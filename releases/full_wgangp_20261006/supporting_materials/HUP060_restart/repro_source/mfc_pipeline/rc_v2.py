"""Leakage-safe V2 reservoir forecasting with direct multihorizon readouts.

The V2 API keeps every array in ``(time, channel)`` orientation.  Scaling,
latent dimension selection, PCA, reservoir states, and ridge readouts are fit
from the supplied ictal training sequences only.  A direct-horizon readout
maps one teacher-forced reservoir state to every future lead separately; it is
therefore deliberately distinct from an autonomous recursive rollout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .rc_koopman import RCKoopmanConfig, RCKoopmanForecaster, TrainOnlyLatentTransform


FloatArray = NDArray[np.float64]
RC_V2_SCHEMA_VERSION = 1
V2Method = Literal[
    "persistence",
    "ridge_var",
    "delay_linear",
    "esn_plain",
    "esn_topology",
    "esn_delay",
    "rc_topology_delay",
]
ReadoutForm = Literal["persistence", "recursive", "direct_horizon"]

SUPPORTED_V2_METHODS = (
    "persistence",
    "ridge_var",
    "delay_linear",
    "esn_plain",
    "esn_topology",
    "esn_delay",
    "rc_topology_delay",
)


def _finite_2d(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError(f"{name} must have shape (time, channel), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def validate_sequences(sequences: Sequence[ArrayLike]) -> list[FloatArray]:
    if not sequences:
        raise ValueError("at least one sequence is required")
    runs = [_finite_2d(value, f"sequences[{index}]") for index, value in enumerate(sequences)]
    if len({run.shape[1] for run in runs}) != 1:
        raise ValueError("all sequences must use the same channel basis")
    return runs


def select_latent_dimension(
    sequences: Sequence[ArrayLike],
    *,
    variance_threshold: float = 0.95,
    minimum_components: int = 8,
    maximum_components: int = 24,
) -> tuple[int, dict[str, float | int | bool]]:
    """Choose q from training samples only using a bounded variance threshold."""

    runs = validate_sequences(sequences)
    if not 0.0 < variance_threshold <= 1.0:
        raise ValueError("variance_threshold must lie in (0, 1]")
    if minimum_components < 1 or maximum_components < minimum_components:
        raise ValueError("invalid latent component bounds")
    training = np.vstack(runs)
    scaled = StandardScaler().fit_transform(training)
    supported = min(scaled.shape)
    full = PCA(n_components=supported, svd_solver="full").fit(scaled)
    cumulative = np.cumsum(full.explained_variance_ratio_)
    threshold_q = int(np.searchsorted(cumulative, variance_threshold, side="left") + 1)
    lower = min(int(minimum_components), supported)
    upper = min(int(maximum_components), supported)
    q = int(np.clip(threshold_q, lower, upper))
    return q, {
        "latent_components": q,
        "threshold_components_unclipped": threshold_q,
        "variance_threshold": float(variance_threshold),
        "explained_variance_at_q": float(cumulative[q - 1]),
        "variance_threshold_reached": bool(cumulative[q - 1] >= variance_threshold),
        "minimum_components": int(minimum_components),
        "maximum_components": int(maximum_components),
        "training_samples": int(training.shape[0]),
        "training_channels": int(training.shape[1]),
    }


@dataclass(frozen=True)
class RCV2Config:
    method: str = "rc_topology_delay"
    readout_form: str = "direct_horizon"
    variance_threshold: float = 0.95
    latent_components_min: int = 8
    latent_components_max: int = 24
    reservoir_size: int = 96
    spectral_radius: float = 0.75
    leak_rate: float = 0.25
    input_scale: float = 0.10
    reservoir_sparsity: float = 0.90
    ridge_alpha: float = 1e-3
    delays_samples: tuple[int, ...] = (16, 64)
    washout: int = 24
    maximum_direct_horizon: int = 288
    random_seed: int = 11

    def validate(self) -> None:
        if self.method not in SUPPORTED_V2_METHODS:
            raise ValueError(f"unsupported V2 method {self.method!r}")
        if self.readout_form not in {"persistence", "recursive", "direct_horizon"}:
            raise ValueError(f"unsupported readout form {self.readout_form!r}")
        if self.method == "persistence" and self.readout_form != "persistence":
            raise ValueError("persistence requires readout_form='persistence'")
        if self.method in {"ridge_var"} and self.readout_form != "recursive":
            raise ValueError("ridge_var is a recursive one-step comparator")
        if self.method == "delay_linear" and self.readout_form != "direct_horizon":
            raise ValueError("delay_linear requires a direct-horizon readout")
        if self.maximum_direct_horizon < 1:
            raise ValueError("maximum_direct_horizon must be positive")
        if self.ridge_alpha < 0.0:
            raise ValueError("ridge_alpha cannot be negative")
        if any(int(delay) < 1 for delay in self.delays_samples):
            raise ValueError("delay samples must be positive")


class MultiHorizonRidge:
    """Ridge maps from one feature vector to all future latent leads.

    The Gram factorization is shared across leads and future targets are built
    in small chunks, avoiding an ``n_samples x (horizon*q)`` allocation.
    """

    def __init__(self, maximum_horizon: int, alpha: float, chunk_horizons: int = 16) -> None:
        self.maximum_horizon = int(maximum_horizon)
        self.alpha = float(alpha)
        self.chunk_horizons = int(chunk_horizons)
        self.coef_: FloatArray | None = None
        self.intercept_: FloatArray | None = None
        self.n_features_: int | None = None
        self.latent_dim_: int | None = None
        self.n_training_pairs_: int = 0
        self.run_pair_counts_: tuple[int, ...] = ()

    def fit(
        self,
        feature_blocks: Sequence[ArrayLike],
        latent_blocks: Sequence[ArrayLike],
        index_blocks: Sequence[ArrayLike],
    ) -> "MultiHorizonRidge":
        if not (
            len(feature_blocks) == len(latent_blocks) == len(index_blocks) and feature_blocks
        ):
            raise ValueError("feature, latent, and index blocks must be non-empty and aligned")
        features: list[FloatArray] = []
        latents: list[FloatArray] = []
        indices: list[NDArray[np.int64]] = []
        counts: list[int] = []
        for block_number, (feature, latent, index) in enumerate(
            zip(feature_blocks, latent_blocks, index_blocks, strict=True)
        ):
            x = _finite_2d(feature, f"feature_blocks[{block_number}]")
            z = _finite_2d(latent, f"latent_blocks[{block_number}]")
            idx = np.asarray(index, dtype=np.int64)
            if idx.ndim != 1 or len(idx) != len(x):
                raise ValueError("each index block must align with its feature rows")
            valid = (idx >= 0) & (idx + self.maximum_horizon < len(z))
            x = x[valid]
            idx = idx[valid]
            if len(x):
                features.append(x)
                latents.append(z)
                indices.append(idx)
                counts.append(len(x))
            else:
                counts.append(0)
        if not features:
            raise ValueError("no within-run direct-horizon training pairs are available")
        if len({block.shape[1] for block in features}) != 1:
            raise ValueError("feature dimensions differ across runs")
        if len({block.shape[1] for block in latents}) != 1:
            raise ValueError("latent dimensions differ across runs")

        x_all = np.vstack(features)
        x_augmented = np.column_stack([x_all, np.ones(len(x_all), dtype=np.float64)])
        penalty = np.eye(x_augmented.shape[1], dtype=np.float64) * self.alpha
        penalty[-1, -1] = 0.0
        gram = x_augmented.T @ x_augmented + penalty
        jitter = max(float(np.trace(gram) / max(len(gram), 1)) * 1e-12, 1e-12)
        gram += jitter * np.eye(len(gram))
        factor = cho_factor(gram, lower=True, check_finite=False)

        latent_dim = latents[0].shape[1]
        coefficients = np.empty(
            (self.maximum_horizon, latent_dim, x_all.shape[1]), dtype=np.float64
        )
        intercepts = np.empty((self.maximum_horizon, latent_dim), dtype=np.float64)
        for start in range(1, self.maximum_horizon + 1, self.chunk_horizons):
            leads = list(range(start, min(start + self.chunk_horizons, self.maximum_horizon + 1)))
            target_parts = []
            for z, idx in zip(latents, indices, strict=True):
                target_parts.append(np.concatenate([z[idx + lead] for lead in leads], axis=1))
            targets = np.vstack(target_parts)
            beta = cho_solve(
                factor,
                x_augmented.T @ targets,
                check_finite=False,
            )
            for offset, lead in enumerate(leads):
                target_slice = slice(offset * latent_dim, (offset + 1) * latent_dim)
                coefficients[lead - 1] = beta[:-1, target_slice].T
                intercepts[lead - 1] = beta[-1, target_slice]

        self.coef_ = coefficients
        self.intercept_ = intercepts
        self.n_features_ = int(x_all.shape[1])
        self.latent_dim_ = int(latent_dim)
        self.n_training_pairs_ = int(len(x_all))
        self.run_pair_counts_ = tuple(int(value) for value in counts)
        return self

    def predict(self, features: ArrayLike, horizon: int | None = None) -> FloatArray:
        if self.coef_ is None or self.intercept_ is None or self.n_features_ is None:
            raise RuntimeError("direct-horizon ridge has not been fitted")
        values = np.asarray(features, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.n_features_:
            raise ValueError(f"features must have shape (sample, {self.n_features_})")
        steps = self.maximum_horizon if horizon is None else int(horizon)
        if not 1 <= steps <= self.maximum_horizon:
            raise ValueError("requested horizon exceeds the fitted direct readout")
        prediction = np.einsum("sf,hqf->shq", values, self.coef_[:steps])
        prediction += self.intercept_[None, :steps, :]
        return np.asarray(prediction, dtype=np.float64)


class RCV2Forecaster:
    """V2 wrapper supporting recursive and truly direct multihorizon forecasts."""

    def __init__(self, config: RCV2Config, adjacency: ArrayLike | None = None) -> None:
        config.validate()
        self.config = config
        self.model_schema_version_ = RC_V2_SCHEMA_VERSION
        self._adjacency_input = None if adjacency is None else np.asarray(adjacency, dtype=float)
        self.base_model_: RCKoopmanForecaster | None = None
        self.preprocessor_: TrainOnlyLatentTransform | None = None
        self.direct_readout_: MultiHorizonRidge | None = None
        self.latent_selection_: dict[str, float | int | bool] = {}
        self.n_channels_: int | None = None
        self.training_sequence_count_: int = 0
        self.is_fitted_: bool = False

    @property
    def latent_dim_(self) -> int:
        if self.preprocessor_ is None:
            raise RuntimeError("model has not been fitted")
        return self.preprocessor_.n_components_

    @property
    def maximum_delay(self) -> int:
        if self.config.method in {"delay_linear", "esn_delay", "rc_topology_delay"}:
            return max(self.config.delays_samples, default=0)
        return 0

    def _base_config(self, latent_components: int) -> RCKoopmanConfig:
        method = self.config.method
        if method == "delay_linear":
            method = "ridge_var"
        return RCKoopmanConfig(
            method=method,
            latent_components=int(latent_components),
            reservoir_size=int(self.config.reservoir_size),
            spectral_radius=float(self.config.spectral_radius),
            leak_rate=float(self.config.leak_rate),
            input_scale=float(self.config.input_scale),
            reservoir_sparsity=float(self.config.reservoir_sparsity),
            ridge_alpha=float(self.config.ridge_alpha),
            delays_samples=tuple(int(value) for value in self.config.delays_samples),
            washout=int(self.config.washout),
            random_seed=int(self.config.random_seed),
        )

    def fit_sequences(self, sequences: Sequence[ArrayLike]) -> "RCV2Forecaster":
        runs = validate_sequences(sequences)
        q, diagnostics = select_latent_dimension(
            runs,
            variance_threshold=self.config.variance_threshold,
            minimum_components=self.config.latent_components_min,
            maximum_components=self.config.latent_components_max,
        )
        self.latent_selection_ = diagnostics
        self.n_channels_ = int(runs[0].shape[1])
        self.training_sequence_count_ = len(runs)

        if self.config.method == "persistence":
            transform = TrainOnlyLatentTransform(q, self.config.random_seed).fit(np.vstack(runs))
            self.preprocessor_ = transform
            self.is_fitted_ = True
            return self

        base = RCKoopmanForecaster(self._base_config(q), adjacency=self._adjacency_input)
        base.fit_sequences(runs, adjacency=self._adjacency_input)
        self.base_model_ = base
        self.preprocessor_ = base.preprocessor

        if self.config.method == "delay_linear":
            self._fit_delay_direct(runs)
        elif self.config.readout_form == "direct_horizon":
            self._fit_reservoir_direct(runs)
        self.is_fitted_ = True
        return self

    def _fit_delay_direct(self, runs: Sequence[FloatArray]) -> None:
        assert self.preprocessor_ is not None
        z_runs = [self.preprocessor_.transform(run) for run in runs]
        feature_blocks: list[FloatArray] = []
        index_blocks: list[NDArray[np.int64]] = []
        delays = tuple(sorted(set(int(value) for value in self.config.delays_samples)))
        for z in z_runs:
            indices = np.arange(max(delays, default=0), len(z), dtype=np.int64)
            features = np.concatenate(
                [z[indices], *(z[indices - delay] for delay in delays)], axis=1
            )
            feature_blocks.append(features)
            index_blocks.append(indices)
        readout = MultiHorizonRidge(
            self.config.maximum_direct_horizon, self.config.ridge_alpha
        )
        readout.fit(feature_blocks, z_runs, index_blocks)
        self.direct_readout_ = readout

    def _fit_reservoir_direct(self, runs: Sequence[FloatArray]) -> None:
        if self.base_model_ is None or self.preprocessor_ is None:
            raise RuntimeError("base reservoir is unavailable")
        z_runs = [self.preprocessor_.transform(run) for run in runs]
        scaled_runs = [self.preprocessor_.standardize(run) for run in runs]
        feature_blocks: list[FloatArray] = []
        index_blocks: list[NDArray[np.int64]] = []
        for z, scaled in zip(z_runs, scaled_runs, strict=True):
            states, indices = self.base_model_._reservoir_states(  # noqa: SLF001
                z, scaled, initial_state=None
            )
            washout = min(self.config.washout, max(0, len(states) - 1))
            feature_blocks.append(states[washout:])
            index_blocks.append(indices[washout:])
        readout = MultiHorizonRidge(
            self.config.maximum_direct_horizon, self.config.ridge_alpha
        )
        readout.fit(feature_blocks, z_runs, index_blocks)
        self.direct_readout_ = readout

    def _delay_features(self, context: FloatArray) -> FloatArray:
        assert self.preprocessor_ is not None
        z = self.preprocessor_.transform(context)
        delays = tuple(sorted(set(int(value) for value in self.config.delays_samples)))
        if len(z) <= max(delays, default=0):
            raise ValueError("context is shorter than the maximum delay")
        return np.concatenate([z[-1], *(z[-1 - delay] for delay in delays)])

    def forecast(
        self,
        context: ArrayLike,
        horizon: int,
        *,
        readout_form: str | None = None,
        return_latent: bool = False,
    ) -> FloatArray:
        if not self.is_fitted_ or self.preprocessor_ is None or self.n_channels_ is None:
            raise RuntimeError("model has not been fitted")
        values = _finite_2d(context, "context")
        if values.shape[1] != self.n_channels_:
            raise ValueError("context channel count differs from training")
        steps = int(horizon)
        if steps < 1:
            raise ValueError("horizon must be positive")
        form = self.config.readout_form if readout_form is None else str(readout_form)

        if self.config.method == "persistence" or form == "persistence":
            if return_latent:
                latent = self.preprocessor_.transform(values[-1:])
                return np.repeat(latent, steps, axis=0)
            return np.repeat(values[-1:], steps, axis=0)

        if form == "recursive":
            if self.base_model_ is None:
                raise RuntimeError("recursive base model is unavailable")
            return self.base_model_.forecast(values, steps, return_latent=return_latent)

        if form != "direct_horizon" or self.direct_readout_ is None:
            raise RuntimeError("a fitted direct-horizon readout is unavailable")
        if self.config.method == "delay_linear":
            features = self._delay_features(values)
        else:
            if self.base_model_ is None:
                raise RuntimeError("reservoir is unavailable")
            features = self.base_model_.encode_context(values)
        latent = self.direct_readout_.predict(features, steps)[0]
        if return_latent:
            return latent
        return self.preprocessor_.inverse_transform(latent)

    def diagnostics(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model_schema_version": self.model_schema_version_,
            "config": asdict(self.config),
            "training_sequence_count": self.training_sequence_count_,
            **self.latent_selection_,
        }
        if self.direct_readout_ is not None:
            payload.update(
                {
                    "direct_training_pairs": self.direct_readout_.n_training_pairs_,
                    "direct_run_pair_counts": self.direct_readout_.run_pair_counts_,
                    "direct_maximum_horizon": self.direct_readout_.maximum_horizon,
                }
            )
        if self.base_model_ is not None:
            payload["stabilization"] = self.base_model_.stabilization_log_
        return payload


def temporal_block_split(
    values: ArrayLike,
    *,
    sampling_rate_hz: float,
    train_end_s: float,
    validation_start_s: float,
    validation_end_s: float | None = None,
) -> tuple[FloatArray, FloatArray, dict[str, float]]:
    """Return chronological train/validation blocks with an explicit gap."""

    array = _finite_2d(values, "values")
    fs = float(sampling_rate_hz)
    if fs <= 0.0:
        raise ValueError("sampling_rate_hz must be positive")
    train_stop = int(round(float(train_end_s) * fs))
    validation_start = int(round(float(validation_start_s) * fs))
    validation_stop = (
        len(array)
        if validation_end_s is None
        else int(round(float(validation_end_s) * fs))
    )
    if not (2 <= train_stop < validation_start < validation_stop <= len(array)):
        raise ValueError("invalid chronological block boundaries")
    return (
        array[:train_stop].copy(),
        array[validation_start:validation_stop].copy(),
        {
            "train_end_s": train_stop / fs,
            "validation_start_s": validation_start / fs,
            "validation_end_s": validation_stop / fs,
            "embargo_s": (validation_start - train_stop) / fs,
        },
    )


def forecast_positions(
    n_samples: int,
    *,
    context_samples: int,
    maximum_horizon: int,
    guard_samples: int,
    count: int,
) -> NDArray[np.int64]:
    """Choose deterministic, evenly spaced forecast boundaries."""

    earliest = int(context_samples)
    latest = int(n_samples) - int(maximum_horizon) - int(guard_samples)
    if latest < earliest:
        raise ValueError("sequence is too short for context, guard, and forecast horizon")
    requested = max(1, int(count))
    positions = np.unique(np.rint(np.linspace(earliest, latest, requested)).astype(np.int64))
    if len(positions) < min(requested, latest - earliest + 1):
        raise RuntimeError("forecast position construction lost avoidable unique positions")
    return positions


__all__ = [
    "RC_V2_SCHEMA_VERSION",
    "SUPPORTED_V2_METHODS",
    "MultiHorizonRidge",
    "RCV2Config",
    "RCV2Forecaster",
    "forecast_positions",
    "select_latent_dimension",
    "temporal_block_split",
    "validate_sequences",
]
