"""Leakage-safe reservoir/Koopman forecasting models.

The public API deliberately uses arrays in ``(time, channel)`` order.  A model
is fitted on one training block and neither :class:`StandardScaler` nor PCA is
ever refitted by :meth:`predict`/:meth:`forecast`.  This makes the data split a
property of the caller rather than an easily overlooked option in the model.

The RC model follows the paper diagram:

``standardised channels -> PCA latent -> [z_t, PLV*x_t, delays] -> ESN``

Two ridge maps are then learned: a readout from reservoir state to latent state
and a one-step Koopman map between successive reservoir states.  Forecasts are
autonomous after the final observed context state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
MODEL_SCHEMA_VERSION = 4
MethodName = Literal[
    "persistence",
    "ridge_var",
    "esn_plain",
    "esn_topology",
    "esn_delay",
    "rc_topology_delay",
]

SUPPORTED_METHODS: tuple[str, ...] = (
    "persistence",
    "ridge_var",
    "esn_plain",
    "esn_topology",
    "esn_delay",
    "rc_topology_delay",
)

_METHOD_ALIASES = {
    "vanilla_esn": "esn_plain",
    "delay_only": "esn_delay",
    "full": "rc_topology_delay",
    "topology_delay": "rc_topology_delay",
}


def _as_2d_finite(x: ArrayLike, name: str = "X") -> FloatArray:
    out = np.asarray(x, dtype=np.float64)
    if out.ndim != 2:
        raise ValueError(f"{name} must have shape (time, channel); got {out.shape}")
    if out.shape[0] < 2 or out.shape[1] < 1:
        raise ValueError(f"{name} is too small: {out.shape}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return out


def spectral_radius(matrix: ArrayLike) -> float:
    """Return the largest eigenvalue modulus (zero for an empty matrix)."""

    a = np.asarray(matrix, dtype=np.float64)
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(np.linalg.eigvals(a))))


def regularized_innovation_covariance(
    residuals: ArrayLike,
    *,
    shrinkage: float = 0.05,
    relative_jitter: float = 1e-8,
) -> tuple[FloatArray, dict[str, float | int | bool]]:
    """Estimate a deterministic symmetric positive-definite innovation covariance.

    Residual rows must already respect run boundaries.  Spherical shrinkage
    makes the estimate usable when the latent dimension is not small relative
    to the number of transitions; jitter and eigenvalue flooring make the PSD
    guarantee explicit rather than relying on numerical luck.
    """

    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("residuals must have shape (innovation, latent)")
    if not np.isfinite(values).all():
        raise ValueError("residuals contain non-finite values")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1]")
    if relative_jitter < 0.0:
        raise ValueError("relative_jitter must be non-negative")

    centered = values - values.mean(axis=0, keepdims=True)
    if values.shape[0] > 1:
        sample = centered.T @ centered / (values.shape[0] - 1)
    else:
        sample = np.outer(values[0], values[0])
    sample = 0.5 * (sample + sample.T)
    latent_dim = sample.shape[0]
    scale = float(np.trace(sample) / latent_dim)
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        scale = max(float(np.mean(values**2)), 1e-12)
    covariance = (1.0 - shrinkage) * sample + shrinkage * scale * np.eye(latent_dim)
    jitter = max(scale * relative_jitter, 1e-12)
    covariance += jitter * np.eye(latent_dim)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    minimum_before = float(np.min(eigenvalues))
    eigenvalues = np.maximum(eigenvalues, jitter)
    covariance = (eigenvectors * eigenvalues) @ eigenvectors.T
    covariance = 0.5 * (covariance + covariance.T)
    information: dict[str, float | int | bool] = {
        "n_innovations": int(values.shape[0]),
        "latent_dimension": int(latent_dim),
        "shrinkage": float(shrinkage),
        "jitter": float(jitter),
        "minimum_eigenvalue_before_floor": minimum_before,
        "minimum_eigenvalue_after_floor": float(np.min(eigenvalues)),
        "within_run_pairs_only": True,
    }
    return np.asarray(covariance, dtype=np.float64), information


def psd_matrix_square_root(matrix: ArrayLike) -> FloatArray:
    """Return the symmetric PSD square root with roundoff-safe eigen clipping."""

    covariance = np.asarray(matrix, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("matrix must be square")
    if not np.isfinite(covariance).all():
        raise ValueError("matrix contains non-finite values")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = max(float(np.max(np.abs(eigenvalues))), 1.0) * 1e-10
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError("matrix is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    return np.asarray(0.5 * (root + root.T), dtype=np.float64)


def stabilize_transition(
    matrix: ArrayLike,
    maximum_radius: float = 0.995,
    *,
    label: str = "transition",
    logger: logging.Logger = LOGGER,
) -> tuple[FloatArray, dict[str, float | bool]]:
    """Radially scale a discrete transition and record the intervention.

    Uniform radial scaling is intentionally used instead of independently
    editing eigenvalues.  It is real-valued, deterministic, and robust for the
    non-normal ridge estimates commonly encountered with short iEEG segments.
    """

    a = np.asarray(matrix, dtype=np.float64).copy()
    before = spectral_radius(a)
    changed = bool(before > maximum_radius)
    if changed:
        a *= maximum_radius / max(before, np.finfo(np.float64).eps)
    after = spectral_radius(a)
    information: dict[str, float | bool] = {
        "spectral_radius_before": before,
        "spectral_radius_after": after,
        "maximum_radius": float(maximum_radius),
        "stabilized": changed,
    }
    if changed:
        logger.warning(
            "Stabilized %s: spectral radius %.6f -> %.6f (limit %.6f)",
            label,
            before,
            after,
            maximum_radius,
        )
    else:
        logger.info("%s spectral radius %.6f required no stabilization", label, before)
    return a, information


@dataclass(frozen=True)
class RCKoopmanConfig:
    """Configuration aligned with the ``prediction`` block in ``config.yaml``."""

    method: str = "rc_topology_delay"
    latent_components: int = 12
    reservoir_size: int = 96
    spectral_radius: float = 0.92
    leak_rate: float = 0.35
    input_scale: float = 0.20
    reservoir_sparsity: float = 0.90  # fraction of recurrent entries set to zero
    ridge_alpha: float = 1e-3
    delays_samples: tuple[int, ...] = (16, 64)
    washout: int = 24
    stability_radius: float = 0.995
    innovation_covariance_shrinkage: float = 0.05
    innovation_covariance_relative_jitter: float = 1e-8
    random_seed: int = 11

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, object], *, method: str | None = None, random_seed: int | None = None
    ) -> "RCKoopmanConfig":
        """Build a config from the YAML ``prediction`` mapping.

        Extra orchestration keys (horizons, methods, context length, and so on)
        are ignored rather than leaking into the numerical model constructor.
        """

        known = {
            "latent_components",
            "reservoir_size",
            "spectral_radius",
            "leak_rate",
            "input_scale",
            "reservoir_sparsity",
            "ridge_alpha",
            "delays_samples",
            "washout",
            "stability_radius",
            "innovation_covariance_shrinkage",
            "innovation_covariance_relative_jitter",
            "random_seed",
        }
        kwargs = {key: values[key] for key in known if key in values}
        if "delays_samples" in kwargs:
            kwargs["delays_samples"] = tuple(int(v) for v in kwargs["delays_samples"])  # type: ignore[arg-type]
        if method is not None:
            kwargs["method"] = method
        if random_seed is not None:
            kwargs["random_seed"] = random_seed
        return cls(**kwargs)  # type: ignore[arg-type]

    def for_method(self, method: str) -> "RCKoopmanConfig":
        canonical = canonical_method(method)
        return replace(self, method=canonical)

    def validate(self) -> None:
        canonical_method(self.method)
        if self.latent_components < 1 or self.reservoir_size < 2:
            raise ValueError("latent_components and reservoir_size must be positive")
        if not 0.0 < self.spectral_radius < 2.0:
            raise ValueError("spectral_radius must lie in (0, 2)")
        if not 0.0 < self.leak_rate <= 1.0:
            raise ValueError("leak_rate must lie in (0, 1]")
        if not 0.0 <= self.reservoir_sparsity < 1.0:
            raise ValueError("reservoir_sparsity is the zero fraction and must lie in [0, 1)")
        if self.ridge_alpha < 0.0:
            raise ValueError("ridge_alpha must be non-negative")
        if not 0.0 <= self.innovation_covariance_shrinkage <= 1.0:
            raise ValueError("innovation_covariance_shrinkage must lie in [0, 1]")
        if self.innovation_covariance_relative_jitter < 0.0:
            raise ValueError("innovation_covariance_relative_jitter must be non-negative")
        if any(int(delay) < 1 for delay in self.delays_samples):
            raise ValueError("all delays_samples must be positive")


def canonical_method(method: str) -> str:
    name = _METHOD_ALIASES.get(str(method).lower(), str(method).lower())
    if name not in SUPPORTED_METHODS:
        raise ValueError(f"unknown prediction method {method!r}; choose from {SUPPORTED_METHODS}")
    return name


class TrainOnlyLatentTransform:
    """StandardScaler plus PCA with an explicit one-time training fit."""

    def __init__(self, n_components: int = 12, random_seed: int = 11) -> None:
        self.requested_components = int(n_components)
        self.random_seed = int(random_seed)
        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self.n_channels_: int | None = None
        self.fit_sample_count_: int | None = None

    @property
    def fitted(self) -> bool:
        return self.pca is not None

    @property
    def n_components_(self) -> int:
        if self.pca is None:
            raise RuntimeError("latent transform has not been fitted")
        return int(self.pca.n_components_)

    @property
    def components_(self) -> FloatArray:
        if self.pca is None:
            raise RuntimeError("latent transform has not been fitted")
        return np.asarray(self.pca.components_, dtype=np.float64)

    def fit(self, x_train: ArrayLike) -> "TrainOnlyLatentTransform":
        x = _as_2d_finite(x_train, "x_train")
        n_components = min(self.requested_components, x.shape[0], x.shape[1])
        if n_components < 1:
            raise ValueError("training data cannot support a PCA component")
        x_scaled = self.scaler.fit_transform(x)
        # Full SVD is deterministic and inexpensive at the configured dimensions.
        self.pca = PCA(n_components=n_components, svd_solver="full")
        self.pca.fit(x_scaled)
        self.n_channels_ = int(x.shape[1])
        self.fit_sample_count_ = int(x.shape[0])
        return self

    def _check(self, x: ArrayLike) -> FloatArray:
        if self.pca is None or self.n_channels_ is None:
            raise RuntimeError("latent transform has not been fitted")
        values = _as_2d_finite(x)
        if values.shape[1] != self.n_channels_:
            raise ValueError(f"expected {self.n_channels_} channels, got {values.shape[1]}")
        return values

    def standardize(self, x: ArrayLike) -> FloatArray:
        values = self._check(x)
        return np.asarray(self.scaler.transform(values), dtype=np.float64)

    def transform(self, x: ArrayLike) -> FloatArray:
        values = self._check(x)
        assert self.pca is not None
        return np.asarray(self.pca.transform(self.scaler.transform(values)), dtype=np.float64)

    def inverse_transform(self, z: ArrayLike) -> FloatArray:
        if self.pca is None:
            raise RuntimeError("latent transform has not been fitted")
        values = np.asarray(z, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.n_components_:
            raise ValueError(
                f"z must have shape (time, {self.n_components_}); got {values.shape}"
            )
        scaled = self.pca.inverse_transform(values)
        return np.asarray(self.scaler.inverse_transform(scaled), dtype=np.float64)


def normalize_adjacency(adjacency: ArrayLike) -> FloatArray:
    """Symmetrically normalise a non-negative channel adjacency matrix."""

    a = np.asarray(adjacency, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("adjacency must be square")
    if not np.isfinite(a).all():
        raise ValueError("adjacency contains non-finite values")
    a = np.maximum(0.0, 0.5 * (a + a.T))
    np.fill_diagonal(a, 0.0)
    degree = a.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    positive = degree > np.finfo(np.float64).eps
    inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    return inv_sqrt[:, None] * a * inv_sqrt[None, :]


class RCKoopmanForecaster:
    """Persistence, ridge-VAR, and topology/delay RC--Koopman forecaster."""

    def __init__(
        self,
        config: RCKoopmanConfig | None = None,
        adjacency: ArrayLike | None = None,
    ) -> None:
        self.config = config or RCKoopmanConfig()
        self.model_schema_version_ = MODEL_SCHEMA_VERSION
        self.config.validate()
        self.method = canonical_method(self.config.method)
        self._adjacency_input = None if adjacency is None else np.asarray(adjacency, dtype=np.float64)
        self.adjacency_: FloatArray | None = None
        self.preprocessor = TrainOnlyLatentTransform(
            self.config.latent_components, self.config.random_seed
        )

        self.W_in_: FloatArray | None = None
        self.W_res_: FloatArray | None = None
        self.K_: FloatArray | None = None
        self.b_K_: FloatArray | None = None
        self.W_out_: FloatArray | None = None
        self.b_out_: FloatArray | None = None
        self.var_coef_: FloatArray | None = None
        self.var_intercept_: FloatArray | None = None
        self.last_reservoir_state_: FloatArray | None = None
        self.last_latent_state_: FloatArray | None = None
        self.train_tail_: FloatArray | None = None
        self.latent_transition_: FloatArray | None = None
        self.latent_bias_: FloatArray | None = None
        self.residual_std_: FloatArray | None = None
        self.residual_covariance_: FloatArray | None = None
        self.residual_covariance_info_: dict[str, float | int | bool] = {}
        self.latent_dynamics_diagnostics_: dict[str, float | int] = {}
        self.direct_latent_transition_: FloatArray | None = None
        self.direct_latent_bias_: FloatArray | None = None
        self.direct_residual_std_: FloatArray | None = None
        self.direct_residual_covariance_: FloatArray | None = None
        self.direct_residual_covariance_info_: dict[str, float | int | bool] = {}
        self.direct_stabilization_log_: dict[str, float | bool] = {}
        self.direct_latent_dynamics_diagnostics_: dict[str, float | int] = {}
        self.direct_diagnostics_: dict[str, float | int] = {}
        self.stabilization_log_: dict[str, dict[str, float | bool]] = {}
        self.training_sequence_count_: int = 0
        self.dynamics_transition_count_: int = 0
        self.is_fitted_: bool = False

    # Compatibility names used by the supplied prototype and downstream control.
    @property
    def W_in(self) -> FloatArray | None:
        return self.W_in_

    @property
    def W_res(self) -> FloatArray | None:
        return self.W_res_

    @property
    def K(self) -> FloatArray | None:
        return self.K_

    @property
    def W_out(self) -> FloatArray | None:
        return self.W_out_

    @property
    def b_K(self) -> FloatArray | None:
        return self.b_K_

    @property
    def b_out(self) -> FloatArray | None:
        return self.b_out_

    @property
    def scaler(self) -> StandardScaler:
        return self.preprocessor.scaler

    @property
    def pca(self) -> PCA | None:
        return self.preprocessor.pca

    @property
    def n_channels_(self) -> int:
        if self.preprocessor.n_channels_ is None:
            raise RuntimeError("model has not been fitted")
        return self.preprocessor.n_channels_

    @property
    def latent_dim_(self) -> int:
        return self.preprocessor.n_components_

    @property
    def uses_topology(self) -> bool:
        return self.method in {"esn_topology", "rc_topology_delay"}

    @property
    def uses_delays(self) -> bool:
        return self.method in {"esn_delay", "rc_topology_delay"}

    @property
    def delays(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.config.delays_samples))) if self.uses_delays else ()

    @property
    def maximum_delay(self) -> int:
        return max(self.delays, default=0)

    def fit(
        self,
        x_train: ArrayLike,
        adjacency: ArrayLike | None = None,
    ) -> "RCKoopmanForecaster":
        """Fit one continuous training sequence.

        For multiple seizures/runs use :meth:`fit_sequences`; it resets the
        reservoir and excludes cross-run regression pairs.
        """

        return self.fit_sequences([x_train], adjacency=adjacency)

    def fit_sequences(
        self,
        sequences: Sequence[ArrayLike],
        adjacency: ArrayLike | None = None,
    ) -> "RCKoopmanForecaster":
        """Fit several training runs without inventing boundary transitions.

        The scaler and PCA are fitted once on all *training* samples.  Dynamic
        pairs are constructed per sequence, and every ESN run starts from a zero
        reservoir state.  Consequently, neither ``end(run_i) -> start(run_j)``
        nor reservoir memory across that boundary enters the regression.
        """

        if len(sequences) == 0:
            raise ValueError("at least one training sequence is required")
        runs = [
            _as_2d_finite(values, f"sequences[{index}]")
            for index, values in enumerate(sequences)
        ]
        channel_counts = {run.shape[1] for run in runs}
        if len(channel_counts) != 1:
            raise ValueError("all training sequences must use the same channel set")
        x_all = np.vstack(runs)
        self.training_sequence_count_ = len(runs)
        self.preprocessor.fit(x_all)
        z_runs = [self.preprocessor.transform(run) for run in runs]
        scaled_runs = [self.preprocessor.standardize(run) for run in runs]
        z_last = z_runs[-1]
        self.last_latent_state_ = z_last[-1].copy()
        keep = max(1, self.maximum_delay + 1)
        self.train_tail_ = runs[-1][-keep:].copy()

        supplied_adjacency = adjacency if adjacency is not None else self._adjacency_input
        if supplied_adjacency is None:
            supplied_adjacency = np.zeros(
                (x_all.shape[1], x_all.shape[1]), dtype=np.float64
            )
        a = np.asarray(supplied_adjacency, dtype=np.float64)
        if a.shape != (x_all.shape[1], x_all.shape[1]):
            raise ValueError(
                f"adjacency shape {a.shape} does not match {x_all.shape[1]} training channels"
            )
        self.adjacency_ = normalize_adjacency(a)

        if self.method == "persistence":
            differences = np.vstack([np.diff(z, axis=0) for z in z_runs])
            self.dynamics_transition_count_ = int(differences.shape[0])
            self._set_innovation_statistics(differences)
        elif self.method == "ridge_var":
            self._fit_var_sequences(z_runs)
        else:
            self._fit_rc_sequences(z_runs, scaled_runs)
        self.is_fitted_ = True
        return self

    def _set_innovation_statistics(self, residuals: FloatArray) -> None:
        covariance, standard_deviation, information = self._estimate_innovation_statistics(
            residuals
        )
        self.residual_covariance_ = covariance
        self.residual_covariance_info_ = information
        self.residual_std_ = standard_deviation

    def _estimate_innovation_statistics(
        self, residuals: FloatArray
    ) -> tuple[FloatArray, FloatArray, dict[str, float | int | bool]]:
        covariance, information = regularized_innovation_covariance(
            residuals,
            shrinkage=self.config.innovation_covariance_shrinkage,
            relative_jitter=self.config.innovation_covariance_relative_jitter,
        )
        standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        return covariance, standard_deviation, information

    def _fit_var(self, z: FloatArray) -> None:
        self._fit_var_sequences([z])

    def _fit_var_sequences(self, z_sequences: Sequence[FloatArray]) -> None:
        valid = [z for z in z_sequences if z.shape[0] >= 2]
        if sum(z.shape[0] - 1 for z in valid) < 2:
            raise ValueError("ridge_var requires at least two within-run transition pairs")
        current = np.vstack([z[:-1] for z in valid])
        following = np.vstack([z[1:] for z in valid])
        self.dynamics_transition_count_ = int(current.shape[0])
        ridge = Ridge(alpha=self.config.ridge_alpha, fit_intercept=True)
        ridge.fit(current, following)
        transition = np.asarray(ridge.coef_, dtype=np.float64)
        transition, info = stabilize_transition(
            transition,
            self.config.stability_radius,
            label="ridge-VAR latent transition",
        )
        self.var_coef_ = transition
        # Radial stabilization changes the slope. Re-center the affine map so
        # the stabilized dynamics retain the empirical training equilibrium.
        self.var_intercept_ = np.asarray(
            following.mean(axis=0) - transition @ current.mean(axis=0),
            dtype=np.float64,
        )
        info["intercept_recentered"] = True
        self.latent_transition_ = transition
        self.latent_bias_ = self.var_intercept_.copy()
        self.stabilization_log_["latent"] = info
        residual = following - (current @ transition.T + self.var_intercept_)
        self._set_innovation_statistics(residual)
        self.latent_dynamics_diagnostics_ = self._dynamics_diagnostics(
            following, following - residual
        )

    def _input_dimension(self) -> int:
        q = self.latent_dim_
        return q * (1 + int(self.uses_topology) + len(self.delays))

    def _initialise_reservoir(self) -> None:
        rng = np.random.default_rng(self.config.random_seed)
        n_res = self.config.reservoir_size
        input_dim = self._input_dimension()
        recurrent = rng.uniform(-1.0, 1.0, size=(n_res, n_res))
        recurrent[rng.random((n_res, n_res)) < self.config.reservoir_sparsity] = 0.0
        radius = spectral_radius(recurrent)
        if radius <= np.finfo(np.float64).eps:
            # Extremely sparse tiny test reservoirs can be nilpotent by chance.
            recurrent = np.eye(n_res, dtype=np.float64)
            radius = 1.0
        self.W_res_ = recurrent * (self.config.spectral_radius / radius)
        # Keep the expected total input drive comparable across the 1x/2x/3x/4x
        # ablations.  Without fan-in normalization, topology+delay variants
        # receive proportionally larger variance and a different tanh saturation
        # regime, confounding any component attribution.
        input_limit = self.config.input_scale / np.sqrt(input_dim)
        self.W_in_ = rng.uniform(
            -input_limit,
            input_limit,
            size=(n_res, input_dim),
        )

    def _topology_latent(self, scaled_channels: FloatArray) -> FloatArray:
        assert self.adjacency_ is not None
        # Rows are observations; channel propagation is A @ x in column form.
        propagated = scaled_channels @ self.adjacency_.T
        return propagated @ self.preprocessor.components_.T

    def _features(
        self,
        z: FloatArray,
        scaled_channels: FloatArray,
        index: int,
    ) -> FloatArray:
        pieces = [z[index]]
        if self.uses_topology:
            pieces.append(self._topology_latent(scaled_channels[index : index + 1])[0])
        if self.uses_delays:
            pieces.extend(z[index - delay] for delay in self.delays)
        return np.concatenate(pieces).astype(np.float64, copy=False)

    def _reservoir_states(
        self,
        z: FloatArray,
        scaled_channels: FloatArray,
        *,
        initial_state: FloatArray | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        if self.W_in_ is None or self.W_res_ is None:
            self._initialise_reservoir()
        assert self.W_in_ is not None and self.W_res_ is not None
        start = self.maximum_delay
        if z.shape[0] <= start:
            raise ValueError(
                f"sequence length {z.shape[0]} must exceed maximum delay {start}"
            )
        r = (
            np.zeros(self.config.reservoir_size, dtype=np.float64)
            if initial_state is None
            else np.asarray(initial_state, dtype=np.float64).copy()
        )
        states: list[FloatArray] = []
        indices: list[int] = []
        leak = self.config.leak_rate
        for index in range(start, z.shape[0]):
            u = self._features(z, scaled_channels, index)
            proposal = np.tanh(self.W_in_ @ u + self.W_res_ @ r)
            r = (1.0 - leak) * r + leak * proposal
            states.append(r.copy())
            indices.append(index)
        return np.vstack(states), np.asarray(indices, dtype=np.int64)

    def _fit_rc(self, z: FloatArray, x_scaled: FloatArray) -> None:
        self._fit_rc_sequences([z], [x_scaled])

    def _fit_rc_sequences(
        self,
        z_sequences: Sequence[FloatArray],
        scaled_sequences: Sequence[FloatArray],
    ) -> None:
        state_blocks: list[FloatArray] = []
        latent_blocks: list[FloatArray] = []
        transition_current: list[FloatArray] = []
        transition_following: list[FloatArray] = []
        aligned_current_latent: list[FloatArray] = []
        aligned_following_latent: list[FloatArray] = []
        last_state: FloatArray | None = None
        for z, scaled in zip(z_sequences, scaled_sequences, strict=True):
            if z.shape[0] <= self.maximum_delay:
                LOGGER.warning(
                    "Skipping training sequence of length %d: maximum delay is %d",
                    z.shape[0],
                    self.maximum_delay,
                )
                continue
            # initial_state=None is deliberate: seizure runs are independent.
            states, indices = self._reservoir_states(z, scaled, initial_state=None)
            washout = min(self.config.washout, max(0, states.shape[0] - 3))
            states_fit = states[washout:]
            z_fit = z[indices[washout:]]
            if states_fit.shape[0] < 2:
                continue
            state_blocks.append(states_fit)
            latent_blocks.append(z_fit)
            transition_current.append(states_fit[:-1])
            transition_following.append(states_fit[1:])
            aligned_current_latent.append(z_fit[:-1])
            aligned_following_latent.append(z_fit[1:])
            last_state = states[-1].copy()

        if sum(block.shape[0] for block in state_blocks) < 3:
            raise ValueError("training sequence is too short after delays and washout")
        states_fit = np.vstack(state_blocks)
        z_fit = np.vstack(latent_blocks)
        reservoir_current = np.vstack(transition_current)
        reservoir_following = np.vstack(transition_following)
        current_latent = np.vstack(aligned_current_latent)
        following_latent = np.vstack(aligned_following_latent)
        self.dynamics_transition_count_ = int(reservoir_current.shape[0])

        readout = Ridge(alpha=self.config.ridge_alpha, fit_intercept=True)
        readout.fit(states_fit, z_fit)
        self.W_out_ = np.asarray(readout.coef_, dtype=np.float64)
        self.b_out_ = np.asarray(readout.intercept_, dtype=np.float64)

        koopman = Ridge(alpha=self.config.ridge_alpha, fit_intercept=True)
        koopman.fit(reservoir_current, reservoir_following)
        raw_k = np.asarray(koopman.coef_, dtype=np.float64)
        self.K_, k_info = stabilize_transition(
            raw_k,
            self.config.stability_radius,
            label="reservoir Koopman operator",
        )
        self.b_K_ = np.asarray(
            reservoir_following.mean(axis=0)
            - self.K_ @ reservoir_current.mean(axis=0),
            dtype=np.float64,
        )
        k_info["intercept_recentered"] = True
        self.stabilization_log_["reservoir"] = k_info
        assert last_state is not None
        self.last_reservoir_state_ = last_state

        self.latent_transition_, self.latent_bias_, latent_info = self.derive_latent_dynamics(
            stabilize=True
        )
        self.stabilization_log_["latent"] = latent_info

        # The controller simulates the projected latent affine map, so its
        # diffusion calibration must include projection error from that map (not
        # merely the lower reservoir-K/readout regression error).
        assert self.latent_transition_ is not None and self.latent_bias_ is not None
        next_latent = current_latent @ self.latent_transition_.T + self.latent_bias_
        residual = following_latent - next_latent
        self._set_innovation_statistics(residual)
        self.latent_dynamics_diagnostics_ = self._dynamics_diagnostics(
            following_latent, next_latent
        )

        # Primary control surrogate: a directly audited latent affine Koopman
        # map using the exact same within-run pairs. The projected RC map above
        # remains available as a sensitivity matching the original diagram.
        direct_ridge = Ridge(alpha=self.config.ridge_alpha, fit_intercept=True)
        direct_ridge.fit(current_latent, following_latent)
        direct_raw = np.asarray(direct_ridge.coef_, dtype=np.float64)
        direct_transition, direct_info = stabilize_transition(
            direct_raw,
            self.config.stability_radius,
            label="direct latent control transition",
        )
        direct_bias = np.asarray(
            following_latent.mean(axis=0)
            - direct_transition @ current_latent.mean(axis=0),
            dtype=np.float64,
        )
        direct_info["intercept_recentered"] = True
        direct_prediction = current_latent @ direct_transition.T + direct_bias
        direct_residual = following_latent - direct_prediction
        direct_covariance, direct_std, direct_covariance_info = (
            self._estimate_innovation_statistics(direct_residual)
        )
        self.direct_latent_transition_ = direct_transition
        self.direct_latent_bias_ = direct_bias
        self.direct_residual_covariance_ = direct_covariance
        self.direct_residual_std_ = direct_std
        self.direct_residual_covariance_info_ = direct_covariance_info
        self.direct_stabilization_log_ = direct_info
        self.direct_latent_dynamics_diagnostics_ = self._dynamics_diagnostics(
            following_latent, direct_prediction
        )
        self.direct_diagnostics_ = dict(self.direct_latent_dynamics_diagnostics_)

    @staticmethod
    def _dynamics_diagnostics(
        observed: FloatArray, predicted: FloatArray
    ) -> dict[str, float | int]:
        error = observed - predicted
        rmse = float(np.sqrt(np.mean(error**2)))
        scale = float(np.std(observed))
        correlation = (
            float(np.corrcoef(observed.ravel(), predicted.ravel())[0, 1])
            if np.std(predicted) > 1e-12 and scale > 1e-12
            else float("nan")
        )
        return {
            "n_transition_pairs": int(observed.shape[0]),
            "one_step_rmse": rmse,
            "one_step_nrmse": rmse / max(scale, np.finfo(np.float64).eps),
            "one_step_correlation": correlation,
        }

    def evaluate_latent_dynamics(
        self,
        sequences: Sequence[ArrayLike],
        *,
        source: str = "projected_rc",
    ) -> dict[str, float | int]:
        """Evaluate a fixed latent control map on boundary-safe held-out runs."""

        self._require_fitted()
        if source == "projected_rc":
            transition, bias = self.latent_transition_, self.latent_bias_
        elif source == "direct_latent":
            transition, bias = self.direct_latent_transition_, self.direct_latent_bias_
        else:
            raise ValueError("source must be 'direct_latent' or 'projected_rc'")
        if transition is None or bias is None:
            raise RuntimeError(f"{self.method} does not expose {source} dynamics")
        if len(sequences) == 0:
            raise ValueError("at least one held-out sequence is required")
        z_runs = [self.preprocessor.transform(values) for values in sequences]
        valid = [z for z in z_runs if z.shape[0] >= 2]
        if not valid:
            raise ValueError("at least one held-out sequence needs two samples")
        current = np.vstack([z[:-1] for z in valid])
        following = np.vstack([z[1:] for z in valid])
        predicted = current @ transition.T + bias
        return self._dynamics_diagnostics(following, predicted)

    def derive_latent_dynamics(
        self, *, stabilize: bool = True
    ) -> tuple[FloatArray, FloatArray, dict[str, float | bool]]:
        """Derive ``A = Wout K pinv(Wout)`` and its affine bias.

        This is the low-dimensional dynamics used by the particle controller.
        Any radial stabilization is both logged and returned to the caller.
        """

        if any(value is None for value in (self.W_out_, self.K_, self.b_out_, self.b_K_)):
            raise RuntimeError("RC readout and Koopman maps have not been fitted")
        assert self.W_out_ is not None and self.K_ is not None
        assert self.b_out_ is not None and self.b_K_ is not None
        raw = self.W_out_ @ self.K_ @ np.linalg.pinv(self.W_out_, rcond=1e-8)
        if stabilize:
            transition, info = stabilize_transition(
                raw,
                self.config.stability_radius,
                label="projected latent A_phys",
            )
        else:
            transition = raw
            radius = spectral_radius(raw)
            info = {
                "spectral_radius_before": radius,
                "spectral_radius_after": radius,
                "maximum_radius": self.config.stability_radius,
                "stabilized": False,
            }
        # If z = Wout*r + bout and r+ = K*r + bK, then
        # z+ = A*z + Wout*bK + bout - A*bout.
        bias = self.W_out_ @ self.b_K_ + self.b_out_ - transition @ self.b_out_
        return transition, np.asarray(bias, dtype=np.float64), info

    def encode_context(self, context: ArrayLike) -> FloatArray:
        """Drive the fixed reservoir with observed context and return its last state."""

        self._require_fitted()
        if self.method in {"persistence", "ridge_var"}:
            raise RuntimeError(f"{self.method} has no reservoir state")
        x = _as_2d_finite(context, "context")
        z = self.preprocessor.transform(x)
        scaled = self.preprocessor.standardize(x)
        states, _ = self._reservoir_states(z, scaled)
        return states[-1].copy()

    def forecast(
        self,
        context: ArrayLike | None,
        horizon: int,
        *,
        return_latent: bool = False,
    ) -> FloatArray:
        """Autonomously forecast ``horizon`` samples after an observed context."""

        self._require_fitted()
        if int(horizon) < 1:
            raise ValueError("horizon must be positive")
        horizon = int(horizon)
        if context is None:
            if self.train_tail_ is None:
                raise RuntimeError("training tail is unavailable")
            context_values = self.train_tail_
        else:
            context_values = _as_2d_finite(context, "context")
        if context_values.shape[1] != self.n_channels_:
            raise ValueError("context channel count differs from training data")
        z_context = self.preprocessor.transform(context_values)

        if self.method == "persistence":
            latent = np.repeat(z_context[-1][None, :], horizon, axis=0)
        elif self.method == "ridge_var":
            assert self.var_coef_ is not None and self.var_intercept_ is not None
            current = z_context[-1].copy()
            values = []
            for _ in range(horizon):
                current = self.var_coef_ @ current + self.var_intercept_
                values.append(current.copy())
            latent = np.vstack(values)
        else:
            assert self.K_ is not None and self.b_K_ is not None
            assert self.W_out_ is not None and self.b_out_ is not None
            if context is None and self.last_reservoir_state_ is not None:
                state = self.last_reservoir_state_.copy()
            else:
                state = self.encode_context(context_values)
            values = []
            for _ in range(horizon):
                state = self.K_ @ state + self.b_K_
                values.append(self.W_out_ @ state + self.b_out_)
            latent = np.vstack(values)
        if return_latent:
            return np.asarray(latent, dtype=np.float64)
        if self.method == "persistence":
            # A conventional persistence baseline repeats the actual last
            # observation.  It must not incur PCA reconstruction error that the
            # baseline itself did not create; this makes the comparison stricter.
            return np.repeat(context_values[-1][None, :], horizon, axis=0)
        return self.preprocessor.inverse_transform(latent)

    def predict(
        self,
        steps: int,
        context: ArrayLike | None = None,
        *,
        return_latent: bool = False,
    ) -> FloatArray:
        """Prototype-compatible alias with ``steps`` as the first argument."""

        return self.forecast(context, steps, return_latent=return_latent)

    def _require_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("model has not been fitted")


# Concise aliases for experiment scripts and backwards-compatible notebooks.
RCKoopmanModel = RCKoopmanForecaster


def make_forecaster(
    method: str,
    prediction_config: RCKoopmanConfig | Mapping[str, object] | None = None,
    *,
    adjacency: ArrayLike | None = None,
    random_seed: int | None = None,
) -> RCKoopmanForecaster:
    """Create any pre-specified forecast ablation through one interface."""

    if prediction_config is None:
        config = RCKoopmanConfig(
            method=canonical_method(method),
            random_seed=11 if random_seed is None else int(random_seed),
        )
    elif isinstance(prediction_config, RCKoopmanConfig):
        config = prediction_config.for_method(method)
        if random_seed is not None:
            config = replace(config, random_seed=int(random_seed))
    else:
        config = RCKoopmanConfig.from_mapping(
            prediction_config, method=canonical_method(method), random_seed=random_seed
        )
    return RCKoopmanForecaster(config=config, adjacency=adjacency)


def build_prediction_models(
    methods: Iterable[str],
    prediction_config: RCKoopmanConfig | Mapping[str, object] | None = None,
    *,
    adjacency: ArrayLike | None = None,
    random_seed: int | None = None,
) -> dict[str, RCKoopmanForecaster]:
    """Return a deterministic model dictionary for the configured ablation list."""

    return {
        canonical_method(method): make_forecaster(
            method,
            prediction_config,
            adjacency=adjacency,
            random_seed=random_seed,
        )
        for method in methods
    }


__all__ = [
    "MODEL_SCHEMA_VERSION",
    "SUPPORTED_METHODS",
    "RCKoopmanConfig",
    "TrainOnlyLatentTransform",
    "RCKoopmanForecaster",
    "RCKoopmanModel",
    "build_prediction_models",
    "canonical_method",
    "make_forecaster",
    "normalize_adjacency",
    "psd_matrix_square_root",
    "regularized_innovation_covariance",
    "spectral_radius",
    "stabilize_transition",
]
