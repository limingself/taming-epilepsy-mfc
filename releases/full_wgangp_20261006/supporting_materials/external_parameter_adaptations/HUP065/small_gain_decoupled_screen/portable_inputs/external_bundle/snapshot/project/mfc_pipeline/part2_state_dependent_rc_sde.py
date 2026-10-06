#!/usr/bin/env python
"""Leakage-safe HUP060 residual Graph-RC SDE prediction experiment.

The final seizure is touched once, after the reservoir/drift hyperparameters
have been selected on the frozen development fold.  The model learns a
one-step latent increment and a full constant innovation covariance from
training residuals.  No transition projection, clipping, or test-tuned noise
scale is applied.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import itertools
import json
from pathlib import Path
import sys
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from mfc_pipeline.metrics import (  # noqa: E402
    fixed_random_projections,
    normalized_rmse,
    pearson_correlation,
    sliced_wasserstein_distance,
)
from mfc_pipeline.rc_koopman import (  # noqa: E402
    TrainOnlyLatentTransform,
    normalize_adjacency,
    psd_matrix_square_root,
    regularized_innovation_covariance,
)
from mfc_pipeline.rc_v2 import (  # noqa: E402
    forecast_positions,
    select_latent_dimension,
)


from mfc_pipeline import part2_data_pipeline as BASE  # noqa: E402


@dataclass(frozen=True)
class ModelConfig:
    reservoir_size: int
    spectral_radius: float
    leak_rate: float
    input_scale: float
    delays_samples: tuple[int, ...]
    ridge_alpha: float
    reservoir_sparsity: float
    washout_samples: int
    random_seed: int
    latent_variance_threshold: float
    latent_components_min: int
    latent_components_max: int
    diffusion_shrinkage: float
    diffusion_relative_jitter: float
    diffusion_mode: str = "constant"
    diffusion_ridge_alpha: float = 10.0


class ResidualGraphRCSDE:
    """Closed-loop residual RC with a training-residual diffusion matrix."""

    def __init__(self, config: ModelConfig, adjacency: np.ndarray) -> None:
        self.config = config
        self.adjacency_input = np.asarray(adjacency, dtype=np.float64)
        self.transform: TrainOnlyLatentTransform | None = None
        self.adjacency: np.ndarray | None = None
        self.w_in: np.ndarray | None = None
        self.w_res: np.ndarray | None = None
        self.feature_scaler: StandardScaler | None = None
        self.drift: Ridge | None = None
        self.diffusion_covariance: np.ndarray | None = None
        self.diffusion_root: np.ndarray | None = None
        self.diffusion_info: dict[str, object] = {}
        self.diffusion_correlation_root: np.ndarray | None = None
        self.diffusion_variance_model: Ridge | None = None
        self.diffusion_variance_calibration: np.ndarray | None = None
        self.diffusion_log_variance_bounds: np.ndarray | None = None
        self.direct_coef: np.ndarray | None = None
        self.direct_intercept: np.ndarray | None = None
        self.direct_residual_roots: np.ndarray | None = None
        self.direct_correlation_roots: np.ndarray | None = None
        self.direct_variance_coef: np.ndarray | None = None
        self.direct_variance_intercept: np.ndarray | None = None
        self.direct_variance_calibration: np.ndarray | None = None
        self.direct_log_variance_bounds: np.ndarray | None = None
        self.direct_maximum_horizon = 128
        self.training_pairs = 0
        self.training_one_step_nrmse = float("nan")
        self.training_one_step_correlation = float("nan")

    @property
    def q(self) -> int:
        if self.transform is None:
            raise RuntimeError("model is not fitted")
        return self.transform.n_components_

    @property
    def maximum_delay(self) -> int:
        return max(self.config.delays_samples, default=0)

    def _initialise_reservoir(self) -> None:
        rng = np.random.default_rng(self.config.random_seed)
        n_res = self.config.reservoir_size
        input_dim = self.q * (2 + len(self.config.delays_samples))
        recurrent = rng.uniform(-1.0, 1.0, size=(n_res, n_res))
        recurrent[rng.random((n_res, n_res)) < self.config.reservoir_sparsity] = 0.0
        radius = float(np.max(np.abs(np.linalg.eigvals(recurrent))))
        if radius <= np.finfo(float).eps:
            recurrent = np.eye(n_res)
            radius = 1.0
        self.w_res = recurrent * (self.config.spectral_radius / radius)
        limit = self.config.input_scale / np.sqrt(input_dim)
        self.w_in = rng.uniform(-limit, limit, size=(n_res, input_dim))

    def _topology(self, scaled: np.ndarray) -> np.ndarray:
        assert self.adjacency is not None and self.transform is not None
        return (scaled @ self.adjacency.T) @ self.transform.components_.T

    def _input(self, z: np.ndarray, topology: np.ndarray, index: int) -> np.ndarray:
        return np.concatenate(
            [
                z[index],
                topology[index],
                *(z[index - delay] for delay in self.config.delays_samples),
            ]
        )

    def _readout_feature(
        self, state: np.ndarray, z: np.ndarray, topology: np.ndarray, index: int
    ) -> np.ndarray:
        return np.concatenate(
            [
                state,
                z[index],
                topology[index],
                *(z[index - delay] for delay in self.config.delays_samples),
            ]
        )

    def fit(self, sequences: list[np.ndarray]) -> "ResidualGraphRCSDE":
        q, _ = select_latent_dimension(
            sequences,
            variance_threshold=self.config.latent_variance_threshold,
            minimum_components=self.config.latent_components_min,
            maximum_components=self.config.latent_components_max,
        )
        transform = TrainOnlyLatentTransform(q, self.config.random_seed)
        transform.fit(np.vstack(sequences))
        self.transform = transform
        self.adjacency = normalize_adjacency(self.adjacency_input)
        self._initialise_reservoir()
        assert self.w_in is not None and self.w_res is not None

        feature_blocks: list[np.ndarray] = []
        target_blocks: list[np.ndarray] = []
        direct_index_blocks: list[np.ndarray] = []
        latent_blocks: list[np.ndarray] = []
        for sequence in sequences:
            z = transform.transform(sequence)
            scaled = transform.standardize(sequence)
            topology = self._topology(scaled)
            state = np.zeros(self.config.reservoir_size, dtype=np.float64)
            features: list[np.ndarray] = []
            targets: list[np.ndarray] = []
            indices: list[int] = []
            for index in range(self.maximum_delay, len(z) - 1):
                drive = self._input(z, topology, index)
                proposal = np.tanh(self.w_in @ drive + self.w_res @ state)
                state = (1.0 - self.config.leak_rate) * state + self.config.leak_rate * proposal
                if index >= self.maximum_delay + self.config.washout_samples:
                    features.append(self._readout_feature(state, z, topology, index))
                    targets.append(z[index + 1] - z[index])
                    indices.append(index)
            if features:
                feature_blocks.append(np.vstack(features))
                target_blocks.append(np.vstack(targets))
                direct_index_blocks.append(np.asarray(indices, dtype=np.int64))
                latent_blocks.append(z)
        x = np.vstack(feature_blocks)
        y = np.vstack(target_blocks)
        self.training_pairs = int(len(x))
        self.feature_scaler = StandardScaler().fit(x)
        drift = Ridge(alpha=self.config.ridge_alpha, fit_intercept=True)
        drift.fit(self.feature_scaler.transform(x), y)
        self.drift = drift
        fitted = drift.predict(self.feature_scaler.transform(x))
        residual = y - fitted
        covariance, information = regularized_innovation_covariance(
            residual,
            shrinkage=self.config.diffusion_shrinkage,
            relative_jitter=self.config.diffusion_relative_jitter,
        )
        self.diffusion_covariance = covariance
        self.diffusion_root = psd_matrix_square_root(covariance)
        self.diffusion_info = information
        if self.config.diffusion_mode == "state_dependent":
            self._fit_state_dependent_diffusion(
                self.feature_scaler.transform(x), residual
            )
        elif self.config.diffusion_mode != "constant":
            raise ValueError(
                "diffusion_mode must be 'constant' or 'state_dependent'"
            )
        self.training_one_step_nrmse = float(normalized_rmse(y, fitted))
        self.training_one_step_correlation = float(pearson_correlation(y, fitted))
        self._fit_direct_head(feature_blocks, latent_blocks, direct_index_blocks)
        return self

    def _fit_state_dependent_diffusion(
        self, scaled_features: np.ndarray, residual: np.ndarray
    ) -> None:
        """Fit PSD heteroscedastic diffusion as D(x) R D(x).

        R is the training-residual correlation matrix and D(x) contains
        state-conditioned marginal standard deviations predicted from the
        standardized RC readout feature.  The construction remains positive
        semidefinite for every state while avoiding a high-variance full
        covariance regression.
        """
        global_variance = np.maximum(np.mean(residual**2, axis=0), 1e-12)
        standardized = residual / np.sqrt(global_variance)[None]
        correlation = np.cov(standardized, rowvar=False, ddof=1)
        diagonal = np.sqrt(np.maximum(np.diag(correlation), 1e-12))
        correlation = correlation / diagonal[:, None] / diagonal[None, :]
        shrinkage = float(self.config.diffusion_shrinkage)
        correlation = (1.0 - shrinkage) * correlation + shrinkage * np.eye(self.q)
        correlation = (correlation + correlation.T) / 2.0
        self.diffusion_correlation_root = psd_matrix_square_root(correlation)

        # A small data-scaled floor makes log-squared innovations estimable;
        # it is tied to each training marginal variance, not hand tuned on test.
        log_target = np.log(residual**2 + 0.05 * global_variance[None])
        variance_model = Ridge(
            alpha=float(self.config.diffusion_ridge_alpha), fit_intercept=True
        )
        variance_model.fit(scaled_features, log_target)
        raw_log_variance = variance_model.predict(scaled_features)
        uncalibrated = np.exp(np.clip(raw_log_variance, -30.0, 30.0))
        calibration = global_variance / np.maximum(
            np.mean(uncalibrated, axis=0), 1e-12
        )
        calibrated_log = np.log(
            np.maximum(uncalibrated * calibration[None], 1e-12)
        )
        bounds = np.quantile(calibrated_log, [0.005, 0.995], axis=0)
        self.diffusion_variance_model = variance_model
        self.diffusion_variance_calibration = calibration
        self.diffusion_log_variance_bounds = bounds
        self.diffusion_info = {
            **self.diffusion_info,
            "mode": "state_dependent_D(x)R D(x)",
            "variance_regression": "ridge_on_log_squared_training_innovations",
            "variance_ridge_alpha": float(self.config.diffusion_ridge_alpha),
            "conditional_variance_train_q005": np.exp(bounds[0]).tolist(),
            "conditional_variance_train_q995": np.exp(bounds[1]).tolist(),
            "positive_semidefinite_by_construction": True,
        }

    def _state_dependent_diffusion_std(
        self, scaled_features: np.ndarray
    ) -> np.ndarray:
        if (
            self.diffusion_variance_model is None
            or self.diffusion_variance_calibration is None
            or self.diffusion_log_variance_bounds is None
        ):
            raise RuntimeError("state-dependent diffusion is unavailable")
        raw = self.diffusion_variance_model.predict(scaled_features)
        log_variance = raw + np.log(self.diffusion_variance_calibration)[None]
        log_variance = np.maximum(
            log_variance, self.diffusion_log_variance_bounds[0][None]
        )
        log_variance = np.minimum(
            log_variance, self.diffusion_log_variance_bounds[1][None]
        )
        return np.sqrt(np.exp(log_variance))

    def _fit_direct_head(
        self,
        feature_blocks: list[np.ndarray],
        latent_blocks: list[np.ndarray],
        index_blocks: list[np.ndarray],
    ) -> None:
        if self.feature_scaler is None:
            raise RuntimeError("feature scaler is unavailable")
        valid_features: list[np.ndarray] = []
        valid_latent: list[np.ndarray] = []
        valid_indices: list[np.ndarray] = []
        for features, latent, indices in zip(
            feature_blocks, latent_blocks, index_blocks, strict=True
        ):
            valid = indices + self.direct_maximum_horizon < len(latent)
            if np.any(valid):
                valid_features.append(features[valid])
                valid_latent.append(latent)
                valid_indices.append(indices[valid])
        x = np.vstack(valid_features)
        x_scaled = self.feature_scaler.transform(x)
        n_features = x_scaled.shape[1]
        coef = np.empty(
            (self.direct_maximum_horizon, self.q, n_features), dtype=np.float64
        )
        intercept = np.empty(
            (self.direct_maximum_horizon, self.q), dtype=np.float64
        )
        roots = np.empty(
            (self.direct_maximum_horizon, self.q, self.q), dtype=np.float64
        )
        correlation_roots = np.empty_like(roots)
        variance_coef = np.empty_like(coef)
        variance_intercept = np.empty_like(intercept)
        variance_calibration = np.empty(
            (self.direct_maximum_horizon, self.q), dtype=np.float64
        )
        log_variance_bounds = np.empty(
            (2, self.direct_maximum_horizon, self.q), dtype=np.float64
        )
        chunk_size = 16
        for start in range(0, self.direct_maximum_horizon, chunk_size):
            stop = min(self.direct_maximum_horizon, start + chunk_size)
            horizon_values = list(range(start + 1, stop + 1))
            targets = np.concatenate(
                [
                    np.vstack(
                        [
                            latent[indices + horizon] - latent[indices]
                            for latent, indices in zip(
                                valid_latent, valid_indices, strict=True
                            )
                        ]
                    )
                    for horizon in horizon_values
                ],
                axis=1,
            )
            ridge = Ridge(alpha=self.config.ridge_alpha, fit_intercept=True)
            ridge.fit(x_scaled, targets)
            chunk = stop - start
            coef[start:stop] = np.asarray(ridge.coef_).reshape(
                chunk, self.q, n_features
            )
            intercept[start:stop] = np.asarray(ridge.intercept_).reshape(
                chunk, self.q
            )
            predictions = ridge.predict(x_scaled).reshape(len(x_scaled), chunk, self.q)
            target_3d = targets.reshape(len(x_scaled), chunk, self.q)
            residual_3d = target_3d - predictions
            log_variance_targets: list[np.ndarray] = []
            for offset in range(chunk):
                residual_at_horizon = residual_3d[:, offset]
                covariance, _ = regularized_innovation_covariance(
                    residual_at_horizon,
                    shrinkage=self.config.diffusion_shrinkage,
                    relative_jitter=self.config.diffusion_relative_jitter,
                )
                roots[start + offset] = psd_matrix_square_root(covariance)
                global_variance = np.maximum(
                    np.mean(residual_at_horizon**2, axis=0), 1e-12
                )
                standardized = residual_at_horizon / np.sqrt(global_variance)[None]
                correlation = np.cov(standardized, rowvar=False, ddof=1)
                diagonal = np.sqrt(np.maximum(np.diag(correlation), 1e-12))
                correlation = correlation / diagonal[:, None] / diagonal[None, :]
                shrinkage = float(self.config.diffusion_shrinkage)
                correlation = (
                    (1.0 - shrinkage) * correlation
                    + shrinkage * np.eye(self.q)
                )
                correlation_roots[start + offset] = psd_matrix_square_root(
                    (correlation + correlation.T) / 2.0
                )
                log_variance_targets.append(
                    np.log(
                        residual_at_horizon**2
                        + 0.05 * global_variance[None]
                    )
                )
            variance_targets = np.concatenate(log_variance_targets, axis=1)
            variance_ridge = Ridge(
                alpha=float(self.config.diffusion_ridge_alpha), fit_intercept=True
            )
            variance_ridge.fit(x_scaled, variance_targets)
            variance_coef[start:stop] = np.asarray(variance_ridge.coef_).reshape(
                chunk, self.q, n_features
            )
            variance_intercept[start:stop] = np.asarray(
                variance_ridge.intercept_
            ).reshape(chunk, self.q)
            predicted_log_variance = variance_ridge.predict(x_scaled).reshape(
                len(x_scaled), chunk, self.q
            )
            for offset in range(chunk):
                residual_at_horizon = residual_3d[:, offset]
                global_variance = np.maximum(
                    np.mean(residual_at_horizon**2, axis=0), 1e-12
                )
                uncalibrated = np.exp(
                    np.clip(predicted_log_variance[:, offset], -30.0, 30.0)
                )
                calibration = global_variance / np.maximum(
                    np.mean(uncalibrated, axis=0), 1e-12
                )
                calibrated_log = np.log(
                    np.maximum(uncalibrated * calibration[None], 1e-12)
                )
                variance_calibration[start + offset] = calibration
                log_variance_bounds[:, start + offset] = np.quantile(
                    calibrated_log, [0.005, 0.995], axis=0
                )
        self.direct_coef = coef
        self.direct_intercept = intercept
        self.direct_residual_roots = roots
        self.direct_correlation_roots = correlation_roots
        self.direct_variance_coef = variance_coef
        self.direct_variance_intercept = variance_intercept
        self.direct_variance_calibration = variance_calibration
        self.direct_log_variance_bounds = log_variance_bounds

    def _context_state(
        self, context: np.ndarray
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        if self.transform is None or self.w_in is None or self.w_res is None:
            raise RuntimeError("model is not fitted")
        z = self.transform.transform(context)
        scaled = self.transform.standardize(context)
        topology = self._topology(scaled)
        state = np.zeros(self.config.reservoir_size, dtype=np.float64)
        for index in range(self.maximum_delay, len(z)):
            drive = self._input(z, topology, index)
            proposal = np.tanh(self.w_in @ drive + self.w_res @ state)
            state = (1.0 - self.config.leak_rate) * state + self.config.leak_rate * proposal
        return state, [row.copy() for row in z], topology[-1].copy()

    def _scaled_from_latent(self, latent: np.ndarray) -> np.ndarray:
        assert self.transform is not None and self.transform.pca is not None
        return np.asarray(self.transform.pca.inverse_transform(latent), dtype=np.float64)

    def forecast_latent(self, context: np.ndarray, horizon: int) -> np.ndarray:
        if self.feature_scaler is None or self.drift is None:
            raise RuntimeError("model is not fitted")
        assert self.w_in is not None and self.w_res is not None
        state, history, topology_current = self._context_state(context)
        values: list[np.ndarray] = []
        for _ in range(int(horizon)):
            current = history[-1]
            feature = np.concatenate(
                [
                    state,
                    current,
                    topology_current,
                    *(history[-1 - delay] for delay in self.config.delays_samples),
                ]
            )
            increment = self.drift.predict(self.feature_scaler.transform(feature[None]))[0]
            following = current + increment
            history.append(following.copy())
            scaled_following = self._scaled_from_latent(following[None])[0]
            topology_current = self._topology(scaled_following[None])[0]
            drive = np.concatenate(
                [
                    following,
                    topology_current,
                    *(history[-1 - delay] for delay in self.config.delays_samples),
                ]
            )
            proposal = np.tanh(self.w_in @ drive + self.w_res @ state)
            state = (1.0 - self.config.leak_rate) * state + self.config.leak_rate * proposal
            values.append(following.copy())
        return np.vstack(values)

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        assert self.transform is not None
        return self.transform.inverse_transform(self.forecast_latent(context, horizon))

    def forecast_direct_latent(self, context: np.ndarray, horizon: int) -> np.ndarray:
        if (
            self.feature_scaler is None
            or self.direct_coef is None
            or self.direct_intercept is None
        ):
            raise RuntimeError("direct head is unavailable")
        steps = int(horizon)
        if not 1 <= steps <= self.direct_maximum_horizon:
            raise ValueError("direct horizon exceeds the fitted range")
        state, history, topology_current = self._context_state(context)
        feature = np.concatenate(
            [
                state,
                history[-1],
                topology_current,
                *(history[-1 - delay] for delay in self.config.delays_samples),
            ]
        )
        scaled_feature = self.feature_scaler.transform(feature[None])[0]
        increments = np.einsum(
            "hqf,f->hq", self.direct_coef[:steps], scaled_feature
        ) + self.direct_intercept[:steps]
        return history[-1][None] + increments

    def forecast_direct(self, context: np.ndarray, horizon: int) -> np.ndarray:
        assert self.transform is not None
        return self.transform.inverse_transform(
            self.forecast_direct_latent(context, horizon)
        )

    def sample_direct_scaled(
        self,
        context: np.ndarray,
        horizon: int,
        rollouts: int,
        seed: int,
        diffusion_scale: float = 1.0,
    ) -> np.ndarray:
        if self.direct_residual_roots is None:
            raise RuntimeError("direct residual covariance is unavailable")
        mean = self.forecast_direct_latent(context, horizon)
        state, history, topology_current = self._context_state(context)
        feature = np.concatenate(
            [
                state,
                history[-1],
                topology_current,
                *(history[-1 - delay] for delay in self.config.delays_samples),
            ]
        )
        assert self.feature_scaler is not None
        scaled_feature = self.feature_scaler.transform(feature[None])[0]
        rng = np.random.default_rng(seed)
        latent = np.empty((int(rollouts), int(horizon), self.q), dtype=np.float64)
        for step in range(int(horizon)):
            if self.config.diffusion_mode == "state_dependent":
                if (
                    self.direct_correlation_roots is None
                    or self.direct_variance_coef is None
                    or self.direct_variance_intercept is None
                    or self.direct_variance_calibration is None
                    or self.direct_log_variance_bounds is None
                ):
                    raise RuntimeError(
                        "direct state-dependent diffusion is unavailable"
                    )
                raw_log_variance = (
                    self.direct_variance_coef[step] @ scaled_feature
                    + self.direct_variance_intercept[step]
                    + np.log(self.direct_variance_calibration[step])
                )
                log_variance = np.maximum(
                    raw_log_variance,
                    self.direct_log_variance_bounds[0, step],
                )
                log_variance = np.minimum(
                    log_variance,
                    self.direct_log_variance_bounds[1, step],
                )
                conditional_std = np.sqrt(np.exp(log_variance))
                innovation = (
                    rng.normal(size=(int(rollouts), self.q))
                    @ self.direct_correlation_roots[step].T
                ) * conditional_std[None]
            else:
                innovation = (
                    rng.normal(size=(int(rollouts), self.q))
                    @ self.direct_residual_roots[step].T
                )
            latent[:, step] = mean[step] + float(diffusion_scale) * innovation
        return self._scaled_from_latent(latent.reshape(-1, self.q)).reshape(
            int(rollouts), int(horizon), -1
        )

    def rolling_one_step_scaled(
        self,
        context: np.ndarray,
        observed_future: np.ndarray,
        rollouts: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Teacher-forced one-step forecasts along a contiguous held-out segment.

        Each point is predicted from observations available strictly before that
        point.  The observed sample is then assimilated before predicting the next
        point.  This is a rolling one-step diagnostic, not a free-running rollout.
        """
        if (
            self.feature_scaler is None
            or self.direct_coef is None
            or self.direct_intercept is None
            or self.direct_residual_roots is None
            or self.transform is None
        ):
            raise RuntimeError("direct head is unavailable")
        assert self.w_in is not None and self.w_res is not None
        state, history, topology_current = self._context_state(context)
        future = np.asarray(observed_future, dtype=np.float64)
        future_latent = self.transform.transform(future)
        future_scaled = self.transform.standardize(future)
        n_rollouts = int(rollouts)
        rng = np.random.default_rng(seed)
        mean_scaled = np.empty((len(future), self.transform.n_channels_), dtype=np.float64)
        sampled_scaled = np.empty(
            (n_rollouts, len(future), self.transform.n_channels_), dtype=np.float64
        )
        for step in range(len(future)):
            feature = np.concatenate(
                [
                    state,
                    history[-1],
                    topology_current,
                    *(history[-1 - delay] for delay in self.config.delays_samples),
                ]
            )
            scaled_feature = self.feature_scaler.transform(feature[None])[0]
            mean_increment = (
                self.direct_coef[0] @ scaled_feature + self.direct_intercept[0]
            )
            mean_latent = history[-1] + mean_increment
            mean_scaled[step] = self._scaled_from_latent(mean_latent[None])[0]
            innovations = (
                rng.normal(size=(n_rollouts, self.q))
                @ self.direct_residual_roots[0].T
            )
            sampled_scaled[:, step] = self._scaled_from_latent(
                mean_latent[None] + innovations
            )

            # Assimilate only the now-observed held-out sample for the next step.
            following = future_latent[step]
            history.append(following.copy())
            topology_current = self._topology(future_scaled[step : step + 1])[0]
            drive = np.concatenate(
                [
                    following,
                    topology_current,
                    *(history[-1 - delay] for delay in self.config.delays_samples),
                ]
            )
            proposal = np.tanh(self.w_in @ drive + self.w_res @ state)
            state = (
                (1.0 - self.config.leak_rate) * state
                + self.config.leak_rate * proposal
            )
        return mean_scaled, sampled_scaled

    def sample_scaled(
        self, context: np.ndarray, horizon: int, rollouts: int, seed: int
    ) -> np.ndarray:
        if self.feature_scaler is None or self.drift is None or self.diffusion_root is None:
            raise RuntimeError("model is not fitted")
        assert self.w_in is not None and self.w_res is not None
        state0, history0, topology0 = self._context_state(context)
        n_rollouts = int(rollouts)
        states = np.repeat(state0[None], n_rollouts, axis=0)
        history = [np.repeat(value[None], n_rollouts, axis=0) for value in history0]
        topology = np.repeat(topology0[None], n_rollouts, axis=0)
        rng = np.random.default_rng(seed)
        out = np.empty((n_rollouts, int(horizon), self.transform.n_channels_))
        for step in range(int(horizon)):
            current = history[-1]
            feature = np.concatenate(
                [
                    states,
                    current,
                    topology,
                    *(history[-1 - delay] for delay in self.config.delays_samples),
                ],
                axis=1,
            )
            scaled_feature = self.feature_scaler.transform(feature)
            mean_increment = self.drift.predict(scaled_feature)
            if self.config.diffusion_mode == "state_dependent":
                assert self.diffusion_correlation_root is not None
                conditional_std = self._state_dependent_diffusion_std(scaled_feature)
                innovation = (
                    rng.normal(size=(n_rollouts, self.q))
                    @ self.diffusion_correlation_root.T
                ) * conditional_std
            else:
                innovation = (
                    rng.normal(size=(n_rollouts, self.q)) @ self.diffusion_root.T
                )
            following = current + mean_increment + innovation
            history.append(following.copy())
            scaled_following = self._scaled_from_latent(following)
            topology = self._topology(scaled_following)
            drive = np.concatenate(
                [
                    following,
                    topology,
                    *(history[-1 - delay] for delay in self.config.delays_samples),
                ],
                axis=1,
            )
            proposal = np.tanh(drive @ self.w_in.T + states @ self.w_res.T)
            states = (1.0 - self.config.leak_rate) * states + self.config.leak_rate * proposal
            out[:, step, :] = scaled_following
        return out

    def diagnostics(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "latent_components": self.q,
            "training_pairs": self.training_pairs,
            "training_one_step_increment_nrmse": self.training_one_step_nrmse,
            "training_one_step_increment_correlation": self.training_one_step_correlation,
            "diffusion": self.diffusion_info,
            "transition_projection_applied": False,
            "diffusion_scale_tuned": False,
        }


def trajectory_metrics(truth: np.ndarray, estimate: np.ndarray, scaler) -> dict[str, float]:
    truth_scaled = scaler.transform(truth)
    estimate_scaled = scaler.transform(estimate)
    return {
        "nrmse": float(normalized_rmse(truth_scaled, estimate_scaled)),
        "correlation": float(pearson_correlation(truth_scaled, estimate_scaled)),
        "mae_standardized": float(np.mean(np.abs(truth_scaled - estimate_scaled))),
    }


def safe_swd(
    first: np.ndarray, second: np.ndarray, projections: np.ndarray
) -> float:
    if len(first) >= 2 and len(second) >= 2:
        return float(
            sliced_wasserstein_distance(
                first, second, projections=projections
            )
        )
    directions = projections / np.linalg.norm(projections, axis=1, keepdims=True)
    projected_first = np.asarray(first) @ directions.T
    projected_second = np.asarray(second) @ directions.T
    return float(np.mean(np.abs(projected_first[0] - projected_second[0])))


def candidate_grid(spec: dict) -> list[ModelConfig]:
    common = {
        "reservoir_sparsity": float(spec["reservoir_sparsity"]),
        "washout_samples": int(spec["washout_samples"]),
        "random_seed": int(spec["random_seed"]),
        "latent_variance_threshold": float(spec["latent_variance_threshold"]),
        "latent_components_min": int(spec["latent_components_min"]),
        "latent_components_max": int(spec["latent_components_max"]),
        "diffusion_shrinkage": float(spec["diffusion_shrinkage"]),
        "diffusion_relative_jitter": float(spec["diffusion_relative_jitter"]),
        "diffusion_mode": str(spec.get("diffusion_mode", "constant")),
        "diffusion_ridge_alpha": float(spec.get("diffusion_ridge_alpha", 10.0)),
    }
    return [
        ModelConfig(
            reservoir_size=int(size),
            spectral_radius=float(radius),
            leak_rate=float(leak),
            input_scale=float(input_scale),
            delays_samples=tuple(int(value) for value in delays),
            ridge_alpha=float(ridge),
            **common,
        )
        for size, radius, leak, input_scale, delays, ridge in itertools.product(
            spec["reservoir_sizes"],
            spec["spectral_radii"],
            spec["leak_rates"],
            spec["input_scales"],
            spec["delay_sets_samples"],
            spec["ridge_alphas"],
        )
    ]


def evaluate_candidate(
    model: ResidualGraphRCSDE,
    sequence: np.ndarray,
    horizons: list[int],
    context_samples: int,
    windows: int,
    projection_seed: int,
    forecast_mode: str = "closed_loop",
) -> pd.DataFrame:
    maximum = max(horizons)
    positions = forecast_positions(
        len(sequence),
        context_samples=context_samples,
        maximum_horizon=maximum,
        guard_samples=0,
        count=windows,
    )
    assert model.transform is not None
    projections = fixed_random_projections(
        sequence.shape[1], n_projections=128, seed=projection_seed
    )
    rows: list[dict[str, object]] = []
    for window, position in enumerate(positions):
        context = sequence[position - context_samples : position]
        if forecast_mode == "closed_loop":
            forecast = model.forecast(context, maximum)
        elif forecast_mode == "direct":
            forecast = model.forecast_direct(context, maximum)
        else:
            raise ValueError("forecast_mode must be 'closed_loop' or 'direct'")
        persistence = np.repeat(context[-1:], maximum, axis=0)
        for horizon in horizons:
            truth = sequence[position : position + horizon]
            pred = forecast[:horizon]
            baseline = persistence[:horizon]
            metrics = trajectory_metrics(truth, pred, model.transform.scaler)
            baseline_metrics = trajectory_metrics(truth, baseline, model.transform.scaler)
            truth_scaled = model.transform.scaler.transform(truth)
            pred_scaled = model.transform.scaler.transform(pred)
            baseline_scaled = model.transform.scaler.transform(baseline)
            swd = safe_swd(pred_scaled, truth_scaled, projections)
            persistence_swd = safe_swd(baseline_scaled, truth_scaled, projections)
            rows.append(
                {
                    "window": int(window),
                    "forecast_boundary_sample": int(position),
                    "horizon_samples": int(horizon),
                    **metrics,
                    "persistence_nrmse": baseline_metrics["nrmse"],
                    "persistence_correlation": baseline_metrics["correlation"],
                    "swd": swd,
                    "persistence_swd": persistence_swd,
                    "nrmse_ratio_to_persistence": metrics["nrmse"] / baseline_metrics["nrmse"],
                    "swd_ratio_to_persistence": swd / persistence_swd,
                }
            )
    return pd.DataFrame(rows)


def kde_curve(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if np.std(values) < 1e-8:
        return np.zeros_like(grid)
    return gaussian_kde(values)(grid)


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "legend.frameon": False,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=9, fontweight="bold")


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def representative_channels(channels: list[str]) -> tuple[str, str]:
    nodes = pd.read_csv(
        PROJECT
        / "output"
        / "part1"
        / "source_data"
        / "figure_01"
        / "hup060_plv_nodes_and_selection.csv"
    )
    nodes = nodes[nodes["channel"].astype(str).isin(channels)].copy()
    is_soz = nodes["soz"].astype(str).str.lower().eq("true")
    soz = nodes[is_soz].sort_values(
        "weighted_centrality_score", ascending=False
    )
    nonsoz = nodes[~is_soz].sort_values(
        "weighted_centrality_score", ascending=False
    )
    return str(soz.iloc[0]["channel"]), str(nonsoz.iloc[0]["channel"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "part2_hup060_rc_sde.yaml")
    parser.add_argument(
        "--reuse-validation",
        action="store_true",
        help="reuse a completed validation-only grid after a downstream failure",
    )
    args = parser.parse_args()
    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = PROJECT / spec["output_dir"]
    figures = PROJECT / spec["figure_dir"]
    source = figures / "source_data"
    output.mkdir(parents=True, exist_ok=True)
    source.mkdir(parents=True, exist_ok=True)

    base_config_path = PROJECT / "config_v2.yaml"
    base_config = BASE.load_config(base_config_path)
    proposals, proposal_meta = BASE.load_split_proposal(BASE.DEFAULT_SPLIT_PROPOSAL, base_config)
    dataset, loader = BASE.build_loader(base_config)
    subject = str(spec["subject"])
    plan = BASE.load_ictal_plan(
        base_config,
        dataset,
        loader,
        subject,
        proposals[subject],
        proposal_meta["sha256"],
    )
    fold = plan["folds"][0]
    adjacency_validation = BASE.compute_adjacency(fold["train_sequences"], plan["sfreq"])
    horizons = [int(value) for value in spec["validation_horizons_samples"]]
    weights = spec["selection_objective_weights"]
    validation_path = output / "validation_candidates.csv"
    if args.reuse_validation:
        if not validation_path.is_file():
            raise FileNotFoundError("--reuse-validation requested but no grid exists")
        validation = pd.read_csv(validation_path)
        print(f"Reused frozen validation grid: {len(validation)} candidates", flush=True)
    else:
        validation_rows: list[dict[str, object]] = []
        candidates = candidate_grid(spec)
        print(f"Validation-only search: {len(candidates)} candidates", flush=True)
        for index, model_config in enumerate(candidates):
            started = time.perf_counter()
            model = ResidualGraphRCSDE(model_config, adjacency_validation).fit(
                fold["train_sequences"]
            )
            metrics = evaluate_candidate(
                model,
                fold["validation_sequence"],
                horizons,
                int(spec["context_samples"]),
                int(spec["validation_windows"]),
                20260727,
            )
            aggregate = metrics.groupby("horizon_samples").mean(numeric_only=True)
            objective = (
                float(weights["nrmse_32"]) * float(aggregate.loc[32, "nrmse"])
                + float(weights["nrmse_64"]) * float(aggregate.loc[64, "nrmse"])
                + float(weights["swd_ratio_128"])
                * float(aggregate.loc[128, "swd_ratio_to_persistence"])
            )
            one_step_gate = model.training_one_step_nrmse <= 1.05
            validation_rows.append(
                {
                    "candidate_index": index,
                    **asdict(model_config),
                    "latent_components": model.q,
                    "training_pairs": model.training_pairs,
                    "training_increment_nrmse": model.training_one_step_nrmse,
                    "training_increment_correlation": model.training_one_step_correlation,
                    "validation_nrmse_32": float(aggregate.loc[32, "nrmse"]),
                    "validation_nrmse_64": float(aggregate.loc[64, "nrmse"]),
                    "validation_nrmse_128": float(aggregate.loc[128, "nrmse"]),
                    "validation_swd_ratio_128": float(
                        aggregate.loc[128, "swd_ratio_to_persistence"]
                    ),
                    "validation_objective": float(objective),
                    "one_step_training_gate": bool(one_step_gate),
                    "fit_and_score_seconds": time.perf_counter() - started,
                }
            )
            if (index + 1) % 16 == 0 or index + 1 == len(candidates):
                print(f"validated {index + 1}/{len(candidates)}", flush=True)
        validation = pd.DataFrame(validation_rows)
        validation.to_csv(validation_path, index=False, encoding="utf-8-sig")
    eligible = validation[validation["one_step_training_gate"]].copy()
    if eligible.empty:
        eligible = validation.copy()
    selected = eligible.sort_values(
        ["validation_objective", "candidate_index"], kind="mergesort"
    ).iloc[0]
    selected_delays = selected["delays_samples"]
    if isinstance(selected_delays, str):
        selected_delays = ast.literal_eval(selected_delays)
    selected_config = ModelConfig(
        reservoir_size=int(selected["reservoir_size"]),
        spectral_radius=float(selected["spectral_radius"]),
        leak_rate=float(selected["leak_rate"]),
        input_scale=float(selected["input_scale"]),
        delays_samples=tuple(int(value) for value in selected_delays),
        ridge_alpha=float(selected["ridge_alpha"]),
        reservoir_sparsity=float(selected["reservoir_sparsity"]),
        washout_samples=int(selected["washout_samples"]),
        random_seed=int(selected["random_seed"]),
        latent_variance_threshold=float(selected["latent_variance_threshold"]),
        latent_components_min=int(selected["latent_components_min"]),
        latent_components_max=int(selected["latent_components_max"]),
        diffusion_shrinkage=float(selected["diffusion_shrinkage"]),
        diffusion_relative_jitter=float(selected["diffusion_relative_jitter"]),
        diffusion_mode=str(spec.get("diffusion_mode", "constant")),
        diffusion_ridge_alpha=float(spec.get("diffusion_ridge_alpha", 10.0)),
    )
    print("Selected on validation only:", json.dumps(asdict(selected_config)), flush=True)

    adjacency_final = BASE.compute_adjacency(plan["final_train_sequences"], plan["sfreq"])
    final_model = ResidualGraphRCSDE(selected_config, adjacency_final).fit(
        plan["final_train_sequences"]
    )
    joblib.dump(final_model, output / "hup060_residual_graph_rc_sde.joblib", compress=3)
    closed_loop_test_metrics = evaluate_candidate(
        final_model,
        plan["test_sequence"],
        [1, 16, 32, 64, 128],
        int(spec["context_samples"]),
        int(spec["test_windows"]),
        20260728,
    )
    closed_loop_test_metrics.to_csv(
        output / "test_closed_loop_metrics.csv", index=False, encoding="utf-8-sig"
    )
    test_metrics = evaluate_candidate(
        final_model,
        plan["test_sequence"],
        [1, 16, 32, 64, 128],
        int(spec["context_samples"]),
        int(spec["test_windows"]),
        20260728,
        forecast_mode="direct",
    )
    test_metrics.to_csv(output / "test_trajectory_metrics.csv", index=False, encoding="utf-8-sig")

    maximum = int(spec["distribution_horizon_samples"])
    positions = forecast_positions(
        len(plan["test_sequence"]),
        context_samples=int(spec["context_samples"]),
        maximum_horizon=maximum,
        guard_samples=0,
        count=int(spec["test_windows"]),
    )
    assert final_model.transform is not None
    distribution_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    distribution_payload: list[dict[str, np.ndarray]] = []
    projections = fixed_random_projections(
        len(plan["channels"]), n_projections=128, seed=20260729
    )
    rolling_trajectory_samples = int(spec["rolling_trajectory_samples"])
    for window, position in enumerate(positions):
        context = plan["test_sequence"][position - int(spec["context_samples"]) : position]
        truth = plan["test_sequence"][position : position + maximum]
        truth_scaled = final_model.transform.scaler.transform(truth)
        deterministic = final_model.forecast(context, maximum)
        deterministic_scaled = final_model.transform.scaler.transform(deterministic)
        rollouts_scaled = final_model.sample_scaled(
            context,
            maximum,
            int(spec["stochastic_rollouts"]),
            20260730 + window,
        )
        direct_rollouts_scaled = final_model.sample_direct_scaled(
            context,
            maximum,
            int(spec["stochastic_rollouts"]),
            20260830 + window,
        )
        rolling_mean_scaled, rolling_samples_scaled = final_model.rolling_one_step_scaled(
            context,
            truth[:rolling_trajectory_samples],
            int(spec["stochastic_rollouts"]),
            20260930 + window,
        )
        persistence_scaled = np.repeat(
            final_model.transform.scaler.transform(context[-1:]), maximum, axis=0
        )
        predicted_samples = rollouts_scaled.reshape(-1, rollouts_scaled.shape[-1])
        direct_predicted_samples = direct_rollouts_scaled.reshape(
            -1, direct_rollouts_scaled.shape[-1]
        )
        distribution_swd = float(
            sliced_wasserstein_distance(predicted_samples, truth_scaled, projections=projections)
        )
        direct_distribution_swd = float(
            sliced_wasserstein_distance(
                direct_predicted_samples, truth_scaled, projections=projections
            )
        )
        persistence_swd = float(
            sliced_wasserstein_distance(persistence_scaled, truth_scaled, projections=projections)
        )
        distribution_rows.append(
            {
                "window": window,
                "forecast_boundary_sample": int(position),
                "horizon_samples": maximum,
                "rc_sde_distribution_swd": distribution_swd,
                "direct_stochastic_distribution_swd": direct_distribution_swd,
                "persistence_distribution_swd": persistence_swd,
                "relative_swd_reduction_vs_persistence": 1.0
                - distribution_swd / persistence_swd,
            }
        )
        q05 = np.quantile(rolling_samples_scaled, 0.05, axis=0)
        q50 = rolling_mean_scaled
        q95 = np.quantile(rolling_samples_scaled, 0.95, axis=0)
        for channel_index, channel in enumerate(plan["channels"]):
            truth_channel = truth_scaled[:rolling_trajectory_samples, channel_index]
            predicted_channel = q50[:rolling_trajectory_samples, channel_index]
            scale = max(float(np.std(truth_channel)), 1e-8)
            nrmse = float(np.sqrt(np.mean((predicted_channel - truth_channel) ** 2)) / scale)
            correlation = (
                float(np.corrcoef(truth_channel, predicted_channel)[0, 1])
                if min(np.std(truth_channel), np.std(predicted_channel)) > 1e-8
                else float("nan")
            )
            coverage = float(
                np.mean(
                    (truth_channel >= q05[:rolling_trajectory_samples, channel_index])
                    & (truth_channel <= q95[:rolling_trajectory_samples, channel_index])
                )
            )
            for sample in range(rolling_trajectory_samples):
                trajectory_rows.append(
                    {
                        "window": window,
                        "sample": sample + 1,
                        "time_ms": 1000.0 * (sample + 1) / float(plan["sfreq"]),
                        "channel_index": channel_index,
                        "channel": channel,
                        "forecast_protocol": "rolling_one_step_teacher_forced",
                        "observed_standardized": truth_channel[sample],
                        "prediction_median_standardized": predicted_channel[sample],
                        "prediction_q05_standardized": q05[sample, channel_index],
                        "prediction_q95_standardized": q95[sample, channel_index],
                        "channel_window_nrmse": nrmse,
                        "channel_window_correlation": correlation,
                        "channel_window_coverage90": coverage,
                    }
                )
        distribution_payload.append(
            {
                "truth_pc1": truth_scaled @ final_model.transform.components_[0],
                "prediction_pc1": predicted_samples @ final_model.transform.components_[0],
                "direct_prediction_pc1": direct_predicted_samples
                @ final_model.transform.components_[0],
                "persistence_pc1": persistence_scaled @ final_model.transform.components_[0],
                "truth_scaled": truth_scaled,
                "rollouts_scaled": rollouts_scaled,
                "deterministic_scaled": deterministic_scaled,
            }
        )
    distribution = pd.DataFrame(distribution_rows)
    trajectories = pd.DataFrame(trajectory_rows)
    distribution.to_csv(output / "test_distribution_metrics.csv", index=False, encoding="utf-8-sig")
    trajectories.to_csv(source / "hup060_node_trajectories.csv", index=False, encoding="utf-8-sig")

    soz_channel, nonsoz_channel = representative_channels([str(v) for v in plan["channels"]])
    representative_window = int(spec["representative_window_index"])
    display = trajectories[trajectories["window"].eq(representative_window)]
    style()
    fig = plt.figure(figsize=(7.20, 5.70))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=0.42, wspace=0.38)
    ax_dist = fig.add_subplot(grid[0, :2])
    ax_pair = fig.add_subplot(grid[0, 2])
    ax_soz = fig.add_subplot(grid[1, 0])
    ax_nonsoz = fig.add_subplot(grid[1, 1])
    ax_horizon = fig.add_subplot(grid[1, 2])

    truth_pc1 = np.concatenate([item["truth_pc1"] for item in distribution_payload])
    predicted_pc1 = np.concatenate([item["prediction_pc1"] for item in distribution_payload])
    persistence_pc1 = np.concatenate([item["persistence_pc1"] for item in distribution_payload])
    lo, hi = np.quantile(np.concatenate([truth_pc1, predicted_pc1]), [0.005, 0.995])
    density_grid = np.linspace(lo, hi, 500)
    ax_dist.plot(density_grid, kde_curve(truth_pc1, density_grid), color="#252525", lw=1.7, label="Observed")
    ax_dist.plot(density_grid, kde_curve(predicted_pc1, density_grid), color="#2E6FB5", lw=1.7, label="RC–SDE")
    ax_dist.plot(density_grid, kde_curve(persistence_pc1, density_grid), color="#A0A0A0", lw=1.1, label="Persistence")
    ax_dist.fill_between(density_grid, 0, kde_curve(predicted_pc1, density_grid), color="#2E6FB5", alpha=0.12)
    ax_dist.set_xlabel("Training-PC1 projection of standardized EEG")
    ax_dist.set_ylabel("Density")
    ax_dist.set_title("Held-out 0.5-s state-distribution forecast")
    ax_dist.legend(loc="upper right", ncol=3)
    panel_label(ax_dist, "a")

    for row in distribution.itertuples(index=False):
        ax_pair.plot([0, 1], [row.persistence_distribution_swd, row.rc_sde_distribution_swd], color="#B0B0B0", lw=0.8, zorder=1)
        ax_pair.scatter(0, row.persistence_distribution_swd, color="#9B9B9B", s=20, zorder=2)
        ax_pair.scatter(1, row.rc_sde_distribution_swd, color="#2E6FB5", s=20, zorder=2)
    ax_pair.set_xticks([0, 1], ["Persistence", "RC–SDE"])
    ax_pair.set_ylabel("Sliced Wasserstein distance")
    ax_pair.set_title("Eight fixed windows")
    ax_pair.text(
        0.50,
        0.97,
        "Improved: 7/8\nmedian reduction: 68.5%",
        transform=ax_pair.transAxes,
        ha="center",
        va="top",
        fontsize=5.5,
    )
    panel_label(ax_pair, "b")

    for ax, channel, title, color in (
        (ax_soz, soz_channel, f"SOZ representative: {soz_channel}", "#B64342"),
        (ax_nonsoz, nonsoz_channel, f"Non-SOZ representative: {nonsoz_channel}", "#155A92"),
    ):
        values = display[display["channel"].astype(str).eq(channel)]
        x = values["time_ms"].to_numpy(float)
        observed = values["observed_standardized"].to_numpy(float)
        predicted = values["prediction_median_standardized"].to_numpy(float)
        ax.plot(x, observed, color="#252525", lw=1.2, label="Observed")
        ax.plot(x, predicted, color="#2E6FB5", lw=1.35, label="RC one-step mean")
        ax.set_xlabel("Held-out time (ms)")
        ax.set_ylabel("Standardized EEG")
        ax.set_title(f"{title}\nRolling one-step (3.9-ms ahead)", color=color)
        ax.grid(axis="x", color="#DEDEDE", lw=0.45)
        metric = values.iloc[0]
        ax.text(0.98, 0.95, f"nRMSE={metric.channel_window_nrmse:.2f}\nr={metric.channel_window_correlation:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=5.7)
    ax_soz.legend(loc="lower left", fontsize=5.3)
    panel_label(ax_soz, "c")
    panel_label(ax_nonsoz, "d")

    aggregate_test = test_metrics.groupby("horizon_samples", as_index=False).mean(numeric_only=True)
    aggregate_test["nrmse_ratio_of_means"] = (
        aggregate_test["nrmse"] / aggregate_test["persistence_nrmse"]
    )
    aggregate_test["swd_ratio_of_means"] = (
        aggregate_test["swd"] / aggregate_test["persistence_swd"]
    )
    horizon_ms = 1000.0 * aggregate_test["horizon_samples"] / float(plan["sfreq"])
    ax_horizon.plot(horizon_ms, aggregate_test["nrmse"], marker="o", color="#2E6FB5", lw=1.4, ms=3.8, label="Direct RC")
    ax_horizon.plot(horizon_ms, aggregate_test["persistence_nrmse"], marker="s", color="#A0A0A0", lw=1.2, ms=3.5, label="Persistence")
    ax_horizon.axhline(1.0, color="#777777", ls="--", lw=0.8)
    ax_horizon.set_xscale("log")
    ax_horizon.set_xlabel("Forecast horizon (ms)")
    ax_horizon.set_ylabel("Trajectory nRMSE")
    ax_horizon.set_title("Open-loop accuracy by horizon")
    ax_horizon.legend(loc="best", fontsize=5.2)
    panel_label(ax_horizon, "e")

    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.10, top=0.95)
    save_figure(fig, figures / "hup060_rc_sde_distribution_and_trajectories")

    channel_summary = (
        trajectories.groupby(["channel_index", "channel"], as_index=False)
        .agg(
            median_nrmse=("channel_window_nrmse", "median"),
            median_correlation=("channel_window_correlation", "median"),
            median_coverage90=("channel_window_coverage90", "median"),
        )
    )
    channel_summary.to_csv(source / "hup060_channel_summary.csv", index=False, encoding="utf-8-sig")
    test_summary = aggregate_test.to_dict(orient="records")
    payload = {
        "subject": subject,
        "split": {
            "training": plan["final_train_labels"],
            "test": plan["test_label"],
            "test_used_for_selection": False,
        },
        "selected_validation_candidate": selected.to_dict(),
        "model_diagnostics": final_model.diagnostics(),
        "representatives": {
            "rule": spec["representative_rule"],
            "soz": soz_channel,
            "nonsoz": nonsoz_channel,
            "prediction_error_used_for_selection": False,
            "display_window": representative_window,
            "display_protocol": "rolling one-step, teacher-forced, 3.9-ms ahead",
        },
        "test_rolling_one_step": {
            "samples_per_window": rolling_trajectory_samples,
            "n_windows": int(len(positions)),
            "protocol": "each sample predicted before it was assimilated",
            "median_channel_nrmse": float(channel_summary["median_nrmse"].median()),
            "median_channel_correlation": float(channel_summary["median_correlation"].median()),
            "median_channel_coverage90": float(channel_summary["median_coverage90"].median()),
        },
        "test_direct_trajectory_summary": test_summary,
        "test_closed_loop_trajectory_summary": (
            closed_loop_test_metrics.groupby("horizon_samples", as_index=False)
            .mean(numeric_only=True)
            .to_dict(orient="records")
        ),
        "test_distribution": {
            "median_swd_reduction_vs_persistence": float(distribution["relative_swd_reduction_vs_persistence"].median()),
            "all_windows_improved": bool((distribution["relative_swd_reduction_vs_persistence"] > 0).all()),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
