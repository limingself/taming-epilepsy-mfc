"""Structured neural feedback for sequential particle covariance steering.

The actor operates on the frozen state-dependent ictal Graph--RC SDE.  Its
interpretable backbone is a smooth time-varying mean/deviation feedback gain;
a small neural residual corrects nonlinear and non-Gaussian effects.  Forward
particles are the empirical Fokker--Planck solver.  This module contains no
recorded seizure-future input API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
import torch
from torch import Tensor, nn

from .causal_ltv_particle_rollout import (
    FrozenIctalGraphRCBatchStepper,
    LTVPolicyAction,
)


def cubic_bspline_basis(
    horizon: int,
    basis_count: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Return a clamped uniform cubic B-spline design matrix."""

    if int(horizon) < 2 or int(basis_count) < 4:
        raise ValueError("horizon>=2 and basis_count>=4 are required")
    degree = 3
    internal_count = int(basis_count) - degree - 1
    internal = (
        np.linspace(0.0, 1.0, internal_count + 2)[1:-1]
        if internal_count > 0
        else np.empty(0, dtype=np.float64)
    )
    knots = np.concatenate(
        [np.zeros(degree + 1), internal, np.ones(degree + 1)]
    )
    times = np.linspace(0.0, 1.0, int(horizon), dtype=np.float64)
    basis = np.zeros((int(horizon), int(basis_count)), dtype=np.float64)
    for index in range(int(basis_count)):
        coefficient = np.zeros(int(basis_count), dtype=np.float64)
        coefficient[index] = 1.0
        basis[:, index] = BSpline(
            knots, coefficient, degree, extrapolate=False
        )(times)
    basis = np.nan_to_num(basis, nan=0.0)
    basis /= np.maximum(basis.sum(axis=1, keepdims=True), 1e-12)
    return torch.as_tensor(basis, dtype=dtype, device=device)


def _mlp(input_dim: int, hidden_size: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        nn.LayerNorm(hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, output_dim),
    )


@dataclass
class ControlledParticleRollout:
    scaled: Tensor
    controls: Tensor
    commands: Tensor
    conditional_std: Tensor
    noise: Tensor


class StructuredSplineCovarianceActor(nn.Module):
    """Smooth mean/deviation actor with a bounded neural feedback residual.

    The common channel tracks the interictal mean.  The exactly centered
    deviation channel modifies covariance without injecting a population-mean
    command.  Both gain families use a small cubic B-spline parameterization;
    the residual network is trust-region bounded by ``residual_scale``.
    Delivered control follows an explicit first-order actuator state, so
    smoothness is part of the controlled dynamics rather than post-processing.
    """

    def __init__(
        self,
        stepper: FrozenIctalGraphRCBatchStepper,
        reference_mean_path: Tensor | np.ndarray,
        reference_variance_path: Tensor | np.ndarray,
        reference_scale: Tensor | np.ndarray,
        base_gain: Tensor | np.ndarray,
        actuated_channel_indices: Tensor | np.ndarray,
        *,
        horizon: int,
        basis_count: int = 8,
        hidden_size: int = 64,
        amplitude_limit: float = 1.8,
        actuator_alpha: float = 0.25,
        residual_scale: float = 0.15,
        local_gain_initial_fraction: float = 0.0,
        local_gain_maximum_fraction: float = 0.05,
        markov_feature_center: Tensor | np.ndarray | None = None,
        markov_feature_scale: Tensor | np.ndarray | None = None,
        markov_residual_scale: float = 0.0,
        markov_hidden_size: int = 96,
        maximum_slew: float | None = None,
    ) -> None:
        super().__init__()
        if amplitude_limit <= 0:
            raise ValueError("amplitude_limit must be positive")
        if not 0.0 < actuator_alpha <= 1.0:
            raise ValueError("actuator_alpha must be in (0,1]")
        if not 0.0 <= residual_scale <= 1.0:
            raise ValueError("residual_scale must be in [0,1]")
        if not 0.0 < local_gain_maximum_fraction <= 1.0:
            raise ValueError("local gain maximum fraction must be in (0,1]")
        if not abs(float(local_gain_initial_fraction)) < local_gain_maximum_fraction:
            raise ValueError(
                "absolute local gain initial fraction must be below the maximum"
            )
        if not 0.0 <= markov_residual_scale <= 1.0:
            raise ValueError("markov_residual_scale must be in [0,1]")
        if int(markov_hidden_size) < 1:
            raise ValueError("markov_hidden_size must be positive")
        if maximum_slew is not None and float(maximum_slew) <= 0.0:
            raise ValueError("maximum_slew must be positive when supplied")
        if (markov_feature_center is None) != (markov_feature_scale is None):
            raise ValueError(
                "markov feature center and scale must be supplied together"
            )
        if markov_residual_scale > 0.0 and markov_feature_scale is None:
            raise ValueError(
                "positive markov_residual_scale requires Markov normalization"
            )
        self.stepper = stepper
        self.horizon = int(horizon)
        self.amplitude_limit = float(amplitude_limit)
        self.actuator_alpha = float(actuator_alpha)
        self.residual_scale = float(residual_scale)
        self.local_gain_maximum_fraction = float(
            local_gain_maximum_fraction
        )
        self.markov_residual_scale = float(markov_residual_scale)
        # Functional ablations may disable particle-specific feedback while
        # preserving the common control branch and the exact same network
        # parameterization.  This scalar is intentionally not a parameter or
        # state-dict entry, so legacy checkpoints remain byte-compatible.
        self.deviation_feedback_scale = 1.0
        self.maximum_slew = (
            None if maximum_slew is None else float(maximum_slew)
        )
        dtype = stepper.dtype
        device = stepper.device
        n = stepper.n_channels
        m = stepper.actuator_dim

        mean_path = torch.as_tensor(
            reference_mean_path, dtype=dtype, device=device
        )
        variance_path = torch.as_tensor(
            reference_variance_path, dtype=dtype, device=device
        )
        scale = torch.as_tensor(reference_scale, dtype=dtype, device=device)
        gain = torch.as_tensor(base_gain, dtype=dtype, device=device)
        actuated = torch.as_tensor(
            actuated_channel_indices, dtype=torch.long, device=device
        )
        if mean_path.shape != (self.horizon, n):
            raise ValueError("reference_mean_path has invalid shape")
        if variance_path.shape != (self.horizon, n):
            raise ValueError("reference_variance_path has invalid shape")
        if scale.shape != (n,) or torch.any(scale <= 0):
            raise ValueError("reference_scale has invalid shape")
        if gain.shape != (m, n):
            raise ValueError("base_gain has invalid shape")
        if actuated.shape != (m,):
            raise ValueError("actuated_channel_indices has invalid shape")
        if torch.any(actuated < 0) or torch.any(actuated >= n):
            raise ValueError("actuated channel index is outside output range")
        self.register_buffer("reference_mean_path", mean_path)
        self.register_buffer(
            "reference_variance_path", variance_path.clamp_min(1e-5)
        )
        self.register_buffer("reference_scale", scale)
        self.register_buffer("base_gain", gain)
        self.register_buffer("actuated_channel_indices", actuated)
        if markov_feature_scale is None:
            markov_center = torch.empty(0, dtype=dtype, device=device)
            markov_scale = torch.empty(0, dtype=dtype, device=device)
        else:
            markov_center = torch.as_tensor(
                markov_feature_center, dtype=dtype, device=device
            )
            markov_scale = torch.as_tensor(
                markov_feature_scale, dtype=dtype, device=device
            )
            if markov_center.shape != (stepper.adapter.state_dim,):
                raise ValueError("markov_feature_center has invalid shape")
            if markov_scale.shape != (stepper.adapter.state_dim,):
                raise ValueError("markov_feature_scale has invalid shape")
            if torch.any(~torch.isfinite(markov_scale)) or torch.any(
                markov_scale <= 0
            ):
                raise ValueError("markov_feature_scale must be finite and positive")
        self.register_buffer("markov_feature_center", markov_center)
        self.register_buffer("markov_feature_scale", markov_scale)
        direct_effect = (
            stepper.adapter.control_step_scale
            * stepper.adapter.control_channel_map[
                torch.arange(m, device=device), actuated
            ]
        )
        if torch.any(direct_effect.abs() < 1e-6):
            raise ValueError("an actuator has negligible effect on its own channel")
        self.register_buffer(
            "local_inverse_effect",
            1.0 / (self.actuator_alpha * direct_effect),
        )
        self.register_buffer(
            "spline_basis",
            cubic_bspline_basis(
                self.horizon,
                int(basis_count),
                dtype=dtype,
                device=device,
            ),
        )
        self.mean_gain_delta = nn.Parameter(
            torch.zeros(int(basis_count), m, n, dtype=dtype, device=device)
        )
        self.deviation_gain_delta = nn.Parameter(
            torch.zeros(int(basis_count), m, n, dtype=dtype, device=device)
        )
        normalized_initial = (
            float(local_gain_initial_fraction)
            / self.local_gain_maximum_fraction
        )
        initial_logit = float(np.arctanh(normalized_initial))
        self.local_deviation_gain_logits = nn.Parameter(
            torch.full(
                (int(basis_count), m),
                initial_logit,
                dtype=dtype,
                device=device,
            )
        )

        common_dim = 2 * n + 3 + m
        deviation_dim = 3 * n + 3 + m
        self.common_residual = _mlp(common_dim, hidden_size, m).to(
            dtype=dtype, device=device
        )
        self.deviation_residual = _mlp(deviation_dim, hidden_size, m).to(
            dtype=dtype, device=device
        )
        if self.markov_residual_scale > 0.0:
            common_markov_dim = stepper.adapter.state_dim + common_dim
            deviation_markov_dim = stepper.adapter.state_dim + deviation_dim
            self.common_markov_residual = _mlp(
                common_markov_dim, int(markov_hidden_size), m
            ).to(dtype=dtype, device=device)
            self.deviation_markov_residual = _mlp(
                deviation_markov_dim, int(markov_hidden_size), m
            ).to(dtype=dtype, device=device)
        else:
            self.common_markov_residual = None
            self.deviation_markov_residual = None
        # Neutral residual initialization makes the initial policy exactly the
        # interpretable full-output feedback backbone.
        nn.init.zeros_(self.common_residual[-1].weight)
        nn.init.zeros_(self.common_residual[-1].bias)
        nn.init.zeros_(self.deviation_residual[-1].weight)
        nn.init.zeros_(self.deviation_residual[-1].bias)
        if self.common_markov_residual is not None:
            nn.init.zeros_(self.common_markov_residual[-1].weight)
            nn.init.zeros_(self.common_markov_residual[-1].bias)
        if self.deviation_markov_residual is not None:
            nn.init.zeros_(self.deviation_markov_residual[-1].weight)
            nn.init.zeros_(self.deviation_markov_residual[-1].bias)

    def gains(self, step: int) -> tuple[Tensor, Tensor]:
        basis = self.spline_basis[int(step)]
        mean_gain = self.base_gain + torch.einsum(
            "b,bmn->mn", basis, self.mean_gain_delta
        )
        deviation_gain = self.base_gain + torch.einsum(
            "b,bmn->mn", basis, self.deviation_gain_delta
        )
        return mean_gain, deviation_gain

    def local_gain_fraction(self, step: int) -> Tensor:
        basis = self.spline_basis[int(step)]
        logits = torch.einsum(
            "b,bm->m", basis, self.local_deviation_gain_logits
        )
        return self.local_gain_maximum_fraction * torch.tanh(logits)

    def _time_features(self, step: int, particles: int) -> Tensor:
        fraction = self.reference_mean_path.new_tensor(
            float(step) / max(self.horizon - 1, 1)
        )
        angle = 2.0 * torch.pi * fraction
        features = torch.stack([fraction, torch.sin(angle), torch.cos(angle)])
        return features[None].expand(particles, -1)

    def propose(
        self,
        *,
        step: int,
        markov_particles: Tensor,
        previous_control: Tensor,
        current_scaled: Tensor,
    ) -> LTVPolicyAction:
        del current_scaled
        index = int(step)
        if not 0 <= index < self.horizon:
            raise IndexError("actor step is outside horizon")
        particles = int(markov_particles.shape[0])
        mean_latent, _ = self.stepper.conditional_mean_and_std(markov_particles)
        predicted_mean_output = (
            mean_latent @ self.stepper.adapter.components
            + self.stepper.adapter.pca_mean
        )
        population_mean = predicted_mean_output.mean(dim=0, keepdim=True)
        deviation = predicted_mean_output - population_mean
        mean_error = population_mean - self.reference_mean_path[index : index + 1]
        population_variance = predicted_mean_output.var(
            dim=0, unbiased=False
        ).clamp_min(1e-5)
        log_variance_error = (
            torch.log(population_variance)
            - torch.log(self.reference_variance_path[index])
        )[None]
        mean_gain, deviation_gain = self.gains(index)
        previous_mean = previous_control.mean(dim=0, keepdim=True)
        previous_deviation = previous_control - previous_mean
        time = self._time_features(index, particles)

        common_input = torch.cat(
            [
                mean_error / self.reference_scale[None],
                log_variance_error,
                time[:1],
                previous_mean / self.amplitude_limit,
            ],
            dim=1,
        )
        common_raw = -mean_error @ mean_gain.T
        common_raw = common_raw + self.residual_scale * self.amplitude_limit * torch.tanh(
            self.common_residual(common_input)
        )
        if self.common_markov_residual is not None:
            markov_mean = markov_particles.mean(dim=0, keepdim=True)
            normalized_markov_mean = (
                markov_mean - self.markov_feature_center[None]
            ) / self.markov_feature_scale[None]
            common_markov_input = torch.cat(
                [normalized_markov_mean, common_input], dim=1
            )
            common_raw = common_raw + (
                self.markov_residual_scale
                * self.amplitude_limit
                * torch.tanh(self.common_markov_residual(common_markov_input))
            )

        deviation_input = torch.cat(
            [
                deviation / self.reference_scale[None],
                mean_error.expand(particles, -1) / self.reference_scale[None],
                log_variance_error.expand(particles, -1),
                time,
                previous_deviation / self.amplitude_limit,
            ],
            dim=1,
        )
        deviation_raw = -deviation @ deviation_gain.T
        local_fraction = self.local_gain_fraction(index)
        local_deviation = deviation[:, self.actuated_channel_indices]
        deviation_raw = deviation_raw - (
            local_fraction[None]
            * local_deviation
            * self.local_inverse_effect[None]
        )
        deviation_raw = deviation_raw + self.residual_scale * self.amplitude_limit * torch.tanh(
            self.deviation_residual(deviation_input)
        )
        if self.deviation_markov_residual is not None:
            markov_mean = markov_particles.mean(dim=0, keepdim=True)
            normalized_markov_deviation = (
                markov_particles - markov_mean
            ) / self.markov_feature_scale[None]
            deviation_markov_input = torch.cat(
                [normalized_markov_deviation, deviation_input], dim=1
            )
            deviation_raw = deviation_raw + (
                self.markov_residual_scale
                * self.amplitude_limit
                * torch.tanh(
                    self.deviation_markov_residual(deviation_markov_input)
                )
            )

        common_command = self.amplitude_limit * torch.tanh(
            common_raw / self.amplitude_limit
        )
        raw_deviation_command = self.amplitude_limit * torch.tanh(
            deviation_raw / self.amplitude_limit
        )
        centered = raw_deviation_command - raw_deviation_command.mean(
            dim=0, keepdim=True
        )
        maximum = centered.abs().amax(dim=0, keepdim=True)
        headroom = self.amplitude_limit - common_command.abs()
        scale = torch.minimum(
            torch.ones_like(maximum), headroom / maximum.clamp_min(1e-8)
        )
        deviation_command = centered * scale * float(
            self.deviation_feedback_scale
        )
        command = common_command.expand(particles, -1) + deviation_command
        filtered = (
            (1.0 - self.actuator_alpha) * previous_control
            + self.actuator_alpha * command
        )
        if self.maximum_slew is None:
            applied = filtered
        else:
            requested_step = filtered - previous_control
            applied = previous_control + self.maximum_slew * torch.tanh(
                requested_step / self.maximum_slew
            )
        applied_common = applied.mean(dim=0, keepdim=True)
        applied_deviation = applied - applied_common
        return LTVPolicyAction(
            command=command,
            applied_control=applied,
            common_command=applied_common,
            deviation_command=applied_deviation,
            variance_contraction_gate=applied.new_tensor(1.0),
        )


def controlled_particle_rollout(
    stepper: FrozenIctalGraphRCBatchStepper,
    actor: StructuredSplineCovarianceActor,
    initial_markov: Tensor,
    standard_normal: Tensor,
) -> ControlledParticleRollout:
    """Differentiably propagate the empirical controlled FP particle law."""

    noise = torch.as_tensor(
        standard_normal, dtype=stepper.dtype, device=stepper.device
    )
    if noise.ndim != 3 or noise.shape[2] != stepper.q:
        raise ValueError("standard_normal must have shape [particle,time,q]")
    particles, horizon, _ = noise.shape
    if horizon != actor.horizon:
        raise ValueError("noise and actor horizons differ")
    state = stepper.repeat_initial(initial_markov, particles)
    previous = state.new_zeros(particles, stepper.actuator_dim)
    current_output = stepper.current_output(state)
    outputs = [current_output]
    controls: list[Tensor] = []
    commands: list[Tensor] = []
    conditional_std: list[Tensor] = []
    for step in range(horizon):
        action = actor.propose(
            step=step,
            markov_particles=state,
            previous_control=previous,
            current_scaled=current_output,
        )
        transition = stepper.step(
            state, action.applied_control, noise[:, step]
        )
        state = transition.next_markov
        current_output = transition.next_scaled
        previous = action.applied_control
        outputs.append(current_output)
        controls.append(previous)
        commands.append(action.command)
        conditional_std.append(transition.conditional_std)
    return ControlledParticleRollout(
        scaled=torch.stack(outputs, dim=1),
        controls=torch.stack(controls, dim=1),
        commands=torch.stack(commands, dim=1),
        conditional_std=torch.stack(conditional_std, dim=1),
        noise=noise,
    )


def uncontrolled_particle_rollout(
    stepper: FrozenIctalGraphRCBatchStepper,
    initial_markov: Tensor,
    standard_normal: Tensor,
) -> Tensor:
    """Propagate the frozen ictal SDE with exactly zero control."""

    noise = torch.as_tensor(
        standard_normal, dtype=stepper.dtype, device=stepper.device
    )
    if noise.ndim != 3 or noise.shape[2] != stepper.q:
        raise ValueError("standard_normal must have shape [particle,time,q]")
    particles, horizon, _ = noise.shape
    state = stepper.repeat_initial(initial_markov, particles)
    zero = state.new_zeros(particles, stepper.actuator_dim)
    outputs = [stepper.current_output(state)]
    for step in range(horizon):
        transition = stepper.step(state, zero, noise[:, step])
        state = transition.next_markov
        outputs.append(transition.next_scaled)
    return torch.stack(outputs, dim=1)


__all__ = [
    "ControlledParticleRollout",
    "StructuredSplineCovarianceActor",
    "controlled_particle_rollout",
    "cubic_bspline_basis",
    "uncontrolled_particle_rollout",
]
