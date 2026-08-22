"""Recovered pre-local/pre-Markov HUP060 structured Actor (root log line 43887).

This class is intentionally isolated from the modern Actor.  It is used only
for HUP080 stages S0--S3, exactly preserving the historical 23-key state-dict
topology.  The S3 checkpoint is migrated once into the verified modern Actor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
import torch
from torch import Tensor, nn

from mfc_pipeline.causal_ltv_particle_rollout import (
    FrozenIctalGraphRCBatchStepper,
    LTVPolicyAction,
)


def cubic_bspline_basis(horizon: int, basis_count: int, *, dtype, device) -> Tensor:
    if int(horizon) < 2 or int(basis_count) < 4:
        raise ValueError("horizon>=2 and basis_count>=4 are required")
    degree = 3
    internal_count = int(basis_count) - degree - 1
    internal = (
        np.linspace(0.0, 1.0, internal_count + 2)[1:-1]
        if internal_count > 0
        else np.empty(0, dtype=np.float64)
    )
    knots = np.concatenate([np.zeros(degree + 1), internal, np.ones(degree + 1)])
    times = np.linspace(0.0, 1.0, int(horizon), dtype=np.float64)
    basis = np.zeros((int(horizon), int(basis_count)), dtype=np.float64)
    for index in range(int(basis_count)):
        coefficient = np.zeros(int(basis_count), dtype=np.float64)
        coefficient[index] = 1.0
        basis[:, index] = BSpline(knots, coefficient, degree, extrapolate=False)(times)
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


class HistoricalStructuredSplineCovarianceActor(nn.Module):
    """The exact pre-local/pre-Markov Actor architecture used through epoch 650."""

    def __init__(
        self,
        stepper: FrozenIctalGraphRCBatchStepper,
        reference_mean_path: Tensor | np.ndarray,
        reference_variance_path: Tensor | np.ndarray,
        reference_scale: Tensor | np.ndarray,
        base_gain: Tensor | np.ndarray,
        *,
        horizon: int,
        basis_count: int = 8,
        hidden_size: int = 64,
        amplitude_limit: float = 1.8,
        actuator_alpha: float = 0.25,
        residual_scale: float = 0.15,
    ) -> None:
        super().__init__()
        if amplitude_limit <= 0 or not 0.0 < actuator_alpha <= 1.0:
            raise ValueError("invalid amplitude/actuator alpha")
        if not 0.0 <= residual_scale <= 1.0:
            raise ValueError("residual_scale must be in [0,1]")
        self.stepper = stepper
        self.horizon = int(horizon)
        self.amplitude_limit = float(amplitude_limit)
        self.actuator_alpha = float(actuator_alpha)
        self.residual_scale = float(residual_scale)
        dtype, device = stepper.dtype, stepper.device
        n, m = stepper.n_channels, stepper.actuator_dim
        mean_path = torch.as_tensor(reference_mean_path, dtype=dtype, device=device)
        variance_path = torch.as_tensor(reference_variance_path, dtype=dtype, device=device)
        scale = torch.as_tensor(reference_scale, dtype=dtype, device=device)
        gain = torch.as_tensor(base_gain, dtype=dtype, device=device)
        if mean_path.shape != (self.horizon, n) or variance_path.shape != (self.horizon, n):
            raise ValueError("reference path shape changed")
        if scale.shape != (n,) or torch.any(scale <= 0) or gain.shape != (m, n):
            raise ValueError("reference scale/base gain shape changed")
        self.register_buffer("reference_mean_path", mean_path)
        self.register_buffer("reference_variance_path", variance_path.clamp_min(1e-5))
        self.register_buffer("reference_scale", scale)
        self.register_buffer("base_gain", gain)
        self.register_buffer(
            "spline_basis",
            cubic_bspline_basis(self.horizon, basis_count, dtype=dtype, device=device),
        )
        self.mean_gain_delta = nn.Parameter(torch.zeros(basis_count, m, n, dtype=dtype, device=device))
        self.deviation_gain_delta = nn.Parameter(torch.zeros(basis_count, m, n, dtype=dtype, device=device))
        common_dim, deviation_dim = 2 * n + 3 + m, 3 * n + 3 + m
        self.common_residual = _mlp(common_dim, hidden_size, m).to(dtype=dtype, device=device)
        self.deviation_residual = _mlp(deviation_dim, hidden_size, m).to(dtype=dtype, device=device)
        nn.init.zeros_(self.common_residual[-1].weight)
        nn.init.zeros_(self.common_residual[-1].bias)
        nn.init.zeros_(self.deviation_residual[-1].weight)
        nn.init.zeros_(self.deviation_residual[-1].bias)

    def gains(self, step: int) -> tuple[Tensor, Tensor]:
        basis = self.spline_basis[int(step)]
        return (
            self.base_gain + torch.einsum("b,bmn->mn", basis, self.mean_gain_delta),
            self.base_gain + torch.einsum("b,bmn->mn", basis, self.deviation_gain_delta),
        )

    def _time_features(self, step: int, particles: int) -> Tensor:
        fraction = self.reference_mean_path.new_tensor(float(step) / max(self.horizon - 1, 1))
        angle = 2.0 * torch.pi * fraction
        return torch.stack([fraction, torch.sin(angle), torch.cos(angle)])[None].expand(particles, -1)

    def propose(self, *, step: int, markov_particles: Tensor, previous_control: Tensor, current_scaled: Tensor) -> LTVPolicyAction:
        del current_scaled
        index = int(step)
        if not 0 <= index < self.horizon:
            raise IndexError("actor step outside horizon")
        particles = int(markov_particles.shape[0])
        mean_latent, _ = self.stepper.conditional_mean_and_std(markov_particles)
        predicted = mean_latent @ self.stepper.adapter.components + self.stepper.adapter.pca_mean
        population_mean = predicted.mean(dim=0, keepdim=True)
        deviation = predicted - population_mean
        mean_error = population_mean - self.reference_mean_path[index : index + 1]
        population_variance = predicted.var(dim=0, unbiased=False).clamp_min(1e-5)
        log_variance_error = (torch.log(population_variance) - torch.log(self.reference_variance_path[index]))[None]
        mean_gain, deviation_gain = self.gains(index)
        previous_mean = previous_control.mean(dim=0, keepdim=True)
        previous_deviation = previous_control - previous_mean
        time = self._time_features(index, particles)
        common_input = torch.cat([
            mean_error / self.reference_scale[None], log_variance_error,
            time[:1], previous_mean / self.amplitude_limit,
        ], dim=1)
        common_raw = -mean_error @ mean_gain.T
        common_raw = common_raw + self.residual_scale * self.amplitude_limit * torch.tanh(self.common_residual(common_input))
        deviation_input = torch.cat([
            deviation / self.reference_scale[None],
            mean_error.expand(particles, -1) / self.reference_scale[None],
            log_variance_error.expand(particles, -1), time,
            previous_deviation / self.amplitude_limit,
        ], dim=1)
        deviation_raw = -deviation @ deviation_gain.T
        deviation_raw = deviation_raw + self.residual_scale * self.amplitude_limit * torch.tanh(self.deviation_residual(deviation_input))
        common_command = self.amplitude_limit * torch.tanh(common_raw / self.amplitude_limit)
        raw_deviation = self.amplitude_limit * torch.tanh(deviation_raw / self.amplitude_limit)
        centered = raw_deviation - raw_deviation.mean(dim=0, keepdim=True)
        maximum = centered.abs().amax(dim=0, keepdim=True)
        headroom = self.amplitude_limit - common_command.abs()
        deviation_command = centered * torch.minimum(torch.ones_like(maximum), headroom / maximum.clamp_min(1e-8))
        command = common_command.expand(particles, -1) + deviation_command
        applied = (1.0 - self.actuator_alpha) * previous_control + self.actuator_alpha * command
        applied_common = applied.mean(dim=0, keepdim=True)
        return LTVPolicyAction(
            command=command,
            applied_control=applied,
            common_command=applied_common,
            deviation_command=applied - applied_common,
            variance_contraction_gate=applied.new_tensor(1.0),
        )


def historical_controlled_particle_rollout(stepper, actor, initial_markov, standard_normal):
    noise = torch.as_tensor(standard_normal, dtype=stepper.dtype, device=stepper.device)
    if noise.ndim != 3 or noise.shape[1] != actor.horizon or noise.shape[2] != stepper.q:
        raise ValueError("noise shape changed")
    particles, horizon, _ = noise.shape
    state = stepper.repeat_initial(initial_markov, particles)
    previous = state.new_zeros(particles, stepper.actuator_dim)
    current_output = stepper.current_output(state)
    outputs, controls, commands, conditional_std = [current_output], [], [], []
    for step in range(horizon):
        action = actor.propose(step=step, markov_particles=state, previous_control=previous, current_scaled=current_output)
        transition = stepper.step(state, action.applied_control, noise[:, step])
        state, current_output, previous = transition.next_markov, transition.next_scaled, action.applied_control
        outputs.append(current_output); controls.append(previous); commands.append(action.command); conditional_std.append(transition.conditional_std)
    return ControlledParticleRollout(
        scaled=torch.stack(outputs, dim=1), controls=torch.stack(controls, dim=1),
        commands=torch.stack(commands, dim=1), conditional_std=torch.stack(conditional_std, dim=1), noise=noise,
    )


__all__ = ["HistoricalStructuredSplineCovarianceActor", "historical_controlled_particle_rollout"]
