"""Two-network full-Markov HJB--FP controller with WGAN-GP law coupling.

The two trainable modules are

1. :class:`FullMarkovHJBGenerator`, a permutation-invariant shared encoder
   with a scalar population-value head and a bounded mean--deviation policy
   head; and
2. :class:`TimeConditionedWassersteinCritic`, a time-conditioned
   Kantorovich critic trained with WGAN-GP.

The frozen state-dependent Graph--RC--SDE is not a third trainable network.
Its particle push-forward is the empirical forward Fokker--Planck solver.
The learned RC transition is a discrete 256-Hz model, so the backward
equation is enforced through a stochastic one-step Bellman (semi-Lagrangian
HJB) residual rather than an uncalibrated continuous-time PDE residual.

No patient data are loaded in this module.  Runners must provide the frozen
plant, prefix-only initial state, and disjoint reference paths explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn

from .causal_ltv_particle_rollout import (
    FrozenIctalGraphRCBatchStepper,
    LTVPolicyAction,
)


@dataclass(frozen=True)
class FullMarkovHJBWGANConfig:
    """Architecture and numerical contract for the two-network solver."""

    horizon_samples: int = 256
    amplitude_limit: float = 1.8
    actuator_alpha: float = 1.0
    local_hidden: int = 128
    local_embedding: int = 64
    global_hidden: int = 128
    context_embedding: int = 96
    critic_hidden: int = 128
    gradient_penalty: float = 10.0
    critic_drift: float = 1.0e-3
    control_energy_weight: float = 0.01
    control_smoothness_weight: float = 0.20
    mean_cost_weight: float = 1.0
    log_variance_cost_weight: float = 0.50
    wasserstein_cost_weight: float = 0.25

    def validate(self) -> None:
        if self.horizon_samples < 2:
            raise ValueError("horizon_samples must be at least two")
        if self.amplitude_limit <= 0.0:
            raise ValueError("amplitude_limit must be positive")
        if not 0.0 < self.actuator_alpha <= 1.0:
            raise ValueError("actuator_alpha must lie in (0,1]")
        widths = (
            self.local_hidden,
            self.local_embedding,
            self.global_hidden,
            self.context_embedding,
            self.critic_hidden,
        )
        if min(widths) <= 0:
            raise ValueError("network widths must be positive")
        penalties = (
            self.gradient_penalty,
            self.critic_drift,
            self.control_energy_weight,
            self.control_smoothness_weight,
            self.mean_cost_weight,
            self.log_variance_cost_weight,
            self.wasserstein_cost_weight,
        )
        if min(penalties) < 0.0:
            raise ValueError("loss weights must be non-negative")


@dataclass(frozen=True)
class HJBFPParticleRollout:
    """Differentiable empirical-FP rollout retaining the augmented state."""

    markov: Tensor
    scaled: Tensor
    controls: Tensor
    commands: Tensor
    conditional_std: Tensor
    noise: Tensor


@dataclass(frozen=True)
class WassersteinGPLoss:
    loss: Tensor
    estimate: Tensor
    gradient_penalty: Tensor
    critic_drift: Tensor
    mean_gradient_norm: Tensor


@dataclass(frozen=True)
class HJBLossAudit:
    bellman_loss: Tensor
    terminal_loss: Tensor
    residuals: Tensor
    current_values: Tensor
    target_values: Tensor


@dataclass(frozen=True)
class HamiltonianPolicyAudit:
    policy_loss: Tensor
    projected_kkt_rms: Tensor
    projected_targets: tuple[Tensor, ...]


def _time_features(
    step: int,
    horizon: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    rows: int = 1,
) -> Tensor:
    if not 0 <= int(step) <= int(horizon):
        raise IndexError("time step lies outside the horizon")
    fraction = torch.as_tensor(
        float(step) / max(int(horizon), 1), dtype=dtype, device=device
    )
    angle = 2.0 * torch.pi * fraction
    features = torch.stack((fraction, torch.sin(angle), torch.cos(angle)))
    return features[None].expand(int(rows), -1)


def _zero_last_linear(module: nn.Module) -> None:
    linears = [item for item in module.modules() if isinstance(item, nn.Linear)]
    if not linears:
        raise ValueError("module does not contain a linear layer")
    nn.init.zeros_(linears[-1].weight)
    nn.init.zeros_(linears[-1].bias)


class FullMarkovHJBGenerator(nn.Module):
    """One shared DeepSets network with population value and policy heads.

    The local encoder receives the complete 538-D Graph--RC Markov state plus
    the previous 13-D command.  The latter makes first-difference control cost
    Markov, so the effective HJB state dimension is 551 for HUP060.
    """

    def __init__(
        self,
        stepper: FrozenIctalGraphRCBatchStepper,
        reference_mean: Tensor,
        reference_variance: Tensor,
        reference_scale: Tensor,
        markov_center: Tensor,
        markov_scale: Tensor,
        config: FullMarkovHJBWGANConfig = FullMarkovHJBWGANConfig(),
    ) -> None:
        super().__init__()
        config.validate()
        horizon = int(config.horizon_samples)
        channels = int(stepper.n_channels)
        state_dim = int(stepper.adapter.state_dim)
        actuator_dim = int(stepper.actuator_dim)
        dtype = stepper.dtype
        device = stepper.device

        reference_mean = torch.as_tensor(
            reference_mean, dtype=dtype, device=device
        )
        reference_variance = torch.as_tensor(
            reference_variance, dtype=dtype, device=device
        )
        reference_scale = torch.as_tensor(
            reference_scale, dtype=dtype, device=device
        )
        markov_center = torch.as_tensor(
            markov_center, dtype=dtype, device=device
        )
        markov_scale = torch.as_tensor(
            markov_scale, dtype=dtype, device=device
        )
        if reference_mean.shape != (horizon, channels):
            raise ValueError("reference_mean has invalid shape")
        if reference_variance.shape != (horizon, channels):
            raise ValueError("reference_variance has invalid shape")
        if reference_scale.shape != (channels,):
            raise ValueError("reference_scale has invalid shape")
        if markov_center.shape != (state_dim,):
            raise ValueError("markov_center has invalid shape")
        if markov_scale.shape != (state_dim,):
            raise ValueError("markov_scale has invalid shape")
        if bool((reference_scale <= 0).any()) or bool((markov_scale <= 0).any()):
            raise ValueError("normalization scales must be positive")

        object.__setattr__(self, "_stepper", stepper)
        self.config = config
        self.horizon = horizon
        self.channels = channels
        self.state_dim = state_dim
        self.actuator_dim = actuator_dim
        self.augmented_state_dim = state_dim + actuator_dim
        self.amplitude_limit = float(config.amplitude_limit)
        self.actuator_alpha = float(config.actuator_alpha)
        self.register_buffer("reference_mean_path", reference_mean.clone())
        self.register_buffer(
            "reference_variance_path", reference_variance.clamp_min(1.0e-6).clone()
        )
        self.register_buffer("reference_scale", reference_scale.clone())
        self.register_buffer("markov_center", markov_center.clone())
        self.register_buffer("markov_scale", markov_scale.clone())

        local_input = self.augmented_state_dim
        self.local_encoder = nn.Sequential(
            nn.LayerNorm(local_input, dtype=dtype, device=device),
            nn.Linear(
                local_input,
                config.local_hidden,
                dtype=dtype,
                device=device,
            ),
            nn.SiLU(),
            nn.Linear(
                config.local_hidden,
                config.local_embedding,
                dtype=dtype,
                device=device,
            ),
            nn.SiLU(),
        )
        global_input = (
            2 * config.local_embedding
            + 2 * channels
            + 3
            + actuator_dim
        )
        self.global_encoder = nn.Sequential(
            nn.LayerNorm(global_input, dtype=dtype, device=device),
            nn.Linear(
                global_input,
                config.global_hidden,
                dtype=dtype,
                device=device,
            ),
            nn.SiLU(),
            nn.Linear(
                config.global_hidden,
                config.context_embedding,
                dtype=dtype,
                device=device,
            ),
            nn.SiLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(
                config.context_embedding,
                config.context_embedding,
                dtype=dtype,
                device=device,
            ),
            nn.SiLU(),
            nn.Linear(config.context_embedding, 1, dtype=dtype, device=device),
        )
        self.common_policy_head = nn.Sequential(
            nn.Linear(
                config.context_embedding,
                config.context_embedding,
                dtype=dtype,
                device=device,
            ),
            nn.SiLU(),
            nn.Linear(
                config.context_embedding,
                actuator_dim,
                dtype=dtype,
                device=device,
            ),
        )
        deviation_input = (
            config.local_embedding
            + config.context_embedding
            + channels
            + actuator_dim
        )
        self.deviation_policy_head = nn.Sequential(
            nn.LayerNorm(deviation_input, dtype=dtype, device=device),
            nn.Linear(
                deviation_input,
                config.global_hidden,
                dtype=dtype,
                device=device,
            ),
            nn.SiLU(),
            nn.Linear(
                config.global_hidden,
                actuator_dim,
                dtype=dtype,
                device=device,
            ),
        )
        _zero_last_linear(self.value_head)
        _zero_last_linear(self.common_policy_head)
        _zero_last_linear(self.deviation_policy_head)

    @property
    def stepper(self) -> FrozenIctalGraphRCBatchStepper:
        return object.__getattribute__(self, "_stepper")

    @property
    def parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.parameters())

    def _reference_index(self, step: int) -> int:
        return min(max(int(step), 0), self.horizon - 1)

    def _encode(
        self,
        *,
        step: int,
        markov_particles: Tensor,
        previous_control: Tensor,
        current_scaled: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if markov_particles.ndim != 2 or markov_particles.shape[1] != self.state_dim:
            raise ValueError("markov_particles has invalid shape")
        particles = int(markov_particles.shape[0])
        if previous_control.shape != (particles, self.actuator_dim):
            raise ValueError("previous_control has invalid shape")
        if current_scaled.shape != (particles, self.channels):
            raise ValueError("current_scaled has invalid shape")
        index = self._reference_index(step)
        normalized_markov = (
            markov_particles - self.markov_center[None]
        ) / self.markov_scale[None]
        augmented = torch.cat(
            [normalized_markov, previous_control / self.amplitude_limit], dim=1
        )
        local = self.local_encoder(augmented)
        pooled_mean = local.mean(dim=0, keepdim=True)
        pooled_second = local.square().mean(dim=0, keepdim=True)

        population_mean = current_scaled.mean(dim=0, keepdim=True)
        population_variance = current_scaled.var(
            dim=0, unbiased=False
        ).clamp_min(1.0e-6)
        mean_error = (
            population_mean - self.reference_mean_path[index : index + 1]
        ) / self.reference_scale[None]
        log_variance_error = (
            torch.log(population_variance)
            - torch.log(self.reference_variance_path[index])
        )[None]
        time = _time_features(
            step,
            self.horizon,
            dtype=current_scaled.dtype,
            device=current_scaled.device,
        )
        previous_mean = previous_control.mean(dim=0, keepdim=True)
        global_input = torch.cat(
            [
                pooled_mean,
                pooled_second,
                mean_error,
                log_variance_error,
                time,
                previous_mean / self.amplitude_limit,
            ],
            dim=1,
        )
        context = self.global_encoder(global_input)
        deviation = current_scaled - population_mean
        previous_deviation = previous_control - previous_mean
        return local, context, deviation, previous_deviation, population_mean

    def population_value(
        self,
        *,
        step: int,
        markov_particles: Tensor,
        previous_control: Tensor,
        current_scaled: Tensor,
    ) -> Tensor:
        """Scalar social-planner value of one empirical particle law."""

        _, context, _, _, _ = self._encode(
            step=step,
            markov_particles=markov_particles,
            previous_control=previous_control,
            current_scaled=current_scaled,
        )
        return self.value_head(context).squeeze()

    def propose(
        self,
        *,
        step: int,
        markov_particles: Tensor,
        previous_control: Tensor,
        current_scaled: Tensor,
    ) -> LTVPolicyAction:
        local, context, deviation, previous_deviation, _ = self._encode(
            step=step,
            markov_particles=markov_particles,
            previous_control=previous_control,
            current_scaled=current_scaled,
        )
        particles = int(markov_particles.shape[0])
        common_raw = self.common_policy_head(context)
        common = self.amplitude_limit * torch.tanh(
            common_raw / self.amplitude_limit
        )
        deviation_input = torch.cat(
            [
                local,
                context.expand(particles, -1),
                deviation / self.reference_scale[None],
                previous_deviation / self.amplitude_limit,
            ],
            dim=1,
        )
        deviation_raw = self.deviation_policy_head(deviation_input)
        deviation_command = self.amplitude_limit * torch.tanh(
            deviation_raw / self.amplitude_limit
        )
        centered = deviation_command - deviation_command.mean(
            dim=0, keepdim=True
        )
        maximum = centered.abs().amax(dim=0, keepdim=True)
        headroom = self.amplitude_limit - common.abs()
        rescale = torch.minimum(
            torch.ones_like(maximum), headroom / maximum.clamp_min(1.0e-8)
        )
        centered = centered * rescale
        command = common.expand(particles, -1) + centered
        applied = (
            (1.0 - self.actuator_alpha) * previous_control
            + self.actuator_alpha * command
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

    def architecture_audit(self) -> dict[str, int | float | str]:
        return {
            "network_role": "shared HJB population-value and policy generator",
            "plant_markov_dimension": self.state_dim,
            "augmented_hjb_state_dimension": self.augmented_state_dim,
            "control_dimension": self.actuator_dim,
            "particle_output_dimension": self.channels,
            "parameter_count": self.parameter_count,
            "amplitude_limit": self.amplitude_limit,
        }


class TimeConditionedWassersteinCritic(nn.Module):
    """Time-conditioned WGAN critic; distinct from the HJB value function."""

    def __init__(
        self,
        reference_center: Tensor,
        reference_scale: Tensor,
        *,
        horizon_samples: int = 256,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        center = torch.as_tensor(reference_center)
        scale = torch.as_tensor(
            reference_scale, dtype=center.dtype, device=center.device
        )
        if center.ndim != 1 or scale.shape != center.shape:
            raise ValueError("critic reference normalization has invalid shape")
        if bool((scale <= 0).any()):
            raise ValueError("critic reference scale must be positive")
        if int(horizon_samples) < 2 or int(hidden_size) <= 0:
            raise ValueError("critic dimensions are invalid")
        self.channels = int(center.numel())
        self.horizon = int(horizon_samples)
        self.register_buffer("reference_center", center.detach().clone())
        self.register_buffer("reference_scale", scale.detach().clone())
        dtype = center.dtype
        device = center.device
        self.network = nn.Sequential(
            nn.Linear(self.channels + 3, hidden_size, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Linear(hidden_size, 1, dtype=dtype, device=device),
        )

    @property
    def parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.parameters())

    def forward(self, samples: Tensor, step: int) -> Tensor:
        if samples.ndim != 2 or samples.shape[1] != self.channels:
            raise ValueError("critic samples have invalid shape")
        normalized = (samples - self.reference_center[None]) / self.reference_scale[None]
        time = _time_features(
            step,
            self.horizon,
            dtype=samples.dtype,
            device=samples.device,
            rows=int(samples.shape[0]),
        )
        return self.network(torch.cat([normalized, time], dim=1)).squeeze(1)


def empirical_fp_rollout(
    stepper: FrozenIctalGraphRCBatchStepper,
    generator: FullMarkovHJBGenerator,
    initial_markov: Tensor,
    standard_normal: Tensor,
) -> HJBFPParticleRollout:
    """Propagate the controlled empirical law and retain all Markov states."""

    noise = torch.as_tensor(
        standard_normal, dtype=stepper.dtype, device=stepper.device
    )
    if noise.ndim != 3 or noise.shape[2] != stepper.q:
        raise ValueError("standard_normal must have shape [particle,time,q]")
    particles, horizon, _ = noise.shape
    if int(horizon) != generator.horizon:
        raise ValueError("noise and generator horizons differ")
    state = stepper.repeat_initial(initial_markov, int(particles))
    previous = state.new_zeros(int(particles), stepper.actuator_dim)
    current = stepper.current_output(state)
    states = [state]
    outputs = [current]
    controls: list[Tensor] = []
    commands: list[Tensor] = []
    conditional_std: list[Tensor] = []
    for step in range(int(horizon)):
        action = generator.propose(
            step=step,
            markov_particles=state,
            previous_control=previous,
            current_scaled=current,
        )
        transition = stepper.step(
            state, action.applied_control, noise[:, step]
        )
        state = transition.next_markov
        current = transition.next_scaled
        previous = action.applied_control
        states.append(state)
        outputs.append(current)
        controls.append(previous)
        commands.append(action.command)
        conditional_std.append(transition.conditional_std)
    return HJBFPParticleRollout(
        markov=torch.stack(states, dim=1),
        scaled=torch.stack(outputs, dim=1),
        controls=torch.stack(controls, dim=1),
        commands=torch.stack(commands, dim=1),
        conditional_std=torch.stack(conditional_std, dim=1),
        noise=noise,
    )


def _matching_rows(real: Tensor, fake: Tensor) -> tuple[Tensor, Tensor]:
    rows = min(int(real.shape[0]), int(fake.shape[0]))
    if rows < 2:
        raise ValueError("WGAN-GP requires at least two real and fake rows")
    return real[:rows], fake[:rows]


def time_conditioned_wgan_gp_loss(
    critic: TimeConditionedWassersteinCritic,
    fake_sequence: Tensor,
    reference_sequence: Tensor,
    time_indices: Iterable[int],
    *,
    gradient_penalty_weight: float = 10.0,
    critic_drift_weight: float = 1.0e-3,
    generator: torch.Generator | None = None,
) -> WassersteinGPLoss:
    """WGAN-GP loss with healthy reference as real and controlled law as fake."""

    if fake_sequence.ndim != 3 or reference_sequence.ndim != 3:
        raise ValueError("WGAN sequences must have shape [sample,time,channel]")
    if fake_sequence.shape[1:] != reference_sequence.shape[1:]:
        raise ValueError("fake and reference time/channel dimensions differ")
    estimates: list[Tensor] = []
    penalties: list[Tensor] = []
    drifts: list[Tensor] = []
    gradient_norms: list[Tensor] = []
    for raw_step in time_indices:
        step = int(raw_step)
        if not 0 <= step < int(fake_sequence.shape[1]):
            raise IndexError("WGAN time index lies outside the sequence")
        real, fake = _matching_rows(
            reference_sequence[:, step], fake_sequence[:, step]
        )
        real_score = critic(real, step)
        fake_score = critic(fake, step)
        estimates.append(real_score.mean() - fake_score.mean())
        drifts.append(0.5 * (real_score.square().mean() + fake_score.square().mean()))
        alpha = torch.rand(
            real.shape[0],
            1,
            dtype=real.dtype,
            device=real.device,
            generator=generator,
        )
        interpolated = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
        interpolated_score = critic(interpolated, step)
        gradient = torch.autograd.grad(
            interpolated_score.sum(),
            interpolated,
            create_graph=True,
            retain_graph=True,
        )[0]
        norm = gradient.reshape(gradient.shape[0], -1).norm(2, dim=1)
        gradient_norms.append(norm.mean())
        penalties.append((norm - 1.0).square().mean())
    if not estimates:
        raise ValueError("at least one WGAN time index is required")
    estimate = torch.stack(estimates).mean()
    gp = torch.stack(penalties).mean()
    drift = torch.stack(drifts).mean()
    loss = (
        -estimate
        + float(gradient_penalty_weight) * gp
        + float(critic_drift_weight) * drift
    )
    return WassersteinGPLoss(
        loss=loss,
        estimate=estimate,
        gradient_penalty=gp,
        critic_drift=drift,
        mean_gradient_norm=torch.stack(gradient_norms).mean(),
    )


def population_stage_cost(
    controlled: Tensor,
    control: Tensor,
    previous_control: Tensor,
    reference_at_time: Tensor,
    reference_scale: Tensor,
    critic: TimeConditionedWassersteinCritic,
    *,
    step: int,
    config: FullMarkovHJBWGANConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    """One social-planner stage cost used by the discrete HJB equation."""

    reference_mean = reference_at_time.mean(dim=0)
    reference_variance = reference_at_time.var(
        dim=0, unbiased=False
    ).clamp_min(1.0e-6)
    controlled_mean = controlled.mean(dim=0)
    controlled_variance = controlled.var(
        dim=0, unbiased=False
    ).clamp_min(1.0e-6)
    mean_cost = (
        (controlled_mean - reference_mean) / reference_scale
    ).square().mean()
    log_variance_cost = (
        torch.log(controlled_variance) - torch.log(reference_variance)
    ).square().mean()
    # Critic parameters may be frozen during the generator/HJB update, but its
    # input gradient is deliberately retained so this term moves the FP law.
    wasserstein_cost = (
        critic(reference_at_time, step).mean()
        - critic(controlled, step).mean()
    )
    normalized = control / config.amplitude_limit
    previous_normalized = previous_control / config.amplitude_limit
    energy = normalized.square().mean()
    smoothness = (normalized - previous_normalized).square().mean()
    total = (
        config.mean_cost_weight * mean_cost
        + config.log_variance_cost_weight * log_variance_cost
        + config.wasserstein_cost_weight * wasserstein_cost
        + config.control_energy_weight * energy
        + config.control_smoothness_weight * smoothness
    )
    return total, {
        "mean": mean_cost,
        "log_variance": log_variance_cost,
        "wasserstein": wasserstein_cost,
        "energy": energy,
        "smoothness": smoothness,
    }


def stochastic_bellman_hjb_loss(
    generator: FullMarkovHJBGenerator,
    target_generator: FullMarkovHJBGenerator,
    critic: TimeConditionedWassersteinCritic,
    stepper: FrozenIctalGraphRCBatchStepper,
    rollout: HJBFPParticleRollout,
    reference_sequence: Tensor,
    reference_scale: Tensor,
    time_indices: Iterable[int],
    antithetic_noise: Tensor,
    *,
    dt: float,
    config: FullMarkovHJBWGANConfig,
) -> HJBLossAudit:
    """Semi-gradient stochastic Bellman residual with +/- noise children."""

    noise = torch.as_tensor(
        antithetic_noise, dtype=stepper.dtype, device=stepper.device
    )
    residuals: list[Tensor] = []
    current_values: list[Tensor] = []
    target_values: list[Tensor] = []
    for position, raw_step in enumerate(time_indices):
        step = int(raw_step)
        if not 0 <= step < generator.horizon:
            raise IndexError("Bellman time index lies outside the horizon")
        if noise.ndim == 3:
            epsilon = noise[position]
        elif noise.ndim == 2:
            epsilon = noise
        else:
            raise ValueError("antithetic_noise must have shape [K,M,q] or [M,q]")
        state = rollout.markov[:, step].detach()
        current = rollout.scaled[:, step].detach()
        previous = (
            torch.zeros_like(rollout.controls[:, 0])
            if step == 0
            else rollout.controls[:, step - 1].detach()
        )
        control = rollout.controls[:, step].detach()
        if epsilon.shape != (state.shape[0], stepper.q):
            raise ValueError("one-step Bellman noise has invalid shape")
        plus = stepper.step(state, control, epsilon)
        minus = stepper.step(state, control, -epsilon)
        next_step = step + 1
        with torch.no_grad():
            plus_value = target_generator.population_value(
                step=next_step,
                markov_particles=plus.next_markov,
                previous_control=control,
                current_scaled=plus.next_scaled,
            )
            minus_value = target_generator.population_value(
                step=next_step,
                markov_particles=minus.next_markov,
                previous_control=control,
                current_scaled=minus.next_scaled,
            )
            stage, _ = population_stage_cost(
                0.5 * (plus.next_scaled + minus.next_scaled),
                control,
                previous,
                reference_sequence[:, step],
                reference_scale,
                critic,
                step=step,
                config=config,
            )
            target = float(dt) * stage + 0.5 * (plus_value + minus_value)
        value = generator.population_value(
            step=step,
            markov_particles=state,
            previous_control=previous,
            current_scaled=current,
        )
        residuals.append(value - target)
        current_values.append(value)
        target_values.append(target)

    final_state = rollout.markov[:, -1].detach()
    final_scaled = rollout.scaled[:, -1].detach()
    final_control = rollout.controls[:, -1].detach()
    terminal_value = generator.population_value(
        step=generator.horizon,
        markov_particles=final_state,
        previous_control=final_control,
        current_scaled=final_scaled,
    )
    terminal_target, _ = population_stage_cost(
        final_scaled,
        torch.zeros_like(final_control),
        torch.zeros_like(final_control),
        reference_sequence[:, -1],
        reference_scale,
        critic,
        step=generator.horizon,
        config=config,
    )
    terminal_target = terminal_target.detach()
    residual_tensor = torch.stack(residuals)
    return HJBLossAudit(
        bellman_loss=residual_tensor.square().mean(),
        terminal_loss=(terminal_value - terminal_target).square(),
        residuals=residual_tensor,
        current_values=torch.stack(current_values),
        target_values=torch.stack(target_values),
    )


def projected_hamiltonian_policy_loss(
    generator: FullMarkovHJBGenerator,
    target_generator: FullMarkovHJBGenerator,
    critic: TimeConditionedWassersteinCritic,
    stepper: FrozenIctalGraphRCBatchStepper,
    rollout: HJBFPParticleRollout,
    reference_sequence: Tensor,
    reference_scale: Tensor,
    time_indices: Iterable[int],
    antithetic_noise: Tensor,
    *,
    dt: float,
    projected_step_size: float,
    config: FullMarkovHJBWGANConfig,
) -> HamiltonianPolicyAudit:
    """First-order box-constrained Hamiltonian policy improvement target."""

    if float(projected_step_size) <= 0.0:
        raise ValueError("projected_step_size must be positive")
    noise = torch.as_tensor(
        antithetic_noise, dtype=stepper.dtype, device=stepper.device
    )
    losses: list[Tensor] = []
    residuals: list[Tensor] = []
    targets: list[Tensor] = []
    for position, raw_step in enumerate(time_indices):
        step = int(raw_step)
        state = rollout.markov[:, step]
        current = rollout.scaled[:, step]
        previous = (
            torch.zeros_like(rollout.controls[:, 0])
            if step == 0
            else rollout.controls[:, step - 1]
        )
        policy_action = generator.propose(
            step=step,
            markov_particles=state,
            previous_control=previous,
            current_scaled=current,
        ).applied_control
        candidate = policy_action.detach().requires_grad_(True)
        epsilon = noise[position] if noise.ndim == 3 else noise
        plus = stepper.step(state.detach(), candidate, epsilon)
        minus = stepper.step(state.detach(), candidate, -epsilon)
        plus_value = target_generator.population_value(
            step=step + 1,
            markov_particles=plus.next_markov,
            previous_control=candidate,
            current_scaled=plus.next_scaled,
        )
        minus_value = target_generator.population_value(
            step=step + 1,
            markov_particles=minus.next_markov,
            previous_control=candidate,
            current_scaled=minus.next_scaled,
        )
        stage, _ = population_stage_cost(
            0.5 * (plus.next_scaled + minus.next_scaled),
            candidate,
            previous.detach(),
            reference_sequence[:, step],
            reference_scale,
            critic,
            step=step,
            config=config,
        )
        hamiltonian = float(dt) * stage + 0.5 * (plus_value + minus_value)
        gradient = torch.autograd.grad(
            hamiltonian,
            candidate,
            create_graph=False,
            retain_graph=True,
        )[0]
        target = torch.clamp(
            candidate - float(projected_step_size) * gradient,
            -config.amplitude_limit,
            config.amplitude_limit,
        ).detach()
        residual = policy_action - target
        losses.append(residual.square().mean())
        residuals.append(residual.detach().square().mean())
        targets.append(target)
    return HamiltonianPolicyAudit(
        policy_loss=torch.stack(losses).mean(),
        projected_kkt_rms=torch.stack(residuals).mean().sqrt(),
        projected_targets=tuple(targets),
    )


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(bool(enabled))


def polyak_update(
    target: nn.Module, source: nn.Module, coefficient: float = 0.01
) -> None:
    tau = float(coefficient)
    if not 0.0 < tau <= 1.0:
        raise ValueError("Polyak coefficient must lie in (0,1]")
    with torch.no_grad():
        target_parameters = dict(target.named_parameters())
        source_parameters = dict(source.named_parameters())
        if target_parameters.keys() != source_parameters.keys():
            raise ValueError("source and target architectures differ")
        for name, target_parameter in target_parameters.items():
            target_parameter.lerp_(source_parameters[name], tau)
        target_buffers = dict(target.named_buffers())
        source_buffers = dict(source.named_buffers())
        for name, target_buffer in target_buffers.items():
            if name in source_buffers and target_buffer.dtype.is_floating_point:
                target_buffer.copy_(source_buffers[name])


__all__ = [
    "FullMarkovHJBGenerator",
    "FullMarkovHJBWGANConfig",
    "HJBLossAudit",
    "HJBFPParticleRollout",
    "HamiltonianPolicyAudit",
    "TimeConditionedWassersteinCritic",
    "WassersteinGPLoss",
    "empirical_fp_rollout",
    "polyak_update",
    "population_stage_cost",
    "projected_hamiltonian_policy_loss",
    "set_requires_grad",
    "stochastic_bellman_hjb_loss",
    "time_conditioned_wgan_gp_loss",
]
