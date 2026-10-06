"""Paired particle/FP rollout for the causal full-Markov LTV backbone.

This module deliberately does not reuse the compact actor callback in
``TorchGraphRCSDE.rollout``: that callback does not receive the reservoir,
complete delay register, or topology state.  Instead, the frozen ictal-only
Graph--RC transition is evaluated in an explicit batch Markov coordinate.

Controlled and uncontrolled arms are advanced in one concatenated batch with
the same standard-normal tensor.  The learned state-dependent diffusion is
recomputed from each arm's current state; only the driving normal variates are
shared.  No interictal diffusion is substituted and diffusion is not directly
controlled.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Protocol

import numpy as np
import torch
from torch import Tensor

from .causal_ltv_riccati import (
    CausalLTVLinearization,
    FrozenGraphRCMarkovAdapter,
    MeanFieldRiccatiBackbone,
)
from .neural_hjb_fp import bounded_mean_deviation_feedback


@dataclass(frozen=True)
class IctalBatchStep:
    """One batch transition of the fitted state-dependent ictal RC--SDE."""

    next_markov: Tensor
    next_scaled: Tensor
    conditional_std: Tensor
    innovation_latent: Tensor


@dataclass(frozen=True)
class LTVPolicyAction:
    """Bounded command and its explicit mean/deviation decomposition."""

    command: Tensor
    applied_control: Tensor
    common_command: Tensor
    deviation_command: Tensor
    variance_contraction_gate: Tensor


@dataclass(frozen=True)
class PairedParticleRollout:
    """Common-random-number forward empirical FP evaluation."""

    controlled_scaled: Tensor
    uncontrolled_scaled: Tensor
    commands: Tensor
    controls: Tensor
    common_controls: Tensor
    deviation_controls: Tensor
    variance_contraction_gate: Tensor
    controlled_conditional_std: Tensor
    uncontrolled_conditional_std: Tensor
    noise: Tensor
    noise_sha256: str
    controlled_markov: Tensor | None = None
    uncontrolled_markov: Tensor | None = None


@dataclass(frozen=True)
class ProjectionClosureAudit:
    """Loss of full-Markov tangent dynamics outside a reduced basis."""

    maximum_state_closure_error: float
    mean_state_closure_error: float
    maximum_control_closure_error: float
    mean_control_closure_error: float
    rank: int
    horizon: int
    uses_recorded_future: bool = False


class FullMarkovMeanFieldPolicy(Protocol):
    """Policy contract that exposes no innovation or recorded future."""

    def propose(
        self,
        *,
        step: int,
        markov_particles: Tensor,
        previous_control: Tensor,
        current_scaled: Tensor,
    ) -> LTVPolicyAction: ...


def tensor_sha256(value: Tensor) -> str:
    """Hash dtype, shape and contiguous bytes of a CPU tensor."""

    array = np.ascontiguousarray(value.detach().cpu().numpy())
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def augment_with_first_order_actuator(
    linearization: CausalLTVLinearization,
    *,
    feedback_alpha: float,
) -> CausalLTVLinearization:
    r"""Add the applied-control state to the finite-horizon HJB system.

    The decision variable is a bounded desired command ``v`` and the applied
    stimulation is the explicit controller state

    ``a[t] = (1-alpha) a[t-1] + alpha v[t]``.

    The plant then receives ``a[t]``.  Thus smoothing is part of the Markov
    dynamics used by the Riccati recursion, rather than a post-hoc plot filter.
    """

    alpha = float(feedback_alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("feedback_alpha must lie in (0, 1]")
    horizon = linearization.horizon
    reduced = linearization.reduced_state_dim
    full = linearization.full_state_dim
    actuators = linearization.actuator_dim
    dtype = linearization.state_transition.dtype
    device = linearization.state_transition.device
    retain = 1.0 - alpha

    transition = torch.zeros(
        horizon,
        reduced + actuators,
        reduced + actuators,
        dtype=dtype,
        device=device,
    )
    persistent = torch.zeros(
        horizon,
        reduced + actuators,
        actuators,
        dtype=dtype,
        device=device,
    )
    output_state = torch.zeros(
        horizon,
        linearization.output_state_jacobian.shape[1],
        reduced + actuators,
        dtype=dtype,
        device=device,
    )
    output_control = torch.zeros_like(
        linearization.output_control_jacobian
    )
    identity = torch.eye(actuators, dtype=dtype, device=device)
    for step in range(horizon):
        a = linearization.state_transition[step]
        b = linearization.persistent_control[step]
        e = linearization.output_state_jacobian[step]
        f = linearization.output_control_jacobian[step]
        transition[step, :reduced, :reduced] = a
        transition[step, :reduced, reduced:] = retain * b
        transition[step, reduced:, reduced:] = retain * identity
        persistent[step, :reduced] = alpha * b
        persistent[step, reduced:] = alpha * identity
        output_state[step, :, :reduced] = e
        output_state[step, :, reduced:] = retain * f
        output_control[step] = alpha * f

    projection = torch.zeros(
        full + actuators,
        reduced + actuators,
        dtype=dtype,
        device=device,
    )
    projection[:full, :reduced] = linearization.projection
    projection[full:, reduced:] = identity
    nominal_states = torch.cat(
        [
            linearization.nominal_markov_states,
            torch.zeros(
                horizon + 1,
                actuators,
                dtype=dtype,
                device=device,
            ),
        ],
        dim=1,
    )
    terminal_output = torch.cat(
        [
            linearization.terminal_output_state_jacobian,
            torch.zeros(
                linearization.terminal_output_state_jacobian.shape[0],
                actuators,
                dtype=dtype,
                device=device,
            ),
        ],
        dim=1,
    )
    return CausalLTVLinearization(
        state_transition=transition,
        persistent_control=persistent,
        output_state_jacobian=output_state,
        output_control_jacobian=output_control,
        terminal_output_state_jacobian=terminal_output,
        nominal_markov_states=nominal_states,
        nominal_outputs=linearization.nominal_outputs,
        projection=projection,
        persistent_latent_control_map=(
            alpha * linearization.persistent_latent_control_map
        ),
        control_step_scale=linearization.control_step_scale,
        nominal_source=(
            linearization.nominal_source
            + "; augmented first-order applied-control state"
        ),
        diffusion_treatment=linearization.diffusion_treatment,
        uses_recorded_future=linearization.uses_recorded_future,
    )


def audit_projection_closure(
    adapter: FrozenGraphRCMarkovAdapter,
    linearization: CausalLTVLinearization,
) -> ProjectionClosureAudit:
    r"""Audit discarded ``(I-VV')A_tV`` and ``(I-VV')B_t`` energy.

    The nominal states and all JVPs are model generated.  No output or
    recorded-future argument is accepted by this function.
    """

    if linearization.full_state_dim != adapter.state_dim:
        raise ValueError("audit must precede actuator-state augmentation")
    basis = linearization.projection
    if basis.shape[0] != adapter.state_dim:
        raise ValueError("projection and adapter state dimensions differ")
    zero = torch.zeros(
        adapter.actuator_dim, dtype=adapter.dtype, device=adapter.device
    )
    reduced = int(basis.shape[1])
    state_directions = torch.cat(
        [
            basis.T,
            torch.zeros(
                adapter.actuator_dim,
                adapter.state_dim,
                dtype=adapter.dtype,
                device=adapter.device,
            ),
        ],
        dim=0,
    )
    control_directions = torch.cat(
        [
            torch.zeros(
                reduced,
                adapter.actuator_dim,
                dtype=adapter.dtype,
                device=adapter.device,
            ),
            torch.eye(
                adapter.actuator_dim,
                dtype=adapter.dtype,
                device=adapter.device,
            ),
        ],
        dim=0,
    )
    state_errors: list[Tensor] = []
    control_errors: list[Tensor] = []
    for step in range(linearization.horizon):
        nominal = linearization.nominal_markov_states[step]

        def transition_map(x: Tensor, u: Tensor) -> Tensor:
            return adapter.deterministic_step(x, u)[0]

        def direction(dx: Tensor, du: Tensor) -> Tensor:
            return torch.func.jvp(
                transition_map, (nominal, zero), (dx, du)
            )[1]

        tangent = torch.vmap(direction)(
            state_directions, control_directions
        ).detach()
        state_tangent = tangent[:reduced].T
        control_tangent = tangent[reduced:].T
        state_projection = basis @ (basis.T @ state_tangent)
        control_projection = basis @ (basis.T @ control_tangent)
        state_errors.append(
            torch.linalg.vector_norm(state_tangent - state_projection)
            / torch.linalg.vector_norm(state_tangent).clamp_min(1e-12)
        )
        control_errors.append(
            torch.linalg.vector_norm(control_tangent - control_projection)
            / torch.linalg.vector_norm(control_tangent).clamp_min(1e-12)
        )
    state = torch.stack(state_errors)
    control = torch.stack(control_errors)
    return ProjectionClosureAudit(
        maximum_state_closure_error=state.max().item(),
        mean_state_closure_error=state.mean().item(),
        maximum_control_closure_error=control.max().item(),
        mean_control_closure_error=control.mean().item(),
        rank=reduced,
        horizon=linearization.horizon,
    )


class FrozenIctalGraphRCBatchStepper:
    """Float-consistent batch implementation of the frozen ictal RC--SDE."""

    def __init__(
        self,
        world,
        adapter: FrozenGraphRCMarkovAdapter,
        *,
        diffusion_scale: float,
    ) -> None:
        if bool(getattr(world, "regime_adaptive", False)):
            raise NotImplementedError(
                "the paired evaluator is intentionally ictal-only"
            )
        if float(diffusion_scale) <= 0:
            raise ValueError("diffusion_scale must be positive")
        self.adapter = adapter
        self.diffusion_scale = float(diffusion_scale)
        self.dtype = adapter.dtype
        self.device = adapter.device
        self.q = adapter.q
        self.n_channels = adapter.n_channels
        self.state_dim = adapter.state_dim
        self.actuator_dim = adapter.actuator_dim

        def frozen(name: str) -> Tensor:
            return torch.as_tensor(
                getattr(world, name), dtype=self.dtype, device=self.device
            ).detach().clone()

        self.variance_coef = frozen("variance_coef")
        self.variance_intercept = frozen("variance_intercept")
        self.variance_calibration = frozen("variance_calibration")
        self.log_variance_bounds = frozen("log_variance_bounds")
        self.correlation_root = frozen("correlation_root")
        if self.variance_coef.shape[0] != self.q:
            raise ValueError("diffusion variance model is incompatible")
        if self.correlation_root.shape != (self.q, self.q):
            raise ValueError("diffusion correlation root is incompatible")

        for name in (
            "components",
            "pca_mean",
            "adjacency",
            "w_in",
            "w_res",
            "feature_mean",
            "feature_scale",
            "drift_coef",
            "drift_intercept",
            "control_map",
            "control_channel_map",
        ):
            world_value = torch.as_tensor(
                getattr(world, name), dtype=self.dtype, device=self.device
            )
            adapter_value = getattr(adapter, name)
            if not torch.equal(world_value, adapter_value):
                raise ValueError(f"world and adapter differ in {name}")

    def repeat_initial(self, initial_state: Tensor, particles: int) -> Tensor:
        state = torch.as_tensor(
            initial_state, dtype=self.dtype, device=self.device
        )
        if state.shape != (self.state_dim,):
            raise ValueError("initial_state has invalid shape")
        if int(particles) < 1:
            raise ValueError("particles must be positive")
        return state[None].expand(int(particles), -1).clone()

    def current_output(self, markov: Tensor) -> Tensor:
        if markov.ndim != 2 or markov.shape[1] != self.state_dim:
            raise ValueError("markov batch has invalid shape")
        current = markov[:, self.adapter.slices.current_latent]
        return current @ self.adapter.components + self.adapter.pca_mean

    def _unpack(self, markov: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if markov.ndim != 2 or markov.shape[1] != self.state_dim:
            raise ValueError("markov batch has invalid shape")
        reservoir = markov[:, self.adapter.slices.reservoir]
        history = markov[:, self.adapter.slices.history].reshape(
            len(markov), self.adapter.history_length, self.q
        )
        topology = markov[:, self.adapter.slices.topology]
        return reservoir, history, topology

    def _feature(
        self, markov: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        reservoir, history, topology = self._unpack(markov)
        current = history[:, -1]
        feature = torch.cat(
            [
                reservoir,
                current,
                topology,
                *(history[:, -1 - delay] for delay in self.adapter.delays),
            ],
            dim=1,
        )
        standardized = (
            feature - self.adapter.feature_mean
        ) / self.adapter.feature_scale
        return reservoir, history, current, standardized

    def conditional_mean_and_std(
        self, markov: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return the no-input conditional latent mean and fitted ictal std."""

        _, _, current, standardized = self._feature(markov)
        mean_increment = (
            standardized @ self.adapter.drift_coef.T
            + self.adapter.drift_intercept
        )
        mean_latent = current + mean_increment
        log_variance = (
            standardized @ self.variance_coef.T
            + self.variance_intercept
            + torch.log(self.variance_calibration)
        )
        log_variance = torch.maximum(
            log_variance, self.log_variance_bounds[0]
        )
        log_variance = torch.minimum(
            log_variance, self.log_variance_bounds[1]
        )
        conditional_std = torch.exp(0.5 * log_variance)
        return mean_latent, conditional_std

    def step(
        self,
        markov: Tensor,
        control: Tensor,
        standard_normal: Tensor,
    ) -> IctalBatchStep:
        """Advance one batch, recomputing diffusion from each current state."""

        batch = int(markov.shape[0])
        expected_control = (batch, self.actuator_dim)
        expected_noise = (batch, self.q)
        if control.shape != expected_control:
            raise ValueError(f"control must have shape {expected_control}")
        if standard_normal.shape != expected_noise:
            raise ValueError(f"standard_normal must have shape {expected_noise}")
        for name, value in (
            ("markov", markov),
            ("control", control),
            ("standard_normal", standard_normal),
        ):
            if value.dtype != self.dtype or value.device != self.device:
                raise ValueError(f"{name} dtype/device differs from the stepper")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")

        reservoir, history, _, _ = self._feature(markov)
        mean_latent, conditional_std = self.conditional_mean_and_std(markov)
        correlated = standard_normal @ self.correlation_root.T
        innovation = self.diffusion_scale * correlated * conditional_std
        latent_control = control @ self.adapter.control_map
        following = (
            mean_latent
            + innovation
            + self.adapter.control_step_scale * latent_control
        )
        updated_history = torch.cat([history[:, 1:], following[:, None]], dim=1)
        next_scaled = following @ self.adapter.components + self.adapter.pca_mean
        if self.adapter.preserve_physical_control_residual:
            channel_control = control @ self.adapter.control_channel_map
            projected = latent_control @ self.adapter.components
            next_scaled = next_scaled + self.adapter.control_step_scale * (
                channel_control - projected
            )
        next_topology = (
            next_scaled @ self.adapter.adjacency.T
        ) @ self.adapter.components.T
        drive = torch.cat(
            [
                following,
                next_topology,
                *(
                    updated_history[:, -1 - delay]
                    for delay in self.adapter.delays
                ),
            ],
            dim=1,
        )
        proposal = torch.tanh(
            drive @ self.adapter.w_in.T
            + reservoir @ self.adapter.w_res.T
        )
        next_reservoir = (
            (1.0 - self.adapter.leak_rate) * reservoir
            + self.adapter.leak_rate * proposal
        )
        next_markov = torch.cat(
            [
                next_reservoir,
                updated_history.reshape(batch, -1),
                next_topology,
            ],
            dim=1,
        )
        return IctalBatchStep(
            next_markov=next_markov,
            next_scaled=next_scaled,
            conditional_std=conditional_std,
            innovation_latent=innovation,
        )


class CausalLTVMeanFieldPolicy:
    """Bounded affine mean plus covariance-contraction LTV feedback."""

    def __init__(
        self,
        backbone: MeanFieldRiccatiBackbone,
        linearization: CausalLTVLinearization,
        *,
        amplitude_limit: float,
        feedback_alpha: float,
        reference_variance: Tensor | None,
        mean_scale: float = 1.0,
        deviation_scale: float = 1.0,
        variance_gate_quantile: float = 0.75,
    ) -> None:
        if amplitude_limit <= 0:
            raise ValueError("amplitude_limit must be positive")
        if not 0.0 < feedback_alpha <= 1.0:
            raise ValueError("feedback_alpha must lie in (0,1]")
        if mean_scale < 0 or deviation_scale < 0:
            raise ValueError("feedback scales must be non-negative")
        if not 0.0 <= variance_gate_quantile <= 1.0:
            raise ValueError("variance_gate_quantile must lie in [0,1]")
        if backbone.mean_gain_reduced.shape[0] != linearization.horizon:
            raise ValueError("backbone and linearization horizons differ")
        self.backbone = backbone
        self.linearization = linearization
        self.amplitude_limit = float(amplitude_limit)
        self.feedback_alpha = float(feedback_alpha)
        self.mean_scale = float(mean_scale)
        self.deviation_scale = float(deviation_scale)
        self.variance_gate_quantile = float(variance_gate_quantile)
        self.reference_variance = (
            None
            if reference_variance is None
            else torch.as_tensor(
                reference_variance,
                dtype=linearization.state_transition.dtype,
                device=linearization.state_transition.device,
            )
        )
        if self.reference_variance is not None:
            expected = (
                linearization.horizon,
                linearization.output_state_jacobian.shape[1],
            )
            if self.reference_variance.shape != expected:
                raise ValueError(
                    f"reference_variance must have shape {expected}"
                )

    def propose(
        self,
        *,
        step: int,
        markov_particles: Tensor,
        previous_control: Tensor,
        current_scaled: Tensor,
    ) -> LTVPolicyAction:
        index = int(step)
        if not 0 <= index < self.linearization.horizon:
            raise IndexError("policy step is outside the horizon")
        particles = int(markov_particles.shape[0])
        actuators = self.linearization.actuator_dim
        if previous_control.shape != (particles, actuators):
            raise ValueError("previous_control has invalid shape")
        if current_scaled.shape[0] != particles:
            raise ValueError("current_scaled batch differs from Markov batch")
        augmented = torch.cat([markov_particles, previous_control], dim=1)
        if augmented.shape[1] != self.linearization.full_state_dim:
            raise ValueError(
                "policy requires the actuator-augmented linearization"
            )
        population_mean = augmented.mean(dim=0, keepdim=True)
        deviation = augmented - population_mean
        nominal = self.linearization.nominal_markov_states[index : index + 1]
        mean_delta = population_mean - nominal
        projection = self.linearization.projection
        reduced_mean = mean_delta @ projection
        reduced_deviation = deviation @ projection
        common_desired = -(
            reduced_mean @ self.backbone.mean_gain_reduced[index].T
            + self.backbone.mean_feedforward[index][None]
        )
        deviation_desired = -(
            reduced_deviation
            @ self.backbone.deviation_gain_reduced[index].T
        )

        if self.reference_variance is None or particles < 2:
            gate = current_scaled.new_tensor(1.0)
        else:
            current_variance = current_scaled.var(dim=0, unbiased=False)
            target_variance = self.reference_variance[index].clamp_min(1e-6)
            contraction_need = (
                1.0
                - torch.sqrt(
                    target_variance / current_variance.clamp_min(1e-6)
                )
            ).clamp(0.0, 1.0)
            gate = torch.quantile(
                contraction_need, self.variance_gate_quantile
            )
        common_desired = self.mean_scale * common_desired
        deviation_desired = (
            self.deviation_scale * gate * deviation_desired
        )
        context_indices = torch.zeros(
            particles, dtype=torch.long, device=markov_particles.device
        )
        command = bounded_mean_deviation_feedback(
            -common_desired.expand(particles, -1) / self.amplitude_limit,
            -deviation_desired / self.amplitude_limit,
            context_indices,
            self.amplitude_limit,
        )
        common_command = command.mean(dim=0, keepdim=True)
        deviation_command = command - common_command
        applied = (
            (1.0 - self.feedback_alpha) * previous_control
            + self.feedback_alpha * command
        )
        return LTVPolicyAction(
            command=command,
            applied_control=applied,
            common_command=common_command,
            deviation_command=deviation_command,
            variance_contraction_gate=gate,
        )


def paired_ictal_batch_rollout(
    stepper: FrozenIctalGraphRCBatchStepper,
    policy: FullMarkovMeanFieldPolicy,
    initial_markov: Tensor,
    standard_normal: Tensor,
    *,
    store_markov: bool = False,
) -> PairedParticleRollout:
    """Evaluate controlled and no-control laws with exact paired normals."""

    noise = torch.as_tensor(
        standard_normal, dtype=stepper.dtype, device=stepper.device
    )
    if noise.ndim != 3 or noise.shape[2] != stepper.q:
        raise ValueError("standard_normal must have shape [particle,time,q]")
    particles, horizon, _ = noise.shape
    uncontrolled = stepper.repeat_initial(initial_markov, particles)
    controlled = uncontrolled.clone()
    zero = torch.zeros(
        particles,
        stepper.actuator_dim,
        dtype=stepper.dtype,
        device=stepper.device,
    )
    previous_control = zero.clone()
    uncontrolled_output = stepper.current_output(uncontrolled)
    controlled_output = uncontrolled_output.clone()
    uncontrolled_values = [uncontrolled_output]
    controlled_values = [controlled_output]
    commands: list[Tensor] = []
    controls: list[Tensor] = []
    common_controls: list[Tensor] = []
    deviation_controls: list[Tensor] = []
    gates: list[Tensor] = []
    uncontrolled_std: list[Tensor] = []
    controlled_std: list[Tensor] = []
    uncontrolled_states: list[Tensor] | None = (
        [uncontrolled] if store_markov else None
    )
    controlled_states: list[Tensor] | None = (
        [controlled] if store_markov else None
    )

    for step in range(horizon):
        action = policy.propose(
            step=step,
            markov_particles=controlled,
            previous_control=previous_control,
            current_scaled=controlled_output,
        )
        combined_state = torch.cat([uncontrolled, controlled], dim=0)
        combined_control = torch.cat([zero, action.applied_control], dim=0)
        combined_noise = torch.cat([noise[:, step], noise[:, step]], dim=0)
        transition = stepper.step(
            combined_state, combined_control, combined_noise
        )
        uncontrolled, controlled = transition.next_markov.chunk(2, dim=0)
        uncontrolled_output, controlled_output = transition.next_scaled.chunk(
            2, dim=0
        )
        std0, std1 = transition.conditional_std.chunk(2, dim=0)
        previous_control = action.applied_control
        applied_common = previous_control.mean(dim=0, keepdim=True)
        uncontrolled_values.append(uncontrolled_output)
        controlled_values.append(controlled_output)
        commands.append(action.command)
        controls.append(previous_control)
        common_controls.append(applied_common)
        deviation_controls.append(previous_control - applied_common)
        gates.append(action.variance_contraction_gate)
        uncontrolled_std.append(std0)
        controlled_std.append(std1)
        if uncontrolled_states is not None and controlled_states is not None:
            uncontrolled_states.append(uncontrolled)
            controlled_states.append(controlled)

    return PairedParticleRollout(
        controlled_scaled=torch.stack(controlled_values, dim=1),
        uncontrolled_scaled=torch.stack(uncontrolled_values, dim=1),
        commands=torch.stack(commands, dim=1),
        controls=torch.stack(controls, dim=1),
        common_controls=torch.stack(common_controls, dim=1),
        deviation_controls=torch.stack(deviation_controls, dim=1),
        variance_contraction_gate=torch.stack(gates),
        controlled_conditional_std=torch.stack(controlled_std, dim=1),
        uncontrolled_conditional_std=torch.stack(uncontrolled_std, dim=1),
        noise=noise,
        noise_sha256=tensor_sha256(noise),
        controlled_markov=(
            None
            if controlled_states is None
            else torch.stack(controlled_states, dim=1)
        ),
        uncontrolled_markov=(
            None
            if uncontrolled_states is None
            else torch.stack(uncontrolled_states, dim=1)
        ),
    )


def reconstruct_paired_normals(
    stepper: FrozenIctalGraphRCBatchStepper,
    initial_markov: Tensor,
    sealed_no_control_scaled: Tensor | np.ndarray,
    *,
    tolerance: float = 2e-6,
) -> tuple[Tensor, dict[str, float | str]]:
    """Reconstruct evaluation normals from a sealed no-control model law.

    This is an evaluation/common-random-number utility, not a synthesis input.
    The caller must freeze the policy before opening the sealed future array.
    """

    target = torch.as_tensor(
        sealed_no_control_scaled,
        dtype=stepper.dtype,
        device=stepper.device,
    )
    if target.ndim != 3 or target.shape[2] != stepper.n_channels:
        raise ValueError("sealed_no_control_scaled has invalid shape")
    particles, horizon, _ = target.shape
    state = stepper.repeat_initial(initial_markov, particles)
    zero = torch.zeros(
        particles,
        stepper.actuator_dim,
        dtype=stepper.dtype,
        device=stepper.device,
    )
    normals: list[Tensor] = []
    maximum_error = 0.0
    squared_error = 0.0
    squared_target = 0.0
    with torch.no_grad():
        for step in range(horizon):
            mean_latent, conditional_std = stepper.conditional_mean_and_std(
                state
            )
            target_latent = (
                target[:, step] - stepper.adapter.pca_mean
            ) @ stepper.adapter.components.T
            innovation = target_latent - mean_latent
            correlated = innovation / (
                stepper.diffusion_scale * conditional_std.clamp_min(1e-10)
            )
            normal = torch.linalg.solve(
                stepper.correlation_root, correlated.T
            ).T
            transition = stepper.step(state, zero, normal)
            error = transition.next_scaled - target[:, step]
            maximum_error = max(maximum_error, error.abs().max().item())
            squared_error += error.square().sum().item()
            squared_target += target[:, step].square().sum().item()
            normals.append(normal)
            state = transition.next_markov
    if maximum_error > float(tolerance):
        raise RuntimeError(
            "sealed no-control reconstruction failed: "
            f"max_abs={maximum_error:.6g}"
        )
    noise = torch.stack(normals, dim=1)
    return noise, {
        "maximum_absolute_error": float(maximum_error),
        "relative_l2_error": float(
            np.sqrt(squared_error / max(squared_target, 1e-30))
        ),
        "noise_sha256": tensor_sha256(noise),
        "role": "evaluation-only reconstruction after policy freeze",
    }


__all__ = [
    "CausalLTVMeanFieldPolicy",
    "FrozenIctalGraphRCBatchStepper",
    "FullMarkovMeanFieldPolicy",
    "IctalBatchStep",
    "LTVPolicyAction",
    "PairedParticleRollout",
    "ProjectionClosureAudit",
    "audit_projection_closure",
    "augment_with_first_order_actuator",
    "paired_ictal_batch_rollout",
    "reconstruct_paired_normals",
    "tensor_sha256",
]
