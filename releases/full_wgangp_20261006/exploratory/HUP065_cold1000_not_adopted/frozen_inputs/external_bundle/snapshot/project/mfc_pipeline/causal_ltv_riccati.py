"""Causal finite-horizon LTV--Riccati backbone for the frozen Graph--RC SDE.

The module is deliberately independent of the Part-III runner.  It exposes the
complete Markov state used by :class:`TorchGraphRCSDE` (reservoir, delay
register, and graph-topology state), rolls its *conditional mean* forward from
an author-supplied past context, and differentiates that model rollout.  No
recorded future is accepted by any public API.

The fitted state-dependent ictal diffusion is not replaced or controlled.
For the risk-neutral quadratic backbone it is treated in the usual
certainty-equivalent way: the Riccati recursion uses the conditional-mean
Jacobian, while the unchanged diffusion remains in the forward particle/FP
solver.  A neural HJB residual may subsequently correct this quadratic prior.

For HUP060 the full Markov state is large (538 coordinates).  The helper
``build_causal_reachability_projection`` therefore constructs an optional
causal, model-only reachable subspace using tangent probes.  The nonlinear
rollout and every Jacobian-vector product are still evaluated in the complete
Markov state; only the dense Riccati recursion is reduced.  Returned gains can
be lifted exactly to the complete Markov coordinate system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class MarkovStateSlices:
    """Slices defining the explicit frozen Graph--RC Markov state."""

    reservoir: slice
    history: slice
    history_blocks: tuple[slice, ...]
    current_latent: slice
    topology: slice


@dataclass(frozen=True)
class ReachabilityProjection:
    """Causal tangent-reachability basis and its numerical audit."""

    matrix: Tensor
    singular_values: Tensor
    explained_energy_fraction: float
    probe_count: int
    snapshot_stride: int
    seed: int
    condition_limit: float
    retained_condition_number: float
    uses_recorded_future: bool = False


@dataclass(frozen=True)
class CausalLTVLinearization:
    """Projected LTV differential along a causal zero-control model rollout.

    ``output_state_jacobian[t]`` and ``output_control_jacobian[t]`` map the
    current reduced state and current actuator command to the *next* decoded
    output.  Consequently the physical PCA-orthogonal residual, when enabled
    in the frozen world, is present in the Riccati stage cost instead of being
    invisible to the Hamiltonian.
    """

    state_transition: Tensor
    persistent_control: Tensor
    output_state_jacobian: Tensor
    output_control_jacobian: Tensor
    terminal_output_state_jacobian: Tensor
    nominal_markov_states: Tensor
    nominal_outputs: Tensor
    projection: Tensor
    persistent_latent_control_map: Tensor
    control_step_scale: float
    nominal_source: str = "causal zero-control conditional-mean Graph-RC rollout"
    diffusion_treatment: str = (
        "fitted ictal state-dependent diffusion remains frozen in the forward "
        "SDE; the risk-neutral Riccati backbone linearizes conditional mean"
    )
    uses_recorded_future: bool = False

    @property
    def horizon(self) -> int:
        return int(self.state_transition.shape[0])

    @property
    def reduced_state_dim(self) -> int:
        return int(self.state_transition.shape[1])

    @property
    def full_state_dim(self) -> int:
        return int(self.projection.shape[0])

    @property
    def actuator_dim(self) -> int:
        return int(self.persistent_control.shape[2])


@dataclass(frozen=True)
class MeanFieldRiccatiBackbone:
    """Time-varying mean/deviation gains for a mean-field quadratic prior.

    The control sign convention is

    ``u = -K_deviation @ (chi_i - mean_chi) - K_mean @ mean_error``.
    """

    mean_gain_reduced: Tensor
    deviation_gain_reduced: Tensor
    mean_feedforward: Tensor
    mean_value_hessian: Tensor
    mean_value_linear: Tensor
    deviation_value_hessian: Tensor
    mean_regularization: Tensor
    deviation_regularization: Tensor
    projection: Tensor
    uses_recorded_future: bool = False

    @property
    def mean_gain_full(self) -> Tensor:
        """Lift the mean gain to the complete Markov state."""

        return torch.einsum(
            "tmr,nr->tmn", self.mean_gain_reduced, self.projection
        )

    @property
    def deviation_gain_full(self) -> Tensor:
        """Lift the deviation gain to the complete Markov state."""

        return torch.einsum(
            "tmr,nr->tmn", self.deviation_gain_reduced, self.projection
        )

    def feedback(
        self,
        step: int,
        deviation_markov: Tensor,
        mean_markov_error: Tensor,
    ) -> Tensor:
        """Evaluate the full-Markov Riccati feedback at one time step."""

        index = int(step)
        return -(
            deviation_markov @ self.deviation_gain_full[index].T
            + mean_markov_error @ self.mean_gain_full[index].T
            + self.mean_feedforward[index]
        )


class FrozenGraphRCMarkovAdapter:
    """Explicit deterministic-mean Markov adapter for ``TorchGraphRCSDE``.

    The adapter copies and detaches every fitted tensor.  It currently targets
    the ictal-only state-dependent RC-SDE used by HUP060.  Regime-adaptive
    mixtures are rejected because their gate window is a different Markov
    contract and must be audited separately rather than silently approximated.
    """

    def __init__(
        self,
        world: Any,
        *,
        control_step_scale: float,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        if float(control_step_scale) <= 0:
            raise ValueError("control_step_scale must be positive")
        if bool(getattr(world, "regime_adaptive", False)):
            raise NotImplementedError(
                "regime-adaptive Graph-RC gates require a separate audited "
                "Markov adapter"
            )

        self.dtype = dtype
        self.device = torch.device(device)
        self.control_step_scale = float(control_step_scale)
        self.q = int(world.q)
        self.n_channels = int(world.n_channels)
        self.reservoir_size = int(world.reservoir_size)
        self.maximum_delay = int(world.maximum_delay)
        self.delays = tuple(int(value) for value in world.delays)
        self.history_length = self.maximum_delay + 1
        self.leak_rate = float(world.leak_rate)
        self.preserve_physical_control_residual = bool(
            getattr(world, "preserve_physical_control_residual", False)
        )

        def frozen(value: Any) -> Tensor:
            return torch.as_tensor(
                value, dtype=self.dtype, device=self.device
            ).detach().clone()

        self.components = frozen(world.components)
        self.pca_mean = frozen(world.pca_mean)
        self.adjacency = frozen(world.adjacency)
        self.w_in = frozen(world.w_in)
        self.w_res = frozen(world.w_res)
        self.feature_mean = frozen(world.feature_mean)
        self.feature_scale = frozen(world.feature_scale)
        self.drift_coef = frozen(world.drift_coef)
        self.drift_intercept = frozen(world.drift_intercept)
        self.control_map = frozen(world.control_map)
        self.control_channel_map = frozen(world.control_channel_map)
        self._context_state = world.context_state

        if self.components.shape != (self.q, self.n_channels):
            raise ValueError("PCA component shape is incompatible with Graph-RC")
        if self.control_map.ndim != 2 or self.control_map.shape[1] != self.q:
            raise ValueError("persistent latent control map has invalid shape")
        if self.control_channel_map.shape != (
            self.control_map.shape[0],
            self.n_channels,
        ):
            raise ValueError("physical control map has invalid shape")

        reservoir_stop = self.reservoir_size
        history_stop = reservoir_stop + self.history_length * self.q
        topology_stop = history_stop + self.q
        blocks = tuple(
            slice(
                reservoir_stop + index * self.q,
                reservoir_stop + (index + 1) * self.q,
            )
            for index in range(self.history_length)
        )
        self.slices = MarkovStateSlices(
            reservoir=slice(0, reservoir_stop),
            history=slice(reservoir_stop, history_stop),
            history_blocks=blocks,
            current_latent=blocks[-1],
            topology=slice(history_stop, topology_stop),
        )
        self.state_dim = topology_stop
        self.actuator_dim = int(self.control_map.shape[0])

        expected_feature_dim = (
            self.reservoir_size
            + self.q
            + self.q
            + len(self.delays) * self.q
        )
        expected_drive_dim = 2 * self.q + len(self.delays) * self.q
        if self.feature_mean.numel() != expected_feature_dim:
            raise ValueError("Graph-RC feature dimension is inconsistent")
        if self.feature_scale.numel() != expected_feature_dim:
            raise ValueError("Graph-RC feature scale dimension is inconsistent")
        if self.drift_coef.shape != (self.q, expected_feature_dim):
            raise ValueError("Graph-RC drift coefficient shape is inconsistent")
        if self.w_in.shape != (self.reservoir_size, expected_drive_dim):
            raise ValueError("reservoir input matrix shape is inconsistent")
        if self.w_res.shape != (self.reservoir_size, self.reservoir_size):
            raise ValueError("reservoir recurrent matrix shape is inconsistent")

    def pack(self, reservoir: Tensor, history: Tensor, topology: Tensor) -> Tensor:
        """Pack one complete Markov state."""

        reservoir = torch.as_tensor(
            reservoir, dtype=self.dtype, device=self.device
        )
        history = torch.as_tensor(history, dtype=self.dtype, device=self.device)
        topology = torch.as_tensor(
            topology, dtype=self.dtype, device=self.device
        )
        if reservoir.shape != (self.reservoir_size,):
            raise ValueError("reservoir state has invalid shape")
        if history.shape != (self.history_length, self.q):
            raise ValueError("delay history has invalid shape")
        if topology.shape != (self.q,):
            raise ValueError("topology state has invalid shape")
        return torch.cat([reservoir, history.reshape(-1), topology])

    def unpack(self, markov_state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Unpack one complete Markov state without copying gradients."""

        if markov_state.shape != (self.state_dim,):
            raise ValueError("Markov state has invalid shape")
        reservoir = markov_state[self.slices.reservoir]
        history = markov_state[self.slices.history].reshape(
            self.history_length, self.q
        )
        topology = markov_state[self.slices.topology]
        return reservoir, history, topology

    def initial_state_from_context(self, past_context: np.ndarray) -> Tensor:
        """Create the initial Markov state from past observations only."""

        context = np.asarray(past_context, dtype=np.float64)
        if context.ndim != 2 or context.shape[1] != self.n_channels:
            raise ValueError("past_context has invalid shape")
        reservoir, history, topology = self._context_state(context)
        history = np.asarray(history, dtype=np.float64)
        if history.shape[0] < self.history_length:
            raise ValueError("past context does not fill the delay register")
        return self.pack(
            torch.as_tensor(reservoir, dtype=self.dtype, device=self.device),
            torch.as_tensor(
                history[-self.history_length :],
                dtype=self.dtype,
                device=self.device,
            ),
            torch.as_tensor(topology, dtype=self.dtype, device=self.device),
        )

    def current_output(self, markov_state: Tensor) -> Tensor:
        """Decode the current latent coordinate of a Markov state."""

        current = markov_state[self.slices.current_latent]
        return current @ self.components + self.pca_mean

    def deterministic_step(
        self, markov_state: Tensor, control: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Advance the frozen conditional mean by one controlled sample.

        The persistent input is exactly ``control @ world.control_map`` and is
        stored in the new latent delay state.  If physical residual preservation
        is enabled, the PCA-orthogonal component is also included in the
        returned output and in the topology/reservoir update.
        """

        if control.shape != (self.actuator_dim,):
            raise ValueError("control has invalid shape")
        reservoir, history, topology = self.unpack(markov_state)
        current = history[-1]
        feature = torch.cat(
            [
                reservoir,
                current,
                topology,
                *(history[-1 - delay] for delay in self.delays),
            ]
        )
        standardized = (feature - self.feature_mean) / self.feature_scale
        mean_increment = standardized @ self.drift_coef.T + self.drift_intercept
        latent_control = control @ self.control_map
        following = (
            current
            + mean_increment
            + self.control_step_scale * latent_control
        )

        updated_history = torch.cat([history[1:], following[None]], dim=0)
        scaled_following = following @ self.components + self.pca_mean
        if self.preserve_physical_control_residual:
            channel_control = control @ self.control_channel_map
            projected_channel_control = latent_control @ self.components
            scaled_following = scaled_following + self.control_step_scale * (
                channel_control - projected_channel_control
            )

        next_topology = (
            scaled_following @ self.adjacency.T
        ) @ self.components.T
        drive = torch.cat(
            [
                following,
                next_topology,
                *(updated_history[-1 - delay] for delay in self.delays),
            ]
        )
        proposal = torch.tanh(drive @ self.w_in.T + reservoir @ self.w_res.T)
        next_reservoir = (
            (1.0 - self.leak_rate) * reservoir
            + self.leak_rate * proposal
        )
        next_state = self.pack(next_reservoir, updated_history, next_topology)
        return next_state, scaled_following


def _orthonormal_projection(
    projection: Tensor | None,
    *,
    state_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if projection is None:
        return torch.eye(state_dim, dtype=dtype, device=device)
    matrix = torch.as_tensor(projection, dtype=dtype, device=device)
    if matrix.ndim != 2 or matrix.shape[0] != state_dim:
        raise ValueError("projection must have shape (full_state, reduced_state)")
    if matrix.shape[1] == 0:
        raise ValueError("projection cannot have zero columns")
    gram = matrix.T @ matrix
    identity = torch.eye(
        matrix.shape[1], dtype=dtype, device=device
    )
    if torch.allclose(gram, identity, rtol=1e-8, atol=1e-10):
        # Preserve the caller's reduced coordinates.  Re-running QR on an
        # already orthonormal reachability basis may flip column signs, which
        # leaves its span unchanged but invalidates saved reduced-state gains.
        return matrix.contiguous()
    q, r = torch.linalg.qr(matrix, mode="reduced")
    diagonal = torch.abs(torch.diagonal(r))
    tolerance = torch.finfo(dtype).eps * max(matrix.shape) * diagonal.max()
    rank = int(torch.sum(diagonal > tolerance).item())
    if rank != matrix.shape[1]:
        raise ValueError("projection columns are linearly dependent")
    return q


def build_causal_reachability_projection(
    adapter: FrozenGraphRCMarkovAdapter,
    initial_state: Tensor,
    *,
    horizon: int,
    rank: int,
    probe_count: int | None = None,
    snapshot_stride: int = 4,
    seed: int = 20260831,
    condition_limit: float = 1e6,
) -> ReachabilityProjection:
    """Build a model-only reachable basis with full-Markov tangent probes.

    Random controls are derivative directions, not applied outcome-seeking
    commands.  The nominal path always uses zero input, and the function has no
    argument through which a recorded future could enter.
    """

    steps = int(horizon)
    target_rank = int(rank)
    if steps <= 0:
        raise ValueError("horizon must be positive")
    if not 1 <= target_rank <= adapter.state_dim:
        raise ValueError("rank is outside the Markov-state range")
    if int(snapshot_stride) <= 0:
        raise ValueError("snapshot_stride must be positive")
    if float(condition_limit) <= 1:
        raise ValueError("condition_limit must exceed one")
    probes = int(probe_count or min(max(target_rank + 8, 16), 64))
    if probes <= 0:
        raise ValueError("probe_count must be positive")

    state = torch.as_tensor(
        initial_state, dtype=adapter.dtype, device=adapter.device
    ).detach()
    if state.shape != (adapter.state_dim,):
        raise ValueError("initial_state has invalid shape")
    tangent_states = torch.zeros(
        probes, adapter.state_dim, dtype=adapter.dtype, device=adapter.device
    )
    generator = torch.Generator(device=adapter.device)
    generator.manual_seed(int(seed))
    directions = torch.randn(
        probes,
        steps,
        adapter.actuator_dim,
        dtype=adapter.dtype,
        device=adapter.device,
        generator=generator,
    ) / max(adapter.actuator_dim, 1) ** 0.5
    zero_control = torch.zeros(
        adapter.actuator_dim, dtype=adapter.dtype, device=adapter.device
    )
    snapshots: list[Tensor] = []

    for step in range(steps):
        nominal_state = state

        def next_state(x: Tensor, u: Tensor) -> Tensor:
            return adapter.deterministic_step(x, u)[0]

        def one_direction(dx: Tensor, du: Tensor) -> Tensor:
            return torch.func.jvp(
                next_state,
                (nominal_state, zero_control),
                (dx, du),
            )[1]

        tangent_states = torch.vmap(one_direction)(
            tangent_states, directions[:, step]
        ).detach()
        state = adapter.deterministic_step(
            nominal_state, zero_control
        )[0].detach()
        if (step + 1) % int(snapshot_stride) == 0 or step == steps - 1:
            snapshots.append(tangent_states.T)

    snapshot_matrix = torch.cat(snapshots, dim=1)
    left, singular_values, _ = torch.linalg.svd(
        snapshot_matrix, full_matrices=False
    )
    tolerance = (
        torch.finfo(adapter.dtype).eps
        * max(snapshot_matrix.shape)
        * singular_values[0]
    )
    stable_threshold = torch.maximum(
        tolerance,
        singular_values[0] / float(condition_limit),
    )
    stable_rank = int(torch.sum(singular_values >= stable_threshold).item())
    retained = min(target_rank, stable_rank)
    if retained == 0:
        raise RuntimeError("causal tangent probes found no reachable direction")
    basis = left[:, :retained].contiguous()
    energy = singular_values.square()
    explained = float(
        (energy[:retained].sum() / energy.sum().clamp_min(1e-18)).item()
    )
    return ReachabilityProjection(
        matrix=basis,
        singular_values=singular_values.detach(),
        explained_energy_fraction=explained,
        probe_count=probes,
        snapshot_stride=int(snapshot_stride),
        seed=int(seed),
        condition_limit=float(condition_limit),
        retained_condition_number=float(
            (singular_values[0] / singular_values[retained - 1]).item()
        ),
    )


def linearize_causal_rollout(
    adapter: FrozenGraphRCMarkovAdapter,
    initial_state: Tensor,
    *,
    horizon: int,
    projection: Tensor | None = None,
) -> CausalLTVLinearization:
    """Linearize a zero-control causal rollout in a fixed state projection."""

    steps = int(horizon)
    if steps <= 0:
        raise ValueError("horizon must be positive")
    state = torch.as_tensor(
        initial_state, dtype=adapter.dtype, device=adapter.device
    ).detach()
    if state.shape != (adapter.state_dim,):
        raise ValueError("initial_state has invalid shape")
    basis = _orthonormal_projection(
        projection,
        state_dim=adapter.state_dim,
        dtype=adapter.dtype,
        device=adapter.device,
    )
    reduced_dim = int(basis.shape[1])
    actuator_dim = adapter.actuator_dim
    zero_control = torch.zeros(
        actuator_dim, dtype=adapter.dtype, device=adapter.device
    )
    state_directions = torch.cat(
        [
            basis.T,
            torch.zeros(
                actuator_dim,
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
                reduced_dim,
                actuator_dim,
                dtype=adapter.dtype,
                device=adapter.device,
            ),
            torch.eye(
                actuator_dim, dtype=adapter.dtype, device=adapter.device
            ),
        ],
        dim=0,
    )

    transitions: list[Tensor] = []
    controls: list[Tensor] = []
    output_states: list[Tensor] = []
    output_controls: list[Tensor] = []
    nominal_states: list[Tensor] = [state]
    nominal_outputs: list[Tensor] = []

    for _ in range(steps):
        nominal_state = state

        def combined(x: Tensor, u: Tensor) -> Tensor:
            next_state, output = adapter.deterministic_step(x, u)
            return torch.cat([next_state, output])

        nominal_combined = combined(nominal_state, zero_control).detach()

        def one_direction(dx: Tensor, du: Tensor) -> Tensor:
            return torch.func.jvp(
                combined,
                (nominal_state, zero_control),
                (dx, du),
            )[1]

        tangent = torch.vmap(one_direction)(
            state_directions, control_directions
        ).detach()
        state_from_state = tangent[:reduced_dim, : adapter.state_dim].T
        state_from_control = tangent[reduced_dim:, : adapter.state_dim].T
        output_from_state = tangent[:reduced_dim, adapter.state_dim :].T
        output_from_control = tangent[reduced_dim:, adapter.state_dim :].T
        transitions.append(basis.T @ state_from_state)
        controls.append(basis.T @ state_from_control)
        output_states.append(output_from_state)
        output_controls.append(output_from_control)
        state = nominal_combined[: adapter.state_dim]
        nominal_states.append(state)
        nominal_outputs.append(nominal_combined[adapter.state_dim :])

    terminal_state = nominal_states[-1]

    def current_output(x: Tensor) -> Tensor:
        return adapter.current_output(x)

    def terminal_direction(dx: Tensor) -> Tensor:
        return torch.func.jvp(
            current_output,
            (terminal_state,),
            (dx,),
        )[1]

    terminal_output = torch.vmap(terminal_direction)(basis.T).T.detach()
    return CausalLTVLinearization(
        state_transition=torch.stack(transitions),
        persistent_control=torch.stack(controls),
        output_state_jacobian=torch.stack(output_states),
        output_control_jacobian=torch.stack(output_controls),
        terminal_output_state_jacobian=terminal_output,
        nominal_markov_states=torch.stack(nominal_states),
        nominal_outputs=torch.stack(nominal_outputs),
        projection=basis,
        persistent_latent_control_map=adapter.control_map.detach().clone(),
        control_step_scale=adapter.control_step_scale,
    )


def _psd_weight(
    value: float | Tensor,
    dimension: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.ndim == 0:
        matrix = tensor * torch.eye(dimension, dtype=dtype, device=device)
    elif tensor.ndim == 1 and tensor.shape == (dimension,):
        matrix = torch.diag(tensor)
    elif tensor.ndim == 2 and tensor.shape == (dimension, dimension):
        matrix = 0.5 * (tensor + tensor.T)
    else:
        raise ValueError(f"{name} has invalid shape")
    eigenvalues = torch.linalg.eigvalsh(matrix)
    if float(eigenvalues.min()) < -1e-9:
        raise ValueError(f"{name} must be positive semidefinite")
    return matrix


def _conditioned_gain(
    hessian: Tensor,
    gradient: Tensor,
    *,
    ridge: float,
    condition_limit: float,
) -> tuple[Tensor, Tensor]:
    symmetric = 0.5 * (hessian + hessian.T)
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    maximum = eigenvalues[-1].clamp_min(1e-12)
    required_floor = torch.maximum(
        maximum / float(condition_limit),
        maximum.new_tensor(float(ridge)),
    )
    added = (required_floor - eigenvalues[0]).clamp_min(0.0)
    regularized = symmetric + added * torch.eye(
        symmetric.shape[0], dtype=symmetric.dtype, device=symmetric.device
    )
    return torch.linalg.solve(regularized, gradient), added


def solve_mean_field_ltv_riccati(
    linearization: CausalLTVLinearization,
    *,
    running_mean_output_weight: float | Tensor,
    running_deviation_output_weight: float | Tensor,
    terminal_mean_output_weight: float | Tensor,
    terminal_deviation_output_weight: float | Tensor,
    control_weight: float | Tensor,
    reference_mean_output: Tensor | None = None,
    terminal_reference_mean_output: Tensor | None = None,
    control_ridge: float = 1e-6,
    condition_limit: float = 1e6,
) -> MeanFieldRiccatiBackbone:
    """Solve two finite-horizon Riccati recursions for mean and deviation.

    The stage output is the next decoded physical state.  Its direct control
    Jacobian is retained, so a PCA-orthogonal physical residual contributes the
    correct ``x-u`` cross term and actuator curvature.

    When ``reference_mean_output`` is supplied, it must have shape
    ``(horizon, output_dim)`` and is used only for the population-mean LQT
    recursion.  With ``delta_state`` measured from the causal zero-control
    nominal Markov path, the returned mean command is

    ``u_mean[t] = -K_mean[t] @ delta_state[t] - k_mean[t]``.

    The deviation recursion remains a zero-reference regulator, as required
    for covariance contraction without shifting the population mean.
    """

    if float(control_ridge) < 0:
        raise ValueError("control_ridge must be non-negative")
    if float(condition_limit) <= 1:
        raise ValueError("condition_limit must exceed one")
    dtype = linearization.state_transition.dtype
    device = linearization.state_transition.device
    output_dim = int(linearization.output_state_jacobian.shape[1])
    actuator_dim = linearization.actuator_dim
    mean_running = _psd_weight(
        running_mean_output_weight,
        output_dim,
        dtype=dtype,
        device=device,
        name="running_mean_output_weight",
    )
    deviation_running = _psd_weight(
        running_deviation_output_weight,
        output_dim,
        dtype=dtype,
        device=device,
        name="running_deviation_output_weight",
    )
    mean_terminal = _psd_weight(
        terminal_mean_output_weight,
        output_dim,
        dtype=dtype,
        device=device,
        name="terminal_mean_output_weight",
    )
    deviation_terminal = _psd_weight(
        terminal_deviation_output_weight,
        output_dim,
        dtype=dtype,
        device=device,
        name="terminal_deviation_output_weight",
    )
    control = _psd_weight(
        control_weight,
        actuator_dim,
        dtype=dtype,
        device=device,
        name="control_weight",
    )

    if reference_mean_output is None:
        mean_output_offset = torch.zeros_like(linearization.nominal_outputs)
        terminal_mean_offset = torch.zeros(
            output_dim, dtype=dtype, device=device
        )
    else:
        reference = torch.as_tensor(
            reference_mean_output, dtype=dtype, device=device
        )
        if reference.shape != (linearization.horizon, output_dim):
            raise ValueError(
                "reference_mean_output must have shape (horizon, output_dim)"
            )
        # The LTV state is a perturbation around the causal zero-control model
        # path, so the affine tracking residual at zero perturbation is y0-yref.
        mean_output_offset = linearization.nominal_outputs - reference
        if terminal_reference_mean_output is None:
            terminal_reference = reference[-1]
        else:
            terminal_reference = torch.as_tensor(
                terminal_reference_mean_output, dtype=dtype, device=device
            )
            if terminal_reference.shape != (output_dim,):
                raise ValueError(
                    "terminal_reference_mean_output has invalid shape"
                )
        terminal_nominal = linearization.nominal_outputs[-1]
        terminal_mean_offset = terminal_nominal - terminal_reference

    def backward(
        running_output: Tensor,
        terminal_output: Tensor,
        output_offset: Tensor,
        terminal_offset: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        horizon = linearization.horizon
        reduced_dim = linearization.reduced_state_dim
        values = torch.zeros(
            horizon + 1,
            reduced_dim,
            reduced_dim,
            dtype=dtype,
            device=device,
        )
        gains = torch.zeros(
            horizon,
            actuator_dim,
            reduced_dim,
            dtype=dtype,
            device=device,
        )
        feedforward = torch.zeros(
            horizon,
            actuator_dim,
            dtype=dtype,
            device=device,
        )
        linear_values = torch.zeros(
            horizon + 1,
            reduced_dim,
            dtype=dtype,
            device=device,
        )
        regularization = torch.zeros(horizon, dtype=dtype, device=device)
        terminal_c = linearization.terminal_output_state_jacobian
        values[-1] = terminal_c.T @ terminal_output @ terminal_c
        linear_values[-1] = terminal_c.T @ terminal_output @ terminal_offset
        for step in range(horizon - 1, -1, -1):
            a = linearization.state_transition[step]
            b = linearization.persistent_control[step]
            e = linearization.output_state_jacobian[step]
            f = linearization.output_control_jacobian[step]
            p_next = values[step + 1]
            q_stage = e.T @ running_output @ e
            n_stage = e.T @ running_output @ f
            r_stage = control + f.T @ running_output @ f
            q_linear = e.T @ running_output @ output_offset[step]
            r_linear = f.T @ running_output @ output_offset[step]
            hessian = r_stage + b.T @ p_next @ b
            gradient = n_stage.T + b.T @ p_next @ a
            gain, added = _conditioned_gain(
                hessian,
                gradient,
                ridge=float(control_ridge),
                condition_limit=float(condition_limit),
            )
            affine_gradient = r_linear + b.T @ linear_values[step + 1]
            affine_gain, affine_added = _conditioned_gain(
                hessian,
                affine_gradient[:, None],
                ridge=float(control_ridge),
                condition_limit=float(condition_limit),
            )
            affine_gain = affine_gain[:, 0]
            gains[step] = gain
            feedforward[step] = affine_gain
            regularization[step] = torch.maximum(added, affine_added)
            left = n_stage + a.T @ p_next @ b
            value = q_stage + a.T @ p_next @ a - left @ gain
            values[step] = 0.5 * (value + value.T)
            linear_values[step] = (
                q_linear
                + a.T @ linear_values[step + 1]
                - left @ affine_gain
            )
        return gains, feedforward, values, linear_values, regularization

    (
        mean_gain,
        mean_feedforward,
        mean_value,
        mean_value_linear,
        mean_regularization,
    ) = backward(
        mean_running,
        mean_terminal,
        mean_output_offset,
        terminal_mean_offset,
    )
    (
        deviation_gain,
        deviation_feedforward,
        deviation_value,
        deviation_value_linear,
        deviation_regularization,
    ) = backward(
        deviation_running,
        deviation_terminal,
        torch.zeros_like(linearization.nominal_outputs),
        torch.zeros(output_dim, dtype=dtype, device=device),
    )
    if torch.count_nonzero(deviation_feedforward) != 0:
        raise RuntimeError("zero-reference deviation recursion gained an affine term")
    if torch.count_nonzero(deviation_value_linear) != 0:
        raise RuntimeError("zero-reference deviation value gained a linear term")
    return MeanFieldRiccatiBackbone(
        mean_gain_reduced=mean_gain,
        deviation_gain_reduced=deviation_gain,
        mean_feedforward=mean_feedforward,
        mean_value_hessian=mean_value,
        mean_value_linear=mean_value_linear,
        deviation_value_hessian=deviation_value,
        mean_regularization=mean_regularization,
        deviation_regularization=deviation_regularization,
        projection=linearization.projection,
    )
