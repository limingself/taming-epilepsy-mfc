"""Mean-field square-wave actor--critic control for a fitted Graph-RC SDE.

The controller in this module is deliberately a model-space intervention.  A
population of RC-SDE particles is summarized by its latent mean and variance.
One shared actor then selects, for every actuator, one of a small odd number of
exact symmetric levels (five by default) through a straight-through
Gumbel--Softmax sample.  The
sample is held for a prescribed number of EEG samples, producing a genuine
piecewise-constant square-wave control rather than a post-hoc stair plot.

Only the supplied actuator columns enter the fitted latent dynamics.  The
Wasserstein critic and the auxiliary distribution losses operate on all
decoded standardized EEG channels, so unactuated nodes remain part of the
optimization target.  No transition projection, state clipping, or terminal
sample replacement is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
from scipy.linalg import expm
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SquareWaveMFCConfig:
    """Hyperparameters for stochastic square-wave mean-field control."""

    horizon_samples: int = 128
    pulse_width_samples: int = 8
    amplitude_limit: float = 0.75
    control_levels: int = 5
    control_mode: str = "square"
    smooth_spline_basis_count: int = 10
    smooth_feedback_alpha: float = 0.18
    smooth_residual_scale: float = 0.35
    control_input_gain: float = 4.0
    actor_hidden_size: int = 96
    critic_hidden_size: int = 96
    train_epochs: int = 180
    critic_steps: int = 2
    batch_size: int = 48
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    gradient_penalty_weight: float = 10.0
    adversarial_weight: float = 0.25
    conditional_adversarial_weight: float = 0.50
    temporal_adversarial_weight: float = 0.30
    critic_conditional_weight: float = 1.0
    critic_temporal_weight: float = 0.6
    critic_marginal_samples: int = 1536
    critic_temporal_samples: int = 384
    temporal_window_samples: int = 8
    marginal_quantile_weight: float = 16.0
    worst_channel_quantile_weight: float = 8.0
    per_time_quantile_weight: float = 14.0
    terminal_distribution_weight: float = 10.0
    reference_band_weight: float = 5.0
    increment_quantile_weight: float = 5.0
    autocorrelation_weight: float = 2.0
    path_mean_weight: float = 3.0
    trajectory_template_weight: float = 0.0
    trajectory_velocity_weight: float = 0.0
    feedback_alignment_weight: float = 8.0
    lyapunov_decrease_weight: float = 6.0
    action_entropy_weight: float = 1.5
    moment_weight: float = 3.0
    mean_matching_weight: float = 0.0
    mean_non_degradation_weight: float = 0.0
    variance_matching_weight: float = 0.0
    worst_moment_weight: float = 0.0
    mmd_weight: float = 2.0
    energy_weight: float = 0.005
    switching_weight: float = 0.006
    control_smoothness_weight: float = 0.0
    control_curvature_weight: float = 0.0
    topology_weight: float = 0.03
    diffusion_scale: float = 1.0
    gumbel_temperature_start: float = 1.25
    gumbel_temperature_end: float = 0.30
    loss_tail_samples: int = 32
    random_seed: int = 11

    def validate(self) -> None:
        if self.horizon_samples < 2:
            raise ValueError("horizon_samples must be at least two")
        if self.pulse_width_samples < 1:
            raise ValueError("pulse_width_samples must be positive")
        if self.horizon_samples % self.pulse_width_samples != 0:
            raise ValueError("horizon_samples must be divisible by pulse_width_samples")
        if self.amplitude_limit <= 0:
            raise ValueError("amplitude_limit must be positive")
        if self.control_levels < 3 or self.control_levels % 2 == 0:
            raise ValueError("control_levels must be an odd integer of at least three")
        if self.control_mode not in {"square", "smooth_feedback"}:
            raise ValueError("control_mode must be 'square' or 'smooth_feedback'")
        if self.smooth_spline_basis_count < 4:
            raise ValueError("smooth_spline_basis_count must be at least four")
        if not 0.0 < self.smooth_feedback_alpha <= 1.0:
            raise ValueError("smooth_feedback_alpha must be in (0, 1]")
        if self.smooth_residual_scale < 0:
            raise ValueError("smooth_residual_scale must be non-negative")
        if self.control_input_gain <= 0:
            raise ValueError("control_input_gain must be positive")
        if self.train_epochs < 1 or self.critic_steps < 1 or self.batch_size < 4:
            raise ValueError("training counts are invalid")
        if not 1 <= self.loss_tail_samples <= self.horizon_samples:
            raise ValueError("loss_tail_samples is outside the rollout horizon")
        if not 2 <= self.temporal_window_samples <= self.loss_tail_samples:
            raise ValueError("temporal_window_samples must fit inside the loss tail")
        if self.critic_marginal_samples < 32 or self.critic_temporal_samples < 16:
            raise ValueError("conditional critic sample counts are too small")


class MeanFieldSquareWaveActor(nn.Module):
    """Population-feedback actor with exact multilevel square-wave outputs."""

    def __init__(
        self,
        latent_dim: int,
        n_actuators: int,
        hidden_size: int,
        amplitude_limit: float,
        control_levels: int = 5,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.n_actuators = int(n_actuators)
        self.amplitude_limit = float(amplitude_limit)
        self.control_levels = int(control_levels)
        # Each particle/agent sees its own state together with population
        # moments.  This is the McKean--Vlasov feedback form pi(x, mu_t), which
        # can shape both the mean and the spread of the particle law.
        input_dim = 6 * self.latent_dim + 3
        self.network = nn.Sequential(
            nn.Linear(input_dim, int(hidden_size)),
            nn.LayerNorm(int(hidden_size)),
            nn.SiLU(),
            nn.Linear(int(hidden_size), int(hidden_size)),
            nn.SiLU(),
            nn.Linear(int(hidden_size), self.n_actuators * self.control_levels),
        )
        # Each channel learns one time-invariant level magnitude.  The temporal
        # actor chooses only polarity/off, keeping every pulse exactly square.
        self.amplitude_logits = nn.Parameter(torch.zeros(self.n_actuators))
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            final.bias.view(self.n_actuators, self.control_levels)[
                :, self.control_levels // 2
            ] = 1.5

    def forward(
        self,
        particles: Tensor,
        reference_mean: Tensor,
        reference_variance: Tensor,
        time_fraction: float,
        *,
        temperature: float,
        stochastic: bool,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        mean = particles.mean(dim=0)
        variance = particles.var(dim=0, unbiased=False).clamp_min(1e-6)
        ref_var = reference_variance.clamp_min(1e-6)
        phase = 2.0 * math.pi * float(time_fraction)
        timing = particles.new_tensor(
            [math.sin(phase), math.cos(phase), 1.0 - float(time_fraction)]
        )
        batch = int(particles.shape[0])
        timing_batch = timing.unsqueeze(0).expand(batch, -1)
        feature = torch.cat(
            [
                particles,
                mean.unsqueeze(0).expand(batch, -1),
                torch.log(variance).unsqueeze(0).expand(batch, -1),
                reference_mean.unsqueeze(0) - particles,
                (reference_mean - mean).unsqueeze(0).expand(batch, -1),
                (torch.log(ref_var) - torch.log(variance)).unsqueeze(0).expand(
                    batch, -1
                ),
                timing_batch,
            ],
            dim=1,
        )
        logits = self.network(feature).view(
            batch, self.n_actuators, self.control_levels
        )
        if stochastic:
            # Explicit generator support makes evaluation reproducible.  The
            # straight-through construction returns exact one-hot states in the
            # forward pass while retaining soft gradients for the actor.
            uniform = torch.rand(
                logits.shape,
                dtype=logits.dtype,
                device=logits.device,
                generator=generator,
            ).clamp_(1e-6, 1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(uniform))
            soft = torch.softmax((logits + gumbel) / float(temperature), dim=-1)
            hard_index = soft.argmax(dim=-1)
            hard = F.one_hot(
                hard_index, num_classes=self.control_levels
            ).to(dtype=soft.dtype)
            states = hard + soft - soft.detach()
        else:
            hard_index = logits.argmax(dim=-1)
            states = F.one_hot(
                hard_index, num_classes=self.control_levels
            ).to(dtype=logits.dtype)
            soft = torch.softmax(logits / max(float(temperature), 1e-3), dim=-1)
        amplitudes = self.amplitude_limit * (
            0.25 + 0.75 * torch.sigmoid(self.amplitude_logits)
        )
        levels = torch.linspace(
            -1.0,
            1.0,
            self.control_levels,
            dtype=logits.dtype,
            device=logits.device,
        )
        if self.training and stochastic:
            # Straight-through values carry gradients during fitting.  In
            # evaluation mode return the hard state itself so saved controls
            # have exactly one of the configured per-actuator levels, without residual
            # floating-point cancellation from soft - soft.detach().
            control = (states @ levels) * amplitudes.unsqueeze(0)
        else:
            control = (hard @ levels) * amplitudes.unsqueeze(0)
        active_probability = 1.0 - soft[:, :, self.control_levels // 2]
        return control, logits, active_probability


class MeanFieldSmoothFeedbackActor(nn.Module):
    """Smooth McKean--Vlasov feedback actor with a directly optimized spline.

    The shared feed-forward command is represented by cubic B-spline
    coefficients, which are ordinary trainable parameters rather than samples
    drawn from a discrete action alphabet.  A state/distribution feedback term
    and a small learned residual correct that common command for each RC-SDE
    particle.  The rollout applies a causal first-order low-pass filter, so the
    delivered action is bounded and continuous without post-hoc smoothing.
    """

    def __init__(
        self,
        latent_dim: int,
        n_actuators: int,
        hidden_size: int,
        amplitude_limit: float,
        control_map: Tensor,
        horizon_samples: int,
        spline_basis_count: int = 10,
        residual_scale: float = 0.35,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.n_actuators = int(n_actuators)
        self.amplitude_limit = float(amplitude_limit)
        self.horizon_samples = int(horizon_samples)
        self.spline_basis_count = int(spline_basis_count)
        self.residual_scale = float(residual_scale)
        input_dim = 6 * self.latent_dim + 3
        self.network = nn.Sequential(
            nn.Linear(input_dim, int(hidden_size)),
            nn.LayerNorm(int(hidden_size)),
            nn.SiLU(),
            nn.Linear(int(hidden_size), int(hidden_size)),
            nn.SiLU(),
            nn.Linear(int(hidden_size), self.n_actuators),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.spline_coefficients = nn.Parameter(
            torch.zeros(self.spline_basis_count, self.n_actuators)
        )
        # Positive gains preserve the interpretation of the explicit terms as
        # feedback toward the patient-specific reference law.
        self.state_feedback_logits = nn.Parameter(
            torch.full((self.n_actuators,), -4.0)
        )
        self.distribution_feedback_logits = nn.Parameter(
            torch.full((self.n_actuators,), -4.0)
        )
        pseudo_inverse = torch.linalg.pinv(control_map.detach())
        column_norm = pseudo_inverse.square().sum(dim=0).sqrt().clamp_min(1e-6)
        self.register_buffer(
            "normalized_control_pseudoinverse",
            pseudo_inverse / column_norm.unsqueeze(0),
        )
        self.register_buffer(
            "spline_basis",
            self._cubic_spline_basis(
                self.horizon_samples,
                self.spline_basis_count,
                dtype=control_map.dtype,
                device=control_map.device,
            ),
        )

    @staticmethod
    def _cubic_spline_basis(
        horizon: int,
        basis_count: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """Return a clamped uniform cubic B-spline design matrix."""

        degree = 3
        internal_count = basis_count - degree - 1
        if internal_count > 0:
            internal = np.linspace(0.0, 1.0, internal_count + 2)[1:-1]
        else:
            internal = np.empty(0, dtype=np.float64)
        knots = np.concatenate(
            [np.zeros(degree + 1), internal, np.ones(degree + 1)]
        )
        times = np.linspace(0.0, 1.0, int(horizon), dtype=np.float64)
        basis = np.zeros((int(horizon), int(basis_count)), dtype=np.float64)
        from scipy.interpolate import BSpline

        for index in range(int(basis_count)):
            coefficient = np.zeros(int(basis_count), dtype=np.float64)
            coefficient[index] = 1.0
            basis[:, index] = BSpline(
                knots, coefficient, degree, extrapolate=False
            )(times)
        basis = np.nan_to_num(basis, nan=0.0)
        basis /= np.maximum(basis.sum(axis=1, keepdims=True), 1e-12)
        return torch.as_tensor(basis, dtype=dtype, device=device)

    def forward(
        self,
        particles: Tensor,
        reference_mean: Tensor,
        reference_variance: Tensor,
        time_fraction: float,
        *,
        temperature: float,
        stochastic: bool,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del temperature, stochastic, generator
        mean = particles.mean(dim=0)
        variance = particles.var(dim=0, unbiased=False).clamp_min(1e-6)
        ref_var = reference_variance.clamp_min(1e-6)
        reference_scale = torch.sqrt(ref_var).clamp_min(0.10)
        phase = 2.0 * math.pi * float(time_fraction)
        timing = particles.new_tensor(
            [math.sin(phase), math.cos(phase), 1.0 - float(time_fraction)]
        )
        batch = int(particles.shape[0])
        feature = torch.cat(
            [
                particles,
                mean.unsqueeze(0).expand(batch, -1),
                torch.log(variance).unsqueeze(0).expand(batch, -1),
                reference_mean.unsqueeze(0) - particles,
                (reference_mean - mean).unsqueeze(0).expand(batch, -1),
                (torch.log(ref_var) - torch.log(variance)).unsqueeze(0).expand(
                    batch, -1
                ),
                timing.unsqueeze(0).expand(batch, -1),
            ],
            dim=1,
        )
        residual = self.network(feature)
        state_error = (reference_mean.unsqueeze(0) - particles) / reference_scale
        scale_ratio = torch.sqrt(ref_var / variance).clamp(0.25, 4.0)
        distribution_target = reference_mean.unsqueeze(0) + scale_ratio.unsqueeze(
            0
        ) * (particles - mean.unsqueeze(0))
        distribution_error = (distribution_target - particles) / reference_scale
        state_command = state_error @ self.normalized_control_pseudoinverse
        distribution_command = (
            distribution_error @ self.normalized_control_pseudoinverse
        )
        state_gain = F.softplus(self.state_feedback_logits).unsqueeze(0)
        distribution_gain = F.softplus(
            self.distribution_feedback_logits
        ).unsqueeze(0)
        step = int(
            np.clip(
                np.rint(float(time_fraction) * max(self.horizon_samples - 1, 1)),
                0,
                self.horizon_samples - 1,
            )
        )
        spline_command = (
            self.spline_basis[step] @ self.spline_coefficients
        ).unsqueeze(0)
        command = (
            spline_command
            + state_gain * state_command
            + distribution_gain * distribution_command
            + self.residual_scale * residual
        )
        control = self.amplitude_limit * torch.tanh(command)
        # A one-element pseudo-logit keeps the critic/controller rollout API
        # unchanged; its entropy is exactly zero for deterministic smooth control.
        pseudo_logits = command.unsqueeze(-1)
        active_fraction = control.abs() / max(self.amplitude_limit, 1e-6)
        return control, pseudo_logits, active_fraction


class WassersteinCritic(nn.Module):
    """Small WGAN-GP critic on decoded standardized full-channel states."""

    def __init__(self, n_channels: int, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(n_channels), int(hidden_size)),
            nn.LeakyReLU(0.2),
            nn.Linear(int(hidden_size), int(hidden_size)),
            nn.LeakyReLU(0.2),
            nn.Linear(int(hidden_size), 1),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values).squeeze(-1)


class ConditionalTrajectoryCritic(nn.Module):
    """Three-head critic for joint, node-conditional and temporal laws.

    The joint head tests instantaneous full-network states.  The conditional
    head tests every marginal with an explicit node identity, preventing large
    easy-to-control channels from dominating the adversarial signal.  The
    temporal head tests short same-node trajectory patches and therefore
    penalizes unrealistic autocorrelation and local waveform statistics.
    """

    def __init__(
        self,
        n_channels: int,
        hidden_size: int,
        temporal_window: int,
        embedding_size: int = 8,
    ) -> None:
        super().__init__()
        self.n_channels = int(n_channels)
        self.temporal_window = int(temporal_window)
        hidden = int(hidden_size)
        embedding = int(embedding_size)
        self.node_embedding = nn.Embedding(self.n_channels, embedding)
        self.state_network = nn.Sequential(
            nn.Linear(self.n_channels, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )
        self.marginal_network = nn.Sequential(
            nn.Linear(1 + embedding, hidden // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden // 2, hidden // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden // 2, 1),
        )
        self.temporal_network = nn.Sequential(
            nn.Linear(self.temporal_window + embedding, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def state_score(self, values: Tensor) -> Tensor:
        return self.state_network(values).squeeze(-1)

    def marginal_score(self, values: Tensor, node_indices: Tensor) -> Tensor:
        embedding = self.node_embedding(node_indices)
        return self.marginal_network(torch.cat([values, embedding], dim=1)).squeeze(-1)

    def temporal_score(self, patches: Tensor, node_indices: Tensor) -> Tensor:
        embedding = self.node_embedding(node_indices)
        return self.temporal_network(torch.cat([patches, embedding], dim=1)).squeeze(-1)

    def forward(self, values: Tensor) -> Tensor:
        return self.state_score(values)


class TorchGraphRCSDE:
    """Differentiable copy of a fitted state-dependent residual Graph-RC SDE."""

    def __init__(
        self,
        fitted_model,
        actuator_indices: Sequence[int],
        sampling_rate_hz: float,
        *,
        control_graph_diffusion_time: float = 0.0,
        preserve_physical_control_residual: bool = False,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        if fitted_model.transform is None or fitted_model.transform.pca is None:
            raise RuntimeError("the RC model must be fitted")
        if fitted_model.config.diffusion_mode != "state_dependent":
            raise ValueError("square-wave MFC requires the state-dependent RC-SDE")
        self.model = fitted_model
        self.dtype = dtype
        self.device = torch.device(device)
        self.dt = 1.0 / float(sampling_rate_hz)
        self.q = int(fitted_model.q)
        self.n_channels = int(fitted_model.transform.n_channels_)
        self.reservoir_size = int(fitted_model.config.reservoir_size)
        self.maximum_delay = int(fitted_model.maximum_delay)
        self.delays = tuple(int(v) for v in fitted_model.config.delays_samples)
        self.leak_rate = float(fitted_model.config.leak_rate)
        self.actuator_indices = np.asarray(actuator_indices, dtype=np.int64)
        self.control_graph_diffusion_time = float(control_graph_diffusion_time)
        self.preserve_physical_control_residual = bool(
            preserve_physical_control_residual
        )
        if self.control_graph_diffusion_time < 0:
            raise ValueError("control_graph_diffusion_time must be non-negative")
        if len(self.actuator_indices) == 0:
            raise ValueError("at least one actuator is required")
        if np.any(self.actuator_indices < 0) or np.any(
            self.actuator_indices >= self.n_channels
        ):
            raise ValueError("actuator index is outside the channel range")

        def tensor(value) -> Tensor:
            return torch.as_tensor(value, dtype=dtype, device=self.device)

        transform = fitted_model.transform
        self.components = tensor(transform.pca.components_)
        self.pca_mean = tensor(transform.pca.mean_)
        self.adjacency = tensor(fitted_model.adjacency)
        self.w_in = tensor(fitted_model.w_in)
        self.w_res = tensor(fitted_model.w_res)
        self.feature_mean = tensor(fitted_model.feature_scaler.mean_)
        self.feature_scale = tensor(fitted_model.feature_scaler.scale_)
        self.regime_adaptive = bool(
            getattr(fitted_model, "is_regime_adaptive", False)
        )
        if self.regime_adaptive:
            self.gate_coef = tensor(fitted_model.gate_coef)
            self.gate_intercept = tensor(fitted_model.gate_intercept)
            self.drift_gate_logit_offset = tensor(
                getattr(fitted_model, "drift_gate_logit_offset", 0.0)
            )
            self.diffusion_gate_logit_offset = tensor(
                getattr(fitted_model, "diffusion_gate_logit_offset", 0.0)
            )
            self.gate_feature_mean = tensor(fitted_model.gate_feature_mean)
            self.gate_feature_scale = tensor(fitted_model.gate_feature_scale)
            self.gate_window_samples = int(fitted_model.gate_window_samples)
            self.gate_feature_mode = str(fitted_model.gate_feature_mode)
            self.regime_diffusion_mode = str(
                fitted_model.regime_diffusion_mode
            )
            self.shared_variance_coef = tensor(
                fitted_model.shared_variance_coef
            )
            self.shared_variance_intercept = tensor(
                fitted_model.shared_variance_intercept
            )
            self.shared_variance_calibration = tensor(
                fitted_model.shared_variance_calibration
            )
            self.shared_log_variance_bounds = tensor(
                fitted_model.shared_log_variance_bounds
            )
            self.shared_correlation_root = tensor(
                fitted_model.shared_correlation_root
            )
            for prefix in ("ictal", "interictal"):
                setattr(
                    self,
                    f"{prefix}_drift_coef",
                    tensor(getattr(fitted_model, f"{prefix}_drift_coef")),
                )
                setattr(
                    self,
                    f"{prefix}_drift_intercept",
                    tensor(getattr(fitted_model, f"{prefix}_drift_intercept")),
                )
                setattr(
                    self,
                    f"{prefix}_variance_coef",
                    tensor(getattr(fitted_model, f"{prefix}_variance_coef")),
                )
                setattr(
                    self,
                    f"{prefix}_variance_intercept",
                    tensor(getattr(fitted_model, f"{prefix}_variance_intercept")),
                )
                setattr(
                    self,
                    f"{prefix}_variance_calibration",
                    tensor(getattr(fitted_model, f"{prefix}_variance_calibration")),
                )
                setattr(
                    self,
                    f"{prefix}_log_variance_bounds",
                    tensor(getattr(fitted_model, f"{prefix}_log_variance_bounds")),
                )
                setattr(
                    self,
                    f"{prefix}_correlation_root",
                    tensor(getattr(fitted_model, f"{prefix}_correlation_root")),
                )
        else:
            self.drift_coef = tensor(fitted_model.drift.coef_)
            self.drift_intercept = tensor(fitted_model.drift.intercept_)
            self.variance_coef = tensor(fitted_model.diffusion_variance_model.coef_)
            self.variance_intercept = tensor(
                fitted_model.diffusion_variance_model.intercept_
            )
            self.variance_calibration = tensor(
                fitted_model.diffusion_variance_calibration
            )
            self.log_variance_bounds = tensor(
                fitted_model.diffusion_log_variance_bounds
            )
            self.correlation_root = tensor(fitted_model.diffusion_correlation_root)
        # Direct stimulation remains confined to the selected nodes.  Its
        # model-space effect may then propagate over the PLV graph through a
        # random-walk heat kernel H_tau = exp(-tau L_rw).  tau=0 recovers the
        # original strictly local input map.
        selector = np.zeros((len(self.actuator_indices), self.n_channels), dtype=np.float64)
        selector[np.arange(len(self.actuator_indices)), self.actuator_indices] = 1.0
        adjacency_np = np.asarray(fitted_model.adjacency, dtype=np.float64)
        degree = adjacency_np.sum(axis=1)
        transition = adjacency_np / np.maximum(degree[:, None], 1e-12)
        random_walk_laplacian = np.eye(self.n_channels) - transition
        heat_kernel = expm(
            -self.control_graph_diffusion_time * random_walk_laplacian
        )
        self.control_channel_map = tensor(selector @ heat_kernel)
        self.control_map = (self.control_channel_map @ self.components.T).contiguous()

    def scaled_from_latent(self, latent: Tensor) -> Tensor:
        return latent @ self.components + self.pca_mean

    def topology_from_scaled(self, scaled: Tensor) -> Tensor:
        return (scaled @ self.adjacency.T) @ self.components.T

    def context_state(
        self, context: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state, history, topology = self.model._context_state(
            np.asarray(context, dtype=np.float64)
        )
        required = self.maximum_delay + 1
        if self.regime_adaptive:
            required = max(required, self.gate_window_samples)
        if len(history) < required:
            raise ValueError("context does not contain the maximum delay")
        return (
            np.asarray(state, dtype=np.float32),
            np.asarray(history[-required:], dtype=np.float32),
            np.asarray(topology, dtype=np.float32),
        )

    def rollout(
        self,
        states0: Tensor,
        history0: Tensor,
        topology0: Tensor,
        *,
        horizon_samples: int,
        noise: Tensor,
        actor: MeanFieldSquareWaveActor | MeanFieldSmoothFeedbackActor | None,
        reference_mean: Tensor,
        reference_variance: Tensor,
        pulse_width_samples: int,
        temperature: float,
        stochastic_actor: bool,
        actor_generator: torch.Generator | None = None,
        diffusion_scale: float = 1.0,
        diffusion_control_strength: float = 0.0,
        independent_diffusion_control: bool = False,
        control_input_gain: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch = int(states0.shape[0])
        horizon = int(horizon_samples)
        if noise.shape != (batch, horizon, self.q):
            raise ValueError("noise has an incompatible shape")
        states = states0
        history = [history0[:, i] for i in range(history0.shape[1])]
        topology = topology0
        scaled_values = [self.scaled_from_latent(history[-1])]
        latent_values = [history[-1]]
        controls: list[Tensor] = []
        active_probabilities: list[Tensor] = []
        action_entropies: list[Tensor] = []
        regime_probabilities: list[Tensor] = []
        regime_logits: list[Tensor] = []
        diffusion_modulations: list[Tensor] = []
        diffusion_actions: list[Tensor] = []
        current_control = states0.new_zeros(batch, len(self.actuator_indices))
        current_diffusion_action = torch.zeros_like(current_control)
        previous_applied_control = torch.zeros_like(current_control)
        current_active = states0.new_zeros(batch, len(self.actuator_indices))
        current_entropy = states0.new_zeros(batch, len(self.actuator_indices))
        for step in range(horizon):
            smooth_actor = isinstance(actor, MeanFieldSmoothFeedbackActor) or bool(
                getattr(actor, "continuous_control", False)
            )
            update_control = actor is not None and (
                smooth_actor or step % int(pulse_width_samples) == 0
            )
            if update_control:
                prior_control = current_control
                step_reference_mean = (
                    reference_mean[step]
                    if reference_mean.ndim == 2
                    else reference_mean
                )
                actor_kwargs = {
                    "temperature": float(temperature),
                    "stochastic": bool(stochastic_actor),
                    "generator": actor_generator,
                }
                if bool(getattr(actor, "uses_control_history", False)):
                    actor_kwargs.update(
                        {
                            "previous_control": current_control,
                            "previous_previous_control": (
                                previous_applied_control
                            ),
                        }
                    )
                proposed_control, current_logits, proposed_active = actor(
                    history[-1],
                    step_reference_mean,
                    reference_variance,
                    step / max(horizon - 1, 1),
                    **actor_kwargs,
                )
                proposed_diffusion_action = getattr(
                    actor,
                    "last_diffusion_control",
                    proposed_control.abs()
                    / max(float(getattr(actor, "amplitude_limit", 1.0)), 1e-6),
                )
                if smooth_actor:
                    alpha = float(getattr(actor, "feedback_alpha", 1.0))
                    current_control = (
                        (1.0 - alpha) * current_control
                        + alpha * proposed_control
                    )
                    current_active = current_control.abs() / max(
                        float(actor.amplitude_limit), 1e-6
                    )
                    current_diffusion_action = (
                        (1.0 - alpha) * current_diffusion_action
                        + alpha * proposed_diffusion_action
                    )
                else:
                    current_control = proposed_control
                    current_active = proposed_active
                    current_diffusion_action = proposed_diffusion_action
                probabilities = torch.softmax(current_logits, dim=-1)
                current_entropy = -(
                    probabilities * torch.log(probabilities.clamp_min(1e-8))
                ).sum(dim=-1)
                previous_applied_control = prior_control
            if actor is None:
                current_control = states0.new_zeros(
                    batch, len(self.actuator_indices)
                )
                current_active = states0.new_zeros(
                    batch, len(self.actuator_indices)
                )
                current_entropy = states0.new_zeros(
                    batch, len(self.actuator_indices)
                )
                current_diffusion_action = states0.new_zeros(
                    batch, len(self.actuator_indices)
                )

            current = history[-1]
            feature = torch.cat(
                [
                    states,
                    current,
                    topology,
                    *(history[-1 - delay] for delay in self.delays),
                ],
                dim=1,
            )
            scaled_feature = (feature - self.feature_mean) / self.feature_scale
            diffusion_modulation = states0.new_zeros(
                batch, self.n_channels
            )
            if self.regime_adaptive:
                gate_window = torch.stack(
                    history[-self.gate_window_samples :], dim=1
                )
                gate_mean = gate_window.mean(dim=1)
                gate_scale = torch.sqrt(
                    gate_window.var(dim=1, unbiased=False).clamp_min(1e-10)
                )
                gate_line_length = torch.mean(
                    torch.abs(gate_window[:, 1:] - gate_window[:, :-1]),
                    dim=1,
                )
                gate_deviation = gate_window[:, -1] - gate_mean
                gate_log_scale = torch.log(gate_scale + 1e-6)
                if self.gate_feature_mode == "mean":
                    gate_feature = gate_mean
                elif self.gate_feature_mode == "mean_scale":
                    gate_feature = torch.cat(
                        [gate_mean, gate_log_scale], dim=1
                    )
                elif self.gate_feature_mode == "mean_scale_deviation":
                    gate_feature = torch.cat(
                        [gate_deviation, gate_mean, gate_log_scale], dim=1
                    )
                elif self.gate_feature_mode == "moments_line_length":
                    gate_feature = torch.cat(
                        [
                            gate_deviation,
                            gate_mean,
                            gate_log_scale,
                            torch.log(gate_line_length + 1e-6),
                        ],
                        dim=1,
                    )
                else:
                    raise ValueError(
                        "unknown regime gate feature mode: "
                        f"{self.gate_feature_mode}"
                    )
                gate_feature = (
                    gate_feature - self.gate_feature_mean
                ) / self.gate_feature_scale
                gate_logit = (
                    gate_feature @ self.gate_coef + self.gate_intercept
                )
                drift_gate = torch.sigmoid(
                    gate_logit - self.drift_gate_logit_offset
                )
                diffusion_gate = torch.sigmoid(
                    gate_logit - self.diffusion_gate_logit_offset
                )
                regime_probability = drift_gate
                regime_logit = gate_logit - self.drift_gate_logit_offset
                expert_means: list[Tensor] = []
                expert_innovations: list[Tensor] = []
                for prefix in ("ictal", "interictal"):
                    drift_coef = getattr(self, f"{prefix}_drift_coef")
                    drift_intercept = getattr(self, f"{prefix}_drift_intercept")
                    variance_coef = getattr(self, f"{prefix}_variance_coef")
                    variance_intercept = getattr(
                        self, f"{prefix}_variance_intercept"
                    )
                    variance_calibration = getattr(
                        self, f"{prefix}_variance_calibration"
                    )
                    log_variance_bounds = getattr(
                        self, f"{prefix}_log_variance_bounds"
                    )
                    correlation_root = getattr(
                        self, f"{prefix}_correlation_root"
                    )
                    expert_means.append(
                        scaled_feature @ drift_coef.T + drift_intercept
                    )
                    log_variance = (
                        scaled_feature @ variance_coef.T
                        + variance_intercept
                        + torch.log(variance_calibration)
                    )
                    log_variance = torch.maximum(
                        log_variance, log_variance_bounds[0]
                    )
                    log_variance = torch.minimum(
                        log_variance, log_variance_bounds[1]
                    )
                    conditional_std = torch.exp(0.5 * log_variance)
                    expert_innovations.append(
                        (noise[:, step] @ correlation_root.T) * conditional_std
                    )
                mean_increment = (
                    drift_gate[:, None] * expert_means[0]
                    + (1.0 - drift_gate[:, None]) * expert_means[1]
                )
                if self.regime_diffusion_mode == "shared_state_dependent":
                    shared_log_variance = (
                        scaled_feature @ self.shared_variance_coef.T
                        + self.shared_variance_intercept
                        + torch.log(self.shared_variance_calibration)
                    )
                    shared_log_variance = torch.maximum(
                        shared_log_variance,
                        self.shared_log_variance_bounds[0],
                    )
                    shared_log_variance = torch.minimum(
                        shared_log_variance,
                        self.shared_log_variance_bounds[1],
                    )
                    shared_std = torch.exp(0.5 * shared_log_variance)
                    innovation = (
                        noise[:, step] @ self.shared_correlation_root.T
                    ) * shared_std
                else:
                    innovation = (
                        diffusion_gate[:, None] * expert_innovations[0]
                        + (1.0 - diffusion_gate[:, None])
                        * expert_innovations[1]
                    )
                    if float(diffusion_control_strength) > 0:
                        if bool(independent_diffusion_control):
                            actuator_intensity = current_diffusion_action
                        else:
                            amplitude_scale = max(
                                float(getattr(actor, "amplitude_limit", 1.0)),
                                1e-6,
                            )
                            actuator_intensity = (
                                current_control.abs() / amplitude_scale
                            )
                        channel_intensity = (
                            actuator_intensity @ self.control_channel_map.abs()
                        )
                        diffusion_modulation = 1.0 - torch.exp(
                            -float(diffusion_control_strength)
                            * channel_intensity
                        )
                        mixed_channel_innovation = innovation @ self.components
                        interictal_channel_innovation = (
                            expert_innovations[1] @ self.components
                        )
                        innovation = (
                            mixed_channel_innovation
                            + diffusion_modulation
                            * (
                                interictal_channel_innovation
                                - mixed_channel_innovation
                            )
                        ) @ self.components.T
            else:
                regime_probability = states0.new_zeros(batch)
                regime_logit = states0.new_zeros(batch)
                mean_increment = scaled_feature @ self.drift_coef.T + self.drift_intercept
                log_variance = (
                    scaled_feature @ self.variance_coef.T
                    + self.variance_intercept
                    + torch.log(self.variance_calibration)
                )
                log_variance = torch.maximum(log_variance, self.log_variance_bounds[0])
                log_variance = torch.minimum(log_variance, self.log_variance_bounds[1])
                conditional_std = torch.exp(0.5 * log_variance)
                innovation = (noise[:, step] @ self.correlation_root.T) * conditional_std
            latent_control = current_control @ self.control_map
            channel_control = current_control @ self.control_channel_map
            applied_control_scale = self.dt * float(control_input_gain)
            following = (
                current
                + mean_increment
                + applied_control_scale * latent_control
                + float(diffusion_scale) * innovation
            )
            history.append(following)
            retained_history = self.maximum_delay + 1
            if self.regime_adaptive:
                retained_history = max(
                    retained_history, self.gate_window_samples
                )
            if len(history) > retained_history:
                history.pop(0)
            scaled_following = self.scaled_from_latent(following)
            if self.preserve_physical_control_residual:
                # Passive PCA is a model-reduction coordinate, not a physical
                # actuator constraint.  Retain the component of the selected-
                # electrode input that lies outside the passive PCA subspace
                # in the decoded physical state.  Its PCA projection already
                # enters ``following`` above; adding only the orthogonal
                # residual avoids double counting and preserves exact zero-
                # control parity.
                projected_channel_control = latent_control @ self.components
                scaled_following = scaled_following + applied_control_scale * (
                    channel_control - projected_channel_control
                )
            topology = self.topology_from_scaled(scaled_following)
            drive = torch.cat(
                [
                    following,
                    topology,
                    *(history[-1 - delay] for delay in self.delays),
                ],
                dim=1,
            )
            proposal = torch.tanh(drive @ self.w_in.T + states @ self.w_res.T)
            states = (1.0 - self.leak_rate) * states + self.leak_rate * proposal
            scaled_values.append(scaled_following)
            latent_values.append(following)
            controls.append(current_control)
            active_probabilities.append(current_active)
            action_entropies.append(current_entropy)
            regime_probabilities.append(regime_probability)
            regime_logits.append(regime_logit)
            diffusion_modulations.append(diffusion_modulation)
            diffusion_actions.append(current_diffusion_action)
        self.last_regime_probabilities = torch.stack(
            regime_probabilities, dim=1
        )
        self.last_regime_logits = torch.stack(regime_logits, dim=1)
        self.last_diffusion_modulation = torch.stack(
            diffusion_modulations, dim=1
        )
        self.last_diffusion_actions = torch.stack(diffusion_actions, dim=1)
        return (
            torch.stack(scaled_values, dim=1),
            torch.stack(latent_values, dim=1),
            torch.stack(controls, dim=1),
            torch.stack(active_probabilities, dim=1),
            torch.stack(action_entropies, dim=1),
        )


def build_context_bank(
    world: TorchGraphRCSDE,
    sequence: np.ndarray,
    *,
    context_samples: int,
    horizon_samples: int,
    positions: Iterable[int],
) -> dict[str, np.ndarray]:
    """Precompute reservoir/history states at fixed forecast boundaries."""

    values = np.asarray(sequence, dtype=np.float64)
    states: list[np.ndarray] = []
    histories: list[np.ndarray] = []
    topologies: list[np.ndarray] = []
    accepted: list[int] = []
    for position in positions:
        index = int(position)
        if index < int(context_samples) or index + int(horizon_samples) > len(values):
            continue
        state, history, topology = world.context_state(
            values[index - int(context_samples) : index]
        )
        states.append(state)
        histories.append(history)
        topologies.append(topology)
        accepted.append(index)
    if not states:
        raise ValueError("no valid context positions were supplied")
    return {
        "states": np.stack(states),
        "history": np.stack(histories),
        "topology": np.stack(topologies),
        "positions": np.asarray(accepted, dtype=np.int64),
    }


def multiscale_mmd(first: Tensor, second: Tensor) -> Tensor:
    """Biased multi-scale RBF MMD in standardized channel space."""

    dimension = max(int(first.shape[1]), 1)
    xx = torch.cdist(first, first).square() / dimension
    yy = torch.cdist(second, second).square() / dimension
    xy = torch.cdist(first, second).square() / dimension
    total = first.new_zeros(())
    for bandwidth in (0.5, 1.0, 2.0):
        scale = 2.0 * bandwidth * bandwidth
        total = total + torch.exp(-xx / scale).mean()
        total = total + torch.exp(-yy / scale).mean()
        total = total - 2.0 * torch.exp(-xy / scale).mean()
    return total / 3.0


def marginal_quantile_losses(
    samples: Tensor,
    reference_quantiles: Tensor,
    probabilities: Tensor,
    channel_weights: Tensor | None = None,
    channel_scales: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    quantiles = torch.quantile(samples, probabilities, dim=0)
    errors = (quantiles - reference_quantiles).abs()
    if channel_scales is not None:
        errors = errors / channel_scales[None, :].clamp_min(0.15)
    per_channel = errors.mean(dim=0)
    if channel_weights is None:
        weighted = per_channel.mean()
    else:
        weights = channel_weights / channel_weights.mean().clamp_min(1e-6)
        weighted = (per_channel * weights).mean()
    worst_count = max(1, int(math.ceil(0.25 * len(per_channel))))
    return weighted, torch.topk(per_channel, worst_count).values.mean()


def per_time_quantile_loss(
    sequence: Tensor,
    reference_quantiles: Tensor,
    probabilities: Tensor,
    channel_weights: Tensor,
    channel_scales: Tensor | None = None,
) -> Tensor:
    """Match the particle law at every time, not only after time pooling."""

    quantiles = torch.quantile(sequence, probabilities, dim=0)
    errors = (quantiles - reference_quantiles[:, None, :]).abs()
    if channel_scales is not None:
        errors = errors / channel_scales[None, None, :].clamp_min(0.15)
    per_channel = errors.mean(dim=(0, 1))
    weights = channel_weights / channel_weights.mean().clamp_min(1e-6)
    return (per_channel * weights).mean()


def lagged_correlation(sequence: Tensor, lag: int) -> Tensor:
    """Per-channel correlation over particles and time for one lag."""

    first = sequence[:, :-int(lag)]
    second = sequence[:, int(lag) :]
    first = first - first.mean(dim=(0, 1), keepdim=True)
    second = second - second.mean(dim=(0, 1), keepdim=True)
    covariance = (first * second).mean(dim=(0, 1))
    scale = torch.sqrt(
        first.square().mean(dim=(0, 1)).clamp_min(1e-6)
        * second.square().mean(dim=(0, 1)).clamp_min(1e-6)
    )
    return covariance / scale


class SquareWaveMeanFieldController:
    """Train a stochastic square-wave actor against a full-channel critic."""

    def __init__(
        self,
        world: TorchGraphRCSDE,
        reference_scaled: np.ndarray,
        reference_latent: np.ndarray,
        config: SquareWaveMFCConfig,
        channel_weights: np.ndarray | None = None,
        reference_trajectory_scaled: np.ndarray | None = None,
        reference_trajectory_latent: np.ndarray | None = None,
    ) -> None:
        config.validate()
        self.world = world
        self.config = config
        torch.manual_seed(int(config.random_seed))
        np.random.seed(int(config.random_seed))
        if config.control_mode == "smooth_feedback":
            self.actor = MeanFieldSmoothFeedbackActor(
                world.q,
                len(world.actuator_indices),
                config.actor_hidden_size,
                config.amplitude_limit,
                world.control_map,
                config.horizon_samples,
                config.smooth_spline_basis_count,
                config.smooth_residual_scale,
            ).to(world.device)
            self.actor.feedback_alpha = float(config.smooth_feedback_alpha)
        else:
            self.actor = MeanFieldSquareWaveActor(
                world.q,
                len(world.actuator_indices),
                config.actor_hidden_size,
                config.amplitude_limit,
                config.control_levels,
            ).to(world.device)
        self.critic = ConditionalTrajectoryCritic(
            world.n_channels,
            config.critic_hidden_size,
            config.temporal_window_samples,
        ).to(world.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=float(config.actor_learning_rate),
            betas=(0.5, 0.9),
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=float(config.critic_learning_rate),
            betas=(0.5, 0.9),
        )
        self.reference_scaled = torch.as_tensor(
            np.asarray(reference_scaled), dtype=world.dtype, device=world.device
        )
        reference_latent_tensor = torch.as_tensor(
            np.asarray(reference_latent), dtype=world.dtype, device=world.device
        )
        self.reference_latent_mean = reference_latent_tensor.mean(dim=0)
        self.reference_latent_variance = reference_latent_tensor.var(
            dim=0, unbiased=False
        ).clamp_min(1e-6)
        if reference_trajectory_scaled is None:
            scaled_template = np.repeat(
                np.asarray(reference_scaled, dtype=np.float64).mean(axis=0)[None],
                config.horizon_samples,
                axis=0,
            )
        else:
            scaled_template = np.asarray(
                reference_trajectory_scaled, dtype=np.float64
            )
        if reference_trajectory_latent is None:
            latent_template = np.repeat(
                np.asarray(reference_latent, dtype=np.float64).mean(axis=0)[None],
                config.horizon_samples,
                axis=0,
            )
        else:
            latent_template = np.asarray(
                reference_trajectory_latent, dtype=np.float64
            )
        if scaled_template.shape != (
            config.horizon_samples,
            world.n_channels,
        ):
            raise ValueError("reference_trajectory_scaled has an invalid shape")
        if latent_template.shape != (config.horizon_samples, world.q):
            raise ValueError("reference_trajectory_latent has an invalid shape")
        self.reference_trajectory_scaled = torch.as_tensor(
            scaled_template, dtype=world.dtype, device=world.device
        )
        self.reference_trajectory_latent = torch.as_tensor(
            latent_template, dtype=world.dtype, device=world.device
        )
        probabilities = self.reference_scaled.new_tensor(
            [
                0.01,
                0.025,
                0.05,
                0.10,
                0.20,
                0.30,
                0.40,
                0.50,
                0.60,
                0.70,
                0.80,
                0.90,
                0.95,
                0.975,
                0.99,
            ]
        )
        self.quantile_probabilities = probabilities
        self.reference_quantiles = torch.quantile(
            self.reference_scaled, probabilities, dim=0
        )
        self.reference_mean = self.reference_scaled.mean(dim=0)
        self.reference_variance = self.reference_scaled.var(
            dim=0, unbiased=False
        ).clamp_min(1e-6)
        self.reference_scale = torch.sqrt(self.reference_variance).clamp_min(0.15)
        increments = self.reference_scaled[1:] - self.reference_scaled[:-1]
        self.reference_increment_quantiles = torch.quantile(
            increments, probabilities, dim=0
        )
        reference_sequence = self.reference_scaled.unsqueeze(0)
        self.reference_autocorrelations = torch.stack(
            [lagged_correlation(reference_sequence, lag) for lag in (1, 2, 4)],
            dim=0,
        )
        if channel_weights is None:
            weights = np.ones(world.n_channels, dtype=np.float32)
        else:
            weights = np.asarray(channel_weights, dtype=np.float32)
            if weights.shape != (world.n_channels,):
                raise ValueError("channel_weights must have one entry per EEG channel")
            if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
                raise ValueError("channel_weights must be finite and positive")
        self.channel_weights = torch.as_tensor(
            weights, dtype=world.dtype, device=world.device
        )
        self.rng = np.random.default_rng(int(config.random_seed) + 1000)
        self.history: list[dict[str, float]] = []

    def _temperature(self, epoch: int) -> float:
        fraction = epoch / max(self.config.train_epochs - 1, 1)
        start = float(self.config.gumbel_temperature_start)
        end = float(self.config.gumbel_temperature_end)
        return start * (end / start) ** fraction

    def _sample_bank(self, bank: dict[str, np.ndarray], count: int) -> tuple[Tensor, Tensor, Tensor]:
        indices = self.rng.integers(0, len(bank["states"]), size=int(count))
        return (
            torch.as_tensor(
                bank["states"][indices], dtype=self.world.dtype, device=self.world.device
            ),
            torch.as_tensor(
                bank["history"][indices], dtype=self.world.dtype, device=self.world.device
            ),
            torch.as_tensor(
                bank["topology"][indices], dtype=self.world.dtype, device=self.world.device
            ),
        )

    def _sample_reference(self, count: int) -> Tensor:
        indices = self.rng.integers(0, len(self.reference_scaled), size=int(count))
        return self.reference_scaled[torch.as_tensor(indices, device=self.world.device)]

    def _rollout(
        self,
        bank: dict[str, np.ndarray],
        *,
        actor: bool,
        temperature: float,
        differentiable: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        states, history, topology = self._sample_bank(bank, self.config.batch_size)
        noise = torch.randn(
            self.config.batch_size,
            self.config.horizon_samples,
            self.world.q,
            dtype=self.world.dtype,
            device=self.world.device,
        )
        context = torch.enable_grad() if differentiable else torch.no_grad()
        with context:
            return self.world.rollout(
                states,
                history,
                topology,
                horizon_samples=self.config.horizon_samples,
                noise=noise,
                actor=self.actor if actor else None,
                reference_mean=self.reference_trajectory_latent,
                reference_variance=self.reference_latent_variance,
                pulse_width_samples=self.config.pulse_width_samples,
                temperature=temperature,
                stochastic_actor=True,
                diffusion_scale=self.config.diffusion_scale,
                control_input_gain=self.config.control_input_gain,
            )

    def _tail(self, trajectory_scaled: Tensor) -> Tensor:
        tail = trajectory_scaled[:, -self.config.loss_tail_samples :]
        return tail.reshape(-1, self.world.n_channels)

    def _conditional_samples(
        self, trajectory_scaled: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Stratified marginal values and chronological same-node patches."""

        tail = trajectory_scaled[:, -self.config.loss_tail_samples :]
        batch, length, channels = tail.shape
        marginal_count = int(self.config.critic_marginal_samples)
        particle_np = self.rng.integers(0, batch, size=marginal_count)
        time_np = self.rng.integers(0, length, size=marginal_count)
        node_np = np.arange(marginal_count, dtype=np.int64) % channels
        self.rng.shuffle(node_np)
        particle = torch.as_tensor(particle_np, device=self.world.device)
        time_index = torch.as_tensor(time_np, device=self.world.device)
        nodes = torch.as_tensor(node_np, device=self.world.device)
        fake_values = tail[particle, time_index, nodes].unsqueeze(1)
        real_time = torch.as_tensor(
            self.rng.integers(0, len(self.reference_scaled), size=marginal_count),
            device=self.world.device,
        )
        real_values = self.reference_scaled[real_time, nodes].unsqueeze(1)

        window = int(self.config.temporal_window_samples)
        temporal_count = int(self.config.critic_temporal_samples)
        particle_np = self.rng.integers(0, batch, size=temporal_count)
        start_np = self.rng.integers(0, length - window + 1, size=temporal_count)
        temporal_node_np = np.arange(temporal_count, dtype=np.int64) % channels
        self.rng.shuffle(temporal_node_np)
        particle = torch.as_tensor(particle_np, device=self.world.device)
        starts = torch.as_tensor(start_np, device=self.world.device)
        temporal_nodes = torch.as_tensor(temporal_node_np, device=self.world.device)
        offsets = torch.arange(window, device=self.world.device)
        fake_patches = tail[
            particle[:, None], starts[:, None] + offsets[None, :], temporal_nodes[:, None]
        ]
        reference_starts = torch.as_tensor(
            self.rng.integers(
                0, len(self.reference_scaled) - window + 1, size=temporal_count
            ),
            device=self.world.device,
        )
        real_patches = self.reference_scaled[
            reference_starts[:, None] + offsets[None, :], temporal_nodes[:, None]
        ]
        return (
            real_values,
            fake_values,
            nodes,
            real_patches,
            fake_patches,
            temporal_nodes,
        )

    def _gradient_penalty(
        self,
        real: Tensor,
        fake: Tensor,
        scorer,
    ) -> Tensor:
        count = min(len(real), len(fake))
        real = real[:count]
        fake = fake[:count]
        alpha = torch.rand(count, 1, dtype=real.dtype, device=real.device)
        mixed = alpha * real + (1.0 - alpha) * fake
        mixed.requires_grad_(True)
        score = scorer(mixed)
        gradient = torch.autograd.grad(
            outputs=score.sum(),
            inputs=mixed,
            create_graph=True,
            retain_graph=True,
        )[0]
        return (gradient.norm(2, dim=1) - 1.0).square().mean()

    def critic_step(self, bank: dict[str, np.ndarray], temperature: float) -> dict[str, float]:
        self.critic_optimizer.zero_grad(set_to_none=True)
        fake_trajectory, _, _, _, _ = self._rollout(
            bank, actor=True, temperature=temperature, differentiable=False
        )
        fake = self._tail(fake_trajectory).detach()
        if len(fake) > self.config.critic_marginal_samples:
            subset = torch.as_tensor(
                self.rng.choice(
                    len(fake), self.config.critic_marginal_samples, replace=False
                ),
                device=self.world.device,
            )
            fake = fake[subset]
        real = self._sample_reference(len(fake))
        (
            real_values,
            fake_values,
            nodes,
            real_patches,
            fake_patches,
            temporal_nodes,
        ) = self._conditional_samples(fake_trajectory.detach())
        state_wasserstein = (
            self.critic.state_score(real).mean()
            - self.critic.state_score(fake).mean()
        )
        marginal_weights = self.channel_weights[nodes]
        temporal_weights = self.channel_weights[temporal_nodes]

        def weighted_score_mean(scores: Tensor, weights: Tensor) -> Tensor:
            return (scores * weights).sum() / weights.sum().clamp_min(1e-6)

        marginal_wasserstein = (
            weighted_score_mean(
                self.critic.marginal_score(real_values, nodes), marginal_weights
            )
            - weighted_score_mean(
                self.critic.marginal_score(fake_values, nodes), marginal_weights
            )
        )
        temporal_wasserstein = (
            weighted_score_mean(
                self.critic.temporal_score(real_patches, temporal_nodes),
                temporal_weights,
            )
            - weighted_score_mean(
                self.critic.temporal_score(fake_patches, temporal_nodes),
                temporal_weights,
            )
        )
        state_penalty = self._gradient_penalty(
            real, fake, self.critic.state_score
        )
        marginal_penalty = self._gradient_penalty(
            real_values,
            fake_values,
            lambda values: self.critic.marginal_score(values, nodes),
        )
        temporal_penalty = self._gradient_penalty(
            real_patches,
            fake_patches,
            lambda values: self.critic.temporal_score(values, temporal_nodes),
        )
        wasserstein = (
            state_wasserstein
            + self.config.critic_conditional_weight * marginal_wasserstein
            + self.config.critic_temporal_weight * temporal_wasserstein
        )
        penalty = state_penalty + 0.5 * marginal_penalty + 0.5 * temporal_penalty
        loss = -wasserstein + self.config.gradient_penalty_weight * penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        return {
            "critic_loss": float(loss.detach().cpu()),
            "critic_wasserstein": float(wasserstein.detach().cpu()),
            "critic_state_wasserstein": float(state_wasserstein.detach().cpu()),
            "critic_marginal_wasserstein": float(marginal_wasserstein.detach().cpu()),
            "critic_temporal_wasserstein": float(temporal_wasserstein.detach().cpu()),
            "gradient_penalty": float(penalty.detach().cpu()),
        }

    def actor_step(self, bank: dict[str, np.ndarray], temperature: float) -> dict[str, float]:
        self.actor_optimizer.zero_grad(set_to_none=True)
        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        trajectory, latent, controls, active, action_entropy = self._rollout(
            bank, actor=True, temperature=temperature, differentiable=True
        )
        fake = self._tail(trajectory)
        real = self._sample_reference(len(fake))
        (
            _,
            fake_values,
            nodes,
            _,
            fake_patches,
            temporal_nodes,
        ) = self._conditional_samples(trajectory)
        adversarial = -self.critic.state_score(fake).mean()
        marginal_weights = self.channel_weights[nodes]
        temporal_weights = self.channel_weights[temporal_nodes]
        conditional_scores = self.critic.marginal_score(fake_values, nodes)
        temporal_scores = self.critic.temporal_score(fake_patches, temporal_nodes)
        conditional_adversarial = -(
            conditional_scores * marginal_weights
        ).sum() / marginal_weights.sum().clamp_min(1e-6)
        temporal_adversarial = -(
            temporal_scores * temporal_weights
        ).sum() / temporal_weights.sum().clamp_min(1e-6)
        quantile, worst_quantile = marginal_quantile_losses(
            fake,
            self.reference_quantiles,
            self.quantile_probabilities,
            self.channel_weights,
            self.reference_scale,
        )
        tail_sequence = trajectory[:, -self.config.loss_tail_samples :]
        per_time_quantile = per_time_quantile_loss(
            tail_sequence,
            self.reference_quantiles,
            self.quantile_probabilities,
            self.channel_weights,
            self.reference_scale,
        )
        terminal_length = min(8, self.config.loss_tail_samples)
        terminal_distribution = per_time_quantile_loss(
            tail_sequence[:, -terminal_length:],
            self.reference_quantiles,
            self.quantile_probabilities,
            self.channel_weights,
            self.reference_scale,
        )
        lower = torch.quantile(self.reference_scaled, 0.05, dim=0)
        upper = torch.quantile(self.reference_scaled, 0.95, dim=0)
        outside = (
            (
                F.relu(lower[None, None, :] - tail_sequence)
                / self.reference_scale[None, None, :]
            ).square()
            + (
                F.relu(tail_sequence - upper[None, None, :])
                / self.reference_scale[None, None, :]
            ).square()
        ).mean(dim=(0, 1))
        normalized_weights = self.channel_weights / self.channel_weights.mean()
        reference_band = (outside * normalized_weights).mean()
        increments = tail_sequence[:, 1:] - tail_sequence[:, :-1]
        increment_quantiles = torch.quantile(
            increments.reshape(-1, self.world.n_channels),
            self.quantile_probabilities,
            dim=0,
        )
        increment_per_channel = (
            increment_quantiles - self.reference_increment_quantiles
        ).abs().mean(dim=0)
        increment_quantile = (
            increment_per_channel * normalized_weights
        ).mean()
        autocorrelation = trajectory.new_zeros(())
        for row, lag in enumerate((1, 2, 4)):
            difference = (
                lagged_correlation(tail_sequence, lag)
                - self.reference_autocorrelations[row]
            ).abs()
            autocorrelation = autocorrelation + (
                difference * normalized_weights
            ).mean()
        autocorrelation = autocorrelation / 3.0
        path_mean = (
            (
                (tail_sequence.mean(dim=0) - self.reference_mean[None, :])
                / self.reference_scale[None, :]
            ).square()
            * normalized_weights[None, :]
        ).mean()
        ensemble_path = trajectory[:, 1:].mean(dim=0)
        reference_path = self.reference_trajectory_scaled
        time_weight = torch.linspace(
            0.25,
            1.0,
            self.config.horizon_samples,
            dtype=trajectory.dtype,
            device=trajectory.device,
        )
        trajectory_template = (
            (
                (ensemble_path - reference_path)
                / self.reference_scale[None, :]
            ).square()
            * normalized_weights[None, :]
            * time_weight[:, None]
        ).mean()
        trajectory_velocity = (
            (
                (
                    (ensemble_path[1:] - ensemble_path[:-1])
                    - (reference_path[1:] - reference_path[:-1])
                )
                / self.reference_scale[None, :]
            ).square()
            * normalized_weights[None, :]
        ).mean()
        controlled_current = trajectory[:, :-1]
        desired_direction = reference_path[None, :, :] - controlled_current
        latent_control = controls @ self.world.control_map
        decoded_control = (
            self.world.dt
            * float(self.config.control_input_gain)
            * (latent_control @ self.world.components)
        )
        weighted_control = decoded_control * torch.sqrt(
            normalized_weights[None, None, :]
        )
        weighted_desired = desired_direction * torch.sqrt(
            normalized_weights[None, None, :]
        )
        cosine = (weighted_control * weighted_desired).sum(dim=2) / (
            weighted_control.norm(dim=2).clamp_min(1e-6)
            * weighted_desired.norm(dim=2).clamp_min(1e-6)
        )
        active_steps = decoded_control.square().sum(dim=2) > 1e-10
        feedback_alignment = (
            (1.0 - cosine)[active_steps].mean()
            if torch.any(active_steps)
            else trajectory.new_zeros(())
        )
        current_energy = (
            (controlled_current - reference_path[None, :, :]).square()
            * normalized_weights[None, None, :]
        ).mean(dim=2)
        next_energy = (
            (trajectory[:, 1:] - reference_path[None, :, :]).square()
            * normalized_weights[None, None, :]
        ).mean(dim=2)
        lyapunov_decrease = F.relu(next_energy - current_energy).mean()
        entropy = action_entropy.mean()
        mean_loss = (fake.mean(dim=0) - self.reference_mean).square().mean()
        log_variance_loss = (
            torch.log(fake.var(dim=0, unbiased=False).clamp_min(1e-6))
            - torch.log(self.reference_variance)
        ).square().mean()
        moment = mean_loss + 0.5 * log_variance_loss
        # Explicit, scale-normalized moment matching is evaluated at every
        # time point.  This separates location recovery from dispersion
        # recovery and prevents broad, easy nodes from hiding a narrow SOZ
        # marginal in a pooled all-channel average.
        time_means = tail_sequence.mean(dim=0)
        time_variances = tail_sequence.var(dim=0, unbiased=False).clamp_min(1e-6)
        normalized_mean_error = (
            (time_means - self.reference_mean[None, :])
            / self.reference_scale[None, :]
        ).square()
        normalized_variance_error = (
            torch.log(time_variances)
            - torch.log(self.reference_variance[None, :])
        ).square()
        per_channel_mean = normalized_mean_error.mean(dim=0)
        per_channel_variance = normalized_variance_error.mean(dim=0)
        mean_matching = (per_channel_mean * normalized_weights).mean()
        initial_mean_error = (
            (
                trajectory[:, 0].mean(dim=0) - self.reference_mean
            )
            / self.reference_scale
        ).square()
        mean_non_degradation = (
            F.relu(per_channel_mean - initial_mean_error)
            * normalized_weights
        ).mean()
        variance_matching = (
            per_channel_variance * normalized_weights
        ).mean()
        per_channel_moment = per_channel_mean + 0.5 * per_channel_variance
        worst_count = min(4, self.world.n_channels)
        worst_moment = torch.topk(
            per_channel_moment * normalized_weights,
            worst_count,
        ).values.mean()
        mmd_count = min(len(fake), len(real), 512)
        mmd_fake = fake[:mmd_count]
        mmd_real = real[:mmd_count]
        mmd = multiscale_mmd(mmd_fake, mmd_real)
        energy = controls.square().mean()
        block_controls = controls[:, :: self.config.pulse_width_samples]
        switching = (
            (block_controls[:, 1:] - block_controls[:, :-1]).abs().mean()
            if block_controls.shape[1] > 1
            else controls.new_zeros(())
        )
        normalized_controls = controls / max(
            float(self.config.amplitude_limit), 1e-6
        )
        first_difference = normalized_controls[:, 1:] - normalized_controls[:, :-1]
        control_smoothness = (
            first_difference.square().mean()
            if first_difference.shape[1] > 0
            else controls.new_zeros(())
        )
        second_difference = (
            normalized_controls[:, 2:]
            - 2.0 * normalized_controls[:, 1:-1]
            + normalized_controls[:, :-2]
        )
        control_curvature = (
            second_difference.square().mean()
            if second_difference.shape[1] > 0
            else controls.new_zeros(())
        )
        # Penalize discordance across graph edges in the controlled terminal
        # mean.  This is a mean-field topology regularizer, not a state edit.
        terminal_mean = trajectory[:, -1].mean(dim=0)
        laplacian = torch.diag(self.world.adjacency.sum(dim=1)) - self.world.adjacency
        topology = (terminal_mean @ laplacian @ terminal_mean) / self.world.n_channels
        loss = (
            self.config.adversarial_weight * adversarial
            + self.config.conditional_adversarial_weight * conditional_adversarial
            + self.config.temporal_adversarial_weight * temporal_adversarial
            + self.config.marginal_quantile_weight * quantile
            + self.config.worst_channel_quantile_weight * worst_quantile
            + self.config.per_time_quantile_weight * per_time_quantile
            + self.config.terminal_distribution_weight * terminal_distribution
            + self.config.reference_band_weight * reference_band
            + self.config.increment_quantile_weight * increment_quantile
            + self.config.autocorrelation_weight * autocorrelation
            + self.config.path_mean_weight * path_mean
            + self.config.trajectory_template_weight * trajectory_template
            + self.config.trajectory_velocity_weight * trajectory_velocity
            + self.config.feedback_alignment_weight * feedback_alignment
            + self.config.lyapunov_decrease_weight * lyapunov_decrease
            + self.config.action_entropy_weight * entropy
            + self.config.moment_weight * moment
            + self.config.mean_matching_weight * mean_matching
            + self.config.mean_non_degradation_weight * mean_non_degradation
            + self.config.variance_matching_weight * variance_matching
            + self.config.worst_moment_weight * worst_moment
            + self.config.mmd_weight * mmd
            + self.config.energy_weight * energy
            + self.config.switching_weight * switching
            + self.config.control_smoothness_weight * control_smoothness
            + self.config.control_curvature_weight * control_curvature
            + self.config.topology_weight * topology
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)
        return {
            "actor_loss": float(loss.detach().cpu()),
            "adversarial_loss": float(adversarial.detach().cpu()),
            "conditional_adversarial_loss": float(
                conditional_adversarial.detach().cpu()
            ),
            "temporal_adversarial_loss": float(temporal_adversarial.detach().cpu()),
            "quantile_loss": float(quantile.detach().cpu()),
            "worst_channel_quantile_loss": float(worst_quantile.detach().cpu()),
            "per_time_quantile_loss": float(per_time_quantile.detach().cpu()),
            "terminal_distribution_loss": float(
                terminal_distribution.detach().cpu()
            ),
            "reference_band_loss": float(reference_band.detach().cpu()),
            "increment_quantile_loss": float(increment_quantile.detach().cpu()),
            "autocorrelation_loss": float(autocorrelation.detach().cpu()),
            "path_mean_loss": float(path_mean.detach().cpu()),
            "trajectory_template_loss": float(
                trajectory_template.detach().cpu()
            ),
            "trajectory_velocity_loss": float(
                trajectory_velocity.detach().cpu()
            ),
            "feedback_alignment_loss": float(feedback_alignment.detach().cpu()),
            "lyapunov_decrease_loss": float(lyapunov_decrease.detach().cpu()),
            "action_entropy_loss": float(entropy.detach().cpu()),
            "moment_loss": float(moment.detach().cpu()),
            "mean_matching_loss": float(mean_matching.detach().cpu()),
            "mean_non_degradation_loss": float(
                mean_non_degradation.detach().cpu()
            ),
            "variance_matching_loss": float(variance_matching.detach().cpu()),
            "worst_moment_loss": float(worst_moment.detach().cpu()),
            "mmd_loss": float(mmd.detach().cpu()),
            "control_energy": float(energy.detach().cpu()),
            "switching_loss": float(switching.detach().cpu()),
            "control_smoothness_loss": float(control_smoothness.detach().cpu()),
            "control_curvature_loss": float(control_curvature.detach().cpu()),
            "topology_loss": float(topology.detach().cpu()),
            "active_probability": float(active.mean().detach().cpu()),
            "terminal_latent_norm": float(
                latent[:, -1].norm(dim=1).mean().detach().cpu()
            ),
        }

    def fit(self, bank: dict[str, np.ndarray], *, verbose_every: int = 20) -> list[dict[str, float]]:
        self.actor.train()
        self.critic.train()
        for epoch in range(self.config.train_epochs):
            temperature = self._temperature(epoch)
            critic_rows = [
                self.critic_step(bank, temperature)
                for _ in range(self.config.critic_steps)
            ]
            actor_row = self.actor_step(bank, temperature)
            row = {
                "epoch": float(epoch + 1),
                "temperature": float(temperature),
                **{
                    key: float(np.mean([entry[key] for entry in critic_rows]))
                    for key in critic_rows[0]
                },
                **actor_row,
            }
            self.history.append(row)
            if verbose_every and (
                epoch == 0
                or (epoch + 1) % int(verbose_every) == 0
                or epoch + 1 == self.config.train_epochs
            ):
                print(
                    "epoch %d/%d actor=%.4f criticW=%.4f q=%.4f active=%.3f"
                    % (
                        epoch + 1,
                        self.config.train_epochs,
                        row["actor_loss"],
                        row["critic_wasserstein"],
                        row["quantile_loss"],
                        row["active_probability"],
                    ),
                    flush=True,
                )
        return self.history

    def evaluate(
        self,
        bank: dict[str, np.ndarray],
        *,
        rollouts_per_context: int,
        seed: int,
        stochastic_actor: bool = True,
    ) -> dict[str, np.ndarray]:
        self.actor.eval()
        self.critic.eval()
        repeats = int(rollouts_per_context)
        states = np.repeat(bank["states"], repeats, axis=0)
        history = np.repeat(bank["history"], repeats, axis=0)
        topology = np.repeat(bank["topology"], repeats, axis=0)
        states_t = torch.as_tensor(states, dtype=self.world.dtype, device=self.world.device)
        history_t = torch.as_tensor(history, dtype=self.world.dtype, device=self.world.device)
        topology_t = torch.as_tensor(topology, dtype=self.world.dtype, device=self.world.device)
        generator = torch.Generator(device=self.world.device).manual_seed(int(seed))
        noise = torch.randn(
            len(states),
            self.config.horizon_samples,
            self.world.q,
            dtype=self.world.dtype,
            device=self.world.device,
            generator=generator,
        )
        with torch.no_grad():
            controlled = self.world.rollout(
                states_t,
                history_t,
                topology_t,
                horizon_samples=self.config.horizon_samples,
                noise=noise,
                actor=self.actor,
                reference_mean=self.reference_trajectory_latent,
                reference_variance=self.reference_latent_variance,
                pulse_width_samples=self.config.pulse_width_samples,
                temperature=self.config.gumbel_temperature_end,
                stochastic_actor=bool(stochastic_actor),
                actor_generator=generator,
                diffusion_scale=self.config.diffusion_scale,
                control_input_gain=self.config.control_input_gain,
            )
            uncontrolled = self.world.rollout(
                states_t,
                history_t,
                topology_t,
                horizon_samples=self.config.horizon_samples,
                noise=noise,
                actor=None,
                reference_mean=self.reference_trajectory_latent,
                reference_variance=self.reference_latent_variance,
                pulse_width_samples=self.config.pulse_width_samples,
                temperature=self.config.gumbel_temperature_end,
                stochastic_actor=False,
                diffusion_scale=self.config.diffusion_scale,
                control_input_gain=self.config.control_input_gain,
            )
        return {
            "controlled_scaled": controlled[0].cpu().numpy().astype(np.float64),
            "controlled_latent": controlled[1].cpu().numpy().astype(np.float64),
            "controls": controlled[2].cpu().numpy().astype(np.float64),
            "active_probability": controlled[3].cpu().numpy().astype(np.float64),
            "action_entropy": controlled[4].cpu().numpy().astype(np.float64),
            "uncontrolled_scaled": uncontrolled[0].cpu().numpy().astype(np.float64),
            "uncontrolled_latent": uncontrolled[1].cpu().numpy().astype(np.float64),
            "positions": np.asarray(bank["positions"], dtype=np.int64),
            "rollouts_per_context": np.asarray([repeats], dtype=np.int64),
        }


__all__ = [
    "ConditionalTrajectoryCritic",
    "MeanFieldSmoothFeedbackActor",
    "MeanFieldSquareWaveActor",
    "SquareWaveMFCConfig",
    "SquareWaveMeanFieldController",
    "TorchGraphRCSDE",
    "WassersteinCritic",
    "build_context_bank",
    "marginal_quantile_losses",
    "multiscale_mmd",
    "per_time_quantile_loss",
]
