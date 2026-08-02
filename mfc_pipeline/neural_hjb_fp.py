"""Neural HJB--Fokker--Planck mean-field control for a fitted Graph-RC SDE.

The value network supplies the Hamiltonian gradient used by the bounded
feedback law.  The controlled RC-SDE particle ensemble is the forward
Monte-Carlo discretization of the Fokker--Planck equation.  A discrete
Bellman residual couples the backward value solve to the forward particle law.
No adversarial critic, state projection, or terminal replacement is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math

import numpy as np
from scipy.interpolate import BSpline
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class NeuralHJBFPConfig:
    horizon_samples: int = 128
    tail_samples: int = 24
    hidden_size: int = 64
    hidden_layers: int = 3
    context_basis_count: int = 12
    state_curvature_floor: float = 0.05
    distribution_curvature_floor: float = 0.50
    epochs: int = 140
    learning_rate: float = 8e-4
    context_learning_rate_scale: float = 10.0
    context_matrix_learning_rate_scale: float = 5.0
    train_repeats: int = 4
    amplitude_limit: float = 1.2
    feedback_alpha: float = 0.16
    control_input_gain: float = 4.0
    # Causal, past-only multiplicative calibration of the fitted conditional
    # diffusion.  The same value is used for the uncontrolled and controlled
    # arms; it never changes the learned state dependence or correlation.
    plant_diffusion_scale: float = 1.0
    preserve_physical_control_residual: bool = False
    # When positive, this is the actual one-sample coefficient multiplying
    # u @ B in the discrete RC-SDE.  The Hamiltonian still uses the associated
    # continuous-time gain ``control_step_scale / dt`` because the factor dt
    # cancels between the drift and running-control terms in the HJB
    # first-order condition.  Zero preserves the dt * control_input_gain plant
    # convention.
    control_step_scale: float = 0.0
    hamiltonian_control_cost: float = 2.0
    hamiltonian_rate_cost: float = 0.0
    hamiltonian_acceleration_cost: float = 0.0
    gramian_control_cost: bool = False
    control_cost_ridge: float = 0.05
    value_gradient_scale: float = 1.0
    # Mean-field LQ feedback separates the common (population-mean) command
    # from the centered particle command that contracts/expands covariance.
    # Unit values recover the original Hamiltonian law exactly.
    mean_feedback_scale: float = 1.0
    deviation_feedback_scale: float = 1.0
    # Apply the actuator saturation separately to the population-mean and
    # centered particle commands, then re-center and rescale the deviation
    # command inside the remaining actuator headroom.  This preserves the
    # exact zero-mean covariance channel after the nonlinear bound.
    separate_mean_deviation_saturation: bool = False
    # For seizure termination, the covariance value prior may contract excess
    # ictal dispersion but does not actively expand below-target directions.
    # The unchanged fitted plant diffusion remains responsible for stochastic
    # variability in both controlled and uncontrolled arms.
    contractive_distribution_feedback_only: bool = False
    trajectory_weight: float = 48.0
    # Relative weight assigned to the first controlled sample in the
    # full-horizon trajectory objective.  A value of 1.0 makes the 1-s
    # tracking loss uniform; values below one emphasize terminal recovery.
    trajectory_time_weight_start: float = 0.25
    tail_trajectory_weight: float = 24.0
    mean_weight: float = 28.0
    variance_weight: float = 12.0
    quantile_weight: float = 32.0
    worst_quantile_weight: float = 12.0
    terminal_quantile_weight: float = 48.0
    joint_density_weight: float = 32.0
    joint_projection_count: int = 32
    regime_transition_weight: float = 0.0
    bures_value_weight: float = 0.0
    bures_covariance_ridge: float = 0.05
    decoded_state_value_weight: float = 0.0
    decoded_distribution_value_weight: float = 0.0
    line_length_weight: float = 0.0
    diffusion_control_strength: float = 0.0
    independent_diffusion_control: bool = False
    diffusion_control_cost: float = 0.05
    diffusion_policy_initial_logit: float = -1.0
    # Positive values enable an explicit mean-field variance feedback term in
    # the trainable diffusion-policy head.  The nonnegative gain can strengthen
    # the action only when particle variance exceeds the interictal target.
    diffusion_variance_feedback_initial_gain: float = 0.0
    path_cvar_weight: float = 0.0
    tail_cvar_weight: float = 0.0
    tube_cvar_weight: float = 0.0
    cvar_tail_fraction: float = 0.25
    tube_radius_std: float = 2.0
    hjb_residual_weight: float = 0.25
    hjb_law_cost_scale: float = 1.0
    terminal_value_weight: float = 0.20
    energy_weight: float = 0.01
    smoothness_weight: float = 1.5
    curvature_weight: float = 5.0
    context_value_regularization_weight: float = 0.05
    context_matrix_regularization_weight: float = 0.02
    gradient_clip: float = 5.0
    seed: int = 20260811

    def validate(self) -> None:
        if self.horizon_samples < 8:
            raise ValueError("horizon_samples must be at least eight")
        if not 4 <= self.tail_samples <= self.horizon_samples:
            raise ValueError("tail_samples must lie inside the horizon")
        if self.hidden_size < 16 or self.hidden_layers < 2:
            raise ValueError("value-network architecture is too small")
        if self.context_basis_count < 4:
            raise ValueError("context_basis_count must be at least four")
        if self.state_curvature_floor < 0 or self.distribution_curvature_floor < 0:
            raise ValueError("value curvature floors must be non-negative")
        if self.epochs < 1 or self.learning_rate <= 0:
            raise ValueError("optimization settings must be positive")
        if self.context_learning_rate_scale <= 0:
            raise ValueError("context_learning_rate_scale must be positive")
        if self.context_matrix_learning_rate_scale <= 0:
            raise ValueError("context_matrix_learning_rate_scale must be positive")
        if self.train_repeats < 2 or self.train_repeats % 2:
            raise ValueError("train_repeats must be a positive even integer")
        if self.amplitude_limit <= 0 or self.hamiltonian_control_cost <= 0:
            raise ValueError("control bounds and Hamiltonian cost must be positive")
        if (
            self.hamiltonian_rate_cost < 0
            or self.hamiltonian_acceleration_cost < 0
        ):
            raise ValueError("Hamiltonian actuator-dynamics costs must be non-negative")
        if self.control_cost_ridge <= 0:
            raise ValueError("control_cost_ridge must be positive")
        if self.value_gradient_scale <= 0:
            raise ValueError("value_gradient_scale must be positive")
        if self.mean_feedback_scale <= 0 or self.deviation_feedback_scale <= 0:
            raise ValueError("mean/deviation feedback scales must be positive")
        if not 0.0 < self.trajectory_time_weight_start <= 1.0:
            raise ValueError(
                "trajectory_time_weight_start must lie in (0, 1]"
            )
        if not 0.0 < self.feedback_alpha <= 1.0:
            raise ValueError("feedback_alpha must lie in (0, 1]")
        if self.control_step_scale < 0:
            raise ValueError("control_step_scale must be non-negative")
        if self.plant_diffusion_scale <= 0:
            raise ValueError("plant_diffusion_scale must be positive")
        if not 0.0 < self.cvar_tail_fraction <= 1.0:
            raise ValueError("cvar_tail_fraction must lie in (0, 1]")
        if self.tube_radius_std <= 0:
            raise ValueError("tube_radius_std must be positive")
        if self.joint_density_weight < 0 or self.joint_projection_count < 4:
            raise ValueError("joint-density settings are invalid")
        if self.hjb_law_cost_scale < 0:
            raise ValueError("HJB law-cost scale must be non-negative")
        if self.regime_transition_weight < 0:
            raise ValueError("regime-transition weight must be non-negative")
        if self.bures_value_weight < 0 or self.bures_covariance_ridge <= 0:
            raise ValueError("Bures value-functional settings are invalid")
        if (
            self.decoded_state_value_weight < 0
            or self.decoded_distribution_value_weight < 0
        ):
            raise ValueError("decoded value-functional weights must be non-negative")
        if self.line_length_weight < 0:
            raise ValueError("line-length matching weight must be non-negative")
        if self.diffusion_control_strength < 0:
            raise ValueError("diffusion-control strength must be non-negative")
        if self.diffusion_control_cost < 0:
            raise ValueError("diffusion-control cost must be non-negative")
        if self.diffusion_variance_feedback_initial_gain < 0:
            raise ValueError(
                "diffusion-variance feedback gain must be non-negative"
            )


class MeanFieldValueNetwork(nn.Module):
    """Smooth value ansatz with explicit state and law-tracking components."""

    def __init__(
        self,
        latent_dim: int,
        hidden_size: int,
        hidden_layers: int,
        context_dim: int = 0,
        state_curvature_floor: float = 0.05,
        distribution_curvature_floor: float = 0.50,
        contractive_distribution_feedback_only: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.context_dim = int(context_dim)
        self.state_curvature_floor = float(state_curvature_floor)
        self.distribution_curvature_floor = float(distribution_curvature_floor)
        self.contractive_distribution_feedback_only = bool(
            contractive_distribution_feedback_only
        )
        if self.context_dim > 0:
            self.context_encoder = nn.Sequential(
                nn.Linear(self.context_dim, int(hidden_size)),
                nn.LayerNorm(int(hidden_size)),
                nn.SiLU(),
                nn.Linear(int(hidden_size), int(hidden_size)),
                nn.SiLU(),
            )
            context_embedding_dim = int(hidden_size)
        else:
            self.context_encoder = None
            context_embedding_dim = 0
        field_dim = 2 * self.latent_dim + 1 + context_embedding_dim
        layers: list[nn.Module] = [nn.Linear(field_dim, int(hidden_size)), nn.SiLU()]
        for _ in range(int(hidden_layers) - 1):
            layers.extend([nn.Linear(int(hidden_size), int(hidden_size)), nn.SiLU()])
        self.field_network = nn.Sequential(*layers)
        self.state_head = nn.Linear(int(hidden_size), self.latent_dim)
        self.distribution_head = nn.Linear(int(hidden_size), self.latent_dim)
        self.linear_head = nn.Linear(int(hidden_size), self.latent_dim)
        self.residual_network = nn.Sequential(
            nn.Linear(3 * self.latent_dim + int(hidden_size) + 1, int(hidden_size)),
            nn.SiLU(),
            nn.Linear(int(hidden_size), int(hidden_size)),
            nn.SiLU(),
            nn.Linear(int(hidden_size), 1),
        )
        # Start with a weak convex value landscape so the bounded Hamiltonian
        # law retains usable gradients instead of immediately saturating.
        nn.init.constant_(self.state_head.bias, -1.5)
        nn.init.constant_(self.distribution_head.bias, -2.5)
        nn.init.zeros_(self.linear_head.weight)
        nn.init.zeros_(self.linear_head.bias)
        nn.init.zeros_(self.residual_network[-1].weight)
        nn.init.zeros_(self.residual_network[-1].bias)

    @staticmethod
    def context_moments(values: Tensor, context_indices: Tensor) -> tuple[Tensor, Tensor]:
        count = int(context_indices.max().item()) + 1
        sums = values.new_zeros(count, values.shape[1])
        sums.index_add_(0, context_indices, values)
        sizes = torch.bincount(context_indices, minlength=count).to(values.dtype).clamp_min(1.0)
        means = sums / sizes[:, None]
        centered = values - means[context_indices]
        variance_sums = values.new_zeros(count, values.shape[1])
        variance_sums.index_add_(0, context_indices, centered.square())
        variances = (variance_sums / sizes[:, None]).clamp_min(1e-6)
        return means, variances

    def forward(
        self,
        particles: Tensor,
        context_indices: Tensor,
        reference_mean: Tensor,
        reference_variance: Tensor,
        time_fraction: float | Tensor,
        context_features: Tensor | None = None,
    ) -> Tensor:
        means, variances = self.context_moments(particles, context_indices)
        ref_var = reference_variance.clamp_min(1e-6)
        ref_scale = torch.sqrt(ref_var).clamp_min(0.10)
        mean_error = (means - reference_mean[None, :]) / ref_scale[None, :]
        log_variance_error = torch.log(variances) - torch.log(ref_var[None, :])
        if isinstance(time_fraction, Tensor):
            scalar_time = time_fraction.to(dtype=particles.dtype, device=particles.device)
        else:
            scalar_time = particles.new_tensor(float(time_fraction))
        field_parts = [
            scalar_time.expand(len(means), 1),
            mean_error,
            log_variance_error,
        ]
        if self.context_encoder is not None:
            if context_features is None or len(context_features) != len(means):
                raise ValueError("context features do not match the mean-field contexts")
            field_parts.append(self.context_encoder(context_features))
        field = torch.cat(field_parts, dim=1)
        embedding = self.field_network(field)
        state_weight = (
            F.softplus(self.state_head(embedding)) + self.state_curvature_floor
        )
        distribution_weight = (
            F.softplus(self.distribution_head(embedding))
            + self.distribution_curvature_floor
        )
        linear_coefficient = self.linear_head(embedding)
        centered = particles - means[context_indices]
        maximum_scale_ratio = (
            1.0 if self.contractive_distribution_feedback_only else 5.0
        )
        scale_ratio = torch.sqrt(ref_var[None, :] / variances).clamp(
            0.20, maximum_scale_ratio
        )
        distribution_target = (
            reference_mean[None, :] + scale_ratio[context_indices] * centered
        )
        state_error = (particles - reference_mean[None, :]) / ref_scale[None, :]
        distribution_error = (particles - distribution_target) / ref_scale[None, :]
        quadratic = 0.5 * (
            state_weight[context_indices] * state_error.square()
            + distribution_weight[context_indices] * distribution_error.square()
        ).sum(dim=1)
        # A time- and law-dependent linear value term represents the
        # feed-forward component of the HJB solution while control remains
        # exactly the Hamiltonian value gradient.
        linear_value = (
            linear_coefficient[context_indices] * state_error
        ).sum(dim=1)
        residual_features = torch.cat(
            [
                state_error,
                distribution_error,
                centered / torch.sqrt(variances[context_indices]).clamp_min(0.10),
                embedding[context_indices],
                scalar_time.expand(len(particles), 1),
            ],
            dim=1,
        )
        residual = 0.05 * self.residual_network(residual_features).squeeze(1)
        return quadratic + linear_value + residual


def bounded_mean_deviation_feedback(
    common_scaled_direction: Tensor,
    deviation_scaled_direction: Tensor,
    context_indices: Tensor,
    amplitude_limit: float,
) -> Tensor:
    """Return bounded feedback with an exactly centered deviation channel.

    Elementwise saturation does not commute with empirical centering.  The
    common and deviation commands are therefore saturated separately; the
    latter is re-centered and uniformly rescaled per context/actuator to fit
    the exact box-constraint headroom without changing its zero sum.
    """

    if common_scaled_direction.shape != deviation_scaled_direction.shape:
        raise ValueError("mean and deviation directions must share a shape")
    if context_indices.shape != (len(common_scaled_direction),):
        raise ValueError("context_indices must identify every particle")
    if len(common_scaled_direction) == 0 or float(amplitude_limit) <= 0:
        raise ValueError("feedback batch and amplitude limit must be positive")
    context_count = int(context_indices.max().item()) + 1
    context_sizes = torch.bincount(
        context_indices, minlength=context_count
    ).to(common_scaled_direction.dtype)
    if bool(torch.any(context_sizes == 0)):
        raise ValueError("context_indices must be contiguous from zero")
    common_control = -float(amplitude_limit) * torch.tanh(
        common_scaled_direction
    )
    raw_deviation_control = -float(amplitude_limit) * torch.tanh(
        deviation_scaled_direction
    )
    deviation_sums = raw_deviation_control.new_zeros(
        context_count, raw_deviation_control.shape[1]
    )
    deviation_sums.index_add_(0, context_indices, raw_deviation_control)
    deviation_means = deviation_sums / context_sizes[:, None]
    centered_control = raw_deviation_control - deviation_means[context_indices]
    maximum_centered = torch.stack(
        [
            centered_control[context_indices == index].abs().amax(dim=0)
            for index in range(context_count)
        ],
        dim=0,
    )
    common_sums = common_control.new_zeros(
        context_count, common_control.shape[1]
    )
    common_sums.index_add_(0, context_indices, common_control)
    common_by_context = common_sums / context_sizes[:, None]
    headroom = (
        float(amplitude_limit) - common_by_context.abs()
    ).clamp_min(0.0)
    deviation_scale = torch.minimum(
        torch.ones_like(maximum_centered),
        headroom / maximum_centered.clamp_min(1e-8),
    )
    return common_control + centered_control * deviation_scale[context_indices]


class NeuralHJBActor(nn.Module):
    """Bounded Hamiltonian feedback derived from the neural value gradient."""

    continuous_control = True
    uses_control_history = True

    def __init__(
        self,
        value_network: MeanFieldValueNetwork,
        context_indices: Tensor,
        control_map: Tensor,
        context_features: Tensor | None,
        *,
        horizon_samples: int,
        context_basis_count: int,
        amplitude_limit: float,
        feedback_alpha: float,
        control_input_gain: float,
        hamiltonian_control_cost: float,
        hamiltonian_rate_cost: float,
        hamiltonian_acceleration_cost: float,
        gramian_control_cost: bool,
        control_cost_ridge: float,
        value_gradient_scale: float,
        mean_feedback_scale: float,
        deviation_feedback_scale: float,
        separate_mean_deviation_saturation: bool,
        contractive_distribution_feedback_only: bool,
        reference_covariance: Tensor,
        bures_value_weight: float,
        bures_covariance_ridge: float,
        decoder_components: Tensor,
        decoder_mean: Tensor,
        reference_scaled: Tensor,
        channel_weights: Tensor,
        decoded_state_value_weight: float,
        decoded_distribution_value_weight: float,
        independent_diffusion_control: bool,
        diffusion_policy_initial_logit: float,
        diffusion_variance_feedback_initial_gain: float,
    ) -> None:
        super().__init__()
        self.value_network = value_network
        self.register_buffer("context_indices", context_indices.to(torch.long).contiguous())
        self.register_buffer("control_map", control_map)
        self.register_buffer("context_features", context_features)
        self.horizon_samples = int(horizon_samples)
        self.register_buffer(
            "context_spline_basis",
            cubic_spline_basis(
                self.horizon_samples,
                int(context_basis_count),
                dtype=control_map.dtype,
                device=control_map.device,
            ),
        )
        context_count = int(context_indices.max().item()) + 1
        self.context_value_coefficients = nn.Parameter(
            torch.zeros(
                context_count,
                int(context_basis_count),
                value_network.latent_dim,
                dtype=control_map.dtype,
                device=control_map.device,
            )
        )
        self.context_state_matrix_coefficients = nn.Parameter(
            torch.zeros(
                context_count,
                int(context_basis_count),
                value_network.latent_dim,
                value_network.latent_dim,
                dtype=control_map.dtype,
                device=control_map.device,
            )
        )
        self.context_distribution_matrix_coefficients = nn.Parameter(
            torch.zeros_like(self.context_state_matrix_coefficients)
        )
        self.context_diffusion_coefficients = nn.Parameter(
            torch.zeros(
                context_count,
                int(context_basis_count),
                int(control_map.shape[0]),
                dtype=control_map.dtype,
                device=control_map.device,
            )
        )
        self.diffusion_direction_gain_logits = nn.Parameter(
            torch.zeros(
                int(control_map.shape[0]),
                dtype=control_map.dtype,
                device=control_map.device,
            )
        )
        self.independent_diffusion_control = bool(
            independent_diffusion_control
        )
        self.diffusion_policy_initial_logit = float(
            diffusion_policy_initial_logit
        )
        self.diffusion_variance_feedback_initial_gain = float(
            diffusion_variance_feedback_initial_gain
        )
        if self.diffusion_variance_feedback_initial_gain > 0:
            initial_raw_gain = math.log(
                math.expm1(self.diffusion_variance_feedback_initial_gain)
            )
        else:
            initial_raw_gain = -20.0
        self.diffusion_variance_feedback_gain_logits = nn.Parameter(
            torch.full(
                (int(control_map.shape[0]),),
                float(initial_raw_gain),
                dtype=control_map.dtype,
                device=control_map.device,
            )
        )
        self.amplitude_limit = float(amplitude_limit)
        self.feedback_alpha = float(feedback_alpha)
        self.control_input_gain = float(control_input_gain)
        self.hamiltonian_control_cost = float(hamiltonian_control_cost)
        self.value_gradient_scale = float(value_gradient_scale)
        self.mean_feedback_scale = float(mean_feedback_scale)
        self.deviation_feedback_scale = float(deviation_feedback_scale)
        self.separate_mean_deviation_saturation = bool(
            separate_mean_deviation_saturation
        )
        self.contractive_distribution_feedback_only = bool(
            contractive_distribution_feedback_only
        )
        self.bures_value_weight = float(bures_value_weight)
        covariance = reference_covariance.to(
            dtype=control_map.dtype, device=control_map.device
        )
        identity_latent = torch.eye(
            covariance.shape[0],
            dtype=control_map.dtype,
            device=control_map.device,
        )
        ridge = float(bures_covariance_ridge) * torch.diagonal(
            covariance
        ).mean().clamp_min(1e-6)
        covariance = 0.5 * (covariance + covariance.T) + ridge * identity_latent
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        inverse_root = (
            eigenvectors
            @ torch.diag(eigenvalues.clamp_min(1e-6).rsqrt())
            @ eigenvectors.T
        )
        self.register_buffer("reference_covariance_inverse_root", inverse_root)
        self.register_buffer("decoder_components", decoder_components)
        self.register_buffer("decoder_mean", decoder_mean)
        self.register_buffer(
            "reference_channel_variance",
            reference_scaled.var(dim=0, unbiased=False).clamp_min(1e-4),
        )
        self.register_buffer(
            "normalized_channel_weights",
            channel_weights / channel_weights.mean().clamp_min(1e-6),
        )
        self.decoded_state_value_weight = float(decoded_state_value_weight)
        self.decoded_distribution_value_weight = float(
            decoded_distribution_value_weight
        )
        actuator_count = int(control_map.shape[0])
        identity = torch.eye(
            actuator_count, dtype=control_map.dtype, device=control_map.device
        )
        if bool(gramian_control_cost):
            base_cost = (
                self.control_input_gain**2
                * (control_map @ control_map.T)
                + float(control_cost_ridge) * identity
            )
        else:
            base_cost = identity
        control_cost_matrix = self.hamiltonian_control_cost * base_cost
        self.register_buffer("control_cost_matrix", control_cost_matrix)
        self.register_buffer(
            "control_cost_inverse", torch.linalg.inv(control_cost_matrix)
        )
        self.hamiltonian_rate_cost = float(hamiltonian_rate_cost)
        self.hamiltonian_acceleration_cost = float(
            hamiltonian_acceleration_cost
        )
        proximal_cost_matrix = control_cost_matrix + (
            self.hamiltonian_rate_cost
            + self.hamiltonian_acceleration_cost
        ) * identity
        self.register_buffer(
            "hamiltonian_proximal_inverse",
            torch.linalg.inv(proximal_cost_matrix),
        )

    def value(
        self,
        particles: Tensor,
        reference_mean: Tensor,
        reference_variance: Tensor,
        time_fraction: float,
    ) -> Tensor:
        base_value = self.value_network(
            particles,
            self.context_indices,
            reference_mean,
            reference_variance,
            time_fraction,
            self.context_features,
        )
        step = int(
            np.clip(
                np.rint(float(time_fraction) * max(self.horizon_samples - 1, 1)),
                0,
                self.horizon_samples - 1,
            )
        )
        linear_coefficient = torch.einsum(
            "k,ckq->cq",
            self.context_spline_basis[step],
            self.context_value_coefficients,
        )
        reference_scale = torch.sqrt(reference_variance.clamp_min(1e-6)).clamp_min(
            0.10
        )
        normalized_state = (
            particles - reference_mean[None, :]
        ) / reference_scale[None, :]
        context_linear_value = (
            linear_coefficient[self.context_indices] * normalized_state
        ).sum(dim=1)
        raw_state_matrix = torch.einsum(
            "k,ckqr->cqr",
            self.context_spline_basis[step],
            self.context_state_matrix_coefficients,
        )
        state_matrix = self._positive_semidefinite_matrix(raw_state_matrix)
        state_quadratic_value = 0.5 * torch.einsum(
            "bq,bqr,br->b",
            normalized_state,
            state_matrix[self.context_indices],
            normalized_state,
        )
        means, variances = self.value_network.context_moments(
            particles, self.context_indices
        )
        centered = particles - means[self.context_indices]
        maximum_scale_ratio = (
            1.0 if self.contractive_distribution_feedback_only else 5.0
        )
        scale_ratio = torch.sqrt(
            reference_variance.clamp_min(1e-6)[None, :] / variances
        ).clamp(0.20, maximum_scale_ratio)
        distribution_target = (
            reference_mean[None, :] + scale_ratio[self.context_indices] * centered
        )
        distribution_error = (
            particles - distribution_target
        ) / reference_scale[None, :]
        raw_distribution_matrix = torch.einsum(
            "k,ckqr->cqr",
            self.context_spline_basis[step],
            self.context_distribution_matrix_coefficients,
        )
        distribution_matrix = self._positive_semidefinite_matrix(
            raw_distribution_matrix
        )
        distribution_quadratic_value = 0.5 * torch.einsum(
            "bq,bqr,br->b",
            distribution_error,
            distribution_matrix[self.context_indices],
            distribution_error,
        )
        value = (
            base_value
            + context_linear_value
            + state_quadratic_value
            + distribution_quadratic_value
        )
        if self.bures_value_weight > 0:
            whitened = (
                particles - reference_mean[None]
            ) @ self.reference_covariance_inverse_root
            context_count = int(self.context_indices.max().item()) + 1
            law_values: list[Tensor] = []
            identity = torch.eye(
                whitened.shape[1],
                dtype=whitened.dtype,
                device=whitened.device,
            )
            for context in range(context_count):
                group = whitened[self.context_indices == context]
                group_mean = group.mean(dim=0)
                centered_group = group - group_mean[None]
                covariance = (
                    centered_group.T @ centered_group
                ) / max(int(len(group)), 1)
                covariance = 0.5 * (covariance + covariance.T) + 1e-5 * identity
                eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(1e-6)
                covariance_bures = (
                    torch.sqrt(eigenvalues) - 1.0
                ).square().sum()
                law_values.append(
                    0.5 * group_mean.square().sum()
                    + 0.5 * covariance_bures
                )
            law_value = torch.stack(law_values)
            # Each representative agent carries the same mean-field value
            # functional. Summing per-agent values then yields the Lions
            # derivative scale rather than an artificial 1/N attenuation.
            value = value + self.bures_value_weight * law_value[
                self.context_indices
            ]
        if (
            self.decoded_state_value_weight > 0
            or self.decoded_distribution_value_weight > 0
        ):
            decoded = particles @ self.decoder_components + self.decoder_mean
            decoded_reference = (
                reference_mean @ self.decoder_components + self.decoder_mean
            )
            channel_scale = torch.sqrt(
                self.reference_channel_variance
            ).clamp_min(0.05)
            decoded_means, decoded_variances = (
                self.value_network.context_moments(
                    decoded, self.context_indices
                )
            )
            state_error = (
                decoded - decoded_reference[None]
            ) / channel_scale[None]
            decoded_centered = decoded - decoded_means[self.context_indices]
            scale_ratio = torch.sqrt(
                self.reference_channel_variance[None]
                / decoded_variances
            ).clamp(0.20, maximum_scale_ratio)
            distribution_target = (
                decoded_reference[None]
                + scale_ratio[self.context_indices] * decoded_centered
            )
            distribution_error = (
                decoded - distribution_target
            ) / channel_scale[None]
            channel_state_value = 0.5 * (
                self.normalized_channel_weights[None]
                * state_error.square()
            ).sum(dim=1)
            channel_distribution_value = 0.5 * (
                self.normalized_channel_weights[None]
                * distribution_error.square()
            ).sum(dim=1)
            value = (
                value
                + self.decoded_state_value_weight * channel_state_value
                + self.decoded_distribution_value_weight
                * channel_distribution_value
            )
        return value

    @staticmethod
    def _positive_semidefinite_matrix(raw_matrix: Tensor) -> Tensor:
        """Map unconstrained coefficients to a smooth positive-semidefinite value Hessian.

        The triangular factor preserves learnable cross-state couplings, while
        the softplus diagonal prevents the zero-gradient initialization of a
        plain L L^T parameterization.  Convexity makes the corresponding
        Hamiltonian feedback locally contractive instead of allowing an
        indefinite value surface to amplify particle dispersion.
        """

        lower = torch.tril(raw_matrix, diagonal=-1)
        diagonal = F.softplus(torch.diagonal(raw_matrix, dim1=-2, dim2=-1) - 2.5)
        factor = lower + torch.diag_embed(diagonal)
        return factor @ factor.transpose(-1, -2)

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
        previous_control: Tensor | None = None,
        previous_previous_control: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del temperature, stochastic, generator
        with torch.enable_grad():
            if particles.requires_grad:
                differentiable_particles = particles
            else:
                differentiable_particles = particles.detach().requires_grad_(True)
            value = self.value(
                differentiable_particles,
                reference_mean,
                reference_variance,
                time_fraction,
            )
            value_gradient = torch.autograd.grad(
                value.sum(),
                differentiable_particles,
                create_graph=self.training,
            )[0]
            hamiltonian_direction = (
                self.value_gradient_scale
                * self.control_input_gain
                * (value_gradient @ self.control_map.T)
            )
            if previous_control is not None:
                hamiltonian_direction = (
                    hamiltonian_direction
                    - self.hamiltonian_rate_cost * previous_control
                )
                if previous_previous_control is None:
                    previous_previous_control = torch.zeros_like(
                        previous_control
                    )
                hamiltonian_direction = (
                    hamiltonian_direction
                    - self.hamiltonian_acceleration_cost
                    * (2.0 * previous_control - previous_previous_control)
                )
            # The common command steers the population mean; the exactly
            # centered command steers covariance without introducing an
            # additional population-mean input.  Distinct positive penalties
            # for these two orthogonal mean-field subspaces give the scales
            # below (equivalently, a block-diagonal R in mean/deviation space).
            context_count = int(self.context_indices.max().item()) + 1
            direction_sums = hamiltonian_direction.new_zeros(
                context_count, hamiltonian_direction.shape[1]
            )
            direction_sums.index_add_(
                0, self.context_indices, hamiltonian_direction
            )
            context_sizes = torch.bincount(
                self.context_indices, minlength=context_count
            ).to(hamiltonian_direction.dtype)
            direction_means = direction_sums / context_sizes[:, None]
            common_direction = direction_means[self.context_indices]
            centered_direction = hamiltonian_direction - common_direction
            if self.separate_mean_deviation_saturation:
                common_scaled_direction = (
                    self.mean_feedback_scale
                    * (common_direction @ self.hamiltonian_proximal_inverse)
                ) / self.amplitude_limit
                deviation_scaled_direction = (
                    self.deviation_feedback_scale
                    * (centered_direction @ self.hamiltonian_proximal_inverse)
                ) / self.amplitude_limit
                control = bounded_mean_deviation_feedback(
                    common_scaled_direction,
                    deviation_scaled_direction,
                    self.context_indices,
                    self.amplitude_limit,
                )
                scaled_direction = (
                    common_scaled_direction + deviation_scaled_direction
                )
            else:
                hamiltonian_direction = (
                    self.mean_feedback_scale * common_direction
                    + self.deviation_feedback_scale * centered_direction
                )
                scaled_direction = (
                    hamiltonian_direction @ self.hamiltonian_proximal_inverse
                ) / self.amplitude_limit
                control = -self.amplitude_limit * torch.tanh(scaled_direction)
            if self.independent_diffusion_control:
                step = int(
                    np.clip(
                        np.rint(
                            float(time_fraction)
                            * max(self.horizon_samples - 1, 1)
                        ),
                        0,
                        self.horizon_samples - 1,
                    )
                )
                diffusion_context_logit = torch.einsum(
                    "k,cka->ca",
                    self.context_spline_basis[step],
                    self.context_diffusion_coefficients,
                )
                direction_gain = F.softplus(
                    self.diffusion_direction_gain_logits
                )
                diffusion_logit = (
                    self.diffusion_policy_initial_logit
                    + diffusion_context_logit[self.context_indices]
                    + direction_gain[None]
                    * torch.log1p(scaled_direction.abs())
                )
                if self.diffusion_variance_feedback_initial_gain > 0:
                    _, current_variance = self.value_network.context_moments(
                        differentiable_particles, self.context_indices
                    )
                    log_variance_excess = torch.relu(
                        torch.log(current_variance.clamp_min(1e-6))
                        - torch.log(reference_variance[None].clamp_min(1e-6))
                    )
                    actuator_variance_weights = self.control_map.abs()
                    actuator_variance_weights = actuator_variance_weights / (
                        actuator_variance_weights.sum(dim=1, keepdim=True)
                        .clamp_min(1e-6)
                    )
                    variance_excess_by_actuator = (
                        log_variance_excess @ actuator_variance_weights.T
                    )
                    variance_feedback_gain = F.softplus(
                        self.diffusion_variance_feedback_gain_logits
                    )
                    diffusion_logit = diffusion_logit + (
                        variance_feedback_gain[None]
                        * variance_excess_by_actuator[self.context_indices]
                    )
                self.last_diffusion_control = torch.sigmoid(
                    diffusion_logit
                )
            else:
                self.last_diffusion_control = (
                    control.abs() / self.amplitude_limit
                )
        if not self.training:
            control = control.detach()
        pseudo_logits = control.unsqueeze(-1)
        active = control.abs() / self.amplitude_limit
        return control, pseudo_logits, active


def antithetic_noise(
    context_count: int,
    repeats: int,
    horizon: int,
    latent_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> Tensor:
    if repeats % 2:
        raise ValueError("repeats must be even")
    generator = torch.Generator(device=device).manual_seed(int(seed))
    half = torch.randn(
        int(context_count),
        int(repeats) // 2,
        int(horizon),
        int(latent_dim),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    return torch.cat([half, -half], dim=1).reshape(
        int(context_count) * int(repeats), int(horizon), int(latent_dim)
    )


def cubic_spline_basis(
    horizon: int,
    basis_count: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
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
        basis[:, index] = BSpline(knots, coefficient, degree)(times)
    basis = np.nan_to_num(basis, nan=0.0)
    basis /= np.maximum(basis.sum(axis=1, keepdims=True), 1e-12)
    return torch.as_tensor(basis, dtype=dtype, device=device)


def repeat_bank(
    bank: dict[str, np.ndarray], repeats: int, *, dtype: torch.dtype, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    context_count = len(bank["states"])
    states = torch.as_tensor(
        np.repeat(bank["states"], int(repeats), axis=0), dtype=dtype, device=device
    )
    history = torch.as_tensor(
        np.repeat(bank["history"], int(repeats), axis=0), dtype=dtype, device=device
    )
    topology = torch.as_tensor(
        np.repeat(bank["topology"], int(repeats), axis=0), dtype=dtype, device=device
    )
    indices = torch.arange(context_count, device=device).repeat_interleave(int(repeats))
    return states, history, topology, indices


def bank_context_features(bank: dict[str, np.ndarray]) -> np.ndarray:
    """Return the fixed augmented RC Markov context for each forecast boundary."""

    states = np.asarray(bank["states"], dtype=np.float64)
    history = np.asarray(bank["history"], dtype=np.float64).reshape(len(states), -1)
    topology = np.asarray(bank["topology"], dtype=np.float64)
    return np.concatenate([states, history, topology], axis=1)


def _validate_actor_world_for_plant(world, actor_world) -> None:
    """Ensure a paired rollout changes only the plant-side control transport.

    ``actor_world`` supplies the Hamiltonian control map used to synthesize the
    feedback law, whereas ``world`` supplies the map actually applied by the
    RC--SDE plant.  A graph-transport ablation is interpretable only when both
    wrappers refer to the identical fitted Part-II model and differ solely in
    their explicit control heat-kernel map.
    """

    if actor_world is world:
        return
    if world.model is not actor_world.model:
        raise ValueError(
            "actor and plant worlds must wrap the identical fitted Part-II model"
        )
    scalar_fields = (
        "q",
        "n_channels",
        "reservoir_size",
        "maximum_delay",
        "delays",
        "leak_rate",
        "dtype",
        "device",
    )
    mismatched = [
        name
        for name in scalar_fields
        if getattr(world, name) != getattr(actor_world, name)
    ]
    if not np.isclose(float(world.dt), float(actor_world.dt), rtol=0.0, atol=0.0):
        mismatched.append("dt")
    if not np.array_equal(world.actuator_indices, actor_world.actuator_indices):
        mismatched.append("actuator_indices")
    for name in ("components", "pca_mean", "adjacency"):
        if not torch.equal(getattr(world, name), getattr(actor_world, name)):
            mismatched.append(name)
    if mismatched:
        raise ValueError(
            "actor and plant worlds differ outside control graph transport: "
            + ", ".join(mismatched)
        )


def make_actor(
    value_network: MeanFieldValueNetwork,
    context_indices: Tensor,
    context_features: Tensor | None,
    world,
    config: NeuralHJBFPConfig,
    reference_latent_covariance: Tensor,
    reference_scaled: Tensor,
    channel_weights: Tensor,
) -> NeuralHJBActor:
    # The plant applies ``dt * B_c u`` (or the explicitly supplied discrete
    # step coefficient), whereas the continuous-time Hamiltonian minimizes
    # ``<grad V, B_c u> + 0.5 u^T R u``.  Passing the one-step coefficient here
    # would introduce an extra factor dt and make the synthesized feedback
    # artificially weak.  Keep the plant scale and Hamiltonian scale distinct.
    hamiltonian_control_gain = (
        float(config.control_step_scale) / float(world.dt)
        if config.control_step_scale > 0
        else float(config.control_input_gain)
    )
    return NeuralHJBActor(
        value_network,
        context_indices,
        world.control_map,
        context_features,
        horizon_samples=config.horizon_samples,
        context_basis_count=config.context_basis_count,
        amplitude_limit=config.amplitude_limit,
        feedback_alpha=config.feedback_alpha,
        control_input_gain=hamiltonian_control_gain,
        hamiltonian_control_cost=config.hamiltonian_control_cost,
        hamiltonian_rate_cost=config.hamiltonian_rate_cost,
        hamiltonian_acceleration_cost=(
            config.hamiltonian_acceleration_cost
        ),
        gramian_control_cost=config.gramian_control_cost,
        control_cost_ridge=config.control_cost_ridge,
        value_gradient_scale=config.value_gradient_scale,
        mean_feedback_scale=config.mean_feedback_scale,
        deviation_feedback_scale=config.deviation_feedback_scale,
        separate_mean_deviation_saturation=(
            config.separate_mean_deviation_saturation
        ),
        contractive_distribution_feedback_only=(
            config.contractive_distribution_feedback_only
        ),
        reference_covariance=reference_latent_covariance,
        bures_value_weight=config.bures_value_weight,
        bures_covariance_ridge=config.bures_covariance_ridge,
        decoder_components=world.components,
        decoder_mean=world.pca_mean,
        reference_scaled=reference_scaled,
        channel_weights=channel_weights,
        decoded_state_value_weight=config.decoded_state_value_weight,
        decoded_distribution_value_weight=(
            config.decoded_distribution_value_weight
        ),
        independent_diffusion_control=config.independent_diffusion_control,
        diffusion_policy_initial_logit=config.diffusion_policy_initial_logit,
        diffusion_variance_feedback_initial_gain=(
            config.diffusion_variance_feedback_initial_gain
        ),
    ).to(world.device)


def train_neural_hjb_fp(
    world,
    train_bank: dict[str, np.ndarray],
    reference_scaled: Tensor,
    reference_latent_mean: Tensor,
    reference_latent_variance: Tensor,
    reference_latent_covariance: Tensor,
    reference_path_scaled: Tensor,
    reference_path_latent: Tensor,
    channel_weights: Tensor,
    config: NeuralHJBFPConfig,
    *,
    initial_value_state_dict: dict[str, Tensor | np.ndarray] | None = None,
    initial_context_policy_state: dict[str, Tensor | np.ndarray] | None = None,
    fixed_context_feature_mean: np.ndarray | None = None,
    fixed_context_feature_scale: np.ndarray | None = None,
) -> tuple[
    MeanFieldValueNetwork,
    dict[str, Tensor],
    list[dict[str, float]],
    np.ndarray,
    np.ndarray,
]:
    """Train a global value policy with backward HJB and forward particle FP coupling."""

    config.validate()
    torch.manual_seed(int(config.seed))
    states, history, topology, context_indices = repeat_bank(
        train_bank,
        config.train_repeats,
        dtype=world.dtype,
        device=world.device,
    )
    raw_context_features = bank_context_features(train_bank)
    if fixed_context_feature_mean is None:
        context_feature_mean = raw_context_features.mean(axis=0)
    else:
        context_feature_mean = np.asarray(
            fixed_context_feature_mean, dtype=np.float64
        ).copy()
    if fixed_context_feature_scale is None:
        context_feature_scale = raw_context_features.std(axis=0)
        context_feature_scale[context_feature_scale < 1e-6] = 1.0
    else:
        context_feature_scale = np.asarray(
            fixed_context_feature_scale, dtype=np.float64
        ).copy()
    if context_feature_mean.shape != (raw_context_features.shape[1],):
        raise ValueError("fixed context-feature mean has the wrong shape")
    if context_feature_scale.shape != (raw_context_features.shape[1],):
        raise ValueError("fixed context-feature scale has the wrong shape")
    if np.any(context_feature_scale <= 0):
        raise ValueError("fixed context-feature scale must be positive")
    standardized_context_features = (
        raw_context_features - context_feature_mean[None]
    ) / context_feature_scale[None]
    context_features = torch.as_tensor(
        standardized_context_features, dtype=world.dtype, device=world.device
    )
    value_network = MeanFieldValueNetwork(
        world.q,
        config.hidden_size,
        config.hidden_layers,
        context_dim=context_features.shape[1],
        state_curvature_floor=config.state_curvature_floor,
        distribution_curvature_floor=config.distribution_curvature_floor,
        contractive_distribution_feedback_only=(
            config.contractive_distribution_feedback_only
        ),
    ).to(world.device)
    if initial_value_state_dict is not None:
        value_network.load_state_dict(initial_value_state_dict, strict=True)
    actor = make_actor(
        value_network,
        context_indices,
        context_features,
        world,
        config,
        reference_latent_covariance,
        reference_scaled,
        channel_weights,
    )
    if initial_context_policy_state is not None:
        policy_parameters = {
            "linear": actor.context_value_coefficients,
            "state_matrix": actor.context_state_matrix_coefficients,
            "distribution_matrix": actor.context_distribution_matrix_coefficients,
            "diffusion": actor.context_diffusion_coefficients,
            "diffusion_gain": actor.diffusion_direction_gain_logits,
            "diffusion_variance_feedback_gain": (
                actor.diffusion_variance_feedback_gain_logits
            ),
        }
        with torch.no_grad():
            for name, parameter in policy_parameters.items():
                if name not in initial_context_policy_state:
                    # Historical drift-only meta-checkpoints predate all
                    # diffusion-policy parameters.  They remain valid only
                    # when diffusion control is disabled; in that case the
                    # newly constructed actor's deterministic defaults are
                    # inactive and are therefore the unique neutral
                    # initialization.  Never silently accept the omission
                    # when the current configuration activates diffusion
                    # control.
                    if (
                        name
                        in {
                            "diffusion",
                            "diffusion_gain",
                            "diffusion_variance_feedback_gain",
                        }
                        and float(config.diffusion_control_strength) <= 0.0
                    ):
                        continue
                    raise ValueError(
                        f"initial context policy is missing parameter {name}"
                    )
                value = torch.as_tensor(
                    initial_context_policy_state[name],
                    dtype=world.dtype,
                    device=world.device,
                )
                if value.shape != parameter.shape:
                    # A meta-policy learned from several development contexts
                    # initializes a new seizure context by its context-wise
                    # mean.  This aggregation rule is frozen before test and
                    # never depends on the future test trajectory.
                    if (
                        value.ndim == parameter.ndim
                        and value.shape[1:] == parameter.shape[1:]
                        and parameter.shape[0] == len(train_bank["states"])
                    ):
                        value = value.mean(dim=0, keepdim=True).expand_as(
                            parameter
                        )
                    else:
                        raise ValueError(
                            f"initial context policy parameter {name} has "
                            "an incompatible shape"
                        )
                parameter.copy_(value)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(value_network.parameters()),
                "lr": config.learning_rate,
                "weight_decay": 1e-5,
            },
            {
                "params": [actor.context_value_coefficients],
                "lr": config.learning_rate * config.context_learning_rate_scale,
                "weight_decay": 0.0,
            },
            {
                "params": [
                    actor.context_state_matrix_coefficients,
                    actor.context_distribution_matrix_coefficients,
                ],
                "lr": config.learning_rate
                * config.context_matrix_learning_rate_scale,
                "weight_decay": 0.0,
            },
            {
                "params": [
                    actor.context_diffusion_coefficients,
                    actor.diffusion_direction_gain_logits,
                ],
                "lr": config.learning_rate * config.context_learning_rate_scale,
                "weight_decay": 0.0,
            },
        ],
    )
    reference_mean = reference_scaled.mean(dim=0)
    reference_variance = reference_scaled.var(dim=0, unbiased=False).clamp_min(1e-6)
    reference_scale = torch.sqrt(reference_variance).clamp_min(0.10)
    normalized_weights = channel_weights / channel_weights.mean().clamp_min(1e-6)
    # With few antithetic training particles, equally spaced interior order
    # statistics provide a lower-variance differentiable approximation of the
    # marginal one-dimensional Wasserstein distance than extreme quantiles.
    probabilities = torch.linspace(
        0.10,
        0.90,
        config.train_repeats,
        dtype=world.dtype,
        device=world.device,
    )
    reference_quantiles = torch.quantile(reference_scaled, probabilities, dim=0)
    # Fixed full-network projections define a differentiable empirical
    # sliced-Wasserstein functional.  Unlike marginal quantiles, it responds
    # to cross-channel dependence in the forward Fokker--Planck particle law.
    projection_rng = np.random.default_rng(int(config.seed) + 13007)
    projection_np = projection_rng.normal(
        size=(world.n_channels, int(config.joint_projection_count))
    )
    projection_np /= np.maximum(
        np.linalg.norm(projection_np, axis=0, keepdims=True), 1e-12
    )
    joint_projections = torch.as_tensor(
        projection_np, dtype=world.dtype, device=world.device
    )
    joint_probabilities = torch.linspace(
        0.05, 0.95, 19, dtype=world.dtype, device=world.device
    )
    joint_channel_scale = torch.sqrt(normalized_weights)[None]
    normalized_reference_joint = (
        (reference_scaled - reference_mean[None])
        / reference_scale[None]
    ) * joint_channel_scale
    reference_joint_quantiles = torch.quantile(
        normalized_reference_joint @ joint_projections,
        joint_probabilities,
        dim=0,
    )
    reference_latent_full = (
        reference_scaled - world.pca_mean[None]
    ) @ world.components.T
    reference_increment = torch.abs(
        reference_latent_full[1:] - reference_latent_full[:-1]
    )
    reference_log_line_length = torch.log(
        reference_increment.mean(dim=0).clamp_min(1e-6)
    )
    if bool(getattr(world, "regime_adaptive", False)):
        line_length_scale = world.gate_feature_scale[-world.q :].clamp_min(1e-4)
        line_length_importance = world.gate_coef[-world.q :].abs()
        line_length_importance = line_length_importance / (
            line_length_importance.mean().clamp_min(1e-6)
        )
        gate_window_samples = int(world.gate_window_samples)
    else:
        line_length_scale = torch.ones_like(reference_log_line_length)
        line_length_importance = torch.ones_like(reference_log_line_length)
        gate_window_samples = min(64, config.horizon_samples)
    time_weights = torch.linspace(
        config.trajectory_time_weight_start,
        1.0,
        config.horizon_samples,
        dtype=world.dtype,
        device=world.device,
    )
    history_rows: list[dict[str, float]] = []
    best_loss = math.inf
    best_epoch = -1
    best_value_state: dict[str, Tensor] | None = None
    best_policy_state: dict[str, Tensor] | None = None
    context_count = len(train_bank["states"])
    for epoch in range(config.epochs):
        actor.train()
        noise = antithetic_noise(
            context_count,
            config.train_repeats,
            config.horizon_samples,
            world.q,
            dtype=world.dtype,
            device=world.device,
            seed=config.seed + 104729 * (epoch + 1),
        )
        optimizer.zero_grad(set_to_none=True)
        scaled, latent, controls, _, _ = world.rollout(
            states,
            history,
            topology,
            horizon_samples=config.horizon_samples,
            noise=noise,
            actor=actor,
            reference_mean=reference_path_latent,
            reference_variance=reference_latent_variance,
            pulse_width_samples=1,
            temperature=1.0,
            stochastic_actor=False,
            diffusion_scale=config.plant_diffusion_scale,
            diffusion_control_strength=config.diffusion_control_strength,
            independent_diffusion_control=config.independent_diffusion_control,
            control_input_gain=(
                config.control_step_scale / world.dt
                if config.control_step_scale > 0
                else config.control_input_gain
            ),
        )
        diffusion_actions = world.last_diffusion_actions
        if bool(getattr(world, "regime_adaptive", False)):
            tail_regime_logit = world.last_regime_logits[
                :, -config.tail_samples :
            ]
            # softplus(logit) == -log(1-sigmoid(logit)), evaluated in a
            # numerically stable form whose gradient remains non-zero even
            # when the ictal probability is close to one.
            regime_transition_loss = F.softplus(tail_regime_logit).mean()
        else:
            regime_transition_loss = controls.new_zeros(())
        sequence = scaled[:, 1:].reshape(
            context_count,
            config.train_repeats,
            config.horizon_samples,
            world.n_channels,
        )
        ensemble_mean = sequence.mean(dim=1)
        normalized_path_error = (
            (ensemble_mean - reference_path_scaled[None])
            / reference_scale[None, None, :]
        ).square()
        trajectory_loss = (
            normalized_path_error
            * normalized_weights[None, None, :]
            * time_weights[None, :, None]
        ).mean()
        per_particle_path_error = (
            (
                (
                    (sequence - reference_path_scaled[None, None])
                    / reference_scale[None, None, None, :]
                ).square()
                * normalized_weights[None, None, None, :]
                * time_weights[None, None, :, None]
            ).mean(dim=(2, 3))
        )
        cvar_count = max(
            1, int(math.ceil(config.cvar_tail_fraction * config.train_repeats))
        )
        path_cvar_loss = torch.topk(
            per_particle_path_error, cvar_count, dim=1
        ).values.mean()
        tail = sequence[:, :, -config.tail_samples :]
        tail_path_loss = (
            normalized_path_error[:, -config.tail_samples :]
            * normalized_weights[None, None, :]
        ).mean()
        per_particle_tail_error = (
            (
                (
                    (
                        tail
                        - reference_path_scaled[None, None, -config.tail_samples :]
                    )
                    / reference_scale[None, None, None, :]
                ).square()
                * normalized_weights[None, None, None, :]
            ).mean(dim=(2, 3))
        )
        tail_cvar_loss = torch.topk(
            per_particle_tail_error, cvar_count, dim=1
        ).values.mean()
        tail_standardized_distance = (
            tail - reference_mean[None, None, None, :]
        ).abs() / reference_scale[None, None, None, :]
        per_particle_tube_exceedance = (
            torch.relu(tail_standardized_distance - config.tube_radius_std).square()
            * normalized_weights[None, None, None, :]
        ).mean(dim=(2, 3))
        tube_cvar_loss = torch.topk(
            per_particle_tube_exceedance, cvar_count, dim=1
        ).values.mean()
        # Match the Fokker--Planck particle law at each time.  Pooling time and
        # particles would let temporal mean changes conceal an over-dispersed
        # particle cloud, which was the dominant failure mode in v6.
        tail_mean = tail.mean(dim=1)
        tail_variance = tail.var(dim=1, unbiased=False).clamp_min(1e-6)
        mean_loss = (
            (
                (tail_mean - reference_mean[None, None])
                / reference_scale[None, None]
            ).square()
            * normalized_weights[None, None]
        ).mean()
        variance_loss = (
            (
                torch.log(tail_variance)
                - torch.log(reference_variance[None, None])
            ).square()
            * normalized_weights[None, None]
        ).mean()
        predicted_quantiles = torch.quantile(
            tail, probabilities, dim=1
        )
        per_channel_quantile = (
            (
                predicted_quantiles
                - reference_quantiles[:, None, None, :]
            ).abs()
            / reference_scale[None, None, None, :]
        ).mean(dim=(0, 1, 2))
        weighted_quantile = per_channel_quantile * normalized_weights
        quantile_loss = weighted_quantile.mean()
        worst_quantile_loss = torch.topk(
            weighted_quantile, min(4, world.n_channels)
        ).values.mean()
        terminal_length = min(16, config.tail_samples)
        terminal = tail[:, :, -terminal_length:]
        terminal_quantiles = torch.quantile(terminal, probabilities, dim=1)
        terminal_channel_quantile = (
            (
                terminal_quantiles
                - reference_quantiles[:, None, None, :]
            ).abs()
            / reference_scale[None, None, None, :]
        ).mean(dim=(0, 1, 2))
        terminal_quantile_loss = (
            terminal_channel_quantile * normalized_weights
        ).mean()
        joint_time_indices = torch.linspace(
            0,
            config.tail_samples - 1,
            min(8, config.tail_samples),
            dtype=torch.long,
            device=world.device,
        ).unique()
        joint_terms: list[Tensor] = []
        for joint_index in joint_time_indices:
            predicted_joint = tail[:, :, joint_index].reshape(
                -1, world.n_channels
            )
            normalized_predicted_joint = (
                (predicted_joint - reference_mean[None])
                / reference_scale[None]
            ) * joint_channel_scale
            predicted_joint_quantiles = torch.quantile(
                normalized_predicted_joint @ joint_projections,
                joint_probabilities,
                dim=0,
            )
            joint_terms.append(
                (
                    predicted_joint_quantiles
                    - reference_joint_quantiles
                ).square().mean()
            )
        joint_density_loss = torch.stack(joint_terms).mean()
        if config.line_length_weight > 0:
            latent_increment = torch.abs(latent[:, 1:] - latent[:, :-1])
            difference_window = max(
                1,
                min(gate_window_samples - 1, latent_increment.shape[1]),
            )
            cumulative_increment = F.pad(
                latent_increment.cumsum(dim=1), (0, 0, 1, 0)
            )
            rolling_line_length = (
                cumulative_increment[:, difference_window:]
                - cumulative_increment[:, :-difference_window]
            ) / float(difference_window)
            predicted_log_line_length = torch.log(
                rolling_line_length.clamp_min(1e-6)
            )
            line_length_loss = (
                (
                    (
                        predicted_log_line_length
                        - reference_log_line_length[None, None]
                    )
                    / line_length_scale[None, None]
                ).square()
                * line_length_importance[None, None]
            ).mean()
        else:
            line_length_loss = controls.new_zeros(())
        normalized_control = controls / config.amplitude_limit
        energy = normalized_control.square().mean()
        diffusion_control_energy = diffusion_actions.square().mean()
        first_difference = normalized_control[:, 1:] - normalized_control[:, :-1]
        second_difference = (
            normalized_control[:, 2:]
            - 2.0 * normalized_control[:, 1:-1]
            + normalized_control[:, :-2]
        )
        smoothness = first_difference.square().mean()
        curvature = second_difference.square().mean()
        context_value_regularization = actor.context_value_coefficients.square().mean()
        context_matrix_regularization = (
            actor.context_state_matrix_coefficients.square().mean()
            + actor.context_distribution_matrix_coefficients.square().mean()
        )
        variance_feedback_regularization = (
            actor.diffusion_variance_feedback_gain_logits.square().mean()
            if config.diffusion_variance_feedback_initial_gain > 0
            else controls.new_zeros(())
        )
        diffusion_policy_regularization = (
            actor.context_diffusion_coefficients.square().mean()
            + 0.1 * actor.diffusion_direction_gain_logits.square().mean()
            + 0.01 * variance_feedback_regularization
        )

        # Discrete stochastic HJB residual evaluated on a sparse time grid.
        hjb_terms: list[Tensor] = []
        stride = max(1, config.horizon_samples // 8)
        latent_reference_scale = torch.sqrt(reference_latent_variance).clamp_min(0.10)
        for step in range(0, config.horizon_samples - 1, stride):
            current_value = actor.value(
                latent[:, step],
                reference_path_latent[step],
                reference_latent_variance,
                step / max(config.horizon_samples - 1, 1),
            )
            next_value = actor.value(
                latent[:, step + 1],
                reference_path_latent[step + 1],
                reference_latent_variance,
                (step + 1) / max(config.horizon_samples - 1, 1),
            )
            local_state_cost = 0.5 * (
                (
                    (latent[:, step] - reference_path_latent[step][None])
                    / latent_reference_scale[None]
                ).square()
            ).mean(dim=1)
            law_mean, law_variance = MeanFieldValueNetwork.context_moments(
                latent[:, step], actor.context_indices
            )
            local_law_by_context = 0.5 * (
                (
                    (law_mean - reference_path_latent[step][None])
                    / latent_reference_scale[None]
                ).square().mean(dim=1)
                + 0.25
                * (
                    torch.log(law_variance.clamp_min(1e-6))
                    - torch.log(reference_latent_variance[None])
                ).square().mean(dim=1)
            )
            local_law_cost = local_law_by_context[actor.context_indices]
            step_control = controls[:, step]
            local_control_cost = 0.5 * torch.einsum(
                "bi,ij,bj->b",
                step_control,
                actor.control_cost_matrix,
                step_control,
            )
            # Keep the Bellman running cost identical to the quadratic form
            # used in the Hamiltonian first-order condition.  Dividing this
            # term by the actuator count while leaving R unchanged in the
            # actor would make the two problems differ by a factor of 13.
            previous_step_control = (
                controls[:, step - 1]
                if step > 0
                else torch.zeros_like(step_control)
            )
            previous_previous_step_control = (
                controls[:, step - 2]
                if step > 1
                else torch.zeros_like(step_control)
            )
            local_control_cost = local_control_cost + 0.5 * (
                actor.hamiltonian_rate_cost
                * (step_control - previous_step_control).square().mean(dim=1)
                + actor.hamiltonian_acceleration_cost
                * (
                    step_control
                    - 2.0 * previous_step_control
                    + previous_previous_step_control
                ).square().mean(dim=1)
            )
            local_control_cost = local_control_cost + 0.5 * (
                config.diffusion_control_cost
                * diffusion_actions[:, step].square().mean(dim=1)
            )
            bellman_target = world.dt * (
                local_state_cost
                + config.hjb_law_cost_scale * local_law_cost
                + local_control_cost
            ) + next_value
            hjb_terms.append((current_value - bellman_target).square().mean())
        hjb_residual = torch.stack(hjb_terms).mean()
        terminal_value = actor.value(
            latent[:, -1],
            reference_path_latent[-1],
            reference_latent_variance,
            1.0,
        )
        terminal_target = 0.5 * (
            (
                (latent[:, -1] - reference_path_latent[-1][None])
                / latent_reference_scale[None]
            ).square()
        ).sum(dim=1)
        terminal_value_loss = (terminal_value - terminal_target).square().mean()

        loss = (
            config.trajectory_weight * trajectory_loss
            + config.tail_trajectory_weight * tail_path_loss
            + config.mean_weight * mean_loss
            + config.variance_weight * variance_loss
            + config.quantile_weight * quantile_loss
            + config.worst_quantile_weight * worst_quantile_loss
            + config.terminal_quantile_weight * terminal_quantile_loss
            + config.joint_density_weight * joint_density_loss
            + config.regime_transition_weight * regime_transition_loss
            + config.line_length_weight * line_length_loss
            + config.path_cvar_weight * path_cvar_loss
            + config.tail_cvar_weight * tail_cvar_loss
            + config.tube_cvar_weight * tube_cvar_loss
            + config.hjb_residual_weight * hjb_residual
            + config.terminal_value_weight * terminal_value_loss
            + config.energy_weight * energy
            + config.diffusion_control_cost * diffusion_control_energy
            + config.smoothness_weight * smoothness
            + config.curvature_weight * curvature
            + config.context_value_regularization_weight
            * context_value_regularization
            + config.context_matrix_regularization_weight
            * context_matrix_regularization
            + config.context_value_regularization_weight
            * diffusion_policy_regularization
        )
        current_loss = float(loss.detach().cpu())
        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch
            # Store the parameters that produced the audited objective, not
            # simply the final optimizer iterate.  The historical run could
            # finish well past its minimum and silently return a worse policy.
            best_value_state = copy.deepcopy(value_network.state_dict())
            best_policy_state = {
                "linear": actor.context_value_coefficients.detach().clone(),
                "state_matrix": (
                    actor.context_state_matrix_coefficients.detach().clone()
                ),
                "distribution_matrix": (
                    actor.context_distribution_matrix_coefficients.detach().clone()
                ),
                "diffusion": actor.context_diffusion_coefficients.detach().clone(),
                "diffusion_gain": (
                    actor.diffusion_direction_gain_logits.detach().clone()
                ),
                "diffusion_variance_feedback_gain": (
                    actor.diffusion_variance_feedback_gain_logits.detach().clone()
                ),
            }
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), config.gradient_clip)
        optimizer.step()
        row = {
            "epoch": float(epoch + 1),
            "loss": current_loss,
            "trajectory_loss": float(trajectory_loss.detach().cpu()),
            "tail_path_loss": float(tail_path_loss.detach().cpu()),
            "mean_loss": float(mean_loss.detach().cpu()),
            "variance_loss": float(variance_loss.detach().cpu()),
            "quantile_loss": float(quantile_loss.detach().cpu()),
            "worst_quantile_loss": float(worst_quantile_loss.detach().cpu()),
            "terminal_quantile_loss": float(
                terminal_quantile_loss.detach().cpu()
            ),
            "joint_density_loss": float(joint_density_loss.detach().cpu()),
            "regime_transition_loss": float(
                regime_transition_loss.detach().cpu()
            ),
            "line_length_loss": float(line_length_loss.detach().cpu()),
            "path_cvar_loss": float(path_cvar_loss.detach().cpu()),
            "tail_cvar_loss": float(tail_cvar_loss.detach().cpu()),
            "tube_cvar_loss": float(tube_cvar_loss.detach().cpu()),
            "hjb_residual": float(hjb_residual.detach().cpu()),
            "terminal_value_loss": float(terminal_value_loss.detach().cpu()),
            "energy": float(energy.detach().cpu()),
            "diffusion_control_energy": float(
                diffusion_control_energy.detach().cpu()
            ),
            "smoothness": float(smoothness.detach().cpu()),
            "curvature": float(curvature.detach().cpu()),
            "context_value_regularization": float(
                context_value_regularization.detach().cpu()
            ),
            "context_matrix_regularization": float(
                context_matrix_regularization.detach().cpu()
            ),
            "diffusion_policy_regularization": float(
                diffusion_policy_regularization.detach().cpu()
            ),
        }
        history_rows.append(row)
        if epoch == 0 or (epoch + 1) % 20 == 0 or epoch + 1 == config.epochs:
            print(
                "HJB-FP epoch %d/%d loss=%.3f path=%.3f q=%.3f hjb=%.3f"
                % (
                    epoch + 1,
                    config.epochs,
                    row["loss"],
                    row["trajectory_loss"],
                    row["quantile_loss"],
                    row["hjb_residual"],
                ),
                flush=True,
            )
    if best_value_state is None or best_policy_state is None or best_epoch < 0:
        raise RuntimeError("training did not produce a finite checkpoint")
    value_network.load_state_dict(best_value_state, strict=True)
    for row_index, row in enumerate(history_rows):
        row["selected_checkpoint"] = float(row_index == best_epoch)
    return (
        value_network,
        {name: value.detach().cpu() for name, value in best_policy_state.items()},
        history_rows,
        context_feature_mean,
        context_feature_scale,
    )


def evaluate_neural_hjb_fp(
    world,
    value_network: MeanFieldValueNetwork,
    bank: dict[str, np.ndarray],
    reference_path_latent: Tensor,
    reference_latent_variance: Tensor,
    reference_latent_covariance: Tensor,
    reference_scaled: Tensor,
    channel_weights: Tensor,
    config: NeuralHJBFPConfig,
    context_feature_mean: np.ndarray,
    context_feature_scale: np.ndarray,
    context_policy_state: dict[str, Tensor | np.ndarray] | None,
    *,
    repeats: int,
    seed: int,
    actor_world=None,
) -> dict[str, np.ndarray]:
    """Evaluate a fixed neural HJB policy on a Graph-RC SDE particle law.

    By default the policy and plant use the same world.  Supplying
    ``actor_world`` freezes the Hamiltonian map to that world while ``world``
    remains the rollout plant.  This separation is intended for a strict
    plant-side graph-transport ablation; it does not retrain or remap the
    policy.
    """

    policy_world = world if actor_world is None else actor_world
    _validate_actor_world_for_plant(world, policy_world)
    states, history, topology, context_indices = repeat_bank(
        bank, repeats, dtype=world.dtype, device=world.device
    )
    raw_context_features = bank_context_features(bank)
    standardized_context_features = (
        raw_context_features - np.asarray(context_feature_mean)[None]
    ) / np.asarray(context_feature_scale)[None]
    context_features = torch.as_tensor(
        standardized_context_features, dtype=world.dtype, device=world.device
    )
    actor = make_actor(
        value_network,
        context_indices,
        context_features,
        policy_world,
        config,
        reference_latent_covariance,
        reference_scaled,
        channel_weights,
    )
    if context_policy_state is not None:
        parameters = {
            "linear": actor.context_value_coefficients,
            "state_matrix": actor.context_state_matrix_coefficients,
            "distribution_matrix": actor.context_distribution_matrix_coefficients,
            "diffusion": actor.context_diffusion_coefficients,
            "diffusion_gain": actor.diffusion_direction_gain_logits,
            "diffusion_variance_feedback_gain": (
                actor.diffusion_variance_feedback_gain_logits
            ),
        }
        with torch.no_grad():
            for name, parameter in parameters.items():
                if name not in context_policy_state:
                    if name == "diffusion_variance_feedback_gain":
                        continue
                    raise ValueError(
                        f"context policy state is missing parameter {name}"
                    )
                value = torch.as_tensor(
                    context_policy_state[name], dtype=world.dtype, device=world.device
                )
                if value.shape != parameter.shape:
                    raise ValueError(
                        f"context policy parameter {name} does not match the evaluation bank"
                    )
                parameter.copy_(value)
    actor.eval()
    noise = antithetic_noise(
        len(bank["states"]),
        repeats,
        config.horizon_samples,
        world.q,
        dtype=world.dtype,
        device=world.device,
        seed=seed,
    )
    with torch.no_grad():
        controlled = world.rollout(
            states,
            history,
            topology,
            horizon_samples=config.horizon_samples,
            noise=noise,
            actor=actor,
            reference_mean=reference_path_latent,
            reference_variance=reference_latent_variance,
            pulse_width_samples=1,
            temperature=1.0,
            stochastic_actor=False,
            diffusion_scale=config.plant_diffusion_scale,
            diffusion_control_strength=config.diffusion_control_strength,
            independent_diffusion_control=config.independent_diffusion_control,
            control_input_gain=(
                config.control_step_scale / world.dt
                if config.control_step_scale > 0
                else config.control_input_gain
            ),
        )
        controlled_regime_probability = (
            world.last_regime_probabilities.detach().clone()
        )
        controlled_diffusion_modulation = (
            world.last_diffusion_modulation.detach().clone()
        )
        controlled_diffusion_action = (
            world.last_diffusion_actions.detach().clone()
        )
        uncontrolled = world.rollout(
            states,
            history,
            topology,
            horizon_samples=config.horizon_samples,
            noise=noise,
            actor=None,
            reference_mean=reference_path_latent,
            reference_variance=reference_latent_variance,
            pulse_width_samples=1,
            temperature=1.0,
            stochastic_actor=False,
            diffusion_scale=config.plant_diffusion_scale,
            diffusion_control_strength=config.diffusion_control_strength,
            independent_diffusion_control=config.independent_diffusion_control,
            control_input_gain=(
                config.control_step_scale / world.dt
                if config.control_step_scale > 0
                else config.control_input_gain
            ),
        )
        uncontrolled_regime_probability = (
            world.last_regime_probabilities.detach().clone()
        )
    return {
        "controlled_scaled": controlled[0].cpu().numpy().astype(np.float64),
        "controlled_latent": controlled[1].cpu().numpy().astype(np.float64),
        "controls": controlled[2].cpu().numpy().astype(np.float64),
        "uncontrolled_scaled": uncontrolled[0].cpu().numpy().astype(np.float64),
        "controlled_ictal_probability": controlled_regime_probability.cpu()
        .numpy()
        .astype(np.float64),
        "controlled_diffusion_modulation": controlled_diffusion_modulation.cpu()
        .numpy()
        .astype(np.float64),
        "controlled_diffusion_action": controlled_diffusion_action.cpu()
        .numpy()
        .astype(np.float64),
        "diffusion_variance_feedback_gain": F.softplus(
            actor.diffusion_variance_feedback_gain_logits
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64),
        "uncontrolled_ictal_probability": uncontrolled_regime_probability.cpu()
        .numpy()
        .astype(np.float64),
        "positions": np.asarray(bank["positions"], dtype=np.int64),
        "rollouts_per_context": np.asarray([int(repeats)], dtype=np.int64),
    }


__all__ = [
    "MeanFieldValueNetwork",
    "NeuralHJBActor",
    "NeuralHJBFPConfig",
    "bank_context_features",
    "bounded_mean_deviation_feedback",
    "evaluate_neural_hjb_fp",
    "train_neural_hjb_fp",
]
