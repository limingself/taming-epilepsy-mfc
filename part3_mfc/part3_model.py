#!/usr/bin/env python
"""Final paper Part III: Actor--WGAN mean-field control workflow.

The public commands are ``train``, ``evaluate``, ``evaluate-ablations`` and
``source-data``.
Training is prefix-only; evaluation opens the frozen run-02 development
window only after checkpoint freezing.  The forward empirical Fokker--Planck
law is propagated by the frozen state-dependent Graph-RC-SDE from Part II.

Figure 6/8 use the frozen main-result run
``seed20261011_adv050_anchor020``.  Figure 7 uses the separately retrained,
matched-budget full checkpoint and its three matched ablations.  These are
the same architecture but are not claimed to be the same numerical weights.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import torch


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from mfc_pipeline.actor_wgan import (
    actor_action_anchor_loss,
    actor_wasserstein_loss,
    balanced_wgan_gp_loss,
)
from mfc_pipeline.causal_ltv_particle_rollout import (
    FrozenIctalGraphRCBatchStepper,
    paired_ictal_batch_rollout,
    reconstruct_paired_normals,
)
from mfc_pipeline.causal_ltv_riccati import FrozenGraphRCMarkovAdapter
from mfc_pipeline.full_markov_hjb_fp_wgan import (
    TimeConditionedWassersteinCritic,
    empirical_fp_rollout,
    set_requires_grad,
)
from mfc_pipeline.sequential_covariance_hjb import (
    StructuredSplineCovarianceActor,
    uncontrolled_particle_rollout,
)
from mfc_pipeline.square_wave_mfc import TorchGraphRCSDE


HORIZON = 256
PARTICLES = 32
FS = 256.0
CONTROL_STEP_SCALE = 0.5
DIFFUSION_SCALE = 0.79451175
GRAPH_DIFFUSION_TIME = 0.5
AMPLITUDE_LIMIT = 1.8
SEED = 20261011
VALIDATION_SEED = 20260922

SYNTHESIS_CONTRACT = PROJECT / "artifacts" / "part3_hup060_causal_synthesis_contract_v1" / "synthesis_only_inputs.npz"
MODEL_PATH = PROJECT / "artifacts" / "part2_hup060_state_dependent_optimized" / "hup060_state_dependent_rc.joblib"
SEALED_CALIBRATED = PROJECT / "artifacts" / "part3_hup060_neural_hjb_fp" / "physical_residual_identity_run02_context3_frozen_e5_v1" / "test_rollout.npz"
LEGAL_EVALUATION = PROJECT / "artifacts" / "part2_hup060_context3_window_contract_v1" / "same_window_legal_inputs.npz"
ORIGINAL_DIR = PROJECT / "artifacts" / "part3_hup060_sequential_covariance_v1" / "run02_context3_markov_precision_joint_v1"
ORIGINAL_ACTOR = ORIGINAL_DIR / "frozen_actor.pt"
ORIGINAL_ROLLOUT = ORIGINAL_DIR / "paired_rollout.npz"
ORIGINAL_SUMMARY = ORIGINAL_DIR / "summary.json"
TEACHER_CHECKPOINT = ORIGINAL_ACTOR
OUTPUT_ROOT = PROJECT / "artifacts" / "part3_hup060_actor_wgan_v1"
CANDIDATE_ROOT = OUTPUT_ROOT

# Aliases retained inside the matched-ablation evaluation contract.
SYNTHESIS = SYNTHESIS_CONTRACT
MODEL = MODEL_PATH
SEALED = SEALED_CALIBRATED
LEGAL = LEGAL_EVALUATION
ROOT = OUTPUT_ROOT


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_part2_model_definitions() -> None:
    """Load pickle class definitions from the single public Part II model file."""
    path = PROJECT / "part2_rc_sde" / "part2_model.py"
    if not path.is_file():
        raise FileNotFoundError(f"Part II model entry was not found: {path}")
    load_script("part2_state_dependent_base", path)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def antithetic_noise(seed: int, q: int) -> torch.Tensor:
    rng = np.random.default_rng(int(seed))
    half = rng.standard_normal((PARTICLES // 2, HORIZON, q))
    return torch.as_tensor(
        np.concatenate([half, -half], axis=0), dtype=torch.float64
    )


def channelwise_time_w1(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.mean(
                [
                    wasserstein_distance(
                        predicted[:, step, channel], reference[:, step, channel]
                    )
                    for step in range(predicted.shape[1])
                ]
            )
            for channel in range(predicted.shape[2])
        ],
        dtype=np.float64,
    )


def channelwise_occupation_w1(
    predicted: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    return np.asarray(
        [
            wasserstein_distance(
                predicted[:, :, channel].reshape(-1),
                reference[:, :, channel].reshape(-1),
            )
            for channel in range(predicted.shape[2])
        ],
        dtype=np.float64,
    )


def per_time_w1(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.mean(
                [
                    wasserstein_distance(
                        predicted[:, step, channel], reference[:, step, channel]
                    )
                    for step in range(predicted.shape[1])
                ]
            )
            for channel in range(predicted.shape[2])
        ],
        dtype=np.float64,
    )


def occupation_w1(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            wasserstein_distance(
                predicted[:, :, channel].reshape(-1),
                reference[:, :, channel].reshape(-1),
            )
            for channel in range(predicted.shape[2])
        ],
        dtype=np.float64,
    )


def control_diagnostics(control: np.ndarray) -> dict[str, float]:
    return {
        "peak": float(np.max(np.abs(control))),
        "rms": float(np.sqrt(np.mean(control**2))),
        "maximum_first_difference": float(
            np.max(np.abs(np.diff(control, axis=1)))
        ),
        "maximum_second_difference": float(
            np.max(np.abs(np.diff(control, n=2, axis=1)))
        ),
        "saturation_fraction_over_95pct": float(
            np.mean(np.abs(control) >= 0.95 * 1.8)
        ),
    }


def paired_channel_bootstrap(
    candidate: np.ndarray,
    original: np.ndarray,
    *,
    seed: int,
    replicates: int = 10000,
) -> dict[str, float | str]:
    """Descriptive within-patient channel bootstrap; not patient inference."""

    difference = np.asarray(candidate) - np.asarray(original)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(difference), size=(int(replicates), len(difference)))
    means = difference[indices].mean(axis=1)
    return {
        "mean_candidate_minus_original": float(difference.mean()),
        "median_candidate_minus_original": float(np.median(difference)),
        "descriptive_95pct_interval_low": float(np.quantile(means, 0.025)),
        "descriptive_95pct_interval_high": float(np.quantile(means, 0.975)),
        "unit_warning": "36 channels from one patient are not independent patients",
    }


def sliced_w1(
    predicted: np.ndarray,
    reference: np.ndarray,
    projections: np.ndarray,
) -> tuple[float, float]:
    projected_predicted = np.einsum("mtd,dp->mtp", predicted, projections)
    projected_reference = np.einsum("rtd,dp->rtp", reference, projections)
    time_values = []
    for step in range(predicted.shape[1]):
        for projection in range(projections.shape[1]):
            time_values.append(
                wasserstein_distance(
                    projected_predicted[:, step, projection],
                    projected_reference[:, step, projection],
                )
            )
    occupation_values = [
        wasserstein_distance(
            projected_predicted[:, :, projection].reshape(-1),
            projected_reference[:, :, projection].reshape(-1),
        )
        for projection in range(projections.shape[1])
    ]
    return float(np.mean(time_values)), float(np.mean(occupation_values))


def reference_statistics(reference: torch.Tensor):
    pooled_variance = reference.reshape(-1, 36).var(dim=0, unbiased=False)
    scale = torch.sqrt(pooled_variance.clamp_min(0.10**2))
    inverse = 1.0 / pooled_variance.clamp_min(0.10**2)
    inverse = inverse / inverse.mean()
    weights = (0.50 + 0.50 * inverse).clamp(0.50, 3.0)
    weights = weights / weights.mean()
    mean = reference.mean(dim=0)
    variance = (
        0.25 * reference.var(dim=0, unbiased=False)
        + 0.75 * pooled_variance[None]
    ).clamp_min(1.0e-5)
    return mean, variance, scale, weights


def build_markov_normalization(
    adapter: FrozenGraphRCMarkovAdapter,
    initial: torch.Tensor,
    fit_reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    center = initial.detach().clone()
    scale = torch.ones_like(initial)
    scale[adapter.slices.reservoir] = 0.25
    latent_reference = (
        fit_reference - adapter.pca_mean[None, None]
    ) @ adapter.components.T
    latent_scale = latent_reference.reshape(-1, adapter.q).std(
        dim=0, unbiased=False
    ).clamp_min(0.10)
    scale[adapter.slices.history] = latent_scale.repeat(adapter.history_length)
    topology_reference = (
        fit_reference @ adapter.adjacency.T
    ) @ adapter.components.T
    topology_scale = topology_reference.reshape(-1, adapter.q).std(
        dim=0, unbiased=False
    ).clamp_min(0.10)
    scale[adapter.slices.topology] = topology_scale
    return center, scale


def build_teacher(
    stepper: FrozenIctalGraphRCBatchStepper,
    reference_mean: torch.Tensor,
    reference_variance: torch.Tensor,
    reference_scale: torch.Tensor,
    selected_indices: np.ndarray,
    markov_center: torch.Tensor,
    markov_scale: torch.Tensor,
) -> StructuredSplineCovarianceActor:
    base_gain = torch.zeros(
        stepper.actuator_dim, stepper.n_channels, dtype=torch.float64
    )
    teacher = StructuredSplineCovarianceActor(
        stepper,
        reference_mean,
        reference_variance,
        reference_scale,
        base_gain,
        selected_indices,
        horizon=HORIZON,
        basis_count=8,
        hidden_size=64,
        amplitude_limit=AMPLITUDE_LIMIT,
        actuator_alpha=1.0,
        residual_scale=0.15,
        local_gain_initial_fraction=0.0,
        local_gain_maximum_fraction=0.05,
        markov_feature_center=markov_center,
        markov_feature_scale=markov_scale,
        markov_residual_scale=0.6,
        markov_hidden_size=96,
        maximum_slew=None,
    )
    payload = torch.load(TEACHER_CHECKPOINT, map_location="cpu", weights_only=False)
    teacher.load_state_dict(payload["actor_state_dict"], strict=True)
    teacher.eval()
    set_requires_grad(teacher, False)
    return teacher


def law_objective(
    sequence: torch.Tensor,
    controls: torch.Tensor,
    commands: torch.Tensor,
    reference: torch.Tensor,
    channel_weights: torch.Tensor,
    reference_scale: torch.Tensor,
    joint_projections: torch.Tensor,
    actor: StructuredSplineCovarianceActor,
    *,
    include_parameter_regularization: bool,
    baseline_sequence: torch.Tensor | None,
    objective_mode: str,
    smoothness_weight: float,
    curvature_weight: float,
    slew_barrier_weight: float,
    curvature_barrier_weight: float,
    maximum_first_difference: float,
    maximum_second_difference: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Time-resolved empirical FP loss with explicit covariance targets."""

    particles, horizon, channels = sequence.shape
    if horizon != HORIZON or channels != len(channel_weights):
        raise ValueError("sequence dimensions differ from frozen contract")
    reference_mean = reference.mean(dim=0)
    pooled_variance = reference.reshape(-1, channels).var(
        dim=0, unbiased=False
    ).clamp_min(1e-5)
    time_variance = reference.var(dim=0, unbiased=False)
    reference_variance = (
        0.25 * time_variance + 0.75 * pooled_variance[None]
    ).clamp_min(1e-5)
    predicted_mean = sequence.mean(dim=0)
    predicted_variance = sequence.var(dim=0, unbiased=False).clamp_min(1e-5)
    time_weights = torch.linspace(
        0.25, 1.0, horizon, dtype=sequence.dtype, device=sequence.device
    )
    time_weights = time_weights / time_weights.mean()

    weighted_mean_error = (
        (
            (predicted_mean - reference_mean)
            / reference_scale[None]
        ).square()
        * channel_weights[None]
        * time_weights[:, None]
    )
    mean_loss = weighted_mean_error.mean()
    per_channel_mean = weighted_mean_error.mean(dim=0)
    weighted_log_variance_error = (
        (
            torch.log(predicted_variance)
            - torch.log(reference_variance)
        ).square()
        * channel_weights[None]
        * time_weights[:, None]
    )
    log_variance_loss = weighted_log_variance_error.mean()
    per_channel_log_variance = weighted_log_variance_error.mean(dim=0)
    worst_log_variance_loss = torch.topk(
        per_channel_log_variance, min(4, channels)
    ).values.mean()

    probabilities = torch.linspace(
        0.10, 0.90, 9, dtype=sequence.dtype, device=sequence.device
    )
    time_indices = torch.arange(0, horizon, 4, device=sequence.device)
    predicted_quantiles = torch.quantile(
        sequence[:, time_indices], probabilities, dim=0
    )
    reference_quantiles = torch.quantile(
        reference[:, time_indices], probabilities, dim=0
    )
    weighted_quantile_error = (
        (
            (predicted_quantiles - reference_quantiles)
            / reference_scale[None, None]
        ).square()
        * channel_weights[None, None]
        * time_weights[time_indices][None, :, None]
    )
    quantile_loss = weighted_quantile_error.mean()
    per_channel_quantile = weighted_quantile_error.mean(dim=(0, 1))
    worst_quantile_loss = torch.topk(
        per_channel_quantile, min(4, channels)
    ).values.mean()
    if baseline_sequence is None:
        no_harm_mean_loss = mean_loss.new_zeros(())
        no_harm_quantile_loss = mean_loss.new_zeros(())
    else:
        if baseline_sequence.shape != sequence.shape:
            raise ValueError("baseline_sequence must match controlled sequence")
        baseline_mean = baseline_sequence.mean(dim=0)
        baseline_weighted_mean = (
            (
                (baseline_mean - reference_mean)
                / reference_scale[None]
            ).square()
            * channel_weights[None]
            * time_weights[:, None]
        )
        baseline_per_channel_mean = baseline_weighted_mean.mean(dim=0)
        baseline_quantiles = torch.quantile(
            baseline_sequence[:, time_indices], probabilities, dim=0
        )
        baseline_weighted_quantile = (
            (
                (baseline_quantiles - reference_quantiles)
                / reference_scale[None, None]
            ).square()
            * channel_weights[None, None]
            * time_weights[time_indices][None, :, None]
        )
        baseline_per_channel_quantile = baseline_weighted_quantile.mean(
            dim=(0, 1)
        )
        mean_harm = torch.relu(
            per_channel_mean - baseline_per_channel_mean
        )
        quantile_harm = torch.relu(
            per_channel_quantile - baseline_per_channel_quantile
        )
        no_harm_mean_loss = (
            mean_harm.mean()
            + torch.topk(mean_harm, min(4, channels)).values.mean()
        )
        no_harm_quantile_loss = (
            quantile_harm.mean()
            + torch.topk(quantile_harm, min(4, channels)).values.mean()
        )

    occupancy_probabilities = torch.linspace(
        0.05, 0.95, 19, dtype=sequence.dtype, device=sequence.device
    )
    predicted_occupancy = torch.quantile(
        sequence.reshape(-1, channels), occupancy_probabilities, dim=0
    )
    reference_occupancy = torch.quantile(
        reference.reshape(-1, channels), occupancy_probabilities, dim=0
    )
    occupancy_loss = (
        (
            (predicted_occupancy - reference_occupancy)
            / reference_scale[None]
        ).square()
        * channel_weights[None]
    ).mean()

    lower = torch.quantile(reference, 0.10, dim=0)
    upper = torch.quantile(reference, 0.90, dim=0)
    below = torch.relu(lower[None] - sequence)
    above = torch.relu(sequence - upper[None])
    tube_loss = (
        ((below + above) / reference_scale[None, None]).square()
        * channel_weights[None, None]
        * time_weights[None, :, None]
    ).mean()

    joint_indices = torch.arange(0, horizon, 16, device=sequence.device)
    joint_terms: list[torch.Tensor] = []
    joint_probabilities = torch.linspace(
        0.10, 0.90, 9, dtype=sequence.dtype, device=sequence.device
    )
    channel_scale = torch.sqrt(channel_weights)[None]
    for step in joint_indices:
        predicted_joint = (
            (sequence[:, step] - reference_mean[step][None])
            / reference_scale[None]
        ) * channel_scale
        reference_joint = (
            (reference[:, step] - reference_mean[step][None])
            / reference_scale[None]
        ) * channel_scale
        predicted_projection = predicted_joint @ joint_projections
        reference_projection = reference_joint @ joint_projections
        joint_terms.append(
            (
                torch.quantile(
                    predicted_projection, joint_probabilities, dim=0
                )
                - torch.quantile(
                    reference_projection, joint_probabilities, dim=0
                )
            ).square().mean()
        )
    joint_loss = torch.stack(joint_terms).mean()

    normalized_control = controls / AMPLITUDE_LIMIT
    energy = normalized_control.square().mean()
    first = normalized_control[:, 1:] - normalized_control[:, :-1]
    second = (
        normalized_control[:, 2:]
        - 2.0 * normalized_control[:, 1:-1]
        + normalized_control[:, :-2]
    )
    smoothness = first.square().mean()
    curvature = second.square().mean()
    slew_excess = torch.relu(
        first.abs() - float(maximum_first_difference) / AMPLITUDE_LIMIT
    ).square().mean()
    curvature_excess = torch.relu(
        second.abs() - float(maximum_second_difference) / AMPLITUDE_LIMIT
    ).square().mean()
    saturation = torch.relu(
        commands.abs() / AMPLITUDE_LIMIT - 0.95
    ).square().mean()
    gain_regularization = (
        actor.mean_gain_delta.square().mean()
        + actor.deviation_gain_delta.square().mean()
        + 0.1 * actor.local_deviation_gain_logits.square().mean()
    )
    markov_regularization = mean_loss.new_zeros(())
    for module in (
        actor.common_markov_residual,
        actor.deviation_markov_residual,
    ):
        if module is not None:
            markov_regularization = markov_regularization + sum(
                parameter.square().mean() for parameter in module.parameters()
            )
    gain_regularization = gain_regularization + 0.01 * markov_regularization
    if objective_mode == "balanced":
        mean_weight = 5.0
        log_variance_weight = 3.0
        worst_log_variance_weight = 2.0
        quantile_weight = 5.0
        worst_quantile_weight = 3.0
        no_harm_weight = 10.0
    elif objective_mode == "covariance":
        mean_weight = 10.0
        log_variance_weight = 10.0
        worst_log_variance_weight = 5.0
        quantile_weight = 8.0
        worst_quantile_weight = 5.0
        no_harm_weight = 20.0
    else:
        raise ValueError(f"unknown objective_mode: {objective_mode}")
    total = (
        mean_weight * mean_loss
        + log_variance_weight * log_variance_loss
        + worst_log_variance_weight * worst_log_variance_loss
        + quantile_weight * quantile_loss
        + worst_quantile_weight * worst_quantile_loss
        + no_harm_weight * no_harm_mean_loss
        + no_harm_weight * no_harm_quantile_loss
        + 2.0 * occupancy_loss
        + 2.0 * tube_loss
        + 0.50 * joint_loss
        + 0.01 * energy
        + float(smoothness_weight) * smoothness
        + float(curvature_weight) * curvature
        + float(slew_barrier_weight) * slew_excess
        + float(curvature_barrier_weight) * curvature_excess
        + 1.00 * saturation
    )
    if include_parameter_regularization:
        total = total + 1e-3 * gain_regularization
    terms = {
        "loss": total,
        "mean": mean_loss,
        "log_variance": log_variance_loss,
        "worst_log_variance": worst_log_variance_loss,
        "quantile": quantile_loss,
        "worst_quantile": worst_quantile_loss,
        "no_harm_mean": no_harm_mean_loss,
        "no_harm_quantile": no_harm_quantile_loss,
        "occupancy": occupancy_loss,
        "tube": tube_loss,
        "joint": joint_loss,
        "energy": energy,
        "smoothness": smoothness,
        "curvature": curvature,
        "slew_excess": slew_excess,
        "curvature_excess": curvature_excess,
        "saturation": saturation,
        "gain_regularization": gain_regularization,
    }
    return total, terms


def train_model(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation",
        choices=("full", "no_wgan", "no_graph_spread", "no_deviation"),
        default="full",
        help="Matched-budget component ablation; full reproduces the complete model.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--validation-every", type=int, default=4)
    parser.add_argument("--critic-pretrain-steps", type=int, default=24)
    parser.add_argument("--critic-pretrain-banks", type=int, default=3)
    parser.add_argument("--n-critic", type=int, default=3)
    parser.add_argument("--actor-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--critic-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--adversarial-weight", type=float, default=0.50)
    parser.add_argument("--anchor-weight", type=float, default=0.20)
    parser.add_argument("--gradient-penalty", type=float, default=10.0)
    parser.add_argument("--critic-drift", type=float, default=1.0e-3)
    parser.add_argument(
        "--output-tag", default="seed20261011_adv050_anchor020"
    )
    args = parser.parse_args(argv)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if min(
        args.epochs,
        args.validation_every,
        args.critic_pretrain_steps,
        args.critic_pretrain_banks,
        args.n_critic,
    ) < 1:
        raise ValueError("all training loop counts must be positive")
    if min(args.actor_learning_rate, args.critic_learning_rate) <= 0.0:
        raise ValueError("learning rates must be positive")
    if min(
        args.adversarial_weight,
        args.anchor_weight,
        args.gradient_penalty,
        args.critic_drift,
    ) < 0.0:
        raise ValueError("loss weights must be non-negative")
    if any(character in args.output_tag for character in "\\/:"):
        raise ValueError("output tag must be a safe directory name")

    use_wgan = args.ablation != "no_wgan"
    effective_adversarial_weight = (
        float(args.adversarial_weight) if use_wgan else 0.0
    )
    graph_diffusion_time = (
        0.0 if args.ablation == "no_graph_spread" else GRAPH_DIFFUSION_TIME
    )
    deviation_feedback_scale = (
        0.0 if args.ablation == "no_deviation" else 1.0
    )

    output = OUTPUT_ROOT / args.output_tag
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    locked_hashes_before = {
        "original_actor_sha256": sha256_file(ORIGINAL_ACTOR),
        "original_rollout_sha256": sha256_file(ORIGINAL_ROLLOUT),
        "original_summary_sha256": sha256_file(ORIGINAL_SUMMARY),
    }


    synthesis = np.load(SYNTHESIS_CONTRACT)
    forbidden = {
        key
        for key in synthesis.files
        if "future" in key.lower() or "observed" in key.lower()
    }
    if forbidden:
        raise RuntimeError(f"prefix-only contract contains forbidden keys: {forbidden}")
    expected_hash = str(synthesis["model_sha256"][0])
    model_hash = sha256_file(MODEL_PATH)
    if model_hash != expected_hash:
        raise RuntimeError("frozen Part-II model hash changed")
    load_part2_model_definitions()
    model = joblib.load(MODEL_PATH)
    reference_np = np.asarray(
        synthesis["run01_reference_fit_pool_scaled"], dtype=np.float64
    ).reshape(30, HORIZON, 36)
    fit_reference = torch.as_tensor(reference_np[:15], dtype=torch.float64)
    validation_reference = torch.as_tensor(reference_np[15:], dtype=torch.float64)
    past_context = np.asarray(synthesis["past_context_scaled"], dtype=np.float64)
    selected_indices = np.asarray(synthesis["selected_indices"], dtype=np.int64)

    world = TorchGraphRCSDE(
        model,
        selected_indices,
        FS,
        control_graph_diffusion_time=graph_diffusion_time,
        preserve_physical_control_residual=True,
        dtype=torch.float64,
        device="cpu",
    )
    adapter = FrozenGraphRCMarkovAdapter(
        world, control_step_scale=CONTROL_STEP_SCALE, dtype=torch.float64
    )
    initial = adapter.initial_state_from_context(past_context)
    stepper = FrozenIctalGraphRCBatchStepper(
        world, adapter, diffusion_scale=DIFFUSION_SCALE
    )
    reference_mean, reference_variance, reference_scale, channel_weights = (
        reference_statistics(fit_reference)
    )
    markov_center, markov_scale = build_markov_normalization(
        adapter, initial, fit_reference
    )
    original_actor = build_teacher(
        stepper,
        reference_mean,
        reference_variance,
        reference_scale,
        selected_indices,
        markov_center,
        markov_scale,
    )
    actor = build_teacher(
        stepper,
        reference_mean,
        reference_variance,
        reference_scale,
        selected_indices,
        markov_center,
        markov_scale,
    )
    # Checkpoint buffers encode the original graph map.  Recompute the sole
    # plant-dependent local inverse for the graph-spread ablation, while all
    # trainable weights retain the identical locked initialization.
    actuator_rows = torch.arange(
        stepper.actuator_dim, dtype=torch.long, device=stepper.device
    )
    direct_effect = (
        stepper.adapter.control_step_scale
        * stepper.adapter.control_channel_map[
            actuator_rows, actor.actuated_channel_indices
        ]
    )
    if torch.any(direct_effect.abs() < 1.0e-6):
        raise RuntimeError("ablation actuator has negligible direct effect")
    for current_actor in (original_actor, actor):
        current_actor.local_inverse_effect.copy_(
            1.0 / (current_actor.actuator_alpha * direct_effect)
        )
        current_actor.deviation_feedback_scale = deviation_feedback_scale
    set_requires_grad(actor, True)
    if args.ablation == "no_deviation":
        actor.deviation_gain_delta.requires_grad_(False)
        actor.local_deviation_gain_logits.requires_grad_(False)
        for module in (
            actor.deviation_residual,
            actor.deviation_markov_residual,
        ):
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
    actor.train()
    critic = TimeConditionedWassersteinCritic(
        fit_reference.reshape(-1, 36).mean(dim=0),
        reference_scale,
        horizon_samples=HORIZON,
        hidden_size=128,
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=float(args.critic_learning_rate),
        betas=(0.0, 0.9),
    )
    actor_optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=float(args.actor_learning_rate),
        weight_decay=1.0e-5,
    )

    rng = np.random.default_rng(SEED)
    projections = rng.normal(size=(36, 16))
    projections /= np.maximum(np.linalg.norm(projections, axis=0), 1.0e-12)
    joint_projections = torch.as_tensor(projections, dtype=torch.float64)
    wgan_indices = tuple(range(15, HORIZON, 16))
    anchor_indices = tuple(range(0, HORIZON, 16))
    # Reuse the locked Actor's original run-01 validation noise bank so epoch
    # zero has exact methodological parity with the published baseline.
    validation_noise = antithetic_noise(VALIDATION_SEED, stepper.q)
    with torch.no_grad():
        validation_baseline = uncontrolled_particle_rollout(
            stepper, initial, validation_noise
        )[:, 1:]

    # Pretrain only the critic against locked-Actor particles.  In the
    # no-WGAN arm the critic is retained only as an unused serialization
    # placeholder and receives neither data nor optimizer steps.
    pretrain_banks = []
    if use_wgan:
        with torch.no_grad():
            for bank in range(int(args.critic_pretrain_banks)):
                pretrain_banks.append(
                    empirical_fp_rollout(
                        stepper,
                        original_actor,
                        initial,
                        antithetic_noise(SEED + 101 * (bank + 1), stepper.q),
                    ).scaled[:, 1:].detach()
                )
    pretrain_history: list[dict[str, float]] = []
    for critic_step in range(
        int(args.critic_pretrain_steps) if use_wgan else 0
    ):
        critic.train()
        critic_optimizer.zero_grad(set_to_none=True)
        gp_rng = torch.Generator(device="cpu")
        gp_rng.manual_seed(SEED + 50000 + critic_step)
        audit = balanced_wgan_gp_loss(
            critic,
            pretrain_banks[critic_step % len(pretrain_banks)],
            fit_reference,
            wgan_indices,
            gradient_penalty_weight=float(args.gradient_penalty),
            critic_drift_weight=float(args.critic_drift),
            generator=gp_rng,
        )
        audit.loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
        critic_optimizer.step()
        pretrain_history.append(
            {
                "stage": "critic_pretrain",
                "step": float(critic_step + 1),
                "critic_loss": float(audit.loss.detach()),
                "critic_estimate": float(audit.estimate.detach()),
                "critic_gp": float(audit.gradient_penalty.detach()),
                "critic_mean_gradient_norm": float(audit.mean_gradient_norm.detach()),
                "critic_parameter_gradient_norm": float(
                    torch.as_tensor(gradient).detach()
                ),
            }
        )

    def validate(current_actor) -> dict[str, Any]:
        current_actor.eval()
        with torch.no_grad():
            rollout = empirical_fp_rollout(
                stepper, current_actor, initial, validation_noise
            )
            law, terms = law_objective(
                rollout.scaled[:, 1:],
                rollout.controls,
                rollout.commands,
                validation_reference,
                channel_weights,
                reference_scale,
                joint_projections,
                current_actor,
                include_parameter_regularization=False,
                baseline_sequence=validation_baseline,
                objective_mode="covariance",
                smoothness_weight=0.20,
                curvature_weight=0.10,
                slew_barrier_weight=0.0,
                curvature_barrier_weight=0.0,
                maximum_first_difference=0.35,
                maximum_second_difference=0.50,
            )
        controlled_np = rollout.scaled[:, 1:].detach().cpu().numpy()
        reference_validation_np = validation_reference.cpu().numpy()
        time_w1 = channelwise_time_w1(controlled_np, reference_validation_np)
        occupation_w1 = channelwise_occupation_w1(
            controlled_np, reference_validation_np
        )
        if use_wgan:
            estimate, _ = actor_wasserstein_loss(
                critic,
                rollout.scaled[:, 1:].detach(),
                validation_reference,
                wgan_indices,
            )
            critic_estimate = float(estimate.detach())
        else:
            critic_estimate = float("nan")
        return {
            "rollout": rollout,
            "law": float(law.detach()),
            "law_terms": {
                key: float(value.detach()) for key, value in terms.items()
            },
            "mean_time_w1": float(time_w1.mean()),
            "mean_occupation_w1": float(occupation_w1.mean()),
            "distribution_score": float(time_w1.mean() + occupation_w1.mean()),
            "critic_estimate": critic_estimate,
        }

    # Epoch zero is reported as the locked-Actor baseline but is not eligible
    # to masquerade as a WGAN-updated checkpoint.
    initial_validation = validate(actor)
    initial_law = float(initial_validation["law"])
    initial_time_w1 = float(initial_validation["mean_time_w1"])
    initial_occupation_w1 = float(initial_validation["mean_occupation_w1"])
    history: list[dict[str, float]] = [
        {
            "stage": "actor_wgan",
            "epoch": 0.0,
            "validation_law": initial_validation["law"],
            "validation_mean_time_w1": initial_validation["mean_time_w1"],
            "validation_mean_occupation_w1": initial_validation[
                "mean_occupation_w1"
            ],
            "validation_distribution_score": initial_validation[
                "distribution_score"
            ],
            "validation_selection_score": 1.0,
            "validation_critic_estimate": initial_validation["critic_estimate"],
            "eligible_updated_checkpoint": 0.0,
        }
    ]
    best_score = float("inf")
    best_law = float("inf")
    best_epoch = -1
    best_actor_state = None
    best_critic_state = None
    fallback_score = float("inf")
    fallback_epoch = -1
    fallback_law = float("inf")
    fallback_actor_state = None
    fallback_critic_state = None

    for epoch in range(int(args.epochs)):
        noise = antithetic_noise(SEED + 1009 * (epoch + 1), stepper.q)

        # Critic step: controlled particles are detached, so this phase cannot
        # update the Actor through the plant.
        critic_audit = None
        if use_wgan:
            actor.eval()
            with torch.no_grad():
                detached = empirical_fp_rollout(
                    stepper, actor, initial, noise
                ).scaled[:, 1:].detach()
            set_requires_grad(actor, False)
            set_requires_grad(critic, True)
            critic.train()
            for critic_step in range(int(args.n_critic)):
                critic_optimizer.zero_grad(set_to_none=True)
                gp_rng = torch.Generator(device="cpu")
                gp_rng.manual_seed(
                    SEED + 100000 * (epoch + 1) + critic_step
                )
                critic_audit = balanced_wgan_gp_loss(
                    critic,
                    detached,
                    fit_reference,
                    wgan_indices,
                    gradient_penalty_weight=float(args.gradient_penalty),
                    critic_drift_weight=float(args.critic_drift),
                    generator=gp_rng,
                )
                critic_audit.loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
                critic_optimizer.step()
            assert critic_audit is not None

        # Actor step: Critic weights are frozen, but its derivative with
        # respect to controlled states remains in the computation graph.
        if use_wgan:
            set_requires_grad(critic, False)
        set_requires_grad(actor, True)
        if args.ablation == "no_deviation":
            actor.deviation_gain_delta.requires_grad_(False)
            actor.local_deviation_gain_logits.requires_grad_(False)
            for module in (
                actor.deviation_residual,
                actor.deviation_markov_residual,
            ):
                if module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)
        actor.train()
        actor_optimizer.zero_grad(set_to_none=True)
        rollout = empirical_fp_rollout(stepper, actor, initial, noise)
        with torch.no_grad():
            train_baseline = uncontrolled_particle_rollout(
                stepper, initial, noise
            )[:, 1:]
        law_loss, law_terms = law_objective(
            rollout.scaled[:, 1:],
            rollout.controls,
            rollout.commands,
            fit_reference,
            channel_weights,
            reference_scale,
            joint_projections,
            actor,
            include_parameter_regularization=True,
            baseline_sequence=train_baseline,
            objective_mode="covariance",
            smoothness_weight=0.20,
            curvature_weight=0.10,
            slew_barrier_weight=0.0,
            curvature_barrier_weight=0.0,
            maximum_first_difference=0.35,
            maximum_second_difference=0.50,
        )
        if use_wgan:
            adversarial, signed_estimate = actor_wasserstein_loss(
                critic, rollout.scaled[:, 1:], fit_reference, wgan_indices
            )
        else:
            adversarial = law_loss.new_zeros(())
            signed_estimate = law_loss.new_zeros(())
        anchor = actor_action_anchor_loss(
            actor,
            original_actor,
            rollout,
            anchor_indices,
            amplitude_limit=AMPLITUDE_LIMIT,
        )
        total = (
            law_loss
            + effective_adversarial_weight * adversarial
            + float(args.anchor_weight) * anchor
        )
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite Actor loss at epoch {epoch + 1}")
        total.backward()
        actor_gradient = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        actor_optimizer.step()

        row: dict[str, float] = {
            "stage": "actor_wgan",
            "epoch": float(epoch + 1),
            "train_total": float(total.detach()),
            "train_law": float(law_loss.detach()),
            "train_adversarial": float(adversarial.detach()),
            "train_signed_critic_estimate": float(signed_estimate.detach()),
            "train_action_anchor": float(anchor.detach()),
            "critic_loss": (
                float(critic_audit.loss.detach())
                if critic_audit is not None
                else float("nan")
            ),
            "critic_estimate": (
                float(critic_audit.estimate.detach())
                if critic_audit is not None
                else float("nan")
            ),
            "critic_gp": (
                float(critic_audit.gradient_penalty.detach())
                if critic_audit is not None
                else float("nan")
            ),
            "critic_mean_gradient_norm": (
                float(critic_audit.mean_gradient_norm.detach())
                if critic_audit is not None
                else float("nan")
            ),
            "actor_parameter_gradient_norm": float(
                torch.as_tensor(actor_gradient).detach()
            ),
            "eligible_updated_checkpoint": 1.0,
        }
        for key, value in law_terms.items():
            row[f"train_law_{key}"] = float(value.detach())

        should_validate = (
            epoch == 0
            or (epoch + 1) % int(args.validation_every) == 0
            or epoch + 1 == int(args.epochs)
        )
        if should_validate:
            validation = validate(actor)
            row.update(
                {
                    "validation_law": validation["law"],
                    "validation_mean_time_w1": validation["mean_time_w1"],
                    "validation_mean_occupation_w1": validation[
                        "mean_occupation_w1"
                    ],
                    "validation_distribution_score": validation[
                        "distribution_score"
                    ],
                    "validation_critic_estimate": validation["critic_estimate"],
                }
            )
            score = float(
                0.50 * validation["law"] / max(initial_law, 1.0e-12)
                + 0.25
                * validation["mean_time_w1"]
                / max(initial_time_w1, 1.0e-12)
                + 0.25
                * validation["mean_occupation_w1"]
                / max(initial_occupation_w1, 1.0e-12)
            )
            law_value = float(validation["law"])
            noninferior = law_value <= 1.02 * initial_law
            row["validation_selection_score"] = score
            row["validation_law_noninferior"] = float(noninferior)
            if score < fallback_score or (
                np.isclose(score, fallback_score) and law_value < fallback_law
            ):
                fallback_score = score
                fallback_law = law_value
                fallback_epoch = epoch + 1
                fallback_actor_state = copy.deepcopy(actor.state_dict())
                fallback_critic_state = copy.deepcopy(critic.state_dict())
            if noninferior and (score < best_score or (
                np.isclose(score, best_score) and law_value < best_law
            )):
                best_score = score
                best_law = law_value
                best_epoch = epoch + 1
                best_actor_state = copy.deepcopy(actor.state_dict())
                best_critic_state = copy.deepcopy(critic.state_dict())
            print(
                "epoch %d/%d total=%.4f valW=%.4f valLaw=%.4f gpNorm=%.3f"
                % (
                    epoch + 1,
                    args.epochs,
                    float(total.detach()),
                    score,
                    law_value,
                    (
                        float(critic_audit.mean_gradient_norm.detach())
                        if critic_audit is not None
                        else float("nan")
                    ),
                ),
                flush=True,
            )
        history.append(row)

    selected_from_noninferior_pool = best_actor_state is not None
    if not selected_from_noninferior_pool:
        best_score = fallback_score
        best_law = fallback_law
        best_epoch = fallback_epoch
        best_actor_state = fallback_actor_state
        best_critic_state = fallback_critic_state
    if best_actor_state is None or best_epoch < 1:
        raise RuntimeError("no updated validation checkpoint was selected")
    if use_wgan and best_critic_state is None:
        raise RuntimeError("no WGAN critic checkpoint was selected")
    actor.load_state_dict(best_actor_state, strict=True)
    if best_critic_state is not None:
        critic.load_state_dict(best_critic_state, strict=True)
    actor.eval()
    critic.eval()
    selected_validation = validate(actor)
    for row in history:
        row["selected_checkpoint"] = float(
            int(row["epoch"]) == int(best_epoch)
        )
    pd.DataFrame(pretrain_history + history).to_csv(
        output / "training_history.csv", index=False, encoding="utf-8-sig"
    )

    checkpoint = {
        "actor_state_dict": best_actor_state,
        "critic_state_dict": best_critic_state,
        "best_epoch": best_epoch,
        "best_validation_selection_score": best_score,
        "best_validation_law": best_law,
        "selected_validation": {
            key: value
            for key, value in selected_validation.items()
            if key not in {"rollout", "law_terms"}
        },
        "initial_validation": {
            key: value
            for key, value in initial_validation.items()
            if key not in {"rollout", "law_terms"}
        },
        "model_sha256": model_hash,
        "original_actor_sha256": locked_hashes_before[
            "original_actor_sha256"
        ],
        "critic_hidden_size": 128,
        "training_arguments": vars(args),
        "ablation_contract": {
            "variant": args.ablation,
            "wgan_coupled": use_wgan,
            "control_graph_diffusion_time": graph_diffusion_time,
            "deviation_feedback_scale": deviation_feedback_scale,
            "effective_adversarial_weight": effective_adversarial_weight,
            "seed": SEED,
        },
    }
    torch.save(checkpoint, output / "frozen_actor_wgan.pt")
    locked_hashes_after = {
        "original_actor_sha256": sha256_file(ORIGINAL_ACTOR),
        "original_rollout_sha256": sha256_file(ORIGINAL_ROLLOUT),
        "original_summary_sha256": sha256_file(ORIGINAL_SUMMARY),
    }
    if locked_hashes_after != locked_hashes_before:
        raise RuntimeError("an original baseline artifact changed during training")
    summary = {
        "status": "prefix_only_matched_component_ablation",
        "ablation_variant": args.ablation,
        "wgan_coupled": use_wgan,
        "control_graph_diffusion_time": graph_diffusion_time,
        "deviation_feedback_scale": deviation_feedback_scale,
        "effective_adversarial_weight": effective_adversarial_weight,
        "uses_recorded_run02_future": False,
        "network_count": 2 if use_wgan else 1,
        "network_roles": (
            [
                "locked-architecture structured 13-node mean-field Actor, fine-tuned",
                "time-conditioned 36-channel Wasserstein Kantorovich critic",
            ]
            if use_wgan
            else [
                "locked-architecture structured 13-node mean-field Actor, fine-tuned"
            ]
        ),
        "not_an_hjb_value_network": True,
        "fp_solver": "empirical particle push-forward through frozen Graph-RC-SDE",
        "selection_rule": (
            "among checkpoints with held-out legacy-law no more than 2% above "
            "epoch zero, minimize 0.50*(law/law0)+0.25*(time-W1/time-W10)+"
            "0.25*(occupation-W1/occupation-W10); epoch zero is reported but "
            "ineligible as an updated candidate"
        ),
        "selected_from_law_noninferior_pool": selected_from_noninferior_pool,
        "best_epoch": best_epoch,
        "initial_validation_distribution_score": initial_validation[
            "distribution_score"
        ],
        "best_validation_selection_score": best_score,
        "best_validation_law": best_law,
        "selected_validation_mean_time_w1": selected_validation[
            "mean_time_w1"
        ],
        "selected_validation_mean_occupation_w1": selected_validation[
            "mean_occupation_w1"
        ],
        "validation_selection_relative_change": best_score - 1.0,
        "passes_two_percent_validation_law_noninferiority": best_law
        <= 1.02 * initial_law,
        "model_sha256": model_hash,
        "locked_baseline_hashes_before": locked_hashes_before,
        "locked_baseline_hashes_after": locked_hashes_after,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_guardrail": (
            "independent development candidate only; it may be compared with, "
            "but must not overwrite or relabel, the locked original Actor"
        ),
    }
    (output / "training_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False))
    return 0


def evaluate_model(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-tag", default="seed20261011_adv050_anchor020"
    )
    parser.add_argument("--evaluation-tag", default="run02_development_evaluation")
    args = parser.parse_args(argv)
    if any(character in args.candidate_tag for character in "\\/:"):
        raise ValueError("candidate tag must be safe")
    if any(character in args.evaluation_tag for character in "\\/:"):
        raise ValueError("evaluation tag must be safe")
    candidate_dir = CANDIDATE_ROOT / args.candidate_tag
    checkpoint_path = candidate_dir / "frozen_actor_wgan.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"candidate checkpoint not found: {checkpoint_path}")
    output = candidate_dir / args.evaluation_tag
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    original_summary = json.loads(ORIGINAL_SUMMARY.read_text(encoding="utf-8"))

    synthesis = np.load(SYNTHESIS_CONTRACT)
    model_hash = sha256_file(MODEL_PATH)
    original_actor_hash = sha256_file(ORIGINAL_ACTOR)
    if model_hash != str(checkpoint["model_sha256"]):
        raise RuntimeError("candidate/frozen-model hash mismatch")
    if model_hash != str(synthesis["model_sha256"][0]):
        raise RuntimeError("synthesis/frozen-model hash mismatch")
    if original_actor_hash != str(checkpoint["original_actor_sha256"]):
        raise RuntimeError("the locked original Actor changed")

    load_part2_model_definitions()
    model = joblib.load(MODEL_PATH)
    reference_np = np.asarray(
        synthesis["run01_reference_fit_pool_scaled"], dtype=np.float64
    ).reshape(30, HORIZON, 36)
    fit_reference = torch.as_tensor(reference_np[:15], dtype=torch.float64)
    validation_reference = torch.as_tensor(reference_np[15:], dtype=torch.float64)
    validation_np = validation_reference.numpy()
    past_context = np.asarray(synthesis["past_context_scaled"], dtype=np.float64)
    selected_indices = np.asarray(synthesis["selected_indices"], dtype=np.int64)
    representative = np.asarray(
        synthesis["representative_indices"], dtype=np.int64
    )
    channels = np.asarray(synthesis["channels"]).astype(str)

    world = TorchGraphRCSDE(
        model,
        selected_indices,
        FS,
        control_graph_diffusion_time=GRAPH_DIFFUSION_TIME,
        preserve_physical_control_residual=True,
        dtype=torch.float64,
        device="cpu",
    )
    adapter = FrozenGraphRCMarkovAdapter(
        world, control_step_scale=CONTROL_STEP_SCALE, dtype=torch.float64
    )
    initial = adapter.initial_state_from_context(past_context)
    stepper = FrozenIctalGraphRCBatchStepper(
        world, adapter, diffusion_scale=DIFFUSION_SCALE
    )
    reference_mean, reference_variance, reference_scale, _ = (
        reference_statistics(fit_reference)
    )
    markov_center, markov_scale = build_markov_normalization(
        adapter, initial, fit_reference
    )
    original_actor = build_teacher(
        stepper,
        reference_mean,
        reference_variance,
        reference_scale,
        selected_indices,
        markov_center,
        markov_scale,
    )
    candidate_actor = build_teacher(
        stepper,
        reference_mean,
        reference_variance,
        reference_scale,
        selected_indices,
        markov_center,
        markov_scale,
    )
    candidate_actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    critic = TimeConditionedWassersteinCritic(
        fit_reference.reshape(-1, 36).mean(dim=0),
        reference_scale,
        horizon_samples=HORIZON,
        hidden_size=int(checkpoint["critic_hidden_size"]),
    )
    critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)
    original_actor.eval()
    candidate_actor.eval()
    critic.eval()
    set_requires_grad(original_actor, False)
    set_requires_grad(candidate_actor, False)
    set_requires_grad(critic, False)

    # Held-out run-01 audit occurs before the development future is opened.
    validation_noise = antithetic_noise(VALIDATION_SEED, stepper.q)
    with torch.no_grad():
        original_validation = empirical_fp_rollout(
            stepper, original_actor, initial, validation_noise
        ).scaled[:, 1:]
        candidate_validation = empirical_fp_rollout(
            stepper, candidate_actor, initial, validation_noise
        ).scaled[:, 1:]
    original_validation_np = original_validation.numpy()
    candidate_validation_np = candidate_validation.numpy()
    original_validation_time = per_time_w1(
        original_validation_np, validation_np
    )
    candidate_validation_time = per_time_w1(
        candidate_validation_np, validation_np
    )
    original_validation_occ = occupation_w1(
        original_validation_np, validation_np
    )
    candidate_validation_occ = occupation_w1(
        candidate_validation_np, validation_np
    )
    gp_rng = torch.Generator(device="cpu")
    gp_rng.manual_seed(SEED + 3)
    heldout_wgan = balanced_wgan_gp_loss(
        critic,
        candidate_validation.detach(),
        validation_reference,
        tuple(range(15, HORIZON, 16)),
        gradient_penalty_weight=float(
            checkpoint["training_arguments"]["gradient_penalty"]
        ),
        critic_drift_weight=float(
            checkpoint["training_arguments"]["critic_drift"]
        ),
        generator=gp_rng,
    )

    # The checkpoint is frozen.  Only now open the paired run-02 development
    # future and reconstruct its already-sealed Brownian innovations.
    sealed = np.load(SEALED_CALIBRATED)
    sealed_no_control = np.asarray(
        sealed["uncontrolled_scaled"][:, 1:], dtype=np.float64
    )
    normals, reconstruction = reconstruct_paired_normals(
        stepper, initial, sealed_no_control, tolerance=2.0e-6
    )
    with torch.no_grad():
        original_evaluation = paired_ictal_batch_rollout(
            stepper, original_actor, initial, normals
        )
        candidate_evaluation = paired_ictal_batch_rollout(
            stepper, candidate_actor, initial, normals
        )
    uncontrolled = original_evaluation.uncontrolled_scaled[:, 1:].numpy()
    original_controlled = original_evaluation.controlled_scaled[:, 1:].numpy()
    candidate_controlled = candidate_evaluation.controlled_scaled[:, 1:].numpy()
    original_controls = original_evaluation.controls.numpy()
    candidate_controls = candidate_evaluation.controls.numpy()
    original_commands = original_evaluation.commands.numpy()
    candidate_commands = candidate_evaluation.commands.numpy()

    saved_original = np.load(ORIGINAL_ROLLOUT)
    saved_uncontrolled = np.asarray(
        saved_original["uncontrolled_scaled"], dtype=np.float64
    )
    saved_original_controlled = np.asarray(
        saved_original["controlled_scaled"], dtype=np.float64
    )
    old_free_parity = float(np.max(np.abs(uncontrolled - saved_uncontrolled)))
    old_actor_parity = float(
        np.max(np.abs(original_controlled - saved_original_controlled))
    )
    candidate_free_parity = float(
        np.max(
            np.abs(
                candidate_evaluation.uncontrolled_scaled[:, 1:].numpy()
                - saved_uncontrolled
            )
        )
    )
    if max(old_free_parity, old_actor_parity, candidate_free_parity) > 1.0e-6:
        raise RuntimeError("old/candidate paired rollout parity gate failed")

    uncontrolled_time = per_time_w1(uncontrolled, validation_np)
    original_time = per_time_w1(original_controlled, validation_np)
    candidate_time = per_time_w1(candidate_controlled, validation_np)
    uncontrolled_occ = occupation_w1(uncontrolled, validation_np)
    original_occ = occupation_w1(original_controlled, validation_np)
    candidate_occ = occupation_w1(candidate_controlled, validation_np)
    legal = np.load(LEGAL_EVALUATION)
    observed = np.asarray(
        legal["observed_evaluation_only_scaled"], dtype=np.float64
    )
    prediction_occ = np.asarray(
        [
            wasserstein_distance(
                uncontrolled[:, :, channel].reshape(-1), observed[:, channel]
            )
            for channel in range(36)
        ]
    )
    control_channel_map = (
        CONTROL_STEP_SCALE * adapter.control_channel_map.detach().cpu().numpy()
    )
    original_effective_control = original_controls @ control_channel_map
    candidate_effective_control = candidate_controls @ control_channel_map

    representatives = {}
    for index in representative:
        representatives[channels[index]] = {
            "uncontrolled_time_w1": float(uncontrolled_time[index]),
            "original_time_w1": float(original_time[index]),
            "candidate_time_w1": float(candidate_time[index]),
            "uncontrolled_occupation_w1": float(uncontrolled_occ[index]),
            "original_occupation_w1": float(original_occ[index]),
            "candidate_occupation_w1": float(candidate_occ[index]),
        }
    summary = {
        "status": "frozen_actor_wgan_run02_development_evaluation",
        "uses_run02_future_for_training_or_selection": False,
        "candidate_only_original_not_replaced": True,
        "model_sha256": model_hash,
        "candidate_checkpoint_sha256": sha256_file(checkpoint_path),
        "original_actor_sha256": original_actor_hash,
        "original_rollout_sha256": sha256_file(ORIGINAL_ROLLOUT),
        "parity": {
            "old_free_max_abs_error": old_free_parity,
            "old_actor_max_abs_error": old_actor_parity,
            "candidate_free_max_abs_error": candidate_free_parity,
            "passes_1e_minus_6": True,
        },
        "prediction_occupation_w1_mean": float(prediction_occ.mean()),
        "heldout_run01": {
            "original_mean_time_w1": float(original_validation_time.mean()),
            "candidate_mean_time_w1": float(candidate_validation_time.mean()),
            "original_mean_occupation_w1": float(original_validation_occ.mean()),
            "candidate_mean_occupation_w1": float(candidate_validation_occ.mean()),
            "critic_wasserstein_estimate": float(heldout_wgan.estimate.detach()),
            "critic_gradient_penalty": float(
                heldout_wgan.gradient_penalty.detach()
            ),
            "critic_mean_gradient_norm": float(
                heldout_wgan.mean_gradient_norm.detach()
            ),
        },
        "run02_control": {
            "mean_uncontrolled_time_w1": float(uncontrolled_time.mean()),
            "mean_original_time_w1": float(original_time.mean()),
            "mean_candidate_time_w1": float(candidate_time.mean()),
            "candidate_relative_change_vs_original_time": float(
                candidate_time.mean() / original_time.mean() - 1.0
            ),
            "mean_uncontrolled_occupation_w1": float(uncontrolled_occ.mean()),
            "mean_original_occupation_w1": float(original_occ.mean()),
            "mean_candidate_occupation_w1": float(candidate_occ.mean()),
            "candidate_relative_change_vs_original_occupation": float(
                candidate_occ.mean() / original_occ.mean() - 1.0
            ),
            "candidate_channels_better_than_original_time": int(
                np.sum(candidate_time < original_time)
            ),
            "candidate_channels_better_than_original_occupation": int(
                np.sum(candidate_occ < original_occ)
            ),
            "original_channels_better_than_free_time": int(
                np.sum(original_time < uncontrolled_time)
            ),
            "candidate_channels_better_than_free_time": int(
                np.sum(candidate_time < uncontrolled_time)
            ),
            "original_channels_better_than_free_occupation": int(
                np.sum(original_occ < uncontrolled_occ)
            ),
            "candidate_channels_better_than_free_occupation": int(
                np.sum(candidate_occ < uncontrolled_occ)
            ),
            "representatives": representatives,
        },
        "within_patient_descriptive_bootstrap": {
            "time_w1": paired_channel_bootstrap(
                candidate_time, original_time, seed=SEED + 11
            ),
            "occupation_w1": paired_channel_bootstrap(
                candidate_occ, original_occ, seed=SEED + 12
            ),
        },
        "original_control_diagnostics": control_diagnostics(original_controls),
        "candidate_control_diagnostics": control_diagnostics(candidate_controls),
        "noise_reconstruction": reconstruction,
        "baseline_summary_consistency": {
            "saved_original_mean_time_w1": original_summary[
                "sealed_density_control"
            ]["mean_controlled_time_resolved_w1"],
            "recomputed_original_mean_time_w1": float(original_time.mean()),
            "saved_original_mean_occupation_w1": original_summary[
                "sealed_density_control"
            ]["mean_controlled_occupation_w1"],
            "recomputed_original_mean_occupation_w1": float(original_occ.mean()),
        },
        "decision": (
            "retain both as separate development artifacts; the user has not "
            "authorized replacement of the locked original"
        ),
        "claim_guardrail": (
            "Actor+WGAN-GP with empirical particle FP; without a value network, "
            "Bellman residual and Hamiltonian stationarity this is not an HJB solver"
        ),
    }
    (output / "evaluation_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    np.savez_compressed(
        output / "paired_comparison.npz",
        observed_scaled=observed,
        uncontrolled_scaled=uncontrolled,
        original_controlled_scaled=original_controlled,
        candidate_controlled_scaled=candidate_controlled,
        reference_fit_scaled=fit_reference.numpy(),
        reference_validation_scaled=validation_np,
        original_controls=original_controls,
        candidate_controls=candidate_controls,
        original_commands=original_commands,
        candidate_commands=candidate_commands,
        original_effective_control=original_effective_control,
        candidate_effective_control=candidate_effective_control,
        time_w1_uncontrolled=uncontrolled_time,
        time_w1_original=original_time,
        time_w1_candidate=candidate_time,
        occupation_w1_uncontrolled=uncontrolled_occ,
        occupation_w1_original=original_occ,
        occupation_w1_candidate=candidate_occ,
        prediction_occupation_w1=prediction_occ,
        representative_indices=representative,
        selected_indices=selected_indices,
        channels=channels,
        sampling_rate_hz=np.asarray([FS]),
        fixed_particle_index=np.asarray([0], dtype=np.int64),
        fixed_reference_path_index=np.asarray([0], dtype=np.int64),
    )
    print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False))
    return 0


def evaluate_ablations(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-tag", default="ablation_full_seed20261011_v1"
    )
    parser.add_argument(
        "--no-wgan-tag", default="ablation_no_wgan_seed20261011_v1"
    )
    parser.add_argument(
        "--no-graph-tag", default="ablation_no_graph_spread_seed20261011_v1"
    )
    parser.add_argument(
        "--no-deviation-tag", default="ablation_no_deviation_seed20261011_v1"
    )
    parser.add_argument(
        "--output-tag", default="matched_ablation_run02_development_v1"
    )
    args = parser.parse_args(argv)
    for value in vars(args).values():
        if any(character in str(value) for character in "\\/:"):
            raise ValueError("tags must be safe directory names")

    tags = {
        "full": args.full_tag,
        "no_wgan": args.no_wgan_tag,
        "no_graph_spread": args.no_graph_tag,
        "no_deviation": args.no_deviation_tag,
    }
    output = ROOT / args.output_tag
    output.mkdir(parents=True, exist_ok=True)
    checkpoints: dict[str, dict[str, Any]] = {}
    checkpoint_paths: dict[str, Path] = {}
    for variant, tag in tags.items():
        path = ROOT / tag / "frozen_actor_wgan.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing {variant} checkpoint: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        contract = payload.get("ablation_contract", {})
        if contract.get("variant") != variant:
            raise RuntimeError(f"checkpoint/variant mismatch for {variant}")
        checkpoints[variant] = payload
        checkpoint_paths[variant] = path

    model_hash = sha256_file(MODEL)
    if any(str(payload["model_sha256"]) != model_hash for payload in checkpoints.values()):
        raise RuntimeError("an ablation checkpoint uses a different Part-II model")

    synthesis = np.load(SYNTHESIS)
    if str(synthesis["model_sha256"][0]) != model_hash:
        raise RuntimeError("synthesis contract and model hash differ")
    load_part2_model_definitions()
    model = joblib.load(MODEL)
    reference = np.asarray(
        synthesis["run01_reference_fit_pool_scaled"], dtype=np.float64
    ).reshape(30, HORIZON, 36)
    fit = torch.as_tensor(reference[:15], dtype=torch.float64)
    validation = np.asarray(reference[15:], dtype=np.float64)
    past_context = np.asarray(synthesis["past_context_scaled"], dtype=np.float64)
    selected = np.asarray(synthesis["selected_indices"], dtype=np.int64)
    representatives = np.asarray(
        synthesis["representative_indices"], dtype=np.int64
    )
    channels = np.asarray(synthesis["channels"]).astype(str)
    observed = np.asarray(
        np.load(LEGAL)["observed_evaluation_only_scaled"], dtype=np.float64
    )
    sealed_no_control = np.asarray(
        np.load(SEALED)["uncontrolled_scaled"][:, 1:], dtype=np.float64
    )

    projection_rng = np.random.default_rng(SEED + 77)
    projections = projection_rng.normal(size=(36, 64))
    projections /= np.maximum(np.linalg.norm(projections, axis=0), 1.0e-12)

    rollouts: dict[str, np.ndarray] = {}
    controls: dict[str, np.ndarray] = {}
    effective_controls: dict[str, np.ndarray] = {}
    reconstructions: dict[str, dict[str, Any]] = {}
    free_by_variant: dict[str, np.ndarray] = {}
    graph_maps: dict[str, np.ndarray] = {}

    for variant, payload in checkpoints.items():
        contract = payload["ablation_contract"]
        graph_time = float(contract["control_graph_diffusion_time"])
        world = TorchGraphRCSDE(
            model,
            selected,
            FS,
            control_graph_diffusion_time=graph_time,
            preserve_physical_control_residual=True,
            dtype=torch.float64,
            device="cpu",
        )
        adapter = FrozenGraphRCMarkovAdapter(
            world, control_step_scale=CONTROL_STEP_SCALE, dtype=torch.float64
        )
        initial = adapter.initial_state_from_context(past_context)
        stepper = FrozenIctalGraphRCBatchStepper(
            world, adapter, diffusion_scale=DIFFUSION_SCALE
        )
        reference_mean, reference_variance, reference_scale, _ = (
            reference_statistics(fit)
        )
        markov_center, markov_scale = build_markov_normalization(
            adapter, initial, fit
        )
        actor = build_teacher(
            stepper,
            reference_mean,
            reference_variance,
            reference_scale,
            selected,
            markov_center,
            markov_scale,
        )
        actor.load_state_dict(payload["actor_state_dict"], strict=True)
        actuator_rows = torch.arange(
            stepper.actuator_dim, dtype=torch.long, device=stepper.device
        )
        direct_effect = (
            stepper.adapter.control_step_scale
            * stepper.adapter.control_channel_map[
                actuator_rows, actor.actuated_channel_indices
            ]
        )
        actor.local_inverse_effect.copy_(
            1.0 / (actor.actuator_alpha * direct_effect)
        )
        actor.deviation_feedback_scale = float(
            contract["deviation_feedback_scale"]
        )
        actor.eval()
        set_requires_grad(actor, False)
        normals, reconstruction = reconstruct_paired_normals(
            stepper, initial, sealed_no_control, tolerance=2.0e-6
        )
        with torch.no_grad():
            result = paired_ictal_batch_rollout(
                stepper, actor, initial, normals
            )
        rollouts[variant] = result.controlled_scaled[:, 1:].cpu().numpy()
        controls[variant] = result.controls.cpu().numpy()
        free_by_variant[variant] = result.uncontrolled_scaled[:, 1:].cpu().numpy()
        graph_maps[variant] = (
            CONTROL_STEP_SCALE
            * adapter.control_channel_map.detach().cpu().numpy()
        )
        effective_controls[variant] = controls[variant] @ graph_maps[variant]
        reconstructions[variant] = reconstruction

    free = free_by_variant["full"]
    maximum_free_difference = max(
        float(np.max(np.abs(values - free)))
        for values in free_by_variant.values()
    )
    if maximum_free_difference > 1.0e-6:
        raise RuntimeError("control-map ablation changed the zero-control plant")
    unselected_indices = np.asarray(
        [index for index in range(36) if index not in set(selected)],
        dtype=np.int64,
    )
    no_graph_unselected_effect = float(
        np.max(
            np.abs(
                effective_controls["no_graph_spread"][:, :, unselected_indices]
            )
        )
    )
    if no_graph_unselected_effect > 1.0e-12:
        raise RuntimeError("no-graph arm has nonzero direct unselected-node input")
    no_deviation_particle_spread = float(
        np.max(np.std(controls["no_deviation"], axis=0))
    )
    if no_deviation_particle_spread > 1.0e-10:
        raise RuntimeError("no-deviation arm still emits particle-specific actions")

    selected_set = set(int(index) for index in selected)
    role_lookup = {
        int(representatives[0]): "selected SOZ",
        int(representatives[1]): "selected non-SOZ",
        int(representatives[2]): "unselected",
    }
    variants_with_free = {"free": free, **rollouts}
    metric_cache: dict[str, dict[str, Any]] = {}
    channel_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for variant, values in variants_with_free.items():
        time_values = per_time_w1(values, validation)
        occupation_values = occupation_w1(values, validation)
        joint_time, joint_occupation = sliced_w1(
            values, validation, projections
        )
        mean_rmse = float(
            np.sqrt(np.mean((values.mean(axis=0) - validation.mean(axis=0)) ** 2))
        )
        variance_error = float(
            np.mean(
                np.abs(
                    np.log(values.var(axis=0).clip(min=1.0e-5))
                    - np.log(validation.var(axis=0).clip(min=1.0e-5))
                )
            )
        )
        metric_cache[variant] = {
            "time": time_values,
            "occupation": occupation_values,
            "joint_time": joint_time,
            "joint_occupation": joint_occupation,
        }
        diagnostics = (
            {
                "control_rms": 0.0,
                "control_peak": 0.0,
                "control_energy": 0.0,
                "maximum_first_difference": 0.0,
                "maximum_second_difference": 0.0,
                "saturation_fraction": 0.0,
            }
            if variant == "free"
            else control_diagnostics(controls[variant])
        )
        summary_rows.append(
            {
                "variant": variant,
                "mean_time_w1": float(time_values.mean()),
                "mean_occupation_w1": float(occupation_values.mean()),
                "selected_time_w1": float(time_values[selected].mean()),
                "selected_occupation_w1": float(occupation_values[selected].mean()),
                "unselected_time_w1": float(
                    np.delete(time_values, selected).mean()
                ),
                "unselected_occupation_w1": float(
                    np.delete(occupation_values, selected).mean()
                ),
                "joint_time_sliced_w1": joint_time,
                "joint_occupation_sliced_w1": joint_occupation,
                "mean_path_rmse": mean_rmse,
                "log_variance_mae": variance_error,
                **diagnostics,
            }
        )
        for index, channel in enumerate(channels):
            channel_rows.append(
                {
                    "variant": variant,
                    "channel_index": index,
                    "channel": channel,
                    "selected": index in selected_set,
                    "representative_role": role_lookup.get(index, ""),
                    "time_w1": float(time_values[index]),
                    "occupation_w1": float(occupation_values[index]),
                }
            )

    summary_frame = pd.DataFrame(summary_rows)
    channel_frame = pd.DataFrame(channel_rows)
    free_row = summary_frame.loc[summary_frame.variant == "free"].iloc[0]
    for column in (
        "mean_time_w1",
        "mean_occupation_w1",
        "selected_time_w1",
        "selected_occupation_w1",
        "unselected_time_w1",
        "unselected_occupation_w1",
        "joint_time_sliced_w1",
        "joint_occupation_sliced_w1",
    ):
        summary_frame[f"{column}_recovery_percent"] = 100.0 * (
            1.0 - summary_frame[column] / float(free_row[column])
        )
    full_time = metric_cache["full"]["time"]
    full_occupation = metric_cache["full"]["occupation"]
    contrast_rows: list[dict[str, Any]] = []
    for variant in ("no_wgan", "no_graph_spread", "no_deviation"):
        for index, channel in enumerate(channels):
            contrast_rows.append(
                {
                    "ablation": variant,
                    "channel_index": index,
                    "channel": channel,
                    "selected": index in selected_set,
                    "delta_time_w1_vs_full": float(
                        metric_cache[variant]["time"][index] - full_time[index]
                    ),
                    "delta_occupation_w1_vs_full": float(
                        metric_cache[variant]["occupation"][index]
                        - full_occupation[index]
                    ),
                }
            )
    contrast_frame = pd.DataFrame(contrast_rows)

    summary_frame.to_csv(
        output / "ablation_summary.csv", index=False, encoding="utf-8-sig"
    )
    channel_frame.to_csv(
        output / "ablation_per_channel.csv", index=False, encoding="utf-8-sig"
    )
    contrast_frame.to_csv(
        output / "ablation_contrasts_vs_full.csv",
        index=False,
        encoding="utf-8-sig",
    )
    np.savez_compressed(
        output / "ablation_rollouts.npz",
        observed_scaled=observed,
        reference_validation_scaled=validation,
        free_scaled=free,
        full_scaled=rollouts["full"],
        no_wgan_scaled=rollouts["no_wgan"],
        no_graph_spread_scaled=rollouts["no_graph_spread"],
        no_deviation_scaled=rollouts["no_deviation"],
        full_controls=controls["full"],
        no_wgan_controls=controls["no_wgan"],
        no_graph_spread_controls=controls["no_graph_spread"],
        no_deviation_controls=controls["no_deviation"],
        full_effective_control=effective_controls["full"],
        no_wgan_effective_control=effective_controls["no_wgan"],
        no_graph_spread_effective_control=effective_controls["no_graph_spread"],
        no_deviation_effective_control=effective_controls["no_deviation"],
        selected_indices=selected,
        representative_indices=representatives,
        channels=channels,
        fixed_particle_index=np.asarray([0], dtype=np.int64),
        fixed_reference_path_index=np.asarray([0], dtype=np.int64),
        sampling_rate_hz=np.asarray([FS], dtype=np.float64),
    )

    invariant_arguments = (
        "epochs",
        "validation_every",
        "actor_learning_rate",
        "anchor_weight",
        "gradient_penalty",
        "critic_drift",
    )
    argument_audit = {
        key: {
            variant: checkpoints[variant]["training_arguments"].get(key)
            for variant in checkpoints
        }
        for key in invariant_arguments
    }
    invariant_pass = all(
        len({json.dumps(value, sort_keys=True) for value in values.values()}) == 1
        for values in argument_audit.values()
    )
    result = {
        "status": "matched_prefix_trained_ablation_run02_development_evaluation",
        "uses_run02_future_for_training_or_checkpoint_selection": False,
        "patient": "HUP060",
        "window_seconds": 1.0,
        "particles": 32,
        "reference_paths": 15,
        "model_sha256": model_hash,
        "checkpoint_sha256": {
            variant: sha256_file(path) for variant, path in checkpoint_paths.items()
        },
        "fairness_contract": {
            "same_prefix_training_pool": True,
            "same_heldout_run01_selection_pool": True,
            "same_locked_part2_model": True,
            "same_initial_markov_state": True,
            "same_run02_brownian_bank": True,
            "same_13_node_mask": True,
            "same_training_budget": invariant_pass,
            "zero_control_plant_max_abs_difference": maximum_free_difference,
            "no_graph_unselected_effective_input_max_abs": (
                no_graph_unselected_effect
            ),
            "no_deviation_particle_control_std_max": (
                no_deviation_particle_spread
            ),
            "argument_audit": argument_audit,
            "graph_ablation_scope": (
                "control input heat-kernel map only; Graph-RC drift and ictal "
                "state-dependent diffusion remain frozen"
            ),
        },
        "noise_reconstruction": reconstructions,
        "summary": summary_frame.to_dict(orient="records"),
        "interpretation_guardrail": (
            "HUP060 run-02 is a within-patient development ablation. Channels "
            "and numerical particles are not independent biological replicates."
        ),
    }
    (output / "evaluation_summary.json").write_text(
        json.dumps(json_ready(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(json_ready(result), indent=2, ensure_ascii=False))
    return 0

def export_source_data(argv: list[str] | None = None) -> int:
    """Synchronize the frozen Part-III evaluations used by Figs. 6--8.

    This command performs no training and no outcome-based selection.  It only
    copies the already frozen main evaluation and matched-ablation tables into
    the public ``output/part3/source_data`` locations read by the independent
    figure scripts.
    """

    parser = argparse.ArgumentParser(description=export_source_data.__doc__)
    parser.parse_args(argv)
    main_source = (
        OUTPUT_ROOT
        / "seed20261011_adv050_anchor020"
        / "run02_development_evaluation"
    )
    main_destination = (
        PROJECT / "output" / "part3" / "source_data" / "figures_06_08"
    )
    ablation_source = OUTPUT_ROOT / "matched_ablation_run02_development_v1"
    ablation_destination = (
        PROJECT / "output" / "part3" / "source_data" / "figure_07"
    )
    transfers = (
        (
            main_source,
            main_destination,
            (
                "paired_comparison.npz",
                "evaluation_summary.json",
            ),
        ),
        (
            OUTPUT_ROOT / "seed20261011_adv050_anchor020",
            main_destination,
            ("training_history.csv", "training_summary.json"),
        ),
        (
            ablation_source,
            ablation_destination,
            (
                "ablation_summary.csv",
                "ablation_per_channel.csv",
                "ablation_contrasts_vs_full.csv",
                "ablation_rollouts.npz",
                "evaluation_summary.json",
            ),
        ),
    )
    copied: list[dict[str, str]] = []
    for source_directory, destination_directory, names in transfers:
        destination_directory.mkdir(parents=True, exist_ok=True)
        for name in names:
            source = source_directory / name
            if not source.is_file():
                raise FileNotFoundError(f"missing frozen Part-III source: {source}")
            destination = destination_directory / name
            shutil.copy2(source, destination)
            copied.append(
                {
                    "path": destination.relative_to(PROJECT).as_posix(),
                    "sha256": sha256_file(destination),
                }
            )
    print(
        json.dumps(
            {"status": "source_data_synchronized", "files": copied},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(
            "Usage: python part3_model.py {train|evaluate|evaluate-ablations|source-data} [options]\n"
            "  train                prefix-only Actor--WGAN training\n"
            "  evaluate             freeze-then-open main-result evaluation\n"
            "  evaluate-ablations   paired matched-budget ablation evaluation\n"
            "  source-data          copy frozen Fig. 6--8 inputs to output/"
        )
        return 0
    command, command_args = args[0], args[1:]
    dispatch = {
        "train": train_model,
        "evaluate": evaluate_model,
        "evaluate-ablations": evaluate_ablations,
        "source-data": export_source_data,
    }
    if command not in dispatch:
        raise SystemExit(f"unknown command: {command}")
    return dispatch[command](command_args)


if __name__ == "__main__":
    raise SystemExit(main())
