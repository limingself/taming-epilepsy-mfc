from __future__ import annotations

from dataclasses import asdict, replace
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types
from typing import Any, Mapping, Sequence

from .contracts import deterministic_seed, fraction_to_quantile, sha256_file, sha256_json
from .data_model import (
    activate_canonical,
    array_sha256,
    assert_live_part2_model_identity,
    build_part1_network,
    build_part2_plant_adjacency,
    part3_context_block,
    get_run,
    load_development,
    load_network,
    reference_paths,
    save_network,
)
from .staging import atomic_json


PROTOCOL_ID = "HUP060_consistent_sparse_rerun_common_v1"
GATE_B_RELATIVE_REDUCTION_MIN = 0.10
GATE_B_TIME_W1_MAX = 0.35
GATE_B_OCCUPATION_W1_MAX = 0.25
GATE_B_MEAN_ERROR_MAX = 0.10
GATE_B_SYMMETRIC_SD_RATIO_MAX = 2.0


def analytical_candidate_eligible(
    *, protocol_gate_a_integrity_pass: bool, gate_c_energy_finite_pass: bool
) -> bool:
    """Return the preregistered analytical feasibility decision.

    Quantitative plant-fidelity distances are deliberately absent: the shared
    protocol preregisters static Gate-A integrity plus post-rollout Gate-C
    energy/finite safety as the feasibility prefilter.  Plant distances remain
    useful diagnostics but may never reject, rank, rescue, or replace an arm.
    """

    return bool(protocol_gate_a_integrity_pass and gate_c_energy_finite_pass)


def _purge_modules() -> None:
    for name in tuple(sys.modules):
        if name == "mfc_pipeline" or name.startswith("mfc_pipeline."):
            del sys.modules[name]
        if name == "multistep_drift_calibration":
            del sys.modules[name]


def activate_controller(config: Mapping[str, Any]) -> Any:
    """Load hash-pinned controller logic against the immutable D authority."""

    source = Path(str(config["adaptation_locks"]["controller_runner"][0])).resolve()
    controller_dir = source.parent
    expected_source = str(config["adaptation_locks"]["controller_runner"][1]).lower()
    if sha256_file(source) != expected_source:
        raise PermissionError("dimension-generalized controller code reference changed")
    canonical = Path(str(config["canonical_locks"]["part2_core"][0])).parents[1].resolve()
    _purge_modules()
    for path in (str(controller_dir), str(canonical)):
        sys.path[:] = [item for item in sys.path if os.path.normcase(item) != os.path.normcase(path)]
    sys.path.insert(0, str(canonical))
    sys.path.insert(0, str(controller_dir))
    name = "hup065_frozen_fresh_controller_impl"
    module = types.ModuleType(name)
    module.__file__ = str(source)
    module.__package__ = ""
    sys.modules[name] = module
    source_text = source.read_text(encoding="utf-8")
    needle = 'PIPELINE_ROOT = WORKSPACE / "taming-epilepsy-mfc-baseline"'
    replacement = f"PIPELINE_ROOT = Path({str(canonical)!r})"
    if source_text.count(needle) != 1:
        raise RuntimeError("controller authority-root patch point changed")
    source_text = source_text.replace(needle, replacement)
    multistep_import = (
        "from multistep_drift_calibration import build_calibrated_stepper  # noqa: E402\n"
    )
    if source_text.count(multistep_import) != 1:
        raise RuntimeError("controller multistep import-removal point changed")
    source_text = source_text.replace(multistep_import, "")
    graph_time_before = "control_graph_diffusion_time=0.0,"
    graph_time_after = (
        "control_graph_diffusion_time=float(config.graph_diffusion_time),"
    )
    if source_text.count(graph_time_before) != 1:
        raise RuntimeError("controller D graph-time restoration point changed")
    source_text = source_text.replace(graph_time_before, graph_time_after)
    local_override = '''    local_selector = np.zeros(
        (len(selected), int(world.n_channels)), dtype=np.float64
    )
    local_selector[np.arange(len(selected)), selected] = 1.0
    world.control_channel_map = torch.as_tensor(
        local_selector, dtype=world.dtype, device=world.device
    )
    world.control_map = (
        world.control_channel_map @ world.components.T
    ).contiguous()
'''
    if source_text.count(local_override) != 1:
        raise RuntimeError("controller local-selector override patch point changed")
    source_text = source_text.replace(local_override, "")
    feedback_override = '''        feedback_channel_map=graph_aware_feedback_channel_map(
            adapter, float(config.graph_diffusion_time)
        ),
'''
    if source_text.count(feedback_override) != 1:
        raise RuntimeError("controller feedback-only heat-map patch point changed")
    source_text = source_text.replace(feedback_override, "")
    calibrated_stepper = '''    stepper = build_calibrated_stepper(
        world,
        adapter,
        diffusion_scale=float(config.diffusion_scale),
        increment_scale=float(config.increment_scale),
        persistence_skip=float(config.persistence_skip),
    )
'''
    canonical_stepper = '''    stepper = FrozenIctalGraphRCBatchStepper(
        world,
        adapter,
        diffusion_scale=float(config.diffusion_scale),
    )
'''
    if source_text.count(calibrated_stepper) != 1:
        raise RuntimeError("controller D-stepper restoration point changed")
    source_text = source_text.replace(calibrated_stepper, canonical_stepper)
    exec(
        compile(source_text, str(source), "exec"),
        module.__dict__,
    )
    from mfc_pipeline.full_markov_hjb_fp_wgan import empirical_fp_rollout

    # The formal path deliberately bypasses the controller reference's later
    # budget-projection helper.  D's empirical FP rollout is the scientific
    # authority; energy, peak and saturation are evaluated after propagation.
    module.empirical_fp_rollout = empirical_fp_rollout
    module.FORMAL_WGAN_PATCH_RECEIPT = {
        "pretrain_noise": "seed+101*(bank+1)",
        "pretrain_gp": "seed+50000+step",
        "joint_noise": "seed+1009*(epoch+1)",
        "joint_gp": "seed+100000*(epoch+1)+critic_step",
        "actor_gradient_clip": 1.0,
        "critic_gradient_clip": 5.0,
        "critic_adam_betas": [0.0, 0.9],
        "epoch_zero_eligible": False,
        "empty_noninferior_pool": "retain_frozen_s6_teacher_no_arm_reselection",
    }
    module.FORMAL_ROLLOUT_RECEIPT = {
        "implementation": "mfc_pipeline.full_markov_hjb_fp_wgan.empirical_fp_rollout",
        "implementation_sha256": str(config["canonical_locks"]["wgan_gp"][1]).lower(),
        "hard_projection_or_rescaling": False,
        "energy_role": "post_rollout_feasibility_screen_or_veto_only",
    }
    module.FORMAL_CONTROLLER_ADAPTER_RECEIPT = {
        "controller_reference_sha256": expected_source,
        "only_runtime_source_edits": [
            "controller_PIPELINE_ROOT_to_hash_locked_D_canonical",
            "remove_local_multistep_import_and_use_D_FrozenIctalGraphRCBatchStepper",
            "restore_D_formal_tau_as_TorchGraphRCSDE_plant_input",
            "remove_adapted_local_selector_and_control_map_override",
            "restore_analytical_weighted_ridge_gain_to_adapter_control_map",
        ],
        "causal_particle_rollout_sha256": str(
            config["canonical_locks"]["causal_particle_rollout"][1]
        ).lower(),
        "causal_riccati_sha256": str(
            config["canonical_locks"]["causal_riccati"][1]
        ).lower(),
        "equation_or_loss_edit": False,
        "projection_or_gate_edit": False,
        "plant_semantics": (
            "D_formal_graph_diffusion_in_world_no_local_selector_override_and_D_stepper"
        ),
        "unselected_channels_may_receive_graph_mediated_effective_input": True,
    }
    imported = {
        key: Path(value.__file__).resolve()
        for key, value in sys.modules.items()
        if (key == "mfc_pipeline" or key.startswith("mfc_pipeline."))
        and getattr(value, "__file__", None)
    }
    escaped = {
        key: value
        for key, value in imported.items()
        if value != canonical and canonical not in value.parents
    }
    if escaped:
        raise ImportError(f"controller imported non-D mfc_pipeline modules: {escaped}")
    dependency_locks = {
        "mfc_pipeline.actor_wgan": "actor_wgan",
        "mfc_pipeline.causal_ltv_particle_rollout": "causal_particle_rollout",
        "mfc_pipeline.causal_ltv_riccati": "causal_riccati",
        "mfc_pipeline.full_markov_hjb_fp_wgan": "wgan_gp",
        "mfc_pipeline.data": "canonical_data_loader",
        "mfc_pipeline.part2_data_pipeline": "part2_data_pipeline",
        "mfc_pipeline.part2_state_dependent_rc_sde": "part2_core",
        "mfc_pipeline.sequential_covariance_hjb": "structured_actor",
        "mfc_pipeline.square_wave_mfc": "square_wave_mfc",
    }
    for module_name, lock_name in dependency_locks.items():
        if module_name not in imported:
            raise ImportError(f"controller did not import required D module: {module_name}")
        path, expected = config["canonical_locks"][lock_name]
        if imported[module_name] != Path(str(path)).resolve():
            raise ImportError(f"controller dependency path changed: {module_name}")
        if sha256_file(imported[module_name]) != str(expected).lower():
            raise PermissionError(f"controller dependency hash changed: {module_name}")
    return module


def unprojected_rollout(
    ctrl: Any, stepper: Any, actor: Any, initial: Any, standard_normal: Any
) -> Any:
    """Run the D-canonical empirical FP trajectory without rescaling controls."""

    return ctrl.empirical_fp_rollout(stepper, actor, initial, standard_normal)


def network_for_fraction(network: Mapping[str, Any], fraction: float) -> dict[str, Any]:
    import numpy as np

    result = {key: np.asarray(value).copy() for key, value in network.items()}
    q = fraction_to_quantile(float(fraction))
    score = np.asarray(result["centrality_score"], dtype=np.float64)
    threshold = float(np.quantile(score, q, method="linear"))
    mask = score > threshold
    if not mask.any() or bool(mask.all()):
        raise ValueError("fraction produced an empty or dense-all actuator mask")
    direct_fraction = float(mask.mean())
    if direct_fraction > 0.80 + 1.0e-12 or 1.0 - direct_fraction < 0.20 - 1.0e-12:
        raise ValueError("strict centrality mask violates the frozen sparse bound")
    if score.size == 64 and float(fraction) == 0.80 and int(mask.sum()) > 51:
        raise ValueError("HUP065 f=0.80 strict mask may contain at most 51 actuators")
    result["target_mask"] = mask
    result["candidate_fraction"] = np.asarray(float(fraction))
    result["candidate_target_quantile"] = np.asarray(q)
    result["candidate_centrality_threshold"] = np.asarray(threshold)
    return result


def _standardized_reference(model: Any, arrays: Mapping[str, Any], runs: Sequence[str]) -> Any:
    import numpy as np

    paths = []
    for run in runs:
        raw = reference_paths(get_run(arrays, "preictal_reference", run))
        transformed = model.transform.scaler.transform(raw.reshape(-1, raw.shape[-1]))
        paths.append(transformed.reshape(raw.shape))
    return np.concatenate(paths, axis=0)


def standardized_reference_pools(
    model: Any, arrays: Mapping[str, Any], runs: Sequence[str]
) -> tuple[Any, Any, dict[str, Any]]:
    """Deterministically split preictal paths; outcomes never choose the split."""

    import numpy as np

    ordered_runs = list(runs)
    if ordered_runs != ["run-01", "run-02"]:
        raise ValueError("all-development reference pool order must be run01 then run02")
    fit_blocks = []
    validation_blocks = []
    per_run_receipts: list[dict[str, Any]] = []
    for run in ordered_runs:
        raw = reference_paths(get_run(arrays, "preictal_reference", run))
        transformed = model.transform.scaler.transform(raw.reshape(-1, raw.shape[-1]))
        transformed = transformed.reshape(raw.shape)
        if int(transformed.shape[0]) < 30:
            raise ValueError(f"{run} reference provides fewer than 30 one-second paths")
        fit_block = transformed[0:15].copy()
        validation_block = transformed[15:30].copy()
        fit_blocks.append(fit_block)
        validation_blocks.append(validation_block)
        per_run_receipts.append(
            {
                "run": run,
                "fit_source_path_indices_half_open": [0, 15],
                "ctx5_validation_source_path_indices_half_open": [15, 30],
                "fit_block_sha256": array_sha256(fit_block),
                "ctx5_validation_block_sha256": array_sha256(validation_block),
            }
        )
    fit = np.concatenate(fit_blocks, axis=0)
    validation = np.concatenate(validation_blocks, axis=0)
    if fit.shape != validation.shape or fit.shape[0] != 30 or fit.shape[1] != 256:
        raise ValueError("frozen reference fit/validation path shapes changed")
    return fit, validation, {
        "run_order": list(runs),
        "deterministic_pool_construction": (
            "fit_concat_each_run_paths0_15_ctx5_concat_each_run_paths15_30"
        ),
        "per_run_source_paths": per_run_receipts,
        "per_run_fit_path_indices_half_open": [0, 15],
        "per_run_ctx5_validation_path_indices_half_open": [15, 30],
        "unused_per_run_tail_path_indices_half_open": [30, 40],
        "fit_reference_role": "teacher_and_wgan_objective_only",
        "validation_reference_role": "ctx5_checkpoint_epoch_selection_only",
        "same_tensor_for_fit_and_validation": False,
        "fit_reference_sha256": array_sha256(fit),
        "validation_reference_sha256": array_sha256(validation),
        "fit_reference_shape": list(fit.shape),
        "validation_reference_shape": list(validation.shape),
    }


def _contexts_and_futures(
    arrays: Mapping[str, Any], runs: Sequence[str], indices: Sequence[int]
) -> tuple[Any, Any, list[dict[str, Any]]]:
    import numpy as np

    contexts = []
    futures = []
    ledger = []
    for run in runs:
        sequence = get_run(arrays, "ictal", run)
        for index in indices:
            context, future, boundary = part3_context_block(sequence, int(index))
            contexts.append(context)
            futures.append(future)
            ledger.append(
                {"run": run, "context_index": int(index), "boundary_sample": int(boundary)}
            )
    return np.stack(contexts), np.stack(futures), ledger


def _controller_config(
    ctrl: Any, config: Mapping[str, Any], *, fraction: float, gain: float, tau: float,
    train: bool,
) -> Any:
    spec = config["controller"]
    return ctrl.ControllerConfig(
        horizon=int(spec["horizon_samples"]),
        sampling_rate_hz=256.0,
        particles=int(spec["particles"]),
        control_step_scale=float(spec["control_step_scale"]),
        diffusion_scale=float(config["part2"]["effective_diffusion_multiplier"]),
        graph_diffusion_time=float(tau),
        amplitude_limit=float(spec["amplitude_limit"]),
        energy_rms=float(spec["per_actuator_rms_max"]),
        total_episode_energy_budget=float(spec["total_energy_cap"]),
        target_quantile=fraction_to_quantile(float(fraction)),
        base_gain_scale=float(gain),
        base_gain_ridge=float(spec["base_gain_ridge"]),
        markov_residual_scale=float(spec["markov_residual_scale"]),
        teacher_epochs=int(spec["teacher_epochs_total"]) if train else 0,
        teacher_learning_rate=float(spec["teacher_stages"][0]["learning_rate"]),
        wgan_epochs=int(spec["wgan_epochs"]) if train else 0,
        actor_learning_rate=float(spec["actor_learning_rate"]),
        critic_learning_rate=float(spec["critic_learning_rate"]),
        validation_every=int(spec["validation_every_epochs"]),
        critic_pretrain_steps=int(spec["critic_pretrain_steps"]),
        critic_pretrain_banks=int(spec["critic_pretrain_banks"]),
        critic_steps=int(spec["critic_steps"]),
        adversarial_weight=float(spec["adversarial_weight"]),
        teacher_anchor_weight=float(spec["teacher_anchor_weight"]),
        seed=int(spec["seed"]),
    )


def frozen_gate_b_components(
    time_free: Any,
    time_controlled: Any,
    occupation_free: Any,
    occupation_controlled: Any,
    mean_absolute_error: Any,
    symmetric_sd_ratio: Any,
) -> dict[str, Any]:
    """Return the six frozen per-channel Gate-B components and their AND."""

    import numpy as np

    tf = np.asarray(time_free, dtype=np.float64)
    tc = np.asarray(time_controlled, dtype=np.float64)
    of = np.asarray(occupation_free, dtype=np.float64)
    oc = np.asarray(occupation_controlled, dtype=np.float64)
    mae = np.asarray(mean_absolute_error, dtype=np.float64)
    rho = np.asarray(symmetric_sd_ratio, dtype=np.float64)
    if len({value.shape for value in (tf, tc, of, oc, mae, rho)}) != 1:
        raise ValueError("Gate-B component arrays must share a shape")
    finite = np.isfinite(np.stack([tf, tc, of, oc, mae, rho])).all(axis=0)
    relative_time = (tf - tc) / np.maximum(tf, 1.0e-12)
    relative_occupation = (of - oc) / np.maximum(of, 1.0e-12)
    component_passes = {
        "gate_time_w1_relative_reduction_pass": (
            relative_time >= GATE_B_RELATIVE_REDUCTION_MIN
        ),
        "gate_occupation_w1_relative_reduction_pass": (
            relative_occupation >= GATE_B_RELATIVE_REDUCTION_MIN
        ),
        "gate_time_w1_absolute_pass": tc <= GATE_B_TIME_W1_MAX,
        "gate_occupation_w1_absolute_pass": (
            oc <= GATE_B_OCCUPATION_W1_MAX
        ),
        "gate_mean_error_pass": mae <= GATE_B_MEAN_ERROR_MAX,
        "gate_sd_ratio_pass": (
            rho <= GATE_B_SYMMETRIC_SD_RATIO_MAX
        ),
    }
    full = finite & np.logical_and.reduce(
        [np.asarray(value, dtype=bool) for value in component_passes.values()]
    )
    return {
        **component_passes,
        "finite_pass": finite,
        "relative_time_w1_reduction": relative_time,
        "relative_occupation_w1_reduction": relative_occupation,
        "full_gate_b_pass": full,
    }


def evaluate_detailed(
    ctrl: Any,
    actor: Any,
    stepper: Any,
    initial_states: Sequence[Any],
    reference: Any,
    direct_mask: Any,
    controller_config: Any,
    *,
    fold_id: str,
    stage: str,
    ledger: Sequence[Mapping[str, Any]],
    observed_standardized: Any | None = None,
    display_local_index: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    import numpy as np
    import torch

    reference_np = np.asarray(reference, dtype=np.float64)
    direct = np.asarray(direct_mask, dtype=bool)
    bank_count = 3
    channel_passes: list[Any] = []
    both_improved_passes: list[Any] = []
    component_passes: dict[str, list[Any]] = {
        name: []
        for name in (
            "gate_time_w1_relative_reduction_pass",
            "gate_occupation_w1_relative_reduction_pass",
            "gate_time_w1_absolute_pass",
            "gate_occupation_w1_absolute_pass",
            "gate_mean_error_pass",
            "gate_sd_ratio_pass",
        )
    }
    rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, float]] = []
    fixed_display: dict[str, Any] = {}
    actor.eval()
    with torch.no_grad():
        for local_index, initial in enumerate(initial_states):
            for bank in range(bank_count):
                replicate = local_index * bank_count + bank
                seed = deterministic_seed(
                    PROTOCOL_ID, "HUP065", fold_id, stage, replicate,
                    int(controller_config.seed),
                )
                noise = ctrl.antithetic_noise(
                    seed, int(controller_config.particles),
                    int(controller_config.horizon), stepper.q,
                )
                free = ctrl.uncontrolled_particle_rollout(stepper, initial, noise)[:, 1:]
                controlled_rollout = unprojected_rollout(
                    ctrl, stepper, actor, initial, noise
                )
                controlled = controlled_rollout.scaled[:, 1:]
                free_np = free.cpu().numpy()
                controlled_np = controlled.cpu().numpy()
                controls = controlled_rollout.controls.cpu().numpy()
                time_free = ctrl.channelwise_time_w1(free_np, reference_np)
                time_controlled = ctrl.channelwise_time_w1(controlled_np, reference_np)
                occupation_free = ctrl.channelwise_occupation_w1(free_np, reference_np)
                occupation_controlled = ctrl.channelwise_occupation_w1(
                    controlled_np, reference_np
                )
                reference_mean = reference_np.mean(axis=(0, 1))
                reference_sd = reference_np.std(axis=(0, 1), ddof=0)
                controlled_mean = controlled_np.mean(axis=(0, 1))
                controlled_sd = controlled_np.std(axis=(0, 1), ddof=0)
                mean_error = np.abs(controlled_mean - reference_mean)
                symmetric_sd_ratio = np.maximum(
                    controlled_sd / np.maximum(reference_sd, 1.0e-12),
                    reference_sd / np.maximum(controlled_sd, 1.0e-12),
                )
                gate = frozen_gate_b_components(
                    time_free, time_controlled, occupation_free,
                    occupation_controlled, mean_error, symmetric_sd_ratio,
                )
                full_gate_b = np.asarray(gate["full_gate_b_pass"], dtype=bool)
                both_improved = (
                    (time_controlled < time_free)
                    & (occupation_controlled < occupation_free)
                )
                channel_passes.append(full_gate_b)
                both_improved_passes.append(both_improved)
                for name in component_passes:
                    component_passes[name].append(np.asarray(gate[name], dtype=bool))
                total_energy = float(
                    np.mean(np.sum(np.square(controls), axis=(1, 2))) / 256.0
                )
                per_actuator_rms = np.sqrt(np.mean(np.square(controls), axis=(0, 1)))
                peak = float(np.max(np.abs(controls)))
                saturation = float(
                    np.mean(
                        np.abs(controls)
                        >= 0.95 * float(controller_config.amplitude_limit)
                    )
                )
                finite = bool(
                    np.isfinite(free_np).all()
                    and np.isfinite(controlled_np).all()
                    and np.isfinite(controls).all()
                )
                gate_c = bool(
                    finite
                    and total_energy
                    <= float(controller_config.total_episode_energy_budget) + 1.0e-9
                    and float(np.max(per_actuator_rms)) <= 0.405 + 1.0e-9
                    and peak <= float(controller_config.amplitude_limit) + 1.0e-9
                    and saturation < 0.01
                )
                row_summary: dict[str, float] = {
                    "mean_time_w1_free": float(time_free.mean()),
                    "mean_time_w1_controlled": float(time_controlled.mean()),
                    "mean_occupation_w1_free": float(occupation_free.mean()),
                    "mean_occupation_w1_controlled": float(occupation_controlled.mean()),
                    "mean_absolute_mean_error": float(mean_error.mean()),
                    "mean_absolute_sd_error": float(
                        np.mean(np.abs(controlled_sd - reference_sd))
                    ),
                    "both_improved_channel_count": float(both_improved.sum()),
                    "full_six_gate_b_channel_count": float(full_gate_b.sum()),
                    "total_energy": total_energy,
                    "maximum_per_actuator_rms": float(np.max(per_actuator_rms)),
                    "control_peak": peak,
                    "saturation_fraction": saturation,
                    "finite": float(finite),
                    "gate_c": float(gate_c),
                }
                if observed_standardized is not None:
                    plant = ctrl.plant_fidelity_metrics(
                        free_np,
                        np.asarray(observed_standardized[local_index]),
                        reference_np,
                    )
                    row_summary.update(
                        {key: float(value) for key, value in plant.items()}
                    )
                aggregate_rows.append(row_summary)
                identity = {**dict(ledger[local_index]), "crn_bank": bank}
                for channel_index in range(stepper.n_channels):
                    rows.append(
                        {
                            **identity,
                            "fold_id": fold_id,
                            "stage": stage,
                            "noise_seed": seed,
                            "channel_index": channel_index,
                            "direct_actuated": bool(direct[channel_index]),
                            "time_w1_free": float(time_free[channel_index]),
                            "time_w1_controlled": float(time_controlled[channel_index]),
                            "occupation_w1_free": float(occupation_free[channel_index]),
                            "occupation_w1_controlled": float(
                                occupation_controlled[channel_index]
                            ),
                            "relative_time_w1_reduction": float(
                                gate["relative_time_w1_reduction"][channel_index]
                            ),
                            "relative_occupation_w1_reduction": float(
                                gate["relative_occupation_w1_reduction"][channel_index]
                            ),
                            "mean_absolute_error": float(mean_error[channel_index]),
                            "symmetric_sd_ratio": float(
                                symmetric_sd_ratio[channel_index]
                            ),
                            "both_time_and_occupation_improved": bool(
                                both_improved[channel_index]
                            ),
                            **{
                                name: bool(gate[name][channel_index])
                                for name in component_passes
                            },
                            "gate_b_pass": bool(full_gate_b[channel_index]),
                            "gate_c_trajectory_pass": gate_c,
                            "trajectory_total_energy": total_energy,
                            "trajectory_maximum_per_actuator_rms": float(
                                np.max(per_actuator_rms)
                            ),
                            "trajectory_control_peak": peak,
                            "trajectory_saturation_fraction": saturation,
                            "trajectory_finite": finite,
                            "hard_projection_or_rescaling_applied": False,
                        }
                    )
                if local_index == int(display_local_index) and bank == 0:
                    fixed_display = {
                        "free": free_np,
                        "controlled": controlled_np,
                        "reference": reference_np,
                        "controls": controls,
                        "noise": noise.cpu().numpy(),
                        "ledger": identity,
                    }
    passed_all_contexts = np.all(np.stack(channel_passes), axis=0)
    both_improved_all_contexts = np.all(np.stack(both_improved_passes), axis=0)
    component_masks = {
        name: np.all(np.stack(values), axis=0)
        for name, values in component_passes.items()
    }
    keys = aggregate_rows[0].keys()
    means = {key: float(np.mean([row[key] for row in aggregate_rows])) for key in keys}
    plant_diagnostic_evaluated = observed_standardized is not None
    plant_diagnostic_pass: bool | None = None
    if observed_standardized is not None:
        plant_diagnostic_pass = bool(
            means["mean_time_w1_free_observed"] < means["mean_time_w1_free_reference_plant"]
            and means["mean_time_w1_free_observed"] < means["mean_time_w1_observed_reference"]
            and means["mean_occupation_w1_free_observed"] < means["mean_occupation_w1_free_reference_plant"]
            and means["mean_occupation_w1_free_observed"] < means["mean_occupation_w1_observed_reference"]
        )
    aggregate_control_improvement = bool(
        means["mean_time_w1_controlled"] < means["mean_time_w1_free"]
        and means["mean_occupation_w1_controlled"] < means["mean_occupation_w1_free"]
    )
    gate_c = bool(all(bool(row["gate_c"]) for row in aggregate_rows))
    protocol_gate_a_integrity_pass = True
    summary = {
        **means,
        "contexts": len(initial_states),
        "crn_banks_per_context": bank_count,
        "context_bank_evaluations": len(initial_states) * bank_count,
        "protocol_gate_a_integrity_pass": protocol_gate_a_integrity_pass,
        "protocol_gate_a_integrity_role": (
            "static_hash_split_crn_and_uncontrolled_parity_integrity"
        ),
        "aggregate_control_improvement_diagnostic_pass": (
            aggregate_control_improvement
        ),
        "aggregate_control_improvement_role": (
            "diagnostic_only_not_protocol_gate_a_not_eligibility"
        ),
        "gate_b_pass_count": int(passed_all_contexts.sum()),
        "gate_b_pass_fraction": float(passed_all_contexts.mean()),
        "gate_b_pass_mask": passed_all_contexts.astype(bool).tolist(),
        "both_improved_all_context_count": int(both_improved_all_contexts.sum()),
        "both_improved_all_context_mask": both_improved_all_contexts.astype(bool).tolist(),
        "gate_b_component_masks": {
            name: values.astype(bool).tolist()
            for name, values in component_masks.items()
        },
        "gate_b_cross_context_rule": (
            "fail_closed_logical_AND_of_each_component_over_all_contexts_and_crn_banks"
        ),
        "gate_b_thresholds": {
            "relative_time_w1_reduction_min": GATE_B_RELATIVE_REDUCTION_MIN,
            "relative_occupation_w1_reduction_min": GATE_B_RELATIVE_REDUCTION_MIN,
            "controlled_time_w1_max": GATE_B_TIME_W1_MAX,
            "controlled_occupation_w1_max": GATE_B_OCCUPATION_W1_MAX,
            "controlled_reference_mean_abs_error_max": GATE_B_MEAN_ERROR_MAX,
            "symmetric_sd_ratio_max": GATE_B_SYMMETRIC_SD_RATIO_MAX,
        },
        "gate_c_pass": gate_c,
        "energy_control_is_post_rollout_gate_only": True,
        "hard_projection_or_rescaling_applied": False,
        "plant_fidelity_diagnostic_evaluated": plant_diagnostic_evaluated,
        "plant_fidelity_diagnostic_pass": plant_diagnostic_pass,
        "plant_fidelity_diagnostic_role": (
            "report_only_never_eligibility_rank_no_go_rescue_or_replacement"
        ),
        "plant_fidelity_used_for_eligibility_or_rank": False,
        "mean_time_plus_occupation_w1": float(
            means["mean_time_w1_controlled"] + means["mean_occupation_w1_controlled"]
        ),
        # Analytical eligibility is exactly the preregistered static-integrity
        # plus post-rollout energy/finite Gate-C prefilter.  Aggregate control
        # improvement, all six Gate-B components, and the four plant distances
        # remain ranking/reporting diagnostics and cannot silently discard an
        # arm before the declared lexicographic rank.
        "controller_feasible": analytical_candidate_eligible(
            protocol_gate_a_integrity_pass=protocol_gate_a_integrity_pass,
            gate_c_energy_finite_pass=gate_c,
        ),
        "aggregate_control_improvement_or_gate_b_used_as_pre_rank_eligibility": False,
        "plant_diagnostic_used_as_pre_rank_eligibility": False,
        "all_context_gate_b_mask_sha256": array_sha256(passed_all_contexts),
    }
    return summary, rows, fixed_display


def _candidate_id(fraction: float, gain: float, tau: float) -> str:
    return f"f-{fraction:.2f}_q-{1.0-fraction:.2f}_gain-{gain:.3f}_tau-{tau:.3f}"


def loro_analytical_top1(
    config: Mapping[str, Any], development_npz: Path, part2_dir: Path, output: Path
) -> None:
    import joblib
    import numpy as np
    import pandas as pd
    import torch

    ctrl = activate_controller(config)
    arrays = load_development(development_npz)
    runs = list(config["source_data"]["development_runs"])
    controller = config["controller"]
    all_summaries: list[dict[str, Any]] = []
    all_channels: list[dict[str, Any]] = []
    fold_cache: dict[str, dict[str, Any]] = {}
    for validation_run in runs:
        fold_id = f"leave-{validation_run}-out"
        training_runs = [run for run in runs if run != validation_run]
        model = joblib.load(part2_dir / "fold_models" / f"{fold_id}.joblib")
        assert_live_part2_model_identity(
            model, sys.modules["mfc_pipeline.part2_state_dependent_rc_sde"]
        )
        network = load_network(
            part2_dir / "fold_part1_selection_networks" / f"{fold_id}.npz"
        )
        fit_contexts, _fit_futures, _fit_ledger = _contexts_and_futures(
            arrays, training_runs, config["contexts"]["analytical_and_controller_fit"]
        )
        validation_contexts, validation_futures, validation_ledger = _contexts_and_futures(
            arrays, [validation_run], config["contexts"]["analytical_and_controller_fit"]
        )
        fit_reference = _standardized_reference(model, arrays, training_runs)
        validation_reference = _standardized_reference(model, arrays, [validation_run])
        reference_split = {
            "fold_fit_reference_runs": list(training_runs),
            "fold_fit_absolute_seconds_half_open": [30.0, 70.0],
            "fold_fit_reference_sha256": array_sha256(fit_reference),
            "leftout_validation_reference_run": validation_run,
            "leftout_validation_absolute_seconds_half_open": [30.0, 70.0],
            "leftout_validation_reference_sha256": array_sha256(validation_reference),
        }
        observed = np.stack(
            [model.transform.scaler.transform(item) for item in validation_futures]
        )
        fold_cache[fold_id] = {
            "model": model, "network": network,
            "fit_contexts": fit_contexts, "validation_contexts": validation_contexts,
            "fit_reference": fit_reference, "validation_reference": validation_reference,
            "reference_split": reference_split,
            "observed": observed, "ledger": validation_ledger,
        }
    for fraction in controller["fraction_grid"]:
        for gain in controller["gain_grid"]:
            for tau in controller["tau_grid"]:
                candidate_id = _candidate_id(float(fraction), float(gain), float(tau))
                for fold_id, bundle in fold_cache.items():
                    candidate_network = network_for_fraction(bundle["network"], float(fraction))
                    cfg = _controller_config(
                        ctrl, config, fraction=float(fraction), gain=float(gain),
                        tau=float(tau), train=False,
                    )
                    torch.manual_seed(int(cfg.seed))
                    stack = ctrl.build_control_stack(
                        bundle["model"], candidate_network, bundle["fit_contexts"],
                        bundle["fit_reference"], cfg,
                    )
                    stepper = stack[2]
                    initial = [
                        stepper.adapter.initial_state_from_context(item)
                        for item in bundle["validation_contexts"]
                    ]
                    summary, channel_rows, _display = evaluate_detailed(
                        ctrl, stack[-1], stepper, initial, bundle["validation_reference"],
                        candidate_network["target_mask"], cfg,
                        fold_id=fold_id, stage="loro_analytical",
                        ledger=bundle["ledger"], observed_standardized=bundle["observed"],
                    )
                    all_summaries.append(
                        {
                            "candidate_id": candidate_id, "fold_id": fold_id,
                            "fraction": float(fraction),
                            "target_quantile": fraction_to_quantile(float(fraction)),
                            "gain": float(gain), "tau": float(tau),
                            "actuator_count": int(np.asarray(candidate_network["target_mask"]).sum()),
                            "target_mask_sha256": array_sha256(candidate_network["target_mask"]),
                            "fold_fit_reference_sha256": bundle["reference_split"]["fold_fit_reference_sha256"],
                            "leftout_validation_reference_sha256": bundle["reference_split"]["leftout_validation_reference_sha256"],
                            **summary,
                        }
                    )
                    for row in channel_rows:
                        all_channels.append({"candidate_id": candidate_id, **row})
                print(f"analytical {candidate_id} complete", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(all_summaries)
    summary_frame.to_csv(output / "analytical_fold_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_channels).to_csv(
        output / "analytical_channel_context_metrics.csv", index=False, encoding="utf-8-sig"
    )
    ranked_rows: list[dict[str, Any]] = []
    for candidate_id, group in summary_frame.groupby("candidate_id", sort=True):
        first = group.iloc[0]
        row = {
            "candidate_id": candidate_id,
            "fraction": float(first["fraction"]),
            "target_quantile": float(first["target_quantile"]),
            "gain": float(first["gain"]),
            "tau": float(first["tau"]),
            "all_folds_feasible": bool(group["controller_feasible"].astype(bool).all()),
            "worst_fold_full_gate_b_count": int(group["gate_b_pass_count"].min()),
            "worst_fold_mean_time_plus_occupation_w1": float(
                group["mean_time_plus_occupation_w1"].max()
            ),
            "maximum_fold_actuator_count": int(group["actuator_count"].max()),
            "all_fold_plant_diagnostic_pass": bool(
                group["plant_fidelity_diagnostic_pass"].astype(bool).all()
            ),
            "plant_diagnostic_used_for_eligibility_or_rank": False,
            "all_fold_protocol_gate_a_integrity_pass": bool(
                group["protocol_gate_a_integrity_pass"].astype(bool).all()
            ),
            "all_fold_gate_c_pass": bool(group["gate_c_pass"].astype(bool).all()),
        }
        ranked_rows.append(row)
    ranking = pd.DataFrame(ranked_rows)
    eligible = ranking[ranking["all_folds_feasible"]].sort_values(
        ["worst_fold_full_gate_b_count", "worst_fold_mean_time_plus_occupation_w1",
         "maximum_fold_actuator_count", "candidate_id"],
        ascending=[False, True, True, True], kind="mergesort",
    )
    if eligible.empty:
        ranking.to_csv(output / "analytical_ranking.csv", index=False, encoding="utf-8-sig")
        raise RuntimeError(
            "no analytical arm passed every LORO static-integrity and Gate-C energy/finite gate"
        )
    selected = eligible.iloc[0].to_dict()
    ranking["selected_unique_top1"] = ranking["candidate_id"].eq(selected["candidate_id"])
    ranking.to_csv(output / "analytical_ranking.csv", index=False, encoding="utf-8-sig")
    atomic_json(
        output / "frozen_top1.json",
        {
            "schema_version": "hup065-loro-analytical-top1-v1",
            "selection": selected,
            "freeze_top_k": 1,
            "rank_order": controller["analytical_rank"],
            "ctx5_used": False,
            "ctx6_used": False,
            "ctx7_used": False,
            "sealed_run_opened": False,
            "candidate_count": len(ranking),
            "feasible_candidate_count": len(eligible),
            "eligibility_rule": (
                "protocol_gate_a_static_integrity_and_gate_c_energy_finite_only"
            ),
            "plant_fidelity_role": (
                "diagnostic_report_only_never_eligibility_rank_or_no_go"
            ),
            "no_post_training_arm_reselection": True,
        },
    )


def final_refit(
    config: Mapping[str, Any], development_npz: Path, part2_dir: Path,
    analytical_dir: Path, output: Path,
) -> None:
    import joblib
    import numpy as np
    from dataclasses import fields

    _data, _network, part2_module, _part2_pipeline = activate_canonical(config)
    arrays = load_development(development_npz)
    channels = arrays["channels"].astype(str).tolist()
    runs = list(config["source_data"]["development_runs"])
    top1 = json.loads((analytical_dir / "frozen_top1.json").read_text(encoding="utf-8"))["selection"]
    selected_payload = json.loads((part2_dir / "selected_config.json").read_text(encoding="utf-8"))
    model_config = part2_module.ModelConfig(**selected_payload["model_config"])
    sequences = [get_run(arrays, "ictal", run) for run in runs]
    network = build_part1_network(config, sequences, float(arrays["sfreq"]))
    plant_adjacency = build_part2_plant_adjacency(
        config, sequences, float(arrays["sfreq"])
    )
    model = part2_module.ResidualGraphRCSDE(model_config, plant_adjacency).fit(sequences)
    assert_live_part2_model_identity(model, part2_module)
    if array_sha256(np.asarray(model.adjacency_input)) != array_sha256(plant_adjacency):
        raise PermissionError("Part-II model adjacency_input differs from broadband plant graph")
    normalized_plant_adjacency = np.asarray(model.adjacency, dtype=np.float64)
    candidate_network = network_for_fraction(network, float(top1["fraction"]))
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "selected_model.joblib"
    network_path = output / "network.npz"
    plant_path = output / "part2_plant_adjacency.npz"
    joblib.dump(model, model_path, compress=3)
    save_network(network_path, candidate_network, channels)
    np.savez_compressed(
        plant_path,
        adjacency_input=np.asarray(plant_adjacency, dtype=np.float64),
        adjacency_normalized=normalized_plant_adjacency,
        channels=np.asarray(channels, dtype=str),
        graph_role=np.asarray("part2_broadband_2s_graph_rc_plant"),
    )
    mask = np.asarray(candidate_network["target_mask"], dtype=bool)
    mask_rows = [
        {"channel_index": index, "channel": channel, "direct_actuated": bool(mask[index])}
        for index, channel in enumerate(channels)
    ]
    import pandas as pd
    pd.DataFrame(mask_rows).to_csv(output / "frozen_mask.csv", index=False, encoding="utf-8-sig")
    atomic_json(
        output / "refit_receipt.json",
        {
            "schema_version": "hup065-final-development-refit-v1",
            "subject": "HUP065",
            "development_runs": runs,
            "model_config": asdict(model_config),
            "analytical_top1": top1,
            "model_sha256": sha256_file(model_path),
            "part1_selection_network_sha256": sha256_file(network_path),
            "part2_plant_adjacency_sha256": sha256_file(plant_path),
            "part2_plant_adjacency_input_array_sha256": array_sha256(
                np.asarray(plant_adjacency)
            ),
            "part2_model_normalized_adjacency_array_sha256": array_sha256(
                normalized_plant_adjacency
            ),
            "part1_graph_reused_as_plant_graph": False,
            "rolling_block_samples_selected_by_loro": int(
                selected_payload["rolling_block_samples_selected_by_loro"]
            ),
            "target_mask_sha256": array_sha256(mask),
            "direct_actuator_count": int(mask.sum()),
            "channels": channels,
            "ctx5_used_for_refit": False,
            "ctx6_used_for_refit": False,
            "ctx7_used": False,
            "sealed_run_opened": False,
            "old_model_or_checkpoint_used": False,
        },
    )


def _legacy_teacher180_never_dispatched(
    config: Mapping[str, Any], development_npz: Path, part2_dir: Path,
    analytical_dir: Path, refit_dir: Path, output: Path,
) -> None:
    raise RuntimeError("legacy teacher180 route is permanently disabled")

    import joblib
    import numpy as np
    import pandas as pd
    import torch

    ctrl = activate_controller(config)
    arrays = load_development(development_npz)
    runs = list(config["source_data"]["development_runs"])
    model_path = refit_dir / "selected_model.joblib"
    network_path = refit_dir / "network.npz"
    model = joblib.load(model_path)
    network = load_network(network_path)
    top1 = json.loads((analytical_dir / "frozen_top1.json").read_text(encoding="utf-8"))["selection"]
    cfg = _controller_config(
        ctrl, config, fraction=float(top1["fraction"]), gain=float(top1["gain"]),
        tau=float(top1["tau"]), train=True,
    )
    fit_contexts, _fit_futures, fit_ledger = _contexts_and_futures(
        arrays, runs, config["contexts"]["analytical_and_controller_fit"]
    )
    ctx5_contexts, _ctx5_futures, ctx5_ledger = _contexts_and_futures(
        arrays, runs, config["contexts"]["checkpoint_epoch_selection_only"]
    )
    reference, _ctx5_reference, _split_receipt = standardized_reference_pools(
        model, arrays, runs
    )
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    stack = ctrl.build_control_stack(model, network, fit_contexts, reference, cfg)
    stepper = stack[2]
    fit_initial = stack[3]
    selection_initial = [
        stepper.adapter.initial_state_from_context(item) for item in ctx5_contexts
    ]
    reference_tensor = torch.as_tensor(reference, dtype=torch.float64)
    projections = ctrl.fixed_joint_projections(stepper.n_channels, int(cfg.seed) + 811)
    started = time.perf_counter()
    actor, teacher_history, teacher_result, teacher_accepted = ctrl.train_teacher(
        stack[-1], stepper, fit_initial, selection_initial,
        reference_tensor, reference_tensor, None,
        stack[7], stack[6], projections, cfg,
        external_plant_fidelity_gate=True,
    )
    actor, critic, wgan_history, wgan_result, wgan_accepted = ctrl.train_wgan(
        actor, stepper, fit_initial, selection_initial,
        reference_tensor, reference_tensor, None,
        stack[7], stack[6], projections, cfg,
        external_plant_fidelity_gate=True,
    )
    selected_stage = (
        "structured_full_markov_actor_wgan_gp" if wgan_accepted
        else "structured_full_markov_teacher" if teacher_accepted
        else "analytical_weighted_ridge_retained_after_fixed_training"
    )
    final_result = wgan_result if wgan_accepted else teacher_result
    output.mkdir(parents=True, exist_ok=True)
    history = [*teacher_history, *wgan_history]
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(final_result.contexts).to_csv(
        output / "ctx5_epoch_selection_metrics.csv", index=False, encoding="utf-8-sig"
    )
    checkpoint_payload = {
        "schema_version": "hup065-fresh-s0-s6-controller-v1",
        "subject": "HUP065",
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict() if wgan_accepted else None,
        "controller_config": asdict(cfg),
        "selected_stage": selected_stage,
        "analytical_top1": top1,
        "target_mask_sha256": array_sha256(network["target_mask"]),
        "model_sha256": sha256_file(model_path),
        "network_sha256": sha256_file(network_path),
        "teacher_epochs_executed": int(cfg.teacher_epochs),
        "wgan_epochs_executed": int(cfg.wgan_epochs),
        "teacher_accepted": bool(teacher_accepted),
        "wgan_accepted": bool(wgan_accepted),
        "epoch_zero_eligible": False,
        "ctx5_selected_epoch_only": True,
        "ctx5_selected_arm": False,
        "ctx6_used": False,
        "ctx7_used": False,
        "old_checkpoint_used": False,
        "fresh_initialization_seed": int(cfg.seed),
    }
    checkpoint_path = output / "selected_controller.pt"
    torch.save(checkpoint_payload, checkpoint_path)
    display = final_result.display
    np.savez_compressed(
        output / "ctx5_selected_rollout.npz",
        free_standardized=display["free"], controlled_standardized=display["controlled"],
        reference_standardized=display["reference"], controls=display["controls"],
        standard_normal=display["noise"], channels=arrays["channels"],
        direct_mask=np.asarray(network["target_mask"], dtype=bool),
    )
    atomic_json(
        output / "training_receipt.json",
        {
            "schema_version": "hup065-fresh-s0-s6-training-receipt-v1",
            "selected_stage": selected_stage,
            "teacher_epochs_executed": int(cfg.teacher_epochs),
            "wgan_epochs_executed": int(cfg.wgan_epochs),
            "teacher_accepted": bool(teacher_accepted),
            "wgan_accepted": bool(wgan_accepted),
            "elapsed_seconds": time.perf_counter() - started,
            "ctx5_metrics": final_result.aggregate,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "s0_to_s6": [
                "S0 hash-bound development data and plant",
                "S1 train-only transform and reference laws",
                "S2 fresh structured actor initialization",
                "S3 frozen analytical top1 initialization",
                "S4 fixed teacher180 training on contexts0-4",
                "S5 fixed WGAN40 training on contexts0-4",
                "S6 ctx5 checkpoint-epoch selection only"
            ],
            "arm_reselection_after_analytical": False,
            "sealed_run_opened": False,
        },
    )


def _legacy_monolithic_teacher_never_dispatched(
    config: Mapping[str, Any], development_npz: Path, part2_dir: Path,
    analytical_dir: Path, refit_dir: Path, output: Path,
) -> None:
    """Deprecated monolith retained only as code history; backend never dispatches it."""

    del config, development_npz, part2_dir, analytical_dir, refit_dir, output
    raise RuntimeError("deprecated monolithic training route is disabled")


def rebuild_selected_stack(
    config: Mapping[str, Any], arrays: Mapping[str, Any], refit_dir: Path,
    training_dir: Path,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    import joblib
    import numpy as np
    import torch

    ctrl = activate_controller(config)
    model = joblib.load(refit_dir / "selected_model.joblib")
    assert_live_part2_model_identity(
        model, sys.modules["mfc_pipeline.part2_state_dependent_rc_sde"]
    )
    network = load_network(refit_dir / "network.npz")
    checkpoint = torch.load(
        training_dir / "selected_controller.pt", map_location="cpu", weights_only=False
    )
    cfg = ctrl.ControllerConfig(**checkpoint["controller_config"])
    runs = list(config["source_data"]["development_runs"])
    fit_contexts, _future, _ledger = _contexts_and_futures(
        arrays, runs, config["contexts"]["analytical_and_controller_fit"]
    )
    reference, _ctx5_reference, _split_receipt = standardized_reference_pools(
        model, arrays, runs
    )
    torch.manual_seed(int(cfg.seed))
    stack = ctrl.build_control_stack(model, network, fit_contexts, reference, cfg)
    actor = stack[-1]
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    actor.eval()
    return ctrl, model, network, cfg, stack, checkpoint


def _legacy_ctx6_terminal_veto_never_dispatched(
    config: Mapping[str, Any], development_npz: Path, refit_dir: Path,
    training_dir: Path, output: Path,
) -> None:
    raise RuntimeError(
        "standalone legacy ctx6 route is disabled; use formal WGAN40_CTX5_SELECT_CTX6_VETO"
    )

    import numpy as np
    import pandas as pd

    arrays = load_development(development_npz)
    ctrl, model, network, cfg, stack, checkpoint = rebuild_selected_stack(
        config, arrays, refit_dir, training_dir
    )
    runs = list(config["source_data"]["development_runs"])
    contexts, futures, ledger = _contexts_and_futures(
        arrays, runs, config["contexts"]["terminal_veto_only"]
    )
    observed = np.stack([model.transform.scaler.transform(item) for item in futures])
    initial = [stack[2].adapter.initial_state_from_context(item) for item in contexts]
    reference = _standardized_reference(model, arrays, runs)
    summary, rows, display = evaluate_detailed(
        ctrl, stack[-1], stack[2], initial, reference, network["target_mask"], cfg,
        fold_id="all-development", stage="ctx6_terminal_veto",
        ledger=ledger, observed_standardized=observed,
    )
    veto_pass = bool(summary["controller_feasible"])
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "ctx6_channel_metrics.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        output / "ctx6_rollout.npz",
        free_standardized=display["free"], controlled_standardized=display["controlled"],
        reference_standardized=display["reference"], controls=display["controls"],
        standard_normal=display["noise"], channels=arrays["channels"],
        direct_mask=np.asarray(network["target_mask"], dtype=bool),
    )
    atomic_json(
        output / "terminal_veto.json",
        {
            "schema_version": "hup065-ctx6-terminal-veto-v1",
            "terminal_veto_pass": veto_pass,
            "metrics": summary,
            "ctx6_selected_or_ranked_arm": False,
            "ctx6_may_rescue": False,
            "ctx7_used": False,
            "sealed_run_opened": False,
            "checkpoint_sha256": sha256_file(training_dir / "selected_controller.pt"),
        },
    )
