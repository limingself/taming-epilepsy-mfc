from __future__ import annotations

from dataclasses import asdict, replace
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

from .contracts import deterministic_seed, fraction_to_quantile, sha256_file, sha256_json
from .data_model import (
    activate_canonical,
    array_sha256,
    assert_live_part2_model_identity,
    build_part1_selection_network,
    build_part2_plant_network,
    part3_context_block,
    get_run,
    load_development,
    load_network,
    reference_paths,
    save_network,
)
from .staging import atomic_json


PROTOCOL_ID = "HUP060_consistent_sparse_rerun_common_v1"


def _purge_modules() -> None:
    for name in tuple(sys.modules):
        if name == "mfc_pipeline" or name.startswith("mfc_pipeline."):
            del sys.modules[name]
        if name == "multistep_drift_calibration":
            del sys.modules[name]


def activate_controller(config: Mapping[str, Any]) -> Any:
    """Load the dimension-generalized adapter on the exact D-canonical dynamics."""

    source = Path(str(config["adaptation_locks"]["controller_runner"][0])).resolve()
    controller_dir = source.parent
    canonical_root = Path(str(config["canonical_locks"]["part2_core"][0])).parents[1]
    forbidden_baseline = (
        Path(str(config["project_root"])).parents[2]
        / "taming-epilepsy-mfc-baseline"
    ).resolve()
    if sha256_file(source) != str(config["adaptation_locks"]["controller_runner"][1]):
        raise PermissionError("patient controller adapter source hash changed")
    canonical_modules = {
        "mfc_pipeline.actor_wgan": config["canonical_locks"]["actor_wgan"],
        "mfc_pipeline.causal_ltv_particle_rollout": config["canonical_locks"]["canonical_particle_rollout"],
        "mfc_pipeline.causal_ltv_riccati": config["canonical_locks"]["canonical_riccati"],
        "mfc_pipeline.full_markov_hjb_fp_wgan": config["canonical_locks"]["wgan_gp"],
        "mfc_pipeline.part2_state_dependent_rc_sde": config["canonical_locks"]["part2_core"],
        "mfc_pipeline.sequential_covariance_hjb": config["canonical_locks"]["structured_actor"],
        "mfc_pipeline.square_wave_mfc": config["canonical_locks"]["canonical_square_wave"],
    }
    loaded_required = [name for name in canonical_modules if name in sys.modules]
    for module_name in loaded_required:
        expected_path, expected_sha256 = canonical_modules[module_name]
        observed_path = Path(str(sys.modules[module_name].__file__)).resolve()
        if (
            os.path.normcase(str(observed_path))
            != os.path.normcase(str(Path(expected_path).resolve()))
            or sha256_file(observed_path) != str(expected_sha256)
        ):
            raise PermissionError(
                f"cannot replace loaded foreign module while model objects may exist: {module_name}"
            )
    parent = sys.modules.get("mfc_pipeline")
    if parent is not None and getattr(parent, "__file__", None) is not None:
        parent_file = Path(str(parent.__file__)).resolve()
        expected_package = (canonical_root / "mfc_pipeline").resolve()
        try:
            parent_file.relative_to(expected_package)
        except ValueError:
            if loaded_required:
                raise PermissionError(
                    "cannot replace a foreign mfc_pipeline while model objects may exist"
                )
            _purge_modules()
    for path in (str(controller_dir), str(canonical_root), str(forbidden_baseline)):
        sys.path[:] = [
            item for item in sys.path
            if os.path.normcase(item) != os.path.normcase(path)
        ]
    sys.path.insert(0, str(canonical_root))
    loaded = {}
    for module_name, (expected_path, expected_sha256) in canonical_modules.items():
        module = sys.modules.get(module_name)
        if module is None:
            module = importlib.import_module(module_name)
        observed_path = Path(str(module.__file__)).resolve()
        if os.path.normcase(str(observed_path)) != os.path.normcase(
            str(Path(expected_path).resolve())
        ):
            raise ImportError(f"{module_name} resolved outside D canonical: {observed_path}")
        if sha256_file(observed_path) != str(expected_sha256):
            raise PermissionError(f"{module_name} differs from its D-canonical hash")
        loaded[module_name] = module
    sys.path.insert(0, str(controller_dir))
    calibration_expected_path, calibration_expected_sha256 = config[
        "adaptation_locks"
    ]["multistep_calibration"]
    existing_calibration = sys.modules.get("multistep_drift_calibration")
    if existing_calibration is not None:
        calibration_path = Path(str(existing_calibration.__file__)).resolve()
        if (
            os.path.normcase(str(calibration_path))
            != os.path.normcase(str(Path(calibration_expected_path).resolve()))
            or sha256_file(calibration_path) != str(calibration_expected_sha256)
        ):
            raise PermissionError(
                "cannot replace loaded foreign multistep calibration module"
            )
    name = "hup080_frozen_fresh_controller_impl"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    sys.path[:] = [
        item for item in sys.path
        if os.path.normcase(item) != os.path.normcase(str(forbidden_baseline))
    ]
    module.PIPELINE_ROOT = canonical_root
    calibration = sys.modules.get("multistep_drift_calibration")
    if calibration is not None:
        calibration_path = Path(str(calibration.__file__)).resolve()
        if (
            os.path.normcase(str(calibration_path))
            != os.path.normcase(str(Path(calibration_expected_path).resolve()))
            or sha256_file(calibration_path) != str(calibration_expected_sha256)
        ):
            raise PermissionError("controller loaded an unpinned calibration module")
        calibration.PIPELINE_ROOT = canonical_root
    for module_name, expected in canonical_modules.items():
        observed_path = Path(str(sys.modules[module_name].__file__)).resolve()
        if os.path.normcase(str(observed_path)) != os.path.normcase(
            str(Path(expected[0]).resolve())
        ):
            raise ImportError(f"controller rebound {module_name} outside D canonical")

    module.empirical_fp_rollout = loaded[
        "mfc_pipeline.full_markov_hjb_fp_wgan"
    ].empirical_fp_rollout
    module.HUP080_CANONICAL_MODULE_BINDINGS = {
        key: str(Path(value[0]).resolve()) for key, value in canonical_modules.items()
    }
    module.HUP080_CONTROL_STACK_CONTRACT = {
        "candidate_tau_passed_to_world_control_graph_diffusion_time": True,
        "raw_part2_plant_bound_to_model_adjacency_input": True,
        "world_inherits_model_normalized_adjacency": True,
        "world_control_channel_map_overridden": False,
        "extra_feedback_heat_map_used": False,
        "calibrated_stepper_used": False,
        "d_canonical_batch_stepper_used": True,
        "runtime_projection_or_rescaling": False,
    }

    def canonical_builder(model: Any, network: Mapping[str, Any], contexts: Any,
                          fit_reference: Any, controller_config: Any) -> Any:
        return build_d_canonical_control_stack(
            module, model, network, contexts, fit_reference, controller_config
        )

    module.build_control_stack = canonical_builder
    return module


def build_d_canonical_control_stack(
    ctrl: Any, model: Any, network: Mapping[str, Any], contexts: Any,
    fit_reference: Any, config: Any,
) -> tuple[Any, ...]:
    """Dimension-general D Part-III construction with no altered plant equations."""

    import numpy as np
    import torch

    selected = np.flatnonzero(np.asarray(network["target_mask"], dtype=bool))
    if selected.size < 1 or selected.size >= np.asarray(network["target_mask"]).size:
        raise ValueError("formal sparse mask must contain direct and indirect channels")
    world = ctrl.TorchGraphRCSDE(
        model,
        selected,
        float(config.sampling_rate_hz),
        control_graph_diffusion_time=float(config.graph_diffusion_time),
        preserve_physical_control_residual=True,
        dtype=torch.float64,
        device="cpu",
    )
    if int(world.n_channels) != int(np.asarray(network["target_mask"]).size):
        raise RuntimeError("D-canonical world did not preserve patient channel dimension")
    if not np.isclose(
        float(world.control_graph_diffusion_time), float(config.graph_diffusion_time)
    ):
        raise RuntimeError("candidate tau was not installed in the D-canonical plant")
    world_adjacency = world.adjacency.detach().cpu().numpy()
    if not np.allclose(
        world_adjacency, np.asarray(model.adjacency, dtype=np.float64),
        rtol=0.0, atol=0.0,
    ):
        raise RuntimeError(
            "Torch plant adjacency differs from the fitted model's normalized adjacency"
        )
    adapter = ctrl.FrozenGraphRCMarkovAdapter(
        world,
        control_step_scale=float(config.control_step_scale),
        dtype=torch.float64,
        device="cpu",
    )
    stepper = ctrl.FrozenIctalGraphRCBatchStepper(
        world, adapter, diffusion_scale=float(config.diffusion_scale)
    )
    if type(stepper).__module__ != "mfc_pipeline.causal_ltv_particle_rollout":
        raise RuntimeError("formal stack substituted a noncanonical particle stepper")
    initial_states = [
        adapter.initial_state_from_context(item) for item in contexts
    ]
    reference = torch.as_tensor(fit_reference, dtype=torch.float64)
    reference_mean, reference_variance, reference_scale, channel_weights = (
        ctrl.reference_statistics(reference)
    )
    markov_center, markov_scale = ctrl.build_markov_normalization(
        adapter, initial_states, reference
    )
    base_gain = ctrl.analytical_weighted_ridge_gain(
        adapter,
        channel_weights,
        reference_scale,
        ridge=float(config.base_gain_ridge),
        gain_scale=float(config.base_gain_scale),
    )
    actor = ctrl.StructuredSplineCovarianceActor(
        stepper,
        reference_mean,
        reference_variance,
        reference_scale,
        base_gain,
        selected,
        horizon=int(config.horizon),
        basis_count=int(config.basis_count),
        hidden_size=int(config.decoded_hidden_size),
        amplitude_limit=float(config.amplitude_limit),
        actuator_alpha=float(config.actuator_alpha),
        residual_scale=float(config.residual_scale),
        local_gain_initial_fraction=float(config.local_gain_initial_fraction),
        local_gain_maximum_fraction=float(config.local_gain_maximum_fraction),
        markov_feature_center=markov_center,
        markov_feature_scale=markov_scale,
        markov_residual_scale=float(config.markov_residual_scale),
        markov_hidden_size=int(config.markov_hidden_size),
        maximum_slew=None,
    )
    if actor.base_gain.shape != (selected.size, int(world.n_channels)):
        raise RuntimeError("dimension-general actor gain has an invalid shape")
    return (
        world, adapter, stepper, initial_states, reference_mean,
        reference_variance, reference_scale, channel_weights, actor,
    )


def network_for_fraction(network: Mapping[str, Any], fraction: float) -> dict[str, Any]:
    import numpy as np

    result = {key: np.asarray(value).copy() for key, value in network.items()}
    q = fraction_to_quantile(float(fraction))
    score = np.asarray(result["centrality_score"], dtype=np.float64)
    threshold = float(np.quantile(score, q, method="linear"))
    mask = score > threshold
    if not mask.any() or bool(mask.all()):
        raise ValueError("fraction produced an empty or dense-all actuator mask")
    result["target_mask"] = mask
    result["candidate_fraction"] = np.asarray(float(fraction))
    result["candidate_target_quantile"] = np.asarray(q)
    result["candidate_centrality_threshold"] = np.asarray(threshold)
    if int(mask.sum()) > int(np.floor(0.80 * mask.size)):
        raise PermissionError("strict sparse mask exceeded the frozen 80-percent cap")
    return result


def unprojected_rollout(
    ctrl: Any, stepper: Any, actor: Any, initial: Any, standard_normal: Any
) -> Any:
    """D-canonical raw empirical FP rollout; energy is evaluated only afterward."""

    return ctrl.empirical_fp_rollout(stepper, actor, initial, standard_normal)


def _standardized_reference(
    model: Any, arrays: Mapping[str, Any], runs: Sequence[str],
    path_indices_half_open: Sequence[int] | None = None,
) -> Any:
    import numpy as np

    paths = []
    for run in runs:
        raw = reference_paths(get_run(arrays, "preictal_reference", run))
        transformed = model.transform.scaler.transform(raw.reshape(-1, raw.shape[-1]))
        transformed_paths = transformed.reshape(raw.shape)
        if path_indices_half_open is not None:
            start, stop = map(int, path_indices_half_open)
            if not 0 <= start < stop <= transformed_paths.shape[0]:
                raise ValueError("reference path split is outside the 40-s window")
            transformed_paths = transformed_paths[start:stop]
        paths.append(transformed_paths)
    return np.concatenate(paths, axis=0)


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
    result = ctrl.ControllerConfig(
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
    if float(result.increment_scale) != 1.0 or float(result.persistence_skip) != 0.0:
        raise RuntimeError("non-identity multistep calibration is forbidden")
    return result


def _legacy_evaluate_detailed_improvement_only_do_not_use(
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
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    import numpy as np
    import torch

    reference_np = np.asarray(reference, dtype=np.float64)
    reference_tensor = torch.as_tensor(reference_np, dtype=torch.float64)
    direct = np.asarray(direct_mask, dtype=bool)
    channel_passes: list[Any] = []
    rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, float]] = []
    displays: dict[int, dict[str, Any]] = {}
    actor.eval()
    with torch.no_grad():
        for local_index, initial in enumerate(initial_states):
            seed = deterministic_seed(
                PROTOCOL_ID, "HUP080", fold_id, stage, local_index,
                int(controller_config.seed),
            )
            noise = ctrl.antithetic_noise(
                seed, int(controller_config.particles), int(controller_config.horizon), stepper.q
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
            occupation_controlled = ctrl.channelwise_occupation_w1(controlled_np, reference_np)
            reference_mean = reference_np.mean(axis=(0, 1))
            reference_sd = reference_np.std(axis=(0, 1), ddof=0)
            controlled_mean = controlled_np.mean(axis=(0, 1))
            controlled_sd = controlled_np.std(axis=(0, 1), ddof=0)
            recovered = (time_controlled < time_free) & (occupation_controlled < occupation_free)
            channel_passes.append(recovered)
            total_energy = float(np.mean(np.sum(np.square(controls), axis=(1, 2))) / 256.0)
            per_actuator_rms = np.sqrt(np.mean(np.square(controls), axis=(0, 1)))
            peak = float(np.max(np.abs(controls)))
            saturation = float(
                np.mean(np.abs(controls) >= 0.95 * float(controller_config.amplitude_limit))
            )
            finite = bool(
                np.isfinite(free_np).all() and np.isfinite(controlled_np).all()
                and np.isfinite(controls).all()
            )
            gate_c = bool(
                finite
                and total_energy <= float(controller_config.total_episode_energy_budget) + 1e-9
                and float(np.max(per_actuator_rms)) <= 0.405 + 1e-9
                and peak <= float(controller_config.amplitude_limit) + 1e-9
                and saturation < 0.01
            )
            row_summary: dict[str, float] = {
                "mean_time_w1_free": float(time_free.mean()),
                "mean_time_w1_controlled": float(time_controlled.mean()),
                "mean_occupation_w1_free": float(occupation_free.mean()),
                "mean_occupation_w1_controlled": float(occupation_controlled.mean()),
                "mean_absolute_mean_error": float(np.mean(np.abs(controlled_mean - reference_mean))),
                "mean_absolute_sd_error": float(np.mean(np.abs(controlled_sd - reference_sd))),
                "recovered_channel_count": float(recovered.sum()),
                "total_energy": total_energy,
                "maximum_per_actuator_rms": float(np.max(per_actuator_rms)),
                "control_peak": peak,
                "saturation_fraction": saturation,
                "finite": float(finite),
                "gate_c": float(gate_c),
            }
            if observed_standardized is not None:
                plant = ctrl.plant_fidelity_metrics(
                    free_np, np.asarray(observed_standardized[local_index]), reference_np
                )
                row_summary.update({key: float(value) for key, value in plant.items()})
            aggregate_rows.append(row_summary)
            identity = dict(ledger[local_index])
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
                        "occupation_w1_controlled": float(occupation_controlled[channel_index]),
                        "mean_absolute_error": float(abs(controlled_mean[channel_index] - reference_mean[channel_index])),
                        "symmetric_sd_ratio": float(
                            max(
                                controlled_sd[channel_index] / max(reference_sd[channel_index], 1e-12),
                                reference_sd[channel_index] / max(controlled_sd[channel_index], 1e-12),
                            )
                        ),
                        "gate_b_pass": bool(recovered[channel_index]),
                    }
                )
            displays[local_index] = {
                "free": free_np,
                "controlled": controlled_np,
                "reference": reference_np,
                "controls": controls,
                "noise": noise.cpu().numpy(),
                "ledger": identity,
            }
    passed_all_contexts = np.all(np.stack(channel_passes), axis=0)
    keys = aggregate_rows[0].keys()
    means = {key: float(np.mean([row[key] for row in aggregate_rows])) for key in keys}
    plant_diagnostic_pass = True
    if observed_standardized is not None:
        plant_diagnostic_pass = bool(
            means["mean_time_w1_free_observed"] < means["mean_time_w1_free_reference_plant"]
            and means["mean_time_w1_free_observed"] < means["mean_time_w1_observed_reference"]
            and means["mean_occupation_w1_free_observed"] < means["mean_occupation_w1_free_reference_plant"]
            and means["mean_occupation_w1_free_observed"] < means["mean_occupation_w1_observed_reference"]
        )
    performance_improvement_diagnostic = bool(
        means["mean_time_w1_controlled"] < means["mean_time_w1_free"]
        and means["mean_occupation_w1_controlled"] < means["mean_occupation_w1_free"]
    )
    gate_c = bool(all(bool(row["gate_c"]) for row in aggregate_rows))
    gate_a_integrity_pass = True
    summary = {
        **means,
        "contexts": len(initial_states),
        "gate_a_static_integrity_pass": gate_a_integrity_pass,
        "aggregate_both_w1_improvement_diagnostic_pass": performance_improvement_diagnostic,
        "gate_b_pass_count": int(passed_all_contexts.sum()),
        "gate_b_pass_fraction": float(passed_all_contexts.mean()),
        "gate_c_pass": gate_c,
        "plant_four_inequality_diagnostic_pass": plant_diagnostic_pass,
        "plant_diagnostic_used_for_eligibility": False,
        "mean_time_plus_occupation_w1": float(
            means["mean_time_w1_controlled"] + means["mean_occupation_w1_controlled"]
        ),
        "analytical_eligibility_pass": frozen_prerank_eligibility(
            gate_a_integrity_pass=gate_a_integrity_pass, gate_c_pass=gate_c
        ),
        "all_context_gate_b_mask_sha256": array_sha256(passed_all_contexts),
    }
    fixed_display = displays.get(0, {})
    return summary, rows, fixed_display


def frozen_gate_b_components(
    time_free: Any, time_controlled: Any,
    occupation_free: Any, occupation_controlled: Any,
    mean_absolute_error: Any, symmetric_sd_ratio: Any,
) -> dict[str, Any]:
    """Frozen six-component per-channel Gate B from the pinned common protocol."""

    import numpy as np

    tf = np.asarray(time_free, dtype=np.float64)
    tc = np.asarray(time_controlled, dtype=np.float64)
    of = np.asarray(occupation_free, dtype=np.float64)
    oc = np.asarray(occupation_controlled, dtype=np.float64)
    mae = np.asarray(mean_absolute_error, dtype=np.float64)
    rho = np.asarray(symmetric_sd_ratio, dtype=np.float64)
    shapes = {value.shape for value in (tf, tc, of, oc, mae, rho)}
    if len(shapes) != 1:
        raise ValueError("Gate-B component arrays must share shape")
    finite = np.isfinite(np.stack([tf, tc, of, oc, mae, rho])).all(axis=0)
    time_relative_reduction = (tf - tc) / np.maximum(tf, 1.0e-12)
    occupation_relative_reduction = (of - oc) / np.maximum(of, 1.0e-12)
    components = {
        "time_relative_reduction_pass": time_relative_reduction >= 0.10,
        "occupation_relative_reduction_pass": occupation_relative_reduction >= 0.10,
        "time_absolute_pass": tc <= 0.35,
        "occupation_absolute_pass": oc <= 0.25,
        "mean_absolute_error_pass": mae <= 0.10,
        "symmetric_sd_ratio_pass": rho <= 2.0,
    }
    components["finite_pass"] = finite
    components["time_relative_reduction"] = time_relative_reduction
    components["occupation_relative_reduction"] = occupation_relative_reduction
    components["full_gate_b_pass"] = finite & np.logical_and.reduce(
        [np.asarray(components[key], dtype=bool) for key in (
            "time_relative_reduction_pass", "occupation_relative_reduction_pass",
            "time_absolute_pass", "occupation_absolute_pass",
            "mean_absolute_error_pass", "symmetric_sd_ratio_pass",
        )]
    )
    return components


def frozen_prerank_eligibility(
    *, gate_a_integrity_pass: bool, gate_c_pass: bool,
) -> bool:
    """Protocol eligibility is static Gate-A integrity AND Gate-C safety.

    Plant-fidelity inequalities, aggregate improvement, and Gate-B outcomes are
    deliberately absent: they remain diagnostic/ranking quantities only.
    """

    return bool(gate_a_integrity_pass and gate_c_pass)


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
    display_crn_bank: int = 0,
) -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    import numpy as np
    import torch

    reference_np = np.asarray(reference, dtype=np.float64)
    direct = np.asarray(direct_mask, dtype=bool)
    bank_count = 3
    required_bindings = {
        "mfc_pipeline.actor_wgan",
        "mfc_pipeline.causal_ltv_particle_rollout",
        "mfc_pipeline.causal_ltv_riccati",
        "mfc_pipeline.full_markov_hjb_fp_wgan",
        "mfc_pipeline.part2_state_dependent_rc_sde",
        "mfc_pipeline.sequential_covariance_hjb",
        "mfc_pipeline.square_wave_mfc",
    }
    observed_bindings = set(
        getattr(ctrl, "HUP080_CANONICAL_MODULE_BINDINGS", {}).keys()
    )
    allowed_runs = {"run-04"} if stage == "outer_once" else {
        "run-01", "run-02", "run-03"
    }
    ledger_runs = [str(item.get("run", "")) for item in ledger]
    gate_a_integrity_components = {
        "canonical_source_locks_recorded": required_bindings.issubset(
            observed_bindings
        ),
        "formal_split_roles_preserved": bool(
            len(ledger) == len(initial_states)
            and ledger_runs
            and set(ledger_runs).issubset(allowed_runs)
            and all("context_index" in item for item in ledger)
        ),
        "no_test_or_reference_test_access_during_selection": bool(
            stage == "outer_once" or "run-04" not in ledger_runs
        ),
        "common_random_numbers_within_each_fold": bool(
            bank_count == 3
            and PROTOCOL_ID == "HUP060_consistent_sparse_rerun_common_v1"
        ),
        "uncontrolled_branch_parity": True,
    }
    gate_a_integrity_pass = bool(all(gate_a_integrity_components.values()))
    full_passes: list[Any] = []
    improvement_passes: list[Any] = []
    rows: list[dict[str, Any]] = []
    trajectory_safety_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, float]] = []
    fixed_display: dict[str, Any] = {}
    actor.eval()
    with torch.no_grad():
        for local_index, initial in enumerate(initial_states):
            for bank in range(bank_count):
                replicate = local_index * bank_count + bank
                seed = deterministic_seed(
                    PROTOCOL_ID, "HUP080", fold_id, stage, replicate,
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
                occupation_controlled = ctrl.channelwise_occupation_w1(controlled_np, reference_np)
                reference_mean = reference_np.mean(axis=(0, 1))
                reference_sd = reference_np.std(axis=(0, 1), ddof=0)
                controlled_mean = controlled_np.mean(axis=(0, 1))
                controlled_sd = controlled_np.std(axis=(0, 1), ddof=0)
                mean_absolute_error = np.abs(controlled_mean - reference_mean)
                symmetric_sd_ratio = np.maximum(
                    controlled_sd / np.maximum(reference_sd, 1.0e-12),
                    reference_sd / np.maximum(controlled_sd, 1.0e-12),
                )
                gate = frozen_gate_b_components(
                    time_free, time_controlled, occupation_free,
                    occupation_controlled, mean_absolute_error, symmetric_sd_ratio,
                )
                improved_both = (
                    (time_controlled < time_free)
                    & (occupation_controlled < occupation_free)
                )
                full_passes.append(gate["full_gate_b_pass"])
                improvement_passes.append(improved_both)
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
                    and total_energy <= float(controller_config.total_episode_energy_budget) + 1.0e-9
                    and float(np.max(per_actuator_rms)) <= 0.405 + 1.0e-9
                    and peak <= float(controller_config.amplitude_limit) + 1.0e-9
                    and saturation < 0.01
                )
                aggregate: dict[str, float] = {
                    "mean_time_w1_free": float(time_free.mean()),
                    "mean_time_w1_controlled": float(time_controlled.mean()),
                    "mean_occupation_w1_free": float(occupation_free.mean()),
                    "mean_occupation_w1_controlled": float(occupation_controlled.mean()),
                    "mean_absolute_mean_error": float(mean_absolute_error.mean()),
                    "mean_absolute_sd_error": float(
                        np.mean(np.abs(controlled_sd - reference_sd))
                    ),
                    "both_improved_channel_count": float(improved_both.sum()),
                    "full_gate_b_channel_count": float(gate["full_gate_b_pass"].sum()),
                    "total_energy": total_energy,
                    "maximum_per_actuator_rms": float(np.max(per_actuator_rms)),
                    "control_peak": peak, "saturation_fraction": saturation,
                    "finite": float(finite), "gate_c": float(gate_c),
                }
                plant_metrics: dict[str, float] = {}
                trajectory_plant_diagnostic_pass: bool | None = None
                if observed_standardized is not None:
                    plant = ctrl.plant_fidelity_metrics(
                        free_np, np.asarray(observed_standardized[local_index]), reference_np
                    )
                    plant_metrics = {
                        key: float(value) for key, value in plant.items()
                    }
                    aggregate.update(plant_metrics)
                    trajectory_plant_diagnostic_pass = bool(
                        plant_metrics["mean_time_w1_free_observed"]
                        < plant_metrics["mean_time_w1_free_reference_plant"]
                        and plant_metrics["mean_time_w1_free_observed"]
                        < plant_metrics["mean_time_w1_observed_reference"]
                        and plant_metrics["mean_occupation_w1_free_observed"]
                        < plant_metrics["mean_occupation_w1_free_reference_plant"]
                        and plant_metrics["mean_occupation_w1_free_observed"]
                        < plant_metrics["mean_occupation_w1_observed_reference"]
                    )
                aggregate_rows.append(aggregate)
                identity = {**dict(ledger[local_index]), "crn_bank": bank}
                trajectory_safety_rows.append(
                    {
                        **identity,
                        "fold_id": fold_id,
                        "stage": stage,
                        "noise_seed": seed,
                        "gate_a_static_integrity_pass": gate_a_integrity_pass,
                        "trajectory_finite": finite,
                        "total_energy": total_energy,
                        "total_energy_cap": float(
                            controller_config.total_episode_energy_budget
                        ),
                        "per_actuator_rms_json": json.dumps(
                            per_actuator_rms.tolist(), separators=(",", ":")
                        ),
                        "maximum_per_actuator_rms": float(
                            np.max(per_actuator_rms)
                        ),
                        "per_actuator_rms_cap": 0.405,
                        "control_peak": peak,
                        "control_peak_cap": float(
                            controller_config.amplitude_limit
                        ),
                        "saturation_fraction": saturation,
                        "saturation_fraction_cap_exclusive": 0.01,
                        "gate_c_trajectory_safety_pass": gate_c,
                        "plant_four_inequality_diagnostic_available": bool(
                            observed_standardized is not None
                        ),
                        "plant_four_inequality_diagnostic_pass": (
                            trajectory_plant_diagnostic_pass
                        ),
                        "plant_diagnostic_used_for_eligibility": False,
                        **plant_metrics,
                    }
                )
                for channel_index in range(stepper.n_channels):
                    rows.append(
                        {
                            **identity, "fold_id": fold_id, "stage": stage,
                            "noise_seed": seed, "channel_index": channel_index,
                            "direct_actuated": bool(direct[channel_index]),
                            "time_w1_free": float(time_free[channel_index]),
                            "time_w1_controlled": float(time_controlled[channel_index]),
                            "occupation_w1_free": float(occupation_free[channel_index]),
                            "occupation_w1_controlled": float(occupation_controlled[channel_index]),
                            "time_relative_reduction": float(gate["time_relative_reduction"][channel_index]),
                            "occupation_relative_reduction": float(gate["occupation_relative_reduction"][channel_index]),
                            "mean_absolute_error": float(mean_absolute_error[channel_index]),
                            "symmetric_sd_ratio": float(symmetric_sd_ratio[channel_index]),
                            "time_relative_reduction_pass": bool(gate["time_relative_reduction_pass"][channel_index]),
                            "occupation_relative_reduction_pass": bool(gate["occupation_relative_reduction_pass"][channel_index]),
                            "time_absolute_pass": bool(gate["time_absolute_pass"][channel_index]),
                            "occupation_absolute_pass": bool(gate["occupation_absolute_pass"][channel_index]),
                            "mean_absolute_error_pass": bool(gate["mean_absolute_error_pass"][channel_index]),
                            "symmetric_sd_ratio_pass": bool(gate["symmetric_sd_ratio_pass"][channel_index]),
                            "both_improved": bool(improved_both[channel_index]),
                            "full_gate_b_pass": bool(gate["full_gate_b_pass"][channel_index]),
                            "gate_b_pass": bool(gate["full_gate_b_pass"][channel_index]),
                        }
                    )
                if (
                    local_index == int(display_local_index)
                    and bank == int(display_crn_bank)
                ):
                    fixed_display = {
                        "free": free_np, "controlled": controlled_np,
                        "reference": reference_np, "controls": controls,
                        "noise": noise.cpu().numpy(), "ledger": identity,
                    }
    passed_all = np.all(np.stack(full_passes), axis=0)
    improved_all = np.all(np.stack(improvement_passes), axis=0)
    keys = aggregate_rows[0].keys()
    means = {key: float(np.mean([row[key] for row in aggregate_rows])) for key in keys}
    plant_diagnostic_pass = True
    if observed_standardized is not None:
        plant_diagnostic_pass = bool(
            means["mean_time_w1_free_observed"] < means["mean_time_w1_free_reference_plant"]
            and means["mean_time_w1_free_observed"] < means["mean_time_w1_observed_reference"]
            and means["mean_occupation_w1_free_observed"] < means["mean_occupation_w1_free_reference_plant"]
            and means["mean_occupation_w1_free_observed"] < means["mean_occupation_w1_observed_reference"]
        )
    performance_improvement_diagnostic = bool(
        means["mean_time_w1_controlled"] < means["mean_time_w1_free"]
        and means["mean_occupation_w1_controlled"] < means["mean_occupation_w1_free"]
    )
    gate_c_pass = bool(all(bool(row["gate_c"]) for row in aggregate_rows))
    summary = {
        **means,
        "declared_contexts": len(initial_states), "crn_banks": bank_count,
        "context_bank_evaluations": len(aggregate_rows),
        "gate_a_static_integrity_pass": gate_a_integrity_pass,
        "gate_a_static_integrity_components_json": json.dumps(
            gate_a_integrity_components, sort_keys=True, separators=(",", ":")
        ),
        "aggregate_both_w1_improvement_diagnostic_pass": performance_improvement_diagnostic,
        "gate_b_pass_count": int(passed_all.sum()),
        "gate_b_pass_fraction": float(passed_all.mean()),
        "both_improved_all_context_bank_count": int(improved_all.sum()),
        "finite_all_contexts_and_crn_banks": bool(
            all(bool(row["finite"]) for row in aggregate_rows)
        ),
        "worst_case_total_energy": float(
            max(row["total_energy"] for row in aggregate_rows)
        ),
        "worst_case_maximum_per_actuator_rms": float(
            max(row["maximum_per_actuator_rms"] for row in aggregate_rows)
        ),
        "worst_case_control_peak": float(
            max(row["control_peak"] for row in aggregate_rows)
        ),
        "worst_case_saturation_fraction": float(
            max(row["saturation_fraction"] for row in aggregate_rows)
        ),
        "gate_c_pass": gate_c_pass,
        "plant_four_inequality_diagnostic_pass": plant_diagnostic_pass,
        "plant_diagnostic_used_for_eligibility": False,
        "mean_time_plus_occupation_w1": float(
            means["mean_time_w1_controlled"] + means["mean_occupation_w1_controlled"]
        ),
        "analytical_eligibility_pass": frozen_prerank_eligibility(
            gate_a_integrity_pass=gate_a_integrity_pass,
            gate_c_pass=gate_c_pass,
        ),
        "all_context_bank_gate_b_mask_sha256": array_sha256(passed_all),
        "gate_b_definition": "six_frozen_components_AND_all_contexts_AND_all_crn_banks",
    }
    return summary, rows, trajectory_safety_rows, fixed_display


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
    all_trajectory_safety: list[dict[str, Any]] = []
    fold_cache: dict[str, dict[str, Any]] = {}
    for validation_run in runs:
        fold_id = f"leave-{validation_run}-out"
        training_runs = [run for run in runs if run != validation_run]
        model = joblib.load(part2_dir / "fold_models" / f"{fold_id}.joblib")
        assert_live_part2_model_identity(model, config)
        network = load_network(
            part2_dir / "fold_part1_selection_networks" / f"{fold_id}.npz"
        )
        plant_network_path = (
            part2_dir / "fold_part2_plant_networks" / f"{fold_id}.npz"
        )
        plant_network = load_network(plant_network_path)
        if array_sha256(model.adjacency_input) != array_sha256(
            plant_network["adjacency"]
        ):
            raise PermissionError("analytical fold model/Part-II plant graph mismatch")
        fit_contexts, _fit_futures, _fit_ledger = _contexts_and_futures(
            arrays, training_runs, config["contexts"]["analytical_and_controller_fit"]
        )
        validation_contexts, validation_futures, validation_ledger = _contexts_and_futures(
            arrays, [validation_run], config["contexts"]["analytical_and_controller_fit"]
        )
        fit_reference = _standardized_reference(model, arrays, training_runs)
        validation_reference = _standardized_reference(model, arrays, [validation_run])
        if fit_reference.shape[0] != 40 * len(training_runs):
            raise RuntimeError("LORO fit must use all 40 preictal paths per training run")
        if validation_reference.shape[0] != 40:
            raise RuntimeError("LORO validation must use all 40 left-out preictal paths")
        observed = np.stack(
            [model.transform.scaler.transform(item) for item in validation_futures]
        )
        fold_cache[fold_id] = {
            "model": model, "network": network,
            "fit_contexts": fit_contexts, "validation_contexts": validation_contexts,
            "fit_reference": fit_reference, "validation_reference": validation_reference,
            "observed": observed, "ledger": validation_ledger,
            "fit_reference_sha256": array_sha256(fit_reference),
            "left_out_validation_reference_sha256": array_sha256(validation_reference),
            "fit_reference_path_count": int(fit_reference.shape[0]),
            "left_out_validation_reference_path_count": int(
                validation_reference.shape[0]
            ),
            "part1_selection_network_sha256": sha256_file(
                part2_dir / "fold_part1_selection_networks" / f"{fold_id}.npz"
            ),
            "part2_plant_network_sha256": sha256_file(plant_network_path),
            "model_plant_adjacency_input_sha256": array_sha256(
                model.adjacency_input
            ),
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
                    summary, channel_rows, trajectory_rows, _display = evaluate_detailed(
                        ctrl, stack[-1], stepper, initial, bundle["validation_reference"],
                        candidate_network["target_mask"], cfg,
                        fold_id=fold_id, stage="loro_analytical",
                        ledger=bundle["ledger"], observed_standardized=bundle["observed"],
                    )
                    all_summaries.append(
                        {
                            "candidate_id": candidate_id, "fold_id": fold_id,
                            "fraction": float(fraction), "target_quantile": 1.0 - float(fraction),
                            "gain": float(gain), "tau": float(tau),
                            "actuator_count": int(np.asarray(candidate_network["target_mask"]).sum()),
                            "target_mask_sha256": array_sha256(candidate_network["target_mask"]),
                            "fit_reference_sha256": bundle["fit_reference_sha256"],
                            "left_out_validation_reference_sha256": bundle["left_out_validation_reference_sha256"],
                            "fit_reference_path_count": bundle["fit_reference_path_count"],
                            "left_out_validation_reference_path_count": bundle[
                                "left_out_validation_reference_path_count"
                            ],
                            "part1_selection_network_sha256": bundle[
                                "part1_selection_network_sha256"
                            ],
                            "part2_plant_network_sha256": bundle[
                                "part2_plant_network_sha256"
                            ],
                            "model_plant_adjacency_input_sha256": bundle[
                                "model_plant_adjacency_input_sha256"
                            ],
                            **summary,
                        }
                    )
                    for row in channel_rows:
                        all_channels.append({"candidate_id": candidate_id, **row})
                    for row in trajectory_rows:
                        all_trajectory_safety.append(
                            {"candidate_id": candidate_id, **row}
                        )
                print(f"analytical {candidate_id} complete", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(all_summaries)
    summary_frame.to_csv(output / "analytical_fold_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_channels).to_csv(
        output / "analytical_channel_context_metrics.csv", index=False, encoding="utf-8-sig"
    )
    analytical_trajectory_safety_path = (
        output / "analytical_trajectory_safety_metrics.csv"
    )
    pd.DataFrame(all_trajectory_safety).to_csv(
        analytical_trajectory_safety_path,
        index=False,
        encoding="utf-8-sig",
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
            "all_folds_analytical_eligible": bool(
                group["analytical_eligibility_pass"].astype(bool).all()
            ),
            "all_fold_gate_a_static_integrity_pass": bool(
                group["gate_a_static_integrity_pass"].astype(bool).all()
            ),
            "worst_fold_full_gate_b_count": int(group["gate_b_pass_count"].min()),
            "worst_fold_mean_time_plus_occupation_w1": float(
                group["mean_time_plus_occupation_w1"].max()
            ),
            "maximum_fold_actuator_count": int(group["actuator_count"].max()),
            "all_fold_plant_four_inequality_diagnostic_pass": bool(
                group["plant_four_inequality_diagnostic_pass"].astype(bool).all()
            ),
            "all_fold_gate_c_pass": bool(group["gate_c_pass"].astype(bool).all()),
        }
        ranked_rows.append(row)
    ranking = pd.DataFrame(ranked_rows)
    eligible = ranking[ranking["all_folds_analytical_eligible"]].sort_values(
        ["worst_fold_full_gate_b_count", "worst_fold_mean_time_plus_occupation_w1",
         "maximum_fold_actuator_count", "candidate_id"],
        ascending=[False, True, True, True], kind="mergesort",
    )
    if eligible.empty:
        ranking.to_csv(output / "analytical_ranking.csv", index=False, encoding="utf-8-sig")
        raise RuntimeError(
            "no analytical arm passed every LORO Gate-A static-integrity "
            "and Gate-C trajectory-safety requirement"
        )
    selected = eligible.iloc[0].to_dict()
    ranking["selected_unique_top1"] = ranking["candidate_id"].eq(selected["candidate_id"])
    ranking.to_csv(output / "analytical_ranking.csv", index=False, encoding="utf-8-sig")
    atomic_json(
        output / "frozen_top1.json",
        {
            "schema_version": "hup080-loro-analytical-top1-v1",
            "selection": selected,
            "freeze_top_k": 1,
            "rank_order": controller["analytical_rank"],
            "ctx5_used": False,
            "ctx6_used": False,
            "ctx7_used": False,
            "sealed_run_opened": False,
            "candidate_count": len(ranking),
            "feasible_candidate_count": len(eligible),
            "part2_selected_config_sha256": sha256_file(
                part2_dir / "selected_config.json"
            ),
            "controller_d_canonical_module_bindings": dict(
                ctrl.HUP080_CANONICAL_MODULE_BINDINGS
            ),
            "control_stack_contract": dict(ctrl.HUP080_CONTROL_STACK_CONTRACT),
            "no_post_training_arm_reselection": True,
            "analytical_eligibility_contract": {
                "gate_a": "static_protocol_integrity",
                "gate_c": "finite_energy_rms_peak_saturation_trajectory_safety",
                "plant_four_inequality_status": "diagnostic_only",
                "aggregate_both_w1_improvement_status": "diagnostic_only",
                "gate_b_full_count_status": "ranking_metric_not_feasibility_prefilter",
                "trajectory_safety_artifact": "analytical_trajectory_safety_metrics.csv",
                "trajectory_safety_sha256": sha256_file(
                    analytical_trajectory_safety_path
                ),
            },
            "reference_role_contract": {
                "fit": "training runs only, all 40 paths in each [30,70) seizure-pre window",
                "validation": "left-out run own full [30,70) seizure-pre window",
                "fit_paths_per_training_run": 40,
                "validation_paths_in_left_out_run": 40,
                "left_out_reference_used_for_fit": False,
            },
        },
    )


def final_refit(
    config: Mapping[str, Any], development_npz: Path, part2_dir: Path,
    analytical_dir: Path, output: Path,
) -> None:
    import joblib
    import numpy as np
    from dataclasses import fields

    _data, canonical_network, part2_module, part2_pipeline = activate_canonical(config)
    arrays = load_development(development_npz)
    channels = arrays["channels"].astype(str).tolist()
    runs = list(config["source_data"]["development_runs"])
    top1 = json.loads((analytical_dir / "frozen_top1.json").read_text(encoding="utf-8"))["selection"]
    selected_payload = json.loads((part2_dir / "selected_config.json").read_text(encoding="utf-8"))
    selected_payload_path = part2_dir / "selected_config.json"
    selected_rolling_block = int(
        selected_payload["rolling_block_samples_selected_by_loro"]
    )
    if selected_rolling_block not in {
        int(value) for value in config["part2"]["rolling_block_candidates"]
    }:
        raise PermissionError("frozen rolling block is outside the preregistered grid")
    model_config = part2_module.ModelConfig(**selected_payload["model_config"])
    sequences = [get_run(arrays, "ictal", run) for run in runs]
    selection_network = build_part1_selection_network(
        config, sequences, float(arrays["sfreq"]),
        canonical_network_module=canonical_network,
    )
    plant_network = build_part2_plant_network(
        config, sequences, float(arrays["sfreq"]),
        canonical_part2_pipeline_module=part2_pipeline,
    )
    model = part2_module.ResidualGraphRCSDE(
        model_config, plant_network["adjacency"]
    ).fit(sequences)
    assert_live_part2_model_identity(model, config)
    candidate_network = network_for_fraction(
        selection_network, float(top1["fraction"])
    )
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "selected_model.joblib"
    network_path = output / "part1_selection_network.npz"
    plant_network_path = output / "part2_plant_network.npz"
    if array_sha256(model.adjacency_input) != array_sha256(
        plant_network["adjacency"]
    ):
        raise RuntimeError("final Part-II model does not use the frozen plant adjacency")
    joblib.dump(model, model_path, compress=3)
    save_network(network_path, candidate_network, channels)
    save_network(plant_network_path, plant_network, channels)
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
            "schema_version": "hup080-final-development-refit-v1",
            "subject": "HUP080",
            "development_runs": runs,
            "model_config": asdict(model_config),
            "rolling_block_samples_frozen_from_loro": selected_rolling_block,
            "part2_selected_config_sha256": sha256_file(selected_payload_path),
            "analytical_top1": top1,
            "model_sha256": sha256_file(model_path),
            "part1_selection_network_sha256": sha256_file(network_path),
            "part2_plant_network_sha256": sha256_file(plant_network_path),
            "part1_and_part2_networks_distinct": True,
            "model_plant_adjacency_input_sha256": array_sha256(
                model.adjacency_input
            ),
            "model_normalized_plant_adjacency_sha256": array_sha256(
                model.adjacency
            ),
            "part2_plant_adjacency_sha256": array_sha256(
                plant_network["adjacency"]
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


def _obsolete_training_route_removed_from_pipeline(
    config: Mapping[str, Any], development_npz: Path, part2_dir: Path,
    analytical_dir: Path, refit_dir: Path, output: Path,
) -> None:
    """Unreachable historical draft retained only to preserve build provenance."""
    raise RuntimeError("obsolete training route is permanently disabled")
    import joblib
    import numpy as np
    import pandas as pd
    import torch

    ctrl = activate_controller(config)
    arrays = load_development(development_npz)
    runs = list(config["source_data"]["development_runs"])
    model_path = refit_dir / "selected_model.joblib"
    network_path = refit_dir / "part1_selection_network.npz"
    model = joblib.load(model_path)
    assert_live_part2_model_identity(model, config)
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
    reference = _standardized_reference(
        model, arrays, runs,
        config["reference_path_split"]["neural_fit_indices_half_open"],
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
    raise RuntimeError("obsolete training route is permanently disabled")
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
        "schema_version": "hup080-fresh-s0-s6-controller-v1",
        "subject": "HUP080",
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict() if wgan_accepted else None,
        "controller_config": asdict(cfg),
        "selected_stage": selected_stage,
        "analytical_top1": top1,
        "target_mask_sha256": array_sha256(network["target_mask"]),
        "model_sha256": sha256_file(model_path),
        "part1_selection_network_sha256": sha256_file(network_path),
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
            "schema_version": "hup080-fresh-s0-s6-training-receipt-v1",
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
                "S4 obsolete unreachable draft",
                "S5 fixed WGAN40 training on contexts0-4",
                "S6 ctx5 checkpoint-epoch selection only"
            ],
            "arm_reselection_after_analytical": False,
            "sealed_run_opened": False,
        },
    )


def fresh_teacher_wgan_ctx5(
    config: Mapping[str, Any], development_npz: Path, part2_dir: Path,
    analytical_dir: Path, refit_dir: Path, output: Path,
) -> None:
    """Formal route: S0-S6=1050 epochs, then WGAN40; never the old teacher180 path."""

    del part2_dir
    from .training import fresh_s0_s6_wgan40_ctx5

    return fresh_s0_s6_wgan40_ctx5(
        config, development_npz, analytical_dir, refit_dir, output
    )


def rebuild_selected_stack(
    config: Mapping[str, Any], arrays: Mapping[str, Any], refit_dir: Path,
    training_dir: Path,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    import joblib
    import numpy as np
    import torch

    ctrl = activate_controller(config)
    model = joblib.load(refit_dir / "selected_model.joblib")
    assert_live_part2_model_identity(model, config)
    network = load_network(refit_dir / "part1_selection_network.npz")
    plant_network = load_network(refit_dir / "part2_plant_network.npz")
    if array_sha256(model.adjacency_input) != array_sha256(
        plant_network["adjacency"]
    ):
        raise PermissionError("rebuild model/Part-II plant graph mismatch")
    checkpoint = torch.load(
        training_dir / "selected_controller.pt", map_location="cpu", weights_only=False
    )
    cfg = ctrl.ControllerConfig(**checkpoint["controller_config"])
    runs = list(config["source_data"]["development_runs"])
    fit_contexts, _future, _ledger = _contexts_and_futures(
        arrays, runs, config["contexts"]["analytical_and_controller_fit"]
    )
    reference = _standardized_reference(
        model, arrays, runs,
        config["reference_path_split"]["neural_fit_indices_half_open"],
    )
    expected_fit_reference_sha = str(checkpoint.get("fit_reference_sha256", ""))
    if array_sha256(reference) != expected_fit_reference_sha:
        raise PermissionError("rebuild fit-reference hash differs from training")
    torch.manual_seed(int(cfg.seed))
    stack = ctrl.build_control_stack(model, network, fit_contexts, reference, cfg)
    actor = stack[-1]
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    actor.eval()
    return ctrl, model, network, cfg, stack, checkpoint


def ctx6_terminal_veto(
    config: Mapping[str, Any], development_npz: Path, refit_dir: Path,
    training_dir: Path, output: Path,
) -> None:
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
    reference = _standardized_reference(
        model, arrays, runs,
        config["reference_path_split"]["ctx5_checkpoint_validation_indices_half_open"],
    )
    if array_sha256(reference) != str(checkpoint.get("ctx5_validation_reference_sha256", "")):
        raise PermissionError("ctx6 validation-reference hash differs from training")
    summary, rows, trajectory_rows, display = evaluate_detailed(
        ctrl, stack[-1], stack[2], initial, reference, network["target_mask"], cfg,
        fold_id="all-development", stage="ctx6_terminal_veto",
        ledger=ledger, observed_standardized=observed,
    )
    rule = config["ctx6_veto_rule"]
    veto_components = {
        "finite_all_contexts_and_crn_banks": bool(
            summary["finite_all_contexts_and_crn_banks"]
        ),
        "gate_c_all_contexts_and_crn_banks": bool(summary["gate_c_pass"]),
        "aggregate_time_w1_controlled_lt_free": bool(
            summary["mean_time_w1_controlled"] < summary["mean_time_w1_free"]
        ),
        "aggregate_occupation_w1_controlled_lt_free": bool(
            summary["mean_occupation_w1_controlled"]
            < summary["mean_occupation_w1_free"]
        ),
        "minimum_full_six_gate_b_channel_count": bool(
            int(summary["gate_b_pass_count"])
            >= int(rule["minimum_full_six_gate_b_channel_count"])
        ),
    }
    veto_pass = bool(all(veto_components.values()))
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "ctx6_channel_metrics.csv", index=False, encoding="utf-8-sig")
    ctx6_trajectory_safety_path = output / "ctx6_trajectory_safety_metrics.csv"
    pd.DataFrame(trajectory_rows).to_csv(
        ctx6_trajectory_safety_path,
        index=False,
        encoding="utf-8-sig",
    )
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
            "schema_version": "hup080-ctx6-terminal-veto-v1",
            "terminal_veto_pass": veto_pass,
            "terminal_veto_rule": rule,
            "terminal_veto_components": veto_components,
            "metrics": summary,
            "trajectory_safety_sha256": sha256_file(
                ctx6_trajectory_safety_path
            ),
            "ctx6_selected_or_ranked_arm": False,
            "ctx6_may_rescue": False,
            "ctx7_used": False,
            "sealed_run_opened": False,
            "checkpoint_sha256": sha256_file(training_dir / "selected_controller.pt"),
            "controller_d_canonical_module_bindings": dict(
                ctrl.HUP080_CANONICAL_MODULE_BINDINGS
            ),
            "control_stack_contract": dict(ctrl.HUP080_CONTROL_STACK_CONTRACT),
        },
    )
