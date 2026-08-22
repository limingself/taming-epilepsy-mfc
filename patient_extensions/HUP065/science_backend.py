#!/usr/bin/env python
"""Real scientific phase backend for the fresh HUP065 sparse rerun.

The module is imported only after runner authorization and lock checks.  No
scientific dependency is imported until the writable runtime/cache firewall is
installed.  Every dispatch below calls a numerical implementation; there is no
receipt-only or simulated science path.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT = Path(__file__).resolve().parent
DEFAULT_RUNTIME = ROOT / ".runtime"
PHASE_OUTPUT_ROOT_DIRECTORY = "p"


def _install_runtime_firewall(runtime: Path) -> dict[str, str]:
    """Install writable cache paths before MNE/matplotlib/science imports."""

    if "mne" in sys.modules or "matplotlib" in sys.modules:
        raise RuntimeError("MNE/matplotlib was imported before the runtime firewall")
    runtime = Path(runtime).resolve()
    mapping = {
        "MNE_HOME": runtime / "mne_home",
        "_MNE_FAKE_HOME_DIR": runtime / "mne_home",
        "MNE_DATA": runtime / "mne_data",
        "MPLCONFIGDIR": runtime / "matplotlib",
        "XDG_CACHE_HOME": runtime / "xdg_cache",
        "JOBLIB_TEMP_FOLDER": runtime / "joblib",
        "TMPDIR": runtime / "tmp",
        "TEMP": runtime / "tmp",
        "TMP": runtime / "tmp",
    }
    runtime.mkdir(parents=True, exist_ok=True)
    for key, directory in mapping.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    os.environ["MNE_LOGGING_LEVEL"] = "ERROR"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    if not sys.dont_write_bytecode or os.environ["PYTHONDONTWRITEBYTECODE"] != "1":
        raise RuntimeError("bytecode firewall is inactive")
    return {key: str(value) for key, value in mapping.items()}


# This is intentionally the only module-import-time write and occurs only after
# the parent runner has explicit science authorization.  It prevents MNE from
# ever resolving an unwritable user-profile cache directory.
_IMPORT_RUNTIME_RECEIPT = _install_runtime_firewall(DEFAULT_RUNTIME)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_code_references(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    receipts = []
    for name, pair in sorted(config["code_reference_locks"].items()):
        path, expected = Path(pair[0]).resolve(), str(pair[1]).lower()
        observed = _sha256(path)
        if observed != expected:
            raise PermissionError(
                f"code reference changed: {name}; expected={expected}, observed={observed}"
            )
        receipts.append({"name": name, "path": str(path), "sha256": observed})
    return receipts


def _science_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt the frozen public config to the numerical module interface."""

    p2 = config["part2"]
    p3 = config["part3"]
    training = p3["training"]
    split = config["split"]
    screen = config["analytical_screen"]
    return {
        "schema_version": config["schema_version"],
        "subject": "HUP065",
        "shared_protocol_sha256": config["shared_protocol"]["sha256"],
        "project_root": config["code_root"],
        "runtime_root": config["runtime_root"],
        "canonical_locks": dict(config["canonical"]["locks"]),
        "adaptation_locks": dict(config["code_reference_locks"]),
        "source_data": {
            "zip_root": config["raw_data"]["zip_root"],
            "raw_zip": config["raw_data"]["subject_archive"],
            "raw_zip_sha256": "NOT_COMPUTED_WHOLE_ZIP_RUN03_PAYLOAD_IS_SEALED",
            "split_proposal": config["canonical"]["locks"]["formal_patient_split_proposal"][0],
            "split_proposal_sha256": config["canonical"]["locks"]["formal_patient_split_proposal"][1],
            "task": "ictal",
            "expected_channels": 64,
            "development_runs": ["run-01", "run-02"],
            "sealed_run": "run-03",
            "ictal_window_absolute_seconds_half_open": [120.0, 152.0],
            "development_reference_window_absolute_seconds_half_open": [30.0, 70.0],
            "sealed_reference_window_absolute_seconds_half_open": [75.0, 115.0],
            "reference_label": "preictal_reference",
            "reference_isolation": "fold-aligned seizure-level",
            "forbidden_task": "interictal",
        },
        "preprocessing": {
            "target_sampling_rate_hz": float(config["preprocessing"]["target_sampling_rate_hz"]),
            "bandpass_hz": list(config["preprocessing"]["bandpass_hz"]),
            "filter_order": int(config["preprocessing"]["filter_order"]),
            "filter_burn_in_s": float(config["preprocessing"]["filter_burn_in_seconds"]),
            "reference": config["preprocessing"]["reference"],
            "resampling_mode": config["preprocessing"]["resampling_mode"],
            "causal_fir_half_length_factor": int(
                config["preprocessing"]["causal_fir_half_length_factor"]
            ),
            "use_existing_cache": False,
        },
        "part1": {
            "bands_hz": {
                name: list(band)
                for name, band in zip(
                    ("delta", "theta", "alpha", "beta", "gamma"),
                    config["part1"]["plv_bands_hz"],
                )
            },
            "plv_window_s": float(config["part1"]["plv_window_seconds"]),
            "plv_overlap": float(config["part1"]["plv_overlap"]),
            "band_aggregation": config["part1"]["band_aggregation"],
            "network_density": float(config["part1"]["graph_density"]),
            "graph_rule": config["part1"]["graph_rule"],
            "centrality_weights": dict(config["part1"]["centrality_weights"]),
            "clinical_labels_used_for_selection": False,
        },
        "part2": {
            "plant_network_source": p2["plant_network_source"],
            "plant_plv_window_seconds": float(p2["plant_plv_window_seconds"]),
            "plant_plv_window_overlap_fraction": float(
                p2["plant_plv_window_overlap_fraction"]
            ),
            "plant_graph_density": float(p2["plant_graph_density"]),
            "part1_graph_may_be_reused_as_plant_graph": bool(
                p2["part1_graph_may_be_reused_as_plant_graph"]
            ),
            "grid": {
                "reservoir_sizes": list(p2["reservoir_sizes"]),
                "spectral_radii": list(p2["spectral_radii"]),
                "leak_rates": list(p2["leak_rates"]),
                "input_scales": list(p2["input_scales"]),
                "delay_sets_samples": list(p2["delay_sets_samples"]),
                "ridge_alphas": list(p2["ridge_alphas"]),
            },
            "candidate_count": int(p2["candidate_count"]),
            "reservoir_sparsity": float(p2["reservoir_sparsity"]),
            "washout_samples": int(p2["washout_samples"]),
            "random_seed": int(p2["random_seed"]),
            "latent_variance_threshold": float(p2["latent_variance_threshold"]),
            "latent_components_min": int(p2["latent_components_min"]),
            "latent_components_max": int(p2["latent_components_max"]),
            "diffusion_shrinkage": float(p2["diffusion_shrinkage"]),
            "diffusion_relative_jitter": float(p2["diffusion_relative_jitter"]),
            "diffusion_mode": p2["diffusion_mode"],
            "diffusion_ridge_alphas": list(p2["diffusion_ridge_alphas"]),
            "effective_diffusion_multiplier": float(p2["effective_diffusion_multiplier"]),
            "context_samples": int(p2["context_samples"]),
            "validation_windows": int(p2["validation_windows"]),
            "validation_horizons_samples": [16, 32, 64, 128],
            "objective_weights": dict(p2["direct_validation_objective"]),
            "fold_aggregation": p2["fold_aggregation"],
            "rolling_block_candidates_samples": list(
                p2["rolling_block_candidates_samples"]
            ),
            "rolling_display_samples": int(p2["rolling_display_samples"]),
            "rolling_selection_min_correlation": float(
                p2["rolling_selection_min_correlation"]
            ),
            "rolling_selection_max_nrmse": float(
                p2["rolling_selection_max_nrmse"]
            ),
            "rolling_no_eligible_block": p2["rolling_no_eligible_block"],
        },
        "controller": {
            "fraction_grid": list(screen["actuator_fraction_grid"]),
            "tau_grid": list(screen["graph_diffusion_time_grid"]),
            "gain_grid": list(screen["analytical_base_gain_grid"]),
            "analytical_candidate_count": 120,
            "horizon_samples": 256,
            "particles": 32,
            "control_step_scale": float(p3["actor"]["control_step_scale"]),
            "amplitude_limit": float(config["energy"]["peak_control_max"]),
            "per_actuator_rms_max": float(config["energy"]["per_actuator_rms_max"]),
            "total_energy_cap": float(config["energy"]["total_energy_cap"]),
            "base_gain_ridge": 0.20,
            "markov_residual_scale": float(p3["actor"]["markov_residual_scale"]),
            "teacher_epochs_total": 1050,
            "s3_to_s4_modern_actor_initialization_seed": int(
                training["s3_to_s4_modern_actor_initialization_seed"]
            ),
            "teacher_stage_ids": list(training["teacher_formal_stage_ids"]),
            "teacher_stages": list(training["teacher_stages"]),
            "objective_weights": dict(training["objective_weights"]),
            "discarded_s2r_executed": False,
            "wgan_epochs": 40,
            "validation_every_epochs": int(training["validation_every_epochs"]),
            "actor_learning_rate": float(training["actor_learning_rate"]),
            "critic_learning_rate": float(training["critic_learning_rate"]),
            "critic_pretrain_steps": int(training["critic_pretrain_steps"]),
            "critic_pretrain_banks": int(training["critic_pretrain_banks"]),
            "critic_steps": int(training["critic_updates_per_epoch"]),
            "adversarial_weight": float(training["adversarial_weight"]),
            "teacher_anchor_weight": float(training["action_anchor_weight"]),
            "seed": int(training["seed"]),
            "validation_seed": int(training["validation_seed"]),
            "teacher_rollout_base_seed": int(training["teacher_rollout_base_seed"]),
            "teacher_rollout_seed_rule": training["teacher_rollout_seed_rule"],
            "teacher_validation_noise_seed": int(
                training["teacher_validation_noise_seed"]
            ),
            "analytical_rank": list(screen["ranking"]),
            "freeze_top_k": 1,
        },
        "contexts": {
            "context_samples": 256,
            "future_samples": 256,
            "count_per_run": 8,
            "analytical_and_controller_fit": [0, 1, 2, 3, 4],
            "checkpoint_epoch_selection_only": [5],
            "terminal_veto_only": [6],
            "posthoc_only_never_success": [7],
            "aggregate_context_indices": list(range(8)),
            "fixed_display_context_index": 7,
            "fixed_display_crn_bank": 0,
        },
        "outer_evaluation": {
            "classification": "hash-locked retrospective held-out-seizure reanalysis",
            "open_sealed_ictal_segment_once": True,
            "aggregate_context_indices": list(range(8)),
            "common_random_number_banks": 3,
            "reselection_forbidden": True,
            "aggregate_context_indices_fixed_before_open": list(range(8)),
            "display_context_index_fixed_before_open": 7,
            "display_crn_bank_fixed_before_open": 0,
            "reference_and_ictal_windows_use_independent_canonical_preprocessing_state": True,
            "overleaf_write": False,
        },
        "outer_windows": list(split["outer_preview_windows"]),
    }


def _phase_directory(run_root: Path, phase_id: str) -> Path:
    directory = Path(run_root) / PHASE_OUTPUT_ROOT_DIRECTORY / phase_id
    if not (directory / "COMPLETE.json").is_file():
        raise FileNotFoundError(f"required completed phase is absent: {phase_id}")
    return directory


def _parent_teacher_phase(config: Mapping[str, Any], phase_id: str) -> str | None:
    ids = list(config["part3"]["training"]["teacher_formal_stage_ids"])
    index = ids.index(phase_id)
    return None if index == 0 else ids[index - 1]


def execute_phase(context: Mapping[str, Any]) -> Mapping[str, Any]:
    """Execute one real numerical phase selected by the authenticated runner."""

    config = context["config"]
    phase = str(context["phase"])
    run_root = Path(context["run_root"]).resolve()
    output = Path(context["staging_dir"]).resolve()
    runtime_receipt = _install_runtime_firewall(Path(config["runtime_root"]))
    code_receipts = _verify_code_references(config)
    science = _science_config(config)

    # All scientific imports occur after the runtime/MNE firewall above.
    from h065 import control, data_model, outer, training

    development = run_root / PHASE_OUTPUT_ROOT_DIRECTORY / "DEV_MATERIALIZE"
    development_npz = development / "development_arrays.npz"
    part1 = run_root / PHASE_OUTPUT_ROOT_DIRECTORY / "P1_LORO_GRAPHS"
    part2 = run_root / PHASE_OUTPUT_ROOT_DIRECTORY / "P2_LORO_RC_SDE"
    analytical = run_root / PHASE_OUTPUT_ROOT_DIRECTORY / "ANALYTICAL_TOP1"
    refit = run_root / PHASE_OUTPUT_ROOT_DIRECTORY / "ALLDEV_REFIT"

    if phase == "DEV_MATERIALIZE":
        data_model.assert_signal_run_allowed(phase, "run-01")
        data_model.assert_signal_run_allowed(phase, "run-02")
        data_model.prepare_development(science, output)
        return {
            "phase": phase, "real_compute": "raw_edf_causal_development_materialization",
            "opened_signal_runs": ["run-01", "run-02"], "run03_opened": False,
            "whole_zip_sha256_computed": False, "code_reference_locks": code_receipts,
            "runtime_firewall": runtime_receipt,
            "hard_projection_or_rescaling_applied": False,
        }
    if phase == "P1_LORO_GRAPHS":
        _phase_directory(run_root, "DEV_MATERIALIZE")
        data_model.loro_part1(science, development_npz, output)
        return {
            "phase": phase, "real_compute": "five_band_equal_plv_mst_density_centrality",
            "folds": 2, "clinical_labels_used": False, "run03_opened": False,
            "hard_projection_or_rescaling_applied": False,
        }
    if phase == "P2_LORO_RC_SDE":
        _phase_directory(run_root, "P1_LORO_GRAPHS")
        data_model.loro_part1_part2(
            science, development_npz, output, frozen_part1_dir=part1
        )
        return {
            "phase": phase, "real_compute": "state_dependent_graph_rc_sde_loro_grid",
            "candidate_count": 144, "folds": 2, "run03_opened": False,
            "part1_selection_and_part2_plant_graphs_are_distinct": True,
            "part2_context_samples": 512,
            "part2_direct_validation_windows": 6,
            "rolling_block_candidates": [4, 8, 12, 16, 24, 32],
            "rolling_selection": "largest_jointly_eligible_else_NO_GO_before_part3",
            "hard_projection_or_rescaling_applied": False,
        }
    if phase == "ANALYTICAL_TOP1":
        _phase_directory(run_root, "P2_LORO_RC_SDE")
        control.loro_analytical_top1(science, development_npz, part2, output)
        return {
            "phase": phase, "real_compute": "common_random_number_analytical_control",
            "candidate_count": 120, "freeze_unique_top_k": 1,
            "neural_training_executed": False,
            "fold_fit_reference_isolated_from_leftout_validation": True,
            "run03_opened": False,
            "hard_projection_or_rescaling_applied": False,
        }
    if phase == "ALLDEV_REFIT":
        _phase_directory(run_root, "ANALYTICAL_TOP1")
        control.final_refit(science, development_npz, part2, analytical, output)
        return {
            "phase": phase, "real_compute": "fresh_part1_part2_all_development_refit",
            "development_runs": ["run-01", "run-02"],
            "fresh_initialization": True, "run03_opened": False,
            "hard_projection_or_rescaling_applied": False,
        }
    if phase in training.FORMAL_STAGE_IDS:
        _phase_directory(run_root, "ALLDEV_REFIT")
        parent_id = _parent_teacher_phase(config, phase)
        parent_dir = None if parent_id is None else _phase_directory(run_root, parent_id)
        return training.train_formal_stage(
            science, development_npz, analytical, refit, phase, parent_dir, output
        )
    if phase == "WGAN40_CTX5_SELECT_CTX6_VETO":
        s6 = _phase_directory(run_root, "S6_PRECISION_ALL_PARAMS_870_1050")
        return training.wgan40_and_ctx6_veto(
            science, development_npz, analytical, refit, s6, output
        )
    if phase == "OUTER":
        freeze = _phase_directory(run_root, "WGAN40_CTX5_SELECT_CTX6_VETO")
        report = outer.open_and_evaluate_run03_once(
            science, development_npz, refit, freeze, output
        )
        return {
            "phase": phase,
            "real_compute": "one_time_run03_32s_sealed_segment_8contexts_x_3banks_ofrc",
            "run03_opened": True, "candidate_reselected": False,
            "fallback_or_rescue": False,
            "display_window_id": report["display_window_id"],
            "display_crn_bank": report["display_crn_bank"],
            "aggregate_window_ids": report["aggregate_window_ids"],
            "common_renderer_executed": False,
            "renderer_binding_status": report["binding_status"],
            "ofrc_candidate_sha256": report["candidate_npz_sha256"],
            "hard_projection_or_rescaling_applied": False,
        }
    raise ValueError(f"science backend has no dispatch for phase: {phase}")


__all__ = ["execute_phase"]
