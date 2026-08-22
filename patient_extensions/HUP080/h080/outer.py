from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import sha256_file
from .control import (
    _standardized_reference,
    part3_context_block,
    evaluate_detailed,
    rebuild_selected_stack,
)
from .data_model import (
    activate_canonical,
    array_sha256,
    load_development,
    parse_hup080_split,
    reference_paths,
)
from .staging import atomic_json


def freeze_outer_contract(
    config: Mapping[str, Any], analytical_dir: Path, refit_dir: Path,
    training_dir: Path, ctx6_dir: Path, output: Path,
) -> None:
    terminal = json.loads((ctx6_dir / "terminal_veto.json").read_text(encoding="utf-8"))
    if not bool(terminal["terminal_veto_pass"]):
        raise PermissionError("ctx6 terminal veto failed; run-04 must remain sealed")
    top1_path = analytical_dir / "frozen_top1.json"
    refit_path = refit_dir / "refit_receipt.json"
    checkpoint = training_dir / "selected_controller.pt"
    terminal_path = ctx6_dir / "terminal_veto.json"
    top1 = json.loads(top1_path.read_text(encoding="utf-8"))["selection"]
    outer = config["outer_evaluation"]
    freeze = {
        "schema_version": "hup080-run04-sealed-segment-freeze-v2",
        "subject": "HUP080",
        "classification": config["outer_evaluation"]["classification"],
        "sealed_run": "run-04",
        "ictal_window_seconds_half_open": [120.0, 152.0],
        "reference_window_seconds_half_open": [75.0, 115.0],
        "candidate_id": str(top1["candidate_id"]),
        "display_window_id": str(outer["display_window_id"]),
        "display_context_index": int(outer["display_context_fixed_before_open"]),
        "display_crn_bank": int(outer["display_crn_bank"]),
        "aggregate_window_ids": list(outer["aggregate_window_ids"]),
        "aggregate_context_indices": list(outer["aggregate_context_indices"]),
        "aggregate_rule": str(outer["aggregate_rule"]),
        "validation_crn_banks": int(config["controller"]["validation_crn_banks"]),
        "analytical_top1_sha256": sha256_file(top1_path),
        "refit_receipt_sha256": sha256_file(refit_path),
        "selected_controller_sha256": sha256_file(checkpoint),
        "ctx6_terminal_veto_sha256": sha256_file(terminal_path),
        "ctx6_terminal_veto_pass": True,
        "mask_model_checkpoint_hyperparameters_frozen": True,
        "sealed_ictal_and_reference_segments_open_together_once_after_go": True,
        "all_eight_contexts_evaluated_from_single_ictal_segment": True,
        "outer_may_reselect_or_rescue": False,
        "run04_opened": False,
        "overleaf_write": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "outer_freeze.json", freeze)
    atomic_json(
        output / "RUN04_GO_TEMPLATE_DO_NOT_RENAME.json",
        {
            "decision": "REPLACE_WITH_GO",
            "outer_freeze_sha256": sha256_file(output / "outer_freeze.json"),
            "config_sha256": sha256_file(Path(str(config["project_root"])) / "config.json"),
            "acknowledge_retrospective_one_time_sealed_segment_access": False,
            "acknowledge_no_reselection_rescue_or_success_claim": False,
        },
    )


def verify_run04_go(config: Mapping[str, Any], freeze_dir: Path, go_file: Path) -> dict[str, Any]:
    path = Path(go_file).resolve()
    if not path.is_file():
        raise PermissionError("run-04 GO file is absent")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "decision": "GO",
        "outer_freeze_sha256": sha256_file(freeze_dir / "outer_freeze.json"),
        "config_sha256": sha256_file(Path(str(config["project_root"])) / "config.json"),
        "acknowledge_retrospective_one_time_sealed_segment_access": True,
        "acknowledge_no_reselection_rescue_or_success_claim": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise PermissionError(f"run-04 GO does not bind {key}")
    return payload


def validate_outer_metric_cartesian(
    metric_frame: Any, aggregate_ids: list[str], aggregate_indices: list[int],
    bank_count: int, *, channel_count: int,
) -> dict[str, Any]:
    """Require one and only one row for every channel x outer-window x CRN-bank."""

    expected_pairs = {
        (window_id, bank)
        for window_id in aggregate_ids for bank in range(int(bank_count))
    }
    if len(metric_frame) != int(channel_count) * len(expected_pairs):
        raise RuntimeError(
            "outer metric ledger is not channels x all windows x all banks"
        )
    observed_channels = set(metric_frame["channel_index"].astype(int).tolist())
    if observed_channels != set(range(int(channel_count))):
        raise RuntimeError("outer metric ledger channel identities changed")
    for channel_index, group in metric_frame.groupby("channel_index", sort=True):
        observed_pairs = list(
            zip(group["window_id"].astype(str), group["crn_bank"].astype(int))
        )
        if len(observed_pairs) != len(expected_pairs) or set(observed_pairs) != expected_pairs:
            raise RuntimeError(
                f"channel {int(channel_index)} lacks a unique window x bank ledger"
            )
    window_context_rows = metric_frame[["window_id", "context_index"]].drop_duplicates()
    if len(window_context_rows) != len(aggregate_ids):
        raise RuntimeError("outer window IDs do not map one-to-one to context indices")
    observed_window_context = {
        str(row.window_id): int(row.context_index)
        for row in window_context_rows.itertuples(index=False)
    }
    if observed_window_context != dict(zip(aggregate_ids, aggregate_indices)):
        raise RuntimeError("outer window-id/context-index mapping changed")
    return {
        "channel_count": int(channel_count),
        "window_count": len(aggregate_ids), "crn_bank_count": int(bank_count),
        "unique_window_bank_pairs_per_channel": len(expected_pairs),
        "total_rows": len(metric_frame), "complete_unique_cartesian_product": True,
    }


def open_and_evaluate_run04_once(
    config: Mapping[str, Any], development_npz: Path, refit_dir: Path,
    training_dir: Path, freeze_dir: Path, go_file: Path, output: Path,
) -> None:
    """Open the sealed segments once, evaluate all eight contexts, and never reselect."""

    import numpy as np
    import pandas as pd

    go = verify_run04_go(config, freeze_dir, go_file)
    freeze = json.loads((freeze_dir / "outer_freeze.json").read_text(encoding="utf-8"))
    if sha256_file(training_dir / "selected_controller.pt") != str(
        freeze["selected_controller_sha256"]
    ):
        raise PermissionError("selected controller changed after outer freeze")
    if sha256_file(refit_dir / "refit_receipt.json") != str(
        freeze["refit_receipt_sha256"]
    ):
        raise PermissionError("refit receipt changed after outer freeze")
    output.mkdir(parents=True, exist_ok=True)
    # This durable staging marker is written before the first EDF byte is requested.
    # A crash leaves the parent phase staging directory in place, blocking blind retry.
    atomic_json(
        output / "ACCESS_STARTED.json",
        {
            "schema_version": "hup080-run04-access-started-v1",
            "run": "run-04", "sealed_segment_access": True,
            "all_eight_outer_contexts": True,
            "go_sha256": sha256_file(Path(go_file)),
            "outer_freeze_sha256": go["outer_freeze_sha256"],
            "reselection_forbidden": True,
        },
    )
    split = parse_hup080_split(config)
    data_module, _network_module, _part2, _part2_pipeline = activate_canonical(config)
    data_cfg = config["source_data"]
    prep = config["preprocessing"]
    dataset = data_module.BIDSZipDataset(data_cfg["zip_root"], subjects=["HUP080"])
    records = dataset.selected_records("HUP080", [("ictal", "run-04")])
    if len(records) != 1 or records[0].run != "run-04" or records[0].task.casefold() != "ictal":
        raise PermissionError("sealed EDF identity changed")
    development = load_development(development_npz)
    channels = tuple(development["channels"].astype(str).tolist())
    observed_channels = tuple(dataset.common_good_channels("HUP080", records=records))
    if observed_channels != channels or len(channels) != 96:
        raise PermissionError("run-04 channel order differs from frozen development order")
    runtime = Path(str(config["runtime_root"]))
    loader = data_module.EEGSegmentLoader(
        dataset, cache_dir=runtime / "run04_fresh_cache_never_reused",
        temp_dir=runtime / "run04_edf_tmp",
        target_sfreq=float(prep["target_sampling_rate_hz"]),
        bandpass_hz=tuple(prep["bandpass_hz"]), filter_order=int(prep["filter_order"]),
        burn_in_s=float(prep["filter_burn_in_s"]), reference=str(prep["reference"]),
        resampling_mode=str(prep["resampling_mode"]),
        causal_fir_half_length_factor=int(prep["causal_fir_half_length_factor"]),
    )
    record = records[0]
    sealed_member = config["source_data"]["sealed_run04_edf_member_contract"]
    if (
        record.edf_member != sealed_member["filename"]
        or int(record.edf_crc) != int(sealed_member["crc32"])
        or int(record.edf_size_bytes) != int(sealed_member["uncompressed_bytes"])
        or int(record.edf_compressed_bytes) != int(sealed_member["compressed_bytes"])
    ):
        raise PermissionError("sealed run-04 EDF member contract changed")
    # The two frozen windows use independent canonical preprocessing calls.
    # Each therefore receives its own 20-s causal-filter burn-in, matching the
    # development/HUP060 semantics; both calls remain inside this one durable
    # OUTER authorization and cannot be used to reselect or retry.
    ictal_segment = loader.load(
        record, start_s=120.0, duration_s=32.0,
        channel_names=channels, use_cache=False,
    )
    reference_segment = loader.load(
        record, start_s=75.0, duration_s=40.0,
        channel_names=channels, use_cache=False,
    )
    ictal = np.asarray(ictal_segment.data.T, dtype=np.float64)
    reference_raw = np.asarray(reference_segment.data.T, dtype=np.float64)
    if ictal.shape != (8192, 96) or reference_raw.shape != (10240, 96):
        raise ValueError("sealed-segment processed shape changed")
    if not np.isfinite(ictal).all() or not np.isfinite(reference_raw).all():
        raise ValueError("sealed segment contains non-finite values")
    for segment in (ictal_segment, reference_segment):
        provenance = dict(segment.provenance)
        if not provenance.get("preprocessing_fully_causal") or not provenance.get("resampling_causal"):
            raise PermissionError("sealed preprocessing is not fully causal")
    np.savez_compressed(
        output / "run04_sealed_segments.npz", ictal=ictal,
        preictal_reference=reference_raw, channels=np.asarray(channels),
        sfreq=np.asarray(256.0),
    )
    ctrl, model, network, cfg, stack, checkpoint = rebuild_selected_stack(
        config, development, refit_dir, training_dir
    )
    outer_contract = config["outer_evaluation"]
    aggregate_indices = [int(value) for value in outer_contract["aggregate_context_indices"]]
    aggregate_ids = [str(value) for value in outer_contract["aggregate_window_ids"]]
    display_window_id = str(outer_contract["display_window_id"])
    display_crn_bank = int(outer_contract["display_crn_bank"])
    display_local_index = aggregate_ids.index(display_window_id)
    contexts = []
    futures = []
    ledger = []
    for window_id, context_index in zip(aggregate_ids, aggregate_indices):
        context, future, boundary = part3_context_block(ictal, context_index)
        contexts.append(context)
        futures.append(future)
        ledger.append({
            "run": "run-04", "window_id": window_id,
            "context_index": context_index, "boundary_sample": int(boundary),
            "predeclared_before_open": True,
        })
    initial = [
        stack[2].adapter.initial_state_from_context(context) for context in contexts
    ]
    reference_transformed = model.transform.scaler.transform(reference_raw)
    reference = reference_paths(reference_transformed)
    observed = np.stack(
        [model.transform.scaler.transform(future) for future in futures]
    )
    summary, rows, trajectory_rows, display = evaluate_detailed(
        ctrl, stack[-1], stack[2], initial, reference, network["target_mask"], cfg,
        fold_id="sealed-run04-retrospective", stage="outer_once",
        ledger=ledger, observed_standardized=observed,
        display_local_index=display_local_index,
        display_crn_bank=display_crn_bank,
    )
    if (
        int(summary["declared_contexts"]) != 8
        or int(summary["crn_banks"]) != 3
        or int(summary["context_bank_evaluations"]) != 24
    ):
        raise RuntimeError("outer evaluation did not complete the frozen 8x3 ledger")
    if (
        display.get("ledger", {}).get("window_id") != display_window_id
        or int(display.get("ledger", {}).get("crn_bank", -1)) != display_crn_bank
    ):
        raise AssertionError("representative outer display window was not respected")
    metric_frame = pd.DataFrame(rows)
    bank_count = int(config["controller"]["validation_crn_banks"])
    cartesian_receipt = validate_outer_metric_cartesian(
        metric_frame, aggregate_ids, aggregate_indices, bank_count,
        channel_count=96,
    )
    metric_frame.to_csv(
        output / "run04_all96_channel_metrics.csv", index=False, encoding="utf-8-sig"
    )
    trajectory_safety_path = output / "run04_trajectory_safety_metrics.csv"
    pd.DataFrame(trajectory_rows).to_csv(
        trajectory_safety_path,
        index=False,
        encoding="utf-8-sig",
    )
    component_columns = [
        "time_relative_reduction_pass", "occupation_relative_reduction_pass",
        "time_absolute_pass", "occupation_absolute_pass",
        "mean_absolute_error_pass", "symmetric_sd_ratio_pass",
    ]
    gate_vectors = (
        metric_frame.groupby("channel_index", sort=True)[component_columns]
        .all().reset_index()
    )
    gate_vectors["full_gate_b_pass"] = gate_vectors[component_columns].all(axis=1)
    gate_vectors["safety_gate_pass"] = bool(summary["gate_c_pass"])
    gate_vectors["full_gate_pass"] = (
        gate_vectors["full_gate_b_pass"] & gate_vectors["safety_gate_pass"]
    )
    gate_vectors.insert(1, "channel", np.asarray(channels)[gate_vectors["channel_index"]])
    gate_vectors.insert(
        2, "direct_actuated",
        np.asarray(network["target_mask"], dtype=bool)[gate_vectors["channel_index"]],
    )
    gate_vector_path = output / "run04_all96_aggregate_gate_vectors.csv"
    gate_vectors.to_csv(gate_vector_path, index=False, encoding="utf-8-sig")
    outer_pass_components = {
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
            int(summary["gate_b_pass_count"]) >= 1
        ),
    }
    safety_path = output / "run04_outer_safety.json"
    atomic_json(
        safety_path,
        {
            "schema_version": "hup080-run04-outer-safety-v1",
            "display_window_id": display_window_id,
            "display_crn_bank": display_crn_bank,
            "aggregate_window_ids": aggregate_ids,
            "crn_banks": int(summary["crn_banks"]),
            "context_bank_evaluations": int(summary["context_bank_evaluations"]),
            "cartesian_product_receipt": cartesian_receipt,
            "all_predeclared_windows_x_banks_gate_c_pass": bool(summary["gate_c_pass"]),
            "worst_case_total_energy": float(summary["worst_case_total_energy"]),
            "worst_case_maximum_per_actuator_rms": float(
                summary["worst_case_maximum_per_actuator_rms"]
            ),
            "worst_case_control_peak": float(summary["worst_case_control_peak"]),
            "worst_case_saturation_fraction": float(
                summary["worst_case_saturation_fraction"]
            ),
            "runtime_control_projection_or_rescaling": False,
            "trajectory_safety_sha256": sha256_file(trajectory_safety_path),
        },
    )
    np.savez_compressed(
        output / "run04_frozen_rollout.npz",
        free_standardized=display["free"],
        controlled_standardized=display["controlled"],
        reference_standardized=display["reference"], controls=display["controls"],
        standard_normal=display["noise"],
        observed_standardized=observed[display_local_index],
        channels=np.asarray(channels), direct_mask=np.asarray(network["target_mask"], dtype=bool),
        display_window_id=np.asarray(display_window_id),
        display_crn_bank=np.asarray(display_crn_bank),
        aggregate_window_ids=np.asarray(aggregate_ids),
    )
    atomic_json(
        output / "run04_retrospective_report.json",
        {
            "schema_version": "hup080-run04-retrospective-outer-v1",
            "classification": config["outer_evaluation"]["classification"],
            "run": "run-04", "candidate_id": str(freeze["candidate_id"]),
            "display_window_id": display_window_id,
            "display_crn_bank": display_crn_bank,
            "aggregate_window_ids": aggregate_ids,
            "aggregate_context_indices": aggregate_indices,
            "outer_gate_and_safety_fail_closed_over_all_windows_x_banks": True,
            "cartesian_product_receipt": cartesian_receipt,
            "outer_fail_closed_components": outer_pass_components,
            "outer_fail_closed_pass": bool(all(outer_pass_components.values())),
            "metrics": summary,
            "ictal_array_sha256": array_sha256(ictal),
            "reference_array_sha256": array_sha256(reference_raw),
            "whole_archive_sha256_runtime_gate": False,
            "source_archive_bytes": int(config["source_data"]["raw_zip_bytes"]),
            "source_central_directory_metadata_sha256": config["source_data"]["raw_zip_central_directory_metadata_sha256"],
            "direct_actuator_count": int(np.asarray(network["target_mask"]).sum()),
            "all_channel_count": 96,
            "checkpoint_sha256": sha256_file(training_dir / "selected_controller.pt"),
            "outer_freeze_sha256": go["outer_freeze_sha256"],
            "aggregate_gate_vectors_sha256": sha256_file(gate_vector_path),
            "outer_safety_sha256": sha256_file(safety_path),
            "trajectory_safety_sha256": sha256_file(trajectory_safety_path),
            "sealed_ictal_and_reference_segments_opened_together_once": True,
            "independent_canonical_preprocessing_windows": {
                "ictal": {"start_s": 120.0, "duration_s": 32.0, "burn_in_start_s": 100.0},
                "reference": {"start_s": 75.0, "duration_s": 40.0, "burn_in_start_s": 55.0}
            },
            "all_eight_contexts_evaluated_from_single_ictal_segment": True,
            "candidate_mask_model_checkpoint_reselected": False,
            "outer_result_used_for_success_or_rescue": False,
            "posthoc_context7_used_for_success": False,
            "overleaf_write": False,
        },
    )


__all__ = [
    "freeze_outer_contract", "open_and_evaluate_run04_once",
    "validate_outer_metric_cartesian", "verify_run04_go",
]
