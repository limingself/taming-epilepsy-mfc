from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import sha256_file
from .control import evaluate_detailed, rebuild_selected_stack
from .data_model import (
    array_sha256,
    assert_signal_run_allowed,
    load_development,
    parse_hup065_split,
    reference_paths,
)
from .staging import atomic_json


def _outer_windows(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    windows = tuple(dict(item) for item in config["outer_windows"])
    identities = [str(item["window_id"]) for item in windows]
    expected_ids = [f"outer-context-{index:02d}" for index in range(8)]
    if identities != expected_ids:
        raise PermissionError("outer window identities changed after development freeze")
    expected = (
        (256, 512, 512, 768),
        (1317, 1573, 1573, 1829),
        (2377, 2633, 2633, 2889),
        (3438, 3694, 3694, 3950),
        (4498, 4754, 4754, 5010),
        (5559, 5815, 5815, 6071),
        (6619, 6875, 6875, 7131),
        (7680, 7936, 7936, 8192),
    )
    for item, bounds in zip(windows, expected):
        observed = tuple(
            int(item[key])
            for key in (
                "context_start_sample",
                "context_stop_sample",
                "future_start_sample",
                "future_stop_sample",
            )
        )
        if observed != bounds:
            raise PermissionError(f"frozen outer window changed: {item}")
    return windows


def _component_vectors(summary: Mapping[str, Any], channels: int) -> dict[str, Any]:
    import numpy as np

    names = (
        "gate_time_w1_relative_reduction_pass",
        "gate_occupation_w1_relative_reduction_pass",
        "gate_time_w1_absolute_pass",
        "gate_occupation_w1_absolute_pass",
        "gate_mean_error_pass",
        "gate_sd_ratio_pass",
    )
    source = summary.get("gate_b_component_masks")
    if not isinstance(source, Mapping) or set(source) != set(names):
        raise RuntimeError("outer evaluation lacks the frozen six Gate-B vectors")
    vectors = {
        name: np.asarray(source[name], dtype=bool).reshape(-1)
        for name in names
    }
    if any(value.shape != (channels,) for value in vectors.values()):
        raise RuntimeError("outer Gate-B vector shape changed")
    full = np.logical_and.reduce(list(vectors.values()))
    reported = np.asarray(summary["gate_b_pass_mask"], dtype=bool)
    if not np.array_equal(full, reported):
        raise RuntimeError("outer full Gate-B mask is not the AND of all six components")
    return {**vectors, "full_gate_b_pass": full}


def validate_outer_metric_cartesian(
    rows: list[dict[str, Any]], *, channels: int = 64
) -> dict[str, int]:
    """Reject missing/duplicate context-bank-channel cells before aggregation."""

    window_ids = [f"outer-context-{index:02d}" for index in range(8)]
    expected = {
        (window_id, bank, channel)
        for window_id in window_ids
        for bank in range(3)
        for channel in range(int(channels))
    }
    observed: set[tuple[str, int, int]] = set()
    bool_fields = (
        "gate_time_w1_relative_reduction_pass",
        "gate_occupation_w1_relative_reduction_pass",
        "gate_time_w1_absolute_pass",
        "gate_occupation_w1_absolute_pass",
        "gate_mean_error_pass",
        "gate_sd_ratio_pass",
        "gate_b_pass",
        "gate_c_trajectory_pass",
    )
    for row in rows:
        key = (
            str(row.get("window_id")),
            int(row.get("crn_bank", -1)),
            int(row.get("channel_index", -1)),
        )
        if key in observed:
            raise RuntimeError(f"duplicate outer metric cell: {key}")
        observed.add(key)
        if any(type(row.get(field)) is not bool for field in bool_fields):
            raise RuntimeError(f"outer metric cell has non-boolean Gate field: {key}")
    missing = expected - observed
    extra = observed - expected
    if missing or extra or len(rows) != len(expected):
        raise RuntimeError(
            "outer metric Cartesian product incomplete: "
            f"expected={len(expected)}, observed={len(rows)}, "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    return {
        "contexts": 8,
        "crn_banks": 3,
        "channels": int(channels),
        "cells": len(expected),
    }


def open_and_evaluate_run03_once(
    config: Mapping[str, Any],
    development_npz: Path,
    refit_dir: Path,
    training_dir: Path,
    output: Path,
) -> dict[str, Any]:
    """Open run03 only after parent GO, then publish metrics and renderer inputs.

    The parent runner consumes the one-shot authorization before entering this
    function.  This function never ranks, rescues, recalibrates, or plots.
    """

    import numpy as np
    import pandas as pd

    assert_signal_run_allowed("OUTER", "run-03")
    windows = _outer_windows(config)
    aggregate_ids = [str(item["window_id"]) for item in windows]
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output / "ACCESS_STARTED.json",
        {
            "schema_version": "hup065-run03-access-started-v2",
            "run": "run-03",
            "parent_outer_go_already_consumed": True,
            "sealed_ictal_32s_segment_opened_once_in_outer": True,
            "outer_reference_40s_segment_opened_in_same_authorized_phase": True,
            "aggregate_window_ids": aggregate_ids,
            "display_window_id": "outer-context-07",
            "display_crn_bank": 0,
            "no_retry_or_reselection": True,
        },
    )

    split = parse_hup065_split(config)
    from .data_model import activate_canonical

    data_module, _network_module, _part2_module, _part2_pipeline = activate_canonical(config)
    data_cfg = config["source_data"]
    prep = config["preprocessing"]
    dataset = data_module.BIDSZipDataset(data_cfg["zip_root"], subjects=["HUP065"])
    records = dataset.selected_records("HUP065", [("ictal", "run-03")])
    if (
        len(records) != 1
        or records[0].run != "run-03"
        or records[0].task.casefold() != "ictal"
    ):
        raise PermissionError("sealed EDF identity changed")
    development = load_development(development_npz)
    channels = tuple(development["channels"].astype(str).tolist())
    observed_channels = tuple(dataset.common_good_channels("HUP065", records=records))
    if observed_channels != channels or len(channels) != 64:
        raise PermissionError("run03 channel order differs from development freeze")
    runtime = Path(str(config["runtime_root"]))
    ictal_loader = data_module.EEGSegmentLoader(
        dataset,
        cache_dir=runtime / "run03_ictal_fresh_cache_never_reused",
        temp_dir=runtime / "run03_ictal_edf_tmp",
        target_sfreq=float(prep["target_sampling_rate_hz"]),
        bandpass_hz=tuple(prep["bandpass_hz"]),
        filter_order=int(prep["filter_order"]),
        burn_in_s=float(prep["filter_burn_in_s"]),
        reference=str(prep["reference"]),
        resampling_mode=str(prep["resampling_mode"]),
        causal_fir_half_length_factor=int(prep["causal_fir_half_length_factor"]),
    )
    reference_loader = data_module.EEGSegmentLoader(
        dataset,
        cache_dir=runtime / "run03_reference_fresh_cache_never_reused",
        temp_dir=runtime / "run03_reference_edf_tmp",
        target_sfreq=float(prep["target_sampling_rate_hz"]),
        bandpass_hz=tuple(prep["bandpass_hz"]),
        filter_order=int(prep["filter_order"]),
        burn_in_s=float(prep["filter_burn_in_s"]),
        reference=str(prep["reference"]),
        resampling_mode=str(prep["resampling_mode"]),
        causal_fir_half_length_factor=int(prep["causal_fir_half_length_factor"]),
    )
    record = records[0]
    # These are the first permitted requests for the run03 EDF payload.  The
    # parent OUTER_OPENED ledger has already made this phase non-retryable.
    ictal_segment = ictal_loader.load(
        record,
        start_s=float(split["ictal_test_window"]["start_s"]),
        duration_s=32.0,
        channel_names=channels,
        use_cache=False,
    )
    reference_segment = reference_loader.load(
        record,
        start_s=float(split["reference_test_window"]["start_s"]),
        duration_s=40.0,
        channel_names=channels,
        use_cache=False,
    )
    ictal = np.asarray(ictal_segment.data.T, dtype=np.float64)
    reference_raw = np.asarray(reference_segment.data.T, dtype=np.float64)
    if ictal.shape != (8192, 64) or reference_raw.shape != (10240, 64):
        raise ValueError("sealed ictal/reference processed shape changed")
    if not np.isfinite(ictal).all() or not np.isfinite(reference_raw).all():
        raise ValueError("sealed ictal/reference segments contain non-finite values")
    for segment in (ictal_segment, reference_segment):
        provenance = dict(segment.provenance)
        if (
            not provenance.get("preprocessing_fully_causal")
            or not provenance.get("resampling_causal")
        ):
            raise PermissionError("sealed preprocessing is not fully causal")
    np.savez_compressed(
        output / "run03_sealed_segments.npz",
        ictal=ictal,
        preictal_reference=reference_raw,
        channels=np.asarray(channels),
        sfreq=np.asarray(256.0),
    )
    atomic_json(
        output / "run03_payload_access_ledger.json",
        {
            "schema_version": "hup065-run03-payload-access-v1",
            "first_payload_access_phase": "OUTER",
            "run": "run-03",
            "edf_member": record.edf_member,
            "edf_crc": int(record.edf_crc),
            "edf_size_bytes": int(record.edf_size_bytes),
            "requested_windows_seconds_half_open": [
                [120.0, 152.0],
                [75.0, 115.0],
            ],
            "raw_member_sha256_not_computed": True,
            "sealed_ictal_segment_opened_once_under_consumed_go": True,
            "outer_reference_opened_in_same_authorized_phase": True,
            "reference_and_ictal_preprocessing_state": "independent_loader_instances",
        },
    )

    ctrl, model, network, cfg, stack, checkpoint = rebuild_selected_stack(
        config, development, refit_dir, training_dir
    )
    contexts = []
    futures = []
    ledger = []
    for item in windows:
        contexts.append(
            ictal[
                int(item["context_start_sample"]) : int(item["context_stop_sample"])
            ].copy()
        )
        futures.append(
            ictal[
                int(item["future_start_sample"]) : int(item["future_stop_sample"])
            ].copy()
        )
        ledger.append(
            {
                "run": "run-03",
                "window_id": str(item["window_id"]),
                "context_index": int(str(item["window_id"]).rsplit("-", 1)[1]),
                "boundary_sample": int(item["context_stop_sample"]),
                "predeclared_before_open": True,
            }
        )
    observed = np.stack(
        [model.transform.scaler.transform(item) for item in futures]
    )
    initials = [
        stack[2].adapter.initial_state_from_context(item) for item in contexts
    ]
    reference_transformed = model.transform.scaler.transform(reference_raw)
    reference = reference_paths(reference_transformed)
    summary, rows, display = evaluate_detailed(
        ctrl,
        stack[-1],
        stack[2],
        initials,
        reference,
        network["target_mask"],
        cfg,
        fold_id="sealed-run03-retrospective",
        stage="outer_once_8contexts_x_3banks",
        ledger=ledger,
        observed_standardized=observed,
        display_local_index=7,
    )
    if display.get("ledger", {}).get("window_id") != "outer-context-07":
        raise RuntimeError("representative OFRC display window is not frozen context07")
    if int(display.get("ledger", {}).get("crn_bank", -1)) != 0:
        raise RuntimeError("representative OFRC display bank is not frozen bank0")
    cartesian = validate_outer_metric_cartesian(rows, channels=len(channels))
    metrics_path = output / "run03_all64_channel_context_bank_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    np.savez_compressed(
        output / "run03_frozen_display_context07_bank0_rollout.npz",
        free_standardized=display["free"],
        controlled_standardized=display["controlled"],
        reference_standardized=display["reference"],
        controls=display["controls"],
        standard_normal=display["noise"],
        observed_standardized=observed[7],
        channels=np.asarray(channels),
        direct_mask=np.asarray(network["target_mask"], dtype=bool),
        display_window_id=np.asarray("outer-context-07"),
        aggregate_window_ids=np.asarray(
            aggregate_ids
        ),
        display_crn_bank=np.asarray(0),
    )

    components = _component_vectors(summary, len(channels))
    direct_mask = np.asarray(network["target_mask"], dtype=bool)
    safety = np.full(64, bool(summary["gate_c_pass"]), dtype=bool)
    full_gate = components["full_gate_b_pass"] & safety
    gate_source = output / "outer_gate_vectors.json"
    atomic_json(
        gate_source,
        {
            "schema_version": "hup065-outer-gate-vectors-v1",
            "aggregation": "fail_closed_AND_over_eight_declared_contexts_and_three_crn_banks",
            "aggregate_window_ids": aggregate_ids,
            "crn_banks_per_window": 3,
            "context_bank_channel_cartesian": cartesian,
            **{
                name: value.astype(bool).tolist()
                for name, value in components.items()
            },
        },
    )
    safety_source = output / "outer_safety_vectors.json"
    atomic_json(
        safety_source,
        {
            "schema_version": "hup065-outer-safety-vectors-v1",
            "finite_and_energy_peak_saturation_all_window_banks": bool(
                summary["gate_c_pass"]
            ),
            "safety_gate_pass": safety.tolist(),
            "energy_is_post_rollout_gate_only": True,
            "hard_projection_or_rescaling_applied": False,
        },
    )
    frozen_manifest = training_dir / "final_development_freeze.json"
    frozen_manifest_sha = sha256_file(frozen_manifest)
    gate_source_sha = sha256_file(gate_source)
    safety_source_sha = sha256_file(safety_source)
    candidate_id = str(checkpoint["analytical_top1"]["candidate_id"])
    candidate_path = output / "hup065_run03_context07_ofrc_candidate.npz"
    np.savez_compressed(
        candidate_path,
        schema_version=np.asarray("ofrc-final-candidate-v1"),
        shared_protocol_sha256=np.asarray(config["shared_protocol_sha256"]),
        frozen_evaluator_manifest_sha256=np.asarray(frozen_manifest_sha),
        outer_evaluation_role=np.asarray("outer_evaluation"),
        gate_vector_source_sha256=np.asarray(gate_source_sha),
        safety_vector_source_sha256=np.asarray(safety_source_sha),
        subject_id=np.asarray("HUP065"),
        candidate_id=np.asarray(candidate_id),
        channels=np.asarray(channels),
        observed_scaled=observed[7],
        free_scaled=display["free"],
        reference_scaled=display["reference"],
        controlled_scaled=display["controlled"],
        direct_mask=direct_mask,
        **{
            name: value
            for name, value in components.items()
            if name != "full_gate_b_pass"
        },
        full_gate_b_pass=components["full_gate_b_pass"],
        safety_gate_pass=safety,
        full_gate_pass=full_gate,
        evaluation_context=np.asarray("outer-context-07"),
        display_window_id=np.asarray("outer-context-07"),
        display_crn_bank=np.asarray(0),
        aggregate_window_ids=np.asarray(aggregate_ids),
        aggregate_context_indices=np.asarray(list(range(8)), dtype=np.int64),
        aggregate_crn_banks=np.asarray([0, 1, 2], dtype=np.int64),
        context_bank_evaluations=np.asarray(24),
        data_role=np.asarray("representative_display_from_frozen_outer_aggregate"),
        model_contract_sha256=np.asarray(frozen_manifest_sha),
    )
    candidate_sha = sha256_file(candidate_path)
    binding_template = {
        "binding_schema_version": "ofrc-final-binding-v1",
        "status": "NO_GO_PENDING_INDEPENDENT_AUDIT",
        "required_status_for_renderer": "GO",
        "subject_id": "HUP065",
        "candidate_id": candidate_id,
        "candidate_npz_sha256": candidate_sha,
        "channel_context_bank_metrics_sha256": sha256_file(metrics_path),
        "shared_protocol_sha256": config["shared_protocol_sha256"],
        "frozen_evaluator_manifest_sha256": frozen_manifest_sha,
        "outer_evaluation_role": "outer_evaluation",
        "gate_vector_source_sha256": gate_source_sha,
        "safety_vector_source_sha256": safety_source_sha,
        "display_window_id": "outer-context-07",
        "display_crn_bank": 0,
        "aggregate_window_ids": aggregate_ids,
        "renderer_was_not_executed_by_outer": True,
    }
    atomic_json(output / "COMMON_RENDERER_BINDING_PENDING_AUDIT.json", binding_template)
    report = {
        "schema_version": "hup065-run03-retrospective-outer-v2",
        "phase": "OUTER",
        "classification": config["outer_evaluation"]["classification"],
        "run": "run-03",
        "metrics": summary,
        "ictal_array_sha256": array_sha256(ictal),
        "reference_array_sha256": array_sha256(reference_raw),
        "direct_actuator_count": int(direct_mask.sum()),
        "all_channel_count": 64,
        "checkpoint_sha256": sha256_file(training_dir / "selected_controller.pt"),
        "frozen_evaluator_manifest_sha256": frozen_manifest_sha,
        "candidate_npz": candidate_path.name,
        "candidate_npz_sha256": candidate_sha,
        "display_window_id": "outer-context-07",
        "display_crn_bank": 0,
        "aggregate_window_ids": aggregate_ids,
        "aggregate_crn_banks_per_window": 3,
        "context_bank_evaluations": 24,
        "context_bank_channel_cartesian": cartesian,
        "sealed_ictal_32s_segment_opened_once_under_consumed_go": True,
        "outer_reference_40s_segment_opened_in_same_authorized_phase": True,
        "reference_and_ictal_preprocessing_state": "independent_loader_instances",
        "candidate_mask_model_checkpoint_reselected": False,
        "outer_result_used_for_success_or_rescue": False,
        "posthoc_context7_used_for_selection": False,
        "common_renderer_executed": False,
        "binding_status": "NO_GO_PENDING_INDEPENDENT_AUDIT",
        "hard_projection_or_rescaling_applied": False,
        "overleaf_write": False,
    }
    atomic_json(output / "run03_retrospective_report.json", report)
    return report


__all__ = ["open_and_evaluate_run03_once"]
