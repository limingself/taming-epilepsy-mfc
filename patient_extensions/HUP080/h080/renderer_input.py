from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import sha256_file
from .staging import atomic_json


def _strict_bool_vector(values: Any, *, name: str) -> Any:
    import numpy as np

    result = []
    for value in values:
        token = str(value).strip().casefold()
        if token in {"true", "1"}:
            result.append(True)
        elif token in {"false", "0"}:
            result.append(False)
        else:
            raise ValueError(f"{name} contains a non-boolean value: {value!r}")
    return np.asarray(result, dtype=bool)


def freeze_common_renderer_input(
    config: Mapping[str, Any], outer_dir: Path, output: Path,
) -> None:
    """Freeze a renderer-v1 candidate; leave its audit GO binding unavailable."""

    import numpy as np
    import pandas as pd

    report_path = outer_dir / "run04_retrospective_report.json"
    rollout_path = outer_dir / "run04_frozen_rollout.npz"
    gate_path = outer_dir / "run04_all96_aggregate_gate_vectors.csv"
    safety_path = outer_dir / "run04_outer_safety.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    safety = json.loads(safety_path.read_text(encoding="utf-8"))
    if bool(report.get("outer_result_used_for_success_or_rescue")):
        raise PermissionError("outer result was improperly routed into selection")
    if not bool(report.get("outer_gate_and_safety_fail_closed_over_all_windows_x_banks")):
        raise PermissionError("outer aggregate window/bank evaluation is incomplete")
    outer_contract = config["outer_evaluation"]
    display_window_id = str(outer_contract["display_window_id"])
    display_crn_bank = int(outer_contract["display_crn_bank"])
    aggregate_window_ids = [str(value) for value in outer_contract["aggregate_window_ids"]]
    if report["display_window_id"] != display_window_id:
        raise PermissionError("outer report display window differs from preregistration")
    if int(report["display_crn_bank"]) != display_crn_bank:
        raise PermissionError("outer report display CRN bank differs from preregistration")
    if list(report["aggregate_window_ids"]) != aggregate_window_ids:
        raise PermissionError("outer report aggregate-window ledger differs from preregistration")
    if list(report["aggregate_context_indices"]) != list(range(8)):
        raise PermissionError("outer report context-index ledger differs from preregistration")
    if int(report["metrics"]["declared_contexts"]) != 8:
        raise PermissionError("outer report does not aggregate all eight windows")
    if int(report["metrics"]["crn_banks"]) != 3:
        raise PermissionError("outer report does not aggregate all three CRN banks")
    if int(report["metrics"]["context_bank_evaluations"]) != 24:
        raise PermissionError("outer report does not contain the full 8x3 ledger")
    if safety["display_window_id"] != display_window_id:
        raise PermissionError("safety receipt display window differs from preregistration")
    if int(safety["display_crn_bank"]) != display_crn_bank:
        raise PermissionError("safety receipt display CRN bank differs from preregistration")
    if list(safety["aggregate_window_ids"]) != aggregate_window_ids:
        raise PermissionError("safety receipt aggregate-window ledger differs from preregistration")
    if int(safety["crn_banks"]) != 3 or int(safety["context_bank_evaluations"]) != 24:
        raise PermissionError("safety receipt does not cover the full 8x3 ledger")

    with np.load(rollout_path, allow_pickle=False) as archive:
        rollout = {name: np.asarray(archive[name]) for name in archive.files}
    required_rollout = {
        "free_standardized", "controlled_standardized", "reference_standardized",
        "observed_standardized", "channels", "direct_mask", "display_window_id",
        "display_crn_bank", "aggregate_window_ids",
    }
    if not required_rollout.issubset(rollout):
        raise KeyError(
            "outer rollout lacks common-renderer fields: "
            + repr(sorted(required_rollout - set(rollout)))
        )
    if str(rollout["display_window_id"].item()) != display_window_id:
        raise PermissionError("rollout display window differs from preregistration")
    if int(rollout["display_crn_bank"].item()) != display_crn_bank:
        raise PermissionError("rollout display CRN bank differs from preregistration")
    if rollout["aggregate_window_ids"].astype(str).tolist() != aggregate_window_ids:
        raise PermissionError("rollout aggregate-window ledger differs from preregistration")
    channels = np.asarray(rollout["channels"]).astype(str).reshape(-1)
    direct_mask = np.asarray(rollout["direct_mask"], dtype=bool).reshape(-1)
    if channels.shape != (96,) or direct_mask.shape != (96,):
        raise ValueError("common-renderer input must describe exactly 96 channels")
    if len(set(channels.tolist())) != 96:
        raise ValueError("common-renderer channels must be unique")
    if float(direct_mask.mean()) > 0.80 + 1.0e-12:
        raise PermissionError("renderer candidate exceeds the frozen 80% direct-actuation limit")

    gate = pd.read_csv(gate_path)
    if gate.shape[0] != 96 or gate["channel_index"].astype(int).tolist() != list(range(96)):
        raise ValueError("aggregate Gate vector must contain channel indices 0..95")
    if gate["channel"].astype(str).tolist() != channels.tolist():
        raise ValueError("aggregate Gate vector channel order changed")
    if _strict_bool_vector(
        gate["direct_actuated"].to_numpy(), name="direct_actuated"
    ).tolist() != direct_mask.tolist():
        raise ValueError("aggregate Gate vector direct mask changed")
    component_mapping = {
        "gate_time_w1_relative_reduction_pass": "time_relative_reduction_pass",
        "gate_occupation_w1_relative_reduction_pass": "occupation_relative_reduction_pass",
        "gate_time_w1_absolute_pass": "time_absolute_pass",
        "gate_occupation_w1_absolute_pass": "occupation_absolute_pass",
        "gate_mean_error_pass": "mean_absolute_error_pass",
        "gate_sd_ratio_pass": "symmetric_sd_ratio_pass",
    }
    components = {
        target: _strict_bool_vector(gate[source].to_numpy(), name=source)
        for target, source in component_mapping.items()
    }
    full_gate_b = np.logical_and.reduce(list(components.values()))
    if not np.array_equal(
        full_gate_b,
        _strict_bool_vector(gate["full_gate_b_pass"].to_numpy(), name="full_gate_b_pass"),
    ):
        raise PermissionError("stored full Gate-B vector is not the AND of all six components")
    safety_gate = _strict_bool_vector(
        gate["safety_gate_pass"].to_numpy(), name="safety_gate_pass"
    )
    full_gate = full_gate_b & safety_gate
    if not np.array_equal(
        full_gate,
        _strict_bool_vector(gate["full_gate_pass"].to_numpy(), name="full_gate_pass"),
    ):
        raise PermissionError("stored full Gate vector is not Gate-B AND safety")

    output.mkdir(parents=True, exist_ok=True)
    candidate_path = output / "common_renderer_input.npz"
    np.savez_compressed(
        candidate_path,
        schema_version=np.asarray("ofrc-final-candidate-v1"),
        shared_protocol_sha256=np.asarray(
            config["shared_protocol"].get(
                "executed_sha256", config["shared_protocol"]["sha256"]
            )
        ),
        frozen_evaluator_manifest_sha256=np.asarray(report["outer_freeze_sha256"]),
        outer_evaluation_role=np.asarray("outer_evaluation"),
        gate_vector_source_sha256=np.asarray(sha256_file(gate_path)),
        safety_vector_source_sha256=np.asarray(sha256_file(safety_path)),
        subject_id=np.asarray("HUP080"),
        candidate_id=np.asarray(report["candidate_id"]),
        channels=channels,
        observed_scaled=np.asarray(rollout["observed_standardized"]),
        free_scaled=np.asarray(rollout["free_standardized"]),
        reference_scaled=np.asarray(rollout["reference_standardized"]),
        controlled_scaled=np.asarray(rollout["controlled_standardized"]),
        direct_mask=direct_mask,
        **components,
        full_gate_b_pass=full_gate_b,
        safety_gate_pass=safety_gate,
        full_gate_pass=full_gate,
        evaluation_context=np.asarray(display_window_id),
        data_role=np.asarray(config["outer_evaluation"]["classification"]),
        model_contract_sha256=np.asarray(report["checkpoint_sha256"]),
        candidate_artifact_sha256=np.asarray(sha256_file(rollout_path)),
        display_window_id=np.asarray(display_window_id),
        display_crn_bank=np.asarray(display_crn_bank),
        aggregate_window_ids=np.asarray(aggregate_window_ids),
    )
    candidate_sha256 = sha256_file(candidate_path)
    binding_fields = {
        "binding_schema_version": "ofrc-final-binding-v1",
        "status": "PENDING_INDEPENDENT_RESULT_AUDIT_DO_NOT_RENDER",
        "subject_id": "HUP080",
        "candidate_id": str(report["candidate_id"]),
        "candidate_npz_sha256": candidate_sha256,
        "shared_protocol_sha256": str(config["shared_protocol"]["sha256"]),
        "frozen_evaluator_manifest_sha256": str(report["outer_freeze_sha256"]),
        "outer_evaluation_role": "outer_evaluation",
        "gate_vector_source_sha256": sha256_file(gate_path),
        "safety_vector_source_sha256": sha256_file(safety_path),
        "display_window_id": display_window_id,
        "display_crn_bank": display_crn_bank,
        "aggregate_window_ids": aggregate_window_ids,
        "independent_result_audit_complete": False,
        "render_authorized": False,
    }
    pending_binding_path = output / "PENDING_FINAL_CANDIDATE_BINDING.json"
    atomic_json(pending_binding_path, binding_fields)
    atomic_json(
        output / "common_renderer_input_manifest.json",
        {
            "schema_version": "common-all-channel-renderer-input-v1",
            "subject": "HUP080", "channel_count": 96,
            "candidate_id": str(report["candidate_id"]),
            "display_window_id": display_window_id,
            "display_crn_bank": display_crn_bank,
            "aggregate_window_ids": aggregate_window_ids,
            "direct_actuator_count": int(direct_mask.sum()),
            "full_gate_b_pass_count": int(full_gate_b.sum()),
            "full_gate_pass_count": int(full_gate.sum()),
            "source_outer_report_sha256": sha256_file(report_path),
            "source_rollout_sha256": sha256_file(rollout_path),
            "source_gate_vectors_sha256": sha256_file(gate_path),
            "source_safety_sha256": sha256_file(safety_path),
            "frozen_input_sha256": candidate_sha256,
            "pending_binding_sha256": sha256_file(pending_binding_path),
            "figure_contract": config["figure_contract"],
            "rendered": False,
            "embedded_patient_renderer_enabled": False,
            "common_renderer_v1_required_after_independent_result_audit": True,
            "go_binding_present": False,
            "overleaf_write": False,
        },
    )


__all__ = ["freeze_common_renderer_input"]
