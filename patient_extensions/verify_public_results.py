#!/usr/bin/env python
"""Verify the safe public HUP065/HUP080 extension without patient arrays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import struct
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
MANIFEST = ROOT / "PUBLIC_MANIFEST.json"
LOCAL_ONLY = {
    "HUP065/config.json",
    "HUP065/SOURCE_ADAPTATION_AUDIT.json",
    "HUP080/config.json",
    "HUP080/SOURCE_BINDING_AUDIT.json",
    "HUP080/renderer_adapter/adapter_config.json",
    "common/protocol/protocol.json",
    "common/legacy_adapters/stage_contract.json",
}
FORBIDDEN_SUFFIXES = {
    ".npz", ".npy", ".pt", ".pth", ".joblib", ".pkl", ".pickle",
    ".edf", ".zip",
}
FORBIDDEN_DIRS = {
    ".runtime", ".staging", "science_run", "raw", "raw_data", "cache",
    "runs",
}
FORBIDDEN_BASENAMES = {
    "OUTER_GO.json", "RUN04_GO.json", "ACCESS_STARTED.json", "OUTER_OPENED.json",
}
FORBIDDEN_NAME_FRAGMENTS = {
    "channel_density_long", "selection_long", "training_history",
    "trajectory_safety_metrics",
}
TEXT_SUFFIXES = {
    ".py", ".json", ".md", ".txt", ".csv", ".yaml", ".yml", ".gitignore",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(relative: str) -> dict[str, Any]:
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"JSON root is not an object: {relative}")
    return value


def public_files(*, include_manifest: bool) -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative in LOCAL_ONLY or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if not include_manifest and path == MANIFEST:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def assert_close(observed: float, expected: float, *, tolerance: float = 1e-15) -> None:
    if abs(float(observed) - expected) > tolerance:
        raise AssertionError(f"{observed!r} != {expected!r}")


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError(f"not a PNG: {path}")
    return struct.unpack(">II", header[16:24])


def verify_patient_semantics() -> list[str]:
    checks: list[str] = []
    expected = {
        "HUP065": {
            "channels": 64, "direct": 23, "gate_b": 21, "block": 8,
            "candidate": "f-0.35_q-0.65_gain-0.350_tau-0.200",
            "time_free": 0.27179461046098846,
            "time_controlled": 0.2090708032460317,
            "occ_free": 0.17950937749047155,
            "occ_controlled": 0.13887517391509355,
        },
        "HUP080": {
            "channels": 96, "direct": 76, "gate_b": 65, "block": 2,
            "candidate": "f-0.80_q-0.20_gain-0.350_tau-0.000",
            "time_free": 0.3286193598915133,
            "time_controlled": 0.0960328004826287,
            "occ_free": 0.32035737631043043,
            "occ_controlled": 0.07818781426942978,
        },
    }
    for subject, frozen in expected.items():
        summary = load_json(f"{subject}/results/outer_summary.json")
        selection = load_json(f"{subject}/results/controller_selection.json")
        safety = load_json(f"{subject}/results/safety_summary.json")
        assert summary["subject"] == subject
        assert summary["channels"] == frozen["channels"]
        assert summary["direct_actuators"] == frozen["direct"]
        assert summary["outer_gate_b_pass_count"] == frozen["gate_b"]
        assert summary["rolling_block_samples"] == frozen["block"]
        assert summary["candidate_id"] == frozen["candidate"]
        assert summary["outer_gate_c_pass"] is True
        assert summary["outer_used_for_selection_or_rescue"] is False
        assert_close(summary["mean_time_w1_free"], frozen["time_free"])
        assert_close(summary["mean_time_w1_controlled"], frozen["time_controlled"])
        assert_close(summary["mean_occupation_w1_free"], frozen["occ_free"])
        assert_close(summary["mean_occupation_w1_controlled"], frozen["occ_controlled"])
        direct = selection["direct_channel_indices_zero_based"]
        indirect = selection["indirect_channel_indices_zero_based"]
        assert len(direct) == frozen["direct"] == selection["direct_actuator_count"]
        assert len(indirect) == frozen["channels"] - frozen["direct"]
        assert sorted(direct + indirect) == list(range(frozen["channels"]))
        assert len(set(direct).intersection(indirect)) == 0
        mask = "".join("1" if index in set(direct) else "0" for index in range(frozen["channels"]))
        assert hashlib.sha256(mask.encode("ascii")).hexdigest() == selection["direct_mask_bitstring_sha256"]
        if subject == "HUP065":
            gates = load_json("HUP065/results/gate_vectors.json")
            assert sum(gates["full_gate_b_pass"]) == 21
            assert len(safety["safety_gate_pass"]) == 64
            assert all(safety["safety_gate_pass"])
        else:
            assert safety["all_predeclared_windows_x_banks_gate_c_pass"] is True
            assert safety["context_bank_evaluations"] == 24
            assert summary["outer_both_improved_all_context_bank_count"] == 96
            with (ROOT / "HUP080/results/gate_vectors.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == 96
            assert sum(row["direct_actuated"] == "True" for row in rows) == 76
            assert sum(row["full_gate_b_pass"] == "True" for row in rows) == 65
            assert all(row["safety_gate_pass"] == "True" for row in rows)
        checks.append(f"{subject} frozen aggregate/count/vector contract")

    v1 = load_json("HUP080/results/v1_rolling_no_go.json")
    assert v1["parent_v1_candidates_samples"] == [4, 8, 12, 16, 24, 32]
    assert v1["parent_v1_implementation_freeze_sha256"] == (
        "359745077c159ec9ea5a6aebca045e8f55f3d8af0b9024777f68df0004982ffd"
    )
    assert v1["outcome"] == "NO_GO_before_part3"
    assert v1["outer_run04_seen"] is False
    assert v1["private_failed_stage_required_for_public_verification"] is False
    assert v1["failure_artifacts_read_by_v2_science"] is False
    inventory = v1["failure_artifact_inventory"]
    assert len(inventory) == 13
    for row in inventory.values():
        assert int(row["bytes"]) > 0
        assert len(str(row["sha256"])) == 64
        int(str(row["sha256"]), 16)
    v2 = v1["exploratory_v2"]
    assert v2["extended_candidates_samples"] == [1, 2, 4, 8, 12, 16, 24, 32]
    assert v2["thresholds_changed"] is False
    assert v2["joint_fold_rule_changed"] is False
    assert v2["parent_no_go_remains_valid"] is True
    assert v2["may_be_described_as_original_preregistered_replication"] is False
    template = load_json("HUP080/config.template.json")
    amendment = template["exploratory_v2_delta"]
    assert amendment["v1_failure_public_receipt"] == (
        "${PATIENT_RUN_ROOT}/results/v1_rolling_no_go.json"
    )
    assert amendment["v1_failure_public_receipt_sha256"] == sha256_file(
        ROOT / "HUP080/results/v1_rolling_no_go.json"
    )
    execution_paths = [
        ROOT / "HUP080/runner.py",
        *(ROOT / "HUP080/h080" / name for name in (
            "data_model.py", "control.py", "training.py", "outer.py", "pipeline.py",
            "renderer_input.py",
        )),
    ]
    execution_text = "\n".join(path.read_text(encoding="utf-8") for path in execution_paths)
    assert "v1_failure_public_receipt" not in execution_text
    assert "failure_artifact_inventory" not in execution_text
    checks.append("HUP080 hash-only parent-v1 NO_GO receipt and exploratory-v2 amendment contract")

    h65_failure = load_json("HUP065/results/failure_provenance.json")
    assert h65_failure["status"] == "HASH_ONLY_AUDIT_PROVENANCE_NOT_A_SCIENCE_INPUT"
    assert h65_failure["private_failure_stages_bundled"] is False
    assert h65_failure["private_failure_stages_required_for_public_self_test"] is False
    assert h65_failure["selection_use"] == "DO_NOT_REUSE_FOR_SELECTION"
    assert h65_failure["preflight_path_length_failure"]["artifact_count"] == 8
    analytical_failure = h65_failure["analytical_prefilter_failure"]
    assert analytical_failure["artifact_count"] == 42
    assert analytical_failure["total_bytes"] == 106976589
    assert analytical_failure["archive_inventory_sha256"] == (
        "a512cd649bb7148a1cec7a7b6fffd2eb3c5a3509dfd403ead707c1516eca8941"
    )
    h65_template = load_json("HUP065/config.template.json")
    assert h65_template["public_failure_provenance"]["sha256"] == sha256_file(
        ROOT / "HUP065/results/failure_provenance.json"
    )
    h65_science_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "HUP065/runner.py",
            *(ROOT / "HUP065/h065" / name for name in (
                "data_model.py", "control.py", "training.py", "outer.py"
            )),
        )
    )
    assert "failure_provenance.json" not in h65_science_text
    checks.append("HUP065 hash-only private-failure provenance receipt contract")

    selection = load_json("HUP080/results/controller_selection.json")
    assert selection["parameters"]["tau"] == 0.0
    assert selection["indirect_channel_count"] == 20
    semantics = selection["tau_semantics"].casefold()
    assert "no instantaneous heat-kernel control input" in semantics
    assert "plant dynamics" in semantics
    plant_source = (REPO_ROOT / "mfc_pipeline/square_wave_mfc.py").read_text(encoding="utf-8")
    assert "selector @ heat_kernel" in plant_source
    assert "control_graph_diffusion_time" in plant_source
    checks.append("HUP080 tau_G=0 identity-map/20-indirect-channel semantics")
    return checks


def verify_publication_figures() -> list[str]:
    checks: list[str] = []
    for subject, pages in (("HUP065", 2), ("HUP080", 3)):
        qa = load_json(f"{subject}/qa/publication_qa.json")
        assert qa["status"] == "PASS"
        if subject == "HUP080":
            assert qa["accepted_package"] == "package_overleaf_final_v2"
            assert qa["authorized_visible_revision"]["to"].startswith("Direct actuation: 76/96")
        dimensions = tuple(qa["png_dimensions_px_at_300_dpi"] if subject == "HUP065" else qa["export_contract"]["png_dimensions_px_at_300_dpi"])
        for page in range(1, pages + 1):
            label = f"page_{page:02d}"
            prefix = ROOT / subject / "publication" / f"{subject.casefold()}_ofrc_all_channels_{label}"
            pdf = prefix.with_suffix(".pdf")
            png = prefix.with_suffix(".png")
            assert sha256_file(pdf) == qa["accepted_pdf_sha256"][label]
            assert sha256_file(png) == qa["accepted_png_sha256"][label]
            assert png_dimensions(png) == dimensions
        checks.append(f"{subject} accepted publication PDF/PNG hashes and dimensions")
    return checks


def verify_portability_map() -> list[str]:
    mapping = load_json("PORTABILITY_MAP.json")
    assert mapping["status"] == (
        "PATH_AND_AUTHORIZATION_ONLY_PUBLIC_RELOCATION_SCIENTIFIC_PARAMETERS_UNCHANGED"
    )
    bindings = mapping["executed_config_bindings"]
    assert bindings["HUP065"]["executed_config_sha256"] == (
        "7f436ba67234c1e0007d29d925d06d2f353483da0cbb026538be322a83268ccf"
    )
    assert bindings["HUP080_exploratory_v2"]["executed_config_sha256"] == (
        "4e46bb903166dd98b2805bf5aa9ca581744aa6fe797893b57c669a87dbcdfcab"
    )
    assert bindings["HUP080_exploratory_v2"]["parent_v1_executed_config_sha256"] == (
        "0539881d94304cd7116839bf7a9ffa1fcb8a0382cce5bb90dc1eb641aa0fac1a"
    )
    assert bindings["shared_protocol"]["executed_protocol_sha256"] == (
        "a59141b4e239eb6676c915108aaa82ba330cf9574a40158c4702479f6d2f1399"
    )
    assert mapping["private_archive_manifest_anchors"] == {
        "HUP065": "df6752e1916eb12e1bb0d16330fc77ce63e4444846d675cddcc4213c3f36a637",
        "HUP080": "6382eb5aec69643855866765ac39eb981f93abb6b99727e145ddb7d803ad4f0d",
    }
    checkout = mapping["checkout_byte_portability"]
    assert checkout["executed_windows_byte_hashes_retained_in_committed_templates"] is True
    assert checkout["materialized_checkout_local_locks_use_current_file_bytes"] is True
    assert checkout["requires_core_autocrlf"] is False
    assert checkout["tracked_hup060_source_blobs_modified"] is False
    assert checkout["serialized_models_or_raw_archives_refreshed_or_published"] is False
    assert checkout["public_directory_allowlist_additions"] == ["HUP065", "HUP080"]
    materializer = (ROOT / "materialize_config.py").read_text(encoding="utf-8")
    for source_fragment in (
        "def refresh_checkout_locks(",
        '"eol_only"',
        'for public_name in ("HUP065", "HUP080")',
        '"executed_canonical_hup060_lock_sha256"',
        "NON_SOURCE_SUFFIXES",
    ):
        assert source_fragment in materializer
    protocol_selftest = (
        ROOT / "common/protocol/static_self_test.py"
    ).read_text(encoding="utf-8")
    assert 'protocol["public_portability"]' in protocol_selftest
    assert 'sha256(Path(locks[name]["path"])) == locks[name]["sha256"]' in protocol_selftest
    for binding in bindings.values():
        template = binding.get("public_template")
        if template:
            assert sha256_file(ROOT / template) == binding["public_template_sha256"]
    for row in mapping["source_adaptations"]:
        assert sha256_file(ROOT / row["public_path"]) == row["public_sha256"]
        assert row.get("scientific_equations_or_parameters_changed", False) is False
    for row in mapping["byte_identical_science_core_examples"]:
        assert sha256_file(ROOT / row["path"]) == row["sha256"]
    assert mapping["source_adaptations"][-1]["objective_function_ast_equal_to_executed_source"] is True
    return [
        "executed-freeze/public-template provenance and portability-map hashes",
        "common protocol executed-provenance/runtime-checkout lock separation",
    ]


def sensitive_scan() -> dict[str, Any]:
    files = public_files(include_manifest=True)
    forbidden_files: list[str] = []
    large_csv: list[str] = []
    text_findings: list[dict[str, str]] = []
    absolute_path = re.compile(r"(?i)(?:[a-z]:[\\/]|/(?:users|home)/[^/\s]+/)")
    credential_patterns = (
        ("private_key", re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY")),
        ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}")),
        ("fixed_hup065_authorization", re.compile("I_" + "AUTHORIZE_FRESH_HUP065")),
        ("fixed_hup080_authorization", re.compile("HUP080_FRESH_SPARSE_" + "SCIENCE_GO_V1")),
        ("private_username", re.compile("Li" + "Ming", re.IGNORECASE)),
    )
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        lower = relative.casefold()
        if path.suffix.casefold() in FORBIDDEN_SUFFIXES:
            forbidden_files.append(relative)
        if any(part.casefold() in FORBIDDEN_DIRS for part in path.parts):
            forbidden_files.append(relative)
        if path.name in FORBIDDEN_BASENAMES:
            forbidden_files.append(relative)
        if any(fragment in lower for fragment in FORBIDDEN_NAME_FRAGMENTS):
            forbidden_files.append(relative)
        if path.suffix.casefold() == ".csv" and path.stat().st_size > 128 * 1024:
            large_csv.append(relative)
        if path.suffix.casefold() in TEXT_SUFFIXES or path.name == ".gitignore":
            text = path.read_text(encoding="utf-8", errors="replace")
            if absolute_path.search(text):
                text_findings.append({"file": relative, "type": "absolute_user_path"})
            for name, pattern in credential_patterns:
                if pattern.search(text):
                    text_findings.append({"file": relative, "type": name})
    unique_forbidden = sorted(set(forbidden_files))
    status = "PASS" if not unique_forbidden and not large_csv and not text_findings else "FAIL"
    return {
        "schema_version": "patient-extensions-sensitive-scan-v1",
        "status": status,
        "files_scanned": len(files),
        "forbidden_payload_files": unique_forbidden,
        "large_csv_files_over_128_kib": sorted(large_csv),
        "absolute_path_or_credential_findings": text_findings,
        "checks": [
            "no patient arrays, serialized models, EEG archives, raw/cache/run/staging directories",
            "no large or long-form result tables",
            "no one-time GO/access receipt files",
            "no machine-absolute user paths, private username, fixed authorization values, private keys, or GitHub token forms",
        ],
        "note": "Protocol code may name local receipt filenames and output extensions; the corresponding private payloads are not distributed.",
    }


def build_manifest() -> dict[str, Any]:
    rows = []
    for path in public_files(include_manifest=False):
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith("HUP065/publication/") or relative.startswith("HUP080/publication/"):
            role = "accepted_publication_figure"
        elif "/results/" in relative:
            role = "lightweight_result"
        elif "/qa/" in relative or relative.startswith("qa/"):
            role = "quality_assurance"
        elif relative.endswith(".template.json"):
            role = "portable_template"
        elif relative.endswith(".py"):
            role = "portable_code"
        else:
            role = "documentation_or_environment"
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "role": role,
            }
        )
    return {
        "schema_version": "patient-extensions-public-manifest-v1",
        "scope": "patient_extensions; excludes this self-referential manifest and ignored machine-local materializations",
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def verify_manifest() -> list[str]:
    manifest = load_json("PUBLIC_MANIFEST.json")
    actual = public_files(include_manifest=False)
    actual_names = [path.relative_to(ROOT).as_posix() for path in actual]
    expected_names = [row["path"] for row in manifest["files"]]
    assert actual_names == expected_names
    assert manifest["file_count"] == len(actual)
    assert manifest["total_bytes"] == sum(path.stat().st_size for path in actual)
    for row, path in zip(manifest["files"], actual):
        assert row["bytes"] == path.stat().st_size
        assert row["sha256"] == sha256_file(path)
    return ["PUBLIC_MANIFEST exact file-set/size/SHA-256 closure"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="write deterministic QA reports and rebuild PUBLIC_MANIFEST before verification",
    )
    args = parser.parse_args()
    checks = verify_patient_semantics() + verify_publication_figures() + verify_portability_map()
    scan = sensitive_scan()
    if scan["status"] != "PASS":
        raise AssertionError(json.dumps(scan, ensure_ascii=False, indent=2))
    checks.append("sensitive/public-payload scan")
    if args.refresh:
        write_json(ROOT / "qa" / "SENSITIVE_SCAN_REPORT.json", scan)
        write_json(
            ROOT / "qa" / "TEST_REPORT.json",
            {
                "schema_version": "patient-extensions-test-report-v1",
                "status": "PASS",
                "checks": checks,
                "test_scope": "stdlib-only aggregate, vector, provenance, figure-hash/dimension, and sensitive-payload verification",
                "patient_signal_arrays_loaded": False,
            },
        )
        write_json(MANIFEST, build_manifest())
    checks += verify_manifest()
    result = {
        "status": "PASS",
        "checks": checks,
        "manifest_sha256": sha256_file(MANIFEST),
        "manifest_file_count": load_json("PUBLIC_MANIFEST.json")["file_count"],
        "manifest_total_bytes": load_json("PUBLIC_MANIFEST.json")["total_bytes"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
