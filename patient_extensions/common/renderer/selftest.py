#!/usr/bin/env python
"""Deterministic synthetic contract test; contains no patient data."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from render_channel_plate import (
    BINDING_SCHEMA_VERSION,
    ContractError,
    SCHEMA_VERSION,
    _validate_formal_output_root,
    render_atomic,
)


HERE = Path(__file__).resolve().parent
SHARED_PROTOCOL_SHA256 = "a59141b4e239eb6676c915108aaa82ba330cf9574a40158c4702479f6d2f1399"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_synthetic(path: Path, subject_id: str, count: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    samples = 48
    channel_shift = np.linspace(-0.4, 0.4, count)
    observed = rng.normal(0.85 + channel_shift, 0.95, size=(samples, count))
    free = rng.normal(0.55 + channel_shift, 0.82, size=(samples, count))
    reference = rng.normal(0.00 + 0.2 * channel_shift, 0.64, size=(samples, count))
    controlled = rng.normal(0.20 + 0.3 * channel_shift, 0.69, size=(samples, count))
    direct_mask = np.zeros(count, dtype=bool)
    direct_mask[::2] = True
    channel_index = np.arange(count)
    gate_even = channel_index % 2 == 0
    gate_third = channel_index % 3 == 0
    full_gate_b_pass = gate_even & gate_third
    safety_gate_pass = channel_index % 5 != 0
    full_gate_pass = full_gate_b_pass & safety_gate_pass
    np.savez_compressed(
        path,
        schema_version=np.asarray(SCHEMA_VERSION),
        shared_protocol_sha256=np.asarray(SHARED_PROTOCOL_SHA256),
        frozen_evaluator_manifest_sha256=np.asarray("a" * 64),
        outer_evaluation_role=np.asarray("outer_evaluation"),
        gate_vector_source_sha256=np.asarray("b" * 64),
        safety_vector_source_sha256=np.asarray("c" * 64),
        subject_id=np.asarray(subject_id),
        candidate_id=np.asarray("SYNTHETIC_CONTRACT_TEST"),
        channels=np.asarray([f"SYN{index + 1:03d}" for index in range(count)]),
        observed_scaled=observed,
        free_scaled=free,
        reference_scaled=reference,
        controlled_scaled=controlled,
        direct_mask=direct_mask,
        gate_time_w1_relative_reduction_pass=gate_even,
        gate_occupation_w1_relative_reduction_pass=gate_third,
        gate_time_w1_absolute_pass=gate_even,
        gate_occupation_w1_absolute_pass=gate_third,
        gate_mean_error_pass=gate_even,
        gate_sd_ratio_pass=gate_third,
        full_gate_b_pass=full_gate_b_pass,
        safety_gate_pass=safety_gate_pass,
        full_gate_pass=full_gate_pass,
        evaluation_context=np.asarray("synthetic_only"),
        data_role=np.asarray("synthetic_selftest"),
    )


def _write_binding(path: Path, *, candidate_path: Path, subject_id: str) -> None:
    binding = {
        "binding_schema_version": BINDING_SCHEMA_VERSION,
        "status": "GO",
        "subject_id": subject_id,
        "candidate_id": "SYNTHETIC_CONTRACT_TEST",
        "candidate_npz_sha256": _sha256(candidate_path),
        "shared_protocol_sha256": SHARED_PROTOCOL_SHA256,
        "frozen_evaluator_manifest_sha256": "a" * 64,
        "outer_evaluation_role": "outer_evaluation",
        "gate_vector_source_sha256": "b" * 64,
        "safety_vector_source_sha256": "c" * 64,
    }
    path.write_text(
        json.dumps(binding, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _assert_preview(run_dir: Path, expected_occupancy: list[int]) -> None:
    qa_path = run_dir / "qa" / "machine_qa.json"
    manifest_path = run_dir / "manifest.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert qa["status"] == "PASS"
    assert qa["stage"] == "preview"
    assert qa["page_occupancy"] == expected_occupancy
    assert qa["page_file_count"] == len(expected_occupancy)
    assert qa["renderer_infers_gate_or_recovery"] is False
    assert len(qa["gate_b_component_pass_count"]) == 6
    assert qa["full_gate_pass_count"] <= qa["full_gate_b_pass_count"]
    assert manifest["status"] == "PASS"
    assert len(list((run_dir / "preview").glob("*.png"))) == len(expected_occupancy)
    assert not list(run_dir.rglob("*.svg"))
    assert not list(run_dir.rglob("*.pdf"))
    assert not list(run_dir.rglob("*.tiff"))
    assert (run_dir / "source" / "channel_density_long.csv").stat().st_size > 0
    assert (run_dir / "source" / "channel_metadata.csv").stat().st_size > 0
    assert (run_dir / "source" / "canonical_ofrc_input.npz").stat().st_size > 0
    assert (run_dir / "source" / "final_candidate_binding.json").stat().st_size > 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--work-directory", type=Path, default=None)
    args = parser.parse_args()
    if args.work_directory is None:
        # Keep the disposable tree at the workspace root: it is writable in the
        # restricted environment and short enough for legacy Windows path limits.
        root = Path(
            tempfile.mkdtemp(
                prefix=".ofrc_renderer_selftest_",
                dir=HERE.parents[3],
            )
        )
        remove_after = not args.keep
    else:
        root = args.work_directory.resolve()
        root.mkdir(parents=True, exist_ok=False)
        remove_after = False

    production_config = HERE / "config.json"
    test_config = json.loads(production_config.read_text(encoding="utf-8"))
    # Exercise the complete page topology while keeping a routine self-test
    # quick; production preview DPI remains frozen at 300 in config.json.
    test_config["export"]["preview_dpi"] = 72
    config = root / "selftest_config.json"
    config.write_text(
        json.dumps(test_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report: dict[str, object] = {
        "selftest_schema_version": "ofrc-renderer-selftest-v1",
        "synthetic_only": True,
        "production_config_sha256": _sha256(production_config),
        "synthetic_test_config_sha256": _sha256(config),
        "checks": [],
    }
    try:
        forbidden_roots = (
            Path(HERE.anchor) / "forbidden_preview",
            HERE.parents[3] / "Overleaf" / "preview",
            HERE.parents[3] / "HUP060_results" / "preview",
            HERE.parents[3] / "arbitrary_preview",
        )
        for forbidden_root in forbidden_roots:
            try:
                _validate_formal_output_root(forbidden_root.resolve())
            except ContractError:
                continue
            raise AssertionError(f"forbidden output-root was accepted: {forbidden_root}")
        report["checks"].append(
            {
                "formal_output_allowlist": "PASS",
                "rejected_categories": ["D_drive", "Overleaf", "HUP060", "arbitrary"],
            }
        )

        specifications = (
            ("HUP065", 64, [36, 28], 65001),
            ("HUP080", 96, [36, 36, 24], 80001),
        )
        for subject_id, count, occupancy, seed in specifications:
            candidate = root / f"{subject_id.lower()}_synthetic.npz"
            _write_synthetic(candidate, subject_id, count, seed)
            binding_manifest = root / f"{subject_id.lower()}_binding.json"
            _write_binding(
                binding_manifest,
                candidate_path=candidate,
                subject_id=subject_id,
            )
            output_root = root / "outputs"
            run_name = f"{subject_id.lower()}_preview"
            run_dir = render_atomic(
                input_path=candidate,
                binding_manifest_path=binding_manifest,
                output_root=output_root,
                run_name=run_name,
                config_path=config,
                stage="preview",
                _synthetic_selftest=True,
            )
            _assert_preview(run_dir, occupancy)
            manifest_before = _sha256(run_dir / "manifest.json")
            try:
                render_atomic(
                    input_path=candidate,
                    binding_manifest_path=binding_manifest,
                    output_root=output_root,
                    run_name=run_name,
                    config_path=config,
                    stage="preview",
                    _synthetic_selftest=True,
                )
            except FileExistsError:
                pass
            else:
                raise AssertionError("no-overwrite contract did not reject an existing run")
            assert _sha256(run_dir / "manifest.json") == manifest_before
            report["checks"].append(
                {
                    "subject_id": subject_id,
                    "channel_count": count,
                    "page_occupancy": occupancy,
                    "preview_contract": "PASS",
                    "no_overwrite_contract": "PASS",
                }
            )

        candidate = root / "hup065_synthetic.npz"
        binding_manifest = root / "hup065_binding.json"
        non_go_binding = root / "hup065_non_go_binding.json"
        non_go_payload = json.loads(binding_manifest.read_text(encoding="utf-8"))
        non_go_payload["status"] = "STOP"
        non_go_binding.write_text(
            json.dumps(non_go_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        try:
            render_atomic(
                input_path=candidate,
                binding_manifest_path=non_go_binding,
                output_root=root / "outputs",
                run_name="non_go_preview",
                config_path=config,
                stage="preview",
                _synthetic_selftest=True,
            )
        except ContractError:
            report["checks"].append({"hash_binding_go_gate": "PASS"})
        else:
            raise AssertionError("non-GO binding manifest was not blocked")

        try:
            render_atomic(
                input_path=candidate,
                binding_manifest_path=binding_manifest,
                output_root=root / "outputs",
                run_name="unconfirmed_final",
                config_path=config,
                stage="final",
                _synthetic_selftest=True,
            )
        except ContractError:
            report["checks"].append({"unconfirmed_final_lock": "PASS"})
        else:
            raise AssertionError("unconfirmed final export was not blocked")

        staging_leftovers = list((root / "outputs").glob(".*.staging-*"))
        assert not staging_leftovers, staging_leftovers
        report["checks"].append({"atomic_staging_cleanup": "PASS"})
        report["status"] = "PASS"
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    finally:
        if remove_after and root.exists():
            shutil.rmtree(root)


if __name__ == "__main__":
    raise SystemExit(main())
