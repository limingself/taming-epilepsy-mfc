"""Hash-locked path adapter for the frozen common HUP060-style renderer.

This module changes one thing only: the common renderer's formal output-root
guard is extended to the isolated HUP080 v2 ``preview`` directory.  Candidate
validation, binding validation, KDE, layout, styling, export, QA, staging, and
no-overwrite behavior remain the byte-locked common implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any


sys.dont_write_bytecode = True

ADAPTER_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ADAPTER_ROOT.parent
ADAPTER_CONFIG_PATH = ADAPTER_ROOT / "adapter_config.json"
EXPECTED_SCHEMA = "hup080-v2-common-renderer-path-adapter-v1"


class AdapterContractError(RuntimeError):
    """Raised before common-renderer execution when a frozen binding drifts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_adapter_config() -> dict[str, Any]:
    payload = json.loads(ADAPTER_CONFIG_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != EXPECTED_SCHEMA:
        raise AdapterContractError("adapter config schema changed")
    if payload.get("fixed_stage") != "preview":
        raise AdapterContractError("adapter may invoke preview stage only")
    if payload.get("synthetic_bypass_permitted") is not False:
        raise AdapterContractError("formal adapter cannot enable a test bypass")
    if payload.get("patient_specific_renderer_permitted") is not False:
        raise AdapterContractError("patient-specific renderer is forbidden")
    if payload.get("final_export_permitted") is not False:
        raise AdapterContractError("final-format export is not authorized")
    if payload.get("overleaf_write_permitted") is not False:
        raise AdapterContractError("Overleaf writes are forbidden")
    return payload


def _resolve_locked_file(entry: dict[str, Any], *, label: str) -> Path:
    path = Path(str(entry["path"])).resolve()
    if not path.is_file():
        raise AdapterContractError(f"{label} is absent: {path}")
    actual = sha256_file(path)
    if actual != str(entry["sha256"]):
        raise AdapterContractError(
            f"{label} SHA-256 drift: expected {entry['sha256']}, got {actual}"
        )
    return path


def formal_paths(config: dict[str, Any]) -> dict[str, Path]:
    output_root = Path(str(config["allowed_output_root"])).resolve()
    expected_output = (PROJECT_ROOT / "preview").resolve()
    if output_root != expected_output:
        raise AdapterContractError("allowed output root is not the HUP080 v2 preview root")
    return {
        "common_renderer": _resolve_locked_file(
            config["common_renderer"], label="common renderer"
        ),
        "common_config": _resolve_locked_file(
            config["common_config"], label="common renderer config"
        ),
        "candidate": _resolve_locked_file(
            config["formal_candidate"], label="formal candidate"
        ),
        "binding": _resolve_locked_file(
            config["formal_binding"], label="formal binding"
        ),
        "output_root": output_root,
    }


def validate_v2_output_root(path: Path, *, expected: Path) -> None:
    resolved = Path(path).resolve()
    if resolved != Path(expected).resolve():
        raise AdapterContractError(
            "unsafe output root; this adapter accepts only the isolated HUP080 v2 preview root"
        )
    lowered = tuple(part.casefold() for part in resolved.parts)
    if resolved.drive.casefold() == "d:":
        raise AdapterContractError("D: output is forbidden")
    if any("overleaf" in part for part in lowered):
        raise AdapterContractError("Overleaf output is forbidden")
    if any(part.startswith("hup060") for part in lowered):
        raise AdapterContractError("HUP060 output is forbidden")


def load_common_renderer(paths: dict[str, Path]) -> ModuleType:
    module_path = paths["common_renderer"]
    module_name = "hup080_v2_hash_locked_common_renderer"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise AdapterContractError("cannot load the common renderer module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(str(module.__file__)).resolve() != module_path:
        raise AdapterContractError("common renderer module origin changed")
    for name in (
        "render_atomic",
        "_validate_formal_output_root",
        "_load_config",
        "_load_candidate",
        "_load_binding",
    ):
        if not callable(getattr(module, name, None)):
            raise AdapterContractError(f"common renderer interface missing {name}")
    return module


def preflight() -> dict[str, Any]:
    config = load_adapter_config()
    paths = formal_paths(config)
    validate_v2_output_root(paths["output_root"], expected=paths["output_root"])
    module = load_common_renderer(paths)
    common_config = module._load_config(paths["common_config"])
    if bool(common_config["export"]["final_export_enabled"]):
        raise AdapterContractError("common renderer final export unexpectedly enabled")
    candidate = module._load_candidate(paths["candidate"], common_config)
    binding = module._load_binding(
        paths["binding"], input_path=paths["candidate"], candidate=candidate
    )
    return {
        "schema_version": EXPECTED_SCHEMA,
        "status": "PASS",
        "subject_id": str(candidate["subject_id"]),
        "candidate_id": str(candidate["candidate_id"]),
        "channel_count": int(len(candidate["channels"])),
        "direct_actuator_count": int(candidate["direct_mask"].sum()),
        "full_gate_b_pass_count": int(candidate["full_gate_b_pass"].sum()),
        "safety_gate_pass_count": int(candidate["safety_gate_pass"].sum()),
        "binding_status": str(binding["status"]),
        "output_root": str(paths["output_root"]),
        "common_renderer_sha256": sha256_file(paths["common_renderer"]),
        "common_config_sha256": sha256_file(paths["common_config"]),
        "candidate_sha256": sha256_file(paths["candidate"]),
        "binding_sha256": sha256_file(paths["binding"]),
        "renderer_imported_without_matplotlib": "matplotlib" not in sys.modules,
    }


def render_preview(*, run_name: str) -> Path:
    config = load_adapter_config()
    paths = formal_paths(config)
    validate_v2_output_root(paths["output_root"], expected=paths["output_root"])
    module = load_common_renderer(paths)
    original_validator = module._validate_formal_output_root

    def extended_validator(output_root: Path) -> None:
        resolved = Path(output_root).resolve()
        if resolved == paths["output_root"]:
            validate_v2_output_root(resolved, expected=paths["output_root"])
            return
        original_validator(resolved)

    module._validate_formal_output_root = extended_validator
    before = {
        "renderer": sha256_file(paths["common_renderer"]),
        "config": sha256_file(paths["common_config"]),
    }
    try:
        target = module.render_atomic(
            input_path=paths["candidate"],
            binding_manifest_path=paths["binding"],
            output_root=paths["output_root"],
            run_name=run_name,
            config_path=paths["common_config"],
            stage="preview",
        )
    finally:
        module._validate_formal_output_root = original_validator
    after = {
        "renderer": sha256_file(paths["common_renderer"]),
        "config": sha256_file(paths["common_config"]),
    }
    if after != before:
        raise AdapterContractError("common renderer authority changed during execution")
    return Path(target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the frozen HUP080 v2 candidate through the hash-locked common renderer."
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate all hashes, candidate, binding, and path contracts without rendering.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.preflight_only:
            print(json.dumps(preflight(), indent=2, ensure_ascii=False))
            return 0
        print(render_preview(run_name=args.run_name))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
