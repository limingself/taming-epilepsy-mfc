"""Freeze/verify hashes of the exact reproducibility source and input bundle.

This inventories only explicitly scoped bundle folders, not any original
project, patient EDF directory, or broad drive root. Training outputs are not
part of this source/input manifest; their hashes are in training/evaluation
contracts. Execute with --freeze before archive and --verify to check later.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "reproducibility_manifest.json"
PACKAGES = (
    "torch", "numpy", "pandas", "scipy", "scikit-learn", "joblib",
    "matplotlib", "networkx", "PyYAML", "threadpoolctl", "pillow",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scoped_files() -> list[Path]:
    files = [ROOT / name for name in (
        "run_experiment.py", "plot_all_channels.py", "reproduce_portable.py",
        "bundle_manifest.py", "audit_results.py", "README.md", "RESULTS.md",
    )]
    files.extend(sorted((ROOT / "repro_source").rglob("*.py")))
    files.extend(sorted((ROOT / "repro_source" / "artifacts").rglob("*.npz")))
    files.extend(sorted((ROOT / "repro_source" / "artifacts").rglob("*.pt")))
    files.extend(sorted((ROOT / "repro_source" / "artifacts").rglob("*.joblib")))
    files.append(ROOT / "repro_inputs" / "paper_baseline_paired_comparison.npz")
    missing = [str(path.relative_to(ROOT)) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing bundle files: " + ", ".join(missing))
    return sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix())


def role(relative: str) -> str:
    if relative == "repro_inputs/paper_baseline_paired_comparison.npz":
        return "Frozen run02 evaluation and paper comparison only; never loaded during training"
    if relative.endswith("/frozen_actor.pt") or relative.endswith("/paired_rollout.npz"):
        return "Historical artifact hash protection only; not loaded as teacher, initialization, or training target"
    if relative.endswith("/synthesis_only_inputs.npz"):
        return "Run01 fit/validation reference pools, past context, mask and frozen-plant contract"
    if relative.endswith(".joblib"):
        return "Trusted local frozen Graph-RC-SDE plant; not retrained"
    if relative.startswith("repro_source/"):
        return "Unchanged implementation snapshot imported by the portable runner"
    return "Independent experiment, plotting, or reproducibility helper"


def environment() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    import torch
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": versions,
        "torch_runtime_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_built_with_mkl": torch.backends.mkl.is_available(),
        "torch_built_with_openmp": torch.backends.openmp.is_available(),
        "training_device": "CPU",
        "training_threads": 4,
        "training_interop_threads": 1,
        "floating_point_dtype": "float64",
        "bitwise_cross_platform_reproduction_guaranteed": False,
    }


def freeze() -> dict:
    files = scoped_files()
    metadata = environment()
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Source/input snapshot for exploratory single-seed HUP060 fresh hybrid WGAN-GP controller training",
        "files": [{
            "relative_path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "role": role(path.relative_to(ROOT).as_posix()),
        } for path in files],
        "environment": metadata,
        "requires_original_absolute_D_drive_paths": False,
        "contains_raw_patient_edf": False,
        "run02_is_independent_external_validation": False,
        "training_seeds": 1,
        "adopted_in_manuscript": False,
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "environment_versions.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    requirements = [
        "# Exact installed distributions recorded for this CPU experiment.",
        "# Use Python " + metadata["python_version"] + "; package index availability may differ.",
        "# A PyTorch +cpu build may require the official PyTorch CPU wheel index.",
    ]
    requirements += [name + "==" + version for name, version in metadata["packages"].items() if version]
    (ROOT / "requirements-recorded.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")
    return {"manifest": str(MANIFEST), "file_count": len(files), "total_bytes": sum(p.stat().st_size for p in files)}


def verify() -> dict:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mismatches = []
    for record in payload["files"]:
        relative = Path(record["relative_path"])
        path = (ROOT / relative).resolve()
        if ROOT.resolve() not in path.parents:
            raise RuntimeError("Manifest path escapes bundle root")
        if not path.is_file():
            mismatches.append({"relative_path": record["relative_path"], "problem": "missing"})
        elif path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            mismatches.append({"relative_path": record["relative_path"], "problem": "content mismatch"})
    return {
        "manifest": str(MANIFEST), "checked_files": len(payload["files"]),
        "all_present_and_matching": not mismatches, "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--freeze", action="store_true")
    modes.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = freeze() if args.freeze else verify()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.verify and not result["all_present_and_matching"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
