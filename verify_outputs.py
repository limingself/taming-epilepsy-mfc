#!/usr/bin/env python
"""Validate the complete paper-figure bundle and record provenance hashes."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

from PIL import Image


ROOT = Path(__file__).resolve().parent
FORMATS = ("svg", "pdf", "png", "tiff")


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_signature(path: Path, extension: str) -> dict[str, object]:
    header = path.read_bytes()[:16]
    if extension == "pdf" and not header.startswith(b"%PDF"):
        raise RuntimeError(f"invalid PDF signature: {path}")
    if extension == "png" and not header.startswith(b"\x89PNG"):
        raise RuntimeError(f"invalid PNG signature: {path}")
    if extension == "tiff" and header[:4] not in (b"II*\x00", b"MM\x00*"):
        raise RuntimeError(f"invalid TIFF signature: {path}")
    if extension == "svg" and "<svg" not in path.read_text(encoding="utf-8", errors="ignore")[:4096]:
        raise RuntimeError(f"invalid SVG document: {path}")
    result: dict[str, object] = {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if extension in {"png", "tiff"}:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            result["pixel_size"] = [int(image.width), int(image.height)]
            result["dpi"] = [float(value) for value in image.info.get("dpi", (0.0, 0.0))]
    return result


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "networkx",
        "mne",
        "torch",
        "PyYAML",
        "tqdm",
        "joblib",
        "Pillow",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def source_records(path: Path) -> list[dict[str, object]]:
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(item for item in path.rglob("*") if item.is_file())
    else:
        raise FileNotFoundError(f"missing frozen source data: {path}")
    return [
        {
            "path": str(item.relative_to(ROOT)).replace("\\", "/"),
            "bytes": item.stat().st_size,
            "sha256": sha256(item),
        }
        for item in files
    ]


def verify(parts: set[int] | None = None) -> dict[str, object]:
    contract = json.loads((ROOT / "paper_figures.json").read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    for figure in contract["figures"]:
        if parts is not None and int(figure["part"]) not in parts:
            continue
        directory = ROOT / figure["output_directory"]
        stem = str(figure["stem"])
        generator = ROOT / figure["generator"]
        if not generator.is_file():
            raise FileNotFoundError(f"missing figure generator: {generator}")
        source = ROOT / figure["source_data"]
        derived_source = figure.get("derived_source_data")
        exports = []
        for extension in FORMATS:
            path = directory / f"{stem}.{extension}"
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"missing or empty paper output: {path}")
            exports.append(validate_signature(path, extension))
        record = {
            "number": int(figure["number"]),
            "part": int(figure["part"]),
            "title": figure["title"],
            "generator": figure["generator"],
            "generator_sha256": sha256(generator),
            "source_data": figure["source_data"],
            "source_files": source_records(source),
            "exports": exports,
        }
        if derived_source is not None:
            derived_path = ROOT / str(derived_source)
            record["derived_source_data"] = str(derived_source)
            record["derived_source_files"] = source_records(derived_path)
        records.append(record)
    payload = {
        "status": "pass",
        "validated_figure_count": len(records),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "figures": records,
    }
    if parts is None:
        manifest_name = "reproduction_manifest.json"
    else:
        label = "_".join(str(value) for value in sorted(parts))
        manifest_name = f"reproduction_manifest_part_{label}.json"
    destination = ROOT / "output" / manifest_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=("all", "1", "2", "3"), default="all")
    args = parser.parse_args()
    parts = None if args.part == "all" else {int(args.part)}
    result = verify(parts)
    print(json.dumps({"status": result["status"], "validated": result["validated_figure_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
