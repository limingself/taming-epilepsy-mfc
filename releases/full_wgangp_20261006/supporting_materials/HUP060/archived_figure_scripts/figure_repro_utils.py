"""Shared helpers for deterministic, publication-grade paper-figure exports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt


FORMATS = (".svg", ".pdf", ".png", ".tiff")


def configure_publication_style() -> None:
    """Apply the paper's Python/matplotlib-only visual and export contract."""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.5,
            "axes.labelsize": 6.5,
            "axes.titlesize": 7.0,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 5.7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "hup060-distribution-control-2026",
            "savefig.facecolor": "white",
        }
    )


def save_bundle(figure: plt.Figure, stem: Path, *, png_dpi: int = 300) -> None:
    """Export editable vectors plus preview and 600-dpi submission raster."""

    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        stem.with_suffix(".svg"),
        bbox_inches="tight",
        metadata={"Date": "2026-08-02", "Creator": "Python/matplotlib"},
    )
    figure.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={
            "Creator": "Python/matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    figure.savefig(stem.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")
    figure.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)


def copy_bundle(source_stem: Path, destination_directory: Path) -> list[Path]:
    """Copy one complete four-format figure bundle into ``output``."""

    destination_directory.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for suffix in FORMATS:
        source = source_stem.with_suffix(suffix)
        if not source.is_file():
            raise FileNotFoundError(f"missing figure export: {source}")
        destination = destination_directory / source.name
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def copy_files(files: Iterable[Path], destination_directory: Path) -> list[Path]:
    destination_directory.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for source in files:
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = destination_directory / source.name
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
