#!/usr/bin/env python
"""Candidate-neutral all-channel O/F/R/C occupation-law renderer.

This module is deliberately presentation-only. It consumes a frozen candidate
artifact, displays externally supplied direct/indirect and full-Gate metadata,
and never infers Gate passage or scientific recovery from curve appearance.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid

import numpy as np
from scipy.stats import wasserstein_distance

SCHEMA_VERSION = "ofrc-final-candidate-v1"
BINDING_SCHEMA_VERSION = "ofrc-final-binding-v1"
SERIES_ORDER = ("O", "F", "R", "C")
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
plt = None
Line2D = None


def _initialize_python_plotting(mpl_config_dir: Path) -> None:
    """Import and configure matplotlib only after the runtime path is frozen."""
    global plt, Line2D
    if plt is not None:
        return
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot
    from matplotlib.lines import Line2D as MatplotlibLine2D

    plt = pyplot
    Line2D = MatplotlibLine2D
    # Mandatory publication settings: Python is the exclusive drawing backend.
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8


class ContractError(ValueError):
    """Raised when an input violates the frozen rendering contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ContractError(
            f"config schema_version must be {SCHEMA_VERSION!r}, got "
            f"{config.get('schema_version')!r}"
        )
    return config


def _decode_scalar(value: np.ndarray | str | bytes) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ContractError("string metadata must be scalar")
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _require_sha256(value: str, *, name: str) -> str:
    normalized = value.strip().lower()
    if not SHA256_RE.fullmatch(normalized):
        raise ContractError(f"{name} must be a lowercase 64-character SHA-256")
    return normalized


def _strict_bool_vector(raw: np.ndarray, *, name: str, count: int) -> np.ndarray:
    vector = np.asarray(raw).reshape(-1)
    if vector.shape != (count,):
        raise ContractError(f"{name} must have shape ({count},), got {vector.shape}")
    if vector.dtype == np.bool_:
        return vector.copy()
    if not np.all(np.isin(vector, [0, 1])):
        raise ContractError(f"{name} must contain only booleans or 0/1")
    return vector.astype(bool)


def _normalize_samples(raw: np.ndarray, *, key: str, channels: int) -> np.ndarray:
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim < 2 or array.shape[-1] != channels:
        raise ContractError(
            f"{key} must have at least two dimensions and end in {channels} channels; "
            f"got {array.shape}"
        )
    normalized = array.reshape(-1, channels)
    if not np.all(np.isfinite(normalized)):
        raise ContractError(f"{key} contains NaN or infinite values")
    return normalized


def _load_candidate(path: Path, config: dict) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"candidate NPZ does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        controlled_key = (
            "controlled_scaled"
            if "controlled_scaled" in keys
            else "candidate_controlled_scaled"
            if "candidate_controlled_scaled" in keys
            else None
        )
        required = {
            "schema_version",
            "shared_protocol_sha256",
            "frozen_evaluator_manifest_sha256",
            "outer_evaluation_role",
            "gate_vector_source_sha256",
            "safety_vector_source_sha256",
            "subject_id",
            "candidate_id",
            "channels",
            "observed_scaled",
            "free_scaled",
            "reference_scaled",
            "direct_mask",
            "safety_gate_pass",
            "full_gate_b_pass",
            "full_gate_pass",
        }
        required.update(
            component["key"] for component in config["gate_b_components"]
        )
        missing = sorted(required - keys)
        if controlled_key is None:
            missing.append("controlled_scaled (or candidate_controlled_scaled)")
        if missing:
            raise ContractError("candidate NPZ is missing: " + ", ".join(missing))

        schema = _decode_scalar(data["schema_version"])
        if schema != SCHEMA_VERSION:
            raise ContractError(
                f"candidate schema_version must be {SCHEMA_VERSION!r}, got {schema!r}"
            )
        shared_protocol_sha256 = _require_sha256(
            _decode_scalar(data["shared_protocol_sha256"]),
            name="shared_protocol_sha256",
        )
        required_protocol_sha256 = str(
            config["provenance"]["required_shared_protocol_sha256"]
        )
        if shared_protocol_sha256 != required_protocol_sha256:
            raise ContractError(
                "candidate shared_protocol_sha256 does not match the frozen common "
                f"Part I--III protocol: {shared_protocol_sha256!r} != "
                f"{required_protocol_sha256!r}"
            )
        subject_id = _decode_scalar(data["subject_id"])
        candidate_id = _decode_scalar(data["candidate_id"])
        if not candidate_id.strip():
            raise ContractError("candidate_id must be non-empty")
        frozen_evaluator_manifest_sha256 = _require_sha256(
            _decode_scalar(data["frozen_evaluator_manifest_sha256"]),
            name="frozen_evaluator_manifest_sha256",
        )
        gate_vector_source_sha256 = _require_sha256(
            _decode_scalar(data["gate_vector_source_sha256"]),
            name="gate_vector_source_sha256",
        )
        safety_vector_source_sha256 = _require_sha256(
            _decode_scalar(data["safety_vector_source_sha256"]),
            name="safety_vector_source_sha256",
        )
        outer_evaluation_role = _decode_scalar(data["outer_evaluation_role"])
        allowed_roles = set(config["integrity"]["allowed_outer_evaluation_roles"])
        if outer_evaluation_role not in allowed_roles:
            raise ContractError(
                f"outer_evaluation_role must be one of {sorted(allowed_roles)}, got "
                f"{outer_evaluation_role!r}"
            )
        channels = np.asarray(data["channels"]).astype(str).reshape(-1)
        count = int(channels.size)
        if count == 0:
            raise ContractError("channels cannot be empty")
        if any(not channel.strip() for channel in channels):
            raise ContractError("channel names must be non-empty")
        if len(set(channels.tolist())) != count:
            raise ContractError("channel names must be unique")

        profiles = config["profiles"]
        if subject_id not in profiles:
            raise ContractError(
                f"subject_id {subject_id!r} has no frozen page profile; "
                f"allowed: {', '.join(sorted(profiles))}"
            )
        expected_count = int(profiles[subject_id]["channel_count"])
        if count != expected_count:
            raise ContractError(
                f"{subject_id} profile requires {expected_count} channels, got {count}"
            )

        samples = {
            "O": _normalize_samples(data["observed_scaled"], key="observed_scaled", channels=count),
            "F": _normalize_samples(data["free_scaled"], key="free_scaled", channels=count),
            "R": _normalize_samples(data["reference_scaled"], key="reference_scaled", channels=count),
            "C": _normalize_samples(data[controlled_key], key=controlled_key, channels=count),
        }
        minimum = int(config["integrity"]["minimum_samples_per_series_per_channel"])
        for code, values in samples.items():
            if values.shape[0] < minimum:
                raise ContractError(
                    f"series {code} has {values.shape[0]} samples per channel; minimum is {minimum}"
                )

        direct_mask = _strict_bool_vector(data["direct_mask"], name="direct_mask", count=count)
        direct_fraction = float(direct_mask.mean())
        maximum_direct_fraction = float(config["integrity"]["maximum_direct_fraction"])
        minimum_indirect_fraction = float(config["integrity"]["minimum_indirect_fraction"])
        if direct_fraction > maximum_direct_fraction + 1.0e-12:
            raise ContractError(
                f"candidate is not sparse enough: direct fraction {direct_fraction:.6f} "
                f"exceeds {maximum_direct_fraction:.6f}"
            )
        if 1.0 - direct_fraction < minimum_indirect_fraction - 1.0e-12:
            raise ContractError(
                f"candidate must retain at least {minimum_indirect_fraction:.1%} indirect channels"
            )
        gate_b_components = {
            component["key"]: _strict_bool_vector(
                data[component["key"]], name=component["key"], count=count
            )
            for component in config["gate_b_components"]
        }
        if len(gate_b_components) != 6:
            raise ContractError("the frozen Gate-B contract must contain exactly six metrics")
        expected_gate_b = np.logical_and.reduce(list(gate_b_components.values()))
        full_gate_b_pass = _strict_bool_vector(
            data["full_gate_b_pass"], name="full_gate_b_pass", count=count
        )
        if not np.array_equal(full_gate_b_pass, expected_gate_b):
            mismatch = np.flatnonzero(full_gate_b_pass != expected_gate_b).tolist()
            raise ContractError(
                "full_gate_b_pass must equal the AND of all six Gate-B metric passes; "
                f"mismatched channel indices: {mismatch[:12]}"
            )
        safety_gate_pass = _strict_bool_vector(
            data["safety_gate_pass"], name="safety_gate_pass", count=count
        )
        gate_pass = _strict_bool_vector(data["full_gate_pass"], name="full_gate_pass", count=count)
        expected_full_gate = full_gate_b_pass & safety_gate_pass
        if not np.array_equal(gate_pass, expected_full_gate):
            mismatch = np.flatnonzero(gate_pass != expected_full_gate).tolist()
            raise ContractError(
                "full_gate_pass must equal full_gate_b_pass AND safety_gate_pass; "
                f"mismatched channel indices: {mismatch[:12]}"
            )
        optional_strings = {}
        for name in (
            "evaluation_context",
            "data_role",
            "model_contract_sha256",
            "candidate_artifact_sha256",
        ):
            optional_strings[name] = _decode_scalar(data[name]) if name in keys else ""

    return {
        "subject_id": subject_id,
        "candidate_id": candidate_id,
        "shared_protocol_sha256": shared_protocol_sha256,
        "frozen_evaluator_manifest_sha256": frozen_evaluator_manifest_sha256,
        "outer_evaluation_role": outer_evaluation_role,
        "gate_vector_source_sha256": gate_vector_source_sha256,
        "safety_vector_source_sha256": safety_vector_source_sha256,
        "channels": channels,
        "samples": samples,
        "direct_mask": direct_mask,
        "gate_b_components": gate_b_components,
        "full_gate_b_pass": full_gate_b_pass,
        "safety_gate_pass": safety_gate_pass,
        "gate_pass": gate_pass,
        "optional_strings": optional_strings,
    }


def _load_binding(path: Path, *, input_path: Path, candidate: dict) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"binding manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        binding = json.load(handle)
    required = {
        "binding_schema_version",
        "status",
        "subject_id",
        "candidate_id",
        "candidate_npz_sha256",
        "shared_protocol_sha256",
        "frozen_evaluator_manifest_sha256",
        "outer_evaluation_role",
        "gate_vector_source_sha256",
        "safety_vector_source_sha256",
    }
    missing = sorted(required - set(binding))
    if missing:
        raise ContractError("binding manifest is missing: " + ", ".join(missing))
    if binding["binding_schema_version"] != BINDING_SCHEMA_VERSION:
        raise ContractError(
            f"binding_schema_version must be {BINDING_SCHEMA_VERSION!r}"
        )
    if binding["status"] != "GO":
        raise ContractError("formal preview requires binding manifest status=GO")
    exact_fields = (
        "subject_id",
        "candidate_id",
        "shared_protocol_sha256",
        "frozen_evaluator_manifest_sha256",
        "outer_evaluation_role",
        "gate_vector_source_sha256",
        "safety_vector_source_sha256",
    )
    for field in exact_fields:
        if str(binding[field]) != str(candidate[field]):
            raise ContractError(
                f"binding manifest {field} does not match the candidate NPZ"
            )
    bound_candidate_sha256 = _require_sha256(
        str(binding["candidate_npz_sha256"]), name="candidate_npz_sha256"
    )
    actual_candidate_sha256 = _sha256(input_path)
    if bound_candidate_sha256 != actual_candidate_sha256:
        raise ContractError(
            "binding candidate_npz_sha256 does not match the input candidate artifact"
        )
    return binding


def _page_occupancy(subject_id: str, channel_count: int, config: dict) -> list[int]:
    profile = config["profiles"][subject_id]
    occupancy = [int(value) for value in profile["page_occupancy"]]
    capacity = int(config["layout"]["rows"]) * int(config["layout"]["columns"])
    if sum(occupancy) != channel_count:
        raise ContractError(
            f"page occupancy {occupancy} does not sum to {channel_count} for {subject_id}"
        )
    if any(value < 1 or value > capacity for value in occupancy):
        raise ContractError(f"invalid page occupancy {occupancy} for capacity {capacity}")
    return occupancy


def _shared_grid(candidate: dict, config: dict) -> np.ndarray:
    all_values = np.concatenate([candidate["samples"][code].reshape(-1) for code in SERIES_ORDER])
    kde = config["kde"]
    low, high = np.quantile(
        all_values,
        [float(kde["lower_quantile"]), float(kde["upper_quantile"])],
    )
    span = max(float(high - low), 1.0)
    margin = float(kde["margin_fraction"]) * span
    return np.linspace(
        float(low - margin),
        float(high + margin),
        int(kde["grid_points"]),
    )


def _kde_curve(values: np.ndarray, grid: np.ndarray, *, bandwidth: float) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    # The canonical HUP060 implementation passes h/sample_sd to gaussian_kde,
    # making the resulting 1-D kernel standard deviation exactly h. This
    # closed-form evaluation is equivalent and avoids expensive per-channel
    # scipy object construction for 64/96-channel plates.
    h = float(bandwidth)
    if h <= 0.0:
        raise ContractError("KDE bandwidth must be positive")
    standardized = (grid[:, None] - data[None, :]) / h
    density = np.exp(-0.5 * standardized * standardized).mean(axis=1)
    density /= h * np.sqrt(2.0 * np.pi)
    if not np.all(np.isfinite(density)):
        raise ContractError("KDE produced non-finite density values")
    return density


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig, base: Path, *, stage: str, config: dict) -> list[Path]:
    preview_path = base.with_suffix(".png")
    fig.savefig(
        preview_path,
        dpi=int(config["export"]["preview_dpi"]),
        bbox_inches="tight",
    )
    saved = [preview_path]
    if stage == "final":
        svg_path = base.with_suffix(".svg")
        pdf_path = base.with_suffix(".pdf")
        tiff_path = base.with_suffix(".tiff")
        fig.savefig(svg_path, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(
            tiff_path,
            dpi=int(config["export"]["tiff_dpi"]),
            bbox_inches="tight",
        )
        saved.extend([svg_path, pdf_path, tiff_path])
    plt.close(fig)
    return saved


def _render(candidate: dict, config: dict, output: Path, *, stage: str) -> dict:
    rows = int(config["layout"]["rows"])
    columns = int(config["layout"]["columns"])
    width = float(config["layout"]["width_mm"]) / 25.4
    height = float(config["layout"]["height_mm"]) / 25.4
    grid = _shared_grid(candidate, config)
    bandwidth = float(config["kde"]["absolute_bandwidth"])
    occupancy = _page_occupancy(
        candidate["subject_id"], len(candidate["channels"]), config
    )
    styles = config["style"]
    output.mkdir(parents=True, exist_ok=False)
    figure_dir = output / ("final" if stage == "final" else "preview")
    source_dir = output / "source"
    qa_dir = output / "qa"
    figure_dir.mkdir()
    source_dir.mkdir()
    qa_dir.mkdir()

    density_rows: list[dict] = []
    metadata_rows: list[dict] = []
    page_files: list[Path] = []
    metrics: list[dict] = []
    for channel_index, channel_name in enumerate(candidate["channels"]):
        reference = candidate["samples"]["R"][:, channel_index]
        w1 = {
            code: float(wasserstein_distance(candidate["samples"][code][:, channel_index], reference))
            for code in ("O", "F", "C")
        }
        metric = {
            "w1_observed_to_reference": w1["O"],
            "w1_free_to_reference": w1["F"],
            "w1_controlled_to_reference": w1["C"],
            "delta_w1_controlled_minus_free": w1["C"] - w1["F"],
            "controlled_w1_below_free_w1": bool(w1["C"] < w1["F"]),
        }
        metrics.append(metric)
        component_status = {
            key: bool(values[channel_index])
            for key, values in candidate["gate_b_components"].items()
        }
        metadata_rows.append(
            {
                "subject_id": candidate["subject_id"],
                "candidate_id": candidate["candidate_id"],
                "channel_index": channel_index,
                "channel": channel_name,
                "actuation": "direct" if candidate["direct_mask"][channel_index] else "indirect",
                **component_status,
                "full_gate_b_status": "PASS"
                if candidate["full_gate_b_pass"][channel_index]
                else "FAIL",
                "safety_gate_status": "PASS"
                if candidate["safety_gate_pass"][channel_index]
                else "FAIL",
                "full_gate_status": "PASS" if candidate["gate_pass"][channel_index] else "FAIL",
                "gate_status_source": "input_final_candidate",
                **metric,
            }
        )

    offset = 0
    for page_index, page_count in enumerate(occupancy, start=1):
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(width, height),
            sharex=True,
            squeeze=False,
        )
        flat_axes = axes.reshape(-1)
        last_occupied_row = (page_count - 1) // columns
        for local_index, ax in enumerate(flat_axes):
            if local_index >= page_count:
                ax.set_visible(False)
                continue
            channel_index = offset + local_index
            channel_name = candidate["channels"][channel_index]
            maximum_density = 0.0
            for code in SERIES_ORDER:
                density = _kde_curve(
                    candidate["samples"][code][:, channel_index],
                    grid,
                    bandwidth=bandwidth,
                )
                style = styles[code]
                ax.plot(
                    grid,
                    density,
                    color=style["color"],
                    lw=float(style["linewidth"]),
                    ls=style["linestyle"],
                    alpha=float(style["alpha"]),
                )
                maximum_density = max(maximum_density, float(density.max()))
                density_rows.extend(
                    {
                        "subject_id": candidate["subject_id"],
                        "candidate_id": candidate["candidate_id"],
                        "page": page_index,
                        "channel_index": channel_index,
                        "channel": channel_name,
                        "actuation": "direct" if candidate["direct_mask"][channel_index] else "indirect",
                        "full_gate_status": "PASS" if candidate["gate_pass"][channel_index] else "FAIL",
                        "series_code": code,
                        "series_label": style["label"],
                        "standardized_amplitude": float(x_value),
                        "density": float(y_value),
                    }
                    for x_value, y_value in zip(grid, density)
                )

            is_direct = bool(candidate["direct_mask"][channel_index])
            gate_pass = bool(candidate["gate_pass"][channel_index])
            gate_b_pass = bool(candidate["full_gate_b_pass"][channel_index])
            safety_pass = bool(candidate["safety_gate_pass"][channel_index])
            gate_bits = "".join(
                "1" if candidate["gate_b_components"][component["key"]][channel_index] else "0"
                for component in config["gate_b_components"]
            )
            actuation_code = "D" if is_direct else "I"
            label_color = "#5B7FCA" if is_direct else "#606060"
            gate_color = "#2E8B57" if gate_pass else "#B64342"
            ax.text(
                0.02,
                0.96,
                f"{channel_name} · {actuation_code}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=5.6,
                fontweight="bold",
                color=label_color,
            )
            ax.text(
                0.98,
                0.96,
                f"B:{gate_bits}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=5.0,
                fontweight="bold",
                color="#2E8B57" if gate_b_pass else "#B64342",
            )
            ax.text(
                0.98,
                0.82,
                f"S:{'P' if safety_pass else 'F'} · Full:{'P' if gate_pass else 'F'}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=4.7,
                fontweight="bold",
                color=gate_color,
            )
            metric = metrics[channel_index]
            delta_color = (
                "#2E8B57"
                if metric["controlled_w1_below_free_w1"]
                else "#B64342"
            )
            ax.text(
                0.98,
                0.68,
                "W1 to R: F "
                f"{metric['w1_free_to_reference']:.3f} → C "
                f"{metric['w1_controlled_to_reference']:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=4.8,
                color=delta_color,
            )
            ax.set_ylim(0.0, maximum_density * 1.10)
            ax.set_xlim(float(grid[0]), float(grid[-1]))
            ax.set_yticks([])
            row_index = local_index // columns
            if row_index == last_occupied_row:
                ax.set_xlabel("Amplitude", fontsize=5.6)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            ax.tick_params(axis="x", labelsize=5.2, length=2)

        handles = [
            Line2D(
                [0],
                [0],
                color=styles[code]["color"],
                lw=1.2,
                ls=styles[code]["linestyle"],
                label=f"{code} · {styles[code]['label']}",
            )
            for code in SERIES_ORDER
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.988),
            ncol=4,
            fontsize=6.0,
            handlelength=2.0,
            columnspacing=1.2,
        )
        fig.suptitle(
            f"{candidate['subject_id']}: all-channel O/F/R/C occupation laws "
            f"(candidate preview; page {page_index}/{len(occupancy)})",
            x=0.055,
            y=1.015,
            ha="left",
            fontsize=8.8,
            fontweight="bold",
        )
        fig.text(
            0.055,
            0.965,
            f"candidate={candidate['candidate_id']}  |  "
            f"direct={int(candidate['direct_mask'].sum())}/{len(candidate['channels'])}  |  "
            f"Gate-B pass={int(candidate['full_gate_b_pass'].sum())}/{len(candidate['channels'])}  |  "
            f"full Gate pass={int(candidate['gate_pass'].sum())}/{len(candidate['channels'])}",
            ha="left",
            va="top",
            fontsize=5.7,
            color="#4D4D4D",
        )
        fig.text(
            0.5,
            0.012,
            "B bits: tW1-rel, oW1-rel, tW1-abs, oW1-abs, mean, SD (1=pass); "
            "S=safety; Full=B&S; D/I=direct/indirect. Status supplied, not inferred.",
            ha="center",
            va="bottom",
            fontsize=4.8,
        )
        fig.subplots_adjust(
            left=0.045,
            right=0.995,
            bottom=0.06,
            top=0.925,
            wspace=0.12,
            hspace=0.19,
        )
        base = figure_dir / (
            f"{candidate['subject_id'].lower()}_ofrc_all_channels_page_{page_index:02d}"
        )
        page_files.extend(_save_figure(fig, base, stage=stage, config=config))
        offset += page_count

    _write_csv(
        source_dir / "channel_density_long.csv",
        density_rows,
        [
            "subject_id",
            "candidate_id",
            "page",
            "channel_index",
            "channel",
            "actuation",
            "full_gate_status",
            "series_code",
            "series_label",
            "standardized_amplitude",
            "density",
        ],
    )
    _write_csv(
        source_dir / "channel_metadata.csv",
        metadata_rows,
        [
            "subject_id",
            "candidate_id",
            "channel_index",
            "channel",
            "actuation",
            *[component["key"] for component in config["gate_b_components"]],
            "full_gate_b_status",
            "safety_gate_status",
            "full_gate_status",
            "gate_status_source",
            "w1_observed_to_reference",
            "w1_free_to_reference",
            "w1_controlled_to_reference",
            "delta_w1_controlled_minus_free",
            "controlled_w1_below_free_w1",
        ],
    )
    np.savez_compressed(
        source_dir / "canonical_ofrc_input.npz",
        schema_version=np.asarray(SCHEMA_VERSION),
        shared_protocol_sha256=np.asarray(candidate["shared_protocol_sha256"]),
        frozen_evaluator_manifest_sha256=np.asarray(
            candidate["frozen_evaluator_manifest_sha256"]
        ),
        outer_evaluation_role=np.asarray(candidate["outer_evaluation_role"]),
        gate_vector_source_sha256=np.asarray(candidate["gate_vector_source_sha256"]),
        safety_vector_source_sha256=np.asarray(candidate["safety_vector_source_sha256"]),
        subject_id=np.asarray(candidate["subject_id"]),
        candidate_id=np.asarray(candidate["candidate_id"]),
        channels=candidate["channels"],
        observed_scaled=candidate["samples"]["O"],
        free_scaled=candidate["samples"]["F"],
        reference_scaled=candidate["samples"]["R"],
        controlled_scaled=candidate["samples"]["C"],
        direct_mask=candidate["direct_mask"],
        **candidate["gate_b_components"],
        full_gate_b_pass=candidate["full_gate_b_pass"],
        safety_gate_pass=candidate["safety_gate_pass"],
        full_gate_pass=candidate["gate_pass"],
    )
    with (source_dir / "final_candidate_binding.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(candidate["binding"], handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    expected_density_rows = len(candidate["channels"]) * len(SERIES_ORDER) * len(grid)
    if len(density_rows) != expected_density_rows:
        raise AssertionError(
            f"density row count {len(density_rows)} != expected {expected_density_rows}"
        )
    expected_formats = ["png"] if stage == "preview" else ["png", "svg", "pdf", "tiff"]
    expected_page_files = len(occupancy) * len(expected_formats)
    if len(page_files) != expected_page_files:
        raise AssertionError(
            f"figure count {len(page_files)} != expected {expected_page_files}"
        )
    for path in page_files:
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"missing or empty export: {path}")
    if stage == "preview" and any(
        path.suffix.lower() in {".svg", ".pdf", ".tif", ".tiff"}
        for path in output.rglob("*")
        if path.is_file()
    ):
        raise AssertionError("preview stage emitted a final-format file")

    artifact_paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.parent != qa_dir
    )
    artifact_hashes = {
        path.relative_to(output).as_posix(): _sha256(path) for path in artifact_paths
    }
    qa = {
        "qa_schema_version": "ofrc-renderer-qa-v1",
        "status": "PASS",
        "backend": "Python/matplotlib only",
        "stage": stage,
        "subject_id": candidate["subject_id"],
        "candidate_id": candidate["candidate_id"],
        "input_binding_manifest_sha256": candidate[
            "input_binding_manifest_sha256"
        ],
        "shared_protocol_sha256": candidate["shared_protocol_sha256"],
        "frozen_evaluator_manifest_sha256": candidate[
            "frozen_evaluator_manifest_sha256"
        ],
        "outer_evaluation_role": candidate["outer_evaluation_role"],
        "gate_vector_source_sha256": candidate["gate_vector_source_sha256"],
        "safety_vector_source_sha256": candidate["safety_vector_source_sha256"],
        "channel_count": len(candidate["channels"]),
        "page_occupancy": occupancy,
        "direct_count": int(candidate["direct_mask"].sum()),
        "indirect_count": int((~candidate["direct_mask"]).sum()),
        "gate_b_component_pass_count": {
            component["key"]: int(
                candidate["gate_b_components"][component["key"]].sum()
            )
            for component in config["gate_b_components"]
        },
        "full_gate_b_pass_count": int(candidate["full_gate_b_pass"].sum()),
        "safety_gate_pass_count": int(candidate["safety_gate_pass"].sum()),
        "full_gate_pass_count": int(candidate["gate_pass"].sum()),
        "full_gate_fail_count": int((~candidate["gate_pass"]).sum()),
        "kde_absolute_bandwidth": bandwidth,
        "shared_grid_points": len(grid),
        "shared_grid_min": float(grid[0]),
        "shared_grid_max": float(grid[-1]),
        "density_row_count": len(density_rows),
        "expected_density_row_count": expected_density_rows,
        "page_file_count": len(page_files),
        "expected_page_file_count": expected_page_files,
        "gate_status_source": "input_final_candidate",
        "renderer_infers_gate_or_recovery": False,
        "atomic_staging": True,
        "no_overwrite": True,
        "artifact_sha256": artifact_hashes,
    }
    with (qa_dir / "machine_qa.json").open("w", encoding="utf-8") as handle:
        json.dump(qa, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return qa


def _write_manifest(
    stage_dir: Path,
    *,
    input_path: Path,
    binding_manifest_path: Path,
    config_path: Path,
    qa: dict,
) -> None:
    manifest = {
        "manifest_schema_version": "ofrc-renderer-manifest-v1",
        "input_filename": input_path.name,
        "input_sha256": _sha256(input_path),
        "input_binding_manifest_filename": binding_manifest_path.name,
        "input_binding_manifest_sha256": _sha256(binding_manifest_path),
        "config_filename": config_path.name,
        "config_sha256": _sha256(config_path),
        "subject_id": qa["subject_id"],
        "candidate_id": qa["candidate_id"],
        "shared_protocol_sha256": qa["shared_protocol_sha256"],
        "frozen_evaluator_manifest_sha256": qa[
            "frozen_evaluator_manifest_sha256"
        ],
        "outer_evaluation_role": qa["outer_evaluation_role"],
        "gate_vector_source_sha256": qa["gate_vector_source_sha256"],
        "safety_vector_source_sha256": qa["safety_vector_source_sha256"],
        "stage": qa["stage"],
        "status": qa["status"],
        "interpretation_limit": (
            "Gate status is copied from the input; the renderer does not infer recovery, "
            "validation, or Gate passage from visual overlap."
        ),
    }
    with (stage_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _formal_output_roots() -> tuple[Path, Path]:
    paper_exact_v2 = Path(__file__).resolve().parents[2]
    return (
        (
            paper_exact_v2
            / "HUP065_hup060_sparse_rerun_v1"
            / "preview"
        ).resolve(),
        (
            paper_exact_v2
            / "HUP080_hup060_sparse_rerun_v1"
            / "preview"
        ).resolve(),
    )


def _validate_formal_output_root(output_root: Path) -> None:
    allowed = _formal_output_roots()
    if output_root in allowed:
        return
    lowered_parts = [part.casefold() for part in output_root.parts]
    if output_root.drive.casefold() == "d:":
        reason = "D: destinations are forbidden"
    elif any("overleaf" in part for part in lowered_parts):
        reason = "Overleaf destinations are forbidden"
    elif any(part.startswith("hup060") for part in lowered_parts):
        reason = "HUP060 destinations are forbidden"
    else:
        reason = "destination is outside the two isolated patient preview roots"
    raise ContractError(
        f"unsafe output-root ({reason}); allowed roots are: "
        + "; ".join(str(path) for path in allowed)
    )


def render_atomic(
    *,
    input_path: Path,
    binding_manifest_path: Path,
    output_root: Path,
    run_name: str,
    config_path: Path,
    stage: str,
    _synthetic_selftest: bool = False,
) -> Path:
    if not RUN_NAME_RE.fullmatch(run_name):
        raise ContractError(
            "run-name must start with an alphanumeric character and contain only "
            "letters, digits, dot, underscore, or hyphen"
        )
    config = _load_config(config_path)
    if stage not in {"preview", "final"}:
        raise ContractError("stage must be preview or final")
    if stage == "final" and not bool(config["export"]["final_export_enabled"]):
        raise ContractError(
            "final export is fail-closed in this frozen renderer; only PNG preview is "
            "authorized before explicit user review"
        )
    if stage == "final":
        raise ContractError(
            "final export requires a separately reviewed renderer revision; changing the "
            "configuration alone cannot enable it"
        )

    output_root = output_root.resolve()
    if not _synthetic_selftest:
        _validate_formal_output_root(output_root)
    candidate = _load_candidate(input_path, config)
    binding = _load_binding(
        binding_manifest_path,
        input_path=input_path,
        candidate=candidate,
    )
    candidate["binding"] = binding
    candidate["input_binding_manifest_sha256"] = _sha256(binding_manifest_path)
    output_root.mkdir(parents=True, exist_ok=True)
    _initialize_python_plotting(output_root / "runtime" / "matplotlib")
    target = output_root / run_name
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {target}")
    stage_dir = output_root / f".{run_name}.staging-{uuid.uuid4().hex}"
    if stage_dir.exists():
        raise FileExistsError(f"unexpected staging collision: {stage_dir}")

    try:
        qa = _render(candidate, config, stage_dir, stage=stage)
        _write_manifest(
            stage_dir,
            input_path=input_path,
            binding_manifest_path=binding_manifest_path,
            config_path=config_path,
            qa=qa,
        )
        if target.exists():
            raise FileExistsError(f"refusing to overwrite concurrently created run: {target}")
        os.replace(stage_dir, target)
    except Exception:
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        raise
    return target


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a frozen HUP065/HUP080 all-channel O/F/R/C candidate artifact."
    )
    parser.add_argument("--input", type=Path, required=True, help="Frozen candidate NPZ")
    parser.add_argument(
        "--binding-manifest",
        type=Path,
        required=True,
        help="Hash-binding GO manifest for the frozen candidate NPZ",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
    )
    parser.add_argument("--stage", choices=("preview",), default="preview")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        target = render_atomic(
            input_path=args.input.resolve(),
            binding_manifest_path=args.binding_manifest.resolve(),
            output_root=args.output_root,
            run_name=args.run_name,
            config_path=args.config.resolve(),
            stage=args.stage,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
