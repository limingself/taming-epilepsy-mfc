#!/usr/bin/env python
"""Replot a frozen controller evaluation in the manuscript's original layout.

Contract: a 3x3 quantitative grid links representative trajectories, occupation
laws and inputs; a 6x6 grid shows all-contact occupation laws. Only the fresh
controlled samples and their numerical summaries replace the previous Full
samples. Layout, plotting grids, channel selection, colour definitions and
normalization are preserved. No training, new rollout or curve manipulation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, wasserstein_distance


plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9, "axes.linewidth": .8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
})
LABELS = ("Observed ictal", "Free Graph–RC", "Preictal reference", "WGAN-GP Full")
COLORS = dict(zip(LABELS, ("#272727", "#9A9A9A", "#3C8D62", "#2166AC")))
LS = dict(zip(LABELS, ("-", ":", "-.", "-")))
BANDWIDTH = .16


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def density(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    sd = float(values.std(ddof=1))
    if sd < 1e-10:
        z = (grid - values.mean()) / BANDWIDTH
        return np.exp(-.5 * z * z) / (BANDWIDTH * np.sqrt(2 * np.pi))
    return gaussian_kde(values, bw_method=BANDWIDTH / sd)(grid)


def handles() -> list[Line2D]:
    return [Line2D([0], [0], color=COLORS[s], lw=1.2, ls=LS[s], label=s)
            for s in LABELS]


def panel_label(ax, label: str) -> None:
    ax.text(-.17, 1.08, label, transform=ax.transAxes, fontsize=10,
            fontweight="bold", ha="left", va="bottom")


def load_inputs(args):
    with np.load(args.input, allow_pickle=False) as data:
        arrays = {k: np.asarray(data[k]) for k in data.files}
    with np.load(args.paper, allow_pickle=False) as data:
        paper = {k: np.asarray(data[k]) for k in data.files}
    if not np.array_equal(arrays["channels"], paper["channels"]):
        raise ValueError("Channel ordering differs from the original plate.")
    for key in ("observed_scaled", "uncontrolled_scaled", "reference_validation_scaled", "selected_indices"):
        if not np.array_equal(arrays[key], paper[key]):
            raise ValueError(f"Frozen baseline or reference changed: {key}")
    # Source imports are read-only, including suppression of Python byte caches
    # in the existing model and code directories.
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("frozen_fresh_experiment_reader", args.run_script)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        source = module.source_module()
        setup = module.setup(source)
        adapter, selected = setup[4], setup[3]
        if not np.array_equal(selected, arrays["selected_indices"]):
            raise ValueError("Frozen adapter actuator mask changed.")
        if float(adapter.control_step_scale) != .5:
            raise ValueError("Expected original control-step scale 0.5.")
        effective_map = .5 * adapter.control_channel_map.detach().cpu().numpy()
        model_hash = setup[-1]
    finally:
        sys.dont_write_bytecode = previous
    old_rebuilt = paper["candidate_controls"] @ effective_map
    map_error = float(np.abs(old_rebuilt - paper["candidate_effective_control"]).max())
    if map_error > 1e-12:
        raise ValueError(f"Effective input map does not reproduce original source: {map_error}")
    fresh_effective = arrays["candidate_controls"] @ effective_map
    np_effective_error = None
    if "candidate_effective_control" in arrays:
        np_effective_error = float(np.abs(fresh_effective - arrays["candidate_effective_control"]).max())
        if np_effective_error > 1e-12:
            raise ValueError("New effective input source disagrees with frozen adapter map.")
    return arrays, paper, fresh_effective, effective_map, {
        "effective_map_reproduces_original_maximum_error": map_error,
        "new_effective_input_existing_array_error": np_effective_error,
        "control_step_scale": .5, "model_sha256": model_hash,
        "source_module_sha256": sha(module.SOURCE),
        "run_script_sha256": sha(args.run_script),
        "no_model_rollout_or_training": True,
    }


def export(fig, base: Path, width: int, height: int) -> dict:
    fig.canvas.draw()
    text_sizes = [text.get_fontsize() for text in fig.findobj(matplotlib.text.Text)
                  if text.get_text()]
    for suffix in (".pdf", ".svg", ".png"):
        kwargs = {"metadata": {"CreationDate": None, "ModDate": None}} if suffix == ".pdf" else {}
        fig.savefig(base.with_suffix(suffix), dpi=300, facecolor="white", **kwargs)
    lines = sum(len(ax.lines) for ax in fig.axes)
    plt.close(fig)
    return {"width_mm": width, "height_mm": height, "axis_count": len(fig.axes),
            "line_count": lines, "minimum_native_font_size_pt": min(text_sizes),
            "editable_pdf_and_svg_text": True,
            "pdf_sha256": sha(base.with_suffix(".pdf")),
            "svg_sha256": sha(base.with_suffix(".svg")),
            "png_sha256": sha(base.with_suffix(".png"))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--run-script", type=Path, required=True)
    parser.add_argument("--representative-source", type=Path, required=True,
                        help="Original representative density CSV, for its unchanged grids.")
    parser.add_argument("--all-source", type=Path, required=True,
                        help="Original all-contact density CSV, for its unchanged grid.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arrays, paper, effective, effective_map, input_qa = load_inputs(args)
    args.output.mkdir(parents=True, exist_ok=True)
    channels = arrays["channels"].astype(str)
    selected = set(arrays["selected_indices"].astype(int).tolist())
    representative = paper["representative_indices"].astype(int)
    if len(representative) != 3:
        raise ValueError("Expected three original representative channels.")
    values = dict(zip(LABELS, (arrays["observed_scaled"], arrays["uncontrolled_scaled"],
                              arrays["reference_validation_scaled"], arrays["candidate_controlled_scaled"])))
    free = arrays["uncontrolled_scaled"]
    full = arrays["candidate_controlled_scaled"]
    reference = arrays["reference_validation_scaled"]
    free_w1 = np.array([wasserstein_distance(free[..., c].ravel(), reference[..., c].ravel()) for c in range(36)])
    full_w1 = np.array([wasserstein_distance(full[..., c].ravel(), reference[..., c].ravel()) for c in range(36)])
    if np.max(np.abs(full_w1 - arrays["occupation_w1_candidate"])) > 1e-12:
        raise ValueError("Recomputed W1 differs from the frozen evaluation metric.")
    time = np.arange(256) / float(arrays["sampling_rate_hz"][0])
    old_representative = pd.read_csv(args.representative_source)
    old_all = pd.read_csv(args.all_source)
    representative_rows, density_rows, control_rows, metric_rows = [], [], [], []
    grid_qa = {}

    fig, axes = plt.subplots(3, 3, figsize=(170/25.4, 175/25.4))
    for i, ax in enumerate(axes.flat):
        row, col = divmod(i, 3)
        ax.set_position([.105 + col*.305, (.800, .455, .207)[row] - (.195, .160, .130)[row],
                         .235, (.195, .160, .130)[row]])
        if row != 1:
            ax.set_xticks([0, .5, 1]); ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis="both", labelsize=9)
    for col, c in enumerate(representative):
        channel = channels[c]; role = ("selected SOZ", "selected non-SOZ", "unselected")[col]
        ax = axes[0, col]
        for label in LABELS:
            arr = values[label]
            trajectory = arr[:, c] if arr.ndim == 2 else arr[0, :, c]
            ax.plot(time, trajectory, color=COLORS[label], lw=1.05 if label == "WGAN-GP Full" else .85,
                    alpha=.85 if label == "Observed ictal" else 1)
            representative_rows.extend({"node": str(channel), "role": role, "time_s": float(t),
                                        "series": label, "standardized_amplitude": float(v),
                                        "fixed_particle": 0} for t, v in zip(time, trajectory))
        ax.set_title(f"{channel} · {role}", fontsize=9, fontweight="bold", loc="left")
        ax.set(xlim=(0, 1), xlabel="Time (s)")
        if col == 0: ax.set_ylabel("Standardized amplitude")
        panel_label(ax, "abc"[col])
        frame = old_representative[old_representative.node == channel]
        grid = frame[frame.series == "Observed ictal"].standardized_amplitude.to_numpy()
        if grid.shape != (300,): raise ValueError("Original representative grid must have 300 points.")
        ax = axes[1, col]
        for label in LABELS:
            y = density(values[label][..., c], grid)
            ax.plot(grid, y, color=COLORS[label], lw=1.05 if label == "WGAN-GP Full" else .85, ls=LS[label])
            if label == "Preictal reference": ax.fill_between(grid, y, color=COLORS[label], alpha=.12)
            density_rows.extend({"plot": "representative", "node": str(channel), "role": role,
                                 "standardized_amplitude": float(x), "series": label, "density": float(d)}
                                for x, d in zip(grid, y))
        ax.text(.98, 1.025, f"$W_1$\nFree → Ref. {free_w1[c]:.3f}\nFull → Ref. {full_w1[c]:.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
        ax.set_xlabel("Standardized amplitude"); ax.set_yticks([])
        if col == 0: ax.set_ylabel("Density")
        panel_label(ax, "def"[col])
        ax = axes[2, col]
        direct = np.flatnonzero(arrays["selected_indices"] == c)
        if len(direct):
            control = arrays["candidate_controls"][0, :, int(direct[0])]
            definition = "normalized direct input"
        else:
            control = effective[0, :, c]
            definition = "normalized graph-mediated input"
        ax.plot(time, control, color=COLORS["WGAN-GP Full"], lw=1.05)
        ax.axhline(0, color="#C7C7C7", lw=.6, zorder=0)
        ax.set(xlim=(0, 1), xlabel="Time (s)")
        if col == 0: ax.set_ylabel("Control input")
        panel_label(ax, "ghi"[col])
        control_rows.extend({"node": str(channel), "role": role, "time_s": float(t),
                             "series": "WGAN-GP Full", "input_value": float(v),
                             "input_definition": definition, "fixed_particle": 0}
                            for t, v in zip(time, control))
        grid_qa[str(channel)] = {"density_grid_points": len(grid), "x_min": float(grid.min()), "x_max": float(grid.max())}
    fig.legend(handles=handles(), loc="upper center", bbox_to_anchor=(.54, .940), ncol=2,
               fontsize=9, handlelength=1.8, columnspacing=1.8)
    fig.text(.09, .984, "HUP060 run-02: reference-directed control (WGAN-GP Full)",
             va="top", ha="left", fontsize=10, fontweight="bold")
    rep_export = export(fig, args.output / "chaos_hup060_control_wgangp_full", 170, 175)

    fig, axes = plt.subplots(6, 6, figsize=(170/25.4, 180/25.4), sharex=True)
    clinical = {"RPFa1", "RPFa2", "RPFa3", "RPFb1", "RPFc1"}
    grid = old_all[(old_all.channel == channels[0]) & (old_all.series == "Observed ictal")].standardized_amplitude.to_numpy()
    if grid.shape != (240,): raise ValueError("Original all-contact grid must have 240 points.")
    for c, ax in enumerate(axes.flat):
        row, col = divmod(c, 6)
        ax.set_position([.030+col*.160, (180-29.7-row*24.66-17.1)/180, .145, 17.1/180])
        maximum = 0
        for label in LABELS:
            y = density(values[label][..., c], grid)
            width = .95 if label == "WGAN-GP Full" else .90 if label == "Preictal reference" else .75
            alpha = .78 if label == "Observed ictal" else .82 if label == "Free Graph–RC" else .92 if label == "Preictal reference" else 1
            ax.plot(grid, y, color=COLORS[label], lw=width, ls=LS[label], alpha=alpha)
            maximum = max(maximum, float(y.max()))
            density_rows.extend({"plot": "all36", "node": str(channels[c]), "role": "selected" if c in selected else "unselected",
                                 "standardized_amplitude": float(x), "series": label, "density": float(d)}
                                for x, d in zip(grid, y))
        color = "#B64342" if channels[c] in clinical else "#5B7FCA" if c in selected else "#606060"
        label = str(channels[c]) + ("●" if c in selected else "")
        ax.set_title(label + f"\n{free_w1[c]:.3f} → {full_w1[c]:.3f}", loc="center",
                     fontweight="bold", color=color, fontsize=9, pad=2)
        ax.set_ylim(0, maximum*1.50); ax.set_xlim(float(grid[0]), float(grid[-1])); ax.set_yticks([])
        ax.set_xticks([v for v in (-1, 0, 1) if grid[0] <= v <= grid[-1]])
        ax.tick_params(axis="x", labelbottom=row == 5, labelsize=9, length=2)
        metric_rows.append({"channel": str(channels[c]), "direct_actuator": bool(c in selected),
                            "occupation_w1_free_to_reference": float(free_w1[c]),
                            "occupation_w1_wgangp_full_to_reference": float(full_w1[c]),
                            "reference_mean": float(reference[..., c].mean()),
                            "free_mean": float(free[..., c].mean()), "full_mean": float(full[..., c].mean()),
                            "reference_sd": float(reference[..., c].std()),
                            "free_sd": float(free[..., c].std()), "full_sd": float(full[..., c].std())})
    fig.legend(handles=handles(), loc="upper center", bbox_to_anchor=(.52, .947), ncol=4,
               fontsize=9, columnspacing=.5, handlelength=1.3)
    fig.text(.09, .984, "HUP060: all-contact control (WGAN-GP Full)", va="top", ha="left", fontsize=10, fontweight="bold")
    fig.supxlabel("Standardized amplitude", fontsize=9, y=2.16/180)
    all_export = export(fig, args.output / "chaos_hup060_all36_control_wgangp_full", 170, 180)

    write_csv(args.output / "source_data_representative_trajectories.csv", representative_rows)
    write_csv(args.output / "source_data_control_densities.csv", density_rows)
    write_csv(args.output / "source_data_representative_controls.csv", control_rows)
    write_csv(args.output / "source_data_all36_metrics.csv", metric_rows)
    write_csv(args.output / "source_data_effective_control_map.csv",
              [{"actuator_index": a, "actuator_channel": str(channels[arrays["selected_indices"][a]]),
                "target_channel": str(channels[c]), "step_scaled_map_value": float(effective_map[a, c])}
               for a in range(13) for c in range(36)])
    cropped = {label: {"sample_fraction_outside_original_all36_axis": float(np.mean((arr < grid[0]) | (arr > grid[-1]))) }
               for label, arr in values.items()}
    qa = {
        "backend": "Python/matplotlib", "archetype": "quantitative grid",
        "fresh_evaluation_sha256": sha(args.input), "original_evaluation_sha256": sha(args.paper),
        "original_representative_density_source_sha256": sha(args.representative_source),
        "original_all36_density_source_sha256": sha(args.all_source),
        "original_representative_indices": representative.tolist(),
        "original_representative_channels": channels[representative].tolist(),
        "selected_direct_actuator_count": len(selected),
        "figure_dimensions_mm": {"representative": [170, 175], "all36": [170, 180]},
        "four_curve_colours": COLORS, "four_curve_line_styles": LS,
        "kde_absolute_bandwidth": BANDWIDTH, "no_new_normalization": True,
        "metric_definition": "Sample empirical occupation W1, pooling all particles and times; not KDE-integral distance.",
        "all36_mean_free_occupation_w1": float(free_w1.mean()),
        "all36_mean_wgangp_full_occupation_w1": float(full_w1.mean()),
        "occupation_channels_improved_vs_free": int(np.sum(full_w1 < free_w1)),
        "representative_density_grids": grid_qa,
        "original_all36_density_grid": {"points": len(grid), "min": float(grid[0]), "max": float(grid[-1])},
        "view_window_policy": "Original figure x-grid retained exactly; W1 computed from all samples including axis-outside tails.",
        "axis_outside_sample_fractions": cropped,
        "input_definition": "Selected contacts: direct normalized actuator input. Unselected contact: graph-mediated step-scaled effective input.",
        "map_qa": input_qa, "representative_export": rep_export, "all36_export": all_export,
        "overleaf_modified": False, "historical_code_or_figures_modified": False,
    }
    with (args.output / "original_style_control_numeric_qa.json").open("w", encoding="utf-8") as handle:
        json.dump(qa, handle, ensure_ascii=False, indent=2)
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
