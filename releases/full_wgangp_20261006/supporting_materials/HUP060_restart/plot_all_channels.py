#!/usr/bin/env python
"""Render auditable four-law HUP060 controller-evaluation plates.

Figure contract: quantitative grid comparing recorded ictal, free prediction,
reference and freshly trained WGAN-GP controlled occupation laws, without
assuming the new controller is better. All panels use one standardized-amplitude
axis; each channel has its own density scale. No curve-specific rescaling,
tail trimming, distance annotation or post-hoc recentering is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.stats import gaussian_kde, wasserstein_distance


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "legend.frameon": False,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
})

BANDWIDTH = 0.16
SERIES = {
    "Recorded ictal": ("observed_scaled", "#272727", "-", 0.80, 0.88),
    "u=0 prediction": ("uncontrolled_scaled", "#D17A22", "--", 0.85, 0.95),
    "Reference": ("reference_validation_scaled", "#3C8D62", "-.", 0.90, 1.00),
    "Fresh WGAN-GP": ("candidate_controlled_scaled", "#2166AC", "-", 1.10, 1.00),
}
ZORDER = {"Recorded ictal": 1, "Fresh WGAN-GP": 2,
          "u=0 prediction": 3, "Reference": 4}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def kde(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    sd = float(np.std(samples, ddof=1))
    if sd < 1e-10:
        # The limiting Gaussian kernel for a constant sample; no noise injection.
        z = (x - float(samples.mean())) / BANDWIDTH
        return np.exp(-0.5 * z * z) / (BANDWIDTH * math.sqrt(2 * math.pi))
    return gaussian_kde(samples, bw_method=BANDWIDTH / sd)(x)


def load_data(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        values = {item[0]: np.asarray(saved[item[0]], dtype=np.float64)
                  for item in SERIES.values()}
        channels = np.asarray(saved["channels"]).astype(str)
        selected = np.asarray(saved["selected_indices"], dtype=int) if "selected_indices" in saved else np.array([], dtype=int)
    if channels.shape != (36,) or len(set(channels)) != 36:
        raise ValueError("Expected 36 unique channel labels.")
    observed = values["observed_scaled"]
    if observed.ndim != 2 or observed.shape[1] != 36:
        raise ValueError("observed_scaled must have shape (time, 36).")
    for key, arr in values.items():
        if not np.isfinite(arr).all():
            raise ValueError(f"Nonfinite data in {key}.")
        if key != "observed_scaled" and (arr.ndim != 3 or arr.shape[1:] != observed.shape):
            raise ValueError(f"{key} must have shape (paths, time, 36).")
    if values["uncontrolled_scaled"].shape != values["candidate_controlled_scaled"].shape:
        raise ValueError("Free and controlled evaluation ensembles must have matching shapes.")
    return values, channels, selected


def time_w1(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.mean([
        wasserstein_distance(candidate[:, t], reference[:, t])
        for t in range(candidate.shape[1])
    ]))


def percent_gain(before: float, after: float) -> float:
    return 100 * (before - after) / before if before > 0 else float("nan")


def metrics(values: dict[str, np.ndarray], channels: np.ndarray, selected: np.ndarray,
            old_paper: Path | None) -> list[dict[str, object]]:
    old_full = None
    if old_paper is not None:
        with np.load(old_paper, allow_pickle=False) as paper:
            if not np.array_equal(np.asarray(paper["channels"]).astype(str), channels):
                raise ValueError("Old paper channel ordering does not match.")
            if not np.array_equal(paper["reference_validation_scaled"], values["reference_validation_scaled"]):
                raise ValueError("Old and new references differ; old-control comparison would be invalid.")
            if not np.array_equal(paper["uncontrolled_scaled"], values["uncontrolled_scaled"]):
                raise ValueError("Old and new free evaluations differ; paired comparison would be invalid.")
            old_full = np.asarray(paper["candidate_controlled_scaled"], dtype=np.float64)
    rows = []
    for c, name in enumerate(channels):
        recorded = values["observed_scaled"][:, c]
        free = values["uncontrolled_scaled"][:, :, c]
        controlled = values["candidate_controlled_scaled"][:, :, c]
        reference = values["reference_validation_scaled"][:, :, c]
        occ_free = float(wasserstein_distance(free.ravel(), reference.ravel()))
        occ_ctrl = float(wasserstein_distance(controlled.ravel(), reference.ravel()))
        occ_recorded = float(wasserstein_distance(recorded, reference.ravel()))
        tw_free = time_w1(free, reference)
        tw_ctrl = time_w1(controlled, reference)
        row = {
            "channel": str(name), "channel_index": c,
            "direct_actuator": bool(c in selected),
            "occupation_w1_recorded_to_reference": occ_recorded,
            "occupation_w1_free_to_reference": occ_free,
            "occupation_w1_fresh_controlled_to_reference": occ_ctrl,
            "occupation_reduction_percent_vs_free": percent_gain(occ_free, occ_ctrl),
            "occupation_reduction_percent_vs_recorded": percent_gain(occ_recorded, occ_ctrl),
            "time_w1_free_to_reference": tw_free,
            "time_w1_fresh_controlled_to_reference": tw_ctrl,
            "time_reduction_percent_vs_free": percent_gain(tw_free, tw_ctrl),
            "recorded_mean": float(recorded.mean()),
            "free_mean": float(free.mean()),
            "fresh_controlled_mean": float(controlled.mean()),
            "reference_mean": float(reference.mean()),
            "free_absolute_mean_bias": float(abs(free.mean() - reference.mean())),
            "fresh_controlled_absolute_mean_bias": float(abs(controlled.mean() - reference.mean())),
            "recorded_sd": float(recorded.std()),
            "free_sd": float(free.std()),
            "fresh_controlled_sd": float(controlled.std()),
            "reference_sd": float(reference.std()),
            "free_absolute_sd_mismatch": float(abs(free.std() - reference.std())),
            "fresh_controlled_absolute_sd_mismatch": float(abs(controlled.std() - reference.std())),
        }
        if old_full is not None:
            old_occ = float(wasserstein_distance(old_full[:, :, c].ravel(), reference.ravel()))
            old_tw = time_w1(old_full[:, :, c], reference)
            row.update({
                "occupation_w1_paper_controlled_to_reference": old_occ,
                "time_w1_paper_controlled_to_reference": old_tw,
                "occupation_reduction_percent_vs_paper": percent_gain(old_occ, occ_ctrl),
                "time_reduction_percent_vs_paper": percent_gain(old_tw, tw_ctrl),
            })
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_plate(densities: dict[str, np.ndarray], channels: np.ndarray, x: np.ndarray,
               indices: np.ndarray, columns: int, out: Path, title: str) -> None:
    nrows = int(math.ceil(len(indices) / columns))
    # The 36-channel overview matches the compact manuscript plate; two 18-channel
    # plates give each channel approximately double the horizontal plotting room.
    fig, axes = plt.subplots(nrows, columns, figsize=(183 / 25.4, 190 / 25.4),
                             sharex=True, squeeze=False)
    for panel, c in enumerate(indices):
        ax = axes.flat[panel]
        ymax = 0.0
        for label, (_, color, linestyle, width, alpha) in SERIES.items():
            curve = densities[label][c]
            ax.plot(x, curve, color=color, ls=linestyle, lw=width, alpha=alpha,
                    zorder=ZORDER[label])
            ymax = max(ymax, float(curve.max()))
        ax.text(0.03, 0.95, str(channels[c]), transform=ax.transAxes, ha="left",
                va="top", fontsize=6.0 if columns == 6 else 7.2,
                fontweight="bold", color="#343434")
        ax.set_xlim(float(x[0]), float(x[-1]))
        ax.set_ylim(0, ymax * 1.12)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.tick_params(axis="x", labelsize=5.5 if columns == 6 else 6.5, length=2)
        if panel // columns != nrows - 1:
            ax.tick_params(axis="x", labelbottom=False)
    for ax in list(axes.flat)[len(indices):]:
        ax.set_visible(False)
    handles = [Line2D([0], [0], color=style[1], ls=style[2], lw=1.2, label=label)
               for label, style in SERIES.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.51, 0.970),
               ncol=4, fontsize=6.3, handlelength=2.1, columnspacing=1.2)
    fig.suptitle(title, y=0.990, fontsize=9.0, fontweight="bold")
    fig.supxlabel("Standardized amplitude", y=0.014, fontsize=8)
    fig.supylabel("Probability density", x=0.007, fontsize=8)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.918, bottom=0.061,
                        wspace=0.14 if columns == 6 else 0.13, hspace=0.21)
    for extension in ("svg", "pdf", "png"):
        fig.savefig(out.with_suffix("." + extension), dpi=400, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper-reference", type=Path, default=None,
                        help="Optional previous paired evaluation for numerical comparison only.")
    args = parser.parse_args()
    values, channels, selected = load_data(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    low = min(float(a.min()) for a in values.values()) - 5 * BANDWIDTH
    high = max(float(a.max()) for a in values.values()) + 5 * BANDWIDTH
    low = min(-4.0, math.floor(low * 2) / 2)
    high = max(4.0, math.ceil(high * 2) / 2)
    grid_count = max(401, int(math.ceil((high - low) / 0.025)) + 1)
    x = np.linspace(low, high, grid_count)
    densities = {label: np.stack([kde(values[style[0]][..., c], x) for c in range(36)])
                 for label, style in SERIES.items()}
    density_rows = [{"channel": str(channels[c]), "series": label,
                     "standardized_amplitude": float(point), "density": float(density)}
                    for c in range(36) for label in SERIES
                    for point, density in zip(x, densities[label][c])]
    write_csv(args.output / "density_source.csv", density_rows)
    rows = metrics(values, channels, selected, args.paper_reference)
    write_csv(args.output / "per_channel_metrics.csv", rows)
    plot_plate(densities, channels, x, np.arange(36), 6,
               args.output / "hup060_fresh_wgangp_all36",
               "HUP060: WGAN-GP control trained from scratch")
    for part, indices in enumerate((np.arange(18), np.arange(18, 36)), 1):
        plot_plate(densities, channels, x, indices, 3,
                   args.output / f"hup060_fresh_wgangp_channels_{part}",
                   f"HUP060: WGAN-GP control trained from scratch ({part}/2)")
    def average(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))
    summary = {
        "input_sha256": sha256(args.input),
        "plot_contract": "Four occupation-law KDEs per channel, common standardized-amplitude x axis; per-channel y axes.",
        "channel_count": 36,
        "free_paths": values["uncontrolled_scaled"].shape[0],
        "controlled_paths": values["candidate_controlled_scaled"].shape[0],
        "reference_paths": values["reference_validation_scaled"].shape[0],
        "time_points": values["observed_scaled"].shape[0],
        "kde_absolute_bandwidth": BANDWIDTH,
        "shared_x_limits": [low, high],
        "kde_grid_points": len(x),
        "tail_handling": "All raw samples plus at least five Gaussian kernel bandwidths are inside the shared axis; no quantile clipping.",
        "curve_processing": "No curve-specific min-max normalization, recentering, smoothing beyond identical absolute KDE bandwidth or manual shape edits.",
        "metric_definition": "Sample empirical 1D Wasserstein distance; occupation pools paths and time. Time-resolved distance averages per-time empirical W1. No KDE-based metric.",
        "uncertainty": "One controller evaluation; ensemble paths are not independent training seeds. No confidence interval is claimed.",
        "mean_occupation_w1_free": average("occupation_w1_free_to_reference"),
        "mean_occupation_w1_fresh": average("occupation_w1_fresh_controlled_to_reference"),
        "mean_time_w1_free": average("time_w1_free_to_reference"),
        "mean_time_w1_fresh": average("time_w1_fresh_controlled_to_reference"),
        "occupation_channels_improved_vs_free": sum(r["occupation_w1_fresh_controlled_to_reference"] < r["occupation_w1_free_to_reference"] for r in rows),
        "time_channels_improved_vs_free": sum(r["time_w1_fresh_controlled_to_reference"] < r["time_w1_free_to_reference"] for r in rows),
        "kde_integral_range": [float(min(np.trapezoid(y, x) for a in densities.values() for y in a)),
                               float(max(np.trapezoid(y, x) for a in densities.values() for y in a))],
    }
    summary["aggregate_occupation_reduction_percent_vs_free"] = percent_gain(summary["mean_occupation_w1_free"], summary["mean_occupation_w1_fresh"])
    summary["aggregate_time_reduction_percent_vs_free"] = percent_gain(summary["mean_time_w1_free"], summary["mean_time_w1_fresh"])
    if args.paper_reference is not None:
        summary["paper_reference_sha256"] = sha256(args.paper_reference)
        summary["mean_occupation_w1_paper"] = average("occupation_w1_paper_controlled_to_reference")
        summary["mean_time_w1_paper"] = average("time_w1_paper_controlled_to_reference")
    with (args.output / "plot_and_metric_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))
    print(args.output / "hup060_fresh_wgangp_all36.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
