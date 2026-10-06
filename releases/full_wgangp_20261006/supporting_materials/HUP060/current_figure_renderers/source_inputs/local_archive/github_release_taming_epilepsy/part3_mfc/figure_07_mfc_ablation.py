#!/usr/bin/env python
"""Uncluttered publication layouts for the HUP060 matched ablations.

The v2 layouts deliberately avoid overlaying near-identical ablation curves.
Every variant occupies its own small-multiple column, while references and
the paired no-control baseline remain identical across columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "artifacts" / "part3_hup060_actor_wgan_v1"

ORDER = ("free", "no_wgan", "no_graph_spread", "no_deviation", "full")
CONTROLLED = ("no_wgan", "no_graph_spread", "no_deviation", "full")
ABLATIONS = ("no_wgan", "no_graph_spread", "no_deviation")
LABELS = {
    "free": "No control",
    "no_wgan": "− WGAN",
    "no_graph_spread": "− Graph spread",
    "no_deviation": "− Deviation",
    "full": "Full",
}
SHORT_LABELS = {
    "free": "No ctrl.",
    "no_wgan": "−WGAN",
    "no_graph_spread": "−Graph",
    "no_deviation": "−Dev.",
    "full": "Full",
}
COLORS = {
    "free": "#8C8C8C",
    "no_wgan": "#D5962B",
    "no_graph_spread": "#8267A8",
    "no_deviation": "#3E9787",
    "full": "#2166AC",
    "reference": "#238B57",
    "observed": "#202020",
}
def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "legend.frameon": False,
        }
    )


def save_bundle(fig: plt.Figure, stem: Path) -> None:
    kwargs = {"bbox_inches": "tight", "pad_inches": 0.06}
    fig.savefig(stem.with_suffix(".svg"), **kwargs)
    fig.savefig(stem.with_suffix(".pdf"), **kwargs)
    fig.savefig(stem.with_suffix(".png"), dpi=360, **kwargs)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, **kwargs)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.17) -> None:
    ax.text(
        x,
        1.055,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=8.8,
        ha="left",
        va="bottom",
    )


def channel_summary(
    ax: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    ylabel: str,
) -> None:
    x = np.arange(len(ORDER), dtype=np.float64)
    offsets = np.linspace(-0.12, 0.12, 36)
    for position, variant in zip(x, ORDER):
        values = frame.loc[frame.variant == variant, metric].to_numpy(float)
        ax.scatter(
            position + offsets,
            values,
            s=7,
            color=COLORS[variant],
            alpha=0.28,
            linewidth=0,
            zorder=1,
        )
        mean = float(np.mean(values))
        ax.plot(
            [position - 0.22, position + 0.22],
            [mean, mean],
            color=COLORS[variant],
            linewidth=2.2,
            solid_capstyle="round",
            zorder=3,
        )
        ax.text(
            position,
            mean + 0.012,
            f"{mean:.3f}",
            color=COLORS[variant],
            ha="center",
            va="bottom",
            fontsize=5.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[item] for item in ORDER])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.5)
    ax.margins(x=0.06)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paper Figure 7 only.")
    parser.add_argument("--source-directory", type=Path, default=PROJECT / "output" / "part3" / "source_data" / "figure_07")
    parser.add_argument("--output-directory", type=Path, default=PROJECT / "output" / "part3" / "figure_07")
    args = parser.parse_args()
    source = args.source_directory
    summary_path = source / "ablation_summary.csv"
    channel_path = source / "ablation_per_channel.csv"
    contrast_path = source / "ablation_contrasts_vs_full.csv"
    rollout_path = source / "ablation_rollouts.npz"
    required = (summary_path, channel_path, contrast_path, rollout_path)
    if not all(path.exists() for path in required):
        raise FileNotFoundError("matched-ablation source data are incomplete")
    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)

    configure_style()
    summary = pd.read_csv(summary_path).set_index("variant")
    channel = pd.read_csv(channel_path)
    contrast = pd.read_csv(contrast_path)
    # ------------------------------------------------------------------
    # Figure 1: uncluttered quantitative summary.
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(7.20, 5.25))
    grid = fig.add_gridspec(
        2,
        3,
        left=0.075,
        right=0.975,
        bottom=0.105,
        top=0.900,
        wspace=0.40,
        hspace=0.49,
    )
    axes = [fig.add_subplot(grid[r, c]) for r in range(2) for c in range(3)]
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes

    channel_summary(ax_a, channel, "time_w1", "Time-resolved $W_1$")
    channel_summary(ax_b, channel, "occupation_w1", "Occupation $W_1$")

    full = summary.loc["full"]
    y = np.arange(len(ABLATIONS), dtype=float)
    time_increase = np.asarray(
        [100.0 * (summary.loc[item, "mean_time_w1"] / full["mean_time_w1"] - 1.0) for item in ABLATIONS]
    )
    occupation_increase = np.asarray(
        [100.0 * (summary.loc[item, "mean_occupation_w1"] / full["mean_occupation_w1"] - 1.0) for item in ABLATIONS]
    )
    for row, left, right in zip(y, time_increase, occupation_increase):
        ax_c.plot([left, right], [row, row], color="#B8B8B8", linewidth=1.2, zorder=1)
    ax_c.scatter(time_increase, y, marker="o", s=26, color="#3D6FA3", label="Time", zorder=3)
    ax_c.scatter(occupation_increase, y, marker="s", s=24, color="#B76542", label="Occupation", zorder=3)
    for values, dy, color in ((time_increase, -0.12, "#3D6FA3"), (occupation_increase, 0.18, "#B76542")):
        for row, value in zip(y, values):
            label = f"{value:.2f}%" if value < 1.0 else f"{value:.1f}%"
            ax_c.text(
                value * 1.08,
                row + dy,
                label,
                color=color,
                fontsize=5.4,
                ha="left",
            )
    ax_c.set_xscale("log")
    ax_c.set_xlim(0.075, 410.0)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([LABELS[item] for item in ABLATIONS])
    ax_c.invert_yaxis()
    ax_c.set_ylim(2.35, -0.35)
    ax_c.set_xlabel("Increase relative to Full (%) — log scale")
    ax_c.grid(axis="x", color="#E6E6E6", linewidth=0.5, which="both")
    ax_c.legend(loc="upper right", fontsize=5.8)

    recovery_columns = (
        "mean_time_w1_recovery_percent",
        "mean_occupation_w1_recovery_percent",
        "unselected_time_w1_recovery_percent",
        "unselected_occupation_w1_recovery_percent",
    )
    recovery = summary.loc[list(CONTROLLED), list(recovery_columns)].to_numpy(float)
    image = ax_d.imshow(recovery, cmap="Blues", vmin=0.0, vmax=75.0, aspect="auto")
    ax_d.set_xticks(np.arange(4))
    ax_d.set_xticklabels(("All\ntime", "All\noccup.", "Unsel.\ntime", "Unsel.\noccup."))
    ax_d.set_yticks(np.arange(4))
    ax_d.set_yticklabels([LABELS[item] for item in CONTROLLED])
    for row in range(4):
        for column in range(4):
            value = recovery[row, column]
            ax_d.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=5.6,
                color="white" if value > 42.0 else "#222222",
            )
    ax_d.set_title("Recovery vs no control (%)", fontsize=7.0, loc="left")
    ax_d.spines[:].set_visible(False)

    label_offsets = {
        "no_wgan": (6, 14),
        "no_graph_spread": (-5, 7),
        "no_deviation": (6, 4),
        "full": (7, -7),
    }
    for variant in CONTROLLED:
        row = summary.loc[variant]
        marker = "*" if variant == "no_wgan" else "o"
        marker_size = 150 if variant == "no_wgan" else (26 if variant == "full" else 30)
        ax_e.scatter(
            row["control_rms"],
            row["mean_time_w1"],
            s=marker_size,
            marker=marker,
            color=COLORS[variant],
            edgecolor="white",
            linewidth=0.8 if variant == "no_wgan" else 0.6,
            zorder=3 if variant == "no_wgan" else 4,
        )
        dx, dy = label_offsets[variant]
        ax_e.annotate(
            LABELS[variant],
            (row["control_rms"], row["mean_time_w1"]),
            xytext=(dx, dy),
            textcoords="offset points",
            color=COLORS[variant],
            fontsize=5.7,
            ha="right" if dx < 0 else "left",
        )
    ax_e.set_xlim(0.18, 0.435)
    ax_e.set_ylim(0.120, 0.305)
    ax_e.set_xlabel("Control RMS")
    ax_e.set_ylabel("Mean time-resolved $W_1$")
    ax_e.grid(color="#E6E6E6", linewidth=0.5)

    counts_time = []
    counts_occupation = []
    for variant in ABLATIONS:
        rows = contrast.loc[contrast.ablation == variant]
        counts_time.append(int(np.sum(rows.delta_time_w1_vs_full > 0.0)))
        counts_occupation.append(int(np.sum(rows.delta_occupation_w1_vs_full > 0.0)))
    x = np.arange(len(ABLATIONS), dtype=float)
    width = 0.34
    ax_f.bar(x - width / 2, counts_time, width, color="#6F92B8", label="Time")
    ax_f.bar(x + width / 2, counts_occupation, width, color="#C98565", label="Occupation")
    for positions, values, lift in (
        (x - width / 2, counts_time, 1.55),
        (x + width / 2, counts_occupation, 0.55),
    ):
        for position, value in zip(positions, values):
            ax_f.text(position, value + lift, f"{value}/36", ha="center", fontsize=5.6)
    ax_f.set_xticks(x)
    ax_f.set_xticklabels([SHORT_LABELS[item] for item in ABLATIONS])
    ax_f.set_ylim(0.0, 39.5)
    ax_f.set_ylabel("Channels worse than Full")
    ax_f.legend(fontsize=5.8, loc="upper left")
    ax_f.grid(axis="y", color="#E6E6E6", linewidth=0.5)

    for label, ax in zip("abcdef", axes):
        add_panel_label(ax, label)
    fig.suptitle(
        "HUP060 run-02: matched component ablation",
        x=0.075,
        y=0.975,
        ha="left",
        fontsize=11.2,
        fontweight="bold",
    )
    summary_stem = output / "hup060_matched_mfc_ablation_summary_v2"
    save_bundle(fig, summary_stem)
    plt.close(fig)
    contract = {
        "core_conclusion": "Deviation feedback is dominant; graph spread gives a moderate marginal-law benefit and WGAN-GP gives a small matched-budget refinement.",
        "archetype": "quantitative grid",
        "backend": "Python/matplotlib",
        "main_result_checkpoint": "seed20261011_adv050_anchor020 (Figures 6 and 8 only)",
        "matched_ablation_checkpoint": "ablation_full_seed20261011_v1 plus three matched retraining arms (Figure 7)",
        "checkpoint_note": "The main-result and matched-ablation full checkpoints share the architecture but are distinct training runs and distinct weights.",
        "source_data": [path.relative_to(PROJECT).as_posix() for path in required],
    }
    (output / "figure_contract_v2.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(summary_stem.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
