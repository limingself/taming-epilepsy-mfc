#!/usr/bin/env python
"""Generate manuscript Fig. 4: all-electrode distribution-error audit."""

from __future__ import annotations

from pathlib import Path
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure_repro_utils import configure_publication_style, save_bundle


SOURCE = ROOT / "output" / "part2" / "source_data" / "figures_02_04"
OUTPUT = ROOT / "output" / "part2"
NODE_SPECS = (
    ("RPFa3", "selected SOZ"),
    ("RA3", "selected non-SOZ"),
    ("RAFa4", "unselected"),
)
COLORS = {
    "observed": "#202020",
    "prediction": "#747474",
    "prediction_fill": "#D9DCE1",
    "error": "#C47A3A",
    "error_fill": "#F0D8C4",
    "selected SOZ/resection": "#B5443C",
    "selected non-SOZ": "#2F6EA3",
    "unselected": "#858585",
}


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.15,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=8.0,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def make_figure_04() -> None:
    metrics = pd.read_csv(SOURCE / "all36_distribution_metrics.csv")
    ordered = metrics.sort_values("occupation_w1", ascending=True).reset_index(drop=True)
    colors = [COLORS[value] for value in ordered["display_class"]]
    y = np.arange(len(ordered), dtype=float)
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(183 / 25.4, 145 / 25.4),
        gridspec_kw={"width_ratios": [1.45, 0.90, 0.90]},
        sharey=True,
        constrained_layout=False,
    )
    figure.subplots_adjust(left=0.12, right=0.985, bottom=0.13, top=0.82, wspace=0.22)
    specifications = (
        ("occupation_w1", "Occupation $W_1$", "Distribution error", "a"),
        ("absolute_mean_error", "$|\\Delta$ mean$|$", "Location error", "b"),
        ("absolute_sd_error", "$|\\Delta$ SD$|$", "Spread error", "c"),
    )
    for axis_index, (axis, (metric, xlabel, title, label)) in enumerate(zip(axes, specifications, strict=True)):
        values = ordered[metric].to_numpy(float)
        axis.hlines(y, 0.0, values, color="#D8D8D8", lw=0.75, zorder=1)
        axis.scatter(values, y, c=colors, s=18, edgecolor="white", linewidth=0.35, zorder=2)
        median = float(np.median(values))
        axis.axvline(median, color="#3F3F3F", lw=0.8, ls="--")
        axis.text(0.98, 0.025, f"median={median:.3f}", transform=axis.transAxes, ha="right", va="bottom", fontsize=5.3, color="#444444")
        axis.set_xlabel(f"{xlabel} (lower is better)")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="x", color="#E8E8E8", linewidth=0.5)
        panel_label(axis, label)
        if axis_index == 0:
            axis.set_yticks(y, ordered["channel"].tolist())
            axis.set_ylabel("Electrode")
        else:
            axis.tick_params(labelleft=False)
    axes[0].set_ylim(-0.8, len(ordered) - 0.2)
    handles = [
        mpl.lines.Line2D([], [], color=COLORS[name], marker="o", lw=0, label=name)
        for name in ("selected SOZ/resection", "selected non-SOZ", "unselected")
    ]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.58, 0.885), ncol=3, columnspacing=1.3)
    figure.suptitle(
        "HUP060 run-02: unified zero-control distribution error",
        x=0.12,
        y=0.975,
        ha="left",
        fontsize=10.0,
        fontweight="bold",
    )
    figure.text(
        0.12,
        0.92,
        "The same 36 electrodes are aligned across panels. Small mean error but larger SD error indicates that most W1 mismatch comes from distribution width.",
        fontsize=5.9,
        color="#555555",
    )
    figure.text(
        0.12,
        0.035,
        "Rows are ordered by W1 in panel a. Dashed lines show all-electrode medians. Electrode summaries are descriptive within one patient; no independent-sample inference is made.",
        fontsize=5.4,
        color="#555555",
    )
    save_bundle(
        figure,
        OUTPUT / "figure_04" / "hup060_context3_all36_distribution_w1_summary",
    )


def main() -> int:
    configure_publication_style()
    make_figure_04()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
