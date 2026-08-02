#!/usr/bin/env python
"""Generate manuscript Fig. 3: all-36-electrode distribution plate."""

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


def make_figure_03() -> None:
    curves = pd.read_csv(SOURCE / "all36_distribution_curves.csv")
    metrics = pd.read_csv(SOURCE / "all36_distribution_metrics.csv").sort_values("channel_index")
    for column in ("selected", "soz", "resection"):
        metrics[column] = metrics[column].astype(str).str.lower().eq("true")

    figure, axes = plt.subplots(
        6,
        6,
        figsize=(183 / 25.4, 181 / 25.4),
        sharex=True,
        constrained_layout=False,
    )
    figure.subplots_adjust(left=0.065, right=0.99, bottom=0.085, top=0.86, wspace=0.26, hspace=0.72)
    for index, (axis, row) in enumerate(zip(axes.flat, metrics.itertuples(), strict=True)):
        values = curves.loc[curves["channel"].eq(row.channel)].sort_values("standardized_amplitude")
        x = values["standardized_amplitude"].to_numpy(float)
        predicted = values["predicted_density"].to_numpy(float)
        observed = values["observed_density"].to_numpy(float)
        axis.fill_between(x, 0.0, predicted, color=COLORS["prediction_fill"], alpha=0.70, linewidth=0)
        axis.plot(x, predicted, color=COLORS["prediction"], lw=0.85)
        axis.plot(x, observed, color=COLORS["observed"], lw=0.85)
        marker = "*" if bool(row.selected) else ""
        axis.set_title(
            f"{row.channel}{marker}  W1={row.occupation_w1:.2f}",
            color=COLORS[str(row.display_class)],
            fontsize=5.5,
            fontweight="bold" if bool(row.selected) else "normal",
            pad=2.0,
        )
        axis.set_xlim(x[0], x[-1])
        axis.set_yticks([])
        if index // 6 == 5:
            axis.set_xlabel("z", fontsize=5.3)
        else:
            axis.tick_params(labelbottom=False)
        axis.tick_params(axis="x", labelsize=5.0, length=2.0)

    handles = [
        mpl.lines.Line2D([], [], color=COLORS["observed"], lw=1.2, label="Observed actual"),
        mpl.lines.Line2D([], [], color=COLORS["prediction"], lw=1.2, label="Free RC-SDE, 32-particle law"),
        mpl.lines.Line2D([], [], color=COLORS["selected SOZ/resection"], lw=0, marker="o", label="selected SOZ/resection"),
        mpl.lines.Line2D([], [], color=COLORS["selected non-SOZ"], lw=0, marker="o", label="selected non-SOZ"),
        mpl.lines.Line2D([], [], color=COLORS["unselected"], lw=0, marker="o", label="unselected"),
    ]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.92), ncol=3, columnspacing=1.2, handlelength=1.8)
    figure.suptitle(
        "HUP060 run-02: distribution prediction across all 36 electrodes",
        x=0.065,
        y=0.985,
        ha="left",
        fontsize=10.0,
        fontweight="bold",
    )
    figure.text(
        0.065,
        0.947,
        "Each panel uses the same fixed context-3 1-s observation and all 32 saved uncontrolled particles; * marks a selected actuator.",
        fontsize=5.9,
        color="#555555",
    )
    figure.text(
        0.065,
        0.025,
        "Black: observed occupation density; gray: predicted occupation density. Panel W1 is lower when the two distributions are closer. Curves share one z-axis scale.",
        fontsize=5.4,
        color="#555555",
    )
    save_bundle(
        figure,
        OUTPUT / "figure_03" / "hup060_context3_all36_distribution_plate_6x6",
    )


def main() -> int:
    configure_publication_style()
    make_figure_03()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
