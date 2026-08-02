#!/usr/bin/env python
"""Generate manuscript Fig. 2: representative RC-SDE prediction audit."""

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


def make_figure_02() -> None:
    trajectories = pd.read_csv(SOURCE / "representative_trajectory_and_error.csv")
    densities = pd.read_csv(SOURCE / "representative_distribution_curves.csv")
    metrics = pd.read_csv(SOURCE / "representative_metrics.csv").set_index("channel")

    figure, axes = plt.subplots(
        3,
        3,
        figsize=(183 / 25.4, 162 / 25.4),
        constrained_layout=False,
        gridspec_kw={"height_ratios": [1.0, 0.92, 0.78]},
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.115,
        top=0.80,
        wspace=0.36,
        hspace=0.66,
    )

    for column, (channel, node_class) in enumerate(NODE_SPECS):
        trace = trajectories.loc[trajectories["channel"].eq(channel)].sort_values("time_s")
        density = densities.loc[densities["channel"].eq(channel)].sort_values(
            "standardized_amplitude"
        )
        metric = metrics.loc[channel]

        axis = axes[0, column]
        axis.plot(trace["time_s"], trace["observed"], color=COLORS["observed"], lw=1.15, label="Observed actual")
        axis.plot(trace["time_s"], trace["predicted"], color=COLORS["prediction"], lw=1.10, label="Free RC-SDE, particle 0")
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Time (s)")
        if column == 0:
            axis.set_ylabel("Standardized amplitude")
        axis.set_title(f"{channel}: {node_class}", loc="left", fontweight="bold")
        axis.text(
            0.98,
            0.04,
            f"nRMSE={metric['nrmse']:.3f}\nr={metric['correlation']:.3f}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
        panel_label(axis, chr(ord("a") + column))

        axis = axes[1, column]
        x = density["standardized_amplitude"].to_numpy(float)
        predicted = density["predicted_density"].to_numpy(float)
        observed = density["observed_density"].to_numpy(float)
        axis.fill_between(x, 0.0, predicted, color=COLORS["prediction_fill"], alpha=0.75, linewidth=0)
        axis.plot(x, predicted, color=COLORS["prediction"], lw=1.15, label="Free RC-SDE, 32 particles")
        axis.plot(x, observed, color=COLORS["observed"], lw=1.15, label="Observed actual")
        axis.set_xlabel("Standardized amplitude")
        if column == 0:
            axis.set_ylabel("Density")
        axis.set_title(f"{channel}: same-window distribution", loc="left", fontweight="bold")
        axis.text(
            0.98,
            0.05,
            f"distribution $W_1$={metric['occupation_w1']:.3f}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
        panel_label(axis, chr(ord("d") + column))

        axis = axes[2, column]
        absolute = trace["absolute_error"].to_numpy(float)
        time = trace["time_s"].to_numpy(float)
        axis.fill_between(time, 0.0, absolute, color=COLORS["error_fill"], alpha=0.75, linewidth=0)
        axis.plot(time, absolute, color=COLORS["error"], lw=1.0, label="Instantaneous absolute error")
        axis.axhline(metric["mae"], color="#7A4C27", lw=0.8, ls="--", label="Mean absolute error")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(bottom=0.0)
        axis.set_xlabel("Time (s)")
        if column == 0:
            axis.set_ylabel("Absolute error")
        axis.set_title(f"{channel}: particle-0 trajectory error", loc="left", fontweight="bold")
        axis.text(
            0.98,
            0.94,
            f"MAE={metric['mae']:.3f}\nRMSE={metric['rmse']:.3f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=5.2,
        )
        panel_label(axis, chr(ord("g") + column))

    unique: dict[str, object] = {}
    for row in range(3):
        handles, labels = axes[row, 0].get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            unique.setdefault(label, handle)
    figure.legend(
        list(unique.values()),
        list(unique.keys()),
        loc="upper center",
        bbox_to_anchor=(0.54, 0.895),
        ncol=3,
        columnspacing=1.3,
        handlelength=2.2,
    )
    figure.suptitle(
        "HUP060 run-02: fixed 1-s free RC-SDE prediction, distribution, and error",
        x=0.075,
        y=0.975,
        ha="left",
        fontsize=10.0,
        fontweight="bold",
    )
    figure.text(
        0.075,
        0.925,
        textwrap.fill(
            "Rows 1 and 3 use the predeclared context 3 and fixed stochastic particle 0. "
            "Row 2 uses all 32 saved particles from exactly the same 1-s context.",
            width=125,
        ),
        fontsize=6.1,
        color="#555555",
    )
    figure.text(
        0.075,
        0.025,
        textwrap.fill(
            "The orange curve is |prediction - observation| at each time point; its dashed line is MAE. "
            "The fixed particle is illustrative and was not chosen by accuracy. Distribution W1 uses the full 32-particle law. "
            "Particles are technical stochastic realizations, not biological replicates.",
            width=158,
        ),
        fontsize=5.4,
        color="#555555",
    )
    save_bundle(
        figure,
        OUTPUT / "figure_02" / "hup060_fixed_context3_particle0_prediction_distribution_error_3x3",
    )


def main() -> int:
    configure_publication_style()
    make_figure_02()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
