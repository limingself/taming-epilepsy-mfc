#!/usr/bin/env python
"""Replot paper Fig. 5 from the frozen matched-input-ablation tables."""

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


SOURCE = ROOT / "output" / "part2" / "source_data" / "figure_05"
OUTPUT = ROOT / "output" / "part2" / "figure_05"
NODE_SPECS = (
    ("RPFa3", "selected SOZ"),
    ("RA3", "selected non-SOZ"),
    ("RAFa4", "unselected"),
)
VARIANTS = (
    ("state_only", "State only", "#9A9A9A", "-."),
    ("state_delay", "State + delay", "#4F9AA3", "--"),
    ("state_delay_graph", "State + delay + graph", "#2F6FAF", "-"),
)


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.14,
        1.07,
        label,
        transform=axis.transAxes,
        fontsize=8.0,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def main() -> int:
    configure_publication_style()
    # The original Fig. 5 script left tick labels at matplotlib's 6.5-pt
    # default; preserve that exact geometry in the frozen-data replot.
    mpl.rcParams.update({"xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
    densities = pd.read_csv(SOURCE / "fixed_context_ablation_density_curves.csv")
    metrics = pd.read_csv(SOURCE / "fixed_context_ablation_metrics.csv")

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(183 / 25.4, 118 / 25.4),
        gridspec_kw={"height_ratios": [1.08, 0.82]},
        constrained_layout=False,
    )
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.14, top=0.70, wspace=0.36, hspace=0.55)

    for column, (channel, node_class) in enumerate(NODE_SPECS):
        axis = axes[0, column]
        observed = densities.loc[
            densities["channel"].eq(channel) & densities["variant"].eq("observed")
        ].sort_values("standardized_amplitude")
        axis.plot(
            observed["standardized_amplitude"],
            observed["density"],
            color="#202020",
            lw=1.35,
            label="Observed actual",
        )
        w1_values: list[float] = []
        for key, label, color, linestyle in VARIANTS:
            curve = densities.loc[
                densities["channel"].eq(channel) & densities["variant"].eq(key)
            ].sort_values("standardized_amplitude")
            axis.plot(
                curve["standardized_amplitude"],
                curve["density"],
                color=color,
                lw=1.05,
                ls=linestyle,
                label=label,
            )
            value = float(
                metrics.loc[
                    metrics["channel"].eq(channel) & metrics["variant"].eq(key),
                    "occupation_w1",
                ].iloc[0]
            )
            w1_values.append(value)
        axis.set_title(f"{channel}: {node_class}", loc="left", fontweight="bold")
        axis.set_xlabel("Standardized amplitude")
        if column == 0:
            axis.set_ylabel("Density")
        panel_label(axis, chr(ord("a") + column))

        axis = axes[1, column]
        x = np.arange(len(VARIANTS), dtype=float)
        axis.bar(
            x,
            w1_values,
            color=[item[2] for item in VARIANTS],
            width=0.62,
            alpha=0.92,
            edgecolor="white",
            linewidth=0.5,
        )
        for index, value in enumerate(w1_values):
            axis.text(index, value + 0.018 * max(w1_values), f"{value:.3f}", ha="center", va="bottom", fontsize=5.7)
        axis.set_xticks(x, ["State", "State\n+ delay", "State + delay\n+ graph"])
        axis.set_ylim(0.0, max(w1_values) * 1.22)
        if column == 0:
            axis.set_ylabel("Occupation $W_1$ (lower is better)")
        axis.set_title(f"{channel}: distribution error", loc="left", fontweight="bold")
        axis.grid(axis="y", color="#E8E8E8", linewidth=0.5)
        panel_label(axis, chr(ord("d") + column))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 0.825), ncol=4, handlelength=2.2, columnspacing=1.3)
    figure.suptitle(
        "HUP060 run-02: protocol-matched Graph–RC input ablation",
        x=0.075,
        y=0.985,
        ha="left",
        fontsize=10.0,
        fontweight="bold",
    )
    figure.text(
        0.075,
        0.915,
        textwrap.fill(
            "All variants use the same run-01 fit data, context-3 initial state, diffusion scale, and reconstructed Part III innovation tensor.",
            width=130,
        ),
        fontsize=6.0,
        color="#555555",
    )
    figure.text(
        0.075,
        0.035,
        textwrap.fill(
            "All predictions are autonomous 256-step RC-SDE rollouts with u=0 and no future-observation refresh. The full graph arm exactly reproduces the Part III zero-control law.",
            width=155,
        ),
        fontsize=5.4,
        color="#555555",
    )
    save_bundle(figure, OUTPUT / "hup060_context3_1s_free_state_delay_graph_ablation")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
