#!/usr/bin/env python
"""Publication figure for the reviewer-requested trivial-baseline audit."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from figure_repro_utils import configure_publication_style, save_bundle, write_json


SOURCE = (
    PROJECT
    / "output"
    / "part3"
    / "source_data"
    / "figure_09_trivial_baselines"
)
OUTPUT = PROJECT / "output" / "part3" / "figure_09"
STEM = OUTPUT / "hup060_trivial_controller_baselines"

COLORS = {
    "time": "#315A7D",
    "occupation": "#D18B47",
    "free": "#7A7A7A",
    "full": "#2B6F9E",
    "damping": "#C65D3B",
    "diffusion": "#7A5195",
    "observed": "#1F1F1F",
    "location": "#8DB3D3",
    "scale": "#D8A15D",
    "shape": "#B8B8B8",
}


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.14,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="top",
    )


def main() -> int:
    configure_publication_style()
    summary = pd.read_csv(SOURCE / "trivial_baseline_summary.csv")
    damping = pd.read_csv(SOURCE / "linear_damping_gain_screen.csv").sort_values(
        "control_rms"
    )
    diffusion = pd.read_csv(SOURCE / "diffusion_scale_sensitivity.csv")
    moment = pd.read_csv(SOURCE / "location_scale_moment_audit.csv")
    experiment = json.loads((SOURCE / "experiment_summary.json").read_text("utf-8"))

    rows = summary.set_index("variant")
    order = [
        "free",
        "diffusion_reference_oracle",
        "constant_input",
        "white_input",
        "linear_damping_rms_matched",
        "full_actor_wgan",
    ]
    labels = [
        "Free",
        r"Diff. $\kappa=0.2$*",
        "Constant",
        "White",
        "P damping",
        "Full",
    ]

    fig = plt.figure(figsize=(7.2, 5.75), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], wspace=0.25, hspace=0.25)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    sub_d = grid[1, 1].subgridspec(2, 1, height_ratios=[0.95, 1.25], hspace=0.28)
    ax_d1 = fig.add_subplot(sub_d[0, 0])
    ax_d2 = fig.add_subplot(sub_d[1, 0])

    # a | Primary endpoint comparison.
    x = np.arange(len(order), dtype=np.float64)
    width = 0.36
    time_values = np.asarray([rows.loc[key, "mean_time_w1"] for key in order])
    occ_values = np.asarray([rows.loc[key, "mean_occupation_w1"] for key in order])
    ax_a.bar(
        x - width / 2,
        time_values,
        width,
        color=COLORS["time"],
        label=r"Time-resolved $W_1$",
        zorder=2,
    )
    ax_a.bar(
        x + width / 2,
        occ_values,
        width,
        color=COLORS["occupation"],
        label=r"Occupation $W_1$",
        zorder=2,
    )
    for index, key in enumerate(order):
        if key not in {"constant_input", "white_input"}:
            continue
        row = rows.loc[key]
        for offset, metric, color in [
            (-width / 2, "mean_time_w1", COLORS["time"]),
            (width / 2, "mean_occupation_w1", COLORS["occupation"]),
        ]:
            low = float(row[f"{metric}_q025"])
            high = float(row[f"{metric}_q975"])
            center = float(row[metric])
            ax_a.errorbar(
                index + offset,
                center,
                yerr=[[center - low], [high - center]],
                fmt="none",
                ecolor="#202020",
                elinewidth=0.7,
                capsize=1.8,
                zorder=4,
            )
    ax_a.set_xticks(x, labels, rotation=26, ha="right")
    ax_a.get_xticklabels()[-1].set_color(COLORS["full"])
    ax_a.get_xticklabels()[-1].set_fontweight("bold")
    ax_a.set_ylabel(r"Mean channel-wise $W_1$")
    ax_a.set_ylim(0, max(1.02, float(np.max([time_values, occ_values])) * 1.12))
    ax_a.grid(axis="y", color="#E7E7E7", linewidth=0.55, zorder=0)
    ax_a.legend(loc="upper left", ncol=2, handlelength=1.2, columnspacing=0.9)
    ax_a.text(
        0.01,
        0.02,
        "*Post-hoc plant-misspecification sensitivity; not a control arm",
        transform=ax_a.transAxes,
        fontsize=5.2,
        color=COLORS["diffusion"],
        va="bottom",
    )
    panel_label(ax_a, "a")

    # b | Full K--RMS curve rather than winner-only reporting.
    positive = damping.loc[damping["mean_occupation_w1"] > 0].copy()
    ax_b.plot(
        positive["control_rms"],
        positive["mean_occupation_w1"],
        color="#A75C48",
        linewidth=1.1,
        marker="o",
        markersize=2.6,
        markeredgewidth=0,
        label=r"Scalar $u=-Kx_{\mathcal{A}}$",
    )
    full_row = rows.loc["full_actor_wgan"]
    damping_row = rows.loc["linear_damping_rms_matched"]
    ax_b.scatter(
        [full_row["control_rms"]],
        [full_row["mean_occupation_w1"]],
        s=38,
        color=COLORS["full"],
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
        label="Full Actor--WGAN",
    )
    ax_b.scatter(
        [damping_row["control_rms"]],
        [damping_row["mean_occupation_w1"]],
        s=32,
        color=COLORS["damping"],
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
        label="RMS-matched damping",
    )
    ax_b.axvline(
        float(full_row["control_rms"]),
        color="#606060",
        linestyle="--",
        linewidth=0.75,
    )
    ax_b.set_yscale("log")
    ax_b.set_xlabel("Actuator-space control RMS")
    ax_b.set_ylabel(r"Mean occupation $W_1$ (log scale)")
    ax_b.grid(axis="y", which="both", color="#E7E7E7", linewidth=0.5)
    best_low_energy = positive.loc[positive["mean_occupation_w1"].idxmin()]
    ax_b.annotate(
        rf"lowest P: $K={best_low_energy['gain']:.2f}$" + "\n"
        rf"RMS={best_low_energy['control_rms']:.3f}",
        xy=(best_low_energy["control_rms"], best_low_energy["mean_occupation_w1"]),
        xytext=(0.04, 0.47),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.6},
        fontsize=5.5,
    )
    ax_b.legend(loc="upper left", handlelength=1.3)
    panel_label(ax_b, "b")

    # c | Diffusion shrinkage must be judged against prediction fidelity.
    ax_c.plot(
        diffusion["kappa_sigma"],
        diffusion["mean_occupation_w1"],
        color=COLORS["diffusion"],
        marker="o",
        markersize=3,
        linewidth=1.2,
        label=r"To interictal reference",
    )
    ax_c.plot(
        diffusion["kappa_sigma"],
        diffusion["mean_occupation_w1_to_observed_ictal"],
        color=COLORS["observed"],
        marker="s",
        markersize=2.8,
        linewidth=1.1,
        label="To observed ictal record",
    )
    ax_c.axvline(1.0, color="#606060", linestyle="--", linewidth=0.75)
    oracle_kappa = float(
        experiment["diffusion_sensitivity"]["posthoc_reference_oracle_kappa"]
    )
    ax_c.axvline(
        oracle_kappa, color=COLORS["diffusion"], linestyle=":", linewidth=0.8
    )
    ax_c.annotate(
        r"prediction-calibrated $\kappa=1$",
        xy=(1.0, float(diffusion.loc[diffusion["kappa_sigma"] == 1.0, "mean_occupation_w1_to_observed_ictal"].iloc[0])),
        xytext=(1.07, 0.19),
        textcoords="data",
        arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 0.6},
        fontsize=5.4,
        color="#4A4A4A",
    )
    ax_c.annotate(
        r"reference-oracle $\kappa=0.2$*",
        xy=(oracle_kappa, float(diffusion.loc[diffusion["kappa_sigma"] == oracle_kappa, "mean_occupation_w1"].iloc[0])),
        xytext=(0.34, 0.058),
        textcoords="data",
        arrowprops={"arrowstyle": "-", "color": COLORS["diffusion"], "lw": 0.6},
        fontsize=5.4,
        color=COLORS["diffusion"],
    )
    ax_c.set_xlabel(r"Relative diffusion multiplier $\kappa_\sigma$")
    ax_c.set_ylabel(r"Mean occupation $W_1$")
    ax_c.set_xlim(-0.03, 1.53)
    ax_c.grid(axis="y", color="#E7E7E7", linewidth=0.55)
    ax_c.legend(loc="upper center", bbox_to_anchor=(0.60, 1.0))
    panel_label(ax_c, "c")

    # d | Descriptive location--scale Shapley bookkeeping and raw SD ratios.
    location = float(moment["shapley_location_contribution"].mean())
    scale = float(moment["shapley_scale_contribution"].mean())
    shape = float(moment["shape_residual_contribution"].mean())
    total = location + scale + shape
    left = 0.0
    for value, color, label in [
        (location, COLORS["location"], "Location"),
        (scale, COLORS["scale"], "Scale"),
        (shape, COLORS["shape"], "Residual shape"),
    ]:
        ax_d1.barh([0], [value], left=left, color=color, height=0.46, label=label)
        left += value
    ax_d1.set_xlim(0, total * 1.02)
    ax_d1.set_yticks([])
    ax_d1.set_xlabel(r"Contribution to Free$\rightarrow$reference occupation $W_1$")
    ax_d1.legend(loc="upper center", ncol=3, columnspacing=0.8, handlelength=1.2)
    ax_d1.text(
        location + scale / 2,
        0,
        f"Scale\n{100.0 * scale / total:.1f}%",
        ha="center",
        va="center",
        fontsize=5.8,
        color="#3B2B1C",
        fontweight="bold",
    )
    panel_label(ax_d1, "d")

    ratio_obs = moment["reference_to_observed_sd_ratio"].to_numpy()
    ratio_free = moment["reference_to_free_sd_ratio"].to_numpy()
    rng = np.random.default_rng(20261109)
    for index, (values, color) in enumerate(
        [(ratio_obs, COLORS["observed"]), (ratio_free, COLORS["free"])]
    ):
        jitter = rng.uniform(-0.08, 0.08, size=values.size)
        ax_d2.scatter(
            index + jitter,
            values,
            s=9,
            color=color,
            alpha=0.52,
            edgecolor="none",
            zorder=2,
        )
        quartiles = np.quantile(values, [0.25, 0.5, 0.75])
        ax_d2.plot([index - 0.16, index + 0.16], [quartiles[1]] * 2, color=color, lw=1.5)
        ax_d2.plot([index, index], [quartiles[0], quartiles[2]], color=color, lw=2.4)
    ax_d2.axhline(1.0, color="#777777", linestyle="--", linewidth=0.7)
    ax_d2.set_xticks([0, 1], ["Reference /\nobserved ictal", "Reference /\nfree model"])
    ax_d2.set_ylabel("Channel-wise SD ratio")
    ax_d2.set_xlim(-0.45, 1.45)
    ax_d2.set_ylim(0, max(1.1, float(max(ratio_obs.max(), ratio_free.max())) * 1.08))
    ax_d2.grid(axis="y", color="#E7E7E7", linewidth=0.55)
    ax_d2.text(
        0.98,
        0.96,
        "n=36 channels\n(descriptive technical units)\n"
        f"median |SD$_{{Full}}$-SD$_{{ref}}$|={moment['absolute_full_reference_sd_difference'].median():.3f}",
        transform=ax_d2.transAxes,
        ha="right",
        va="top",
        fontsize=5.3,
    )

    full_occ = float(rows.loc["full_actor_wgan", "mean_occupation_w1"])
    damping_occ = float(rows.loc["linear_damping_rms_matched", "mean_occupation_w1"])
    improvement = 100.0 * (damping_occ - full_occ) / damping_occ
    fig.suptitle(
        "RMS-matched generic inputs do not reproduce structured control; "
        f"Full lowers occupation $W_1$ by {improvement:.1f}% versus matched damping",
        fontsize=8.2,
        fontweight="bold",
    )

    save_bundle(fig, STEM, png_dpi=350)
    contract = {
        "core_conclusion": (
            "At matched actuator-space RMS, scalar damping and generic open-loop "
            "inputs do not reproduce Full Actor-WGAN control, whereas post-hoc "
            "diffusion shrinkage can lower reference W1 only by sacrificing ictal "
            "prediction fidelity."
        ),
        "figure_archetype": "quantitative grid",
        "backend": "Python/matplotlib only",
        "target_output": "editable SVG/PDF plus PNG preview and 600-dpi TIFF",
        "final_size_inches": [7.2, 5.75],
        "panel_map": {
            "a": "primary time-resolved and occupation W1 comparison",
            "b": "complete proportional gain versus RMS curve",
            "c": "diffusion-reference benefit versus observed-ictal fidelity",
            "d": "location-scale Shapley bookkeeping and channel-wise SD ratios",
        },
        "statistics": (
            "36 contacts and 32 random-input seeds are technical units; no naive "
            "patient-level p-values are reported"
        ),
        "reviewer_risks": [
            "post-hoc kappa=0.2 must never be called a control arm",
            "the RMS-matched damping row is selected without reference-W1 access",
            "location-scale contributions are descriptive and non-causal",
        ],
    }
    write_json(OUTPUT / "figure_contract.json", contract)
    print(json.dumps({"status": "ok", "stem": str(STEM)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
