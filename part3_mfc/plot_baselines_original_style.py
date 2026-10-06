#!/usr/bin/env python
"""Original-layout input audit using frozen, exact-paired-noise source data.

Five axes retain the manuscript arrangement. Panel a uses bare point-estimate
bars with an adaptive axis, not mixed uncertainty intervals. Its random-input
arms summarize 32 technical seeds by medians, not independent patients. The
other panels retain the full gain screen, diffusion sensitivity, descriptive
location/scale accounting and channel SD ratios. No model evaluation or tuning.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "axes.linewidth": .7, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    "svg.hashsalt": "hup060-distribution-control-2026", "savefig.facecolor": "white",
})
COLORS = {"time": "#315A7D", "occupation": "#D18B47", "free": "#7A7A7A",
          "full": "#2B6F9E", "damping": "#C65D3B", "diffusion": "#7A5195",
          "observed": "#1F1F1F", "location": "#8DB3D3", "scale": "#D8A15D", "shape": "#B8B8B8"}
ORDER = ["free", "diffusion_reference_oracle", "constant_input", "white_input",
         "linear_damping_rms_matched", "full_actor_wgan"]
POSITIONS = [[.115, .580, .350, .290], [.605, .580, .350, .290],
             [.115, .105, .350, .290], [.605, .360, .350, .110], [.605, .105, .350, .135]]


def sha(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def panel_label(axis, label):
    axis.text(-.17, 1.08, label, transform=axis.transAxes, fontsize=10,
              fontweight="bold", ha="left", va="bottom")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = {name: args.source/name for name in ["experiment_summary.json", "trivial_baseline_summary.csv",
             "linear_damping_gain_screen.csv", "diffusion_scale_sensitivity.csv",
             "location_scale_moment_audit.csv", "random_input_seed_metrics.csv"]}
    if not files["experiment_summary.json"].is_file():
        raise FileNotFoundError("Exact-paired-noise experiment must finish before plotting.")
    experiment = json.loads(files["experiment_summary.json"].read_text("utf-8"))
    noise_error = float(experiment["paired_noise_reconstruction"]["maximum_absolute_error"])
    if noise_error >= 1e-12:
        raise ValueError("Source does not meet the exact paired-noise contract.")
    summary = pd.read_csv(files["trivial_baseline_summary.csv"])
    damping = pd.read_csv(files["linear_damping_gain_screen.csv"]).sort_values("control_rms")
    diffusion = pd.read_csv(files["diffusion_scale_sensitivity.csv"])
    moment = pd.read_csv(files["location_scale_moment_audit.csv"])
    seeds = pd.read_csv(files["random_input_seed_metrics.csv"])
    rows = summary.set_index("variant")
    full = rows.loc["full_actor_wgan"]; matched = rows.loc["linear_damping_rms_matched"]
    expected_rms = float(experiment["full_raw_control_rms_target"])
    if abs(float(full.control_rms)-expected_rms) > 1e-12:
        raise ValueError("Full RMS is not bound to this frozen experiment.")
    seed_qa = {}
    for arm in ("constant_input", "white_input"):
        arm_seeds = seeds[seeds.variant == arm]
        if len(arm_seeds) != 32:
            raise ValueError("Expected 32 predeclared command seeds per random-input arm.")
        for metric in ("mean_time_w1", "mean_occupation_w1", "control_rms"):
            err = abs(float(arm_seeds[metric].median()) - float(rows.loc[arm, metric]))
            if err > 1e-12: raise ValueError(f"Random-input median source mismatch: {arm}/{metric}")
        seed_qa[arm] = {"n_technical_seeds": 32,
                        "command_rms_min": float(arm_seeds.control_rms.min()),
                        "command_rms_max": float(arm_seeds.control_rms.max()),
                        "median_time_w1": float(arm_seeds.mean_time_w1.median()),
                        "median_occupation_w1": float(arm_seeds.mean_occupation_w1.median()),
                        "maximum_time_w1": float(arm_seeds.mean_time_w1.max()),
                        "maximum_occupation_w1": float(arm_seeds.mean_occupation_w1.max())}
    args.output.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(170/25.4, 174/25.4))
    a, b, c, d1, d2 = [fig.add_axes(pos) for pos in POSITIONS]

    # a: same bare-bar contract, with enough headroom for the new Constant arm.
    x = np.arange(6); width = .36
    time_values = np.array([float(rows.loc[key, "mean_time_w1"]) for key in ORDER])
    occ_values = np.array([float(rows.loc[key, "mean_occupation_w1"]) for key in ORDER])
    time_bars = a.bar(x-width/2, time_values, width, color=COLORS["time"], label=r"Time-resolved $W_1$", zorder=2)
    occ_bars = a.bar(x+width/2, occ_values, width, color=COLORS["occupation"], label=r"Occupation $W_1$", zorder=2)
    a.set_xticks(x, ["Free", "Noise 0.2", "Constant", "White", "Damping", "WGAN-GP Full"], rotation=40, ha="right")
    a.get_xticklabels()[-1].set_color(COLORS["full"]); a.get_xticklabels()[-1].set_fontweight("bold")
    a.set_ylabel(r"Mean channel-wise $W_1$")
    # Preserve the original in-panel legend without letting taller new bars
    # intersect it; only the y-axis headroom changes, never the bar heights.
    y_max = max(1.20, float(max(time_values.max(), occ_values.max()))*1.38)
    a.set_ylim(0, y_max)
    a.grid(axis="y", color="#E7E7E7", linewidth=.55, zorder=0)
    a.legend(loc="upper left", ncol=1, fontsize=9, handlelength=1)
    panel_label(a, "a")
    if not np.array_equal(np.array([bar.get_height() for bar in time_bars]), time_values):
        raise AssertionError("Time bar heights changed.")
    if not np.array_equal(np.array([bar.get_height() for bar in occ_bars]), occ_values):
        raise AssertionError("Occupation bar heights changed.")

    # b: retain every positive screening point, including high-gain failures.
    positive = damping[damping.mean_occupation_w1 > 0].copy()
    b.plot(positive.control_rms, positive.mean_occupation_w1, color="#A75C48", linewidth=1.1,
            marker="o", markersize=2.6, markeredgewidth=0, label="Scalar damping")
    b.scatter([full.control_rms], [full.mean_occupation_w1], s=38, color=COLORS["full"], edgecolor="white", linewidth=.7, zorder=5, label="WGAN-GP Full")
    b.scatter([matched.control_rms], [matched.mean_occupation_w1], s=32, color=COLORS["damping"], edgecolor="white", linewidth=.7, zorder=5, label="RMS-matched damping")
    b.axvline(float(full.control_rms), color="#606060", ls="--", lw=.75)
    b.set_yscale("log"); b.set_xlabel("Command RMS"); b.set_ylabel(r"Occupation $W_1$ (log)")
    b.grid(axis="y", which="both", color="#E7E7E7", linewidth=.5)
    best = positive.loc[positive.mean_occupation_w1.idxmin()]
    b.annotate(f"Best screened damping\n$K={best.gain:.2f}$, RMS={best.control_rms:.3f}",
               xy=(best.control_rms, best.mean_occupation_w1), xytext=(.08, .46), textcoords="axes fraction",
               arrowprops={"arrowstyle": "-", "color": "#777777", "lw": .6}, fontsize=9)
    b.legend(loc="upper left", fontsize=9, handlelength=1)
    panel_label(b, "b")

    # c: diffusion changes are sensitivities of the predictive plant, not control.
    c.plot(diffusion.kappa_sigma, diffusion.mean_occupation_w1, color=COLORS["diffusion"], marker="o", markersize=3, lw=1.2, label="To preictal reference")
    c.plot(diffusion.kappa_sigma, diffusion.mean_occupation_w1_to_observed_ictal, color=COLORS["observed"], marker="s", markersize=2.8, lw=1.1, label="To recorded ictal")
    c.axvline(1., color="#606060", ls="--", lw=.75)
    oracle = float(experiment["diffusion_sensitivity"]["posthoc_reference_oracle_kappa"])
    c.axvline(oracle, color=COLORS["diffusion"], ls=":", lw=.8)
    c.annotate(r"Frozen $\kappa_\sigma=1$", xy=(1., float(diffusion.loc[diffusion.kappa_sigma == 1., "mean_occupation_w1_to_observed_ictal"].iloc[0])),
               xytext=(.88, .18), textcoords="data", arrowprops={"arrowstyle": "-", "color": "#666666", "lw": .6}, fontsize=9, color="#4A4A4A")
    c.annotate(rf"Post hoc $\kappa_\sigma={oracle:g}$", xy=(oracle, float(diffusion.loc[diffusion.kappa_sigma == oracle, "mean_occupation_w1"].iloc[0])),
               xytext=(.38, .056), textcoords="data", arrowprops={"arrowstyle": "-", "color": COLORS["diffusion"], "lw": .6}, fontsize=9, color=COLORS["diffusion"])
    c.set_xlabel(r"Relative noise multiplier $\kappa_\sigma$"); c.set_ylabel(r"Occupation $W_1$")
    c.set_xlim(-.03, 1.53); c.grid(axis="y", color="#E7E7E7", linewidth=.55)
    c.legend(loc="upper left", fontsize=9, handlelength=1)
    panel_label(c, "c")

    # d1/d2: descriptive bookkeeping and unchanged reference/free SD ratios.
    location = float(moment.shapley_location_contribution.mean())
    scale = float(moment.shapley_scale_contribution.mean())
    shape = float(moment.shape_residual_contribution.mean())
    total = location+scale+shape
    per_channel_error = float(np.max(np.abs(moment.shapley_location_contribution + moment.shapley_scale_contribution + moment.shape_residual_contribution - moment.free_to_reference_w1)))
    if per_channel_error > 1e-12: raise ValueError("Location-scale bookkeeping does not sum to total W1.")
    left = 0.
    for value, color, label in [(location, COLORS["location"], "Location"), (scale, COLORS["scale"], "Scale"), (shape, COLORS["shape"], "Residual shape")]:
        d1.barh([0], [value], left=left, color=color, height=.46, label=label); left += value
    d1.set_xlim(0, total*1.02); d1.set_yticks([])
    d1.set_xlabel(r"Free → reference $W_1$ contribution"); d1.set_xticks([0, .1, .2, .3])
    d1.legend(loc="lower center", bbox_to_anchor=(.50, 1.12), ncol=3, fontsize=9, columnspacing=.7, handlelength=1)
    d1.text(location+scale/2, 0, f"Scale\n{100*scale/total:.1f}%", ha="center", va="center", fontsize=9, color="#3B2B1C", fontweight="bold")
    panel_label(d1, "d")
    rng = np.random.default_rng(20261109)
    sd_plot = []
    for index, (values, color) in enumerate([(moment.reference_to_observed_sd_ratio.to_numpy(), COLORS["observed"]),
                                           (moment.reference_to_free_sd_ratio.to_numpy(), COLORS["free"])]):
        jitter = rng.uniform(-.08, .08, values.size)
        d2.scatter(index+jitter, values, s=9, color=color, alpha=.52, edgecolor="none", zorder=2)
        quartiles = np.quantile(values, [.25, .5, .75])
        d2.plot([index-.16, index+.16], [quartiles[1]]*2, color=color, lw=1.5)
        d2.plot([index, index], [quartiles[0], quartiles[2]], color=color, lw=2.4)
        sd_plot.extend({"group": "Reference / recorded ictal" if index == 0 else "Reference / free model",
                        "channel": str(channel), "x_with_jitter": float(x), "sd_ratio": float(v)}
                       for channel, x, v in zip(moment.channel, index+jitter, values))
    d2.axhline(1., color="#777777", ls="--", lw=.7)
    d2.set_xticks([0, 1], ["Reference /\nrecorded ictal", "Reference /\nfree model"])
    d2.set_ylabel("SD ratio"); d2.set_xlim(-.45, 1.45); d2.set_yticks([0, .5, 1])
    d2.set_ylim(0, max(1.1, float(max(moment.reference_to_observed_sd_ratio.max(), moment.reference_to_free_sd_ratio.max()))*1.08))
    d2.grid(axis="y", color="#E7E7E7", linewidth=.55)
    fig.text(.09, .984, "HUP060 run-02: input audit (WGAN-GP Full) and noise sensitivity", va="top", ha="left", fontsize=10, fontweight="bold")
    fig.canvas.draw()
    stem = args.output/"hup060_trivial_controller_baselines_wgangp_full"
    for extension in ("pdf", "png", "svg"):
        kwargs = {"metadata": {"CreationDate": None, "ModDate": None}} if extension == "pdf" else {}
        fig.savefig(stem.with_suffix("."+extension), dpi=300, **kwargs)
    plotted_bars = pd.DataFrame({"variant": ORDER, "mean_time_w1": time_values, "mean_occupation_w1": occ_values,
                                "command_rms": [float(rows.loc[key, "control_rms"]) for key in ORDER]})
    plotted_bars.to_csv(args.output/"plotted_bar_values.csv", index=False, encoding="utf-8-sig")
    positive.to_csv(args.output/"plotted_damping_screen.csv", index=False, encoding="utf-8-sig")
    diffusion.to_csv(args.output/"plotted_diffusion_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sd_plot).to_csv(args.output/"plotted_sd_ratios.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"location": location, "scale": scale, "shape": shape, "total": total,
                   "scale_percent": 100*scale/total}]).to_csv(args.output/"plotted_location_scale_summary.csv", index=False, encoding="utf-8-sig")
    qa = {"source_sha256": {name: sha(path) for name, path in files.items()},
          "exact_paired_noise_maximum_error": noise_error, "all_values_from_exact_paired_sources": True,
          "dimensions_mm": [170, 174], "axis_positions": POSITIONS, "axis_count": len(fig.axes),
          "panel_a_bars": plotted_bars.to_dict(orient="records"), "panel_a_y_upper_limit": y_max,
          "panel_a_all_bars_visible": bool(y_max > max(time_values.max(), occ_values.max())),
          "panel_a_error_bars": False, "panel_a_extra_explanatory_annotations": False,
          "random_input_seed_audit": seed_qa,
          "full_command_rms": float(full.control_rms), "matched_damping_command_rms": float(matched.control_rms),
          "matched_damping_gain": float(matched.gain), "best_screened_damping": {"gain": float(best.gain), "rms": float(best.control_rms), "occupation_w1": float(best.mean_occupation_w1)},
          "damping_screen_points_retained": len(positive), "damping_occupation_w1_max": float(positive.mean_occupation_w1.max()),
          "moment_sum_maximum_error": per_channel_error,
          "statistics": "Constant/White medians of 32 technical command seeds; others fixed evaluations. No error intervals or patient-level inference.",
          "no_training_or_parameter_change": True, "overleaf_modified": False,
          "output_sha256": {extension: sha(stem.with_suffix("."+extension)) for extension in ("pdf", "png", "svg")}}
    with (args.output/"baseline_original_style_numeric_qa.json").open("w", encoding="utf-8") as f:
        json.dump(qa, f, ensure_ascii=False, indent=2)
    plt.close(fig)
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
