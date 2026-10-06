#!/usr/bin/env python
"""Create publication-grade comparison figures for the Actor--WGAN candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


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


PROJECT = Path(__file__).resolve().parents[1]

COLORS = {
    "observed": "#272727",
    "free": "#9A9A9A",
    "reference": "#3C8D62",
    "original": "#D17A22",
    "candidate": "#2166AC",
    "selected": "#5B7FCA",
    "unselected": "#A8A8A8",
    "accent": "#B64342",
}


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.16,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def kde_curve(
    values: np.ndarray, grid: np.ndarray, *, bandwidth: float = 0.16
) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if np.std(data) < 1.0e-10:
        data = data + np.linspace(-1.0e-6, 1.0e-6, len(data))
    sample_sd = float(np.std(data, ddof=1))
    factor = float(bandwidth) / max(sample_sd, 1.0e-8)
    return gaussian_kde(data, bw_method=factor)(grid)


def save_bundle(fig, base: Path, *, overleaf_pdf: Path | None = None) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    if overleaf_pdf is not None:
        overleaf_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(overleaf_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate paper Figure 6 only.')
    parser.add_argument("--source-directory", type=Path, default=PROJECT / "output" / "part3" / "source_data" / "figures_06_08")
    parser.add_argument("--output-directory", type=Path, default=PROJECT / "output" / "part3" / "figure_06")
    args = parser.parse_args()
    data_path = args.source_directory / "paired_comparison.npz"
    if not data_path.is_file():
        raise FileNotFoundError(f"missing frozen paired evaluation: {data_path}")
    data = np.load(data_path)
    figures = args.output_directory
    figures.mkdir(parents=True, exist_ok=True)
    source_output = PROJECT / "output" / "part3" / "source_data" / "figure_06"
    source_output.mkdir(parents=True, exist_ok=True)

    observed = np.asarray(data["observed_scaled"])
    free = np.asarray(data["uncontrolled_scaled"])
    original = np.asarray(data["original_controlled_scaled"])
    candidate = np.asarray(data["candidate_controlled_scaled"])
    reference = np.asarray(data["reference_validation_scaled"])
    original_controls = np.asarray(data["original_controls"])
    candidate_controls = np.asarray(data["candidate_controls"])
    original_effective = np.asarray(data["original_effective_control"])
    candidate_effective = np.asarray(data["candidate_effective_control"])
    representative = np.asarray(data["representative_indices"], dtype=int)
    selected = np.asarray(data["selected_indices"], dtype=int)
    channels = np.asarray(data["channels"]).astype(str)
    fixed_particle = int(np.asarray(data["fixed_particle_index"])[0])
    fixed_reference = int(np.asarray(data["fixed_reference_path_index"])[0])
    fs = float(np.asarray(data["sampling_rate_hz"])[0])
    time = np.arange(observed.shape[0]) / fs
    roles = ("selected SOZ", "selected non-SOZ", "unselected")
    # Figure contract: the candidate should show an externally measured,
    # modest law-matching gain without hiding control cost or particle choice.
    contract = {
        "core_conclusion": (
            "Adding a time-conditioned WGAN-GP critic to the locked structured "
            "Actor yields a modest paired improvement in reference-law matching."
        ),
        "figure_archetype": "quantitative grid",
        "target_output": "double-column manuscript figure",
        "backend": "Python/matplotlib only",
        "final_size_mm": {"width": 183, "representative_height": 184},
        "panel_map": {
            "representative_a_c": "fixed-particle 1-s trajectories",
            "representative_d_f": "complete-particle occupation laws",
            "representative_g_i": "direct or graph-mediated control input",
            "summary_a": "all-channel primary W1 endpoints",
            "summary_b": "paired per-channel occupation W1",
            "summary_c": "time-localized mean W1",
            "summary_d": "critic and validation diagnostics",
        },
        "statistics": {
            "particles": 32,
            "reference_paths": 15,
            "channels": 36,
            "fixed_particle": fixed_particle,
            "fixed_reference_path": fixed_reference,
            "uncertainty_warning": (
                "particles and channels are technical/model units, not independent patients"
            ),
        },
        "reviewer_risks": [
            "run-02 is a development window",
            "critic score is not used as the final W1 metric",
            "single-particle traces are illustrative; distributions use all particles",
            "unselected-node input is graph-mediated, not direct stimulation",
        ],
    }
    (figures / "figure_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    trajectory_rows: list[dict[str, object]] = []
    density_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(183 / 25.4, 184 / 25.4),
        gridspec_kw={"height_ratios": [1.0, 0.95, 0.72]},
    )
    for column, (channel, role) in enumerate(zip(representative, roles)):
        label = channels[channel]
        ax = axes[0, column]
        trajectory_series = {
            "Observed ictal": observed[:, channel],
            "Free RC-SDE": free[fixed_particle, :, channel],
            "Interictal reference": reference[fixed_reference, :, channel],
            "Original Actor": original[fixed_particle, :, channel],
            "Actor + WGAN-GP": candidate[fixed_particle, :, channel],
        }
        styles = {
            "Observed ictal": (COLORS["observed"], 0.85, "-", 0.82),
            "Free RC-SDE": (COLORS["free"], 0.85, "-", 0.85),
            "Interictal reference": (COLORS["reference"], 1.00, "-", 0.95),
            "Original Actor": (COLORS["original"], 0.90, "--", 0.95),
            "Actor + WGAN-GP": (COLORS["candidate"], 1.05, "-", 1.00),
        }
        for series_label, values in trajectory_series.items():
            color, linewidth, linestyle, alpha = styles[series_label]
            ax.plot(
                time,
                values,
                color=color,
                lw=linewidth,
                ls=linestyle,
                alpha=alpha,
                label=series_label,
            )
            trajectory_rows.extend(
                {
                    "node": label,
                    "role": role,
                    "time_s": float(t),
                    "series": series_label,
                    "standardized_amplitude": float(value),
                    "fixed_particle": fixed_particle,
                    "fixed_reference_path": fixed_reference,
                }
                for t, value in zip(time, values)
            )
        ax.set_title(f"{label} · {role}", fontsize=8, fontweight="bold", loc="left")
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xlabel("Time (s)")
        if column == 0:
            ax.set_ylabel("Standardized amplitude")
        panel_label(ax, "abc"[column])

        ax = axes[1, column]
        pooled = np.concatenate(
            [
                observed[:, channel],
                free[:, :, channel].reshape(-1),
                reference[:, :, channel].reshape(-1),
                original[:, :, channel].reshape(-1),
                candidate[:, :, channel].reshape(-1),
            ]
        )
        low, high = np.quantile(pooled, [0.002, 0.998])
        margin = 0.08 * max(high - low, 1.0)
        grid = np.linspace(low - margin, high + margin, 300)
        density_series = {
            "Observed ictal": observed[:, channel],
            "Free RC-SDE": free[:, :, channel],
            "Interictal reference": reference[:, :, channel],
            "Original Actor": original[:, :, channel],
            "Actor + WGAN-GP": candidate[:, :, channel],
        }
        for series_label, values in density_series.items():
            density = kde_curve(values, grid)
            color, linewidth, linestyle, alpha = styles[series_label]
            ax.plot(
                grid,
                density,
                color=color,
                lw=linewidth,
                ls=linestyle,
                alpha=alpha,
            )
            if series_label == "Interictal reference":
                ax.fill_between(grid, density, color=color, alpha=0.12)
            density_rows.extend(
                {
                    "node": label,
                    "role": role,
                    "standardized_amplitude": float(x),
                    "series": series_label,
                    "density": float(y),
                }
                for x, y in zip(grid, density)
            )
        old_w1 = float(data["occupation_w1_original"][channel])
        new_w1 = float(data["occupation_w1_candidate"][channel])
        ax.text(
            0.98,
            0.95,
            f"occupation $W_1$\nold {old_w1:.3f}  new {new_w1:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.3,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )
        ax.set_xlabel("Standardized amplitude")
        if column == 0:
            ax.set_ylabel("Density")
        ax.set_yticks([])
        panel_label(ax, "def"[column])

        ax = axes[2, column]
        actuator_location = np.flatnonzero(selected == channel)
        if len(actuator_location):
            actuator = int(actuator_location[0])
            old_input = original_controls[fixed_particle, :, actuator]
            new_input = candidate_controls[fixed_particle, :, actuator]
            input_label = "Direct Actor input"
            input_unit = "normalized direct input"
        else:
            old_input = original_effective[fixed_particle, :, channel]
            new_input = candidate_effective[fixed_particle, :, channel]
            input_label = "Graph-mediated effective input\n(no direct Actor channel)"
            input_unit = "normalized graph-mediated input"
        ax.plot(
            time,
            old_input,
            color=COLORS["original"],
            lw=0.9,
            ls="--",
            label="Original Actor",
        )
        ax.plot(
            time,
            new_input,
            color=COLORS["candidate"],
            lw=1.05,
            label="Actor + WGAN-GP",
        )
        ax.axhline(0.0, color="#C7C7C7", lw=0.6, zorder=0)
        ax.text(
            0.02,
            0.94,
            input_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.2,
            color="#4D4D4D",
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xlabel("Time (s)")
        if column == 0:
            ax.set_ylabel("Control input")
        panel_label(ax, "ghi"[column])
        for series_label, values in {
            "Original Actor": old_input,
            "Actor + WGAN-GP": new_input,
        }.items():
            control_rows.extend(
                {
                    "node": label,
                    "role": role,
                    "time_s": float(t),
                    "series": series_label,
                    "input_value": float(value),
                    "input_definition": input_unit,
                    "fixed_particle": fixed_particle,
                }
                for t, value in zip(time, values)
            )

    handles = [
        Line2D([0], [0], color=COLORS["observed"], lw=1.2, label="Observed ictal"),
        Line2D([0], [0], color=COLORS["free"], lw=1.2, label="Free RC-SDE"),
        Line2D([0], [0], color=COLORS["reference"], lw=1.2, label="Interictal reference"),
        Line2D([0], [0], color=COLORS["original"], lw=1.2, ls="--", label="Original Actor"),
        Line2D([0], [0], color=COLORS["candidate"], lw=1.3, label="Actor + WGAN-GP"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        handlelength=2.0,
        columnspacing=1.25,
        fontsize=6.5,
    )
    fig.suptitle(
        "HUP060 run-02: paired evaluation of the original Actor and Actor + WGAN-GP",
        x=0.08,
        y=1.035,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.075, top=0.91, wspace=0.32, hspace=0.52)
    save_bundle(
        fig,
        figures / "hup060_actor_wgan_mfc_preview",
        overleaf_pdf=(
            None
        ),
    )
    pd.DataFrame(trajectory_rows).to_csv(
        source_output / "source_data_representative_trajectories.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(density_rows).to_csv(
        source_output / "source_data_representative_densities.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(control_rows).to_csv(
        source_output / "source_data_representative_controls.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(figures / "hup060_actor_wgan_mfc_preview.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
