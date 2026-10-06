#!/usr/bin/env python
"""Create publication-grade comparison figures for the Actor--WGAN candidate."""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description='Generate paper Figure 8 only.')
    parser.add_argument("--source-directory", type=Path, default=PROJECT / "output" / "part3" / "source_data" / "figures_06_08")
    parser.add_argument("--output-directory", type=Path, default=PROJECT / "output" / "part3" / "figure_08")
    args = parser.parse_args()
    data_path = args.source_directory / "paired_comparison.npz"
    if not data_path.is_file():
        raise FileNotFoundError(f"missing frozen paired evaluation: {data_path}")
    data = np.load(data_path)
    figures = args.output_directory
    figures.mkdir(parents=True, exist_ok=True)
    source_output = PROJECT / "output" / "part3" / "source_data" / "figure_08"
    source_output.mkdir(parents=True, exist_ok=True)

    observed = np.asarray(data["observed_scaled"])
    free = np.asarray(data["uncontrolled_scaled"])
    original = np.asarray(data["original_controlled_scaled"])
    candidate = np.asarray(data["candidate_controlled_scaled"])
    reference = np.asarray(data["reference_validation_scaled"])
    selected = np.asarray(data["selected_indices"], dtype=int)
    channels = np.asarray(data["channels"]).astype(str)
    # Dense all-channel occupation-law plate.  Density scales are allowed to
    # vary by channel because several interictal laws are much narrower than
    # others; the standardized-amplitude axis is shared exactly.
    clinical_selected = {"RPFa1", "RPFa2", "RPFa3", "RPFb1", "RPFc1"}
    all_values = np.concatenate(
        [
            observed.reshape(-1),
            free.reshape(-1),
            reference.reshape(-1),
            original.reshape(-1),
            candidate.reshape(-1),
        ]
    )
    x_low, x_high = np.quantile(all_values, [0.002, 0.998])
    x_margin = 0.05 * max(x_high - x_low, 1.0)
    shared_grid = np.linspace(x_low - x_margin, x_high + x_margin, 240)
    fig, grid_axes = plt.subplots(
        6,
        6,
        figsize=(183 / 25.4, 190 / 25.4),
        sharex=True,
    )
    all36_density_rows: list[dict[str, object]] = []
    plate_styles = {
        "Observed ictal": (COLORS["observed"], 0.70, "-", 0.78),
        "Free RC-SDE": (COLORS["free"], 0.75, ":", 0.82),
        "Interictal reference": (COLORS["reference"], 0.90, "-.", 0.92),
        "Original Actor": (COLORS["original"], 0.78, "--", 0.92),
        "Actor + WGAN-GP": (COLORS["candidate"], 0.95, "-", 1.00),
    }
    for channel_index, ax in enumerate(grid_axes.reshape(-1)):
        channel_name = channels[channel_index]
        series_values = {
            "Observed ictal": observed[:, channel_index],
            "Free RC-SDE": free[:, :, channel_index],
            "Interictal reference": reference[:, :, channel_index],
            "Original Actor": original[:, :, channel_index],
            "Actor + WGAN-GP": candidate[:, :, channel_index],
        }
        maximum_density = 0.0
        for series_label, raw_values in series_values.items():
            density = kde_curve(raw_values, shared_grid)
            color, linewidth, linestyle, alpha = plate_styles[series_label]
            ax.plot(
                shared_grid,
                density,
                color=color,
                lw=linewidth,
                ls=linestyle,
                alpha=alpha,
            )
            maximum_density = max(maximum_density, float(density.max()))
            all36_density_rows.extend(
                {
                    "channel": channel_name,
                    "series": series_label,
                    "standardized_amplitude": float(value),
                    "density": float(density_value),
                }
                for value, density_value in zip(shared_grid, density)
            )
        if channel_name in clinical_selected:
            label_color = COLORS["accent"]
            suffix = "●"
        elif channel_index in selected:
            label_color = COLORS["selected"]
            suffix = "●"
        else:
            label_color = "#606060"
            suffix = ""
        ax.text(
            0.02,
            0.96,
            f"{channel_name}{suffix}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            fontweight="bold",
            color=label_color,
        )
        old_value = float(data["occupation_w1_original"][channel_index])
        new_value = float(data["occupation_w1_candidate"][channel_index])
        delta_color = "#2E8B57" if new_value < old_value else COLORS["accent"]
        ax.text(
            0.98,
            0.94,
            f"{old_value:.3f}→{new_value:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=5.0,
            color=delta_color,
        )
        ax.set_ylim(0.0, maximum_density * 1.10)
        ax.set_xlim(float(shared_grid[0]), float(shared_grid[-1]))
        ax.set_yticks([])
        if channel_index // 6 == 5:
            ax.set_xlabel("Amplitude", fontsize=5.6)
        else:
            ax.tick_params(axis="x", labelbottom=False)
        ax.tick_params(axis="x", labelsize=5.2, length=2)
    plate_handles = [
        Line2D([0], [0], color=plate_styles[label][0], lw=1.2,
               ls=plate_styles[label][2], label=label)
        for label in plate_styles
    ]
    fig.legend(
        handles=plate_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.988),
        ncol=5,
        fontsize=6.0,
        handlelength=2.0,
        columnspacing=1.0,
    )
    fig.suptitle(
        "HUP060 run-02: all-channel occupation laws after WGAN-GP Actor fine-tuning",
        x=0.06,
        y=1.015,
        ha="left",
        fontsize=8.8,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "Panel text: Original Actor → Actor + WGAN-GP occupation $W_1$; "
        "green/red values indicate improvement/worsening. ● denotes a selected actuator.",
        ha="center",
        va="bottom",
        fontsize=5.6,
    )
    fig.subplots_adjust(left=0.045, right=0.995, bottom=0.06, top=0.935, wspace=0.12, hspace=0.19)
    save_bundle(
        fig,
        figures / "hup060_actor_wgan_all36_distribution_grid",
        overleaf_pdf=(
            None
        ),
    )
    pd.DataFrame(all36_density_rows).to_csv(
        source_output / "source_data_actor_wgan_all36_densities.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(figures / "hup060_actor_wgan_all36_distribution_grid.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
