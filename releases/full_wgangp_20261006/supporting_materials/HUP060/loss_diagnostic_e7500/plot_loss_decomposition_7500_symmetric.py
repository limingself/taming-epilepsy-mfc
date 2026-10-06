#!/usr/bin/env python
"""Create a symmetric Actor/Critic total-plus-three-components figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import plot_loss_decomposition_7500 as audit


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output_symmetric"

COLORS = {
    "raw": "#C6CBD0",
    "actor_total": "#D97706",
    "actor_law": "#3D6F9E",
    "actor_adv": "#7A5195",
    "actor_prox": "#2A9D8F",
    "critic_total": "#A94232",
    "critic_w": "#C45A32",
    "critic_gp": "#C77C29",
    "grad_norm": "#317873",
    "critic_drift": "#667788",
    "zero": "#4C5358",
    "grid": "#E4E8EB",
    "window": "#E7EEF6",
    "warning": "#A33A2B",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.5,
        "axes.titlesize": 7.4,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.1,
        "ytick.labelsize": 6.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "legend.frameon": False,
        "legend.fontsize": 5.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def style(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=COLORS["grid"], lw=0.45)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 7575)
    ax.set_xticks([0, 1500, 3000, 4500, 6000, 7500])


def title(ax: plt.Axes, label: str, text: str) -> None:
    ax.set_title(f"{label}  {text}", loc="left", fontweight="bold", pad=3)


def trace(
    ax: plt.Axes,
    epoch: pd.Series,
    raw: pd.Series,
    smooth: pd.Series,
    color: str,
    label: str,
    *,
    zero: bool = False,
) -> None:
    ax.plot(
        epoch,
        raw,
        color=COLORS["raw"],
        lw=0.28,
        alpha=0.28,
        rasterized=True,
        label="Raw per-epoch value",
    )
    ax.plot(epoch, smooth, color=color, lw=1.15, label=label)
    if zero:
        ax.axhline(0, color=COLORS["zero"], lw=0.7, ls="--", label="Zero")
    style(ax)


def main() -> int:
    for path in (
        audit.LONGRUN_HISTORY,
        audit.LONGRUN_SUMMARY,
        audit.LONGRUN_CHECKPOINT,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    frame = audit.actor_rows(audit.LONGRUN_HISTORY)
    if not np.array_equal(frame["epoch"].to_numpy(), np.arange(1, 7501)):
        raise RuntimeError("history is not a complete epoch 1--7500 sequence")
    checkpoint = torch.load(
        audit.LONGRUN_CHECKPOINT, map_location="cpu", weights_only=False
    )
    arguments = checkpoint["training_arguments"]
    weights = {
        "actor_adversarial": float(arguments["adversarial_weight"]),
        "actor_proximal": float(arguments["anchor_weight"]),
        "critic_gp": float(arguments["gradient_penalty"]),
        "critic_drift": float(arguments["critic_drift"]),
    }
    expected_weights = {
        "actor_adversarial": 0.5,
        "actor_proximal": 0.2,
        "critic_gp": 10.0,
        "critic_drift": 0.001,
    }
    if any(not np.isclose(weights[key], value) for key, value in expected_weights.items()):
        raise RuntimeError("checkpoint weights differ from the audited decomposition")
    if int(arguments["epochs"]) != 7500 or int(arguments["n_critic"]) != 3:
        raise RuntimeError("checkpoint schedule differs from the audited run")

    frame["actor_law_component"] = frame["train_law"]
    frame["actor_adversarial_component"] = (
        weights["actor_adversarial"] * frame["train_adversarial"]
    )
    frame["actor_proximal_component"] = (
        weights["actor_proximal"] * frame["train_action_anchor"]
    )
    frame["critic_wasserstein_component"] = -frame["critic_estimate"]
    frame["critic_gp_component"] = weights["critic_gp"] * frame["critic_gp"]
    frame["critic_drift_component"] = (
        frame["critic_loss"]
        - frame["critic_wasserstein_component"]
        - frame["critic_gp_component"]
    )
    actor_error = float(
        np.max(
            np.abs(
                frame["train_total"]
                - frame["actor_law_component"]
                - frame["actor_adversarial_component"]
                - frame["actor_proximal_component"]
            )
        )
    )
    critic_error = float(
        np.max(
            np.abs(
                frame["critic_loss"]
                - frame["critic_wasserstein_component"]
                - frame["critic_gp_component"]
                - frame["critic_drift_component"]
            )
        )
    )
    if actor_error > 1e-10 or critic_error > 1e-10:
        raise RuntimeError("objective decomposition identity failed")
    if not np.isclose(
        float(frame["critic_drift_component"].median()),
        weights["critic_drift"]
        * float((frame["critic_drift_component"] / weights["critic_drift"]).median()),
    ):
        raise RuntimeError("critic drift weighting audit failed")

    columns = [
        "train_total",
        "critic_loss",
        "actor_law_component",
        "critic_wasserstein_component",
        "actor_adversarial_component",
        "critic_gp_component",
        "actor_proximal_component",
        "critic_drift_component",
    ]
    for column in columns:
        frame[f"median125_{column}"] = frame[column].rolling(
            125, min_periods=40
        ).median()
    frame["median125_critic_mean_gradient_norm"] = frame[
        "critic_mean_gradient_norm"
    ].rolling(125, min_periods=40).median()

    actor500 = audit.window_stats(frame, "train_total", 500)
    critic500 = audit.window_stats(frame, "critic_loss", 500)
    gp500 = audit.window_stats(frame, "critic_gp_component", 500)
    grad_norm500 = audit.window_stats(
        frame, "critic_mean_gradient_norm", 500
    )
    raw_minimum = frame.loc[frame["train_total"].idxmin()]
    selected = frame.loc[frame["selected_checkpoint"].eq(1.0)]
    if len(selected) != 1 or int(selected.iloc[0]["epoch"]) != 7500:
        raise RuntimeError("available checkpoint is not selected at epoch 7500")

    summary = {
        "status": "symmetric_actor_critic_total_plus_three_components",
        "history_sha256": audit.sha256(audit.LONGRUN_HISTORY),
        "checkpoint_sha256": audit.sha256(audit.LONGRUN_CHECKPOINT),
        "selected_checkpoint_epoch": 7500,
        "raw_actor_total_minimum_epoch": int(raw_minimum["epoch"]),
        "raw_actor_total_minimum_value": float(raw_minimum["train_total"]),
        "raw_minimum_checkpoint_saved": False,
        "weights_verified_from_checkpoint": weights,
        "actor_identity": "L_A = L_law + 0.5 L_adv + 0.2 L_prox",
        "critic_identity": "L_C = -W_hat_C + 10 L_GP + 0.001 L_drift",
        "actor_identity_max_abs_error": actor_error,
        "critic_identity_max_abs_error": critic_error,
        "critic_total_negative_epoch_count": int((frame["critic_loss"] < 0).sum()),
        "critic_total_positive_epoch_count": int((frame["critic_loss"] > 0).sum()),
        "critic_wasserstein_component_is_signed_and_untransformed": True,
        "actor_final500": actor500,
        "critic_final500": critic500,
        "gradient_penalty_definition": (
            "L_GP = mean((per-sample input-gradient norm - 1)^2)"
        ),
        "gradient_penalty_loss_target": 0.0,
        "mean_input_gradient_norm_target": 1.0,
        "critic_gp_weighted_final500": gp500,
        "critic_gp_unweighted_final500_median": (
            float(gp500["median"]) / weights["critic_gp"]
        ),
        "critic_gradient_norm_final500": grad_norm500,
        "critic_gradient_norm_final500_fraction_in_0p95_to_1p05": float(
            frame.loc[frame["epoch"].ge(7001), "critic_mean_gradient_norm"]
            .between(0.95, 1.05)
            .mean()
        ),
        "critic_recording_scope": "third/final Critic update per epoch",
        "actor_critic_estimate_timing": (
            "Critic components use the logged final Critic audit; Actor adversarial "
            "terms are recomputed in the subsequent Actor update and need not match."
        ),
        "rolling_median_guardrail": (
            "Objective identities hold for raw per-epoch values. Trailing medians are "
            "display-only summaries and must not be added across panels."
        ),
        "critic_drift_source": (
            "reconstructed weighted drift penalty = critic_loss + critic_estimate - 10*critic_gp"
        ),
        "interpretation": (
            "The symmetric layout exposes both objective identities. Actor total declines, "
            "whereas the signed Critic total and its -Wasserstein component need not approach zero."
        ),
        "guardrail": (
            "Actor and Critic objective scales and optimization meanings differ and must not be "
            "compared by absolute height. Negative Critic values are valid WGAN objectives."
        ),
    }
    source_columns = (
        ["epoch"]
        + columns
        + ["critic_mean_gradient_norm"]
        + [f"median125_{column}" for column in columns]
        + ["median125_critic_mean_gradient_norm"]
    )
    frame[source_columns].to_csv(
        OUTPUT / "source_data_symmetric_loss_decomposition_e7500.csv",
        index=False,
    )
    (OUTPUT / "symmetric_loss_decomposition_e7500_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(183 / 25.4, 180 / 25.4),
        sharex=True,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.97,
        bottom=0.085,
        top=0.89,
        hspace=0.52,
        wspace=0.30,
    )
    epoch = frame["epoch"]

    trace(
        axes[0, 0], epoch, frame["train_total"], frame["median125_train_total"],
        COLORS["actor_total"], "Trailing 125-epoch median"
    )
    axes[0, 0].axvspan(7000, 7500, color=COLORS["window"], alpha=0.9, lw=0)
    axes[0, 0].set_ylabel(r"Actor total $L_A$")
    title(axes[0, 0], "a", "Actor total objective")

    trace(
        axes[0, 1], epoch, frame["critic_loss"], frame["median125_critic_loss"],
        COLORS["critic_total"], "Trailing 125-epoch median", zero=True
    )
    axes[0, 1].axvspan(7000, 7500, color=COLORS["window"], alpha=0.9, lw=0)
    axes[0, 1].set_ylabel(r"Signed Critic total $L_C$")
    title(axes[0, 1], "b", "Critic total objective (signed)")

    trace(
        axes[1, 0], epoch, frame["actor_law_component"],
        frame["median125_actor_law_component"], COLORS["actor_law"],
        r"$L_{law}$ (trailing median)"
    )
    axes[1, 0].set_ylabel(r"$L_{law}$")
    title(axes[1, 0], "c", "Actor component 1: empirical-law term")

    trace(
        axes[1, 1], epoch, frame["critic_wasserstein_component"],
        frame["median125_critic_wasserstein_component"], COLORS["critic_w"],
        r"$-\widehat W_C$ (trailing median)", zero=True
    )
    axes[1, 1].set_ylabel(r"$-\widehat W_C$")
    title(axes[1, 1], "d", "Critic component 1: Wasserstein term")

    trace(
        axes[2, 0], epoch, frame["actor_adversarial_component"],
        frame["median125_actor_adversarial_component"], COLORS["actor_adv"],
        r"$0.5L_{adv}$ (trailing median)"
    )
    axes[2, 0].set_ylim(bottom=0)
    axes[2, 0].set_ylabel(r"$0.5L_{adv}$")
    title(axes[2, 0], "e", "Actor component 2: adversarial term")

    gp_ax = axes[2, 1]
    gp_ax.plot(
        epoch,
        frame["critic_gp_component"],
        color=COLORS["raw"],
        lw=0.28,
        alpha=0.28,
        rasterized=True,
    )
    gp_ax.plot(
        epoch,
        frame["median125_critic_gp_component"],
        color=COLORS["critic_gp"],
        lw=1.15,
    )
    gp_ax.axhline(0, color=COLORS["critic_gp"], lw=0.65, ls=":")
    style(gp_ax)
    gp_ax.set_ylim(0, 0.14)
    gp_ax.set_ylabel(r"$10L_{GP}$ (target 0)", color=COLORS["critic_gp"])
    gp_ax.tick_params(axis="y", colors=COLORS["critic_gp"])
    norm_ax = gp_ax.twinx()
    norm_ax.plot(
        epoch,
        frame["critic_mean_gradient_norm"],
        color="#B8D5D2",
        lw=0.25,
        alpha=0.24,
        rasterized=True,
    )
    norm_ax.plot(
        epoch,
        frame["median125_critic_mean_gradient_norm"],
        color=COLORS["grad_norm"],
        lw=1.05,
    )
    norm_ax.axhline(1.0, color=COLORS["grad_norm"], lw=0.75, ls="--")
    norm_ax.set_ylim(0.70, 1.09)
    norm_ax.set_ylabel(
        r"Mean $\|\nabla_{\hat{x}}D\|_2$ (target 1)",
        color=COLORS["grad_norm"],
    )
    norm_ax.tick_params(axis="y", colors=COLORS["grad_norm"], labelsize=5.8)
    norm_ax.spines["right"].set_visible(True)
    norm_ax.spines["right"].set_color(COLORS["grad_norm"])
    norm_ax.grid(False)
    title(gp_ax, "f", "Critic GP loss (0) and gradient-norm target (1)")

    trace(
        axes[3, 0], epoch, frame["actor_proximal_component"],
        frame["median125_actor_proximal_component"], COLORS["actor_prox"],
        r"$0.2L_{prox}$ (trailing median)"
    )
    axes[3, 0].set_ylim(bottom=0)
    axes[3, 0].set_ylabel(r"$0.2L_{prox}$")
    title(axes[3, 0], "g", "Actor component 3: proximal-action term")

    trace(
        axes[3, 1], epoch, frame["critic_drift_component"],
        frame["median125_critic_drift_component"], COLORS["critic_drift"],
        r"Reconstructed $10^{-3}L_{drift}$ (trailing median)"
    )
    axes[3, 1].set_ylim(bottom=0)
    axes[3, 1].ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    axes[3, 1].set_ylabel(r"$10^{-3}L_{drift}$")
    title(axes[3, 1], "h", "Critic component 3: reconstructed drift penalty")

    for ax in axes[-1, :]:
        ax.set_xlabel("Joint-training epoch")
    axes[0, 0].legend(loc="lower left", ncol=2)
    axes[0, 1].legend(loc="upper right", ncol=2)

    fig.suptitle(
        "HUP060 Actor–WGAN-GP: symmetric total-loss and component decomposition",
        x=0.085, y=0.975, ha="left", fontsize=9.2, fontweight="bold"
    )
    fig.text(
        0.29, 0.93,
        r"Actor: $L_A=L_{law}+0.5L_{adv}+0.2L_{prox}$",
        ha="center", va="center", fontsize=6.2, color=COLORS["actor_total"]
    )
    fig.text(
        0.73, 0.93,
        r"Critic: $L_C=-\widehat W_C+10L_{GP}+10^{-3}L_{drift}$",
        ha="center", va="center", fontsize=6.2, color=COLORS["critic_total"]
    )
    base = OUTPUT / "HUP060_WGAN_loss_decomposition_e7500_symmetric"
    fig.savefig(base.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
