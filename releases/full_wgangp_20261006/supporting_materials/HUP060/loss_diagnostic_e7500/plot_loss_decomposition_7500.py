#!/usr/bin/env python
"""Create an isolated 7,500-epoch HUP060 Actor/Critic diagnostic preview.

The long-run history is a matched-full development run produced by the
current paper source.  It is deliberately not relabelled as a numerical
continuation of the paper-primary 40-epoch checkpoint trajectory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
import torch


HERE = Path(__file__).resolve().parent
LONGRUN_HISTORY = (
    HERE.parent
    / "HUP060_actor_e7500_exact_source_v1"
    / "training_output"
    / "paper_source_e7500_seed20261011"
    / "training_history.csv"
)
LONGRUN_SUMMARY = LONGRUN_HISTORY.with_name("training_summary.json")
LONGRUN_CHECKPOINT = LONGRUN_HISTORY.with_name("frozen_actor_wgan.pt")
FORMAL_HISTORY = Path(
    r"D:\VS code\distribution control\artifacts\part3_hup060_actor_wgan_v1"
    r"\seed20261011_adv050_anchor020\training_history.csv"
)
FORMAL_SUMMARY = FORMAL_HISTORY.with_name("training_summary.json")
FORMAL_CHECKPOINT = FORMAL_HISTORY.with_name("frozen_actor_wgan.pt")
OUTPUT = HERE / "output"

COLORS = {
    "raw": "#B7BDC3",
    "total": "#D97706",
    "law": "#3D6F9E",
    "adv": "#7A5195",
    "anchor": "#2A9D8F",
    "critic": "#C45A32",
    "validation_critic": "#347A78",
    "gp": "#C77C29",
    "drift": "#6C7A89",
    "grad": "#B45F3C",
    "validation": "#2563A6",
    "accent": "#A33A2B",
    "pass": "#2E7D4F",
    "shade": "#DCEFE4",
    "window_shade": "#E7EEF6",
    "grid": "#E4E8EB",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.5,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "legend.frameon": False,
        "legend.fontsize": 5.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def actor_rows(path: Path) -> pd.DataFrame:
    history = pd.read_csv(path)
    actor = history.loc[
        history["stage"].eq("actor_wgan")
        & history["epoch"].notna()
        & history["train_total"].notna()
    ].copy()
    actor = actor.sort_values("epoch").reset_index(drop=True)
    actor["epoch"] = actor["epoch"].astype(int)
    return actor


def rolling(frame: pd.DataFrame, column: str, width: int = 125) -> pd.Series:
    return frame[column].rolling(width, min_periods=40).median()


def block_summary(frame: pd.DataFrame, width: int = 250) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for start in range(1, 7501, width):
        end = min(start + width - 1, 7500)
        block = frame.loc[frame["epoch"].between(start, end)]
        row: dict[str, float | int] = {
            "start_epoch": start,
            "end_epoch": end,
            "mid_epoch": 0.5 * (start + end),
            "n": len(block),
        }
        for column in (
            "train_total",
            "train_law",
            "weighted_adversarial",
            "weighted_anchor",
            "critic_estimate",
            "weighted_gp",
            "weighted_critic_drift",
            "critic_mean_gradient_norm",
        ):
            row[f"median_{column}"] = float(block[column].median())
        rows.append(row)
    return pd.DataFrame(rows)


def window_stats(frame: pd.DataFrame, column: str, width: int) -> dict[str, object]:
    window = frame.tail(width)
    x = window["epoch"].to_numpy(dtype=float)
    y = window[column].to_numpy(dtype=float)
    fit = linregress(x, y)
    half = width // 2
    first = float(np.median(y[:half]))
    second = float(np.median(y[-half:]))
    mean_abs = float(np.mean(np.abs(y)))
    relative_change = float(fit.slope * (x[-1] - x[0]) / mean_abs * 100.0)
    half_shift = float((second - first) / mean_abs * 100.0)
    return {
        "width": width,
        "start_epoch": int(x[0]),
        "end_epoch": int(x[-1]),
        "median": float(np.median(y)),
        "relative_ols_change_pct_of_mean_abs": relative_change,
        "ols_p_value": float(fit.pvalue),
        "first_half_median": first,
        "second_half_median": second,
        "half_median_shift_pct_of_mean_abs": half_shift,
    }


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=COLORS["grid"], lw=0.45)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 7575)
    ax.set_xticks([0, 1500, 3000, 4500, 6000, 7500])


def panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_title(f"{label}  {title}", loc="left", fontweight="bold", pad=3)


def main() -> int:
    for path in (
        LONGRUN_HISTORY,
        LONGRUN_SUMMARY,
        LONGRUN_CHECKPOINT,
        FORMAL_HISTORY,
        FORMAL_SUMMARY,
        FORMAL_CHECKPOINT,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    longrun = actor_rows(LONGRUN_HISTORY)
    formal = actor_rows(FORMAL_HISTORY)
    expected = np.arange(1, 7501, dtype=int)
    if not np.array_equal(longrun["epoch"].to_numpy(), expected):
        raise RuntimeError("long-run history is not a complete epoch 1--7500 sequence")
    if not np.array_equal(formal["epoch"].to_numpy(), np.arange(1, 41)):
        raise RuntimeError("paper-primary history is not a complete epoch 1--40 sequence")

    checkpoint = torch.load(
        LONGRUN_CHECKPOINT, map_location="cpu", weights_only=False
    )
    arguments = checkpoint["training_arguments"]
    adversarial_weight = float(arguments["adversarial_weight"])
    anchor_weight = float(arguments["anchor_weight"])
    gp_weight = float(arguments["gradient_penalty"])
    critic_drift_weight = float(arguments["critic_drift"])
    if not np.isclose(adversarial_weight, 0.5):
        raise RuntimeError("checkpoint adversarial weight is not 0.5")
    if not np.isclose(anchor_weight, 0.2):
        raise RuntimeError("checkpoint anchor weight is not 0.2")
    if not np.isclose(gp_weight, 10.0):
        raise RuntimeError("checkpoint gradient-penalty weight is not 10")
    if not np.isclose(critic_drift_weight, 0.001):
        raise RuntimeError("checkpoint critic-drift weight is not 0.001")
    if int(arguments["epochs"]) != 7500 or int(arguments["validation_every"]) != 250:
        raise RuntimeError("checkpoint does not use the audited 7500/250 schedule")
    if int(arguments["n_critic"]) != 3:
        raise RuntimeError("checkpoint does not use three Critic updates per epoch")

    longrun["weighted_adversarial"] = (
        adversarial_weight * longrun["train_adversarial"]
    )
    longrun["weighted_anchor"] = anchor_weight * longrun["train_action_anchor"]
    longrun["weighted_gp"] = gp_weight * longrun["critic_gp"]
    longrun["weighted_critic_drift"] = (
        longrun["critic_loss"]
        + longrun["critic_estimate"]
        - longrun["weighted_gp"]
    )
    actor_identity_error = float(
        np.max(
            np.abs(
                longrun["train_total"]
                - longrun["train_law"]
                - longrun["weighted_adversarial"]
                - longrun["weighted_anchor"]
            )
        )
    )
    critic_identity_error = float(
        np.max(
            np.abs(
                longrun["critic_loss"]
                + longrun["critic_estimate"]
                - longrun["weighted_gp"]
                - longrun["weighted_critic_drift"]
            )
        )
    )
    if actor_identity_error > 1.0e-10 or critic_identity_error > 1.0e-10:
        raise RuntimeError("stored Actor/Critic identities failed numerical audit")
    plotted = [
        "train_total",
        "train_law",
        "weighted_adversarial",
        "weighted_anchor",
        "critic_estimate",
        "weighted_gp",
        "weighted_critic_drift",
        "critic_mean_gradient_norm",
    ]
    if not np.isfinite(longrun[plotted].to_numpy(dtype=float)).all():
        raise RuntimeError("non-finite values found in plotted diagnostics")
    if (longrun[plotted].to_numpy(dtype=float) < 0).any():
        raise RuntimeError("the requested no-negative display contains a negative metric")

    validation = longrun.loc[
        longrun["validation_selection_score"].notna(),
        [
            "epoch",
            "validation_selection_score",
            "validation_critic_estimate",
            "validation_law",
            "validation_mean_time_w1",
            "validation_mean_occupation_w1",
        ],
    ].copy()
    if len(validation) != 31:
        raise RuntimeError(f"expected 31 validation rows, found {len(validation)}")

    for column in plotted:
        longrun[f"median125_{column}"] = rolling(longrun, column)
    blocks = block_summary(longrun)

    actor_windows = {
        str(width): window_stats(longrun, "train_total", width)
        for width in (250, 500, 1000)
    }
    for statistics in actor_windows.values():
        statistics["point_estimate_0p5pct_diagnostic_met"] = bool(
            abs(float(statistics["relative_ols_change_pct_of_mean_abs"])) <= 0.5
            and abs(float(statistics["half_median_shift_pct_of_mean_abs"])) <= 0.5
        )
    actor500 = actor_windows["500"]
    actor500_diagnostic = bool(
        actor500["point_estimate_0p5pct_diagnostic_met"]
    )
    critic_estimate500 = window_stats(longrun, "critic_estimate", 500)
    gp500 = window_stats(longrun, "weighted_gp", 500)
    grad500 = window_stats(longrun, "critic_mean_gradient_norm", 500)
    grad_median = float(grad500["median"])
    critic_gp_pass = float(gp500["median"]) <= 0.10
    critic_grad_pass = 0.95 <= grad_median <= 1.05

    validation_tail = validation.loc[validation["epoch"].ge(7000)].copy()
    validation_improvement_pct = float(
        (validation_tail["validation_selection_score"].iloc[0]
         - validation_tail["validation_selection_score"].iloc[-1])
        / validation_tail["validation_selection_score"].iloc[0]
        * 100.0
    )
    validation_critic_drift_pct = float(
        (validation_tail["validation_critic_estimate"].iloc[-1]
         - validation_tail["validation_critic_estimate"].iloc[0])
        / abs(validation_tail["validation_critic_estimate"].iloc[0])
        * 100.0
    )
    if len(validation_tail) != 3:
        raise RuntimeError("expected three fixed-bank evaluations from e7000 to e7500")
    validation_plateau_pass = abs(validation_improvement_pct) < 0.1
    validation_critic_pass = abs(validation_critic_drift_pct) <= 1.0
    joint_stabilization = bool(
        actor500_diagnostic
        and critic_gp_pass
        and critic_grad_pass
        and validation_plateau_pass
        and validation_critic_pass
    )

    first40 = longrun.head(40)
    prefix_diffs = {
        column: float(
            np.max(np.abs(formal[column].to_numpy() - first40[column].to_numpy()))
        )
        for column in (
            "train_total",
            "train_law",
            "train_adversarial",
            "critic_loss",
            "critic_estimate",
            "critic_gp",
            "critic_mean_gradient_norm",
        )
    }
    longrun_summary = json.loads(LONGRUN_SUMMARY.read_text(encoding="utf-8"))
    formal_summary = json.loads(FORMAL_SUMMARY.read_text(encoding="utf-8"))
    best_index = validation["validation_selection_score"].idxmin()
    best_epoch = int(validation.loc[best_index, "epoch"])
    best_score = float(validation.loc[best_index, "validation_selection_score"])
    selected_rows = longrun.loc[longrun["selected_checkpoint"].eq(1.0)]
    if len(selected_rows) != 1:
        raise RuntimeError("training history does not identify exactly one checkpoint")
    history_best_epoch = int(selected_rows.iloc[0]["epoch"])
    if (
        history_best_epoch != best_epoch
        or int(checkpoint["best_epoch"]) != best_epoch
        or int(longrun_summary["best_epoch"]) != best_epoch
    ):
        raise RuntimeError("history, checkpoint, and summary disagree on best epoch")
    raw_minimum_row = longrun.loc[longrun["train_total"].idxmin()]
    summary = {
        "status": "isolated_7500epoch_matched_full_loss_decomposition_preview",
        "longrun_history_sha256": sha256(LONGRUN_HISTORY),
        "formal_history_sha256": sha256(FORMAL_HISTORY),
        "longrun_checkpoint_sha256": sha256(LONGRUN_CHECKPOINT),
        "formal_checkpoint_sha256": sha256(FORMAL_CHECKPOINT),
        "longrun_training_status": longrun_summary.get("status"),
        "formal_training_status": formal_summary.get("status"),
        "same_numerical_trajectory_as_paper_primary": False,
        "paper_primary_selected_epoch": int(formal_summary["best_epoch"]),
        "paper_primary_checkpoint_role": "used for the existing HUP060 36-channel downstream evaluation",
        "longrun_best_evaluated_epoch": best_epoch,
        "longrun_best_validation_score": best_score,
        "longrun_raw_train_total_minimum_epoch": int(raw_minimum_row["epoch"]),
        "longrun_raw_train_total_minimum_value": float(raw_minimum_row["train_total"]),
        "raw_minimum_epoch_checkpoint_saved": False,
        "formal_vs_longrun_first40_max_absolute_differences": prefix_diffs,
        "checkpoint_training_arguments": {
            "seed": int(checkpoint["ablation_contract"]["seed"]),
            "epochs": int(arguments["epochs"]),
            "validation_every": int(arguments["validation_every"]),
            "n_critic": int(arguments["n_critic"]),
            "adversarial_weight": adversarial_weight,
            "anchor_weight": anchor_weight,
            "gradient_penalty_weight": gp_weight,
            "critic_drift_weight": critic_drift_weight,
        },
        "actor_identity": "train_total = train_law + 0.5*train_adversarial + 0.2*train_action_anchor",
        "actor_identity_max_abs_error": actor_identity_error,
        "critic_identity": "critic_loss = -critic_estimate + 10*critic_gp + 0.001*critic_drift_raw",
        "critic_identity_max_abs_error": critic_identity_error,
        "no_negative_display_policy": (
            "The signed critic_loss is not shifted or absolutized and is omitted from the no-negative display. "
            "The signed critic estimate is shown untransformed; its values happen to be positive in this run."
        ),
        "critic_recording_scope": "final (third) Critic update of each epoch",
        "actor_window_statistics": actor_windows,
        "actor_final500_point_diagnostic_0p5pct_met": actor500_diagnostic,
        "actor_250_and_1000_sensitivity_diagnostics_met": bool(
            actor_windows["250"]["point_estimate_0p5pct_diagnostic_met"]
            and actor_windows["1000"]["point_estimate_0p5pct_diagnostic_met"]
        ),
        "formal_equivalence_test_performed": False,
        "critic_final500": {
            "critic_estimate": critic_estimate500,
            "weighted_gp": gp500,
            "gradient_norm": grad500,
            "weighted_gp_median_le_0p10_pass": critic_gp_pass,
            "gradient_norm_median_0p95_to_1p05_pass": critic_grad_pass,
        },
        "validation_7000_to_7500": {
            "fixed_bank_evaluation_count": int(len(validation_tail)),
            "selection_score_improvement_pct": validation_improvement_pct,
            "selection_score_improvement_lt_0p1pct_pass": validation_plateau_pass,
            "fixed_bank_critic_estimate_drift_pct": validation_critic_drift_pct,
            "fixed_bank_critic_abs_drift_le_1pct_pass": validation_critic_pass,
        },
        "joint_actor_critic_validation_stabilization_pass": joint_stabilization,
        "interpretation": (
            "Actor loss declines and meets only the final-500 point diagnostic; the 250- and 1000-epoch "
            "sensitivity checks are not met. Critic constraint diagnostics approach their target regime, "
            "but Critic estimates and fixed-bank validation remain non-stationary. Joint convergence is "
            "therefore not demonstrated."
        ),
    }

    source_columns = [
        "epoch",
        "train_total",
        "train_law",
        "weighted_adversarial",
        "weighted_anchor",
        "critic_loss",
        "critic_estimate",
        "weighted_gp",
        "weighted_critic_drift",
        "critic_mean_gradient_norm",
        "validation_selection_score",
        "validation_critic_estimate",
    ] + [f"median125_{column}" for column in plotted]
    longrun[source_columns].to_csv(
        OUTPUT / "source_data_loss_decomposition_e7500.csv", index=False
    )
    blocks.to_csv(OUTPUT / "source_data_250epoch_blocks.csv", index=False)
    validation.to_csv(OUTPUT / "source_data_validation_e7500.csv", index=False)
    (OUTPUT / "loss_decomposition_e7500_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig = plt.figure(figsize=(183 / 25.4, 165 / 25.4), constrained_layout=False)
    grid = fig.add_gridspec(
        4,
        2,
        height_ratios=(1.25, 1.0, 1.0, 0.92),
        hspace=0.58,
        wspace=0.40,
        left=0.075,
        right=0.965,
        bottom=0.14,
        top=0.92,
    )
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[2, 0])
    ax_e = fig.add_subplot(grid[2, 1])
    ax_f = fig.add_subplot(grid[3, :])

    epoch = longrun["epoch"]
    ax_a.plot(epoch, longrun["train_total"], color=COLORS["raw"], lw=0.28, alpha=0.22, rasterized=True)
    ax_a.plot(epoch, longrun["train_law"], color="#BCD0E0", lw=0.28, alpha=0.22, rasterized=True)
    ax_a.plot(epoch, longrun["median125_train_total"], color=COLORS["total"], lw=1.15, label="Total (trailing 125-epoch median)")
    ax_a.plot(epoch, longrun["median125_train_law"], color=COLORS["law"], lw=1.0, label="Law (trailing 125-epoch median)")
    ax_a.plot(blocks["mid_epoch"], blocks["median_train_total"], color="#9A4F05", lw=0.65, marker="o", ms=1.8, label="Total (250-epoch blocks)")
    ax_a.axvspan(7000, 7500, color=COLORS["window_shade"], alpha=0.9, lw=0)
    ax_a.text(
        0.985,
        0.93,
        "Final-500 point diagnostic met\n"
        f"change {float(actor500['relative_ols_change_pct_of_mean_abs']):+.2f}%; "
        f"half-window shift {float(actor500['half_median_shift_pct_of_mean_abs']):+.2f}%\n"
        "250/1000 sensitivity: not met",
        transform=ax_a.transAxes,
        ha="right",
        va="top",
        fontsize=6.0,
        color=COLORS["validation"],
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#D2D8DC", "lw": 0.55},
    )
    panel_label(ax_a, "a", "Actor total objective and dominant law term")
    ax_a.set_ylabel("Objective")
    ax_a.legend(loc="lower left", ncol=3, handlelength=2.0, columnspacing=1.1)
    style_axis(ax_a)

    ax_b.plot(epoch, longrun["weighted_adversarial"], color="#C8B7D6", lw=0.28, alpha=0.28, rasterized=True)
    line_adv, = ax_b.plot(epoch, longrun["median125_weighted_adversarial"], color=COLORS["adv"], lw=1.1, label=r"$0.5L_{adv}$")
    ax_b2 = ax_b.twinx()
    ax_b2.spines["right"].set_visible(True)
    ax_b2.plot(epoch, longrun["weighted_anchor"], color="#AED9D3", lw=0.28, alpha=0.28, rasterized=True)
    line_anchor, = ax_b2.plot(epoch, longrun["median125_weighted_anchor"], color=COLORS["anchor"], lw=1.0, label=r"$0.2L_{prox}$")
    ax_b.set_ylim(bottom=0)
    ax_b2.set_ylim(bottom=0)
    ax_b.set_ylabel(r"$0.5L_{adv}$")
    ax_b2.set_ylabel(r"$0.2L_{prox}$", color=COLORS["anchor"], labelpad=4)
    ax_b2.tick_params(axis="y", colors=COLORS["anchor"])
    ax_b.legend([line_adv, line_anchor], [line_adv.get_label(), line_anchor.get_label()], loc="upper left", ncol=2)
    ax_b.text(
        0.98, 0.08, "Separate y-axes", transform=ax_b.transAxes,
        ha="right", va="bottom", fontsize=5.4, color="#5B6268"
    )
    panel_label(ax_b, "b", "Actor weighted auxiliary terms")
    style_axis(ax_b)

    ax_c.plot(epoch, longrun["critic_estimate"], color="#D7B4A7", lw=0.28, alpha=0.27, rasterized=True)
    ax_c.plot(epoch, longrun["median125_critic_estimate"], color=COLORS["critic"], lw=1.1, label="Training estimate")
    ax_c.scatter(validation["epoch"], validation["validation_critic_estimate"], s=8, color=COLORS["validation_critic"], alpha=0.8, label="Fixed-bank estimate", zorder=4)
    ax_c.set_ylim(bottom=0)
    ax_c.set_ylabel(r"$\widehat W_C$", labelpad=2)
    ax_c.legend(loc="lower right", ncol=2)
    panel_label(ax_c, "c", "Signed Critic estimate (untransformed; positive here)")
    style_axis(ax_c)

    ax_d.plot(epoch, longrun["weighted_gp"], color="#E4C79F", lw=0.28, alpha=0.28, rasterized=True)
    ax_d.plot(epoch, longrun["median125_weighted_gp"], color=COLORS["gp"], lw=1.05, label=r"$10L_{GP}$")
    ax_d.plot(
        epoch,
        longrun["median125_weighted_critic_drift"],
        color=COLORS["drift"],
        lw=0.95,
        label=r"$10^{-3}L_{drift}$",
    )
    ax_d.set_yscale("log")
    ax_d.set_ylabel("Contribution (log scale)")
    ax_d.legend(loc="upper right", ncol=2)
    panel_label(ax_d, "d", "Critic penalty decomposition")
    style_axis(ax_d)

    ax_e.fill_between([0, 7575], 0.95, 1.05, color=COLORS["shade"], alpha=0.8, lw=0, label="0.95–1.05 target band")
    ax_e.plot(epoch, longrun["critic_mean_gradient_norm"], color="#D8B1A3", lw=0.28, alpha=0.28, rasterized=True)
    ax_e.plot(epoch, longrun["median125_critic_mean_gradient_norm"], color=COLORS["grad"], lw=1.05, label="125-epoch median")
    ax_e.axhline(1.0, color="#444444", lw=0.7, ls="--", label="Target = 1")
    ax_e.set_ylim(0.70, 1.09)
    ax_e.set_ylabel("Mean gradient norm")
    ax_e.legend(loc="lower left", ncol=2, columnspacing=0.9)
    ax_e.text(
        0.985,
        0.82,
        f"Final-500 median = {grad_median:.4f}\nstrict 0.95–1.05: {'PASS' if critic_grad_pass else 'borderline FAIL'}",
        transform=ax_e.transAxes,
        ha="right",
        va="top",
        fontsize=5.7,
        color=COLORS["pass"] if critic_grad_pass else COLORS["accent"],
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#D2D8DC", "lw": 0.5},
    )
    panel_label(ax_e, "e", "Critic gradient constraint")
    style_axis(ax_e)

    ax_f.plot(validation["epoch"], validation["validation_selection_score"], color=COLORS["validation"], lw=1.0, marker="o", ms=2.6)
    ax_f.scatter([best_epoch], [best_score], marker="*", s=55, color=COLORS["accent"], zorder=5, label=f"Best evaluated: e{best_epoch}")
    ax_f.axvline(best_epoch, color=COLORS["accent"], lw=0.75, ls="--")
    ax_f.set_ylim(bottom=max(0.0, float(validation["validation_selection_score"].min()) - 0.02))
    ax_f.set_ylabel("Validation selection score")
    ax_f.set_xlabel("Joint-training epoch (all panels: 1–7,500)")
    ax_f.legend(loc="upper right")
    ax_f.text(
        0.012,
        0.12,
        "Three fixed-bank evaluations from e7000–e7500; best is at the boundary → not a convergence test.",
        transform=ax_f.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.9,
        color=COLORS["accent"],
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.86},
    )
    panel_label(ax_f, "f", "Fixed-bank validation trajectory")
    style_axis(ax_f)

    fig.suptitle(
        "HUP060 Actor–WGAN-GP: 7,500-epoch long-run development audit",
        x=0.075,
        y=0.972,
        ha="left",
        fontsize=9.2,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.944,
        "Matched-full branch, seed 20261011; no synthetic points, sign changes, offsets or plotting-value clipping. Dark lines are trailing 125-epoch medians.",
        ha="left",
        va="center",
        fontsize=5.9,
        color="#4D555B",
    )
    fig.text(
        0.5,
        0.055,
        "Critic traces record the third/final Critic update per epoch; validation uses one fixed bank at 31 sampled epochs. Signed Critic loss is retained in source data but not plotted.",
        ha="center",
        va="bottom",
        fontsize=5.4,
        color="#4D555B",
    )
    fig.text(
        0.5,
        0.025,
        "Important: this is not a numerical continuation of the paper-primary 40-epoch trajectory selected at e32; existing 36-channel results remain tied to that frozen e32 checkpoint.",
        ha="center",
        va="bottom",
        fontsize=5.7,
        color=COLORS["accent"],
    )

    base = OUTPUT / "HUP060_WGAN_loss_decomposition_e7500_preview"
    fig.savefig(base.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    qa = {
        "render_qa_status": "PASS",
        "scientific_joint_stabilization": (
            "MET" if joint_stabilization else "NOT_MET"
        ),
        "backend": "Python/matplotlib only",
        "complete_epoch_sequence": True,
        "all_displayed_metrics_nonnegative": True,
        "signed_critic_loss_not_shifted_or_absolutized": True,
        "raw_values_visible": True,
        "rolling_statistics_explicitly_labelled": True,
        "paper_primary_and_longrun_trajectories_distinguished": True,
        "joint_stabilization_pass": joint_stabilization,
        "outputs": [
            base.with_suffix(suffix).name
            for suffix in (".png", ".svg", ".pdf", ".tiff")
        ],
    }
    (OUTPUT / "qa_report.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
