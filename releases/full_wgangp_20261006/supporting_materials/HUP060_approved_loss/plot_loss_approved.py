"""Draw actual fresh-controller losses in the manuscript's eight-panel layout.

Figure contract: this quantitative grid exposes the actor and critic objective
identities from one actual training run, without forcing their signed losses to
have a common limit. Panels a/c/e/g show the actor total and its active/removed
terms; b/d/f/h show the critic total and its Wasserstein, GP and drift terms.
The output is a 170 x 174 mm editable PDF/SVG plus PNG and per-update source CSV.
Raw values are unshifted; the only display summary is a trailing 125-update
median (minimum 40 observations). No uncertainty is inferred from one seed.

The removed proximity term is an explicit zero derived from the verified
training contract, not a purported observed loss. The drift is reconstructed
from the logged critic identity, not independently measured. No input files,
manuscript sources, or original figures are modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import fitz
import torch


HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE.parent / ".fresh_wgangp_20261006/runs/fresh_seed20261011_e1000"
COLORS = {
    "raw": "#C6CBD0", "actor_total": "#D97706", "actor_law": "#3D6F9E",
    "actor_adv": "#7A5195", "actor_prox": "#2A9D8F",
    "critic_total": "#A94232", "critic_w": "#C45A32", "critic_gp": "#C77C29",
    "grad_norm": "#317873", "critic_drift": "#667788", "zero": "#4C5358",
    "grid": "#E4E8EB", "window": "#E7EEF6", "selected": "#818991",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def configure() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .75,
        "legend.frameon": False, "legend.fontsize": 9,
        "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.facecolor": "white",
    })


def style(ax, total_epochs: int) -> None:
    ax.grid(axis="y", color=COLORS["grid"], lw=.45)
    ax.set_axisbelow(True)
    ax.set_xlim(0, total_epochs * 1.01)
    ax.set_xticks(np.linspace(0, total_epochs, 5))
    ax.yaxis.set_major_locator(MaxNLocator(4))


def draw_trace(ax, epoch, raw, smooth, color, total_epochs, zero=False):
    ax.plot(epoch, raw, color=COLORS["raw"], lw=.28, alpha=.28,
            rasterized=True)
    ax.plot(epoch, smooth, color=color, lw=1.15)
    if zero:
        ax.axhline(0, color=COLORS["zero"], lw=.7, ls="--")
    style(ax, total_epochs)


def assert_bounds(ax, raw, column):
    lower, upper = ax.get_ylim()
    values = np.asarray(raw, dtype=float)
    assert np.isfinite(values).all(), f"nonfinite raw loss {column}"
    assert values.min() >= lower - 1e-12 and values.max() <= upper + 1e-12, (
        f"raw loss {column} is clipped by y-axis"
    )
    return {"minimum_raw": float(values.min()), "maximum_raw": float(values.max()),
            "axis_limits": [float(lower), float(upper)], "all_raw_values_visible": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=HERE / "figures/loss")
    parser.add_argument("--subject", default="HUP060")
    args = parser.parse_args()
    run, out = args.run_dir, args.output_dir
    history = run / "training_history.csv"
    summary_path = run / "training_summary.json"
    contract_path = run / "training_contract.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    input_hashes = {str(path): sha(path) for path in (history, summary_path, contract_path)}
    checkpoint_path = run / "frozen_actor_wgan.pt"
    assert sha(checkpoint_path) == summary["checkpoint_sha256"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["training_contract"] == contract
    selected = int(summary.get("best_epoch", summary.get("selected_update")))
    assert int(checkpoint.get("best_epoch", checkpoint.get("best_update"))) == selected
    input_hashes[str(checkpoint_path)] = sha(checkpoint_path)
    total_epochs = int(summary.get("trained_epochs", summary.get("trained_updates")))
    history_frame = pd.read_csv(history)
    counter_column = "epoch" if "epoch" in history_frame.columns else "update"
    unit = "epoch" if counter_column == "epoch" else "update"
    frame = history_frame.loc[history_frame[counter_column].gt(0)].copy()
    assert np.array_equal(frame[counter_column], np.arange(1, total_epochs + 1))
    assert float(contract["teacher_proximity_weight"]) == 0
    assert summary.get("teacher_checkpoint_loaded", summary.get("teacher_loaded")) is False
    if unit == "update":
        assert contract["teacher_or_actor_checkpoint_loaded"] is False
        assert contract["update_unit"] == "one rotating fit context; not a whole-data epoch"
    adv_weight = float(contract["adv_weight"])
    gp_weight = float(contract.get("gradient_penalty", contract.get("gp_weight")))
    drift_weight = float(contract["critic_drift"])
    frame["actor_law_component"] = frame.train_law
    frame["actor_adversarial_component"] = adv_weight * frame.train_adversarial
    frame["actor_proximal_component"] = 0.0
    frame["teacher_proximity_removed_marker"] = 1
    frame["critic_wasserstein_component"] = -frame.critic_estimate
    frame["critic_gp_component"] = gp_weight * frame.critic_gp
    frame["critic_drift_component"] = (
        frame.critic_loss + frame.critic_estimate - gp_weight * frame.critic_gp
    )
    frame["selected_checkpoint"] = frame[counter_column].eq(selected).astype(int)
    actor_error = float(np.max(np.abs(
        frame.train_total - frame.actor_law_component - frame.actor_adversarial_component
    )))
    critic_error = float(np.max(np.abs(
        frame.critic_loss - frame.critic_wasserstein_component
        - frame.critic_gp_component - frame.critic_drift_component
    )))
    assert actor_error < 1e-10 and critic_error < 1e-10
    specs = [
        ("train_total", "actor_total", r"$L_A$", "a  Actor total"),
        ("critic_loss", "critic_total", r"$L_C$", "b  Signed critic total"),
        ("actor_law_component", "actor_law", r"$L_{law}$", "c  Empirical-law term"),
        ("critic_wasserstein_component", "critic_w", r"$-\widehat W_C$", "d  Wasserstein term"),
        ("actor_adversarial_component", "actor_adv", rf"${adv_weight:g}L_{{adv}}$", "e  Adversarial term"),
        ("critic_gp_component", "critic_gp", rf"${gp_weight:g}L_{{GP}}$", "f  Gradient penalty / norm"),
        ("actor_proximal_component", "actor_prox", r"$L_{prox}$ (disabled)", "g  Teacher proximity (disabled)"),
        ("critic_drift_component", "critic_drift", r"$10^{-3}L_{drift}$", "h  Reconstructed drift"),
    ]
    plotted_columns = [s[0] for s in specs] + ["critic_input_gradient_norm"]
    for col in plotted_columns:
        frame["median125_" + col] = frame[col].rolling(125, min_periods=40).median()

    configure()
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 2, figsize=(170 / 25.4, 174 / 25.4), sharex=True)
    bounds = {}
    epoch = frame[counter_column]
    twin = None
    for i, (column, color, ylabel, heading) in enumerate(specs):
        ax = axes.flat[i]
        row, col = divmod(i, 2)
        ax.set_position([.105 + col * .490, .704 - row * .199, .310, .137])
        draw_trace(ax, epoch, frame[column], frame["median125_" + column],
                   COLORS[color], total_epochs, zero=i in (1, 3))
        ax.set_ylabel(ylabel + (" (target 0)" if i == 5 else ""))
        ax.set_title(heading, loc="left", fontweight="bold", pad=6)
        ax.axvline(selected, color=COLORS["selected"], lw=.65, ls=(0, (2, 3)), zorder=0)
        if i in (0, 1):
            ax.axvspan(max(0, total_epochs - 500), total_epochs,
                       color=COLORS["window"], alpha=.9, lw=0, zorder=-1)
        if i in (4, 5, 7) and frame[column].min() >= 0:
            ax.set_ylim(bottom=0)
        if i == 6:
            # This is the verified absence of a term, not a fabricated loss.
            ax.set_ylim(-.005, .055)
            ax.set_yticks([0, .025, .05])
        if i == 7:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
        if i == 5:
            twin = ax.twinx()
            twin.set_position(ax.get_position())
            twin.plot(epoch, frame.critic_input_gradient_norm, color="#B8D5D2",
                      lw=.25, alpha=.24, rasterized=True)
            twin.plot(epoch, frame.median125_critic_input_gradient_norm,
                      color=COLORS["grad_norm"], lw=1.05)
            twin.axhline(1, color=COLORS["grad_norm"], lw=.75, ls="--")
            twin.set_ylabel("Gradient norm (target 1)", color=COLORS["grad_norm"])
            twin.tick_params(axis="y", colors=COLORS["grad_norm"])
            twin.yaxis.set_major_locator(MaxNLocator(4))
            twin.spines["right"].set_visible(True)
            twin.spines["right"].set_color(COLORS["grad_norm"])
            twin.grid(False)
        ax.tick_params(axis="x", labelbottom=row == 3)
        if row == 3:
            ax.set_xlabel("Actor update")
        bounds[column] = assert_bounds(ax, frame[column], column)
    bounds["critic_input_gradient_norm"] = assert_bounds(
        twin, frame.critic_input_gradient_norm, "critic_input_gradient_norm"
    )
    fig.legend(
        [Line2D([], [], color=COLORS["raw"], lw=1), Line2D([], [], color="#444444", lw=1.15)],
        ["Raw update value", "Trailing 125-update median"], loc="upper center",
        bbox_to_anchor=(.54, .933), ncol=2, fontsize=9,
    )
    fig.text(.09, .984, args.subject + ": from-scratch WGAN-GP training loss",
             va="top", ha="left", fontsize=10, fontweight="bold")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        bbox = text.get_window_extent(renderer)
        if bbox.x0 < -.5 or bbox.y0 < -.5 or bbox.x1 > fig.bbox.x1 + .5 or bbox.y1 > fig.bbox.y1 + .5:
            outside.append(text.get_text())
    assert not outside, f"text outside figure: {outside}"
    base = out / (args.subject + "_WGANGP_loss_original_style_" + ("e" if unit == "epoch" else "u") + str(total_epochs))
    fig.savefig(base.with_suffix(".pdf"), metadata={
        "Creator": "Python/matplotlib; actual fresh-controller training history",
        "CreationDate": None, "ModDate": None,
    })
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".png"), dpi=300)
    plt.close(fig)
    source_csv = out / (args.subject + "_WGANGP_loss_source.csv")
    frame.to_csv(source_csv, index=False)
    pdf = fitz.open(base.with_suffix(".pdf"))
    page = pdf[0]
    spans = [s for block in page.get_text("dict")["blocks"]
             for line in block.get("lines", []) for s in line["spans"] if s["text"].strip()]
    width = page.rect.width / 72 * 25.4
    height = page.rect.height / 72 * 25.4
    assert abs(width - 170) < .01 and abs(height - 174) < .01
    assert len(pdf) == 1 and bool(page.get_fonts())
    page.get_pixmap(matrix=fitz.Matrix(220 / 72, 220 / 72), alpha=False).save(
        str(out / (base.stem + "_pdf_render.png"))
    )
    font_sizes = [s["size"] for s in spans if s["size"] >= 8.8]
    pdf.close()
    qa = {
        "status": "complete_preview_not_adopted_in_manuscript",
        "subject": args.subject,
        "archetype": "quantitative grid",
        "claim": "Actual actor and signed critic totals are shown with their objective components, without artificial convergence manipulation.",
        "input_sha256": input_hashes,
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "weights_verified_against_frozen_checkpoint": True,
        "source_data_sha256": sha(source_csv),
        "training_counter_column": counter_column,
        "training_unit": contract.get("update_unit", "one actor update per logged epoch"),
        "trained_actor_updates": total_epochs,
        "selected_checkpoint_update": selected,
        **({"trained_epochs": total_epochs, "selected_checkpoint_epoch": selected} if unit == "epoch" else {}),
        "teacher_checkpoint_loaded": False,
        "teacher_proximity_term": {
            "removed": True, "weight": 0.0,
            "displayed_zero_source": "verified training contract; this term is absent, not separately logged",
        },
        "weights": {"adversarial": adv_weight, "gp": gp_weight, "drift": drift_weight},
        "actor_identity": "L_A = L_law + 0.5 L_adv; teacher proximity absent",
        "critic_identity": "L_C = -W_hat_C + 10 L_GP + 0.001 L_drift",
        "actor_identity_max_abs_error": actor_error,
        "critic_identity_max_abs_error": critic_error,
        "drift_source": "critic_loss + critic_estimate - 10*critic_gp; reconstructed weighted contribution",
        "critic_recording_scope": "final (third) critic audit per actor update; recorded before its optimizer step",
        "actor_critic_timing": "actor adversarial term is evaluated after the critic optimizer step and need not equal the preceding logged critic estimate",
        "smoothing": {"kind": "trailing median", "window_actor_updates": 125, "counter_unit": unit, "minimum_observations": 40,
                      "first_colored_median_update": 40, "raw_values_retained": True,
                      "display_only_not_a_new_observed_loss": True},
        "per_panel_visibility": bounds,
        "caption": (
            f"{args.subject}: training-loss decomposition for the from-scratch WGAN-GP controller. "
            f"Training covers {total_epochs} actor updates; the frozen controller was selected at update {selected} "
            "using the predetermined validation score (vertical dashed lines). "
            f"The horizontal axis denotes {unit}s"
            + (" (one rotating fit context per update, not a whole-data epoch). " if unit == "update" else ". ") +
            "Gray curves show raw updates; "
            "colored curves are trailing 125-update medians, shown after 40 observations. The shaded region "
            "in a and b denotes the final 500 updates. Actor total is empirical-law loss plus 0.5 times the "
            "adversarial loss. The teacher-proximity term in g was removed and is therefore zero. The critic "
            "total is the signed negative Wasserstein estimate plus 10 times the gradient penalty and 0.001 "
            "times the drift penalty. The weighted drift in h is reconstructed from this identity. Panel f "
            "shows the gradient penalty (left; target zero) and mean input-gradient norm (right; target one). "
            "All raw values remain within the plotted ranges; medians are display summaries, not separately "
            "additive components. One technical training seed is shown; no cross-seed interval is claimed."
        ),
        "dimensions_mm": [width, height],
        "axes_geometry": [list(ax.get_position().bounds) for ax in axes.flat],
        "editable_text": True,
        "minimum_ordinary_font_pt": min(font_sizes),
        "text_outside_canvas": outside,
        "visual_qa": {"text_bounds_check_passed": True,
                      "human_or_agent_image_inspection_complete": False,
                      "image_inspection_required_before_author_confirmation": True,
                      "eight_panel_original_positions_preserved": True},
        "overleaf_modified": False,
        "manuscript_modified": False,
        "original_inputs_unchanged": all(sha(Path(p)) == h for p, h in input_hashes.items()),
        "outputs": [str(base.with_suffix(suffix)) for suffix in (".pdf", ".png", ".svg")] + [str(source_csv)],
    }
    assert qa["original_inputs_unchanged"]
    dump(out / (args.subject + "_WGANGP_loss_qa.json"), qa)
    print(json.dumps({"outputs": qa["outputs"], "qa": str(out / (args.subject + "_WGANGP_loss_qa.json")),
                      "actor_error": actor_error, "critic_error": critic_error,
                      "selected_update": selected, "counter_unit": unit, "dimensions_mm": qa["dimensions_mm"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
