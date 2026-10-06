"""Fresh HUP060 component-removal comparison in the original six-panel layout.

Contract: a quantitative grid describes how independently retrained removal of
WGAN, the control-input heat kernel, or particle-deviation feedback changes
empirical law errors, nondirect-node recovery, actuator RMS and channel counts.
No component is presumed beneficial before results are available. Each arm has
one technical training seed and 1000 actor updates, not an equal wall-clock or
equal-energy budget. Output geometry follows the current manuscript: 170 x
178 mm, three rows by two columns, editable PDF/SVG, PNG and clean source CSVs.
Negative relative changes are preserved on a symmetric linear axis; no metric
is clipped or transformed to make Full appear superior.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import fitz

HERE = Path(__file__).resolve().parent
ORDER = ("free", "no_wgan", "no_graph_spread", "no_deviation", "full")
CONTROLLED = ORDER[1:]
ABLATIONS = ORDER[1:-1]
LABELS = {"free": "Free", "no_wgan": "−WGAN", "no_graph_spread": "−Graph",
          "no_deviation": "−Deviation", "full": "WGAN\nFull"}
SHORT = {**LABELS, "no_deviation": "−Dev."}
COLORS = {"free": "#8C8C8C", "no_wgan": "#D5962B", "no_graph_spread": "#8267A8",
          "no_deviation": "#3E9787", "full": "#2166AC"}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 9,
                         "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "axes.linewidth": .75, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 9})


def channel_summary(ax, frame, metric, ylabel):
    x = np.arange(len(ORDER), dtype=float)
    offsets = np.linspace(-.12, .12, 36)
    for position, variant in zip(x, ORDER):
        values = frame.loc[frame.variant.eq(variant)].sort_values("channel_index")[metric].to_numpy(float)
        assert len(values) == 36 and np.isfinite(values).all()
        ax.scatter(position + offsets, values, s=7, color=COLORS[variant], alpha=.28, linewidth=0, zorder=1)
        mean = float(values.mean())
        ax.plot([position - .22, position + .22], [mean, mean], color=COLORS[variant], lw=2.2,
                solid_capstyle="round", zorder=3)
        ax.text(position, mean + .012, f"{mean:.3f}", color=COLORS[variant], ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, [SHORT[v] for v in ORDER])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E6E6E6", lw=.5)
    ax.margins(x=.06, y=.12)
    ax.yaxis.set_major_locator(MaxNLocator(4))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, default=HERE / "ablations")
    p.add_argument("--output-dir", type=Path, default=HERE / "ablations/figures")
    args = p.parse_args()
    source, out = args.source_dir, args.output_dir
    required = [source / name for name in ("ablation_summary.csv", "ablation_per_channel.csv",
                                          "ablation_contrasts_vs_full.csv", "ablation_aggregation_qa.json")]
    if not all(path.is_file() for path in required):
        raise RuntimeError("Only completed fresh ablation aggregation can be plotted")
    hashes = {str(path): sha(path) for path in required}
    audit = json.loads(required[-1].read_text(encoding="utf-8"))
    assert audit["status"] == "completed_fresh_matched_component_removal_preview_not_adopted"
    assert audit["same_actor_update_budget"] == 1000 and audit["teacher_used"] is False
    summary = pd.read_csv(required[0]).set_index("variant")
    channel = pd.read_csv(required[1])
    contrast = pd.read_csv(required[2])
    full = summary.loc["full"]
    configure()
    fig = plt.figure(figsize=(170 / 25.4, 178 / 25.4))
    axes = [fig.add_axes([.120 + col * .465, .710 - row * .300, .335, .205])
            for row in range(3) for col in range(2)]
    a, b, c, d, e, f = axes
    channel_summary(a, channel, "time_w1", "Time-resolved $W_1$")
    channel_summary(b, channel, "occupation_w1", "Occupation $W_1$")
    y = np.arange(3, dtype=float)
    t_inc = np.asarray([100 * (summary.loc[v, "mean_time_w1"] / full.mean_time_w1 - 1) for v in ABLATIONS])
    o_inc = np.asarray([100 * (summary.loc[v, "mean_occupation_w1"] / full.mean_occupation_w1 - 1) for v in ABLATIONS])
    for row, left, right in zip(y, t_inc, o_inc):
        c.plot([left, right], [row, row], color="#B8B8B8", lw=1.2, zorder=1)
    c.scatter(t_inc, y, marker="o", s=26, color="#3D6FA3", label="Time", zorder=3)
    c.scatter(o_inc, y, marker="s", s=24, color="#B76542", label="Occupation", zorder=3)
    all_inc = np.concatenate([t_inc, o_inc])
    linear = bool(np.any(all_inc <= 0))
    if linear:
        limit = max(1., float(np.max(np.abs(all_inc))) * 1.45)
        c.set_xlim(-limit, limit)
        c.axvline(0, color="#7C7C7C", lw=.6, ls="--", zorder=0)
        c.set_xlabel("Change vs WGAN Full (%)\nsymmetric linear scale")
        c.xaxis.set_major_locator(MaxNLocator(5))
        for values, dy, color in ((t_inc, -.13, "#3D6FA3"), (o_inc, .18, "#B76542")):
            for row, value in zip(y, values):
                offset = limit * .02 * (1 if value >= 0 else -1)
                c.text(value + offset, row + dy, f"{value:+.2f}%" if abs(value) < 1 else f"{value:+.1f}%",
                       color=color, fontsize=9, ha="left" if value >= 0 else "right")
    else:
        c.set_xscale("log")
        c.set_xlim(min(.075, float(all_inc.min()) / 1.7), max(410., float(all_inc.max()) * 1.7))
        c.set_xlabel("Increase vs WGAN Full (%)\nlogarithmic scale")
        for values, dy, color in ((t_inc, -.13, "#3D6FA3"), (o_inc, .18, "#B76542")):
            for row, value in zip(y, values):
                rightmost = value >= 100
                c.text(value / 1.08 if rightmost else value * 1.08, row + dy,
                       f"{value:.2f}%" if value < 1 else f"{value:.1f}%",
                       color=color, fontsize=9, ha="right" if rightmost else "left")
    c.set_yticks(y, [LABELS[v] for v in ABLATIONS])
    c.set_ylim(2.35, -.35)
    # LogLocator proposes off-range ticks too. They are not rendered by the
    # axis, but their phantom text boxes can falsely trip the canvas audit.
    # Keep the same in-range ticks without changing limits or scientific data.
    if not linear:
        low, high = c.get_xlim()
        c.set_xticks([tick for tick in c.get_xticks() if low <= tick <= high])
    c.grid(axis="x", color="#E6E6E6", lw=.5, which="both")
    c.legend(loc="upper right", fontsize=9, handlelength=1)

    recovery_cols = ("mean_time_w1_recovery_percent", "mean_occupation_w1_recovery_percent",
                     "unselected_time_w1_recovery_percent", "unselected_occupation_w1_recovery_percent")
    recovery = summary.loc[list(CONTROLLED), list(recovery_cols)].to_numpy(float)
    if recovery.min() < 0:
        limit = max(75., float(np.ceil(np.max(np.abs(recovery)) / 5) * 5))
        image = d.imshow(recovery, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
        color_limits = [-limit, limit]
    else:
        limit = max(75., float(np.ceil(recovery.max() / 5) * 5))
        image = d.imshow(recovery, cmap="Blues", vmin=0, vmax=limit, aspect="auto")
        color_limits = [0., limit]
    d.set_xticks(np.arange(4), ["All\ntime", "All\noccup.", "Unsel.\ntime", "Unsel.\noccup."])
    d.set_yticks(np.arange(4), [LABELS[v] for v in CONTROLLED])
    for row in range(4):
        for col in range(4):
            value = recovery[row, col]
            d.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=9,
                   color="white" if abs(value) > limit * .56 else "#222222")
    d.set_title("Recovery vs free (%)", loc="left", fontsize=9)
    d.spines[:].set_visible(False)

    scatter_xy = []
    scatters = []
    for variant in CONTROLLED:
        row = summary.loc[variant]
        xval = float(row.command_rms)
        yval = float(row.mean_time_w1)
        scatter_xy.append([xval, yval])
        scatters.append(e.scatter(xval, yval, s=105 if variant == "no_wgan" else (40 if variant == "full" else 30),
                                  marker="*" if variant == "no_wgan" else "o", color=COLORS[variant],
                                  edgecolor="white", lw=.6, zorder=4 if variant == "no_wgan" else 3))
    scatter_xy = np.asarray(scatter_xy)
    dx = max(float(np.ptp(scatter_xy[:, 0])), .04)
    dy = max(float(np.ptp(scatter_xy[:, 1])), .015)
    # Reserve upper whitespace for the original shared legend, rather than
    # visually shifting any nearly identical data point.
    e.set_xlim(max(0., float(scatter_xy[:, 0].min()) - .22 * dx), float(scatter_xy[:, 0].max()) + .30 * dx)
    e.set_ylim(max(0., float(scatter_xy[:, 1].min()) - .25 * dy), float(scatter_xy[:, 1].max()) + .60 * dy)
    e.set_xlabel("Command RMS")
    e.set_ylabel("Mean time-resolved $W_1$")
    e.yaxis.set_major_locator(MaxNLocator(4))
    e.xaxis.set_major_locator(MaxNLocator(4))
    e.grid(color="#E6E6E6", lw=.5)
    e.legend(scatters, [LABELS[v].replace("\n", " ") for v in CONTROLLED], loc="upper right",
             fontsize=9, handlelength=1, labelspacing=.35, scatterpoints=1)

    counts_t = [int((contrast.loc[contrast.ablation.eq(v), "delta_time_w1_vs_full"] > 0).sum()) for v in ABLATIONS]
    counts_o = [int((contrast.loc[contrast.ablation.eq(v), "delta_occupation_w1_vs_full"] > 0).sum()) for v in ABLATIONS]
    x = np.arange(3, dtype=float)
    width = .34
    f.bar(x - width / 2, counts_t, width, color="#6F92B8", label="Time")
    f.bar(x + width / 2, counts_o, width, color="#C98565", label="Occupation")
    for pos, values, lift in ((x - width / 2, counts_t, 4.0), (x + width / 2, counts_o, .10)):
        for position, value in zip(pos, values):
            f.text(position, value + lift, f"{value}/36", ha="center", fontsize=9)
    f.set_xticks(x, [SHORT[v] for v in ABLATIONS])
    f.set_ylim(0, 43)
    f.set_ylabel("Contacts worse than WGAN Full")
    f.yaxis.set_major_locator(MaxNLocator(4))
    f.grid(axis="y", color="#E6E6E6", lw=.5)
    f.legend(loc="lower center", bbox_to_anchor=(.5, 1.015), fontsize=9, ncol=2,
             handlelength=1, columnspacing=.7)
    for label, ax in zip("abcdef", axes):
        ax.text(-.17, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=10,
                ha="left", va="bottom")
    fig.text(.09, .984, "HUP060 run-02: fresh-controller component-removal audit", va="top", ha="left",
             fontsize=10, fontweight="bold")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text(): continue
        box = text.get_window_extent(renderer)
        if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.x1 + .5 or box.y1 > fig.bbox.y1 + .5:
            outside.append(text.get_text())
    assert not outside, f"Text extends outside the figure: {outside}"
    out.mkdir(parents=True, exist_ok=True)
    base = out / "HUP060_fresh_WGANGP_matched_ablation_original_style"
    fig.savefig(base.with_suffix(".pdf"), metadata={"Creator": "Python/matplotlib; fresh matched component-removal results", "CreationDate": None, "ModDate": None})
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".png"), dpi=300)
    plt.close(fig)
    pdf = fitz.open(base.with_suffix(".pdf")); page = pdf[0]
    assert len(pdf) == 1 and bool(page.get_fonts())
    dims = [page.rect.width / 72 * 25.4, page.rect.height / 72 * 25.4]
    assert np.allclose(dims, [170, 178], rtol=0, atol=.01)
    page.get_pixmap(matrix=fitz.Matrix(220/72, 220/72), alpha=False).save(str(out / (base.stem + "_pdf_render.png")))
    pdf.close()
    qa = {
        "status": "fresh_ablation_original_style_preview_not_adopted", "archetype": "quantitative grid",
        "claim": "Matched from-scratch component removals quantify changes in law error and input cost without assuming Full is always best.",
        "input_sha256": hashes, "checkpoint_sha256": audit["checkpoint_sha256"],
        "selected_epochs": audit["selected_epochs"], "same_full_as_control_and_loss_figures": True,
        "same_actor_update_budget": 1000, "training_seeds_per_arm": 1,
        "channel_dots": "36 correlated channels within one patient; not independent patient or training replicates",
        "panel_c_axis": "symmetric linear" if linear else "logarithmic positive-only range",
        "panel_c_axis_change_reason": "Preserve negative or zero changes without dropping, offsetting or taking absolute values" if linear else "All empirical changes are strictly positive",
        "panel_c_time_changes_percent": t_inc.tolist(), "panel_c_occupation_changes_percent": o_inc.tolist(),
        "panel_d_recovery_percent": recovery.tolist(), "panel_d_color_limits": color_limits,
        "panel_e_command_rms_time_w1": scatter_xy.tolist(), "panel_e_no_wgan_marker": "orange five-pointed star; coordinates unchanged",
        "panel_e_near_coincident_points_shifted": False,
        "panel_f_time_counts": counts_t, "panel_f_occupation_counts": counts_o,
        "dimensions_mm": dims, "editable_text": True, "text_outside_canvas": outside,
        "caption": (
            "HUP060 run-02: component-removal audit of freshly trained controllers. Full and the three "
            "removal arms start from the same neutral actor initialization and use the same frozen Graph-RC "
            "plant, 13-contact mask, run-01 fitting/validation references, 1000 actor updates, and paired "
            "run-02 evaluation bank. Each arm's checkpoint is selected using the same predetermined run-01 "
            "validation score; Full is the identical selected policy used in the control and loss figures. "
            "a,b, Channel-wise time-resolved and occupation-law W1; dots denote correlated within-patient "
            "contacts and colored horizontal segments their arithmetic mean. c, Relative changes from Full "
            "(positive: larger distance; negative: smaller distance). d, Recovery relative to the free plant "
            "for all 36 contacts and 23 contacts without a dedicated actuator. e, Command RMS versus mean "
            "time-resolved W1; the orange star denotes removal of WGAN and points are not displaced. "
            "f, Number of contacts with larger W1 than Full. Removal of graph spread affects only the input "
            "heat kernel, not the frozen plant's dynamical coupling. Removal of deviation feedback yields "
            "a particle-common action. One training seed per arm is shown; the comparison matches actor-update "
            "budgets, not input energy or wall-clock time."
        ),
        "original_inputs_unchanged": all(sha(path) == h for path, h in hashes.items()),
        "overleaf_modified": False,
    }
    assert qa["original_inputs_unchanged"]
    (out / "ablation_figure_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(base), "panel_c_axis": qa["panel_c_axis"], "selected_epochs": qa["selected_epochs"]}, indent=2))


if __name__ == "__main__": main()
