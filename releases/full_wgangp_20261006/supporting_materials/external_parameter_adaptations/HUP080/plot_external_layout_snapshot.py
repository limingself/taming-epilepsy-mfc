#!/usr/bin/env python
"""Original-layout external occupation-law plates after completed evaluation.

Contract: every contact shows recorded ictal, free prediction, patient-specific
reference and newly controlled model laws. The fixed original plotting grids,
6-column pagination, millimetre axis geometry and existing standardization are
preserved. The display context is predeclared context 7 / CRN bank 0, never
selected by viewing the new outer result. No distances are printed in panels.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, wasserstein_distance


plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 9, "axes.linewidth": .8, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "xtick.major.width": .8, "ytick.major.width": .8,
})
SUBJECTS = {"HUP065": {"n": 64, "m": 23, "pages": (36, 28)},
            "HUP080": {"n": 96, "m": 76, "pages": (36, 36, 24)}}
SERIES = {
    "O": ("observed_standardized", "Recorded ictal", "#272727", .70, "-", .78),
    "F": ("free_standardized", "Free prediction", "#9A9A9A", .75, ":", .82),
    "R": ("reference_standardized", "Preictal reference", "#3C8D62", .90, "-.", .92),
    "C": ("controlled_standardized", "Controlled", "#2166AC", 1.00, "-", 1.00),
}
BANDWIDTH = .16


def sha(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def clean_contact(value: str) -> str:
    return value.removeprefix("EEG ").removesuffix("-Ref")


def density(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    samples = np.asarray(values, dtype=float).ravel()
    sd = float(samples.std(ddof=1))
    if sd < 1e-10:
        z = (x-samples.mean())/BANDWIDTH
        return np.exp(-.5*z*z)/(BANDWIDTH*np.sqrt(2*np.pi))
    return gaussian_kde(samples, bw_method=BANDWIDTH/sd)(x)


def time_w1(values: np.ndarray, ref: np.ndarray) -> float:
    return float(np.mean([wasserstein_distance(values[:, t], ref[:, t]) for t in range(values.shape[1])]))


def load_completed(subject: str, run: Path):
    profile = SUBJECTS[subject]
    training_path = run/"training_summary.json"
    evaluation_path = run/"outer_posthoc_amendment/evaluation_summary.json"
    # Critically, no NPZ outcome file is opened before completed summaries and
    # checkpoint binding have been verified.
    if not training_path.is_file() or not evaluation_path.is_file():
        raise FileNotFoundError("Completed training and outer evaluation summaries are required before reading display data.")
    training = json.loads(training_path.read_text("utf-8"))
    evaluation = json.loads(evaluation_path.read_text("utf-8"))
    if training.get("status") != "completed_fresh_hybrid_training" or training.get("subject") != subject:
        raise ValueError("Training is not complete for the requested subject.")
    if evaluation.get("subject") != subject or evaluation.get("context_bank_evaluations") != 24:
        raise ValueError("Outer evaluation is not complete over the predeclared 8 x 3 contexts/banks.")
    if evaluation.get("outer_used_for_training_or_checkpoint_selection") is not False:
        raise ValueError("Outer arrays must not have been used for selection.")
    if evaluation.get("ctx6_used_for_reselection") is not False:
        raise ValueError("Terminal-context veto must not reselect the policy.")
    if training.get("teacher_loaded") is not False or training.get("outer_opened") is not False:
        raise ValueError("Expected teacher-free, outer-sealed training contract.")
    checkpoint = run/"frozen_actor_wgan.pt"
    checkpoint_sha = sha(checkpoint)
    if training.get("checkpoint_sha256") != checkpoint_sha or evaluation.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("Training and evaluation checkpoints are not the same frozen policy.")
    if evaluation.get("channels") != profile["n"] or evaluation.get("direct_actuators") != profile["m"]:
        raise ValueError("Official channel/actuator dimensions changed.")
    display_path = run/"outer_posthoc_amendment/display_context_rollout.npz"
    with np.load(display_path, allow_pickle=False) as p:
        arrays = {key: np.asarray(p[key]) for key in p.files}
    if int(arrays["display_context_index"]) != 7 or int(arrays["display_crn_bank"]) != 0:
        raise ValueError("Wrong predeclared display context / bank.")
    channels = arrays["channels"].astype(str)
    if channels.shape != (profile["n"],) or len(set(channels)) != profile["n"]:
        raise ValueError("Invalid unique contact names.")
    mask = arrays["direct_mask"].astype(bool)
    if mask.shape != (profile["n"],) or int(mask.sum()) != profile["m"]:
        raise ValueError("Actuator mask differs from the official mask.")
    if not np.array_equal(np.flatnonzero(mask), arrays["selected_indices"]):
        raise ValueError("Selected indices disagree with the official actuator mask.")
    observed_shape = arrays["observed_standardized"].shape
    if observed_shape != (256, profile["n"]):
        raise ValueError("Observed display must have 256 samples per contact.")
    for code, style in SERIES.items():
        arr = arrays[style[0]]
        if not np.isfinite(arr).all(): raise ValueError(f"Nonfinite {code} display samples.")
        if code != "O" and (arr.ndim != 3 or arr.shape[1:] != observed_shape):
            raise ValueError(f"Invalid ensemble shape for {code}.")
    if arrays["free_standardized"].shape != arrays["controlled_standardized"].shape:
        raise ValueError("Free and controlled particle ensembles are not paired.")
    return arrays, training, evaluation, {"checkpoint_sha256": checkpoint_sha,
        "display_npz_sha256": sha(display_path), "training_summary_sha256": sha(training_path),
        "evaluation_summary_sha256": sha(evaluation_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", choices=tuple(SUBJECTS), required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--original-density-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = SUBJECTS[args.subject]
    arrays, training, evaluation, hashes = load_completed(args.subject, args.run_directory)
    old = pd.read_csv(args.original_density_source, encoding="utf-8-sig")
    if set(old.subject_id.astype(str)) != {args.subject} or set(old.series_code.astype(str)) != set(SERIES):
        raise ValueError("Original density source belongs to a different subject or curve contract.")
    names = old[["channel_index", "channel"]].drop_duplicates().sort_values("channel_index")
    if not np.array_equal(names.channel_index.to_numpy(), np.arange(profile["n"])):
        raise ValueError("Original source has a different channel indexing.")
    channels = arrays["channels"].astype(str)
    if not np.array_equal(names.channel.astype(str).to_numpy(), channels):
        raise ValueError("Original and new contact labels/order differ.")
    grids, old_curve = {}, {}
    for c in range(profile["n"]):
        channel_table = old[old.channel_index == c]
        grid = channel_table[channel_table.series_code == "O"].standardized_amplitude.to_numpy(dtype=float)
        if len(grid) != 240 or np.any(np.diff(grid) <= 0):
            raise ValueError("Expected unchanged 240-point original x-grid.")
        for code in SERIES:
            frame = channel_table[channel_table.series_code == code]
            if not np.array_equal(frame.standardized_amplitude.to_numpy(dtype=float), grid):
                raise ValueError("Original series do not share a grid.")
            old_curve[(c, code)] = frame.density.to_numpy(dtype=float)
        grids[c] = grid
    x_min = float(old.standardized_amplitude.min()); x_max = float(old.standardized_amplitude.max())
    args.output.mkdir(parents=True, exist_ok=True)
    curves = {(c, code): density(arrays[style[0]][..., c], grids[c])
              for c in range(profile["n"]) for code, style in SERIES.items()}
    density_rows = []
    metric_rows = []
    ref = arrays["reference_standardized"]
    free = arrays["free_standardized"]; controlled = arrays["controlled_standardized"]
    for c in range(profile["n"]):
        focc = float(wasserstein_distance(free[..., c].ravel(), ref[..., c].ravel()))
        cocc = float(wasserstein_distance(controlled[..., c].ravel(), ref[..., c].ravel()))
        ftw = time_w1(free[..., c], ref[..., c]); ctw = time_w1(controlled[..., c], ref[..., c])
        metric_rows.append({"subject": args.subject, "channel_index": c, "channel": str(channels[c]),
            "direct_actuated": bool(arrays["direct_mask"][c]), "display_context": 7, "crn_bank": 0,
            "time_w1_free": ftw, "time_w1_controlled": ctw,
            "occupation_w1_free": focc, "occupation_w1_controlled": cocc,
            "time_w1_reduction_percent": 100*(ftw-ctw)/ftw if ftw else 0.,
            "occupation_w1_reduction_percent": 100*(focc-cocc)/focc if focc else 0.,
            "reference_mean": float(ref[..., c].mean()), "free_mean": float(free[..., c].mean()),
            "controlled_mean": float(controlled[..., c].mean()), "reference_sd": float(ref[..., c].std()),
            "free_sd": float(free[..., c].std()), "controlled_sd": float(controlled[..., c].std())})
        for code in SERIES:
            density_rows.extend({"subject_id": args.subject, "page": c//36+1, "channel_index": c,
                "channel": str(channels[c]), "series_code": code, "series_label": SERIES[code][1],
                "standardized_amplitude": float(x), "density": float(y)}
                for x, y in zip(grids[c], curves[(c, code)]))
    # Verify displayed endpoints against the evaluation's context/bank source.
    eval_csv = args.run_directory/"outer_posthoc_amendment/all_channel_context_bank_metrics.csv"
    source_metrics = pd.read_csv(eval_csv)
    selected = source_metrics[(source_metrics.context_index == 7) & (source_metrics.crn_bank == 0)].sort_values("channel_index")
    generated = pd.DataFrame(metric_rows).sort_values("channel_index")
    checks = [float(np.max(np.abs(generated[k].to_numpy()-selected[k].to_numpy())))
              for k in ("time_w1_free", "time_w1_controlled", "occupation_w1_free", "occupation_w1_controlled")]
    if len(selected) != profile["n"] or max(checks) > 1e-12:
        raise ValueError("Displayed empirical W1 does not bind to the completed evaluation source.")
    outputs = []
    offset = 0
    for page_index, count in enumerate(profile["pages"], 1):
        height = 180-max(0, 6-int(math.ceil(count/6)))*24.66
        fig, axes = plt.subplots(6, 6, figsize=(170/25.4, height/25.4), sharex=True, squeeze=False)
        for i, ax in enumerate(axes.flat):
            if i >= count:
                ax.set_visible(False); continue
            c = offset+i; row, col = divmod(i, 6)
            ax.set_position([.030+col*.160, (height-29.7-row*24.66-17.1)/height, .145, 17.1/height])
            max_y = 0.
            for code, style in SERIES.items():
                y = curves[(c, code)]
                ax.plot(grids[c], y, color=style[2], lw=style[3], ls=style[4], alpha=style[5])
                max_y = max(max_y, float(y.max()))
            ax.set_xlim(x_min, x_max); ax.set_ylim(0, max_y*1.10); ax.set_yticks([])
            ax.set_title(clean_contact(str(channels[c])), loc="left", fontsize=9,
                         fontweight="bold", color="#4D4D4D", pad=2)
            ax.set_xticks([v for v in (-1, 0, 1) if x_min <= v <= x_max])
            has_below = any(j < count for j in range(i+6, count, 6))
            ax.tick_params(axis="x", labelbottom=not has_below, labelsize=9, length=2)
        fig.text(.09, .984, f"{args.subject}: channel occupation laws ({page_index}/{len(profile['pages'])})",
                 va="top", ha="left", fontsize=10, fontweight="bold")
        legend = [Line2D([0], [0], color=style[2], lw=1.2, ls=style[4], label=style[1])
                  for style in SERIES.values()]
        fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.52, 1-9/height),
                   ncol=4, fontsize=9, handlelength=1.2, columnspacing=.5)
        fig.supxlabel("Standardized amplitude", fontsize=9, y=2.16/height)
        fig.canvas.draw()
        occupied = [ax for ax in fig.axes if ax.get_visible()]
        if len(occupied) != count or any(len(ax.lines) != 4 for ax in occupied):
            raise AssertionError("Each occupied channel must contain exactly four displayed curves.")
        suffix = "_v2" if args.subject == "HUP080" else ""
        base = args.output/f"{args.subject.lower()}_ofrc_all_channels_page_{page_index:02d}{suffix}"
        for ext in ("pdf", "png", "svg"):
            kwargs = {"metadata": {"CreationDate": None, "ModDate": None}} if ext == "pdf" else {}
            fig.savefig(base.with_suffix("."+ext), dpi=300, **kwargs)
        outputs.append({"page": page_index, "channel_count": count, "width_mm": 170, "height_mm": height,
                         "occupied_axis_count": len(occupied), "curves_per_channel": 4,
                         "artifact_sha256": {ext: sha(base.with_suffix("."+ext)) for ext in ("pdf", "png", "svg")}})
        plt.close(fig); offset += count
    pd.DataFrame(density_rows).to_csv(args.output/"source_data_channel_density_long.csv", index=False, encoding="utf-8-sig")
    generated.to_csv(args.output/"source_data_display_context07_bank0_metrics.csv", index=False, encoding="utf-8-sig")
    controls = arrays["controls"]
    display = {"mean_time_w1_free": float(generated.time_w1_free.mean()),
               "mean_time_w1_controlled": float(generated.time_w1_controlled.mean()),
               "mean_occupation_w1_free": float(generated.occupation_w1_free.mean()),
               "mean_occupation_w1_controlled": float(generated.occupation_w1_controlled.mean()),
               "time_channels_improved": int((generated.time_w1_controlled < generated.time_w1_free).sum()),
               "occupation_channels_improved": int((generated.occupation_w1_controlled < generated.occupation_w1_free).sum()),
               "control_rms": float(np.sqrt(np.mean(controls**2))),
               "maximum_per_actuator_rms": float(np.sqrt(np.mean(controls**2, axis=(0, 1))).max()),
               "control_peak": float(np.abs(controls).max()),
               "control_energy": float(np.mean(np.sum(controls**2, axis=-1)))}
    qa = {"subject": args.subject, "backend": "Python/matplotlib", "archetype": "quantitative grid",
          "completed_training_before_display_read": True, "completed_evaluation_before_display_read": True,
          "predeclared_display_context": 7, "predeclared_crn_bank": 0,
          "trained_updates": training["trained_updates"], "selected_update": training["selected_update"],
          "source_sha256": {**hashes, "original_density_source": sha(args.original_density_source), "evaluation_metric_csv": sha(eval_csv)},
          "no_new_training_rollout_normalization_or_rescaling": True, "no_distance_annotation": True,
          "no_gate_annotation": True, "no_original_actor_comparator": True,
          "original_montage_labels_preserved_except_display_prefix_suffix": True,
          "selected_actuator_count": profile["m"], "channel_count": profile["n"],
          "page_occupancy": list(profile["pages"]), "pages": outputs,
          "original_grid_points": 240, "original_x_limits": [x_min, x_max], "kde_absolute_bandwidth": BANDWIDTH,
          "plot_window_note": "Original viewing window retained exactly; empirical W1 includes all samples, including tails outside the plot axis.",
          "outside_axis_sample_fractions": {code: float(np.mean((arrays[s[0]] < x_min) | (arrays[s[0]] > x_max))) for code, s in SERIES.items()},
          "empirical_endpoint_binding_maximum_error": max(checks),
          "display_context_only_metrics": display,
          "outer_all_context_bank_summary": evaluation,
          "scope_warning": "Displayed curves and display metrics concern context 7/bank 0 only; the separate outer summary aggregates 8 contexts x 3 banks. Already-revealed outer run is a post-hoc amendment, not a pristine independent test.",
          "overleaf_modified": False, "historical_source_modified": False}
    with (args.output/"external_original_style_numeric_qa.json").open("w", encoding="utf-8") as f:
        json.dump(qa, f, ensure_ascii=False, indent=2)
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
