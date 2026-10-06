"""Independently audit a completed fresh HUP060 WGAN-GP development run.

This checker never trains a network or changes a source/model/paper artifact.
It refuses to open evaluation arrays until training completion is verified.
Its only output is <run-dir>/qa_summary.json. No superiority is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import torch

HERE = Path(__file__).resolve().parent
EXPECTED_MODEL_SHA = "a2984a7f1042f2fff1231d744405f76ca8b649c8048812d7f48249a3e02e1e43"
EXPECTED_BASELINE_SHA = "bb5e43d4b8774d144e3fd598650ee1653e22f8391764467b13e403d2577d8541"
PAPER_BASELINE = HERE / "repro_inputs" / "paper_baseline_paired_comparison.npz"
SOURCE_SERIES = {
    "Recorded ictal": "observed_scaled",
    "u=0 prediction": "uncontrolled_scaled",
    "Reference": "reference_validation_scaled",
    "Fresh WGAN-GP": "candidate_controlled_scaled",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def array_bytes_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def occupation_w1(sequence: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.array([wasserstein_distance(sequence[..., c].reshape(-1),
                                         reference[..., c].reshape(-1))
                     for c in range(36)])


def time_w1(sequence: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.array([np.mean([wasserstein_distance(sequence[:, t, c], reference[:, t, c])
                              for t in range(sequence.shape[1])])
                     for c in range(36)])


def boolean_column(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy()
    values = series.astype(str).str.lower()
    if not values.isin(["true", "false", "1", "0"]).all():
        raise ValueError("Unexpected boolean column encoding")
    return values.isin(["true", "1"]).to_numpy()


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.details: dict = {}

    def check(self, name: str, passed: bool, detail=None) -> bool:
        item = {"name": name, "passed": bool(passed)}
        if detail is not None:
            item["detail"] = detail
        self.checks.append(item)
        return bool(passed)

    def close(self, name: str, actual, expected, atol=1e-11, rtol=1e-11) -> bool:
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        maximum = float(np.max(np.abs(a - b))) if a.size and a.shape == b.shape else None
        return self.check(name, a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol),
                          {"max_abs_error": maximum, "atol": atol, "rtol": rtol})


def audit_training(audit: Audit, run: Path) -> tuple[dict, dict]:
    required = ["training_contract.json", "training_summary.json", "training_history.csv",
                "initialization_audit.json", "initial_actor.pt", "frozen_actor_wgan.pt"]
    absent = [name for name in required if not (run / name).is_file()]
    if not audit.check("completed_training_files_exist", not absent, absent):
        raise RuntimeError("Training is incomplete; evaluation arrays were not opened")
    contract = read_json(run / "training_contract.json")
    summary = read_json(run / "training_summary.json")
    initialization = read_json(run / "initialization_audit.json")
    expected_epochs = int(contract["epochs"])
    history = pd.read_csv(run / "training_history.csv")
    finished = summary.get("status") == "complete_fresh_hybrid_wgangp_development_experiment"
    finished = finished and int(summary.get("trained_epochs", -1)) == expected_epochs
    if not audit.check("training_completed_before_outcome_access", finished):
        raise RuntimeError("Training is not complete; evaluation arrays were not opened")
    audit.check("epoch_history_exactly_zero_through_budget",
                np.array_equal(history["epoch"].to_numpy(), np.arange(expected_epochs + 1)),
                {"rows": len(history), "expected": expected_epochs + 1})
    updates = history.loc[history.epoch > 0]
    finite_cols = ["train_total", "train_law", "train_adversarial", "critic_loss", "critic_estimate",
                   "critic_gp", "critic_input_gradient_norm", "actor_gradient_norm_before_clip",
                   "critic_gradient_norm_before_clip", "actor_lr"]
    audit.check("all_training_update_diagnostics_finite",
                all(k in updates for k in finite_cols) and
                np.isfinite(updates[finite_cols].to_numpy(dtype=float)).all())
    for key in [k for k in updates if k.startswith("train_law_")]:
        audit.check("finite_" + key, np.isfinite(updates[key]).all())
    adv_weight = float(contract["adv_weight"])
    audit.close("recorded_total_equals_law_plus_weighted_adversarial",
                updates["train_total"], updates["train_law"] + adv_weight * updates["train_adversarial"])
    audit.close("hybrid_adversarial_weight_is_point_five", adv_weight, .5)
    audit.check("no_teacher_in_training_contract", contract.get("teacher_loaded") is False)
    audit.check("no_teacher_in_training_summary", summary.get("teacher_checkpoint_loaded") is False)
    audit.close("no_teacher_proximity", contract.get("teacher_proximity_weight"), 0.)
    audit.check("adversarial_active_from_first_actor_update",
                contract.get("adv_active_from_actor_update") == 1 and summary.get("adversarial_active_from_update") == 1)
    mask = np.asarray(contract["actuators"])
    audit.check("thirteen_unique_valid_actuators", len(mask) == 13 and len(np.unique(mask)) == 13 and
                np.issubdtype(mask.dtype, np.integer) and np.all((mask >= 0) & (mask < 36)))
    audit.check("original_frozen_model_sha", contract.get("model_sha256") == EXPECTED_MODEL_SHA and
                summary.get("model_sha256") == EXPECTED_MODEL_SHA)
    audit.check("initial_actor_checkpoint_not_loaded", initialization.get("no_actor_checkpoint_loaded") is True)
    audit.check("initial_policy_neutral_zero", initialization.get("neutral_control_peak", math.inf) <= 1e-12 and
                initialization.get("initial_to_free_max_abs_error", math.inf) <= 1e-12)
    audit.check("initial_policy_parameters_present", int(initialization.get("trainable_parameters", 0)) > 0)
    audit.check("run02_not_used_for_training_or_selection", contract.get("run02_access_during_training") is False)
    audit.check("run_not_adopted_in_paper", summary.get("not_adopted_in_paper") is True)
    audit.check("originals_reported_unchanged", summary.get("originals_unchanged") is True)
    hashes = contract.get("original_hashes_before", {})
    present_hashes = {name: sha(Path(name)) for name in hashes if Path(name).is_file()}
    absent_originals = [name for name in hashes if not Path(name).is_file()]
    audit.check("present_original_artifact_hashes_unchanged",
                all(present_hashes[name] == hashes[name] for name in present_hashes),
                {"checked_count": len(present_hashes), "absent_historical_paths": absent_originals})
    audit.check("frozen_checkpoint_sha_matches_training_summary",
                sha(run / "frozen_actor_wgan.pt") == summary.get("checkpoint_sha256"))
    initial = torch.load(run / "initial_actor.pt", map_location="cpu", weights_only=False)["actor_state_dict"]
    frozen = torch.load(run / "frozen_actor_wgan.pt", map_location="cpu", weights_only=False)
    trained = frozen["actor_state_dict"]
    audit.check("actor_state_keys_unchanged", set(initial) == set(trained))
    changed = {}
    for key in ["mean_gain_delta", "deviation_gain_delta", "common_residual.5.weight",
                "deviation_residual.5.weight", "common_markov_residual.5.weight",
                "deviation_markov_residual.5.weight"]:
        if key in initial and key in trained:
            changed[key] = float((trained[key] - initial[key]).abs().max())
    audit.check("sampled_controller_weights_changed", bool(changed) and any(v > 1e-12 for v in changed.values()), changed)
    audit.check("all_frozen_actor_tensors_finite", all(torch.isfinite(v).all().item() for v in trained.values()))
    buffers = ["reference_mean_path", "reference_variance_path", "reference_scale", "base_gain",
               "actuated_channel_indices", "markov_feature_center", "markov_feature_scale",
               "local_inverse_effect", "spline_basis"]
    audit.check("nontrainable_actor_buffers_unchanged",
                all(k in initial and k in trained and torch.equal(initial[k], trained[k]) for k in buffers))
    selected_epoch = int(summary["best_epoch"])
    validation = history.loc[history.validation_selection_score.notna() & (history.epoch > 0)]
    initial_row = history.loc[history.epoch == 0].iloc[0]
    expected_score = .5 * validation.validation_law / initial_row.validation_law
    expected_score += .25 * validation.validation_mean_time_w1 / initial_row.validation_mean_time_w1
    expected_score += .25 * validation.validation_mean_occupation_w1 / initial_row.validation_mean_occupation_w1
    audit.close("all_validation_scores_follow_declared_rule", validation.validation_selection_score, expected_score)
    best_row = validation.loc[validation.validation_selection_score.idxmin()]
    audit.check("selected_epoch_is_minimum_recorded_validation_score", selected_epoch == int(best_row.epoch))
    audit.close("selected_score_matches_history", summary["best_selection_score"], best_row.validation_selection_score)
    audit.check("frozen_checkpoint_best_epoch_matches_summary", int(frozen["best_epoch"]) == selected_epoch)
    audit.check("recorded_critic_update_budget", int(summary["critic_updates"]) == 24 + int(contract["n_critic"]) * expected_epochs)
    audit.details["training"] = {"epochs": expected_epochs, "selected_epoch": selected_epoch,
                                  "technical_training_seeds": contract.get("technical_training_seeds"),
                                  "elapsed_seconds": summary.get("elapsed_seconds")}
    return contract, summary


def audit_evaluation(audit: Audit, run: Path, contract: dict) -> tuple[dict, dict, np.ndarray]:
    directory = run / "evaluation"
    summary = read_json(directory / "evaluation_summary.json")
    audit.check("copied_paper_baseline_exact_file_hash", sha(PAPER_BASELINE) == EXPECTED_BASELINE_SHA)
    with np.load(directory / "paired_comparison.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    with np.load(PAPER_BASELINE, allow_pickle=False) as saved:
        baseline = {key: saved[key].copy() for key in saved.files}
    audit.check("evaluation_is_frozen_development_preview",
                summary.get("status") == "frozen_run02_development_preview_only")
    audit.check("evaluation_not_adopted_in_paper", summary.get("not_adopted_in_paper") is True)
    audit.check("recorded_future_not_used_for_training_or_selection",
                summary.get("recorded_future_used_for_training_or_selection") is False)
    audit.check("evaluation_frozen_checkpoint_sha", summary.get("checkpoint_sha256") == sha(run / "frozen_actor_wgan.pt"))
    audit.check("all_evaluation_numeric_arrays_finite", all(np.isfinite(value).all() for value in arrays.values()
                if np.issubdtype(value.dtype, np.number)))
    for key in ["observed_scaled", "uncontrolled_scaled", "reference_fit_scaled", "reference_validation_scaled",
                "selected_indices", "channels"]:
        same = key in arrays and key in baseline and arrays[key].shape == baseline[key].shape
        same = same and arrays[key].dtype == baseline[key].dtype and np.array_equal(arrays[key], baseline[key])
        same = same and array_bytes_hash(arrays[key]) == array_bytes_hash(baseline[key])
        audit.check("unchanged_baseline_" + key, same)
    audit.check("paper_control_comparison_is_unchanged",
                np.array_equal(arrays["paper_controlled_scaled"], baseline["candidate_controlled_scaled"]))
    audit.check("evaluation_mask_matches_training_contract",
                np.array_equal(arrays["selected_indices"], np.asarray(contract["actuators"])))
    audit.close("saved_candidate_time_W1_vector", arrays["time_w1_candidate"],
                time_w1(arrays["candidate_controlled_scaled"], arrays["reference_validation_scaled"]))
    audit.close("saved_candidate_occupation_W1_vector", arrays["occupation_w1_candidate"],
                occupation_w1(arrays["candidate_controlled_scaled"], arrays["reference_validation_scaled"]))
    audit.check("no_control_rollout_parity", float(summary["no_control_parity_max_abs_error"]) <= 1e-6)
    channels = arrays["channels"].astype(str)
    audit.check("exactly_thirty_six_unique_channels", len(channels) == 36 and len(set(channels)) == 36)
    mask = np.isin(np.arange(36), arrays["selected_indices"])
    table = pd.read_csv(directory / "channel_metrics.csv")
    audit.check("metric_table_channel_order", np.array_equal(table.channel.astype(str).to_numpy(), channels))
    audit.check("metric_table_direct_mask", np.array_equal(boolean_column(table.direct_actuator), mask))
    reference = arrays["reference_validation_scaled"]
    rm = reference.mean(axis=(0, 1))
    computed = {}
    for label, key in [("free", "uncontrolled_scaled"), ("paper", "paper_controlled_scaled"),
                       ("fresh", "candidate_controlled_scaled")]:
        seq = arrays[key]
        audit.check(label + "_sequence_shape", seq.shape == (32, 256, 36))
        tw, ow = time_w1(seq, reference), occupation_w1(seq, reference)
        mean, sd = seq.mean(axis=(0, 1)), seq.std(axis=(0, 1))
        computed[label] = {"time": tw, "occupation": ow, "mean": mean, "sd": sd}
        for column, values in [("time_w1_" + label, tw), ("occupation_w1_" + label, ow),
                               ("mean_" + label, mean), ("sd_" + label, sd),
                               ("abs_mean_bias_" + label, abs(mean - rm))]:
            audit.close("empirical_metric_" + column, table[column], values)
        audit.check(label + "_occupation_W1_lower_bounds_absolute_mean_bias",
                    np.all(ow + 1e-12 >= abs(mean - rm)),
                    {"minimum_W1_minus_mean_bias": float(np.min(ow - abs(mean - rm)))})
        audit.close("summary_mean_time_" + label, summary["mean_time_w1_" + label], tw.mean())
        audit.close("summary_mean_occupation_" + label, summary["mean_occupation_w1_" + label], ow.mean())
    audit.close("metric_reference_mean", table.mean_reference, rm)
    audit.close("metric_reference_sd", table.sd_reference, reference.std(axis=(0, 1)))
    free, fresh, paper = [computed[k] for k in ["free", "fresh", "paper"]]
    audit.close("per_channel_occupation_reduction_pct", table.occupation_reduction_vs_free_pct,
                100 * (1 - fresh["occupation"] / free["occupation"]))
    for metric in ["time", "occupation"]:
        audit.close("aggregate_" + metric + "_reduction", summary[metric + "_reduction_vs_free_pct"],
                    100 * (1 - fresh[metric].mean() / free[metric].mean()))
        audit.check("improved_count_vs_free_" + metric,
                    int(summary["channels_improved_vs_free_" + metric]) == int(sum(fresh[metric] < free[metric])))
        audit.check("improved_count_vs_paper_" + metric,
                    int(summary["channels_improved_vs_paper_" + metric]) == int(sum(fresh[metric] < paper[metric])))
    audit.check("nondirect_occupation_improved_count",
                int(summary["nondirect_channels_improved_vs_free_occupation"]) ==
                int(sum(fresh["occupation"][~mask] < free["occupation"][~mask])))
    for key in ["candidate_controls", "candidate_commands"]:
        value = arrays[key]
        peak = float(abs(value).max())
        audit.check(key + "_shape", value.shape == (32, 256, 13))
        audit.check(key + "_amplitude_at_most_one_point_eight", peak <= 1.8 + 1e-12, {"peak": peak})
    controls = arrays["candidate_controls"]
    audit.close("summary_control_peak", summary["candidate_control_diagnostics"]["peak"], abs(controls).max())
    audit.close("summary_control_rms", summary["candidate_control_diagnostics"]["rms"], np.sqrt(np.mean(controls**2)))
    for container_name, container in [("contract", contract), ("evaluation", summary)]:
        for key, value in container.items():
            if "adopted" in key.lower() and not key.lower().startswith("not_"):
                audit.check(container_name + "_" + key + "_false", value is False)
            if key.lower() in ["paper_changed", "overleaf_changed", "paper_modified", "originals_changed"]:
                audit.check(container_name + "_" + key + "_false", value is False)
    audit.details["evaluation"] = {"empirical_recomputed_mean_time_w1_fresh": float(fresh["time"].mean()),
        "empirical_recomputed_mean_occupation_w1_fresh": float(fresh["occupation"].mean()),
        "time_channels_improved_vs_free": int(sum(fresh["time"] < free["time"])),
        "occupation_channels_improved_vs_free": int(sum(fresh["occupation"] < free["occupation"])),
        "no_superiority_requirement": True}
    return arrays, computed, channels


def audit_figures(audit: Audit, figures: Path, arrays: dict, channels: np.ndarray) -> None:
    summary = read_json(figures / "plot_and_metric_summary.json")
    density = pd.read_csv(figures / "density_source.csv")
    audit.check("density_has_four_expected_series", set(density.series) == set(SOURCE_SERIES))
    audit.check("density_has_thirty_six_channel_labels", set(density.channel) == set(channels))
    groups = density.groupby(["channel", "series"], sort=False)
    audit.check("density_has_exactly_one_hundred_forty_four_groups", len(groups) == 144)
    bandwidth = float(summary["kde_absolute_bandwidth"])
    audit.close("identical_absolute_KDE_bandwidth", bandwidth, .16)
    integrals, errors, common_grid = [], [], None
    lookup = {name: c for c, name in enumerate(channels)}
    shared = True
    all_finite = True
    all_nonnegative = True
    for (name, series), frame in groups:
        x = frame.standardized_amplitude.to_numpy(dtype=float)
        y = frame.density.to_numpy(dtype=float)
        all_finite = all_finite and np.isfinite(x).all() and np.isfinite(y).all()
        all_nonnegative = all_nonnegative and np.all(y >= 0)
        shared = shared and np.all(np.diff(x) > 0)
        if common_grid is None:
            common_grid = x
        shared = shared and np.array_equal(x, common_grid)
        integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        integrals.append(float(integrate(y, x)))
        samples = arrays[SOURCE_SERIES[series]][..., lookup[name]].reshape(-1)
        # Independent analytic fixed-bandwidth Gaussian kernel mixture. No
        # fitted gaussian_kde instance, curve normalization, or recentering.
        expected = np.zeros_like(x)
        for start in range(0, len(samples), 2048):
            z = (x[:, None] - samples[None, start:start + 2048]) / bandwidth
            expected += np.exp(-.5 * z**2).sum(axis=1)
        expected /= len(samples) * bandwidth * math.sqrt(2 * math.pi)
        errors.append(float(np.max(np.abs(y - expected))))
    audit.check("all_density_curves_finite", all_finite)
    audit.check("all_density_curves_nonnegative", all_nonnegative)
    audit.check("all_one_hundred_forty_four_curves_share_identical_increasing_grid", shared)
    audit.check("density_integrals_at_least_point_9999", bool(integrals) and min(integrals) >= .9999,
                {"minimum": min(integrals), "maximum": max(integrals)})
    audit.check("density_integrals_at_most_one_plus_roundoff", max(integrals) <= 1 + 1e-8)
    audit.check("densities_match_raw_empirical_samples_without_rescale_or_recentering",
                max(errors) <= 1e-10, {"maximum_pointwise_error": max(errors)})
    audit.close("reported_KDE_integral_range", summary["kde_integral_range"], [min(integrals), max(integrals)])
    audit.close("reported_shared_x_limits", summary["shared_x_limits"], [common_grid[0], common_grid[-1]])
    audit.check("all_raw_tails_retained_plus_five_bandwidths",
                common_grid[0] <= min(arrays[k].min() for k in SOURCE_SERIES.values()) - 5 * bandwidth + 1e-12 and
                common_grid[-1] >= max(arrays[k].max() for k in SOURCE_SERIES.values()) + 5 * bandwidth - 1e-12)
    svg = figures / "hup060_fresh_wgangp_all36.svg"
    png = figures / "hup060_fresh_wgangp_all36.png"
    pdf = figures / "hup060_fresh_wgangp_all36.pdf"
    audit.check("overview_svg_exists", svg.is_file() and svg.stat().st_size > 0)
    audit.check("overview_png_exists", png.is_file() and png.stat().st_size > 0)
    audit.check("overview_pdf_exists", pdf.is_file() and pdf.stat().st_size > 0)
    if png.is_file():
        with png.open("rb") as handle:
            header = handle.read(24)
        width = int.from_bytes(header[16:20], "big")
        height = int.from_bytes(header[20:24], "big")
        audit.check("overview_PNG_header_and_resolution",
                    header[:8] == b"\x89PNG\r\n\x1a\n" and width > 1000 and height > 1000,
                    {"width": width, "height": height})
    tree = ET.parse(svg)
    texts = ["".join(node.itertext()).strip() for node in tree.getroot().iter()
             if node.tag.rsplit("}", 1)[-1] == "text"]
    counts = {name: texts.count(name) for name in channels}
    audit.check("SVG_contains_each_of_thirty_six_labels_exactly_once", all(v == 1 for v in counts.values()), counts)
    audit.check("SVG_contains_all_four_legend_labels", all(label in texts for label in SOURCE_SERIES))
    audit.details["figures"] = {"density_groups": len(groups), "shared_grid_points": len(common_grid),
        "KDE_integral_range": [min(integrals), max(integrals)], "max_independent_KDE_error": max(errors),
        "overview_png_sha256": sha(png), "overview_svg_sha256": sha(svg)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    args = parser.parse_args()
    run, figures = args.run_dir.resolve(), args.figures_dir.resolve()
    if not run.is_dir():
        parser.error("--run-dir must be an existing completed-run directory")
    audit = Audit()
    started = __import__("time").perf_counter()
    try:
        contract, summary = audit_training(audit, run)
        arrays, computed, channels = audit_evaluation(audit, run, contract)
        audit_figures(audit, figures, arrays, channels)
    except Exception as exc:
        audit.check("checker_completed_without_exception", False, {"error": repr(exc), "traceback": traceback.format_exc()})
    failed = [item["name"] for item in audit.checks if not item["passed"]]
    result = {"status": "passed" if not failed else "failed", "all_checks_passed": not failed,
        "checks": audit.checks, "failed_checks": failed, "details": audit.details,
        "run_directory": str(run), "figures_directory": str(figures),
        "checker_sha256": sha(Path(__file__)), "elapsed_seconds": __import__("time").perf_counter() - started,
        "scope": "Read-only artifact audit; no training or paper update; no required new-controller superiority"}
    output = run / "qa_summary.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "failed_checks": failed, "output": str(output)}, ensure_ascii=False))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
