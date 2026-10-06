"""Independent HUP060-only sample audit; never reads external-patient results."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"full": ROOT.parent/".fresh_wgangp_20261006/runs/fresh_seed20261011_e1000",
        "no_wgan": ROOT/"runs/no_wgan_seed20261011_e1000",
        "no_graph_spread": ROOT/"runs/no_graph_seed20261011_e1000",
        "no_deviation": ROOT/"runs/no_deviation_seed20261011_e1000"}


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    summary = pd.read_csv(ROOT/"ablations/ablation_summary.csv").set_index("variant")
    contrasts = pd.read_csv(ROOT/"ablations/ablation_contrasts_vs_full.csv")
    aggregation = json.loads((ROOT/"ablations/ablation_aggregation_qa.json").read_text("utf-8"))
    fig_qa = json.loads((ROOT/"ablations/figures/ablation_figure_qa.json").read_text("utf-8"))
    before_hashes = {name: sha(ROOT/"ablations"/name) for name in
                     ["ablation_summary.csv", "ablation_per_channel.csv", "ablation_contrasts_vs_full.csv", "ablation_aggregation_qa.json"]}
    checks, results = [], []
    def check(name, ok, detail=None):
        checks.append({"name": name, "passed": bool(ok), "detail": detail})
    check("same_1000_actor_update_budget", aggregation["same_actor_update_budget"] == 1000)
    check("not_equal_energy_or_wallclock", aggregation["not_an_equal_energy_comparison"] and not aggregation["same_wallclock_budget"])
    check("teacher_not_used", aggregation["teacher_used"] is False)
    shared = None; sample_metrics = {}
    for variant, run in RUNS.items():
        training = json.loads((run/"training_summary.json").read_text("utf-8"))
        evaluation = json.loads((run/"evaluation/evaluation_summary.json").read_text("utf-8"))
        history = pd.read_csv(run/"training_history.csv")
        check(variant+":complete_1000_history", training["trained_epochs"] == 1000 and np.array_equal(history.epoch, np.arange(1001)))
        frozen_hash = sha(run/"frozen_actor_wgan.pt")
        check(variant+":same_frozen_checkpoint", frozen_hash == training["checkpoint_sha256"] == evaluation["checkpoint_sha256"] == aggregation["checkpoint_sha256"][variant])
        history = history[(history.epoch > 0) & history.validation_selection_score.notna()]
        winner = history.loc[history.validation_selection_score.idxmin()]
        check(variant+":validation_selected_not_outer_selected", int(winner.epoch) == training["best_epoch"] and evaluation["recorded_future_used_for_training_or_selection"] is False)
        with np.load(run/"evaluation/paired_comparison.npz", allow_pickle=False) as data:
            z = {k: np.asarray(data[k]) for k in data.files}
        if shared is None:
            shared = z
        for key in ("observed_scaled", "uncontrolled_scaled", "reference_fit_scaled", "reference_validation_scaled", "selected_indices", "channels"):
            check(variant+":shared_"+key, np.array_equal(z[key], shared[key]))
        reference = z["reference_validation_scaled"]
        values = z["candidate_controlled_scaled"]
        tw = np.array([np.mean([wasserstein_distance(values[:, t, c], reference[:, t, c]) for t in range(256)]) for c in range(36)])
        ow = np.array([wasserstein_distance(values[..., c].ravel(), reference[..., c].ravel()) for c in range(36)])
        sample_metrics[variant] = (tw, ow)
        check(variant+":time_endpoint_independently_recomputed", np.max(np.abs(tw-z["time_w1_candidate"])) < 1e-12, float(np.max(np.abs(tw-z["time_w1_candidate"]))))
        check(variant+":occupation_endpoint_independently_recomputed", np.max(np.abs(ow-z["occupation_w1_candidate"])) < 1e-12, float(np.max(np.abs(ow-z["occupation_w1_candidate"]))))
        check(variant+":aggregate_mean_endpoints", abs(float(tw.mean())-summary.loc[variant,"mean_time_w1"]) < 1e-12 and abs(float(ow.mean())-summary.loc[variant,"mean_occupation_w1"]) < 1e-12)
        mean_bias = abs(values.mean(axis=(0, 1))-reference.mean(axis=(0, 1)))
        check(variant+":W1_mean_bias_lower_bound", np.all(ow+1e-12 >= mean_bias))
        controls, commands = z["candidate_controls"], z["candidate_commands"]
        rms = float(np.sqrt(np.mean(controls**2))); command_rms = float(np.sqrt(np.mean(commands**2)))
        check(variant+":input_RMS_independently_recomputed", abs(rms-summary.loc[variant,"control_rms"]) < 1e-12 and abs(command_rms-summary.loc[variant,"command_rms"]) < 1e-12)
        check(variant+":control_energy_mean_square_definition", abs(rms**2-summary.loc[variant,"control_energy"]) < 1e-12)
        free = z["uncontrolled_scaled"]
        f_tw = np.array([np.mean([wasserstein_distance(free[:, t, c], reference[:, t, c]) for t in range(256)]) for c in range(36)])
        f_ow = np.array([wasserstein_distance(free[..., c].ravel(), reference[..., c].ravel()) for c in range(36)])
        non_direct = np.setdiff1d(np.arange(36), z["selected_indices"])
        results.append({"variant": variant, "selected_epoch": int(training["best_epoch"]),
            "mean_time_w1": float(tw.mean()), "mean_occupation_w1": float(ow.mean()),
            "control_rms": rms, "command_rms": command_rms,
            "actuator_mean_square": rms**2, "sum_actuator_square_mean": float(np.mean(np.sum(controls**2, axis=-1))),
            "time_channels_improved_vs_free": int(np.sum(tw < f_tw)),
            "occupation_channels_improved_vs_free": int(np.sum(ow < f_ow)),
            "nondirect_occupation_channels_improved_vs_free": int(np.sum(ow[non_direct] < f_ow[non_direct])),
            "occupation_not_improved_contacts": z["channels"][ow >= f_ow].astype(str).tolist()})
    comparison = []
    full_t, full_o = sample_metrics["full"]
    for variant in ("no_wgan", "no_graph_spread", "no_deviation"):
        tw, ow = sample_metrics[variant]
        ct = contrasts[contrasts.ablation == variant].sort_values("channel_index")
        check(variant+":all_per_channel_signed_contrasts_retained", np.max(abs(ct.delta_time_w1_vs_full.to_numpy()-(tw-full_t))) < 1e-12 and np.max(abs(ct.delta_occupation_w1_vs_full.to_numpy()-(ow-full_o))) < 1e-12)
        comparison.append({"variant": variant,
            "time_increase_vs_full_percent": float(100*(tw.mean()/full_t.mean()-1)),
            "occupation_increase_vs_full_percent": float(100*(ow.mean()/full_o.mean()-1)),
            "time_contacts_worse_than_full": int(np.sum(tw > full_t)),
            "occupation_contacts_worse_than_full": int(np.sum(ow > full_o)),
            "time_contacts_better_than_full": int(np.sum(tw < full_t)),
            "occupation_contacts_better_than_full": int(np.sum(ow < full_o)),
            "time_delta_min": float((tw-full_t).min()), "occupation_delta_min": float((ow-full_o).min())})
    check("figure_count_time_endpoint_binding", fig_qa["panel_f_time_counts"] == [r["time_contacts_worse_than_full"] for r in comparison])
    check("figure_count_occupation_endpoint_binding", fig_qa["panel_f_occupation_counts"] == [r["occupation_contacts_worse_than_full"] for r in comparison])
    check("figure_percentage_time_endpoint_binding", np.max(abs(np.asarray(fig_qa["panel_c_time_changes_percent"])-[r["time_increase_vs_full_percent"] for r in comparison])) < 1e-10)
    check("figure_percentage_occupation_endpoint_binding", np.max(abs(np.asarray(fig_qa["panel_c_occupation_changes_percent"])-[r["occupation_increase_vs_full_percent"] for r in comparison])) < 1e-10)
    with np.load(ROOT/"ablations/ablation_rollouts.npz", allow_pickle=False) as z:
        unselected = np.setdiff1d(np.arange(36), z["selected_indices"])
        no_graph_input = float(abs(z["no_graph_spread_effective_control"][..., unselected]).max())
        nodev_spread = float(z["no_deviation_controls"].std(axis=0).max())
        check("no_graph_has_zero_nondirect_injected_input", no_graph_input < 1e-12, no_graph_input)
        check("no_deviation_has_particle_common_control", nodev_spread < 1e-12, nodev_spread)
    check("sources_unchanged_during_independent_audit", all(sha(ROOT/"ablations"/name) == digest for name,digest in before_hashes.items()))
    failures = [r["name"] for r in checks if not r["passed"]]
    result = {"status": "passed" if not failures else "failed", "patient": "HUP060",
              "external_outcome_arrays_opened": False, "training_started": False,
              "overleaf_modified": False, "check_count": len(checks), "failures": failures,
              "checks": checks, "summary": results, "comparisons_vs_full": comparison,
              "source_sha256": before_hashes,
              "statistical_scope": "One technical training seed per arm, 36 correlated contacts; descriptive matched-update comparison, not equal-energy or patient-level inference."}
    output = ROOT/"qa/hup060_ablation_independent_audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "check_count": len(checks), "failures": failures, "summary": results,
                      "comparisons_vs_full": comparison, "output": str(output)}, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
