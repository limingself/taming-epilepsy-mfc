"""Qualified old/new budget and paired endpoint reports; no selection is made."""
from pathlib import Path
import json
import pandas as pd

root = Path(__file__).resolve().parent
selected = root / "selected"
previous = Path(r"C:\Users\LiMing\Documents\改论文\.fresh_wgangp_figures_20261006\external\runs\HUP080_seed20261011_u1000_uniformref")
final = json.loads((selected / "final_posthoc_report.json").read_text(encoding="utf-8"))
interpreted = {}
for group in ("ctx6_terminal_veto", "outer_posthoc_amendment"):
    report = final[group]
    gates = pd.read_csv(selected / group / "aggregate_gate_vectors.csv")
    gates["original_0405_budget_pass"] = report["old_budget_pass_count"] == report["context_bank_evaluations"]
    gates["revised_0450_budget_pass"] = report["revised_budget_pass_count"] == report["context_bank_evaluations"]
    gates["full_gate_pass_revised_budget"] = gates.full_gate_b_pass & gates.revised_0450_budget_pass
    gates.to_csv(selected / group / "aggregate_gate_vectors_old_and_revised_budget.csv", index=False)
    old_veto = report["terminal_veto_pass"]
    revised_veto = bool(report["revised_budget_pass"] and report["gate_b_pass_count"] >= 1
        and report["mean_time_w1_controlled"] < report["mean_time_w1_free"]
        and report["mean_occupation_w1_controlled"] < report["mean_occupation_w1_free"])
    interpreted[group] = dict(original_0405_terminal_pass=old_veto, revised_0450_terminal_pass=revised_veto,
        original_budget_pass_count=report["old_budget_pass_count"], revised_budget_pass_count=report["revised_budget_pass_count"],
        trajectory_count=report["context_bank_evaluations"], unchanged_distribution_gate_b_count=report["gate_b_pass_count"],
        full_gate_count_under_revised_budget=int(gates.full_gate_pass_revised_budget.sum()),
        resource_budget_revision_not_physiological_safety=True)
new = pd.read_csv(selected / "outer_posthoc_amendment/all_channel_context_bank_metrics.csv")
old = pd.read_csv(previous / "outer_posthoc_amendment/all_channel_context_bank_metrics.csv")
labels = new.groupby("channel_index")["channel"].first()
a = old.groupby("channel_index").mean(numeric_only=True)
b = new.groupby("channel_index").mean(numeric_only=True)
frame = pd.DataFrame(dict(channel=labels, direct_actuated=b.direct_actuated.astype(bool)))
for metric in ("time_w1_controlled", "occupation_w1_controlled", "mean_abs_error", "symmetric_sd_ratio"):
    frame[metric + "_previous"] = a[metric]
    frame[metric + "_adapted"] = b[metric]
    frame[metric + "_delta"] = b[metric] - a[metric]
frame.to_csv(selected / "outer_posthoc_amendment/paired_channel_comparison_vs_previous_controller.csv", index_label="channel_index")
tradeoff = frame[frame.occupation_w1_controlled_delta > 1e-12]
interpreted["remaining_occupation_tradeoffs_vs_previous_fresh_controller"] = tradeoff.reset_index().to_dict(orient="records")
for metric in ("time_w1_controlled", "occupation_w1_controlled"):
    interpreted[metric + "_additional_relative_reduction_vs_previous_controller"] = float(1 - b[metric].mean() / a[metric].mean())
interpreted["checkpoint_sha256"] = final["outer_posthoc_amendment"]["checkpoint_sha256"]
interpreted["outer_used_for_training_or_reselection"] = False
interpreted["note"] = "Original gate-C/terminal-veto fields remain false at 0.405; the separately named revised-budget fields reflect only the explicit resource policy change to 0.45. New W1 gains come from parameter adaptation, not the policy relabeling."
(selected / "explicit_budget_and_performance_interpretation.json").write_text(json.dumps(interpreted, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(interpreted, ensure_ascii=False), flush=True)
