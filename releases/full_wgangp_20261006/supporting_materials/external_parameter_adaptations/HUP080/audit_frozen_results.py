"""Post-freeze independent numerical checks; never selects a controller."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parent
SELECTED = ROOT / "selected"
PREVIOUS = Path(r"C:\Users\LiMing\Documents\改论文\.fresh_wgangp_figures_20261006\external\runs\HUP080_seed20261011_u1000_uniformref")
checks = []

def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

def require(name, condition, value=None):
    checks.append(dict(name=name, passed=bool(condition), value=value))
    if not condition:
        raise AssertionError(name)

receipt = json.loads((SELECTED / "selection_receipt.json").read_text(encoding="utf-8"))
final = json.loads((SELECTED / "final_posthoc_report.json").read_text(encoding="utf-8"))
execution = json.loads((ROOT / "execution_manifest.json").read_text(encoding="utf-8"))
require("Optimizer source exact executed snapshot", sha(ROOT / "optimize_hup080.py") == execution["source_script_sha256"])
require("Original runner untouched", sha(PREVIOUS.parent.parent / "run_external_fresh_uniform_reference.py") == execution["immutable_base_runner_sha256"])
require("Starting checkpoint untouched", sha(PREVIOUS / "frozen_actor_wgan.pt") == execution["starting_checkpoint_sha256"])
require("Selection checkpoint SHA", sha(SELECTED / "frozen_actor.pt") == receipt["checkpoint_sha256"])
require("No outer / ctx6 selection", receipt["outer_used_for_selection"] is False and receipt["ctx6_used_for_selection"] is False)

arm_summaries = []
for folder in sorted((ROOT / "arms").iterdir()):
    summary = json.loads((folder / "training_summary.json").read_text(encoding="utf-8"))
    arm_summaries.append(summary)
    history = pd.read_csv(folder / "training_history.csv")
    require(folder.name + " finite full 150-update history", len(history) == 151 and np.array_equal(history["update"].to_numpy(), np.arange(151)))
    require(folder.name + " finite actor/critic objective", np.isfinite(history.iloc[1:][["train_total", "train_law", "train_adversarial", "critic_loss"]].to_numpy()).all())
    require(folder.name + " complete 15-context rotation", history.iloc[1:].fit_context_slot.value_counts().to_dict() == {i: 10 for i in range(15)})
    validated = history[history.common_dev_selection_score.notna()].copy()
    feasible = validated.revised_budget_pass_count == validated.budget_case_count
    validated["infeasible"] = ~feasible
    expected = validated.sort_values(["infeasible", "common_dev_selection_score", "update"]).iloc[0]
    require(folder.name + " exact development-only selected update", int(expected["update"]) == summary["selected_added_update"])
    require(folder.name + " selected checkpoint binding", sha(folder / "frozen_actor.pt") == summary["checkpoint_sha256"])
    frame = pd.read_csv(folder / "selected_development/development_channel_metrics.csv")
    by = frame.groupby("channel_index").mean(numeric_only=True)
    tr = frame.time_w1_controlled.mean() / frame.time_w1_free.mean()
    ort = frame.occupation_w1_controlled.mean() / frame.occupation_w1_free.mean()
    channel = .5 * (by.time_w1_controlled / by.time_w1_free + by.occupation_w1_controlled / by.occupation_w1_free)
    common = .4 * tr + .4 * ort + .1 * np.sort(channel.to_numpy())[-10:].mean() + .1 * np.log(by.symmetric_sd_ratio).mean()
    require(folder.name + " independently reconstructed common dev score", abs(common - summary["common_dev_selection_score"]) < 1e-12, float(common))
best = min(arm_summaries, key=lambda item: (not item["revised_budget_feasible"], item["common_dev_selection_score"]))
require("Selected arm global minimum among completed comparable dev outcomes", receipt["selected_arm"] == best["arm"])
require("Selected copy exact chosen weights bytes", receipt["checkpoint_sha256"] == best["checkpoint_sha256"])

budget_comparison = {}
for group, expected_cases in (("ctx6_terminal_veto", 9), ("outer_posthoc_amendment", 24)):
    folder = SELECTED / group
    summary = final[group]
    safety = pd.read_csv(folder / "trajectory_safety_metrics.csv")
    all_channels = pd.read_csv(folder / "all_channel_context_bank_metrics.csv")
    require(group + " complete trajectory table", len(safety) == expected_cases)
    require(group + " complete channel table", len(all_channels) == expected_cases * 96)
    require(group + " finite recorded metrics", np.isfinite(safety.select_dtypes(include="number").to_numpy()).all())
    other = safety.finite.astype(bool) & (safety.total_energy <= 5.6862 + 1e-9) & (safety.control_peak <= 1.8 + 1e-9) & (safety.saturation_fraction < .01)
    old = other & (safety.maximum_per_actuator_rms <= .405 + 1e-9)
    revised = other & (safety.maximum_per_actuator_rms <= .45 + 1e-9)
    require(group + " old gates retained", np.array_equal(old.to_numpy(), safety.gate_c.to_numpy()))
    require(group + " old pass count reconstructed", int(old.sum()) == summary["old_budget_pass_count"])
    require(group + " revised pass count reconstructed", int(revised.sum()) == summary["revised_budget_pass_count"])
    require(group + " metrics linked frozen weights", summary["checkpoint_sha256"] == receipt["checkpoint_sha256"])
    for metric in ("mean_time_w1_free", "mean_time_w1_controlled", "mean_occupation_w1_free", "mean_occupation_w1_controlled", "control_rms", "total_energy"):
        require(group + " " + metric + " summary from all trajectories", abs(summary[metric] - safety[metric].mean()) < 1e-12)
    for metric in ("time_w1_free", "time_w1_controlled", "occupation_w1_free", "occupation_w1_controlled"):
        require(group + " " + metric + " equals channel-average table", abs(all_channels[metric].mean() - summary["mean_" + metric]) < 1e-12)
    budget_comparison[group] = dict(old_pass_count=int(old.sum()), revised_pass_count=int(revised.sum()),
        total_cases=expected_cases, max_actuator_rms=float(safety.maximum_per_actuator_rms.max()),
        max_energy=float(safety.total_energy.max()), max_peak=float(safety.control_peak.max()), max_saturation=float(safety.saturation_fraction.max()))

with np.load(SELECTED / "outer_posthoc_amendment/display_context_rollout.npz", allow_pickle=False) as archive:
    display = {name: np.asarray(archive[name]) for name in archive.files}
with np.load(PREVIOUS / "outer_posthoc_amendment/display_context_rollout.npz", allow_pickle=False) as archive:
    old_display = {name: np.asarray(archive[name]) for name in archive.files}
require("Original context7 bank0 display", int(display["display_context_index"]) == 7 and int(display["display_crn_bank"]) == 0)
require("Original 96/76 mask", display["direct_mask"].size == 96 and display["direct_mask"].sum() == 76)
require("Full mask exact previous", np.array_equal(display["direct_mask"], old_display["direct_mask"]))
for key in ("reference_standardized", "observed_standardized", "free_standardized", "standard_normal", "channels"):
    require("Unchanged paired display source " + key, np.array_equal(display[key], old_display[key]))
reference = display["reference_standardized"]
source = pd.read_csv(SELECTED / "outer_posthoc_amendment/all_channel_context_bank_metrics.csv")
visible = source[(source.context_index == 7) & (source.crn_bank == 0)].sort_values("channel_index")
errors = {}
for series, label in ((display["free_standardized"], "free"), (display["controlled_standardized"], "controlled")):
    occ = np.array([wasserstein_distance(series[..., j].ravel(), reference[..., j].ravel()) for j in range(96)])
    time = np.array([np.mean([wasserstein_distance(series[:, t, j], reference[:, t, j]) for t in range(256)]) for j in range(96)])
    for kind, vector in (("occupation", occ), ("time", time)):
        error = float(np.abs(vector - visible[kind + "_w1_" + label].to_numpy()).max())
        errors[kind + "_" + label] = error
        require("Independent scipy displayed " + kind + " W1 " + label, error < 1e-12, error)
u = display["controls"]
budget_values = dict(control_rms=float(np.sqrt(np.square(u).mean())),
    maximum_per_actuator_rms=float(np.sqrt(np.square(u).mean(axis=(0, 1))).max()),
    total_energy=float(np.square(u).sum(axis=-1).mean()), control_peak=float(np.abs(u).max()),
    saturation_fraction=float((np.abs(u) >= .95 * 1.8).mean()))
visible_budget = pd.read_csv(SELECTED / "outer_posthoc_amendment/trajectory_safety_metrics.csv")
visible_budget = visible_budget[(visible_budget.context_index == 7) & (visible_budget.crn_bank == 0)].iloc[0]
for key, value in budget_values.items():
    require("Independent displayed control budget " + key, abs(value - visible_budget[key]) < 1e-12)

old = pd.read_csv(PREVIOUS / "outer_posthoc_amendment/all_channel_context_bank_metrics.csv").groupby("channel_index").mean(numeric_only=True)
new = source.groupby("channel_index").mean(numeric_only=True)
comparison = {}
for metric in ("time_w1_controlled", "occupation_w1_controlled", "mean_abs_error", "symmetric_sd_ratio"):
    comparison[metric] = dict(previous_mean=float(old[metric].mean()), adapted_mean=float(new[metric].mean()),
        channels_lower=int((new[metric] < old[metric] - 1e-12).sum()), channels_higher=int((new[metric] > old[metric] + 1e-12).sum()))
both = (new.time_w1_controlled < new.time_w1_free) & (new.occupation_w1_controlled < new.occupation_w1_free)
indirect = ~display["direct_mask"].astype(bool)
both_previous = (new.time_w1_controlled < old.time_w1_controlled) & (new.occupation_w1_controlled < old.occupation_w1_controlled)
comparison["both_improved_vs_free_crosscase_mean"] = int(both.sum())
comparison["indirect_both_improved_vs_free_crosscase_mean"] = int(both.to_numpy()[indirect].sum())
comparison["both_improved_vs_previous_controller_crosscase_mean"] = int(both_previous.sum())
comparison["indirect_both_improved_vs_previous_controller_crosscase_mean"] = int(both_previous.to_numpy()[indirect].sum())
comparison["persistent_both_improved_vs_free_all_24_conditions"] = int(source.groupby("channel_index").both_time_and_occupation_improved.all().sum())
result = dict(status="passed", checks_passed=len(checks), checks=checks, budget_comparison=budget_comparison,
    independent_display_empirical_W1_max_errors=errors, outer_paired_comparison_vs_previous_fresh_controller=comparison,
    warm_start=True, outer_used_for_selection=False, no_new_loss_plot=True,
    not_yet_author_approved_or_adopted_external_results=True)
(ROOT / "frozen_result_independent_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False), flush=True)
