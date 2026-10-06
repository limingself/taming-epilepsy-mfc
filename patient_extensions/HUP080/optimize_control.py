"""Finite, development-only parameter adaptation of the existing HUP080 controller.

No plant, architecture, normalization, mask, reference split, noise coefficient,
or loss component is changed. Existing loss coefficients are tuned. The starting
controller is the previously frozen fresh WGAN-GP update-200 controller, NOT a
neutral initialization. The outer run was already revealed; new outer values
are post-hoc diagnostics and never select an arm/checkpoint.
"""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[2]
HERE = PROJECT / 'patient_results/HUP080_sparse_control_final_v2/current_full_wgangp'
PREVIOUS = HERE / 'portable'
RUNNER = PROJECT / 'patient_extensions/common/run_external_fresh_uniform_reference.py'
START = PREVIOUS / 'starting_checkpoint/frozen_actor_wgan.pt'
EXPECTED_START = "cdffedfea81d3e57943fe4f47bde6155efe3661fbeb325db9fda221213c68fc0"
SEED = 20261011
NEW_RMS_CAP = .45
OLD_RMS_CAP = .405
ARMS = {
    "occupancy_mean_lr1e4": dict(lr=1e-4, min_lr=1e-5, adv=.5,
        weights=dict(mean=15., log_variance=8., worst_log_variance=5., quantile=8.,
        worst_quantile=8., no_harm_mean=30., no_harm_quantile=30., occupancy=6., energy=.05)),
    "occupancy_worst_lr3e5": dict(lr=3e-5, min_lr=5e-6, adv=.25,
        weights=dict(mean=20., log_variance=5., worst_log_variance=5., quantile=10.,
        worst_quantile=10., no_harm_mean=40., no_harm_quantile=40., occupancy=10., energy=.1)),
}
BASE_WEIGHTS = dict(mean=10., log_variance=10., worst_log_variance=5., quantile=8.,
    worst_quantile=5., no_harm_mean=20., no_harm_quantile=20., occupancy=2., energy=.01)

def load_runner():
    spec = importlib.util.spec_from_file_location("hup080_frozen_original_runner", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    if mod.sha(START) != EXPECTED_START:
        raise RuntimeError("The immutable warm-start checkpoint changed")
    return mod

def setup(base):
    state = base.setup("HUP080", SEED)
    checkpoint = torch.load(START, map_location="cpu", weights_only=False)
    state["actor"].load_state_dict(checkpoint["actor_state_dict"], strict=True)
    state["critic"].load_state_dict(checkpoint["critic_state_dict"], strict=True)
    return state

def tuned_law(base, state, rollout, reference, baseline, regularize, arm):
    value, terms = base.law(state, rollout, reference, baseline, regularize)
    for key, weight in arm["weights"].items():
        value = value + (weight - BASE_WEIGHTS[key]) * terms[key]
    return value, terms

def dev_evaluate(base, state, cache, destination=None):
    """Same development ctx5/reference15:30 and fixed noise for every arm."""
    rows, budgets = [], []
    state["actor"].eval()
    with torch.no_grad():
        for slot, (initial, noise, uncontrolled) in enumerate(cache):
            rollout = state["core"].empirical_fp_rollout(state["stepper"], state["actor"], initial, noise)
            controlled = rollout.scaled[:, 1:].numpy()
            free = uncontrolled.numpy()
            reference = state["validation"].numpy()
            time_free, occ_free = base.distances(free, reference)
            time_ctrl, occ_ctrl = base.distances(controlled, reference)
            ref_std = reference.std(axis=(0, 1))
            ctrl_std = controlled.std(axis=(0, 1))
            sd_ratio = np.maximum(ctrl_std / np.maximum(ref_std, 1e-12), ref_std / np.maximum(ctrl_std, 1e-12))
            mean_error = np.abs(controlled.mean(axis=(0, 1)) - reference.mean(axis=(0, 1)))
            u = rollout.controls.numpy()
            energy = float(np.square(u).sum(axis=-1).mean())
            maximum_rms = float(np.sqrt(np.square(u).mean(axis=(0, 1))).max())
            peak = float(np.abs(u).max())
            saturation = float((np.abs(u) >= .95 * 1.8).mean())
            finite = bool(np.isfinite(controlled).all() and np.isfinite(u).all())
            other = finite and energy <= 5.6862 + 1e-9 and peak <= 1.8 + 1e-9 and saturation < .01
            budgets.append(dict(development_run_slot=slot, finite=finite, total_energy=energy,
                maximum_per_actuator_rms=maximum_rms, peak=peak, saturation_fraction=saturation,
                old_budget_pass=bool(other and maximum_rms <= OLD_RMS_CAP + 1e-9),
                revised_budget_pass=bool(other and maximum_rms <= NEW_RMS_CAP + 1e-9)))
            for j in range(state["p"]["n"]):
                rows.append(dict(development_run_slot=slot, channel_index=j,
                    channel=str(state["arrays"]["channels"][j]), direct_actuated=bool(state["mask"][j]),
                    time_w1_free=float(time_free[j]), time_w1_controlled=float(time_ctrl[j]),
                    occupation_w1_free=float(occ_free[j]), occupation_w1_controlled=float(occ_ctrl[j]),
                    mean_abs_error=float(mean_error[j]), symmetric_sd_ratio=float(sd_ratio[j])))
    frame = pd.DataFrame(rows)
    by_channel = frame.groupby("channel_index").mean(numeric_only=True)
    time_ratio = float(frame.time_w1_controlled.mean() / frame.time_w1_free.mean())
    occ_ratio = float(frame.occupation_w1_controlled.mean() / frame.occupation_w1_free.mean())
    per_channel_ratio = .5 * (by_channel.time_w1_controlled / np.maximum(by_channel.time_w1_free, 1e-12)
        + by_channel.occupation_w1_controlled / np.maximum(by_channel.occupation_w1_free, 1e-12))
    worst_decile_ratio = float(np.sort(per_channel_ratio)[-10:].mean())
    sd_mismatch = float(np.mean(np.abs(np.log(by_channel.symmetric_sd_ratio))))
    # Common, dimensionless score. It NEVER uses the arm's reweighted law value.
    score = .4 * time_ratio + .4 * occ_ratio + .1 * worst_decile_ratio + .1 * sd_mismatch
    result = dict(common_dev_selection_score=score, time_ratio=time_ratio, occupation_ratio=occ_ratio,
        worst_decile_ratio=worst_decile_ratio, mean_abs_log_sd_ratio=sd_mismatch,
        mean_time_w1_free=float(frame.time_w1_free.mean()), mean_time_w1_controlled=float(frame.time_w1_controlled.mean()),
        mean_occupation_w1_free=float(frame.occupation_w1_free.mean()), mean_occupation_w1_controlled=float(frame.occupation_w1_controlled.mean()),
        mean_abs_error=float(frame.mean_abs_error.mean()),
        max_mean_abs_error=float(by_channel.mean_abs_error.max()),
        max_symmetric_sd_ratio=float(by_channel.symmetric_sd_ratio.max()),
        time_improved_channels=int((by_channel.time_w1_controlled < by_channel.time_w1_free).sum()),
        occupation_improved_channels=int((by_channel.occupation_w1_controlled < by_channel.occupation_w1_free).sum()),
        both_improved_channels=int(((by_channel.time_w1_controlled < by_channel.time_w1_free)
            & (by_channel.occupation_w1_controlled < by_channel.occupation_w1_free)).sum()),
        old_budget_pass_count=sum(row["old_budget_pass"] for row in budgets),
        revised_budget_pass_count=sum(row["revised_budget_pass"] for row in budgets),
        budget_case_count=len(budgets), worst_per_actuator_rms=max(row["maximum_per_actuator_rms"] for row in budgets),
        worst_energy=max(row["total_energy"] for row in budgets))
    if destination:
        destination.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination / "development_channel_metrics.csv", index=False)
        pd.DataFrame(budgets).to_csv(destination / "development_budget_metrics.csv", index=False)
        base.dump(destination / "development_summary.json", result)
    return result

def historical_budget_report(base):
    reports = {}
    for group in ("ctx6_terminal_veto", "outer_posthoc_amendment"):
        path = START.parent / group / "trajectory_safety_metrics.csv"
        frame = pd.read_csv(path)
        finite = frame.finite.astype(bool)
        other = finite & (frame.total_energy <= 5.6862 + 1e-9) & (frame.control_peak <= 1.8 + 1e-9) & (frame.saturation_fraction < .01)
        reports[group] = dict(cases=len(frame), old_pass_count=int((other & (frame.maximum_per_actuator_rms <= .405 + 1e-9)).sum()),
            revised_pass_count=int((other & (frame.maximum_per_actuator_rms <= .45 + 1e-9)).sum()),
            rms_min=float(frame.maximum_per_actuator_rms.min()), rms_max=float(frame.maximum_per_actuator_rms.max()),
            max_energy=float(frame.total_energy.max()), max_peak=float(frame.control_peak.max()), max_saturation=float(frame.saturation_fraction.max()),
            performance_changed=False, classification="post-hoc resource budget revision, not a new performance improvement")
    base.dump(HERE / "previous_checkpoint_budget_reinterpretation.json", dict(old_rms_cap=.405,
        revised_rms_cap=.45, unchanged_energy_cap=5.6862, unchanged_peak_cap=1.8, unchanged_saturation_cap=.01,
        immutable_checkpoint_sha256=EXPECTED_START, evaluations=reports))
    return reports

def preflight(base):
    state = setup(base)
    cache = base.validation_cache(state)
    metrics = dev_evaluate(base, state, cache, HERE / "starting_development")
    report = dict(subject="HUP080", same_model_architecture_and_loss_components=True, model_sha256=base.MODEL_HASHES["HUP080"],
        actor_warm_start=True, original_training_selected_update=200, warm_start_sha256=EXPECTED_START,
        mask_direct_count=int(state["mask"].sum()), channels=96, heat_kernel_time=0., diffusion_scale=.79451175,
        control_step_scale=.5, amplitude_limit=1.8, revised_per_actuator_rms_cap=.45, old_per_actuator_rms_cap=.405,
        energy_cap=5.6862, saturation_cap=.01, development_reference_shape=list(state["validation"].shape),
        fit_contexts=state["fit_ledger"], selection_context_index=5, source_input_hashes=state["hashes"],
        arms=ARMS, planned_updates_per_arm=150, validate_every=25,
        selection_rule="common .4*time_ratio + .4*occupation_ratio + .1*worst_channel_decile_ratio + .1*mean_abs_log_SD_ratio, budget-feasible preferred; starting checkpoint retained if no lower score",
        development_validation_noise_seed=20260922,
        outer_already_revealed=True, outer_used_for_selection=False,
        selection_frozen_before_new_ctx6_or_outer_evaluation=True, no_new_loss_figure=True,
        starting_development_metrics=metrics, historical_budget_reinterpretation=historical_budget_report(base))
    base.dump(HERE / "optimization_contract.json", report)
    print(json.dumps(report), flush=True)

def train_arm(base, name, updates):
    out = HERE / "arms" / name
    if out.exists():
        raise RuntimeError(f"Refuse arm overwrite: {out}")
    out.mkdir(parents=True)
    state = setup(base)
    actor, critic, core = state["actor"], state["critic"], state["core"]
    arm = ARMS[name]
    cache = base.validation_cache(state)
    start_dev = dev_evaluate(base, state, cache, out / "initial_validation")
    history = [dict(update=0, **start_dev)]
    best_score = start_dev["common_dev_selection_score"]
    best_update = 0
    best_feasible = start_dev["revised_budget_pass_count"] == start_dev["budget_case_count"]
    best_actor, best_critic = copy.deepcopy(actor.state_dict()), copy.deepcopy(critic.state_dict())
    optimizer = torch.optim.AdamW(actor.parameters(), lr=arm["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=arm["min_lr"])
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0., .9))
    time_indices = tuple(range(15, 256, 16))
    started = time.perf_counter()
    for update in range(1, updates + 1):
        slot = (update - 1) % len(state["initials"])
        initial = state["initials"][slot]
        noise = core.antithetic_noise(SEED + 30000000 + 1009 * update, state["stepper"].q)
        actor.eval()
        with torch.no_grad():
            detached = core.empirical_fp_rollout(state["stepper"], actor, initial, noise).scaled[:, 1:].detach()
            free = core.uncontrolled_particle_rollout(state["stepper"], initial, noise)[:, 1:]
        core.set_requires_grad(actor, False)
        core.set_requires_grad(critic, True)
        critic.train()
        for critic_index in range(3):
            critic_optimizer.zero_grad(set_to_none=True)
            reference, included = base.uniform_reference(state["fit"], SEED + 90000000 + 1000 * update + critic_index)
            critic_loss = core.balanced_wgan_gp_loss(critic, detached, reference, time_indices,
                gradient_penalty_weight=10., critic_drift_weight=.001,
                generator=torch.Generator().manual_seed(SEED + 40000000 + 100000 * update + critic_index))
            if not torch.isfinite(critic_loss.loss):
                raise RuntimeError("Nonfinite critic objective")
            critic_loss.loss.backward()
            dnorm = torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.)
            critic_optimizer.step()
        core.set_requires_grad(critic, False)
        core.set_requires_grad(actor, True)
        actor.train()
        optimizer.zero_grad(set_to_none=True)
        rollout = core.empirical_fp_rollout(state["stepper"], actor, initial, noise)
        law_value, terms = tuned_law(base, state, rollout, state["fit"], free, True, arm)
        adv, _ = core.actor_wasserstein_loss(critic, rollout.scaled[:, 1:], state["fit"], time_indices)
        total = law_value + arm["adv"] * adv
        if not torch.isfinite(total):
            raise RuntimeError("Nonfinite actor objective")
        total.backward()
        anorm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
        optimizer.step()
        used_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        row = dict(update=update, fit_context_slot=slot, actor_lr=used_lr, train_total=float(total.detach()),
            train_law=float(law_value.detach()), train_adversarial=float(adv.detach()),
            critic_loss=float(critic_loss.loss.detach()), actor_grad_before_clip=float(anorm), critic_grad_before_clip=float(dnorm))
        row.update({"law_" + key: float(value.detach()) for key, value in terms.items()})
        if update % 25 == 0 or update == updates:
            dev = dev_evaluate(base, state, cache)
            row.update(dev)
            feasible = dev["revised_budget_pass_count"] == dev["budget_case_count"]
            score = dev["common_dev_selection_score"]
            if (feasible and not best_feasible) or (feasible == best_feasible and score < best_score):
                best_feasible, best_score, best_update = feasible, score, update
                best_actor, best_critic = copy.deepcopy(actor.state_dict()), copy.deepcopy(critic.state_dict())
                torch.save(dict(actor_state_dict=best_actor, critic_state_dict=best_critic,
                    best_update=best_update, common_dev_selection_score=best_score), out / "best_so_far.pt")
        history.append(row)
        if update % 10 == 0 or update == 1 or update == updates:
            pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
            progress = dict(status="running", arm=name, update=update, planned_updates=updates, best_update=best_update,
                best_common_dev_score=best_score, best_revised_budget_feasible=best_feasible,
                elapsed_seconds=time.perf_counter() - started, outer_opened_during_training=False)
            base.dump(out / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    actor.load_state_dict(best_actor)
    critic.load_state_dict(best_critic)
    final_dev = dev_evaluate(base, state, cache, out / "selected_development")
    torch.save(dict(actor_state_dict=best_actor, critic_state_dict=best_critic, best_update=best_update,
        warm_start_checkpoint_sha256=EXPECTED_START, model_sha256=base.MODEL_HASHES["HUP080"], arm_parameters=arm,
        validation=final_dev, input_hashes=state["hashes"], outer_used_for_selection=False), out / "frozen_actor.pt")
    if any(base.sha(path) != digest for path, digest in state["hashes"].items()) or base.sha(START) != EXPECTED_START:
        raise RuntimeError("Original frozen inputs or checkpoint changed")
    summary = dict(status="completed", arm=name, added_actor_updates=updates, selected_added_update=best_update,
        starting_fresh_selected_update=200, warm_start=True, common_dev_selection_score=best_score,
        revised_budget_feasible=best_feasible, checkpoint_sha256=base.sha(out / "frozen_actor.pt"),
        selected_development=final_dev, outer_used_for_selection=False, elapsed_seconds=time.perf_counter() - started)
    base.dump(out / "training_summary.json", summary)
    base.dump(out / "progress.json", summary)
    print(json.dumps(summary), flush=True)

def freeze_selection(base):
    rows = []
    for name in ARMS:
        path = HERE / "arms" / name / "training_summary.json"
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    selected = min(rows, key=lambda row: (not row["revised_budget_feasible"], row["common_dev_selection_score"]))
    target = HERE / "selected"
    if target.exists():
        raise RuntimeError("Refuse final selection overwrite")
    target.mkdir(parents=True)
    chosen = HERE / "arms" / selected["arm"] / "frozen_actor.pt"
    # Byte-preserving copy is an output snapshot, not an edit of a model.
    import shutil
    shutil.copy2(chosen, target / "frozen_actor.pt")
    receipt = dict(selected_arm=selected["arm"], selected_added_update=selected["selected_added_update"],
        common_dev_selection_score=selected["common_dev_selection_score"], revised_budget_feasible=selected["revised_budget_feasible"],
        checkpoint_sha256=base.sha(target / "frozen_actor.pt"), arms=rows,
        outer_used_for_selection=False, ctx6_used_for_selection=False,
        classification="post-hoc development-only parameter adaptation of the already trained fresh controller")
    base.dump(target / "selection_receipt.json", receipt)
    print(json.dumps(receipt), flush=True)

def evaluate_frozen(base):
    target = HERE / "selected"
    receipt = json.loads((target / "selection_receipt.json").read_text(encoding="utf-8"))
    path = target / "frozen_actor.pt"
    if base.sha(path) != receipt["checkpoint_sha256"]:
        raise RuntimeError("Frozen selection changed")
    state = setup(base)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state["actor"].load_state_dict(checkpoint["actor_state_dict"], strict=True)
    state["actor"].eval()
    args = SimpleNamespace(subject="HUP080", seed=SEED)
    # The original evaluator is run unchanged with its .405 gate. Revised budget
    # classification is saved separately so historical failures are never erased.
    initials, observed = [], []
    for run in state["p"]["development_runs"]:
        ictal = state["arrays"]["ictal_" + run.replace("-", "_")]
        stop = base.boundary(ictal, 6)
        initials.append(state["adapter"].initial_state_from_context(ictal[stop - 256:stop]))
        observed.append(state["model"].transform.scaler.transform(ictal[stop:stop + 256]))
    base.evaluate_set(args, state, initials, state["validation"].numpy(), observed,
        "all-development", "ctx6_terminal_veto", target / "ctx6_terminal_veto")
    with np.load(state["p"]["outer"], allow_pickle=False) as archive:
        outer = {key: np.asarray(archive[key]) for key in archive.files}
    if not np.array_equal(outer["channels"].astype(str), state["arrays"]["channels"].astype(str)):
        raise RuntimeError("Outer channel order changed")
    ictal = outer["ictal"]
    reference = state["model"].transform.scaler.transform(outer["preictal_reference"]).reshape(-1, 256, 96)
    initials, observed = [], []
    for index in range(8):
        stop = base.boundary(ictal, index)
        initials.append(state["adapter"].initial_state_from_context(ictal[stop - 256:stop]))
        observed.append(state["model"].transform.scaler.transform(ictal[stop:stop + 256]))
    base.evaluate_set(args, state, initials, reference, observed,
        state["p"]["outer_fold"], state["p"]["outer_stage"], target / "outer_posthoc_amendment", display=7)
    summaries = {}
    for group in ("ctx6_terminal_veto", "outer_posthoc_amendment"):
        folder = target / group
        old = json.loads((folder / "evaluation_summary.json").read_text(encoding="utf-8"))
        frame = pd.read_csv(folder / "trajectory_safety_metrics.csv")
        other = frame.finite.astype(bool) & (frame.total_energy <= 5.6862 + 1e-9) & (frame.control_peak <= 1.8 + 1e-9) & (frame.saturation_fraction < .01)
        frame["gate_c_old_0405"] = other & (frame.maximum_per_actuator_rms <= .405 + 1e-9)
        frame["gate_c_revised_0450"] = other & (frame.maximum_per_actuator_rms <= .45 + 1e-9)
        frame.to_csv(folder / "trajectory_budget_old_and_revised.csv", index=False)
        revised = dict(old, old_per_actuator_rms_cap=.405, revised_per_actuator_rms_cap=.45,
            old_budget_pass_count=int(frame.gate_c_old_0405.sum()), revised_budget_pass_count=int(frame.gate_c_revised_0450.sum()),
            revised_budget_pass=bool(frame.gate_c_revised_0450.all()), max_per_actuator_rms=float(frame.maximum_per_actuator_rms.max()),
            worst_total_energy=float(frame.total_energy.max()), max_control_peak=float(frame.control_peak.max()),
            max_saturation_fraction=float(frame.saturation_fraction.max()),
            time_w1_relative_reduction=1 - old["mean_time_w1_controlled"] / old["mean_time_w1_free"],
            occupation_w1_relative_reduction=1 - old["mean_occupation_w1_controlled"] / old["mean_occupation_w1_free"],
            budget_revision_classification="post-hoc increase of allowable per-actuator RMS; not clinical safety or unchanged historical gate",
            selected_arm=receipt["selected_arm"], selected_added_update=receipt["selected_added_update"],
            checkpoint_sha256=receipt["checkpoint_sha256"], outer_used_for_selection=False,
            ctx6_used_for_reselection=False, warm_start=True, not_adopted_in_paper=True)
        base.dump(folder / "evaluation_revised_budget_summary.json", revised)
        summaries[group] = revised
    old_report = json.loads(state["p"]["old_outer_report"].read_text(encoding="utf-8"))
    old = old_report.get("metrics", old_report.get("summary", old_report))
    free_parity = max(abs(summaries["outer_posthoc_amendment"][key] - old[key])
        for key in ("mean_time_w1_free", "mean_occupation_w1_free"))
    if free_parity > 1e-7:
        raise RuntimeError("Uncontrolled plant-law parity failed")
    if any(base.sha(path) != digest for path, digest in state["hashes"].items()) or base.sha(START) != EXPECTED_START:
        raise RuntimeError("Original frozen inputs/checkpoint changed")
    summaries["immutable_inputs_verified"] = True
    summaries["old_uncontrolled_metric_parity_max_error"] = free_parity
    base.dump(target / "final_posthoc_report.json", summaries)
    print(json.dumps(summaries), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "screen", "select", "evaluate", "screen-evaluate"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    base = load_runner()
    HERE.mkdir(parents=True, exist_ok=True)
    if args.mode == "preflight":
        preflight(base)
    elif args.mode in ("screen", "screen-evaluate"):
        if not (HERE / "optimization_contract.json").is_file():
            raise RuntimeError("Preflight required")
        for name in ARMS:
            train_arm(base, name, 150)
        freeze_selection(base)
        if args.mode == "screen-evaluate":
            evaluate_frozen(base)
    elif args.mode == "select":
        freeze_selection(base)
    elif args.mode == "evaluate":
        evaluate_frozen(base)

if __name__ == "__main__":
    main()
