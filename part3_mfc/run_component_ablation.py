"""Matched from-scratch HUP060 component removal; no pretrained actor or teacher.

Training imports the existing frozen plant, actor architecture and law loss
read-only. Evaluation accesses run-02 only after a checkpoint is frozen.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
HERE = PROJECT / "artifacts/part3_hup060_actor_wgan_v1/current_comparisons"
SOURCE = PROJECT / "part3_mfc/model_core.py"
PAPER_PAIRED = HERE.parent / "current_full_wgangp/repro_inputs/paper_baseline_paired_comparison.npz"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def source_module():
    spec = importlib.util.spec_from_file_location("isolated_hup060_source", SOURCE)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    private_project = HERE.parent / "current_full_wgangp/repro_source"
    if private_project.is_dir():
        for name in ("SYNTHESIS_CONTRACT", "MODEL_PATH", "SEALED_CALIBRATED", "LEGAL_EVALUATION", "ORIGINAL_DIR", "ORIGINAL_ACTOR", "ORIGINAL_ROLLOUT", "ORIGINAL_SUMMARY", "TEACHER_CHECKPOINT", "SYNTHESIS", "MODEL", "SEALED", "LEGAL"):
            value = getattr(m, name)
            setattr(m, name, private_project / value.relative_to(PROJECT))
        m.PROJECT = private_project
    return m

ACTIVE_ABLATION = "full"


def enforce_deviation_removal(actor):
    if ACTIVE_ABLATION != "no_deviation":
        return
    actor.deviation_feedback_scale = 0.0
    actor.deviation_gain_delta.requires_grad_(False)
    actor.local_deviation_gain_logits.requires_grad_(False)
    for module in (actor.deviation_residual, actor.deviation_markov_residual):
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def setup(m):
    s = np.load(m.SYNTHESIS_CONTRACT)
    if any("future" in k.lower() or "observed" in k.lower() for k in s.files):
        raise RuntimeError("Outcome arrays in training contract")
    model_hash = sha(m.MODEL_PATH)
    if model_hash != str(s["model_sha256"][0]):
        raise RuntimeError("Frozen plant hash mismatch")
    m.load_part2_model_definitions()
    model = joblib.load(m.MODEL_PATH)
    refs = np.asarray(s["run01_reference_fit_pool_scaled"], dtype=np.float64).reshape(30, 256, 36)
    fit, val = [torch.as_tensor(x, dtype=torch.float64) for x in (refs[:15], refs[15:])]
    selected = np.asarray(s["selected_indices"], dtype=np.int64)
    if len(selected) != 13:
        raise RuntimeError("Expected original 13-contact mask")
    world = m.TorchGraphRCSDE(model, selected, m.FS, control_graph_diffusion_time=m.GRAPH_DIFFUSION_TIME,
                            preserve_physical_control_residual=True, dtype=torch.float64, device="cpu")
    adapter = m.FrozenGraphRCMarkovAdapter(world, control_step_scale=m.CONTROL_STEP_SCALE, dtype=torch.float64)
    initial = adapter.initial_state_from_context(np.asarray(s["past_context_scaled"], dtype=np.float64))
    stepper = m.FrozenIctalGraphRCBatchStepper(world, adapter, diffusion_scale=m.DIFFUSION_SCALE)
    mean, variance, scale, weights = m.reference_statistics(fit)
    center, state_scale = m.build_markov_normalization(adapter, initial, fit)
    actor = m.StructuredSplineCovarianceActor(
        stepper, mean, variance, scale, torch.zeros(13, 36, dtype=torch.float64), selected,
        horizon=256, basis_count=8, hidden_size=64, amplitude_limit=1.8, actuator_alpha=1.0,
        residual_scale=0.15, local_gain_initial_fraction=0.0, local_gain_maximum_fraction=0.05,
        markov_feature_center=center, markov_feature_scale=state_scale,
        markov_residual_scale=0.6, markov_hidden_size=96, maximum_slew=None)
    enforce_deviation_removal(actor)
    critic = m.TimeConditionedWassersteinCritic(fit.reshape(-1, 36).mean(dim=0), scale,
                                              horizon_samples=256, hidden_size=128)
    return s, fit, val, selected, adapter, initial, stepper, scale, weights, actor, critic, model_hash

def law(m, rollout, reference, weights, scale, projections, actor, baseline, regularize):
    return m.law_objective(rollout.scaled[:, 1:], rollout.controls, rollout.commands,
        reference, weights, scale, projections, actor, include_parameter_regularization=regularize,
        baseline_sequence=baseline, objective_mode="covariance", smoothness_weight=.20,
        curvature_weight=.10, slew_barrier_weight=0.0, curvature_barrier_weight=0.0,
        maximum_first_difference=.35, maximum_second_difference=.50)

def train(args, m, out):
    if (out / "frozen_actor_wgan.pt").exists():
        raise RuntimeError("Refusing to overwrite a completed experiment")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started = time.perf_counter()
    s, fit, val, selected, adapter, initial, stepper, scale, weights, actor, critic, model_hash = setup(m)
    original_hashes = {str(p): sha(p) for p in (m.MODEL_PATH, m.ORIGINAL_ACTOR, m.ORIGINAL_ROLLOUT, SOURCE)}
    torch.save({"actor_state_dict": actor.state_dict(), "seed": args.seed,
                "initialization": "constructor only; neutral outputs, random hidden layers; no checkpoint loaded"}, out / "initial_actor.pt")
    contract = dict(vars(args))
    contract.update({"model_sha256": model_hash, "source_sha256": sha(SOURCE), "synthesis_sha256": sha(m.SYNTHESIS_CONTRACT),
        "actuators": selected.tolist(), "teacher_loaded": False, "teacher_proximity_weight": 0.0,
        "gradient_penalty": 10.0, "critic_drift": .001, "particles": 32, "horizon_samples": 256,
        "graph_spread": m.GRAPH_DIFFUSION_TIME, "diffusion_scale": m.DIFFUSION_SCALE, "original_hashes_before": original_hashes,
        "component_removal": ACTIVE_ABLATION, "runner_sha256": sha(__file__),
        "same_actor_update_budget_as_full": True, "same_wallclock_budget_as_full": False,
        "selection": "run01 validation only: .5 law/law0+.25 timeW1/timeW10+.25 occW1/occW10",
        "run02_access_during_training": False, "technical_training_seeds": 1,
        "adv_active_from_actor_update": None if ACTIVE_ABLATION == "no_wgan" else 1})
    dump(out / "training_contract.json", contract)
    a_opt = torch.optim.AdamW(actor.parameters(), lr=args.actor_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=args.epochs, eta_min=args.actor_min_lr)
    d_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr, betas=(0.0, .9))
    rng = np.random.default_rng(args.seed)
    directions = rng.normal(size=(36, 16))
    directions /= np.maximum(np.linalg.norm(directions, axis=0), 1e-12)
    projections = torch.as_tensor(directions, dtype=torch.float64)
    indices = tuple(range(15, 256, 16))
    val_noise = m.antithetic_noise(m.VALIDATION_SEED, stepper.q)
    with torch.no_grad():
        val_baseline = m.uncontrolled_particle_rollout(stepper, initial, val_noise)[:, 1:]
        neutral = m.empirical_fp_rollout(stepper, actor, initial, val_noise)
    neutral_error = float((neutral.scaled[:, 1:] - val_baseline).abs().max())
    if neutral_error > 1e-12 or float(neutral.controls.abs().max()) > 1e-12:
        raise RuntimeError("Fresh initialization is not exactly zero control")
    dump(out / "initialization_audit.json", {"neutral_control_peak": float(neutral.controls.abs().max()),
        "initial_to_free_max_abs_error": neutral_error, "trainable_parameters": sum(p.numel() for p in actor.parameters()),
        "no_actor_checkpoint_loaded": True})
    def validate():
        actor.eval()
        with torch.no_grad():
            r = m.empirical_fp_rollout(stepper, actor, initial, val_noise)
            loss, _ = law(m, r, val, weights, scale, projections, actor, val_baseline, False)
        x = r.scaled[:, 1:].numpy()
        return {"validation_law": float(loss), "validation_mean_time_w1": float(m.channelwise_time_w1(x, val.numpy()).mean()),
                "validation_mean_occupation_w1": float(m.channelwise_occupation_w1(x, val.numpy()).mean())}
    initial_val = validate()
    history = [{"epoch": 0, **initial_val, "validation_selection_score": 1.0}]
    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    best_score, best_epoch, best_actor, best_critic = float("inf"), -1, None, None
    with torch.no_grad():
        warm_banks = [m.empirical_fp_rollout(stepper, actor, initial,
            m.antithetic_noise(args.seed + 101 * (b + 1), stepper.q)).scaled[:, 1:].detach() for b in range(3)]
    warm_history = []
    for i in range(0 if ACTIVE_ABLATION == "no_wgan" else 24):
        d_opt.zero_grad(set_to_none=True)
        g = torch.Generator().manual_seed(args.seed + 50000 + i)
        audit = m.balanced_wgan_gp_loss(critic, warm_banks[i % 3], fit, indices,
                    gradient_penalty_weight=10., critic_drift_weight=.001, generator=g)
        audit.loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.)
        d_opt.step()
        warm_history.append({"update": i + 1, "critic_loss": float(audit.loss.detach()),
            "gp": float(audit.gradient_penalty.detach()), "input_gradient_norm": float(audit.mean_gradient_norm.detach())})
    pd.DataFrame(warm_history).to_csv(out / "critic_warmup.csv", index=False)
    print("TRAINING STARTED: fresh actor, no teacher, component removal=" + ACTIVE_ABLATION, flush=True)
    for epoch in range(1, args.epochs + 1):
        noise = m.antithetic_noise(args.seed + 1009 * epoch, stepper.q)
        actor.eval()
        with torch.no_grad():
            detached = m.empirical_fp_rollout(stepper, actor, initial, noise).scaled[:, 1:].detach()
            baseline = m.uncontrolled_particle_rollout(stepper, initial, noise)[:, 1:]
        m.set_requires_grad(actor, False)
        m.set_requires_grad(critic, True)
        critic.train()
        if ACTIVE_ABLATION == "no_wgan":
            from types import SimpleNamespace
            zero = torch.zeros((), dtype=torch.float64)
            audit = SimpleNamespace(loss=zero, estimate=zero, gradient_penalty=zero,
                                    mean_gradient_norm=zero)
            d_grad = zero
        for j in range(0 if ACTIVE_ABLATION == "no_wgan" else args.n_critic):
            d_opt.zero_grad(set_to_none=True)
            g = torch.Generator().manual_seed(args.seed + 100000 * epoch + j)
            audit = m.balanced_wgan_gp_loss(critic, detached, fit, indices,
                        gradient_penalty_weight=10., critic_drift_weight=.001, generator=g)
            if not torch.isfinite(audit.loss):
                raise RuntimeError(f"Nonfinite critic loss at update {epoch}")
            audit.loss.backward()
            d_grad = torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.)
            d_opt.step()
        m.set_requires_grad(critic, False)
        m.set_requires_grad(actor, True)
        enforce_deviation_removal(actor)
        actor.train()
        a_opt.zero_grad(set_to_none=True)
        rollout = m.empirical_fp_rollout(stepper, actor, initial, noise)
        law_loss, terms = law(m, rollout, fit, weights, scale, projections, actor, baseline, True)
        if ACTIVE_ABLATION == "no_wgan":
            adversarial = rollout.controls.new_zeros(())
        else:
            adversarial, _ = m.actor_wasserstein_loss(critic, rollout.scaled[:, 1:], fit, indices)
        total = law_loss + args.adv_weight * adversarial
        if not torch.isfinite(total):
            raise RuntimeError(f"Nonfinite actor loss at update {epoch}")
        total.backward()
        a_grad = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
        a_opt.step()
        used_lr = a_opt.param_groups[0]["lr"]
        scheduler.step()
        row = {"epoch": epoch, "actor_lr": used_lr, "train_total": float(total.detach()),
            "train_law": float(law_loss.detach()), "train_adversarial": float(adversarial.detach()),
            "critic_loss": float(audit.loss.detach()), "critic_estimate": float(audit.estimate.detach()),
            "critic_gp": float(audit.gradient_penalty.detach()), "critic_input_gradient_norm": float(audit.mean_gradient_norm.detach()),
            "actor_gradient_norm_before_clip": float(a_grad), "critic_gradient_norm_before_clip": float(d_grad)}
        row.update({"train_law_" + k: float(v.detach()) for k, v in terms.items()})
        if epoch == 1 or epoch % args.validation_every == 0 or epoch == args.epochs:
            row.update(validate())
            score = .5 * row["validation_law"] / initial_val["validation_law"]
            score += .25 * row["validation_mean_time_w1"] / initial_val["validation_mean_time_w1"]
            score += .25 * row["validation_mean_occupation_w1"] / initial_val["validation_mean_occupation_w1"]
            row["validation_selection_score"] = score
            if score < best_score:
                best_score, best_epoch = score, epoch
                best_actor, best_critic = copy.deepcopy(actor.state_dict()), copy.deepcopy(critic.state_dict())
                torch.save({"actor_state_dict": best_actor, "critic_state_dict": best_critic, "best_epoch": best_epoch,
                    "training_contract": contract, "source_sha256": sha(SOURCE), "model_sha256": model_hash}, out / "best_so_far.pt")
        history.append(row)
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
            elapsed = time.perf_counter() - started
            progress = {"status": "running", "epoch": epoch, "planned_epochs": args.epochs,
                "best_epoch": best_epoch, "best_selection_score": best_score, "elapsed_seconds": elapsed,
                "train_law": row["train_law"], "remaining_seconds_linear_estimate": elapsed / epoch * (args.epochs - epoch)}
            dump(out / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    torch.save({"actor_state_dict": actor.state_dict(), "critic_state_dict": critic.state_dict(),
        "epoch": args.epochs, "training_contract": contract}, out / "last_actor_wgan.pt")
    actor.load_state_dict(best_actor)
    critic.load_state_dict(best_critic)
    selected_val = validate()
    torch.save({"actor_state_dict": best_actor, "critic_state_dict": best_critic, "best_epoch": best_epoch,
        "best_validation_selection_score": best_score, "training_contract": contract, "selected_validation": selected_val,
        "source_sha256": sha(SOURCE), "model_sha256": model_hash}, out / "frozen_actor_wgan.pt")
    if {p: sha(p) for p in original_hashes} != original_hashes:
        raise RuntimeError("Original source or frozen artifact changed")
    summary = {"status": "complete_fresh_hybrid_wgangp_development_experiment", "trained_epochs": args.epochs,
        "best_epoch": best_epoch, "best_selection_score": best_score, "initial_validation": initial_val,
        "selected_validation": selected_val, "elapsed_seconds": time.perf_counter() - started,
        "teacher_checkpoint_loaded": False, "teacher_proximity_weight": 0.,
        "adversarial_active_from_update": None if ACTIVE_ABLATION == "no_wgan" else 1,
        "critic_updates": 0 if ACTIVE_ABLATION == "no_wgan" else 24 + args.n_critic * args.epochs,
        "component_removal": ACTIVE_ABLATION,
        "originals_unchanged": True, "source_sha256": sha(SOURCE),
        "model_sha256": model_hash, "checkpoint_sha256": sha(out / "frozen_actor_wgan.pt"), "not_adopted_in_paper": True}
    dump(out / "training_summary.json", summary)
    dump(out / "progress.json", summary)
    print(json.dumps(summary), flush=True)

def evaluate(args, m, out):
    checkpoint_path = out / "frozen_actor_wgan.pt"
    if not checkpoint_path.is_file() or not (out / "training_summary.json").is_file():
        raise RuntimeError("Training must complete and freeze before evaluation")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if sha(SOURCE) != checkpoint["source_sha256"]:
        raise RuntimeError("Training source changed after freezing")
    s, fit, val, selected, adapter, initial, stepper, scale, weights, actor, critic, model_hash = setup(m)
    if model_hash != checkpoint["model_sha256"]:
        raise RuntimeError("Frozen plant changed")
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    actor.eval()
    m.set_requires_grad(actor, False)
    # Recorded outcome arrays are accessed only after policy freezing above.
    b = np.load(PAPER_PAIRED)
    free, reference = np.asarray(b["uncontrolled_scaled"]), np.asarray(b["reference_validation_scaled"])
    if not np.array_equal(reference, val.numpy()) or not np.array_equal(selected, b["selected_indices"]):
        raise RuntimeError("Paper/experiment reference or mask mismatch")
    normals, reconstruction = m.reconstruct_paired_normals(stepper, initial, free, tolerance=2e-6)
    with torch.no_grad():
        r = m.paired_ictal_batch_rollout(stepper, actor, initial, normals)
    candidate = r.controlled_scaled[:, 1:].numpy()
    parity = float(np.abs(r.uncontrolled_scaled[:, 1:].numpy() - free).max())
    if parity > 1e-6:
        raise RuntimeError("Frozen no-control parity failed")
    tf, of = m.per_time_w1(free, reference), m.occupation_w1(free, reference)
    tc, oc = m.per_time_w1(candidate, reference), m.occupation_w1(candidate, reference)
    paper = np.asarray(b["candidate_controlled_scaled"])
    tp, op = m.per_time_w1(paper, reference), m.occupation_w1(paper, reference)
    channels, mask = np.asarray(b["channels"]).astype(str), np.isin(np.arange(36), selected)
    controls, commands = r.controls.numpy(), r.commands.numpy()
    rm = reference.mean(axis=(0, 1))
    table = pd.DataFrame({"channel": channels, "direct_actuator": mask,
        "time_w1_free": tf, "time_w1_paper": tp, "time_w1_fresh": tc,
        "occupation_w1_free": of, "occupation_w1_paper": op, "occupation_w1_fresh": oc,
        "occupation_reduction_vs_free_pct": 100 * (1 - oc / of), "mean_reference": rm,
        "mean_free": free.mean(axis=(0, 1)), "mean_paper": paper.mean(axis=(0, 1)), "mean_fresh": candidate.mean(axis=(0, 1)),
        "abs_mean_bias_free": abs(free.mean(axis=(0, 1)) - rm), "abs_mean_bias_paper": abs(paper.mean(axis=(0, 1)) - rm),
        "abs_mean_bias_fresh": abs(candidate.mean(axis=(0, 1)) - rm),
        "sd_free": free.std(axis=(0, 1)), "sd_paper": paper.std(axis=(0, 1)),
        "sd_fresh": candidate.std(axis=(0, 1)), "sd_reference": reference.std(axis=(0, 1))})
    ev = out / "evaluation"
    ev.mkdir(exist_ok=True)
    table.to_csv(ev / "channel_metrics.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(ev / "paired_comparison.npz", observed_scaled=b["observed_scaled"], uncontrolled_scaled=free,
        candidate_controlled_scaled=candidate, paper_controlled_scaled=paper, reference_fit_scaled=fit.numpy(),
        reference_validation_scaled=reference, selected_indices=selected, channels=channels,
        candidate_controls=controls, candidate_commands=commands, sampling_rate_hz=np.array([256.]),
        time_w1_candidate=tc, occupation_w1_candidate=oc)
    summary = {"status": "frozen_run02_development_preview_only", "checkpoint_sha256": sha(checkpoint_path),
        "selected_epoch": checkpoint["best_epoch"], "no_control_parity_max_abs_error": parity,
        "noise_reconstruction": reconstruction, "recorded_future_used_for_training_or_selection": False,
        "not_adopted_in_paper": True, "actuators": 13, "channels": 36,
        "mean_time_w1_free": float(tf.mean()), "mean_time_w1_paper": float(tp.mean()), "mean_time_w1_fresh": float(tc.mean()),
        "time_reduction_vs_free_pct": float(100 * (1 - tc.mean() / tf.mean())),
        "mean_occupation_w1_free": float(of.mean()), "mean_occupation_w1_paper": float(op.mean()), "mean_occupation_w1_fresh": float(oc.mean()),
        "occupation_reduction_vs_free_pct": float(100 * (1 - oc.mean() / of.mean())),
        "channels_improved_vs_free_time": int(sum(tc < tf)), "channels_improved_vs_free_occupation": int(sum(oc < of)),
        "channels_improved_vs_paper_time": int(sum(tc < tp)), "channels_improved_vs_paper_occupation": int(sum(oc < op)),
        "nondirect_channels_improved_vs_free_occupation": int(sum(oc[~mask] < of[~mask])),
        "candidate_control_diagnostics": m.control_diagnostics(controls), "paper_control_diagnostics": m.control_diagnostics(b["candidate_controls"]),
        "RAFd3_RAFd4": table.loc[table.channel.isin(["RAFd3", "RAFd4"])].to_dict(orient="records"),
        "training_seeds": 1, "particles_are_not_independent_patient_replicates": True}
    dump(ev / "evaluation_summary.json", m.json_ready(summary))
    print(json.dumps(m.json_ready(summary)), flush=True)

def main():
    global ACTIVE_ABLATION
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "evaluate"])
    p.add_argument("--tag", required=True, help="Explicit unique new run tag")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20261011)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--actor-min-lr", type=float, default=1e-5)
    p.add_argument("--critic-lr", type=float, default=1e-4)
    p.add_argument("--n-critic", type=int, default=3)
    p.add_argument("--adv-weight", type=float, default=.5)
    p.add_argument("--validation-every", type=int, default=50)
    p.add_argument("--ablation", required=True,
                   choices=["no_wgan", "no_graph_spread", "no_deviation"])
    a = p.parse_args()
    ACTIVE_ABLATION = a.ablation
    if a.ablation == "no_wgan":
        a.adv_weight = 0.0
    if any(c in a.tag for c in "/\\:") or min(a.epochs, a.n_critic, a.threads, a.validation_every) < 1:
        raise ValueError("Invalid tag or loop count")
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    out = HERE / "runs" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    m = source_module()
    if a.ablation == "no_graph_spread":
        m.GRAPH_DIFFUSION_TIME = 0.0
    (train if a.mode == "train" else evaluate)(a, m, out)

if __name__ == "__main__":
    main()
