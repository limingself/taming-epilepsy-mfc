"""Isolated HUP080 uniform-reference all-through hybrid WGAN-GP training.

Never loads a teacher or actor checkpoint during setup/training. The official
Part-II models, Part-I masks, references and development splits remain frozen.
Every 'update' sees one rotating development context, not an entire epoch.
Outer arrays are opened only by evaluate(), after policy freezing, and are
explicitly a post-hoc amendment to already-revealed outer evaluations.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / ".runtime"
RUNTIME.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(RUNTIME / "matplotlib")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import joblib
import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / "part3_mfc/model_core.py"
PATIENT_BASE = PROJECT / "patient_results"
MODEL_HASHES = {
    "HUP065": "278591950c0938458fca1fc9dfdfb943eb4d413d1ad95a0f13fa59630b39fa89",
    "HUP080": "10272ee6883777b967f109b57c7dba74d6ef45607053113194fc1274ae0b9734",
}
EXPECTED_SOURCE_HASH = "57da573dbe300ad6bd585fbd69e46102d9f315dba8862e63c7d3e8320b6943e1"
PROTOCOL = "HUP060_consistent_sparse_rerun_common_v1"

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def ready(value):
    if isinstance(value, dict): return {str(k): ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [ready(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    return value

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(ready(value), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)

def load_core():
    if sha(SOURCE) != EXPECTED_SOURCE_HASH:
        raise RuntimeError("Canonical law/plant source changed; refusing silent substitution")
    spec = importlib.util.spec_from_file_location("external_fresh_canonical_part3", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.load_part2_model_definitions()
    return module

def paths(subject):
    if subject == "HUP065":
        base = PATIENT_BASE / "HUP065_sparse_control_final_v1" / "frozen_run"
        science = base / "science_run" / "p"
        refit = science / "ALLDEV_REFIT"
        return dict(base=base, config=base / "config.json",
            development=science / "DEV_MATERIALIZE/development_arrays.npz",
            model=refit / "selected_model.joblib", network=refit / "network.npz",
            plant=refit / "part2_plant_adjacency.npz", refit=refit / "refit_receipt.json",
            outer=science / "OUTER/run03_sealed_segments.npz",
            old_outer_report=science / "OUTER/run03_retrospective_report.json",
            old_display=science / "OUTER/run03_frozen_display_context07_bank0_rollout.npz",
            development_runs=["run-01", "run-02"], n=64, m=23, tau=.2, energy_cap=3.7908,
            outer_run="run-03", outer_fold="sealed-run03-retrospective", outer_stage="outer_once_8contexts_x_3banks")
    base = PATIENT_BASE / "HUP080_sparse_control_final_v2" / "frozen_run"
    science = base / "science_run/artifacts"
    refit = science / "04_final_refit"
    return dict(base=base, config=base / "config.json",
        development=science / "01_prepare_development/development_arrays.npz",
        model=refit / "selected_model.joblib", network=refit / "part1_selection_network.npz",
        plant=refit / "part2_plant_network.npz", refit=refit / "refit_receipt.json",
        outer=science / "15_run04_once/run04_sealed_segments.npz",
        old_outer_report=science / "15_run04_once/run04_retrospective_report.json",
        old_display=science / "15_run04_once/run04_frozen_rollout.npz",
        development_runs=["run-01", "run-02", "run-03"], n=96, m=76, tau=0., energy_cap=5.6862,
        outer_run="run-04", outer_fold="sealed-run04-retrospective", outer_stage="outer_once")

def reference_statistics(reference):
    channels = reference.shape[-1]
    variance = reference.reshape(-1, channels).var(dim=0, unbiased=False)
    scale = variance.clamp_min(.1 ** 2).sqrt()
    inverse = 1 / variance.clamp_min(.1 ** 2)
    inverse = inverse / inverse.mean()
    weights = (.5 + .5 * inverse).clamp(.5, 3.)
    weights = weights / weights.mean()
    return reference.mean(dim=0), (.25 * reference.var(dim=0, unbiased=False) + .75 * variance[None]).clamp_min(1e-5), scale, weights

def markov_normalization(adapter, initials, reference):
    # Same dimension-general normalization as the official patient controller.
    values = torch.stack([state.detach() for state in initials])
    center = values.mean(dim=0)
    scale = torch.ones_like(center)
    scale[adapter.slices.reservoir] = values[:, adapter.slices.reservoir].std(dim=0, unbiased=False).clamp_min(.25)
    latent = (reference - adapter.pca_mean[None, None]) @ adapter.components.T
    scale[adapter.slices.history] = latent.reshape(-1, adapter.q).std(dim=0, unbiased=False).clamp_min(.1).repeat(adapter.history_length)
    topology = (reference @ adapter.adjacency.T) @ adapter.components.T
    scale[adapter.slices.topology] = topology.reshape(-1, adapter.q).std(dim=0, unbiased=False).clamp_min(.1)
    return center, scale

def boundary(sequence, index):
    positions = np.rint(np.linspace(512, len(sequence) - 256, 8)).astype(int)
    return int(positions[index])

def setup(subject, seed):
    p = paths(subject)
    hashes = {str(p[k]): sha(p[k]) for k in ("config", "development", "model", "network", "plant", "refit")}
    if hashes[str(p["model"])] != MODEL_HASHES[subject]:
        raise RuntimeError("Official frozen predictive plant hash mismatch")
    core = load_core()
    model = joblib.load(p["model"])
    with np.load(p["development"], allow_pickle=False) as archive:
        arrays = {k: np.asarray(archive[k]) for k in archive.files}
    with np.load(p["network"], allow_pickle=False) as archive:
        network = {k: np.asarray(archive[k]) for k in archive.files}
    mask = np.asarray(network["target_mask"], dtype=bool)
    selected = np.flatnonzero(mask)
    if mask.size != p["n"] or len(selected) != p["m"]:
        raise RuntimeError("Official Part-I actuator mask dimension/count changed")
    with np.load(p["plant"], allow_pickle=False) as archive:
        if subject == "HUP065":
            raw_adj, norm_adj = archive["adjacency_input"], archive["adjacency_normalized"]
        else:
            raw_adj = archive["adjacency"]
            norm_adj = np.asarray(model.adjacency)
    if not np.array_equal(np.asarray(model.adjacency_input), raw_adj):
        raise RuntimeError("Model no longer matches the official Part-II adjacency")
    if not np.array_equal(np.asarray(model.adjacency), norm_adj):
        raise RuntimeError("Official normalized plant graph changed")
    fit_blocks, validation_blocks, fit_contexts, selection_contexts, fit_ledger = [], [], [], [], []
    for run in p["development_runs"]:
        key = run.replace("-", "_")
        reference_raw = np.asarray(arrays["preictal_reference_" + key], dtype=np.float64)
        reference = model.transform.scaler.transform(reference_raw).reshape(-1, 256, p["n"])
        if reference.shape[0] < 30: raise RuntimeError("Insufficient official reference paths")
        fit_blocks.append(reference[:15].copy())
        validation_blocks.append(reference[15:30].copy())
        ictal = np.asarray(arrays["ictal_" + key], dtype=np.float64)
        for index in range(5):
            stop = boundary(ictal, index)
            fit_contexts.append(ictal[stop - 256:stop].copy())
            fit_ledger.append(dict(run=run, context_index=index, boundary_sample=stop))
        stop = boundary(ictal, 5)
        selection_contexts.append(ictal[stop - 256:stop].copy())
    fit = torch.as_tensor(np.concatenate(fit_blocks), dtype=torch.float64)
    validation = torch.as_tensor(np.concatenate(validation_blocks), dtype=torch.float64)
    torch.manual_seed(seed)
    np.random.seed(seed)
    world = core.TorchGraphRCSDE(model, selected, 256., control_graph_diffusion_time=p["tau"], preserve_physical_control_residual=True, dtype=torch.float64, device="cpu")
    if not np.array_equal(world.adjacency.numpy(), np.asarray(model.adjacency)):
        raise RuntimeError("Controller substituted a different dynamics graph")
    adapter = core.FrozenGraphRCMarkovAdapter(world, control_step_scale=.5, dtype=torch.float64)
    stepper = core.FrozenIctalGraphRCBatchStepper(world, adapter, diffusion_scale=.79451175)
    initials = [adapter.initial_state_from_context(item) for item in fit_contexts]
    selections = [adapter.initial_state_from_context(item) for item in selection_contexts]
    mean, variance, scale, weights = reference_statistics(fit)
    center, state_scale = markov_normalization(adapter, initials, fit)
    actor = core.StructuredSplineCovarianceActor(stepper, mean, variance, scale,
        torch.zeros(p["m"], p["n"], dtype=torch.float64), selected,
        horizon=256, basis_count=8, hidden_size=64, amplitude_limit=1.8, actuator_alpha=1.,
        residual_scale=.15, local_gain_initial_fraction=0., local_gain_maximum_fraction=.05,
        markov_feature_center=center, markov_feature_scale=state_scale,
        markov_residual_scale=.6, markov_hidden_size=96, maximum_slew=None)
    critic = core.TimeConditionedWassersteinCritic(fit.reshape(-1, p["n"]).mean(dim=0), scale, horizon_samples=256, hidden_size=128)
    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(p["n"], 16))
    projection /= np.maximum(np.linalg.norm(projection, axis=0), 1e-12)
    return dict(p=p, hashes=hashes, core=core, model=model, arrays=arrays, mask=mask,
        selected=selected, fit=fit, validation=validation, fit_ledger=fit_ledger,
        world=world, adapter=adapter, stepper=stepper, initials=initials, selections=selections,
        scale=scale, weights=weights, actor=actor, critic=critic,
        projections=torch.as_tensor(projection, dtype=torch.float64))

def law(s, rollout, reference, baseline, regularize):
    return s["core"].law_objective(rollout.scaled[:, 1:], rollout.controls, rollout.commands,
        reference, s["weights"], s["scale"], s["projections"], s["actor"],
        include_parameter_regularization=regularize, baseline_sequence=baseline,
        objective_mode="covariance", smoothness_weight=.2, curvature_weight=.1,
        slew_barrier_weight=0., curvature_barrier_weight=0., maximum_first_difference=.35,
        maximum_second_difference=.5)

def exact_w1(x, y, axis=0):
    x, y = np.moveaxis(np.asarray(x), axis, 0), np.moveaxis(np.asarray(y), axis, 0)
    if x.shape[1:] != y.shape[1:]: raise ValueError("Empirical W1 shape mismatch")
    n, m = x.shape[0], y.shape[0]
    breaks = np.unique(np.r_[np.arange(n + 1) / n, np.arange(m + 1) / m])
    middle = (breaks[1:] + breaks[:-1]) * .5
    ix, iy = np.minimum((middle * n).astype(int), n - 1), np.minimum((middle * m).astype(int), m - 1)
    difference = np.abs(np.sort(x, axis=0)[ix] - np.sort(y, axis=0)[iy])
    return (difference * np.diff(breaks).reshape((-1,) + (1,) * (x.ndim - 1))).sum(axis=0)

def distances(x, reference):
    n = x.shape[-1]
    return exact_w1(x, reference).mean(axis=0), exact_w1(x.reshape(-1, n), reference.reshape(-1, n))

def validation_cache(s):
    core, stepper = s["core"], s["stepper"]
    noise = core.antithetic_noise(20260922, stepper.q)
    with torch.no_grad():
        return [(initial, noise, core.uncontrolled_particle_rollout(stepper, initial, noise)[:, 1:]) for initial in s["selections"]]

def validate(s, cache):
    s["actor"].eval()
    rows = []
    with torch.no_grad():
        for initial, noise, baseline in cache:
            rollout = s["core"].empirical_fp_rollout(s["stepper"], s["actor"], initial, noise)
            value, _ = law(s, rollout, s["validation"], baseline, False)
            tw, ow = distances(rollout.scaled[:, 1:].numpy(), s["validation"].numpy())
            rows.append(dict(validation_law=float(value), validation_mean_time_w1=float(tw.mean()), validation_mean_occupation_w1=float(ow.mean())))
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}

def preflight(args, out):
    s = setup(args.subject, args.seed)
    error, peak = 0., 0.
    with torch.no_grad():
        for initial in s["initials"]:
            noise = s["core"].antithetic_noise(args.seed, s["stepper"].q)
            free = s["core"].uncontrolled_particle_rollout(s["stepper"], initial, noise)
            rollout = s["core"].empirical_fp_rollout(s["stepper"], s["actor"], initial, noise)
            error = max(error, float((rollout.scaled - free).abs().max()))
            peak = max(peak, float(rollout.controls.abs().max()))
    if error > 1e-12 or peak > 1e-12: raise RuntimeError("Fresh actor is not neutral")
    check = dict(subject=args.subject, official_plant_sha256=MODEL_HASHES[args.subject],
        canonical_source_sha256=sha(SOURCE), original_input_hashes=s["hashes"],
        channels=s["p"]["n"], direct_actuators=s["p"]["m"], selected_indices=s["selected"],
        control_heat_time=s["p"]["tau"], diffusion_scale=.79451175, control_step_scale=.5,
        fit_reference_shape=list(s["fit"].shape), validation_reference_shape=list(s["validation"].shape),
        rotating_fit_context_count=len(s["initials"]), selection_ctx5_count=len(s["selections"]),
        fit_ledger=s["fit_ledger"], neutral_peak=peak, neutral_free_parity_max_error=error,
        teacher_or_actor_checkpoint_loaded=False, outer_signal_arrays_opened=False,
        parameter_count=sum(p.numel() for p in s["actor"].parameters()),
        uniform_reference_sampler=True, actor_score_and_law_use_all45=True,
        architecture_dynamics_and_reference_arrays_unchanged=True, passed=True)
    dump(out / "preflight.json", check)
    print(json.dumps(ready(check)), flush=True)

def uniform_reference(reference, seed):
    """Uniform without-replacement order before the unchanged helper truncates.

    The first 32 of 45 are an unbiased real minibatch; GP interpolation keeps a
    different generator. Actor scores and the empirical-law objective use all 45.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(reference.shape[0], generator=generator)
    return reference.index_select(0, order), order[:32].numpy()


def train(args, out):
    if (out / "frozen_actor_wgan.pt").exists(): raise RuntimeError("Refusing completed run overwrite")
    if not (out / "preflight.json").exists(): raise RuntimeError("Run preflight first")
    s = setup(args.subject, args.seed)
    core, actor, critic, stepper = s["core"], s["actor"], s["critic"], s["stepper"]
    started = time.perf_counter()
    contract = dict(subject=args.subject, actor_updates=args.updates, update_unit="one rotating fit context; not a whole-data epoch",
        seed=args.seed, actor_lr=3e-4, actor_min_lr=1e-5, critic_lr=1e-4, n_critic=3,
        adv_weight=.5, teacher_proximity_weight=0., critic_warmup=24, gp_weight=10., critic_drift=.001,
        particles=32, horizon=256, actor_neutral_initialization=True, teacher_or_actor_checkpoint_loaded=False,
        graph_spread=s["p"]["tau"], input_energy_cap=s["p"]["energy_cap"], per_actuator_rms_cap=.405,
        no_runtime_projection_or_rescaling=True, input_paths=s["hashes"], source_sha256=sha(SOURCE),
        model_sha256=MODEL_HASHES[args.subject], source_code_sha256=sha(__file__),
        fit_contexts=s["fit_ledger"], selection_context_index=5, terminal_veto_context_index=6,
        validation_every=50, outer_opened_during_training=False,
        classification="post-hoc exploratory training amendment; previously revealed outer run is not a new locked validation",
        critic_reference_sampling="uniform without-replacement prefix32 from a freshly permuted full45 at every critic update",
        reference_sampler_seed_rule="warm: seed+70000000+i; joint: seed+80000000+1000*update+j",
        independent_gp_generator=True, actor_score_and_law_use_all45=True)
    dump(out / "training_contract.json", contract)
    torch.save({"actor_state_dict": actor.state_dict(), "seed": args.seed, "teacher_loaded": False}, out / "initial_actor.pt")
    actor_opt = torch.optim.AdamW(actor.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_opt, T_max=args.updates, eta_min=1e-5)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0., .9))
    indices = tuple(range(15, 256, 16))
    cache = validation_cache(s)
    initial_validation = validate(s, cache)
    history = [{"update": 0, **initial_validation, "validation_selection_score": 1.}]
    best_score, best_update, best_actor, best_critic = float("inf"), -1, None, None
    with torch.no_grad():
        warm = [core.empirical_fp_rollout(stepper, actor, s["initials"][bank % len(s["initials"])], core.antithetic_noise(args.seed + 101 * (bank + 1), stepper.q)).scaled[:, 1:].detach() for bank in range(3)]
    warm_rows = []
    reference_inclusion_counts = np.zeros(s["fit"].shape[0], dtype=np.int64)
    for i in range(24):
        critic_opt.zero_grad(set_to_none=True)
        critic_reference, sampled_paths = uniform_reference(s["fit"], args.seed + 70000000 + i)
        reference_inclusion_counts[sampled_paths] += 1
        audit = core.balanced_wgan_gp_loss(critic, warm[i % 3], critic_reference, indices,
            gradient_penalty_weight=10., critic_drift_weight=.001, generator=torch.Generator().manual_seed(args.seed + 50000 + i))
        audit.loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.)
        critic_opt.step()
        warm_rows.append(dict(update=i + 1, critic_loss=float(audit.loss.detach()), critic_gp=float(audit.gradient_penalty.detach()), critic_input_gradient_norm=float(audit.mean_gradient_norm.detach()), real_sampled_path_indices=json.dumps(sampled_paths.tolist())))
    pd.DataFrame(warm_rows).to_csv(out / "critic_warmup.csv", index=False)
    for update in range(1, args.updates + 1):
        slot = (update - 1) % len(s["initials"])
        initial = s["initials"][slot]
        noise = core.antithetic_noise(args.seed + 1009 * update, stepper.q)
        actor.eval()
        with torch.no_grad():
            detached = core.empirical_fp_rollout(stepper, actor, initial, noise).scaled[:, 1:].detach()
            baseline = core.uncontrolled_particle_rollout(stepper, initial, noise)[:, 1:]
        core.set_requires_grad(actor, False)
        core.set_requires_grad(critic, True)
        critic.train()
        for j in range(3):
            critic_opt.zero_grad(set_to_none=True)
            critic_reference, sampled_paths = uniform_reference(s["fit"], args.seed + 80000000 + 1000 * update + j)
            reference_inclusion_counts[sampled_paths] += 1
            audit = core.balanced_wgan_gp_loss(critic, detached, critic_reference, indices,
                gradient_penalty_weight=10., critic_drift_weight=.001,
                generator=torch.Generator().manual_seed(args.seed + 100000 * update + j))
            if not torch.isfinite(audit.loss): raise RuntimeError("Nonfinite critic")
            audit.loss.backward()
            dnorm = torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.)
            critic_opt.step()
        core.set_requires_grad(critic, False)
        core.set_requires_grad(actor, True)
        actor.train()
        actor_opt.zero_grad(set_to_none=True)
        rollout = core.empirical_fp_rollout(stepper, actor, initial, noise)
        law_loss, terms = law(s, rollout, s["fit"], baseline, True)
        adv, _ = core.actor_wasserstein_loss(critic, rollout.scaled[:, 1:], s["fit"], indices)
        total = law_loss + .5 * adv
        if not torch.isfinite(total): raise RuntimeError("Nonfinite actor")
        total.backward()
        anorm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
        actor_opt.step()
        used_lr = actor_opt.param_groups[0]["lr"]
        scheduler.step()
        row = dict(update=update, fit_context_slot=slot, actor_lr=used_lr, train_total=float(total.detach()),
            train_law=float(law_loss.detach()), train_adversarial=float(adv.detach()), critic_loss=float(audit.loss.detach()),
            critic_estimate=float(audit.estimate.detach()), critic_gp=float(audit.gradient_penalty.detach()),
            critic_input_gradient_norm=float(audit.mean_gradient_norm.detach()),
            actor_gradient_norm_before_clip=float(anorm), critic_gradient_norm_before_clip=float(dnorm),
            critic_reference_path_indices_last=json.dumps(sampled_paths.tolist()),
            critic_reference_run01_count_last=int(np.sum(sampled_paths < 15)),
            critic_reference_run02_count_last=int(np.sum((sampled_paths >= 15) & (sampled_paths < 30))),
            critic_reference_run03_count_last=int(np.sum(sampled_paths >= 30)))
        row.update({"train_law_" + k: float(v.detach()) for k, v in terms.items()})
        if update == 1 or update % 50 == 0 or update == args.updates:
            row.update(validate(s, cache))
            score = .5 * row["validation_law"] / initial_validation["validation_law"] + .25 * row["validation_mean_time_w1"] / initial_validation["validation_mean_time_w1"] + .25 * row["validation_mean_occupation_w1"] / initial_validation["validation_mean_occupation_w1"]
            row["validation_selection_score"] = score
            if score < best_score:
                best_score, best_update = score, update
                best_actor, best_critic = copy.deepcopy(actor.state_dict()), copy.deepcopy(critic.state_dict())
                torch.save(dict(actor_state_dict=best_actor, critic_state_dict=best_critic, best_update=best_update, training_contract=contract), out / "best_so_far.pt")
        history.append(row)
        if update == 1 or update % 10 == 0 or update == args.updates:
            pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
            elapsed = time.perf_counter() - started
            progress = dict(status="running", subject=args.subject, update=update, planned_updates=args.updates,
                best_update=best_update, best_selection_score=best_score, elapsed_seconds=elapsed,
                remaining_seconds_linear_estimate=elapsed / update * (args.updates - update),
                train_law=row["train_law"], training_context_only=True, outer_opened=False)
            dump(out / "progress.json", progress)
            dump(out / "reference_sampling_audit.json", dict(critic_updates_observed=24 + update * 3,
                sampled_per_critic_update=32, all45_path_inclusion_counts=reference_inclusion_counts,
                source_run_inclusion_counts=[int(reference_inclusion_counts[:15].sum()), int(reference_inclusion_counts[15:30].sum()), int(reference_inclusion_counts[30:].sum())],
                uniform_without_replacement=True, independent_gp_generator=True,
                actor_score_and_law_use_all45=True))
            print(json.dumps(progress), flush=True)
    torch.save(dict(actor_state_dict=actor.state_dict(), critic_state_dict=critic.state_dict(), update=args.updates, training_contract=contract), out / "last_actor_wgan.pt")
    actor.load_state_dict(best_actor)
    critic.load_state_dict(best_critic)
    selected_validation = validate(s, cache)
    torch.save(dict(actor_state_dict=best_actor, critic_state_dict=best_critic, best_update=best_update,
        training_contract=contract, selected_validation=selected_validation), out / "frozen_actor_wgan.pt")
    if any(sha(path) != digest for path, digest in s["hashes"].items()): raise RuntimeError("Original input/source mutation")
    summary = dict(status="completed_fresh_hybrid_training", subject=args.subject, trained_updates=args.updates,
        selected_update=best_update, best_selection_score=best_score, initial_validation=initial_validation,
        selected_validation=selected_validation, elapsed_seconds=time.perf_counter() - started,
        teacher_loaded=False, outer_opened=False, checkpoint_sha256=sha(out / "frozen_actor_wgan.pt"), not_adopted_in_paper=True)
    dump(out / "training_summary.json", summary)
    dump(out / "progress.json", summary)
    print(json.dumps(summary), flush=True)

def deterministic_seed(subject, fold, stage, replicate, base=20261011):
    payload = f"{PROTOCOL}|{subject}|{fold}|{stage}|{replicate}|{base}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2 ** 31 - 1)

def evaluate_set(args, s, initials, reference, observed, fold, stage, output, display=0):
    core, actor, stepper = s["core"], s["actor"], s["stepper"]
    rows, safety_rows, displays = [], [], {}
    with torch.no_grad():
        for index, initial in enumerate(initials):
            for bank in range(3):
                seed = deterministic_seed(args.subject, fold, stage, index * 3 + bank, args.seed)
                noise = core.antithetic_noise(seed, stepper.q)
                free = core.uncontrolled_particle_rollout(stepper, initial, noise)[:, 1:].numpy()
                r = core.empirical_fp_rollout(stepper, actor, initial, noise)
                controlled = r.scaled[:, 1:].numpy()
                controls = r.controls.numpy()
                tf, of = distances(free, reference)
                tc, oc = distances(controlled, reference)
                rm, rs = reference.mean(axis=(0, 1)), reference.std(axis=(0, 1))
                cm, cs = controlled.mean(axis=(0, 1)), controlled.std(axis=(0, 1))
                rho = np.maximum(cs / np.maximum(rs, 1e-12), rs / np.maximum(cs, 1e-12))
                mean_error = np.abs(cm - rm)
                gates = dict(time_relative_reduction_pass=(tf - tc) / np.maximum(tf, 1e-12) >= .1,
                    occupation_relative_reduction_pass=(of - oc) / np.maximum(of, 1e-12) >= .1,
                    time_absolute_pass=tc <= .35, occupation_absolute_pass=oc <= .25,
                    mean_absolute_error_pass=mean_error <= .1, symmetric_sd_ratio_pass=rho <= 2.)
                finite = bool(np.isfinite(free).all() and np.isfinite(controlled).all() and np.isfinite(controls).all())
                energy = float(np.mean(np.sum(controls ** 2, axis=-1)))
                rms = np.sqrt(np.mean(controls ** 2, axis=(0, 1)))
                peak = float(np.max(np.abs(controls)))
                saturation = float(np.mean(np.abs(controls) >= .95 * 1.8))
                safety = bool(finite and energy <= s["p"]["energy_cap"] + 1e-9 and rms.max() <= .405 + 1e-9 and peak <= 1.8 + 1e-9 and saturation < .01)
                sr = dict(context_index=index, crn_bank=bank, noise_seed=seed, mean_time_w1_free=float(tf.mean()),
                    mean_time_w1_controlled=float(tc.mean()), mean_occupation_w1_free=float(of.mean()),
                    mean_occupation_w1_controlled=float(oc.mean()), control_rms=float(np.sqrt(np.mean(controls ** 2))),
                    maximum_per_actuator_rms=float(rms.max()), total_energy=energy, control_peak=peak,
                    saturation_fraction=saturation, finite=finite, gate_c=safety)
                safety_rows.append(sr)
                full = np.logical_and.reduce(list(gates.values())) & np.isfinite(np.stack([tf, tc, of, oc, mean_error, rho])).all(axis=0)
                for channel in range(s["p"]["n"]):
                    rows.append(dict(context_index=index, crn_bank=bank, channel_index=channel,
                        channel=str(s["arrays"]["channels"][channel]), direct_actuated=bool(s["mask"][channel]),
                        time_w1_free=float(tf[channel]), time_w1_controlled=float(tc[channel]),
                        occupation_w1_free=float(of[channel]), occupation_w1_controlled=float(oc[channel]),
                        mean_abs_error=float(mean_error[channel]), symmetric_sd_ratio=float(rho[channel]),
                        both_time_and_occupation_improved=bool(tc[channel] < tf[channel] and oc[channel] < of[channel]),
                        **{k: bool(v[channel]) for k, v in gates.items()}, full_gate_b_pass=bool(full[channel]), gate_c_trajectory_pass=safety))
                if index == display and bank == 0:
                    displays = dict(free_standardized=free, controlled_standardized=controlled,
                        reference_standardized=reference, controls=controls, commands=r.commands.numpy(),
                        standard_normal=noise.numpy(), observed_standardized=observed[index],
                        channels=s["arrays"]["channels"], direct_mask=s["mask"], selected_indices=s["selected"],
                        display_context_index=np.asarray(index), display_crn_bank=np.asarray(0))
                print(json.dumps(dict(subject=args.subject, evaluation_stage=stage, context=index, bank=bank, gate_c=safety)), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    frame, safety_frame = pd.DataFrame(rows), pd.DataFrame(safety_rows)
    frame.to_csv(output / "all_channel_context_bank_metrics.csv", index=False)
    safety_frame.to_csv(output / "trajectory_safety_metrics.csv", index=False)
    components = list(gates)
    vectors = frame.groupby("channel_index", sort=True)[components + ["full_gate_b_pass", "both_time_and_occupation_improved"]].all()
    gate_c = bool(safety_frame.gate_c.all())
    vectors["safety_gate_pass"] = gate_c
    vectors["full_gate_pass"] = vectors.full_gate_b_pass & gate_c
    vectors.to_csv(output / "aggregate_gate_vectors.csv")
    np.savez_compressed(output / "display_context_rollout.npz", **displays)
    summary = dict(subject=args.subject, contexts=len(initials), crn_banks=3, context_bank_evaluations=len(initials) * 3,
        channels=s["p"]["n"], direct_actuators=s["p"]["m"], gate_b_pass_count=int(vectors.full_gate_b_pass.sum()),
        both_improved_all_context_bank_count=int(vectors.both_time_and_occupation_improved.sum()), gate_c_pass=gate_c,
        safety_failed_trajectory_count=int((~safety_frame.gate_c).sum()), energy_cap=s["p"]["energy_cap"],
        per_actuator_rms_cap=.405, no_projection_or_rescaling=True,
        **{k: float(safety_frame[k].mean()) for k in ("mean_time_w1_free", "mean_time_w1_controlled", "mean_occupation_w1_free", "mean_occupation_w1_controlled", "control_rms", "total_energy")})
    summary["terminal_veto_pass"] = bool(gate_c and summary["gate_b_pass_count"] >= 1 and summary["mean_time_w1_controlled"] < summary["mean_time_w1_free"] and summary["mean_occupation_w1_controlled"] < summary["mean_occupation_w1_free"])
    dump(output / "evaluation_summary.json", summary)
    return summary

def evaluate(args, out):
    checkpoint_path = out / "frozen_actor_wgan.pt"
    if not checkpoint_path.is_file() or not (out / "training_summary.json").is_file(): raise RuntimeError("Freeze trained policy before evaluation")
    s = setup(args.subject, args.seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["training_contract"]["model_sha256"] != MODEL_HASHES[args.subject]: raise RuntimeError("Checkpoint predictive-plant mismatch")
    if checkpoint["training_contract"]["input_paths"] != s["hashes"]: raise RuntimeError("Checkpoint input hashes changed")
    s["actor"].load_state_dict(checkpoint["actor_state_dict"], strict=True)
    s["actor"].eval()
    # First terminal context-6 veto, never checkpoint/arm reselection.
    initials, observed = [], []
    for run in s["p"]["development_runs"]:
        ictal = s["arrays"]["ictal_" + run.replace("-", "_")]
        stop = boundary(ictal, 6)
        initials.append(s["adapter"].initial_state_from_context(ictal[stop - 256:stop]))
        observed.append(s["model"].transform.scaler.transform(ictal[stop:stop + 256]))
    veto = evaluate_set(args, s, initials, s["validation"].numpy(), observed,
        "all-development", "ctx6_terminal_veto", out / "ctx6_terminal_veto")
    # Already-revealed outer arrays are diagnostic amendments, not pristine tests.
    with np.load(s["p"]["outer"], allow_pickle=False) as archive:
        outer = {k: np.asarray(archive[k]) for k in archive.files}
    if not np.array_equal(outer["channels"].astype(str), s["arrays"]["channels"].astype(str)): raise RuntimeError("Outer channel order differs")
    ictal = outer["ictal"]
    reference = s["model"].transform.scaler.transform(outer["preictal_reference"]).reshape(-1, 256, s["p"]["n"])
    initials, observed = [], []
    for index in range(8):
        stop = boundary(ictal, index)
        initials.append(s["adapter"].initial_state_from_context(ictal[stop - 256:stop]))
        observed.append(s["model"].transform.scaler.transform(ictal[stop:stop + 256]))
    summary = evaluate_set(args, s, initials, reference, observed,
        s["p"]["outer_fold"], s["p"]["outer_stage"], out / "outer_posthoc_amendment", display=7)
    summary.update(classification="post-hoc exploratory amendment after the outer run had already been revealed",
        selected_update=checkpoint["best_update"], checkpoint_sha256=sha(checkpoint_path),
        ctx6_veto_pass=veto["terminal_veto_pass"], ctx6_used_for_reselection=False,
        outer_used_for_training_or_checkpoint_selection=False, not_adopted_in_paper=True)
    # Check unchanged zero-control law against the earlier frozen report.
    old_report = json.loads(s["p"]["old_outer_report"].read_text(encoding="utf-8"))
    old = old_report.get("metrics", old_report.get("summary", old_report))
    old_tw, old_ow = old.get("mean_time_w1_free"), old.get("mean_occupation_w1_free")
    if old_tw is not None and old_ow is not None:
        discrepancy = max(abs(summary["mean_time_w1_free"] - old_tw), abs(summary["mean_occupation_w1_free"] - old_ow))
        summary["old_zero_control_metric_parity_max_abs_error"] = discrepancy
        if discrepancy > 1e-7: raise RuntimeError(f"Original zero-control outer metric parity failed: {discrepancy}")
    dump(out / "outer_posthoc_amendment/evaluation_summary.json", summary)
    if any(sha(path) != digest for path, digest in s["hashes"].items()): raise RuntimeError("Original inputs changed")
    print(json.dumps(ready(summary)), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "train", "evaluate", "train-evaluate"))
    parser.add_argument("--subject", required=True, choices=("HUP080",))
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    out = ROOT / "runs" / f"{args.subject}_seed{args.seed}_u{args.updates}_uniformref"
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == "preflight": preflight(args, out)
    if args.mode in ("train", "train-evaluate"): train(args, out)
    if args.mode in ("evaluate", "train-evaluate"): evaluate(args, out)

if __name__ == "__main__":
    main()
