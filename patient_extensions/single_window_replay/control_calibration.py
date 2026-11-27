"""Run-01-only simulation input-scale selection (no access to run-02)."""
import gc
import joblib
import numpy as np
import pandas as pd
import torch

import control_core as core

PROTOCOL = {
    "alpha_grid": [.25, .5, 1.],
    "run01_ictal_prefix_boundary": 3547,
    "run01_validation_prefix_boundaries": [6656, 7424],
    "seed": 20261011, "validation_seed": 20260922, "updates": 12, "cosine_horizon": 1000,
    "selection": "Actual epoch 12; average two prefix scores. Each score = .50 law/neutral "
                 "+ .25 per-time W1/neutral + .25 occupation W1/neutral. Tie prefers .50.",
    "reference": "Run-01 preictal EDF [30,45) fit; [45,60) validation.",
    "run02_used": False,
}


def trial(c, alpha, kappa, model, x, ref, sel, names, out):
    out.mkdir()
    b = PROTOCOL["run01_ictal_prefix_boundary"]
    synth = out / "run01_contract_only.npz"
    np.savez_compressed(synth, history_raw=x[b-256:b], reference_scaled=ref,
                        selected_indices=sel, channels=names, model_sha256=np.array(core.sha(model)))
    m, src = core.trainer(c, model, kappa, alpha, synth, pilot=True)
    args = core.options(pilot=True)
    src.train(args, m, out)
    _, fit, val, _, adapter, _, stepper, scale, weights, actor, _, _ = src.setup(m)
    last = torch.load(out / "last_actor_wgan.pt", map_location="cpu", weights_only=False)
    core.require(last["epoch"] == 12, "Pilot must score the actual epoch-12 actor")
    actor.load_state_dict(last["actor_state_dict"])
    actor.eval()
    directions = np.random.default_rng(args.seed).normal(size=(c.channels, 16))
    directions /= np.maximum(np.linalg.norm(directions, axis=0), 1e-12)
    directions = torch.as_tensor(directions, dtype=torch.float64)
    noise = m.antithetic_noise(m.VALIDATION_SEED, stepper.q)
    rows = []
    for b in PROTOCOL["run01_validation_prefix_boundaries"]:
        initial = adapter.initial_state_from_context(x[b-256:b])
        with torch.no_grad():
            free = m.uncontrolled_particle_rollout(stepper, initial, noise)[:, 1:]
            neutral = src.setup(m)[9]
            base = m.empirical_fp_rollout(stepper, neutral, initial, noise)
            core.require(float((base.scaled[:, 1:]-free).abs().max()) < 1e-10, "Neutral parity failure")
            controlled = m.empirical_fp_rollout(stepper, actor, initial, noise)

            def metrics(roll, actor_):
                law, _ = src.law(m, roll, val, weights, scale, directions, actor_, free, False)
                y = roll.scaled[:, 1:].numpy()
                return np.array([float(law), m.channelwise_time_w1(y, val.numpy()).mean(),
                                 m.channelwise_occupation_w1(y, val.numpy()).mean()])

            z, y = metrics(base, neutral), metrics(controlled, actor)
            core.require(np.all(z > 0) and np.isfinite(y).all(), "Invalid pilot score")
        rows.append(dict(alpha=alpha, run01_boundary=b, neutral_law=z[0], neutral_time=z[1],
                         neutral_occ=z[2], law=y[0], time=y[1], occ=y[2],
                         score=float(np.dot(y/z, [.5, .25, .25]))))
    pd.DataFrame(rows).to_csv(out / "scores.csv", index=False)
    return rows


def calibrate(c):
    out, dyn = c.output / "scale_calibration", c.output / "dynamics"
    out.mkdir(exist_ok=True)
    core.require(not any(out.iterdir()), "Use an empty scale_calibration directory")
    core.dump(out / "protocol.json", PROTOCOL)
    model_path = dyn / "plant.joblib"
    noise = core.read_json(dyn / "noise_selection.json")
    k = float(noise["selected_kappa"])
    core.require(k in [.50, .75, 1., 1.25] and noise["global_C"] == core.BASE_DIFFUSION,
                 "Unexpected diffusion scale contract")
    sel, names, _ = core.load_graph(c)
    ref = core.reference(c, joblib.load(model_path), names)
    x = core.segment_slice(c.run01_ictal, names, 0, max(PROTOCOL["run01_validation_prefix_boundaries"]))
    rows = []
    for a in PROTOCOL["alpha_grid"]:
        rows.extend(trial(c, a, k, model_path, x, ref, sel, names, out / f"gain_{a:.2f}"))
        gc.collect()
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "validation_scores.csv", index=False)
    means = frame.groupby("alpha")["score"].mean().to_dict()
    winner = min(means, key=lambda a: (means[a], abs(a-.5)))
    result = dict(subject=c.subject, selected_alpha=winner, selected_kappa=k, mean_scores=means,
                  actual_updates_per_alpha=12, critic_updates_per_alpha=60, run02_used=False,
                  model_sha256=core.sha(model_path), protocol_sha256=core.sha(out / "protocol.json"),
                  noise_calibration_sha256=core.sha(dyn / "noise_selection.json"),
                  canonical_alpha=.5)
    core.dump(out / "selection.json", result)
    print(result, flush=True)
