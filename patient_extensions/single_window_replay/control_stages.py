"""Offline Part III stages; use run.py for the public input/output CLI.

Advanced use: set the SWR_* environment variables as in replay_config.py,
then run python -B control_stages.py calibrate|full|evaluate|verify. Full
requires completed dynamics and scale_calibration stages and a fresh control/
directory. Raw trajectories and checkpoints are local, user-supplied artifacts.
Use trusted joblib/torch files only. All controls are model simulations.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

for thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_var] = "1"

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import control_core as core

COMBINED = "combined_policy_eval.npz"


def bind(c):
    """Create a synthesis allowlist: no recorded forecasting target enters train()."""
    out, dyn, cal = c.output / "control", c.output / "dynamics", c.output / "scale_calibration"
    out.mkdir(exist_ok=True)
    core.require(not any(out.iterdir()), "Use an empty control directory; refusing to overwrite a replay")
    alpha_receipt = core.read_json(cal / "selection.json")
    noise_receipt = core.read_json(dyn / "noise_selection.json")
    k, a = float(noise_receipt["selected_kappa"]), float(alpha_receipt["selected_alpha"])
    model_path = dyn / "plant.joblib"
    core.require(alpha_receipt["selected_kappa"] == k and k in [.5, .75, 1., 1.25] and a in [.25, .5, 1.]
                 and alpha_receipt["run02_used"] is False and alpha_receipt["model_sha256"] == core.sha(model_path)
                 and alpha_receipt["protocol_sha256"] == core.sha(cal / "protocol.json")
                 and noise_receipt["global_C"] == core.BASE_DIFFUSION,
                 "Input/diffusion selection does not match plant/protocol")
    sel, names, raw_graph = core.load_graph(c)
    model = joblib.load(model_path)
    np.testing.assert_array_equal(model.adjacency_input, raw_graph)
    np.testing.assert_allclose(model.adjacency, core.p2.normalize_adjacency(raw_graph), rtol=0, atol=0)
    ref = core.reference(c, model, names)
    hist = core.segment_slice(c.run02_ictal, names, core.BOUNDARY-256, core.BOUNDARY)
    np.save(out / "eval_noise.npy", core.eval_bank(model.q), allow_pickle=False)
    np.savez_compressed(out / "synthesis_only.npz", history_raw=hist, reference_scaled=ref,
                        selected_indices=sel, channels=names, model_sha256=np.array(core.sha(model_path)))
    cfg = {"subject": c.subject, "kappa_sigma": k, "constant_C": core.BASE_DIFFUSION,
           "diffusion_scale": core.BASE_DIFFUSION*k, "control_step_scale": a,
           "model_sha256": core.sha(model_path), "reference_sha256": core.sha(c.run01_reference),
           "eval_input_sha256": core.sha(c.run02_ictal),  # opaque input fingerprint, not a training statistic
           "run02_history_samples": [3291, 3547], "run02_eval_samples": [3547, 3803],
           "eval_noise_seed": core.EVAL_SEED, "eval_noise_sha256": core.sha(out / "eval_noise.npy"),
           "gain_calibration_sha256": core.sha(cal / "selection.json"),
           "noise_calibration_sha256": core.sha(dyn / "noise_selection.json"),
           "training_contract_sha256": core.sha(out / "synthesis_only.npz")}
    core.dump(out / "binding.json", cfg)
    return cfg


def validate_binding(c):
    out, dyn, cal = c.output / "control", c.output / "dynamics", c.output / "scale_calibration"
    cfg = core.read_json(out / "binding.json")
    core.require(cfg["subject"] == c.subject, "Case changed after binding")
    for key, path in [("model_sha256", dyn / "plant.joblib"),
                      ("training_contract_sha256", out / "synthesis_only.npz"),
                      ("eval_noise_sha256", out / "eval_noise.npy"),
                      ("gain_calibration_sha256", cal / "selection.json"),
                      ("noise_calibration_sha256", dyn / "noise_selection.json")]:
        core.require(cfg[key] == core.sha(path), "Frozen input changed: " + key)
    return cfg


def simulate(c, model_path, synth, ckpt, samples, kappa, alpha):
    """Return paired paths from the frozen actor without accessing any recorded future."""
    m, src = core.trainer(c, model_path, kappa, alpha, synth)
    _, _, val, sel, _, initial, stepper, _, _, actor, _, model_hash = src.setup(m)
    noise = torch.as_tensor(samples, dtype=torch.float64)
    core.require(noise.shape == (32, 256, stepper.q), "Invalid evaluation bank")
    np.testing.assert_array_equal(samples[:16], -samples[16:])
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    core.require(ck["model_sha256"] == model_hash, "Frozen actor/plant mismatch")
    actor.load_state_dict(ck["actor_state_dict"], strict=True)
    actor.eval()
    with torch.no_grad():
        free = m.uncontrolled_particle_rollout(stepper, initial, noise)[:, 1:].numpy()
        pair = m.paired_ictal_batch_rollout(stepper, actor, initial, noise)
    parity = float(np.abs(pair.uncontrolled_scaled[:, 1:].numpy()-free).max())
    core.require(parity < 1e-6, "Paired uncontrolled paths do not match the standalone baseline")
    with np.load(synth, allow_pickle=False) as z:
        past = joblib.load(model_path).transform.scaler.transform(z["history_raw"])
        names = z["channels"]
    result = dict(channels=names, past=past, free=free, reference=val.numpy(), selected_indices=sel,
                  controlled=pair.controlled_scaled[:, 1:].numpy(), controls=pair.controls.numpy(),
                  commands=pair.commands.numpy())
    core.require(all(np.isfinite(v).all() for v in result.values() if np.issubdtype(v.dtype, np.number)),
                 "Nonfinite paired evaluation outputs")
    return result, parity


def evaluate(c):
    out = c.output / "control"
    cfg = validate_binding(c)
    ck = out / "frozen_actor_wgan.pt"
    manifest = core.read_json(out / "training_summary.json")
    core.require(manifest["trained_epochs"] == 1000 and manifest["critic_updates"] == 3024
                 and manifest["checkpoint_sha256"] == core.sha(ck), "Full policy not complete/frozen")
    saved = torch.load(ck, map_location="cpu", weights_only=False)
    training = core.read_json(out / "training_contract.json")
    core.require(saved["training_contract"] == training and saved["best_epoch"] == manifest["best_epoch"]
                 and training["frozen_bindings_sha256"] == core.sha(out / "binding.json")
                 and training["synthesis_sha256"] == cfg["training_contract_sha256"]
                 and training["model_sha256"] == cfg["model_sha256"],
                 "The frozen actor does not belong to these synthesis inputs")
    model = c.output / "dynamics" / "plant.joblib"
    result, parity = simulate(c, model, out / "synthesis_only.npz", ck,
                              np.load(out / "eval_noise.npy", allow_pickle=False),
                              cfg["kappa_sigma"], cfg["control_step_scale"])
    # The recorded target is used only for this post-freeze descriptive output.
    core.require(core.sha(c.run02_ictal) == cfg["eval_input_sha256"], "Run-02 input changed")
    raw = core.segment_slice(c.run02_ictal, result["channels"], core.BOUNDARY, core.BOUNDARY+256)
    result["observed"] = joblib.load(model).transform.scaler.transform(raw)
    np.savez_compressed(out / COMBINED, **result)
    core.dump(out / "eval_receipt.json", dict(
        subject=c.subject, model_sha256=cfg["model_sha256"], checkpoint_sha256=core.sha(ck),
        best_epoch=manifest["best_epoch"], max_free_pair_error=parity,
        eval_sha256=core.sha(out / COMBINED), eval_noise_sha256=cfg["eval_noise_sha256"],
        reference="run-01 EDF [45,60) s", recorded_future_used_for_training_or_selection=False))


def verify(c):
    out = c.output / "control"
    cfg = validate_binding(c)
    t, s = core.read_json(out / "training_contract.json"), core.read_json(out / "training_summary.json")
    rec = core.read_json(out / "eval_receipt.json")
    core.require(s["trained_epochs"] == 1000 and s["critic_updates"] == 3024
                 and t["run02_past_context_used"] is True and t["run02_forecast_outcome_used"] is False
                 and t["frozen_bindings_sha256"] == core.sha(out / "binding.json")
                 and t["synthesis_sha256"] == cfg["training_contract_sha256"]
                 and rec["model_sha256"] == t["model_sha256"] == cfg["model_sha256"]
                 and rec["checkpoint_sha256"] == s["checkpoint_sha256"] == core.sha(out / "frozen_actor_wgan.pt")
                 and rec["eval_sha256"] == core.sha(out / COMBINED), "Frozen-run audit mismatch")
    h = pd.read_csv(out / "training_history.csv")
    v = h.dropna(subset=["validation_selection_score"])
    v = v[v.epoch > 0]
    core.require(list(h.epoch) == list(range(1001)) and set(v.epoch) == {1, *range(50, 1001, 50)}
                 and int(v.loc[v.validation_selection_score.idxmin(), "epoch"]) == s["best_epoch"],
                 "Incomplete updates or incorrect frozen validation winner")
    with np.load(out / COMBINED, allow_pickle=False) as ev, np.load(out / "synthesis_only.npz") as syn:
        np.testing.assert_array_equal(ev["reference"], syn["reference_scaled"][15:])
        np.testing.assert_array_equal(ev["selected_indices"], syn["selected_indices"])
        core.require(ev["controlled"].shape == ev["free"].shape == (32, 256, c.channels)
                     and ev["observed"].shape == (256, c.channels)
                     and ev["controls"].shape == ev["commands"].shape == (32, 256, c.actuators)
                     and np.max(np.abs(ev["controls"])) <= 1.8+1e-12, "Trajectory shape/command violation")
    print("Verified:", c.subject, COMBINED, rec["eval_sha256"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["calibrate", "full", "evaluate", "verify"])
    stage = parser.parse_args().stage
    from replay_config import settings
    c = settings()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):
        if stage == "calibrate":
            from control_calibration import calibrate
            calibrate(c)
            return
        if stage == "full":
            cfg = bind(c)
            out = c.output / "control"
            m, src = core.trainer(c, c.output / "dynamics" / "plant.joblib", cfg["kappa_sigma"],
                                  cfg["control_step_scale"], out / "synthesis_only.npz")
            src.train(core.options(), m, out)
        if stage != "verify":
            evaluate(c)
        verify(c)
        if stage == "full":
            subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("render.py")),
                            "--subject", c.subject, "--input", str(c.output / "control" / COMBINED),
                            "--out", str(c.output / "figures")], check=True)


if __name__ == "__main__":
    main()
