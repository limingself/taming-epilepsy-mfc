"""Aggregate only completed, freshly trained HUP060 component-removal arms.

No training is performed here. Optional evaluation is permitted only after all
arms have complete training summaries and frozen checkpoints. The current main
Full policy is the identical policy used in the control and loss previews.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
FULL_RUN = HERE.parent / ".fresh_wgangp_20261006/runs/fresh_seed20261011_e1000"
RUNS = {
    "full": FULL_RUN,
    "no_wgan": HERE / "runs/no_wgan_seed20261011_e1000",
    "no_graph_spread": HERE / "runs/no_graph_seed20261011_e1000",
    "no_deviation": HERE / "runs/no_deviation_seed20261011_e1000",
}
ORDER = ("free", "no_wgan", "no_graph_spread", "no_deviation", "full")
ARMS = ("full", "no_wgan", "no_graph_spread", "no_deviation")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def module_from_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_complete_training():
    incomplete = [variant for variant, run in RUNS.items()
                  if not all((run / name).is_file() for name in
                             ("frozen_actor_wgan.pt", "training_summary.json", "training_contract.json", "initial_actor.pt"))]
    if incomplete:
        raise RuntimeError("Training is not complete and frozen for: " + ", ".join(incomplete))
    contracts, summaries, initial_states, hashes = {}, {}, {}, {}
    for variant, run in RUNS.items():
        contracts[variant] = json.loads((run / "training_contract.json").read_text(encoding="utf-8"))
        summaries[variant] = json.loads((run / "training_summary.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(run / "frozen_actor_wgan.pt", map_location="cpu", weights_only=False)
        assert checkpoint["training_contract"] == contracts[variant]
        assert int(checkpoint["best_epoch"]) == int(summaries[variant]["best_epoch"])
        assert sha(run / "frozen_actor_wgan.pt") == summaries[variant]["checkpoint_sha256"]
        assert summaries[variant]["teacher_checkpoint_loaded"] is False
        assert float(contracts[variant]["teacher_proximity_weight"]) == 0
        assert contracts[variant]["run02_access_during_training"] is False
        assert int(summaries[variant]["trained_epochs"]) == 1000
        history = pd.read_csv(run / "training_history.csv")
        assert np.array_equal(history.epoch, np.arange(1001))
        val_history = history.dropna(subset=["validation_selection_score"]).query("epoch > 0")
        winner = val_history.loc[val_history.validation_selection_score.idxmin()]
        assert int(winner.epoch) == int(summaries[variant]["best_epoch"])
        assert np.isclose(float(winner.validation_selection_score), summaries[variant]["best_selection_score"], rtol=0, atol=1e-12)
        initial_states[variant] = torch.load(run / "initial_actor.pt", map_location="cpu", weights_only=False)["actor_state_dict"]
        for name in ("training_contract.json", "training_summary.json", "training_history.csv", "frozen_actor_wgan.pt", "initial_actor.pt"):
            hashes[str(run / name)] = sha(run / name)
    full = contracts["full"]
    invariant_keys = ("epochs", "seed", "actor_lr", "actor_min_lr", "critic_lr", "n_critic", "validation_every",
                      "model_sha256", "source_sha256", "synthesis_sha256", "actuators", "teacher_proximity_weight",
                      "gradient_penalty", "critic_drift", "particles", "horizon_samples", "diffusion_scale", "selection")
    argument_audit = {key: {v: contracts[v][key] for v in ARMS} for key in invariant_keys}
    assert all(all(contracts[v][key] == full[key] for v in ARMS) for key in invariant_keys)
    assert contracts["no_wgan"]["adv_weight"] == 0
    assert contracts["no_graph_spread"]["graph_spread"] == 0
    assert contracts["no_deviation"]["component_removal"] == "no_deviation"
    # Only the inverse actuator response buffer is expected to change under a
    # different input heat kernel. Random neural parameters are otherwise equal.
    init_keys = set(initial_states["full"])
    initial_differences = {}
    for variant in ARMS:
        assert set(initial_states[variant]) == init_keys
        unequal = [key for key in sorted(init_keys)
                   if not torch.equal(initial_states[variant][key], initial_states["full"][key])]
        allowed = {"local_inverse_effect"} if variant == "no_graph_spread" else set()
        assert set(unequal).issubset(allowed), f"Unexpected initial state changes: {variant}: {unequal}"
        initial_differences[variant] = unequal
    return contracts, summaries, hashes, argument_audit, initial_differences


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=HERE / "ablations")
    p.add_argument("--evaluate-missing", action="store_true")
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    # When invoked by the batch finisher, both external experiments have also
    # frozen and evaluated. Run independent full-result QA before producing
    # the new figure set; a failed scientific check must stop the export.
    external_runs = [HERE / "external/runs/HUP065_seed20261011_u1000",
                     HERE / "external/runs/HUP080_seed20261011_u1000_uniformref"]
    full_audit = HERE / "qa/verify_new_training_artifacts.py"
    if full_audit.is_file() and all((run / "outer_posthoc_amendment/evaluation_summary.json").is_file()
                                   for run in external_runs):
        subprocess.run([sys.executable, str(full_audit), "--stage-root", str(HERE)], check=True)
    torch.set_num_threads(args.threads)
    contracts, train_summaries, input_hashes, argument_audit, initial_differences = load_complete_training()
    for variant, run in RUNS.items():
        if not (run / "evaluation/paired_comparison.npz").is_file():
            if not args.evaluate_missing:
                raise RuntimeError("Missing frozen evaluation for " + variant + "; rerun with --evaluate-missing")
            if variant == "full":
                raise RuntimeError("Main Full evaluation must be completed by its original isolated runner")
            subprocess.run([sys.executable, str(HERE / "run_matched_ablation.py"), "evaluate", "--ablation", variant,
                            "--tag", run.name, "--threads", str(args.threads)], check=True)
    wrapper = module_from_path(HERE / "run_matched_ablation.py", "fresh_ablation_aggregate_source")
    m = wrapper.source_module()
    assert sha(wrapper.SOURCE) == contracts["full"]["source_sha256"]
    rollouts, controls, commands, effective, maps, evaluations = {}, {}, {}, {}, {}, {}
    full_values = None
    shared_max_diff = {}
    for variant, run in RUNS.items():
        path = run / "evaluation/paired_comparison.npz"
        input_hashes[str(path)] = sha(path)
        ev_path = run / "evaluation/evaluation_summary.json"
        input_hashes[str(ev_path)] = sha(ev_path)
        evaluations[variant] = json.loads(ev_path.read_text(encoding="utf-8"))
        assert evaluations[variant]["checkpoint_sha256"] == train_summaries[variant]["checkpoint_sha256"]
        assert evaluations[variant]["recorded_future_used_for_training_or_selection"] is False
        with np.load(path, allow_pickle=False) as data:
            current = {key: data[key] for key in data.files}
        if full_values is None:
            full_values = current
        for key in ("uncontrolled_scaled", "reference_fit_scaled", "reference_validation_scaled", "observed_scaled", "selected_indices", "channels"):
            assert np.array_equal(current[key], full_values[key]), f"Shared array changed: {variant}:{key}"
        rollouts[variant] = np.asarray(current["candidate_controlled_scaled"], dtype=float)
        controls[variant] = np.asarray(current["candidate_controls"], dtype=float)
        commands[variant] = np.asarray(current["candidate_commands"], dtype=float)
        assert rollouts[variant].shape == (32, 256, 36)
        assert controls[variant].shape == commands[variant].shape == (32, 256, 13)
        assert np.isfinite(rollouts[variant]).all() and np.isfinite(controls[variant]).all()
        wrapper.ACTIVE_ABLATION = variant
        m.GRAPH_DIFFUSION_TIME = float(contracts[variant]["graph_spread"])
        _, _, _, _, adapter, _, _, _, _, _, _, _ = wrapper.setup(m)
        maps[variant] = float(adapter.control_step_scale) * adapter.control_channel_map.detach().cpu().numpy()
        effective[variant] = controls[variant] @ maps[variant]
        shared_max_diff[variant] = float(evaluations[variant]["no_control_parity_max_abs_error"])
    free = np.asarray(full_values["uncontrolled_scaled"], dtype=float)
    reference = np.asarray(full_values["reference_validation_scaled"], dtype=float)
    selected = np.asarray(full_values["selected_indices"], dtype=int)
    channels = np.asarray(full_values["channels"]).astype(str)
    unselected = np.asarray([i for i in range(36) if i not in set(selected)], dtype=int)
    no_graph_input = float(np.max(np.abs(effective["no_graph_spread"][:, :, unselected])))
    no_deviation_spread = float(np.max(np.std(controls["no_deviation"], axis=0)))
    assert no_graph_input <= 1e-12, "No-graph arm still injects unselected nodes"
    assert no_deviation_spread <= 1e-10, "No-deviation arm still emits particle-specific controls"
    assert max(shared_max_diff.values()) <= 1e-6
    noise_hashes = {v: evaluations[v]["noise_reconstruction"]["noise_sha256"] for v in ARMS}
    # Floating-point reconstruction may differ under input-map construction;
    # exact reconstructed free parity is checked above. All arms share the
    # identical stored free-bank target and frozen zero-input plant.
    variants_with_free = {"free": free, **rollouts}
    rng = np.random.default_rng(int(contracts["full"]["seed"]) + 77)
    projections = rng.normal(size=(36, 64))
    projections /= np.maximum(np.linalg.norm(projections, axis=0), 1e-12)
    metrics, summary_rows, channel_rows = {}, [], []
    for variant in ORDER:
        values = variants_with_free[variant]
        t = m.per_time_w1(values, reference)
        o = m.occupation_w1(values, reference)
        joint_t, joint_o = m.sliced_w1(values, reference, projections)
        metrics[variant] = {"time": t, "occupation": o}
        control = controls.get(variant)
        command = commands.get(variant)
        diag = m.control_diagnostics(control) if control is not None else {
            "rms": 0., "peak": 0., "maximum_first_difference": 0.,
            "maximum_second_difference": 0., "saturation_fraction_over_95pct": 0.,
        }
        summary_rows.append({
            "variant": variant, "mean_time_w1": float(t.mean()), "mean_occupation_w1": float(o.mean()),
            "selected_time_w1": float(t[selected].mean()), "selected_occupation_w1": float(o[selected].mean()),
            "unselected_time_w1": float(t[unselected].mean()), "unselected_occupation_w1": float(o[unselected].mean()),
            "joint_time_sliced_w1": joint_t, "joint_occupation_sliced_w1": joint_o,
            "mean_path_rmse": float(np.sqrt(np.mean((values.mean(axis=0) - reference.mean(axis=0)) ** 2))),
            "log_variance_mae": float(np.mean(np.abs(np.log(values.var(axis=0).clip(min=1e-5))
                                                     - np.log(reference.var(axis=0).clip(min=1e-5))))),
            "control_rms": float(diag["rms"]), "control_peak": float(diag["peak"]),
            "control_energy": float(diag["rms"]) ** 2,
            "command_rms": float(np.sqrt(np.mean(command ** 2))) if command is not None else 0.,
            "maximum_first_difference": float(diag["maximum_first_difference"]),
            "maximum_second_difference": float(diag["maximum_second_difference"]),
            "saturation_fraction": float(diag["saturation_fraction_over_95pct"]),
            "selected_epoch": int(train_summaries[variant]["best_epoch"]) if variant != "free" else -1,
        })
        for i, ch in enumerate(channels):
            channel_rows.append({"variant": variant, "channel_index": i, "channel": ch,
                                 "selected": bool(i in set(selected)), "time_w1": float(t[i]),
                                 "occupation_w1": float(o[i])})
        if variant != "free":
            with np.load(RUNS[variant] / "evaluation/paired_comparison.npz") as d:
                assert np.allclose(t, d["time_w1_candidate"], rtol=0, atol=1e-12)
                assert np.allclose(o, d["occupation_w1_candidate"], rtol=0, atol=1e-12)
    summary = pd.DataFrame(summary_rows)
    baseline = summary.loc[summary.variant.eq("free")].iloc[0]
    metric_cols = ["mean_time_w1", "mean_occupation_w1", "selected_time_w1", "selected_occupation_w1",
                   "unselected_time_w1", "unselected_occupation_w1", "joint_time_sliced_w1", "joint_occupation_sliced_w1"]
    for column in metric_cols:
        summary[column + "_recovery_percent"] = 100 * (1 - summary[column] / float(baseline[column]))
    contrast_rows = []
    for variant in ARMS[1:]:
        for i, ch in enumerate(channels):
            contrast_rows.append({"ablation": variant, "channel_index": i, "channel": ch,
                                  "selected": bool(i in set(selected)),
                                  "delta_time_w1_vs_full": float(metrics[variant]["time"][i] - metrics["full"]["time"][i]),
                                  "delta_occupation_w1_vs_full": float(metrics[variant]["occupation"][i] - metrics["full"]["occupation"][i])})
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "ablation_summary.csv", index=False)
    pd.DataFrame(channel_rows).to_csv(out / "ablation_per_channel.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(out / "ablation_contrasts_vs_full.csv", index=False)
    np.savez_compressed(out / "ablation_rollouts.npz", observed_scaled=full_values["observed_scaled"],
                        reference_fit_scaled=full_values["reference_fit_scaled"],
                        reference_validation_scaled=reference, free_scaled=free,
                        selected_indices=selected, channels=channels,
                        sampling_rate_hz=full_values["sampling_rate_hz"],
                        **{variant + "_scaled": rollouts[variant] for variant in ARMS},
                        **{variant + "_controls": controls[variant] for variant in ARMS},
                        **{variant + "_commands": commands[variant] for variant in ARMS},
                        **{variant + "_control_channel_map": maps[variant] for variant in ARMS},
                        **{variant + "_effective_control": effective[variant] for variant in ARMS})
    qa = {"status": "completed_fresh_matched_component_removal_preview_not_adopted",
          "patient": "HUP060", "model_sha256": contracts["full"]["model_sha256"],
          "input_sha256": input_hashes,
          "checkpoint_sha256": {v: train_summaries[v]["checkpoint_sha256"] for v in ARMS},
          "selected_epochs": {v: train_summaries[v]["best_epoch"] for v in ARMS},
          "argument_audit": argument_audit, "same_actor_update_budget": 1000,
          "same_wallclock_budget": False, "fresh_constructor_neural_parameters_same": True,
          "initial_state_differences": initial_differences,
          "teacher_used": False, "same_frozen_plant": True, "same_mask": selected.tolist(),
          "same_reference_fit_and_validation_arrays": True, "same_sealed_free_evaluation_bank": True,
          "same_recorded_arrays": True, "no_control_parity_max_abs": shared_max_diff,
          "evaluation_reconstruction_noise_sha256": noise_hashes,
          "exact_reconstructed_noise_sha_same": len(set(noise_hashes.values())) == 1,
          "no_graph_unselected_effective_input_max_abs": no_graph_input,
          "no_deviation_particle_control_std_max": no_deviation_spread,
          "graph_ablation_scope": "actuator-to-channel input heat kernel only; frozen Graph-RC drift and diffusion unchanged",
          "control_energy_definition": "mean squared applied actuator input; no physical unit",
          "technical_training_seeds_per_arm": 1, "no_cross_seed_statistical_claim": True,
          "channels_are_correlated_technical_units_not_patient_replicates": True,
          "not_an_equal_energy_comparison": True, "overleaf_modified": False,
          "original_inputs_unchanged": all(sha(path) == value for path, value in input_hashes.items()),
          "summary": summary.to_dict(orient="records")}
    assert qa["original_inputs_unchanged"]
    dump(out / "ablation_aggregation_qa.json", qa)
    print(json.dumps({"output_dir": str(out), "selected_epochs": qa["selected_epochs"],
                      "no_graph_nondirect_input": no_graph_input, "no_deviation_particle_std": no_deviation_spread,
                      "summary": summary[["variant", "mean_time_w1", "mean_occupation_w1", "control_rms", "selected_epoch"]].to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
