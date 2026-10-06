"""Re-evaluate input-matched baselines for the newly frozen HUP060 policy.

Existing experiment code and all old results are read-only. The proportional
gain is chosen only by command RMS, never by the reference W1 endpoint.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time
import torch
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "artifacts/part3_hup060_actor_wgan_v1/current_comparisons"
SOURCE = PROJECT / "part3_mfc/trivial_baseline_experiment.py"
PAIR = ROOT.parent / "current_full_wgangp/runs/fresh_seed20261011_e1000/evaluation/paired_comparison.npz"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    output = ROOT / "baselines_paired_exact/source_data"
    if (output / "experiment_summary.json").exists():
        raise RuntimeError("Completed baseline evaluation will not be overwritten")
    before = {str(path): sha(path) for path in (SOURCE, PAIR)}
    spec = importlib.util.spec_from_file_location("fresh_baseline_source", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.FULL_PAIR = PAIR
    module.SOURCE_ROOT = output
    module.ARTIFACT_ROOT = ROOT / "baselines_paired_exact/artifacts"
    # Match precisely the innovations reconstructed from the new Full paired
    # evaluation. The historical sealed free array differed only by roundoff,
    # but its reconstructed noise bank is not used for this final comparison.
    original_reconstruct = module.reconstruct_paired_normals
    with np.load(PAIR) as paired:
        exact_free = paired["uncontrolled_scaled"].copy()
    def reconstruct_from_same_free(stepper, initial, _historical_free, **kwargs):
        return original_reconstruct(stepper, initial, exact_free, **kwargs)
    module.reconstruct_paired_normals = reconstruct_from_same_free
    started = time.perf_counter()
    result = module.main(["--random-seeds", "32"])
    after = {path: sha(path) for path in before}
    if before != after:
        raise RuntimeError("Source or frozen policy evaluation input was modified")
    audit = {
        "source_and_policy_inputs_unchanged": True,
        "input_hashes": before,
        "wrapper_sha256": sha(__file__),
        "elapsed_seconds": time.perf_counter() - started,
        "new_full_checkpoint_selected_epoch": 700,
        "random_seed_count": 32,
        "baseline_parameters_selected_by_RMS_only": True,
        "not_adopted_in_overleaf": True,
    }
    (ROOT / "baselines_paired_exact/recalculation_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-tag", required=True)
    parser.add_argument("--allow-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.allow_evaluation:
        parser.error("An explicit saved-frozen-input evaluation decision is required")
    if not args.new_tag or any(character in args.new_tag for character in '/\\:'):
        parser.error("Use a new simple tag")
    ROOT = ROOT / "new_runs" / args.new_tag
    raise SystemExit(main())
