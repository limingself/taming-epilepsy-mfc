"""Reproduce the independent HUP060 experiment using bundled, unchanged inputs.

Run ``python reproduce_portable.py --verify-bundle`` for a non-training check.
Otherwise pass the same CLI as run_experiment.py; portable outputs are isolated
under portable_reproductions/runs/<tag>. No original D-drive path is required.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "repro_source"
TRAINING_SOURCE = PROJECT / "part3_mfc" / "part3_model.py"
BASELINE = ROOT / "repro_inputs" / "paper_baseline_paired_comparison.npz"


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "portable_hup060_fresh_wgangp_runner", ROOT / "run_experiment.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load bundled experiment runner")
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    runner.SOURCE = TRAINING_SOURCE
    runner.PAPER_PAIRED = BASELINE
    runner.HERE = ROOT / "portable_reproductions"
    return runner


def verify_bundle() -> None:
    spec = importlib.util.spec_from_file_location(
        "portable_hup060_bundle_manifest", ROOT / "bundle_manifest.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load manifest verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    verification = module.verify()
    if not verification["all_present_and_matching"]:
        print(json.dumps(verification, indent=2, ensure_ascii=False))
        raise RuntimeError("A required bundled source or input failed verification")
    runner = load_runner()
    source = runner.source_module()
    runner.torch.set_num_threads(4)
    runner.torch.manual_seed(20261011)
    setup = runner.setup(source)
    actor = setup[9]
    print(json.dumps({
        "mode": "non_training_portable_preflight",
        "hash_verification": verification,
        "source_path": str(runner.SOURCE),
        "model_sha256": setup[-1],
        "reference_fit_shape": list(setup[1].shape),
        "reference_validation_shape": list(setup[2].shape),
        "actuators": setup[3].tolist(),
        "actor_trainable_parameters": sum(p.numel() for p in actor.parameters()),
        "portable_output_root": str(runner.HERE),
        "training_started": False,
        "paper_baseline_outcome_arrays_opened": False,
        "teacher_checkpoint_loaded": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if sys.argv[1:] == ["--verify-bundle"]:
        verify_bundle()
    else:
        load_runner().main()
