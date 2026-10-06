"""Portable path-only wrapper for archived HUP060 comparison implementations.

No model equation, loss coefficient, random-seed rule or checkpoint-selection
rule is changed. Published outputs are never overwritten. Use ``verify`` for a
non-training, outcome-free import/setup check.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
FULL = ROOT.parent / "HUP060_restart"
PROJECT = FULL / "repro_source"
SOURCE = PROJECT / "part3_mfc/part3_model.py"
BASELINE_SOURCE = PROJECT / "part3_mfc/trivial_baseline_experiment.py"
PAPER_PAIR = FULL / "repro_inputs/paper_baseline_paired_comparison.npz"
FULL_PAIR = FULL / "runs/fresh_seed20261011_e1000/evaluation/paired_comparison.npz"
EXPECTED_CORE_SHA = "57da573dbe300ad6bd585fbd69e46102d9f315dba8862e63c7d3e8320b6943e1"
EXPECTED_BASELINE_SHA = "ed0c5611af5be1f46dd5328885713f2087e1487396394c9bc8f4ccee531eb0eb"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def runner():
    module = load(ROOT / "run_matched_ablation.py", "portable_hup060_comparison_runner")
    module.SOURCE = SOURCE
    module.PAPER_PAIRED = PAPER_PAIR
    module.HERE = ROOT / "portable_reproductions"
    return module


def baseline_module():
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    return load(BASELINE_SOURCE, "portable_hup060_baseline_core")


def verify():
    manifest = load(FULL / "bundle_manifest.py", "portable_hup060_full_manifest")
    validation = manifest.verify()
    if not validation["all_present_and_matching"]:
        raise RuntimeError("Full frozen-source bundle failed verification")
    if sha(SOURCE) != EXPECTED_CORE_SHA or sha(BASELINE_SOURCE) != EXPECTED_BASELINE_SHA:
        raise RuntimeError("Canonical comparison source hash changed")
    module = runner()
    module.torch.set_num_threads(4)
    setups = []
    for arm in ["no_wgan", "no_graph_spread", "no_deviation"]:
        module.ACTIVE_ABLATION = arm
        module.torch.manual_seed(20261011)
        core = module.source_module()
        if arm == "no_graph_spread":
            core.GRAPH_DIFFUSION_TIME = 0.0
        setup = module.setup(core)
        setups.append({"arm": arm, "model_sha256": setup[-1],
                       "actuators": setup[3].tolist(),
                       "reference_fit_shape": list(setup[1].shape),
                       "reference_validation_shape": list(setup[2].shape)})
    baseline = baseline_module()
    paths = [baseline.core.SYNTHESIS_CONTRACT, baseline.core.MODEL_PATH,
             baseline.core.LEGAL_EVALUATION, baseline.core.SEALED_CALIBRATED, FULL_PAIR]
    for path in paths:
        if not path.is_file() or PROJECT not in path.parents and FULL not in path.parents:
            raise RuntimeError(f"Required input does not resolve inside archive: {path}")
    print(json.dumps({"status": "passed_non_training_portable_preflight",
                      "training_started": False, "outcome_arrays_opened": False,
                      "full_bundle_all_present_and_matching": True,
                      "core_sha256": sha(SOURCE), "baseline_source_sha256": sha(BASELINE_SOURCE),
                      "ablation_setups": setups,
                      "baseline_input_paths": [str(path) for path in paths],
                      "baseline_input_sha256": {str(path): sha(path) for path in paths}},
                     ensure_ascii=False, indent=2))


def run_ablation(command, arguments):
    module = runner()
    if "--archived" in arguments:
        if command != "evaluate-ablation":
            raise ValueError("--archived is for frozen evaluation, not training")
        arguments = [item for item in arguments if item != "--archived"]
        tag = arguments[arguments.index("--tag") + 1]
        if any(character in tag for character in "/\\:"):
            raise ValueError("Invalid tag")
        source = ROOT / "runs" / tag
        target = module.HERE / "runs" / tag
        if target.exists():
            raise RuntimeError("Refusing to overwrite a portable output directory")
        target.mkdir(parents=True)
        for name in ["frozen_actor_wgan.pt", "training_summary.json"]:
            shutil.copy2(source / name, target / name)
    sys.argv = [str(ROOT / "run_matched_ablation.py"),
                "train" if command == "train-ablation" else "evaluate", *arguments]
    module.main()


def run_baselines(arguments):
    if len(arguments) != 2 or arguments[0] != "--tag":
        raise ValueError("Use baselines --tag <new-output-name>")
    tag = arguments[1]
    if not tag or any(character in tag for character in "/\\:"):
        raise ValueError("Invalid tag")
    output = ROOT / "portable_reproductions/baselines" / tag
    if output.exists():
        raise RuntimeError("Refusing to overwrite a portable baseline directory")
    import torch
    import numpy as np
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    module = baseline_module()
    module.FULL_PAIR = FULL_PAIR
    module.SOURCE_ROOT = output / "source_data"
    module.ARTIFACT_ROOT = output / "artifacts"
    original_reconstruct = module.reconstruct_paired_normals
    with np.load(FULL_PAIR, allow_pickle=False) as paired:
        exact_free = paired["uncontrolled_scaled"].copy()

    def exact_reconstruction(stepper, initial, _historical_free, **kwargs):
        return original_reconstruct(stepper, initial, exact_free, **kwargs)

    module.reconstruct_paired_normals = exact_reconstruction
    return module.main(["--random-seeds", "32"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Commands: verify, train-ablation, evaluate-ablation, baselines")
    command, arguments = sys.argv[1], sys.argv[2:]
    if command == "verify" and not arguments:
        verify()
    elif command in ["train-ablation", "evaluate-ablation"]:
        run_ablation(command, arguments)
    elif command == "baselines":
        raise SystemExit(run_baselines(arguments))
    else:
        raise SystemExit("Unknown command. See README_REPRODUCIBILITY.md.")
