"""Path-only portable wrapper for the exact finite HUP080 adaptation script.

The optimizer snapshot is byte-identical to the executed script. Its function
bodies, loss coefficients, training/selection/evaluation logic are unchanged.
Only filesystem bindings and the immutable base-runner loader are relocated.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys
import uuid
import torch

ROOT = Path(__file__).resolve().parent
EXPECTED_OPTIMIZER = "b6b7db88daf222ce852747cdabc90ecd3ab0f67bb6bc4b767262872d2e275570"
EXPECTED_CHECKPOINT = "cdffedfea81d3e57943fe4f47bde6155efe3661fbeb325db9fda221213c68fc0"

def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify-bundle", "preflight", "screen-evaluate", "evaluate"))
    parser.add_argument("--tag", default=None)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    portable = module_from(ROOT / "base_bundle/reproduce_external.py", "hup080_optimization_portable_base")
    manifest = portable.read_manifest()
    count = portable.verify_hashes(manifest)
    if portable.sha(ROOT / "optimize_hup080_snapshot.py") != EXPECTED_OPTIMIZER:
        raise RuntimeError("The executed optimizer snapshot changed")
    checkpoint = ROOT / "starting_checkpoint/frozen_actor_wgan.pt"
    if portable.sha(checkpoint) != EXPECTED_CHECKPOINT:
        raise RuntimeError("Warm-start checkpoint changed")
    opt = module_from(ROOT / "optimize_hup080_snapshot.py", "hup080_optimization_exact_snapshot")
    runtime = ROOT / ".runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    if args.mode == "verify-bundle":
        output = runtime / ("verify_" + uuid.uuid4().hex)
    else:
        if not args.tag or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,79}", args.tag):
            raise RuntimeError("A new explicit alphanumeric --tag is required")
        output = ROOT / "reruns" / args.tag
        if args.mode == "preflight" and output.exists():
            raise RuntimeError("Refuse preflight overwrite; use a new tag")
        if args.mode != "preflight" and not output.is_dir():
            raise RuntimeError("Run tagged preflight first")
    opt.HERE = output
    opt.START = checkpoint
    def relocated_base():
        return portable.load_runner("HUP080", manifest, runtime, output)
    opt.load_runner = relocated_base
    base = relocated_base()
    if args.mode == "verify-bundle":
        state = opt.setup(base)
        final_manifest = json.loads((ROOT / "final_snapshot_manifest.json").read_text(encoding="utf-8"))
        for item in final_manifest["files"]:
            path = ROOT / "final_snapshot" / item["relative_path"]
            if portable.sha(path) != item["sha256"]:
                raise RuntimeError("Frozen final diagnostic snapshot changed")
        final_path = ROOT / "final_snapshot/selected/frozen_actor.pt"
        if portable.sha(final_path) != final_manifest["checkpoint_sha256"]:
            raise RuntimeError("Frozen adapted-controller bytes changed")
        final_checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
        state["actor"].load_state_dict(final_checkpoint["actor_state_dict"], strict=True)
        state["critic"].load_state_dict(final_checkpoint["critic_state_dict"], strict=True)
        if final_checkpoint["input_hashes"] != state["hashes"] or final_checkpoint["model_sha256"] != base.MODEL_HASHES["HUP080"]:
            raise RuntimeError("Final controller's frozen predictive inputs differ")
        for name, module in list(sys.modules.items()):
            if name == "mfc_pipeline" or name.startswith("mfc_pipeline."):
                source = Path(module.__file__).resolve()
                if portable.PROJECT not in source.parents:
                    raise RuntimeError("A scientific dependency imported outside the snapshot")
        portable.verify_hashes(manifest)
        print(json.dumps(dict(status="verified", snapshot_files_verified=count,
            optimizer_sha256=EXPECTED_OPTIMIZER, starting_checkpoint_sha256=EXPECTED_CHECKPOINT,
            channels=96, actuators=len(state["selected"]), predictive_model_sha256=base.MODEL_HASHES["HUP080"],
            old_rms_cap=.405, revised_rms_cap=.45,
            same_optimizer_function_bodies=True, only_filesystem_routing_rebound=True,
            warm_start_checkpoint_loaded=True, outer_arrays_unpacked=False,
            final_adapted_actor_and_critic_strict_load=True,
            final_snapshot_files_verified=len(final_manifest["files"]),
            final_adapted_checkpoint_sha256=final_manifest["checkpoint_sha256"],
            training_or_evaluation_run=False, original_D_bindings_relocated=True)), flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "preflight":
        opt.preflight(base)
    elif args.mode == "screen-evaluate":
        if not (output / "optimization_contract.json").is_file():
            raise RuntimeError("Completed tagged preflight required")
        for name in opt.ARMS:
            opt.train_arm(base, name, 150)
        opt.freeze_selection(base)
        opt.evaluate_frozen(base)
    elif args.mode == "evaluate":
        opt.evaluate_frozen(base)

if __name__ == "__main__":
    main()
