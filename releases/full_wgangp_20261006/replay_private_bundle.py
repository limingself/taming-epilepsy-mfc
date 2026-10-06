"""Explicit, dry-run-first launcher for an independently supplied private bundle.

No terminal or outer evaluation action is provided. Public verification never
calls this script and never opens private arrays or serialized models.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--action", required=True, choices=(
        "verify-hup060", "verify-comparisons", "verify-hup065", "verify-hup080",
        "train-hup060", "train-cold-hup065", "preflight-cold-hup065"))
    parser.add_argument("--tag")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.bundle_root.expanduser().resolve()
    tagged = args.action not in {"verify-hup060", "verify-comparisons"}
    if tagged and (not args.tag or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", args.tag)):
        parser.error("An explicit new simple --tag is required for this action")
    mapping = {
        "verify-hup060": ("supporting_materials/HUP060_restart/reproduce_portable.py", ["--verify-bundle"]),
        "verify-comparisons": ("supporting_materials/HUP060_comparisons/reproduce_hup060_comparisons.py", ["verify"]),
        "verify-hup065": ("supporting_materials/external_parameter_adaptations/HUP065/small_gain_decoupled_screen/reproduce_portable_small_gain.py", ["verify-bundle", "--threads", "2", "--tag", args.tag]),
        "verify-hup080": ("supporting_materials/external_parameter_adaptations/HUP080/portable/reproduce_optimization.py", ["verify-bundle", "--threads", "2", "--tag", args.tag]),
        "train-hup060": ("supporting_materials/HUP060_restart/reproduce_portable.py", ["train", "--tag", args.tag, "--epochs", "1000", "--seed", "20261011", "--threads", "4"]),
        "preflight-cold-hup065": ("run_cold_top32.py", ["preflight", "--tag", args.tag, "--threads", "2"]),
        "train-cold-hup065": ("run_cold_top32.py", ["train", "--tag", args.tag, "--threads", "2"]),
    }
    relative, arguments = mapping[args.action]
    target = root / relative
    command = [sys.executable, "-B", "-X", "utf8", str(target), *arguments]
    print(json.dumps({"private_bundle_explicitly_supplied": str(root), "action": args.action,
                      "command": command, "dry_run": not args.execute,
                      "public_package_has_not_loaded_private_inputs": True}, indent=2))
    if not args.execute:
        return 0
    if not target.is_file() or root not in target.resolve().parents:
        raise SystemExit("Supplied complete bundle entry does not exist or escapes its root")
    return subprocess.run(command, cwd=str(root), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
