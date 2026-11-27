"""Fit, calibrate and replay a registered HUP065/HUP080 single-window experiment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from replay_config import PATIENTS, Replay, WINDOWS, check_segment


HERE = Path(__file__).resolve().parent
STAGES = ("check", "fit", "noise", "calibrate", "full")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", required=True, choices=PATIENTS)
    p.add_argument("--project-root", type=Path, default=HERE.parents[1],
                   help="complete checkout containing mfc_pipeline/ and part3_mfc/")
    p.add_argument("--output", type=Path, required=True, help="new or empty private directory")
    p.add_argument("--input-map", type=Path, help="local JSON created by prepare_inputs.py")
    for key in WINDOWS:
        p.add_argument("--" + key.replace("_", "-"), type=Path)
    p.add_argument("--through", choices=STAGES, default="full")
    args = p.parse_args(argv)
    if args.input_map:
        if any(getattr(args, k) is not None for k in WINDOWS):
            p.error("provide --input-map or all three input paths, not both")
        mapping = json.loads(args.input_map.read_text(encoding="utf-8"))
        if mapping.get("subject") != args.subject:
            p.error("subject does not match --input-map")
        inputs = [(args.input_map.resolve().parent / mapping[k]).resolve() for k in WINDOWS]
    else:
        if not all(getattr(args, k) is not None for k in WINDOWS):
            p.error("provide --input-map or all three --runXX-... paths")
        inputs = [getattr(args, k).resolve() for k in WINDOWS]
    case = Replay(args.subject, args.project_root.resolve(), args.output.resolve(), *inputs)
    receipts = [check_segment(case, key) for key in WINDOWS]
    print(json.dumps({"subject": case.subject, "segments": receipts}, indent=2), flush=True)
    if args.through == "check":
        return
    for rel in ("mfc_pipeline/part2_state_dependent_rc_sde.py", "mfc_pipeline/part2_data_pipeline.py",
                "part3_mfc/model_core.py", "part3_mfc/train_full_wgangp.py"):
        if not (case.project / rel).is_file():
            p.error(f"missing {rel}; use a complete checkout or --project-root")
    if case.output == case.project or case.output.is_relative_to(HERE):
        p.error("use a separate private output folder, not the repository root or package")
    case.output.mkdir(parents=True, exist_ok=True)
    if any(case.output.iterdir()):
        p.error("output is not empty: choose a fresh directory; checkpoints are never overwritten")
    (case.output / "input_contract.json").write_text(json.dumps(
        {"subject": case.subject, "segments": receipts}, indent=2), encoding="utf-8")
    env = dict(os.environ, SWR_SUBJECT=case.subject, SWR_PROJECT=str(case.project),
               SWR_OUTPUT=str(case.output), PYTHONDONTWRITEBYTECODE="1")
    for key in WINDOWS:
        env["SWR_" + key.upper()] = str(getattr(case, key))
    env["PYTHONPATH"] = os.pathsep.join([str(HERE), str(case.project), env.get("PYTHONPATH", "")])
    stages = {
        "fit": ["dynamics/fit_run01.py"],
        "noise": ["dynamics/run01_noise_grid.py"],
        "calibrate": ["control_stages.py", "calibrate"],
        "full": ["control_stages.py", "full"],
    }
    for name, (script, *stage_args) in stages.items():
        print(f"[{case.subject}] {name}", flush=True)
        subprocess.run([sys.executable, "-B", str(HERE / script), *stage_args],
                       env=env, cwd=case.output, check=True)
        if args.through == name:
            break


if __name__ == "__main__":
    main()
