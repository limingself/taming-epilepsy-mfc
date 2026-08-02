#!/usr/bin/env python
"""One-command reproduction entry point for all eight manuscript figures."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PART_SCRIPTS = {
    1: [ROOT / "part1_network" / "figure_01_plv_network_selection.py"],
    2: [
        ROOT / "part2_rc_sde" / "figure_02_rc_sde_prediction.py",
        ROOT / "part2_rc_sde" / "figure_03_all36_distributions.py",
        ROOT / "part2_rc_sde" / "figure_04_distribution_errors.py",
        ROOT / "part2_rc_sde" / "figure_05_input_ablation.py",
    ],
    3: [
        ROOT / "part3_mfc" / "figure_06_actor_wgan_control.py",
        ROOT / "part3_mfc" / "figure_07_mfc_ablation.py",
        ROOT / "part3_mfc" / "figure_08_all36_controlled_distributions.py",
    ],
}


def run_script(path: Path, arguments: tuple[str, ...] = ()) -> None:
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("PYTHONHASHSEED", "0")
    environment.setdefault("MNE_DONTWRITE_HOME", "true")
    environment.setdefault("MNE_LOGGING_LEVEL", "WARNING")
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    runtime = ROOT / ".runtime"
    (runtime / "matplotlib").mkdir(parents=True, exist_ok=True)
    environment.setdefault("MPLCONFIGDIR", str(runtime / "matplotlib"))
    subprocess.run(
        [sys.executable, str(path), *arguments],
        cwd=ROOT,
        env=environment,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("all", "1", "2", "3"), default="all")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()
    parts = (1, 2, 3) if args.part == "all" else (int(args.part),)
    for part in parts:
        if part == 2:
            run_script(
                ROOT / "part2_rc_sde" / "part2_model.py",
                ("source-data",),
            )
        for script in PART_SCRIPTS[part]:
            run_script(script)
    if not args.skip_verify:
        from verify_outputs import verify

        verify(None if args.part == "all" else set(parts))
    print(f"Paper figures are available in: {ROOT / 'output'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
