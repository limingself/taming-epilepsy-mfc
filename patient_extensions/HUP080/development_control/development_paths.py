"""Location of authorised inputs and private replay output, independent of the shell cwd."""
from pathlib import Path
import os

CODE = Path(__file__).resolve().parent
PROJECT = CODE.parents[2]
ART = Path(os.environ.get("HUP080_DEV_ARTIFACTS", PROJECT / "patient_results/HUP080_sparse_control_final_v2/frozen_run/science_run/artifacts")).resolve()
WORK = Path(os.environ.get("HUP080_DEV_WORK", PROJECT / "patient_results/HUP080_development_replay")).resolve()
FREE = WORK / "free"
CALIBRATION = WORK / "calibration"
FIRST = WORK / "controller"
FINAL = WORK / "refined/engine"
