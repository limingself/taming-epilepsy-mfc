"""Immutable per-patient dimensions and the local single-window replay contract."""
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np


# Hash newline-separated UTF-8 channel labels in their recorded order, no trailing LF.
# Compare labels rather than cache names: cache keys can depend on local zip timestamps.
PATIENTS = {
    "HUP065": (64, 32, "a7f2f47c5eca55e59567c6b7ea8850a87779450701c101773dac58a3cdb5abea"),
    "HUP080": (96, 76, "cde0cdb825b79269c3280bb5b6b8c0883fb6ad76f731100c4d7a460739156429"),
}
WINDOWS = {
    "run01_ictal": ("run-01", 120., 32.),
    "run01_reference": ("run-01", 30., 40.),
    "run02_ictal": ("run-02", 120., 32.),
}
PREPROCESSING = {
    "bandpass_causal": True, "bandpass_hz": [1., 50.], "filter_order": 4,
    "filter_type": "Butterworth SOS (forward only)", "filter_burn_in_s_applied": 20.,
    "reference": "common average over selected status=good channels",
    "resampling_mode": "causal_upfirdn", "resampling_causal": True,
    "causal_fir_half_length_factor": 10, "causal_fir_group_delay_s": .0390625,
    "preprocessing_fully_causal": True,
}


@dataclass(frozen=True)
class Replay:
    subject: str
    project: Path
    output: Path
    run01_ictal: Path
    run01_reference: Path
    run02_ictal: Path

    @property
    def channels(self):
        return PATIENTS[self.subject][0]

    @property
    def actuators(self):
        return PATIENTS[self.subject][1]


def settings():
    """Read a contract supplied by run.py to scientific-stage subprocesses."""
    return Replay(os.environ["SWR_SUBJECT"], *[
        Path(os.environ["SWR_" + key.upper()]).resolve()
        for key in ("project", "output", *WINDOWS)
    ])


def check_segment(case, role):
    """Check metadata, recorded channel order and NPY headers without loading signals."""
    path = getattr(case, role)
    run, start, duration = WINDOWS[role]
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata_json"].item()))
        names = z["channel_names"].astype(str)
    with ZipFile(path) as archive, archive.open("data.npy") as fd:
        version = np.lib.format.read_magic(fd)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(fd)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(fd)
        else:
            raise ValueError(f"{role}: unsupported NPY header version {version}")
    expected = dict(subject=case.subject, run=run, start_s=start, duration_s=duration,
                    sfreq=256., source_sfreq=512., units="uV", task="ictal")
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"{role}: expected {key}={value!r}, got {meta.get(key)!r}")
    for key, value in PREPROCESSING.items():
        if meta["provenance"].get(key) != value:
            raise ValueError(f"{role}: preprocessing mismatch at {key}")
    if tuple(shape) != (case.channels, int(256 * duration)) or dtype != np.dtype("float32"):
        raise ValueError(f"{role}: unexpected channel/sample shape or dtype: {shape}, {dtype}")
    basis = sha256("\n".join(names).encode("utf-8")).hexdigest()
    if len(np.unique(names)) != case.channels or basis != PATIENTS[case.subject][2]:
        raise ValueError(f"{role}: channel basis differs from recorded good-channel order")
    return {"role": role, "run": run, "start_s": start, "duration_s": duration,
            "shape": list(shape), "ordered_channel_sha256": basis}
