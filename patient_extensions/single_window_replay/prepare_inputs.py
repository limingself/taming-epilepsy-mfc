"""Recreate the three private causal NPZ inputs from a local BIDS tree or subject ZIP."""
import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
from replay_config import PATIENTS, WINDOWS, Replay, check_segment


def from_tree(tree, subject, destination):
    """The canonical loader consumes a ZIP; package just two ictal runs and sidecars."""
    folder = tree / ("sub-" + subject) / "ses-presurgery" / "ieeg"
    prefix = f"sub-{subject}_ses-presurgery_"
    members = [folder / f"{prefix}task-ictal_acq-ecog_run-{run:02}_{ext}"
               for run in (1, 2) for ext in ("ieeg.edf", "ieeg.json", "channels.tsv", "events.tsv")]
    members += sorted(folder.glob(f"{prefix}acq-ecog_*_coordsystem.json"))
    members += sorted(folder.glob(f"{prefix}acq-ecog_*_electrodes.tsv"))
    with ZipFile(destination / f"sub-{subject}.zip", "x") as z:
        for member in members:
            z.write(member, member.relative_to(tree).as_posix())


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", required=True, choices=PATIENTS)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--zip-root", type=Path, help="directory containing sub-HUPxxx.zip")
    grp.add_argument("--bids-root", type=Path, help="directory containing sub-HUPxxx/")
    p.add_argument("--project-root", type=Path, default=here.parents[1])
    p.add_argument("--output", type=Path, required=True, help="new/empty private input directory")
    args = p.parse_args()
    project = args.project_root.resolve()
    out = args.output.resolve()
    if out.is_relative_to(here) or out == project:
        p.error("use a separate private output directory for signals")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        p.error("output is not empty; select a fresh directory")
    sys.path.insert(0, str(project))
    from mfc_pipeline.data import BIDSZipDataset, EEGSegmentLoader

    with ExitStack() as stack:
        if args.zip_root:
            root = args.zip_root.resolve()
        else:
            # Explicit known-name private adapter archive; loader reads just two runs.
            # Do not depend on platform-specific permissions for mode-0700 temp dirs.
            root = out
            from_tree(args.bids_root.resolve(), args.subject, root)
            stack.callback((root / f"sub-{args.subject}.zip").unlink)
        dataset = BIDSZipDataset(root, subjects=[args.subject])
        records = {key: dataset.get_record(args.subject, "ictal", key) for key in ("run-01", "run-02")}
        names = dataset.common_good_channels(args.subject, records=list(records.values()))
        if sha256("\n".join(names).encode()).hexdigest() != PATIENTS[args.subject][2]:
            raise ValueError("BIDS good-channel intersection/order does not match this case")
        loader = EEGSegmentLoader(dataset, cache_dir=None, temp_dir=out / "_edf_tmp",
                                  target_sfreq=256., bandpass_hz=(1., 50.), filter_order=4,
                                  burn_in_s=20., reference="car", resampling_mode="causal_upfirdn",
                                  causal_fir_half_length_factor=10)
        mapping = {"subject": args.subject}
        digests = {}
        for role, (run, start, duration) in WINDOWS.items():
            meta = dataset.metadata(records[run])
            if meta.onset_s != 120. or (start == 120. and meta.effective_offset_s < start + duration):
                raise ValueError(f"recorded event bounds disagree with the {role} protocol")
            seg = loader.load(records[run], start_s=start, duration_s=duration, channel_names=names,
                              use_cache=False)
            if not np.isfinite(seg.data).all():
                raise ValueError(f"nonfinite samples in {role}")
            path = out / f"{role}.npz"
            seg.to_npz(path)
            mapping[role] = path.name
            digests[role] = sha256(seg.data.astype("<f4", copy=False).tobytes()).hexdigest()
        case = Replay(args.subject, project, out, *[out / mapping[k] for k in WINDOWS])
        for role in WINDOWS:
            check_segment(case, role)
    (out / "input_map.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    (out / "signal_sha256.json").write_text(json.dumps(digests, indent=2), encoding="utf-8")
    print("Validated three causal inputs; pass --input-map", out / "input_map.json")


if __name__ == "__main__":
    main()
