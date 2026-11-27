# Single-window replay: HUP065 and HUP080

HUP065 supplies the updated Fig. S4 in the local red manuscript revision.
HUP080 is a companion comparison from the same pipeline; it does **not**
replace the existing Fig. S5. Both cases use the
HUP060 preprocessing, Graph-RC-SDE, Full controller, initialization and window
rule. The fixed channel/input counts are **64/32** and **96/76**, respectively.
These are retrospective model-based controls, not intervention recordings.
The separate three-case summaries retain HUP065 run-03 and HUP080 run-04;
these examples do not replace those experiments.

## Run from original recordings

Use Python 3.11+ and the dependencies in the repository's
[`requirements-lock.txt`](../../requirements-lock.txt). Obtain OpenNeuro
`ds004100` v1.1.3 independently; see [data and usage notice](../../DATA_NOTICE.md).
From this directory in a **full** clone, replace the illustrative input/output paths
below with local paths. No network requests or private patient files are needed
from the authors' workstations. Use new output directories outside the repository;
raw segments, checkpoints, per-path arrays and training logs remain private.

```sh
python -B prepare_inputs.py --subject HUP065 --bids-root /data/ds004100 --output /work/065-inputs
python -B run.py --subject HUP065 --input-map /work/065-inputs/input_map.json --output /work/065-replay --through full

python -B prepare_inputs.py --subject HUP080 --bids-root /data/ds004100 --output /work/080-inputs
python -B run.py --subject HUP080 --input-map /work/080-inputs/input_map.json --output /work/080-replay --through full
```

`--bids-root` contains the actual `sub-HUPxxx/ses-presurgery/ieeg` directories,
including EDF data (not Git-annex pointers), run sidecars, electrodes and
coordinate-system files. Alternatively pass `--zip-root /data/subject-zips`
to read an existing `sub-HUPxxx.zip` through the same public loader. For a
separate complete source checkout, both scripts accept
`--project-root /path/to/taming-epilepsy-mfc`; fitting and controller training
resolve their canonical code from that same location.

If you already have the canonical local segment NPZs, skip preparation and
pass `--run01-ictal FILE --run01-reference FILE --run02-ictal FILE` instead of
`--input-map`. Roles are checked using embedded subject, run, clock, units,
filter/CAR/resampling metadata, array dimensions and channel-order digests;
filenames or archive timestamps need not match. Fit/scale selection does not
consult run-02 future values. To check only the input headers, use
`--through check` with the same arguments (writes nothing). `fit`, `noise`
and `calibrate` stop after their named stage; `full` runs all stages in order.
Each invocation starts from scratch in an empty output directory.

## What is fitted, selected and evaluated

| Role | Locked setting |
|---|---|
| Input preprocessing | Intersection of status=good ECoG contacts in the two recording sidecars, in run-01 order. Forward fourth-order 1–50 Hz Butterworth, 20 s burn-in, common-average reference and causal FIR 512→256 Hz resampling (0.0390625 s retained group delay). No run-02 z-score fitting. |
| Dynamics/graph | One continuous ictal run-01 EDF interval [120,152) s, 8,192 samples. Run-01 inner fit [0,22) s, 2 s gap, validation [24,32) s relative to ictal onset; final refit uses the full 32 s. Fixed PCA threshold 0.95 **subject to** maximum 24 components. Fixed five-band PLV ranking; offline band decomposition for the ranking uses run-01 only. |
| Noise/input scales | Run-01 only: seven validation forecast origins for κ in {0.50, 0.75, 1.00, 1.25} times 0.79451175. Choose the input-step scale from {0.25,0.50,1.00} using 12-update run-01-only pilots; fresh initialization for the full run. Neither patient is tuned on the run-02 display outcome. |
| Reference | The run-01 preictal NPZ spans EDF [30,70) s. The first 30 s form 30 one-second paths: EDF [30,45) fit the reference, [45,60) validate/select and supply the displayed target law. |
| Run-02 example | Use the transferred HUP060 window rule: ictal-crop sample indices [3291,3547) for the available past and [3547,3803) for the hidden recorded future, i.e. EDF [133.85546875,134.85546875) s for the forecast. The actor can use this **past** when synthesized; the future is not a controller target. |
| Full control | 1,000 actor updates; 3,024 critic updates including warm-up. Paired free and controlled evaluation uses the same 32 paths (16 antithetic pairs), with no mid-window observed-state refresh. |

The public code refits the plant, computes graphs and calibrates κ rather than
loading release weights. The controller adapter calls the pinned repository
[`model_core.py`](../../part3_mfc/model_core.py) and
[`train_full_wgangp.py`](../../part3_mfc/train_full_wgangp.py).
See [reported results](reported_results.json) for frozen aggregate distances
and baseline fidelity. Time-resolved W1 compares channel marginal laws at the
same time, then averages over time. Occupation W1 pools time before comparing.
Observed denotes one measured path, free and controlled denote simulations;
neither a distance to the reference nor a fit increment score is an observed
treatment effect or a held-out phase-accurate waveform correlation.

## Outputs and verification

The private output directory contains the fit and noise receipts under
`dynamics/`, input-scale receipts under `scale_calibration/`, and the
frozen training/evaluation receipts and `combined_policy_eval.npz` under
`control/`. `figures/` contains the all-contact density and trajectory plots
and their numerical metrics. Use [`render.py`](render.py) to redraw the
figures from the locally generated combined archive.

```sh
python -B -m unittest discover -s tests
python -B tests/check_private_dynamics.py --subject HUP065 --input-map /work/065-inputs/input_map.json --output /work/065-replay
python -B tests/check_private_dynamics.py --subject HUP080 --input-map /work/080-inputs/input_map.json --output /work/080-replay
```

The first command is signal-free. The optional real-data checks require
`--through noise` or later and compare sample-array digests, graph rankings,
fitted PCA dimensions and all four κ validation scores. The full run also
audits the paired free trajectories, frozen selected epoch, allowed input
bounds, metric archive and checkpoint/contract hashes. Do not load untrusted
Joblib or Torch checkpoints.
