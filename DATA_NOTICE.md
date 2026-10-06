# Data and clinical-use notice

## Current Full WGAN-GP source package (2026-10-06)

`releases/full_wgangp_20261006/` is an additive current-version code and
lightweight result deposit. It preserves the existing patient-extension
publication boundary: HUP065/HUP080 signal arrays, serialized models and
controller weights, large density tables, training histories, raw ZIP/EDF
archives, and one-time authorization/access receipts are not redistributed.
Exact scientific code, parameter metadata, aggregate/channel metrics,
publication figures and a public-file hash verifier are supplied instead.

The complete offline revision bundles retain those inputs, weights, logs
and candidate outcomes for reproducibility. Replaying a frozen controller
or retraining it requires that separately obtained bundle and the documented
dependencies. Passing the signal-free public verifier is not evidence that
an external evaluation or retraining was executed, nor a claim that all
private input files were uploaded. The unadopted HUP065 cold-start experiment
is kept separate from the adopted manuscript results.

The current source manifest and `CURRENT_RELEASE.json` supersede the old
root `PAPER_FINAL_VERSION.json` only as the version pointer; historical
files and scientific results below are not overwritten.

This repository does not contain the original EEG archives. The subject codes
`HUP060`, `HUP065`, and `HUP080` refer to de-identified participants in OpenNeuro dataset
`ds004100`, version 1.1.3 (DOI: `10.18112/openneuro.ds004100.v1.1.3`; CC0 in
the distributed dataset metadata). Users who retrain the models must obtain
the dataset independently and comply with its current terms, ethics
documentation, and citation requirements.

The committed `artifacts/` and `output/` files are frozen numerical research
derivatives used to reproduce the manuscript figures. Channel labels and
seizure-onset/resection annotations are retained only at the de-identified
electrode level required by the analysis.

The additive `patient_extensions/` package contains portable source code,
configuration templates, lightweight aggregate results and pass vectors,
publication PDF/PNG files, and provenance/QA manifests. It does not redistribute
patient signal arrays, serialized HUP065/HUP080 models, raw ZIP/EDF data, large
channel-density tables, training histories, run caches, failed staging trees,
or one-time authorization/access receipts. The two private-archive manifest
hashes identify offline provenance ledgers only and do not provide data access.

This software is for research and reproducibility only. It is not a medical
device and must not be used to diagnose, predict, or treat an individual
patient.

The committed Joblib and PyTorch files are serialized model artifacts. Load
only copies obtained from this repository and verify them against the hashes
recorded in `PAPER_FINAL_VERSION.json` or the reproduction manifest; do not
load untrusted pickle-compatible files.
