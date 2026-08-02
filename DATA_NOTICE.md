# Data and clinical-use notice

This repository does not contain the original EEG archives. The subject code
`HUP060` refers to a de-identified participant in OpenNeuro dataset
`ds004100`, version 1.1.3 (DOI: `10.18112/openneuro.ds004100.v1.1.3`; CC0 in
the distributed dataset metadata). Users who retrain the models must obtain
the dataset independently and comply with its current terms, ethics
documentation, and citation requirements.

The committed `artifacts/` and `output/` files are frozen numerical research
derivatives used to reproduce the manuscript figures. Channel labels and
seizure-onset/resection annotations are retained only at the de-identified
electrode level required by the analysis.

This software is for research and reproducibility only. It is not a medical
device and must not be used to diagnose, predict, or treat an individual
patient.

The committed Joblib and PyTorch files are serialized model artifacts. Load
only copies obtained from this repository and verify them against the hashes
recorded in `PAPER_FINAL_VERSION.json` or the reproduction manifest; do not
load untrusted pickle-compatible files.
