# HUP060 Part-II context-3 same-window contract

The manifest freezes the fourth predeclared HUP060 run-02 origin (`p=3547`) and its exact 256-sample analysis interval. The last known state is sample 3546; evaluation samples are `3547:3803`. Observed, sealed no-control, and any future controlled trajectories must use the same `k/256`, `k=0,...,255`, coordinate. Densities are occupation laws from this exact interval.

For synthesis, use only the run-01 train-only reference target. The saved run-02 reference arrays are evaluation-only. The manifest contains shapes and content hashes. `same_window_legal_inputs.npz` is the restricted runner package: it deliberately excludes the run-02 reference-evaluation pool/template. Its observed window is explicitly evaluation-only and must never enter synthesis, tuning, early stopping, or path selection.
