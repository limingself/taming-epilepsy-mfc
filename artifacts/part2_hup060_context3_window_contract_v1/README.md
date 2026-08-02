# HUP060 Part-II context-3 same-window contract

> **Status: superseded historical provenance only.** This contract records the former
> `diffusion_scale=1.0` rollout. It remains useful for the fixed window coordinates and
> leakage audit, but `sealed_no_control_scaled` must not be used for current Figs. 2--5
> or for control-gain calculations. The current authority is
> `output/part2/source_data/figures_02_04/unified_protocol_summary.json`.

The manifest freezes the fourth predeclared HUP060 run-02 origin (`p=3547`) and its exact 256-sample analysis interval. The last known state is sample 3546; evaluation samples are `3547:3803`. Observed, sealed no-control, and any future controlled trajectories must use the same `k/256`, `k=0,...,255`, coordinate. Densities are occupation laws from this exact interval.

For synthesis, use only the run-01 train-only reference target. The saved run-02 reference arrays are evaluation-only. The manifest contains shapes and content hashes. `same_window_legal_inputs.npz` is the restricted runner package: it deliberately excludes the run-02 reference-evaluation pool/template. Its observed window is explicitly evaluation-only and must never enter synthesis, tuning, early stopping, or path selection.
