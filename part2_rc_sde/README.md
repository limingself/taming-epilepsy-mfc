# Part II: state-dependent Graph-RC SDE

This directory exposes one Part-II model command and one plotting script for
each manuscript figure.  The frozen model uses state-dependent diffusion; it
is not the superseded constant-diffusion baseline.

## Unified prediction/control protocol

Part II now reports the exact `u=0` particle branch used as the uncontrolled
arm in Part III.  Both parts therefore share the frozen Graph-RC SDE, context-3
initial Markov state, effective diffusion multiplier `0.79451175`, 32
particle-wise innovation paths, and 1-s horizon.  The full arm of Fig. 5 is
required to reproduce that saved branch with zero maximum absolute discrepancy.

For HUP060 run-02, the representative prediction-to-recording occupation-law
Wasserstein-1 distances are `0.077`, `0.128`, and `0.109`; the cross-contact
mean and median across all 36 electrodes are `0.110` and `0.096`.

## Model entry point

```powershell
python part2_rc_sde/part2_model.py inspect
python part2_rc_sde/part2_model.py source-data
python part2_rc_sde/part2_model.py train
```

`source-data` rebuilds the Fig. 2--5 tables from frozen evaluation artifacts.
It does not select a context, particle, or noise path by outcome.  `train`
requires the independently obtained OpenNeuro archives configured in
`config_v2.yaml`.

## Independent figure scripts

- `figure_02_rc_sde_prediction.py`: representative trajectories,
  distributions, and trajectory error.
- `figure_03_all36_distributions.py`: predictive distributions for all 36
  electrodes.
- `figure_04_distribution_errors.py`: cross-electrode Wasserstein, mean, and
  standard-deviation error audit.
- `figure_05_input_ablation.py`: matched state, state+delay, and
  state+delay+PLV-graph input ablation.

Each plotting script reads only its committed Source Data and exports SVG,
PDF, PNG, and 600-dpi TIFF locally.  Shared numerical implementation lives in
`mfc_pipeline/`; it is not a second model version.
