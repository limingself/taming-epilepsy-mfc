# Part II: state-dependent Graph-RC SDE

This directory exposes one Part-II model command and one plotting script for
each manuscript figure.  The frozen model uses state-dependent diffusion; it
is not the superseded constant-diffusion baseline.

## Unified prediction/control protocol

Part II now reports the exact `u=0` particle branch used as the uncontrolled
arm in Part III.  Both parts therefore share the frozen Graph-RC SDE, context-3
initial Markov state, effective diffusion multiplier `0.79451175`, 32
particle-wise innovation paths, and 1-s horizon.  The full arm of manuscript Fig. 3 (historical export ID 05) is
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

`source-data` rebuilds prediction tables from frozen evaluation artifacts. For the new Fig. 3 it uses the matched four-arm artifact if available, or validates the committed two-table Source Data rather than substituting the superseded ablation.
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
- `figure_05_input_ablation.py`: current manuscript Fig. 3, whose historical
  export ID and file stem are retained to preserve existing links.

Each plotting script reads only its committed Source Data and exports SVG,
PDF, PNG, and 600-dpi TIFF locally.  Shared numerical implementation lives in
`mfc_pipeline/`; it is not a second model version.

## Reproduce the nested feature comparison (manuscript Fig. 3)

Ordinary redraw (no patient-level files required): run
`python part2_rc_sde/figure_05_input_ablation.py`. The committed Source Data
contain scores for each of 36 contacts and KDE curves for RPFa3, RA3 and RAFa4,
not participant time series. To verify the exact table digests and scores run
`python part2_rc_sde/nested_ablation_source.py`.

For independent model-based regeneration, supply an authorized copy of the matched
run-01 training cache (120–152 s, 8,192 samples) and original frozen model,
context, and two paired evaluations under the release's `artifacts/` locations
listed in `refit_nested_ablation.py`. The runner checks their SHA-256 digests;
none of its numerical inputs depend on a previously generated four-arm run.
Use a new private destination for the particle trajectories:

```powershell
python part2_rc_sde/refit_nested_ablation.py --training-cache PATH_TO_RUN01_CACHE --out PATH_TO_NEW_PRIVATE_RUN
python part2_rc_sde/nested_ablation_source.py --rollouts PATH_TO_NEW_PRIVATE_RUN/nested_runs.npz --paired-reference artifacts/part3_hup060_actor_wgan_v1/seed20261011_adv050_anchor020/run02_development_evaluation/paired_comparison.npz --out PATH_TO_NEW_TABLES
python part2_rc_sde/figure_05_input_ablation.py --source PATH_TO_NEW_TABLES --out PATH_TO_NEW_FIGURES
```

The four arms add (i) current latent state, (ii) explicit lags (1, 8, 32
samples), (iii) graph features and (iv) the 96-unit reservoir. Only (iv)
uses RC. Drift and conditional variance are refitted for each reduced model.
All four use identical PCA, 8,127 training transitions, context 3 of run-02,
32 paired innovation paths and a 256-sample free forecast. Occupation
Wasserstein-1 is computed from empirical samples, not KDE curves. The contact
means are 0.178, 0.144, 0.158 and 0.110: adding graph terms alone did not
improve this forecast on average, while adding RC reduced the error on 27/36
contacts relative to (iii). This is a matched, single-context development
comparison and not a universal ranking of models or a held-out seizure test.
