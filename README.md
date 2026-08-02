# Taming Epilepsy: Mean Field Control of Whole-Brain Dynamics

Reproducible Python implementation and frozen figure bundle for a three-part
brain-network control workflow:

1. connected PLV functional-network construction and actuator selection;
2. state-dependent-diffusion Graph-RC-SDE prediction;
3. structured Actor + WGAN-GP mean-field distribution control.

The repository is organized so that every manuscript part has exactly one
public model entry point, while every manuscript figure has its own plotting
script.

![Representative mean-field control result](output/part3/figure_06/hup060_actor_wgan_mfc_preview.png)

## Code-to-figure map

| Part | Single model entry | Independent figure scripts | Figures |
|---|---|---|---|
| I. PLV network and actuator selection | `part1_network/part1_model.py` | `figure_01_plv_network_selection.py` | Fig. 1 |
| II. State-dependent Graph-RC-SDE | `part2_rc_sde/part2_model.py` | `figure_02_rc_sde_prediction.py`, `figure_03_all36_distributions.py`, `figure_04_distribution_errors.py`, `figure_05_input_ablation.py` | Figs. 2–5 |
| III. Actor-WGAN mean-field control | `part3_mfc/part3_model.py` | `figure_06_actor_wgan_control.py`, `figure_07_mfc_ablation.py`, `figure_08_all36_controlled_distributions.py` | Figs. 6–8 |

`mfc_pipeline/` contains shared numerical components imported by these three
entry points; it is not a second set of model variants.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python reproduce_all.py --part all
python verify_outputs.py --part all
```

The final command should report `{"status": "pass", "validated": 8}`.
All figures are exported as editable SVG/PDF, preview PNG, and 600-dpi TIFF
under `output/`. To keep the Git repository reasonably sized, generated TIFF
files are not committed; running `reproduce_all.py` recreates them locally.
`paper_figures.json` maps each figure to its generator and
Source Data; `output/reproduction_manifest.json` records hashes, dimensions,
DPI, and the software environment.

Individual parts can be reproduced with:

```powershell
python reproduce_all.py --part 1
python reproduce_all.py --part 2
python reproduce_all.py --part 3
```

## Model and Source Data entry points

```powershell
python part1_network/part1_model.py
python part2_rc_sde/part2_model.py --help
python part3_mfc/part3_model.py --help
```

The frozen figure inputs are committed, so figure reproduction does not
retrain a model or select a favorable trajectory. For a model-to-figure Part
III workflow, run `python part3_mfc/part3_model.py source-data` before the
three Part III figure scripts.

## Data

The original EEG archives are not included. To retrain from OpenNeuro
`ds004100` v1.1.3 (DOI: `10.18112/openneuro.ds004100.v1.1.3`), place the independently obtained subject ZIP files under
`data/ds004100/`, or edit `data.zip_root` in `config_v2.yaml`. That directory
is ignored by Git.

The original sampling rate is read from BIDS metadata. The analysis pipeline
then applies causal resampling to 256 Hz; consequently, a 1-s analysis window
contains 256 samples without implying that the original recording was 256 Hz.
See [DATA_NOTICE.md](DATA_NOTICE.md) for data and clinical-use limitations.

## Methodological scope

Part III implements a structured state-feedback Actor, a time-conditioned
WGAN-GP critic, and empirical-particle Fokker–Planck propagation on the frozen
Graph-RC-SDE. The current implementation does not train a separate HJB value
network and does not evaluate a Bellman/HJB PDE residual; claims should remain
consistent with that implementation.

Fig. 6/8 use the frozen main-result checkpoint. Fig. 7 uses separately
retrained, matched-budget full, no-WGAN, no-graph-spread, and no-deviation
checkpoints. The matched-ablation full checkpoint is not byte-identical to the
Fig. 6/8 main-result checkpoint.

## License

The software is released under the [MIT License](LICENSE). Dataset-derived
research artifacts remain subject to the upstream dataset terms described in
[DATA_NOTICE.md](DATA_NOTICE.md).

If you use this repository, see [CITATION.cff](CITATION.cff) for citation
metadata.
