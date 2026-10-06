# Taming Epilepsy: Mean Field Control of Whole-Brain Dynamics

The existing project paths now contain the current Full WGAN-GP code and
adopted results. Parts I/II and the frozen Graph–RC prediction model are
unchanged. Do not run a separate release folder for the current workflow.

Full combines the frozen patient-specific Graph–RC dynamics, structured
mean–deviation feedback, the existing empirical-law losses and WGAN-GP from
the first fresh actor update. It is not a pure GAN replacement of the plant.

| Patient | Direct inputs | Current controller |
|---|---:|---|
| HUP060 | 13/36 | Neutral initialization, 1,000 actor updates, selected update700 |
| HUP065 | 32/64 | Fresh23-node parent, weighted top32 low-gain expansion, selected added0 of150 |
| HUP080 | 76/96 | Fresh parent selected200, continuation selected added125 of150 |

HUP060 is a development/adaptation case. The current65/80 results are post hoc
amended reanalyses. HUP080's original RMS cap0.405 failed all24 conditions;
the revised resource cap0.45 passes24/24, with the original failures retained.
Model inputs have no physical stimulation units. The separate cold32 HUP065
trial was not adopted because it failed the two-endpoint development comparison.

## Current commands

```powershell
python verify_outputs.py --part 3
python patient_extensions/verify_public_results.py
python reproduce_all.py --part 3
python part3_mfc/part3_model.py source-data
python part3_mfc/part3_model.py train --help
python patient_extensions/HUP065/runner.py --help
python patient_extensions/HUP080/runner.py --help
```

Public Part-III defaults verify the current frozen publication exports and
unrounded Source Data without reading patient arrays, loading weights or
re-evaluating seizures. They do not pretend to rerun training. HUP060 curves,
component/baseline tables and loss Source Data are in the original
`output/part3/source_data/` locations. Optional CSV-only redraw uses a new
output directory, for example:

```powershell
python part3_mfc/figure_08_all36_controlled_distributions.py --redraw --output-dir ./new_public_redraw
```

`part3_mfc/part3_model.py` is the current command entry;
`train_full_wgangp.py` is the executed neutral-training implementation;
`model_core.py` is its unchanged scientific kernel, not an alternative training
default. Control, ablation, RMS-matched baselines and actual loss records all
identify the same Full update700 checkpoint. The current contract is
`PAPER_FINAL_VERSION.json`; `CURRENT_PROJECT_MANIFEST.json` binds current code,
public metrics and approved figures. `paper_figures.json` maps10 HUP060 outputs;
65/80 pages remain under their original patient publication directories.

Private exact replay needs separately supplied frozen inputs/weights. Current
local assets occupy the existing `artifacts/part3_hup060_actor_wgan_v1/` and
`patient_results/` project trees. Public commands fail clearly if these are
absent, never falling back to the historical teacher/controller. Actual
training requires an explicit new tag; evaluation additionally requires an
explicit evaluation decision. See the PartIII and patient READMEs.

Parts I/II retain their original code and reproduction commands:
`python reproduce_all.py --part 1` or `--part 2`. PartII's unchanged free-only
provenance can retain its historical saved zero-input source: those unchanged
free samples are not the newly selected controlled branch.

Historical tags and `releases/full_wgangp_20261006` remain recoverable provenance,
not the current default project. No original EEG archives or new external
arrays/serialized models/full private histories/access receipts are redistributed.
HUP060 loss-figure scalar Source Data includes the plotted raw terms and their
derived rolling median, distinct from the external private training bundles.
Obtain recordings independently from OpenNeuro ds004100 v1.1.3, DOI
`10.18112/openneuro.ds004100.v1.1.3`. See `DATA_NOTICE.md` and the MIT `LICENSE`.
