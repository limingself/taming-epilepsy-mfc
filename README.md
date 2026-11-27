# Taming Epilepsy: Mean Field Control of Whole-Brain Dynamics

The existing project paths now contain the current Full WGAN-GP code and
adopted results. HUP060 Part I and its frozen Graph–RC prediction model are unchanged.
The revised HUP060 forecasting comparison (manuscript Fig. 3) uses four
nested predictors: current state, added delays, added graph features, and
finally the full reservoir. The first three predictors have no reservoir.
See the [Part II reproduction instructions](part2_rc_sde/README.md);
the historical figure filenames are retained. Do not run a separate release
folder for the current workflow.

## Current Physica D figures, in manuscript order

The [HUP065 and HUP080 run-02 replay](patient_extensions/single_window_replay/README.md) uses run-01 to fit each patient's Graph–RC dynamics and select the noise and input scales. Both cases follow the HUP060 preprocessing, mean-field control and evaluation procedure. A fixed one-second run-02 window is shown for all 64 or 96 contacts, with 32 or 76 directly actuated, respectively. The history before the forecast boundary is available to the controller; the recorded future is used only for evaluation. Controlled trajectories are simulations, not recorded responses to stimulation.

Read the cases in this order: **HUP060 → HUP065 (Fig. S4) → HUP080 (Fig. S5)**. For HUP065 and HUP080 the manuscript shows only the all-contact controlled amplitude distributions, not additional ablations. Each percentage below compares the controlled and uncontrolled simulations against the same preictal reference in the registered one-second window.

| Result | Reproduction entry | Scope |
|---|---|---|
| HUP060 Fig. 3 | [Nested forecasting ablation](part2_rc_sde/figure_05_input_ablation.py) | Four nested predictors; unchanged from the verified Nov 24 figure. |
| HUP060 Fig. 7 | [Six-panel controller comparison](part3_mfc/figure_09_trivial_baselines.py) · [inputs and commands](part3_mfc/TRACKING_MPC.md) | All original comparisons plus tracking MPC; the previous split export is historical. |
| HUP065 Fig. S4 | [PDF](patient_extensions/single_window_replay/publication/HUP065/HUP065_run02_full_all64.pdf) · [PNG](patient_extensions/single_window_replay/publication/HUP065/HUP065_run02_full_all64.png) · [per-contact metrics](patient_extensions/single_window_replay/publication/HUP065/contact_metrics.csv) | 64 contacts, 32 direct inputs. Mean time-resolved W1 fell 42.4% (64/64 contacts improved); mean occupation W1 fell 61.6% (54/64). |
| HUP080 Fig. S5 | [PDF](patient_extensions/single_window_replay/publication/HUP080/HUP080_run02_full_all96.pdf) · [PNG](patient_extensions/single_window_replay/publication/HUP080/HUP080_run02_full_all96.png) · [per-contact metrics](patient_extensions/single_window_replay/publication/HUP080/contact_metrics.csv) | 96 contacts, 76 direct inputs. Mean time-resolved W1 fell 68.6% and mean occupation W1 fell 88.4%; both improved at 96/96 contacts. |

Stable pipeline slot 05 corresponds to manuscript Fig. 3, slot 07 to Fig. 6, and slot 09 to Fig. 7. Fig. 7 compares methods on the same frozen HUP060 model and draw bank; realized input energy is an outcome, not a matched quantity.

Full combines the frozen patient-specific Graph–RC dynamics, structured
mean–deviation feedback, the existing empirical-law losses and WGAN-GP from
the first fresh actor update. It is not a pure GAN replacement of the plant.

HUP060 uses 13/36 inputs and the Full checkpoint at update 700. The earlier [HUP065 development display](patient_extensions/HUP065/development_s4/README.md), [HUP080 development display and Table S17](patient_extensions/HUP080/FIGURE_S5.md), and run-03/run-04 evaluations remain available as provenance. They do not supply results for the current S4/S5. Model inputs are in standardized model units.

## Current commands

```powershell
python verify_outputs.py --part 3
python patient_extensions/single_window_replay/verify_publication.py
python reproduce_all.py --part 3
python part3_mfc/part3_model.py source-data
python part3_mfc/part3_model.py train --help
python patient_extensions/single_window_replay/prepare_inputs.py --help
python patient_extensions/single_window_replay/run.py --help
```

Public Part-III defaults verify the current frozen publication exports and
unrounded Source Data without reading patient arrays, loading weights or
re-evaluating seizures. They do not pretend to rerun training. HUP060 curves,
component/baseline tables and loss Source Data are in the original
`output/part3/source_data/` locations. Optional CSV-only redraw uses a new
output directory, for example:

```powershell
python part3_mfc/figure_08_all36_controlled_distributions.py --redraw --output-dir ./new_public_redraw
python part3_mfc/figure_09_trivial_baselines.py --redraw --output-dir ./fig7_complete_redraw
```

`part3_mfc/part3_model.py` is the current command entry;
`train_full_wgangp.py` is the executed neutral-training implementation;
`model_core.py` is its unchanged scientific kernel, not an alternative training
default. HUP060 control, ablation, RMS-matched baselines and loss records all
identify the same Full update700 checkpoint. See `PAPER_FINAL_VERSION.json`
and `paper_figures.json` for the HUP060 release. The
[single-window replay guide](patient_extensions/single_window_replay/README.md)
is the entry point for the current HUP065 and HUP080 results.

Frozen-checkpoint replay of the earlier experiments needs separately supplied
inputs and weights. These assets occupy the existing `artifacts/part3_hup060_actor_wgan_v1/` and
`patient_results/` project trees. Public commands fail clearly if these are
absent, never falling back to the historical teacher/controller. Actual
training requires an explicit new tag; evaluation additionally requires an
explicit evaluation decision. For the new S4/S5, the shared command above
instead fits and trains from separately obtained recordings; see the replay guide.

Parts I/II retain their original reproduction commands:
`python reproduce_all.py --part 1` or `--part 2`. The updated prediction
ablation compares uncontrolled forecasts to recorded ictal activity; the
full predictor and the separately reported controller are unchanged.

Historical tags and `releases/full_wgangp_20261006` remain recoverable provenance,
not the current default project. No original EEG archives or new external
arrays/serialized models/full private histories/access receipts are redistributed.
HUP060 loss-figure scalar Source Data includes the plotted raw terms and their
derived rolling median, distinct from the external private training bundles.
Obtain recordings independently from OpenNeuro ds004100 v1.1.3, DOI
`10.18112/openneuro.ds004100.v1.1.3`. See `DATA_NOTICE.md` and the MIT `LICENSE`.
