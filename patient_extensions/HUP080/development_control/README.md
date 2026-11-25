# HUP080: development predictor and controller for Fig. S5 / Table S17

This is the code path for the November 25 red manuscript. It uses only development recordings 01–03. The earlier run04 controller, evaluation and hashes belong to a different experiment and are left intact.

Start at the project root with Python 3.11+ and the project's scientific dependencies. Use only trusted local joblib/PyTorch inputs. The private artifact directory must supply the original development arrays, fold-specific model/config files and fold-specific input-selection graphs. The code checks fold membership, channel order and input hashes. Use a **new** private work directory; intermediate fits, per-window simulations and checkpoints must not be committed. In PowerShell:

```powershell
# Optional overrides (defaults are under this project's ignored patient_results/).
# $env:HUP080_DEV_ARTIFACTS = "<original science_run/artifacts>"
# $env:HUP080_DEV_WORK = "<new private run directory>"
$src = "patient_extensions/HUP080/development_control"
python "$src/predict_free.py"
python "$src/adaptive_run.py"
foreach ($fold in 1..3) { python "$src/train_control.py" --fold $fold; if ($LASTEXITCODE) { throw "Initial training failed" } }
foreach ($fold in 1..3) { python "$src/refine.py" --fold $fold; if ($LASTEXITCODE) { throw "Refinement failed" } }
foreach ($fold in 1..3) { python "$src/finalize.py" --fold $fold; if ($LASTEXITCODE) { throw "Selection failed" } }
foreach ($fold in 1..3) { python "$src/finalize.py" --fold $fold --evaluate; if ($LASTEXITCODE) { throw "Frozen evaluation failed" } }
python "$src/check_additional.py" --all --summary
```

The predictor is refitted on the other two recordings in each fold. The scale pilot sees only pre-origin data; all methods at an origin use paired evaluation draws. The graph-derived 76-contact actuator mask is separate from the predictor's feature set. The 32-update controller uses the project's structured mean–deviation actor and WGAN-GP objective. Refinement checks actuator-space directions and additional updates on training-fold validation data. The final policies are frozen for all three folds before the final comparison. The 33 windows and two innovation banks per window are not 66 independent seizures.

The single figure uses the private `<work>/refined/engine/display_control.npz`. For the figure CLI and the public Table S17 inputs, see [Fig. S5](../FIGURE_S5.md). The additional-window evaluation is under `<work>/refined/additional/`. `HUP080_HISTORICAL_AUDIT_CSV`, if set, enables an optional numerical cross-check against an older prediction audit; it is not a training input and does not change the estimates.
