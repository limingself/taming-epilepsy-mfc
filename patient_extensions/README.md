# HUP065 and HUP080: current manuscript replays

The manuscript presents HUP060 first, followed by **HUP065 (S4)** and
**HUP080 (S5)**. The last two cases show the all-contact controlled amplitude
distributions. Both follow the HUP060 data processing, model and mean-field
control procedure: fit each patient's dynamics on run-01, calibrate noise and
input scales on run-01, and evaluate paired free/controlled simulations in a
preselected one-second window from run-02. See the
[shared training and replay instructions](single_window_replay/README.md).

| Figure | Controlled distributions | Per-contact metrics |
|---|---|---|
| S4, HUP065 (64 contacts; 32 direct inputs) | [PDF](single_window_replay/publication/HUP065/HUP065_run02_full_all64.pdf) · [PNG](single_window_replay/publication/HUP065/HUP065_run02_full_all64.png) | [CSV](single_window_replay/publication/HUP065/contact_metrics.csv) |
| S5, HUP080 (96 contacts; 76 direct inputs) | [PDF](single_window_replay/publication/HUP080/HUP080_run02_full_all96.pdf) · [PNG](single_window_replay/publication/HUP080/HUP080_run02_full_all96.png) | [CSV](single_window_replay/publication/HUP080/contact_metrics.csv) |

Verify the released figures and metrics from the repository root without patient recordings:

```powershell
python patient_extensions/single_window_replay/verify_publication.py
```

## Historical multi-context experiments

The earlier development displays and run-03/run-04 evaluations are separate
experiments, not sources for S4/S5 or ablations of these single-window results.
For users reproducing the older channel/context/bank tables and figures:

```powershell
python patient_extensions/verify_public_results.py --archive
python patient_extensions/HUP065/runner.py
python patient_extensions/HUP080/runner.py
```

The historical HUP065 experiment used 32/64 weighted selected direct nodes;
the HUP080 experiment used 76/96. Their 24-condition evaluations represent
eight contexts and three paired banks, not separate patients.

The original public arrays/weights exclusion remains: no signal `.npy/.npz`,
external `.pt/.joblib`, raw ZIP/EDF, large density CSVs, training histories or
access receipts are added. To verify a complete trusted historical bundle,
use `runner.py verify-private --new-tag NEW` with an optional `--bundle-root`.
Without that argument the current tree under the same original project's
`patient_results/…/current_full_wgangp` is used only for explicitly requested
private actions. Missing current inputs fail; there is no old-controller fallback.

The original `common/` and patient directories retain their fresh/screen/plot
sources and original runner conventions. Their `current_full_wgangp` template
blocks refer to the corresponding archived experiments, not to the run-02
single-window replays above.
