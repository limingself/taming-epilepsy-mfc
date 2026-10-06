# Part III: current Full WGAN-GP controller

Use the existing `part3_model.py` entry. `source-data` and `verify` inspect the
current saved tables and approved exports only. `train --help` does not open
signals. The scientific kernel `model_core.py` retains SHA
`57da573dbe300ad6bd585fbd69e46102d9f315dba8862e63c7d3e8320b6943e1`;
only path bindings and command dispatch differ from the executed runner.

HUP060 Full starts neutral, trains1,000 actor updates, and selects update700
using development-only validation. It has13 dedicated outputs among36 channels.
The same selected Full is used in control, component-removal, RMS-matched
baseline and actual-loss figures. Component-removal arms are separate matched
1,000-update runs; their input RMS is not forced equal.

| Existing command | Current data/figure |
|---|---|
| `figure_06_actor_wgan_control.py` | Original3×3 representative Full700 |
| `figure_07_mfc_ablation.py` | Same-Full component removals |
| `figure_08_all36_controlled_distributions.py` | Original6×6 four-law Full700 |
| `figure_09_trivial_baselines.py` | RMS-matched Constant/White/damping and diffusion sensitivity |
| `figure_10_training_losses.py` | Actual1000-update raw loss and trailing125-update median |

Defaults verify the exact adopted exports, not secretly redraw from a legacy
actor. `--redraw --output-dir NEW_EMPTY_DIRECTORY` uses permitted public CSVs
for6/7/8/9; it cannot overwrite adopted outputs implicitly. Loss's default
verifies the current raw/median source and figure. Its executed eight-panel
redraw uses `plot_loss_approved.py --run-dir PRIVATE_CURRENT_RUN --output-dir NEW`
with separately supplied real training history, not fabricated traces.

Private current run inputs/results live at
`artifacts/part3_hup060_actor_wgan_v1/current_full_wgangp/`; comparisons are in
the sibling `current_comparisons/`. The untouched PartII/free-only saved source
remains separate. `train_full_wgangp.py` and `run_component_ablation.py` bind
the current original-project paths and require explicit new tags. The public
repository does not supply a complete new serialized-input training bundle.

```powershell
python part3_mfc/part3_model.py train --new-tag NEW_UNIQUE_RUN --epochs 1000
```

Do not reuse the adopted tag. Evaluation is disabled unless explicitly called
with `--allow-evaluation`; component evaluation also requires `--ablation`.
No synchronization command retrains or opens a terminal/outer evaluation.
Original kernel helper definitions are retained for import compatibility;
historical teacher/40-update functions are not the current command dispatcher.
