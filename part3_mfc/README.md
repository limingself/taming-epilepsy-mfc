# Part III: Actor-WGAN mean-field distribution control

This directory exposes one Part-III model command and one plotting script for
each manuscript figure.

| File | Responsibility | Output |
|---|---|---|
| `part3_model.py` | Actor-WGAN training, frozen evaluation, matched-ablation evaluation, and Source Data export | checkpoints and evaluation tables |
| `figure_06_actor_wgan_control.py` | representative trajectories, occupation laws, and control inputs | Fig. 6 |
| `figure_07_mfc_ablation.py` | matched WGAN, graph-spread, and deviation-feedback ablations | Fig. 7 |
| `figure_08_all36_controlled_distributions.py` | all-36-channel distribution audit | Fig. 8 |

## Commands

```powershell
python part3_mfc/part3_model.py train --help
python part3_mfc/part3_model.py evaluate --help
python part3_mfc/part3_model.py evaluate-ablations --help
python part3_mfc/part3_model.py source-data
```

Training uses run-01-only development inputs and cannot read the run-02 future.
Evaluation opens the frozen run-02 development window only after the policy is
fixed.  The Part-III uncontrolled branch is the exact same saved `u=0`
Graph-RC-SDE particle law reported in Part II.

Generate the three figures independently with:

```powershell
python part3_mfc/figure_06_actor_wgan_control.py
python part3_mfc/figure_07_mfc_ablation.py
python part3_mfc/figure_08_all36_controlled_distributions.py
```

Figs. 6 and 8 use the primary `seed20261011_adv050_anchor020` checkpoint.
Fig. 7 uses separately retrained matched-budget full, no-WGAN,
no-graph-spread, and no-deviation checkpoints.  Its full checkpoint has the
same architecture but is not byte-identical to the Figs. 6/8 checkpoint.

The implementation is a structured state-feedback Actor, a time-conditioned
WGAN-GP critic, and empirical-particle Fokker--Planck propagation on the frozen
Graph-RC SDE.  It does not train a separate HJB value network or evaluate an
HJB/Bellman PDE residual; claims should remain within that implemented scope.
