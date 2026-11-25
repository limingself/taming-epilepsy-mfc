# HUP065: Figure S4, development-window replay

The one-page, 64-contact red-manuscript figure is in
[`../publication/HUP065_run01_free_control.png`](../publication/HUP065_run01_free_control.png)
and [vector PDF](../publication/HUP065_run01_free_control.pdf). The PNG is
byte-identical to the red-manuscript figure; both were produced by the same
Matplotlib call. This is a **run01 development display**, not a run03 or
independent cross-seizure validation. The earlier frozen outer analysis in
the parent directory is a different experiment.

## Reproduce

Use the repository's numerical environment and a trusted, privately obtained
HUP065 development bundle. No external patient signal, trajectory ensemble,
density grid, weights, or optimization history is distributed here. The
examples are run from the repository root. Replace `PRIVATE` with your
authorized data directory and use distinct empty private output directories.
Only load known, trusted joblib and PyTorch files.

```bash
python patient_extensions/HUP065/development_s4/fit_predictor.py --arrays PRIVATE/development_arrays.npz --graph PRIVATE/fold1_model.joblib --out PRIVATE/refit/no_graph_fold1.joblib
python patient_extensions/HUP065/development_s4/train_policy.py --arrays PRIVATE/development_arrays.npz --graph PRIVATE/fold1_model.joblib --predictor PRIVATE/refit/no_graph_fold1.joblib --annotations PRIVATE/annotations.csv --out PRIVATE/retraining
python patient_extensions/HUP065/development_s4/plot_all_contacts.py PRIVATE/retraining/display_control.npz --out PRIVATE/rendered --metrics-dir PRIVATE/plot_metrics
```

- **Fast figure replay**: the third command also accepts the archived
  `display_control.npz` directly, with no training or checkpoint required.
- **Fast control replay**: pass `--checkpoint PRIVATE/trained_frozen_policy.pt`
  to the second command, using the exact frozen predictor that checkpoint
  identifies. Optional `--anchor PRIVATE/display_raw.npz` checks the free and
  observed arrays against the independent prediction run. If the private
  predictor has a different serialization, retrain with it instead of bypassing
  the frozen-checkpoint digest check.
- Input NPZ: `ictal_run_01`, `ictal_run_02`, `preictal_reference_run_01`,
  `preictal_reference_run_02`, `channels`, `sfreq`. The CSV has `channel`
  and `direct_paper_mask` columns in recording order. Use the original
  32-contact mask. Run03 is neither needed nor accepted.

## What was fitted, and what is shown?

Part II fits the same state-dependent reservoir recurrence on run02 with zero
explicit graph input, retaining the 35-component PCA transform, 160-unit
reservoir, and delays of 1, 8, and 32 samples. Part III independently uses the
fold-1 graph for input diffusion (heat time 0.2); the noise multiplier is
0.79451175 and the input bound is 1.8. The HUP060 mean–deviation actor and
WGAN-GP critic are initialized afresh. Run02 supplies the training contexts,
reference and validation segments; 16 actor updates and 72 critic updates
were run, and validation selected actor update 16.

Figure S4 starts at run01 sample 5150 with 256 past samples, 32 paired-noise
particles, 256 forecast steps, and 15 nonoverlapping reference segments. Each
panel compares recorded ictal data, free prediction, preictal reference and
closed-loop control. The horizontal axis is a **common fixed display window**:
retain standardized amplitudes in [−6.309818911354839, 5.201366495203709],
then map them linearly to [−1, 1]. Each contact and series is normalized to
unit mass **within that window**. A shared contact-specific bandwidth is
estimated from the displayed recorded ictal samples. For interpretation,
work with sample-based distances, not distances between the smoothed curves.

Mean within-window occupation W1 decreases from 0.101264 to 0.070818; 59/64
contacts improve (30/32 direct; 29/32 indirect). These are conditional
distances, not full-range forecast fidelity. Full-range distances and input
diagnostics are computed separately by the same script with `--metrics-dir`;
the free rollout has large tails. The figure should not be used as evidence
of full-range agreement.

## Identification

SHA-256 of the private originals (digests identify files; they grant no access):

| Input | SHA-256 |
|---|---|
| Development arrays | `a3b28c87987af9209f58970ed0ac1de5767574f24a154b3f5ae94accb429026f` |
| Fold-1 graph | `a55366d14a55e89ef4993cd46f5381e4468997e9d7c8535fe51709cb37542ed4` |
| Fold-1 frozen predictor | `03d1adec1a6859b292bf43e7840f88d143a16705031b56ed4be1c0e7e7503b20` |
| Frozen controller | `46b972f117aab0dc65796a6f1deefd79b4f62d873d1088eaa9e1d2090703b638` |
| Figure input bundle | `659002a731a2277b9380611fbb8a79f0281ef584621ad08b8582667a79a6dfb8` |
| Approved one-page PNG | `c53e88b88fb126829ca46c91c3285fcc5ff8a513473c8a82c2a14a737ac39bcc` |
