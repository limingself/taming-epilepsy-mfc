# Physica D Figure 7 and Supplementary Figure S10

These are figure-level summaries for the HUP060 run-02 development case. They
cover the same one-second window (context 3), 36 recorded contacts and 13
actuators as the manuscript. The primary evaluation uses 32 model paths, 256
time steps and 15 preictal reference segments. Contacts and paths are not
independent patients.

## Reproduce or inspect the figures

From the repository root, with NumPy, pandas and Matplotlib installed:

```sh
python part3_mfc/plot_physicad_controller_baselines.py --check-only
python part3_mfc/plot_physicad_controller_baselines.py --outdir ./reproduced_figures
```

Omit `--outdir` to write to `output/part3/figure_09/` (the historical export
ID). Existing exports are not overwritten; use a fresh directory for a redraw.
The files are `HUP060_controller_baselines_mpc` (Fig. 7) and
`HUP060_noise_and_mismatch` (Fig. S10), each as PDF, PNG and editable SVG.
The check command prints values calculated from these tables, not values
transcribed from the manuscript. It checks that both width-related tables
cover the same 36 contacts, that the initial decomposition sums to its total
for every contact, that both sensitivity sweeps recover the no-control
baseline, and that the reported matched damping point lies on the gain sweep.
It also recalculates Table 1's Free/Full/MPC mean W1 distances and medians
of the absolute pooled mean and SD gaps from `primary_contact_endpoints.csv`.
This sixth table contains only the four distances/errors per arm and
contact, not any trajectories, mean amplitudes or patient samples. A
contact counts as improved only when **both** W1 distances are strictly
smaller than for Free at that same contact (Full: 36/36; tracking MPC:
34/36). This criterion is distinct from the Full-versus-MPC width comparison
in panel 7b.

## Map from panels to data

| Panel | Source table | Quantity plotted |
| --- | --- | --- |
| 7a | `controller_endpoints.csv` | Mean contact-wise Wasserstein-1 distance to reference. Time-resolved distances are evaluated at corresponding times; occupation distances pool across time. Constant and White are medians across 32 sign-paired input outcomes **after** averaging across contacts; the other four arms are fixed evaluations. |
| 7b | `pooled_width_errors.csv` | Each point pairs the Full and tracking-MPC absolute errors in pooled standard deviation at one contact. SD uses all times and paths with a population denominator. The display gives the median of the 36 errors for each controller. |
| 7c | `damping_scan.csv`, `controller_endpoints.csv` | Residual occupation distance versus actual actuator-command RMS. The highlighted damping comparison matches Full's command RMS approximately; the best gain is the post hoc minimum in this scan. |
| S10a | `diffusion_scan.csv` | No-control occupation distances to the preictal reference and to the recorded seizure, as the innovation multiplier varies. This changes the surrogate, not the controller. |
| S10b | `initial_mismatch.csv` | Location, scale and shape contributions to the original Free-to-reference mismatch; reference/ictal and reference/Free SD ratios by contact. Segments show medians and interquartile ranges. |

Full and tracking MPC share a frozen model, input locations, initial state,
evaluation innovations and per-command amplitude bound (1.8); they do not
share realized command RMS. The occupation distances both round to 0.079.
Full is closer in width at 26/36 contacts (median absolute width error 0.036
versus 0.095). Tracking MPC uses less command RMS (0.518 versus 0.673).
The 96.1% scale share in S10 describes the **initial discrepancy**, not the
fraction of improvement attributable to treatment.

These CSVs contain contact-level or controller-level statistics, not iEEG
samples, reference segments, simulation trajectories or checkpoints. The
neighboring historical tables retain the original damping and diffusion scans,
location/scale decomposition and generic-input outcomes. Replotting the
summaries is distinct from retraining a controller or replaying the paired
trajectories.
