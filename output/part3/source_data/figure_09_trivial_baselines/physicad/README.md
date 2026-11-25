# Physica D Figure 7: six-panel baseline and mismatch audit

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
The single current figure is `HUP060_controller_baselines_mpc_complete` (PDF,
PNG and editable SVG). The program also writes a panel-a CSV and QA JSON with
hashes of the eight actual input CSVs.
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
in panel 7e.

## Map from panels to data

| Panel | Source table | Quantity plotted |
| --- | --- | --- |
| 7a | `controller_endpoints.csv`, `../trivial_baseline_summary.csv`, `../random_input_seed_metrics.csv` | Time-resolved and time-pooled (occupation) W1 to the reference for seven conditions. The Noise 0.2 bar represents a post hoc no-input sensitivity setting. Constant and White use the median of 32 sign-paired outcomes after averaging across contacts. |
| 7b | `damping_scan.csv`, `controller_endpoints.csv` | Occupation W1 versus realized input RMS over 27 proportional damping gains, with matched, best and Full comparisons. |
| 7c | `diffusion_scan.csv` | No-control occupation distance to the preictal reference and to the seizure recording for 13 innovation multipliers. This varies the model noise, not the control law. |
| 7d | `initial_mismatch.csv` | Location, scale and residual-shape contributions to the initial Free-to-reference distance, plus reference/recorded and reference/free SD ratios by contact. |
| 7e | `pooled_width_errors.csv`, cross-checked against `primary_contact_endpoints.csv` | Pooled-SD absolute error for tracking MPC versus Full, paired across 36 contacts. SD uses all simulation times and paths with a population denominator. |
| 7f | `damping_scan.csv`, `controller_endpoints.csv` | Occupation W1 versus realized actuator-command RMS for Full, tracking MPC and proportional damping. |

Full and tracking MPC share a frozen model, input locations, initial state,
evaluation innovations and per-command amplitude bound (1.8); they do not
share realized command RMS. The occupation distances both round to 0.079.
Full is closer in width at 26/36 contacts (median absolute width error 0.036
versus 0.095). Tracking MPC uses less command RMS (0.518 versus 0.673).
The 96.1% scale share in panel 7d describes the **initial discrepancy**, not the
fraction of improvement attributable to treatment.

These CSVs contain contact-level or controller-level statistics, not iEEG
samples, reference segments, simulation trajectories or checkpoints. The
neighboring historical tables retain the original damping and diffusion scans,
location/scale decomposition and generic-input outcomes. Replotting the
summaries is distinct from retraining a controller or replaying the paired
trajectories.
