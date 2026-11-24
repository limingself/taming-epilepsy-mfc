# Tracking MPC on the frozen HUP060 surrogate (Physica D red manuscript, Fig. 7)

This baseline asks whether tracking a reference **mean** also reproduces the
distribution of recorded amplitudes. It is a separate policy comparison, not a
component removal or a retraining of Full. The HUP060 run02 development window
has 256 steps at 256 Hz, 32 particles, 36 contacts and 13 graph-selected inputs.
Full and MPC use the same model, starting state, noise draws, actuator map and
dimensionless command bound (1.8); their realized command RMS is not matched.

## Reproduction levels

Run the following commands from the repository root. No command writes patient
trajectories or checkpoints. Summary statistics go to standard output.

1. **Public, no patient inputs:** run the synthetic tests:

   ```sh
   python -B part3_mfc/test_tracking_mpc.py
   ```

   The scalar test checks feasibility, non-increasing MPC objective and
   non-anticipation of future noise. A separate test distinguishes the two
   Wasserstein endpoints by reversing the time order of an artificial series.
   The public figure renderer uses checked-in numerical summaries; these tests
   do not reproduce the patient's trajectories.

2. **Authorized paired trajectory bank:** recompute the manuscript endpoints
   from trajectories (no checkpoint required):

   ```sh
   python -B part3_mfc/tracking_mpc_metrics.py /path/to/primary_rollouts.npz
   ```

   The NPZ has keys `free`, `full`, `mpc` (32 x 256 x 36), `full_u`,
   `mpc_u` (32 x 256 x 13) and `ref` (15 x 256 x 36). The code computes
   empirical time-resolved W1 at each contact and time before averaging;
   occupation W1 pools time and paths **within** each contact. Both endpoints
   are then averaged over contacts. Location and width errors are absolute
   differences of pooled means and population SDs, **median** across contacts.
   Strict joint improvement counts contacts below Free on both endpoints.

3. **Authorized frozen-model workspace:** rerun the actual policies and
   optionally verify the archived arrays point by point:

   ```sh
   python -B part3_mfc/tracking_mpc.py --project-root /path/to/complete/project --check-against /path/to/primary_rollouts.npz
   ```

   The project must contain the existing Part II/III source and frozen
   reproduction inputs expected by `part3_mfc/train_full_wgangp.py`:
   the synthesis-only context/mask/reference contract, serialized stochastic
   plant, the published paired no-control rollout, and the Full checkpoint from
   `current_full_wgangp/runs/fresh_seed20261011_e1000` (update 700). Replay
   checks the checkpoint SHA-256, plant fingerprint and evaluation mask/reference,
   and reconstructs the original primary noise bank from the frozen Free rollout.
   Use trusted local model bundles only. No raw iEEG, standardized reference
   segments, trajectory bank or model weights are distributed with these scripts.
   Omitting `--project-root` uses the repository containing the script; omitting
   `--check-against` skips the optional external trajectory comparison.

## Solver

At every step, each particle plans its own four-step input sequence on the
existing nonlinear Part III stepper, with future innovations set to zero **only
inside the optimizer**. MPC tracks the time-dependent mean of the first 15 run01
fitting-reference segments, using the existing fitting-reference contact weights.
The other 15 segments are used for evaluation, not as a target available to MPC.
The objective is weighted output mean-squared error plus 0.01 times mean squared
normalized command and 0.02 times mean squared normalized command change. Each
command and command change is normalized by 1.8. The first change uses the
last applied input. There is no terminal penalty.

The solver uses nonlinear autograd gradients, an initial-state finite-difference
Gauss–Newton curvature approximation, three outer updates and up to 40 projected
accelerated iterations per outer update, with per-particle backtracking. It
applies the first input, advances the **stochastic** plant with the paired
innovation, shifts the plan, and replans. The horizon shortens at the end of the
window. There is no new plant, hindsight input, patient-specific refit or claim
that the numerical optimizer attains a global minimum.

At manuscript precision both Full and MPC have mean occupation W1 of 0.079.
Full's median width error is 0.036 versus 0.095 for MPC, and Full is closer in
width at 26/36 contacts. Both W1 endpoints improve relative to Free at 36/36
contacts for Full and 34/36 for MPC. Command RMS is 0.673 versus 0.518.
