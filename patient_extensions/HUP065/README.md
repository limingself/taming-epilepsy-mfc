# HUP065: current manuscript Fig. S4

[All 64 controlled contact distributions (PDF)](../single_window_replay/publication/HUP065/HUP065_run02_full_all64.pdf) · [shared replay code](../single_window_replay/run.py) · [training and evaluation instructions](../single_window_replay/README.md).
The current S4 uses the HUP060 workflow: fit dynamics on run-01 and assess control in the preselected one-second run-02 window.

## Archive: multi-context Full WGAN-GP results

The patient-specific `runner.py`, tables and figures below reproduce this historical experiment, not the current S4.

`runner.py` defaults to signal-free current-result verification. The adopted
mask is PartI's weighted ranking top32. A neutral23-node fresh parent was
selected at update550; nine outputs were added at mean/deviation gains0.05/0,
and the150-update continuation selected added0. This is not a new32-node cold
training success. The later neutral32/1000-update trial was not adopted.

Current outer means: time W1 `0.27179461046098846→0.20086363305685798`;
occupation W1 `0.1795093774904715→0.12694506110397122`. Complete Gate-B21/64.
The analysis is a post hoc amended reanalysis. RMS cap0.405 is unchanged.

`results/` and `publication/` now contain this adopted version. Current control
and plotting source lives here; no external arrays/weights/density tables/logs
are public. Explicit private verification uses
`runner.py verify-private --new-tag NEW --bundle-root COMPLETE_CURRENT_TREE`.
Default local tree is the same project's
`patient_results/HUP065_sparse_control_final_v1/current_full_wgangp`.
Public default checks never open context6 or the outer seizure.
