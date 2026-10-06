# HUP080: current76/96 Full WGAN-GP

`runner.py` defaults to signal-free current-result verification. The adopted
controller starts from fresh selected update200 and chooses added125 of150 in
the `occupancy_worst_lr3e5` continuation. This is a post hoc amended reanalysis.

Current outer means: time W1 `0.32861935989151325→0.09385697244094479`;
occupation W1 `0.32035737631043043→0.07285532178847282`. Complete Gate-B57/96.
The old RMS cap0.405 failed0/24; revised0.45 passes24/24. Original trajectory
tables keep the old false budget flags. Reclassification is not the source of
the numerical control improvement or a clinical safety threshold.

Heat-kernel time0 removes instantaneous spreading to20 non-direct channels,
not the frozen surrogate's dynamic coupling. `results/` and `publication/` are
updated to this selected controller; no private signal arrays or weights are
redistributed. Explicit private verification uses `runner.py verify-private
--new-tag NEW --bundle-root COMPLETE_CURRENT_TREE`, or the same project's
`patient_results/HUP080_sparse_control_final_v2/current_full_wgangp` tree.
Screen+evaluation is not exposed as a default synchronization command.
