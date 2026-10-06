# Current HUP065/HUP080 Full WGAN-GP extensions

The existing patient paths are updated in place. Default commands verify the
current unrounded channel/context/bank tables and approved publication figures:

```powershell
python patient_extensions/verify_public_results.py
python patient_extensions/HUP065/runner.py
python patient_extensions/HUP080/runner.py
```

HUP065 uses32/64 weighted selected direct nodes, the23-node fresh parent and
low-gain extension selected at added0. HUP080 uses76/96 nodes, fresh parent200
and selected continuation125. Both current analyses are post hoc amendments,
not newly pristine held-out tests. Gate-B covers21/64 and57/96 channels under
the all24-condition rule. Those24 conditions are8 contexts×3 paired banks,
not24patients.

The original public arrays/weights exclusion remains: no signal `.npy/.npz`,
external `.pt/.joblib`, raw ZIP/EDF, large density CSVs, training histories or
access receipts are added. To verify a complete trusted local current bundle,
use `runner.py verify-private --new-tag NEW` with an optional `--bundle-root`.
Without that argument the current tree under the same original project's
`patient_results/…/current_full_wgangp` is used only for explicitly requested
private actions. Missing current inputs fail; there is no old-controller fallback.

Executed current fresh/screen/plot sources are now in the original `common/`
and patient directories. Kernel equations and scientific results are not
modified. Legacy preprocessing/prediction helpers remain for frozen input
provenance, but their earlier control defaults are not current runner defaults.
The `current_full_wgangp` block in each existing template is authoritative for
the control stage; pre-existing PartI/II preparation values are retained.
Historical freezes/tags/releases remain separate provenance, not default inputs.
