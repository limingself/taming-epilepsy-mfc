# HUP080 exploratory partial-actuation extension

The retained `sparse-rerun` schema identifier and authorization environment
variable are historical machine-readable identifiers from the executed run;
they do not change this extension's exploratory partial-actuation classification.

The original v1 development rolling screen tested blocks 4, 8, 12, 16, 24,
and 32 and produced a valid `NO_GO_before_part3`. The exploratory v2 amendment
added only blocks 1 and 2; thresholds and the every-fold LORO rule were not
changed. It must not be described as the original preregistered replication.
The bundled parent-v1 receipt records only the failed-stage artifact names,
sizes and SHA-256 values. The private stage is not required for public
verification, and no v2 science phase reads the receipt as a runtime dependency.

Development run-01 through run-03 selected a 2-sample rolling block and the
frozen candidate `f-0.80_q-0.20_gain-0.350_tau-0.000`, giving 76 direct and 20
indirect channels. A one-time retrospective run-04 evaluation passed complete
Gate-B on 65/96 channels, the contact-level safety vector on 96/96 channels,
and Gate-C on all 24 predeclared trajectories.

At `tau_G=0`, the random-walk heat kernel is the identity. The 20 indirect
channels therefore receive no instantaneous heat-kernel control input; any
change in them arises only through the frozen plant dynamics. The code path is
the `selector @ heat_kernel` construction in `mfc_pipeline/square_wave_mfc.py`.

To materialize a relocation-specific configuration after independently
obtaining OpenNeuro `ds004100`:

```powershell
python patient_extensions/materialize_config.py --patient HUP080 --dataset-root <dataset-root>
python patient_extensions/verify_public_results.py
python -B patient_extensions/HUP080/runner.py status
python -B patient_extensions/HUP080/runner.py self-test
```

Checkout-local source/reference locks are rebound to the clone's actual bytes
during materialization while the executed Windows byte hashes remain
provenance. LF GitHub archives therefore need no `core.autocrlf` setting.

Real phases require matching locally supplied authorization values. A new
independently hash-bound run-04 receipt is required after the development
freeze. No consumed private access receipt is included in this repository.
Both the public verifier and the full synthetic self-test validate the bundled
hash-only parent-v1 receipt without opening or reusing the private stage.
