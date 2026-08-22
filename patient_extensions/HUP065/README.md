# HUP065 sparse-control extension

HUP065 used run-01 and run-02 for development and one sealed retrospective
run-03 outer evaluation. Development LORO selected an 8-sample rolling block
and the frozen candidate `f-0.35_q-0.65_gain-0.350_tau-0.200`, giving 23 direct
actuators among 64 channels. The outer result passed complete six-component
Gate-B on 21/64 channels, the contact-level safety vector on 64/64 channels,
and Gate-C on all 24 predeclared trajectories.

The code here is a portable, path-relocated snapshot. `config.template.json`
preserves the executed scientific values but is not the byte-identical frozen
configuration. Its executed SHA-256 and every portability edit are recorded in
`../PORTABILITY_MAP.json`.

For a fresh local rerun, obtain OpenNeuro `ds004100` independently, create a
clean clone, and materialize the template:

```powershell
python patient_extensions/materialize_config.py --patient HUP065 --dataset-root <dataset-root>
python -B patient_extensions/HUP065/runner.py static-check
python -B patient_extensions/HUP065/runner.py path-check
python -B patient_extensions/HUP065/static_self_test.py
```

Materialization keeps the executed Windows byte hashes as provenance and
rebinds checkout-local source/reference locks to the clone's actual bytes, so
LF archives work without `core.autocrlf`. The static self-test validates the
bundled hash-only failure-provenance receipt; the excluded 106 MB failed stage
is neither opened nor required and may never be reused for selection.

Continue through the documented development phases in `runner.py`. A new local
authorization supplied through `HUP065_EXPECTED_SCIENCE_AUTHORIZATION` (and,
for the outer phase, `HUP065_EXPECTED_OUTER_AUTHORIZATION`) and a new hash-bound
outer receipt are required before opening run-03. Consumed private access
material is deliberately absent. A relocated rerun has its own configuration
hash and must not be presented as the original freeze.
