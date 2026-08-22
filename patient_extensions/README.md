# HUP065 and HUP080 public reproducibility extensions

This directory adds manuscript-matched HUP065 and HUP080 extensions without
changing the existing HUP060 reproduction entry points or manifests. It
contains portable code, path-relocated configuration templates, lightweight
aggregate and channel-level pass vectors, accepted publication PDF/PNG files,
and provenance/QA records.

| Subject | Development / outer split | Selected controller | Direct nodes | Complete Gate-B | Safety |
|---|---|---|---:|---:|---:|
| HUP065 | run-01--02 / run-03 | `f-0.35_q-0.65_gain-0.350_tau-0.200`, rolling block 8 | 23/64 | 21/64 | 64/64 contact vector; all 24 trajectories met Gate-C |
| HUP080 exploratory v2 | run-01--03 / run-04 | `f-0.80_q-0.20_gain-0.350_tau-0.000`, rolling block 2 | 76/96 | 65/96 | 96/96 contact vector; all 24 trajectories met Gate-C |

HUP080 v1 validly stopped at `NO_GO_before_part3` after testing rolling blocks
4, 8, 12, 16, 24 and 32. Exploratory v2 added only blocks 1 and 2; thresholds
and the every-fold rule were unchanged. It is not the original preregistered
replication. At `tau_G=0`, its 20 indirect channels receive no instantaneous
heat-kernel input and can change only through the frozen plant dynamics.

Run the public, signal-free verifier with Python's standard library:

```powershell
python patient_extensions/verify_public_results.py
```

To prepare a relocation-specific local configuration after independently
obtaining OpenNeuro `ds004100`:

```powershell
python patient_extensions/materialize_config.py --patient HUP065 --dataset-root <ds004100-root>
python patient_extensions/materialize_config.py --patient HUP080 --dataset-root <ds004100-root> --force
```

The templates retain the executed scientific values, but their path and
authorization bindings are portable. Materialized configs therefore have new
hashes and must not be presented as the original freezes. The executed hashes
and exact portability edits are in `PORTABILITY_MAP.json`; private archive
manifest anchors are in `archive_hashes/README.md`.

The committed templates also retain the executed Windows byte hashes as
provenance. During materialization, every source/reference lock that resolves
inside the current checkout is recursively rebound to that checkout's actual
bytes. This makes LF GitHub archives and clones portable without relying on
`core.autocrlf`; EOL-only changes are disclosed in the local relocation
receipt and do not alter scientific parameters. The materialized shared
protocol additionally permits the audited public directory names `HUP065` and
`HUP080`, while preserving the executed private allowlist separately.

The public package deliberately excludes raw EEG/ZIP/EDF data, development or
outer signal arrays, serialized weights, large or long-form tables, training
histories, caches, failed staging trees, and one-time authorization/access
receipts. Full reruns additionally require independently obtained data and any
hash-locked legacy authority roots declared by the materializer options.
