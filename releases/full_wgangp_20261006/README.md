# Full WGAN-GP revision, 2026-10-06

This additive scientific-code and results release supersedes the historical
controller results as the current paper pointer without overwriting the older
HUP060 entry points or `patient_extensions/`. The root-level historical figure
commands remain historical. Use this directory's `results/current_results.json`
and `code_to_figure.json` for the adopted revision.

Full means the frozen patient-specific Graph–RC stochastic recurrence plus a
structured mean–deviation actor, the existing empirical-law objectives, and
WGAN-GP active from the first fresh actor update. It does not replace the
prediction model with a pure GAN. Standardized amplitudes and model inputs have
no physical stimulation units.

| Subject | Direct inputs | Current controller | Analysis role |
|---|---:|---|---|
| HUP060 | 13/36 | Neutral initialization; 1,000 actor updates; selected update 700 | Development/adaptation |
| HUP065 | 32/64 | Fresh 23-node parent selected at 550; low-gain 32-node expansion; selected added update 0 of 150 | Post hoc amended reanalysis |
| HUP080 | 76/96 | Fresh parent selected at 200; selected continuation added update 125 of 150 | Post hoc amended reanalysis |

HUP080's RMS resource cap was revised from 0.405 to 0.45. Old-cap failures are
retained in the lightweight tables; reclassification is not a performance
improvement. A separate 1,000-update neutral 32-node HUP065 trial is under
`exploratory/HUP065_cold1000_not_adopted/`. Its best feasible development
checkpoint, update 900, did not improve both endpoints over the adopted version;
context 6 and the outer seizure were not evaluated, and it is not the paper result.

## Verify without private signals or ML dependencies

```powershell
python -B releases/full_wgangp_20261006/verify_public_results.py
```

This standard-library-only check verifies public bytes, controller/plant SHA
metadata, the 13/32/76 masks, all current channel/context/bank metrics, unrounded
aggregate reductions, syntax, and exclusion rules. It never imports scientific
models, opens signal arrays, deserializes weights, trains, or evaluates a seizure.

## Contents and reproduction scope

- `supporting_materials/`: byte-identical executed training, component/baseline,
  plotting and portable source topology, scientific configuration metadata,
  dependencies, lightweight results, and permitted HUP060 figure Source Data.
- `figures/current/`: exactly the 16 adopted compiled figure assets, not all
  historical PDFs retained in the local LaTeX project. Other figure components
  and previews retain explicit provenance and exploratory labels.
- `framework/`: Python and native-vector `.mjs` builders, real adjacency and
  density sources, and the editable current framework. The JavaScript builder
  uses the existing artifact-tool environment; no `node_modules` are distributed.
- `results/`: current unrounded metrics and the two adopted context/bank CSVs.
- `parameter_table.json`: executed scientific settings and lineage; original
  absolute paths are provenance, not required public file locations.
- `source_provenance.json`, `public_manifest.json`: exact archive-to-public byte
  bindings and checksums. Old source docstrings and preview status flags remain
  unchanged to preserve executed code; `current_results.json` is the adoption
  authority, not those historical flags or their direct legacy CLI defaults.

The public package is **not the complete private frozen-input archive**.
HUP065/HUP080 arrays, serialized weights, raw ZIP/EDF files, long density CSVs,
training histories, and one-time access receipts are not redistributed. Original
recordings can be obtained independently from OpenNeuro ds004100 v1.1.3
(DOI `10.18112/openneuro.ds004100.v1.1.3`). Exact replay also needs the frozen
processed inputs and weights listed by the private manifests, or separately
regenerated inputs; regeneration creates a new run, not the original byte freeze.
See [PRIVATE_BUNDLE_REPLAY.md](PRIVATE_BUNDLE_REPLAY.md).

Use the repository MIT licence for software. Dataset-derived artifacts remain
subject to the upstream terms in the root `DATA_NOTICE.md`. This is retrospective
in-silico research, not a clinical stimulation protocol. Full manuscript/author
declarations awaiting confirmation are not published as a signed submission here.
