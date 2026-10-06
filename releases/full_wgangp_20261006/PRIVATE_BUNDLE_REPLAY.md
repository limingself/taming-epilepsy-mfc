# Exact replay and independently obtained inputs

The executed Python/PowerShell sources are preserved byte-for-byte. Do not edit
their model equations, selection rules, data boundaries, or source-hash checks
to bypass a missing private input. Existing `verify-bundle` commands require
the complete frozen bundle, not just the public code files.

For the author's current local archive, the complete adopted bundle root is:

```
D:/论文修改资料/2026-10-06/Full_WGANGP_all_cases_adopted_20261006
```

It contains sibling `supporting_materials/HUP060_restart`,
`HUP060_comparisons`, the complete external fresh runs, and both external
adaptation trees with their original manifests and trusted serialized artifacts.
Do not copy or publish these omitted artifacts to GitHub. A separate complete
cold-32 bundle is under
`D:/论文修改资料/2026-10-06/HUP065_top32_cold1000_development_preview_20261006`.

The path-only launcher below defaults to a **dry run**. It does not silently
inspect the author's drive or automatically fetch any private data. Set an
explicit bundle root and action:

```powershell
python releases/full_wgangp_20261006/replay_private_bundle.py --bundle-root "D:/论文修改资料/2026-10-06/Full_WGANGP_all_cases_adopted_20261006" --action verify-hup060
```

Only with `--execute` does it call the chosen private bundle's existing entry
point. Verification actions perform the original non-training checks. Training
actions require a unique new `--tag`; outputs remain in the private wrapper's
isolated paths and never overwrite adopted public results. No terminal or outer
evaluation action is exposed by this launcher. Opening those data is a separate
scientific decision, not a repository-synchronization step.

The public `verify_public_results.py` needs none of these inputs and does not use
this launcher. It verifies published numerical tables and provenance only.

For external users, obtain source recordings independently from OpenNeuro
ds004100 v1.1.3, follow the historical `patient_extensions/materialize_config.py`
route to create a **new**, correctly path-bound configuration, then reconstruct
the frozen Part-I/Part-II inputs before using the new controller source.
Exact archived hashes apply only to the executed frozen bytes; new fits and
materialized path/EOL bindings must have their own provenance. No legal or
privacy restriction is invented here: omission is the established public-release
scope, not a claim that the public OpenNeuro dataset requires new access approval.

## Audited scientific entry points

| Scope | Executed entry | Relevant constraint |
|---|---|---|
| HUP060 restart | `HUP060_restart/reproduce_portable.py` | `--verify-bundle`, or `train/evaluate --tag NEW`; direct legacy runner's default is a historical tag |
| HUP060 comparisons | `HUP060_comparisons/reproduce_hup060_comparisons.py` | `verify`, `train-ablation`, `baselines --tag NEW`; sibling restart required |
| HUP065 fresh parent | `external_fresh_training/run_external_fresh.py` | HUP065 original development/reference route |
| HUP080 fresh parent | `external_fresh_training/run_external_fresh_uniform_reference.py` | This executed variant's subject CLI permits HUP080 only |
| HUP065 adopted extension | `HUP065/small_gain_decoupled_screen/reproduce_portable_small_gain.py` | `verify-bundle`/`screen-train`; private old23 warm-start actor required |
| HUP080 adopted extension | `HUP080/portable/reproduce_optimization.py` | `verify-bundle`/`preflight`; private starting/final weights required; preserve Windows long paths |
| HUP065 cold trial, not adopted | `run_cold_top32.py` in its separate bundle | `preflight/train --tag NEW --threads 2`; no old actor/critic load; failed dual-endpoint eligibility |

Use Python 3.12 and the exact archived requirement/environment records. Installed
versions are execution records, not a guarantee that a future package index or
different platform will recreate the same numerical freeze.
