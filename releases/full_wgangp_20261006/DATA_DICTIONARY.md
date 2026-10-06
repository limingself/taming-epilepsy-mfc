# Public numerical data dictionary

`results/current_results.json` is the current adoption authority. Its W1 values
are arithmetic means over channel rows and, for external cases, all 8 contexts
and 3 common-random-number banks. The 24 conditions are numerical repetitions,
not independent patients. Reported percentages are `100*(1-controlled/free)`
from unrounded means, not an average of per-channel percentages.

| Field | Definition / unit |
|---|---|
| `channel` | De-identified electrode label in the fixed patient channel order |
| `channel_index` | Zero-based fixed channel index |
| `direct_actuator` / `direct_actuated` | Boolean: dedicated masked actor output exists, not a claim of clinical electrode implantation |
| `context_index` | Fixed context index 0–7 in the outer seizure; context 5 has a separate development role in development tables |
| `crn_bank` | Paired standard-normal innovation bank 0–2 |
| `time_w1_free` | Mean time-resolved marginal Wasserstein-1 distance from zero-input ensemble to patient reference |
| `time_w1_controlled` / `time_w1_fresh` | Same distance under the adopted controller |
| `occupation_w1_free` | Wasserstein-1 after pooling particles and times, compared with the reference occupation law |
| `occupation_w1_controlled` / `occupation_w1_fresh` | Same pooled-law distance under the adopted controller |
| `*_paper` in HUP060 | Historical-controller comparison, **not the newly adopted Full** |
| `maximum_per_actuator_rms` | Largest per-actuator command RMS in that trajectory; dimensionless |
| `total_energy` | Discrete squared-input aggregate used by the executed resource-budget test; dimensionless |
| `control_peak` | Maximum command magnitude in that trajectory; dimensionless |
| `saturation_fraction` | Fraction meeting the executed amplitude-saturation criterion |
| `gate_c` | Original executed budget test, before HUP080's revised RMS cap; `False` must not be silently relabelled |
| `full_gate_b_pass` | Joint reported channel-quality test defined by the original protocol |
| `symmetric_sd_ratio` | max(control/reference SD, reference/control SD), dimensionless |

The private canonical arrays determine raw W1; fitted KDE curves do not define
the distance. HUP060 density CSVs are permitted plotting Source Data. External
long density CSVs and signal arrays are intentionally absent. Missing optional
fields are unavailable, not zero. Exploratory cold-32 tables are development-only
and must not replace the external adopted table.
