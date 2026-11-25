# HUP080: red-manuscript Fig. S5 and Table S17

These results concern **development runs 01–03**. They are distinct from the earlier run04 evaluation in this directory. The [published Fig. S5](publication/HUP080_refined_all96_overview.pdf) is a one-page, vector PDF showing all 96 contacts in run01 at sample 5150, noise bank 1. Black is the recorded ictal signal, grey dashed the free prediction, green dash-dotted the preictal reference, and blue the refined feedback. In this window, mean occupation W1 to the reference fell from 0.562 to 0.142 and time-resolved W1 from 0.597 to 0.214. Both fell at all 96 contacts, including the 20 without direct input. These are unsmoothed, full-sample distances, not distances between the plotted lines.

## Render the figure

From the repository root, substitute the path to the separately authorized display bundle:

```sh
python patient_extensions/HUP080/plot_development_control.py --input PRIVATE/refined/engine/display_control.npz --output build/HUP080_S5.pdf --caption-metrics
```

The renderer reads `observed` (256 × 96), `free` and `controlled` (32 × 256 × 96), `reference` (15 × 256 × 96), 96 ordered labels, and the 76 direct-input indices. It preserves the window's common amplitude transform: pooled recorded 0.5th–99.5th percentiles with an 8% margin mapped to [−1, 1]. All four curves in a panel use the same observed-data bandwidth and the same 256-bin grid. Gaussian filtering includes guard bins and normalizes to the full sample; the displayed range is not a truncation rule for W1.

The accepted input SHA-256 is `105ca395a722e3239105e8756d58a6d81387692d8ced0530749fda5033f9096f`. The [predictor and controller instructions](development_control/README.md) give the authorized-input route to refit or replay. A newly packed bundle may have a different byte hash even if the tensors are identical; `--expect-input-sha256` accepts an independently checked bundle, but does not certify it as the published one. Neither the private signals nor the contact-density tables are distributed here.

The published PDF SHA-256 is `ce5911945cb55727644521e5022c872387d7cf590b556b2ff7bfd155863104de`. It was reproduced byte-for-byte from the frozen bundle with Python 3.12, NumPy 2.4.6, SciPy 1.17.1, and Matplotlib 3.10.9 (bundled DejaVu Sans, local UTC+8). Renderer, fonts or local time zone can alter PDF bytes on other systems.

## Rebuild Table S17 without patient signals

```sh
python patient_extensions/HUP080/summarize_development_windows.py
```

The [source CSV](results/table_s17_window_banks.csv) contains one row per development window and simulation bank: four contact-mean W1 distances in standardized model units, command peak, largest per-actuator RMS, mean summed squared command (`energy`), and near-saturation fraction (`sat`). `start` is the sample index; `fold` identifies the development run held apart from the other two for evaluation. Two simulation banks are averaged within each window, then distinct windows are weighted equally. A pass is counted **per bank**, not per window: peak ≤ 1.800 (numerical tolerance 1e−12), max RMS ≤ 0.450, energy ≤ 5.6862 and saturation fraction < 0.01. Saturation counts magnitudes at or above 95% of the command cap. Results are 22/24, 24/24 and 16/18 for the three groups (62/66 overall). See the [three-row manuscript table](results/table_s17.csv).

With the six original, non-public scalar evaluation files available, the same table and redistributable scalar extract can be regenerated without any patient array:

```sh
python patient_extensions/HUP080/summarize_development_windows.py --evaluation-root PRIVATE/refined --export-window-banks build/table_s17_window_banks.csv --output build/table_s17.csv
```

The private-source mode reads `engine/fold{1,2,3}/conditions.csv` and `additional/fold{1,2,3}.csv`. It verifies unique fold/start/bank keys and the recorded refined-controller budget decisions. The 33 windows come from three seizures; the two banks are repeated simulation draws, not additional recordings.
