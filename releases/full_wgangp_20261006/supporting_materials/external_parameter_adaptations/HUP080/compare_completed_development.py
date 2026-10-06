"""Compare completed arm development metrics, without reading outer outcomes."""
from pathlib import Path
import json
import pandas as pd

root = Path(__file__).resolve().parent
initial = pd.read_csv(root / "starting_development/development_channel_metrics.csv")
initial_mean = initial.groupby("channel_index").mean(numeric_only=True)
metrics = ["time_w1_controlled", "occupation_w1_controlled", "mean_abs_error", "symmetric_sd_ratio"]
results = {}
for arm in sorted((root / "arms").iterdir()):
    if not (arm / "training_summary.json").is_file():
        continue
    selected = pd.read_csv(arm / "selected_development/development_channel_metrics.csv")
    merged = initial.merge(selected, on=["development_run_slot", "channel_index"], suffixes=("_start", "_adapted"), validate="one_to_one")
    adapted_mean = selected.groupby("channel_index").mean(numeric_only=True)
    report = {"development_cases": 3, "channels": 96, "outer_results_read": False, "metrics": {}}
    for metric in metrics:
        a, b = initial_mean[metric], adapted_mean[metric]
        report["metrics"][metric] = dict(starting_mean=float(a.mean()), adapted_mean=float(b.mean()),
            channels_lower=int((b < a - 1e-12).sum()), channels_higher=int((b > a + 1e-12).sum()),
            channel_cases_lower=int((merged[metric + "_adapted"] < merged[metric + "_start"] - 1e-12).sum()),
            channel_cases_higher=int((merged[metric + "_adapted"] > merged[metric + "_start"] + 1e-12).sum()),
            largest_channel_increase=float((b - a).max()), largest_channel_decrease=float((b - a).min()))
    improved = (adapted_mean.time_w1_controlled < initial_mean.time_w1_controlled) & (adapted_mean.occupation_w1_controlled < initial_mean.occupation_w1_controlled)
    report["both_distances_improved_vs_previous_controller_channels"] = int(improved.sum())
    report["note"] = "This is an exact paired development comparison. It does not claim all channels improve versus the previous controller."
    results[arm.name] = report
(root / "completed_development_comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results), flush=True)
