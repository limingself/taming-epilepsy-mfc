"""Read-only frozen-controller diagnostic; generated tables are new outputs."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("hup080_optimization_diagnostic", root / "optimize_hup080.py")
opt = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = opt
spec.loader.exec_module(opt)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
base = opt.load_runner()
s = opt.setup(base)
rows = []
with torch.no_grad():
    reference = s["validation"].numpy()
    rs = reference.std(axis=(0, 1))
    rm = reference.mean(axis=(0, 1))
    for slot, (initial, noise, uncontrolled) in enumerate(base.validation_cache(s)):
        ctrl = s["core"].empirical_fp_rollout(s["stepper"], s["actor"], initial, noise).scaled[:, 1:].numpy()
        ratio = ctrl.std(axis=(0, 1)) / np.maximum(rs, 1e-12)
        for j in range(96):
            rows.append(dict(run_slot=slot, channel=str(s["arrays"]["channels"][j]),
                reference_sd=float(rs[j]), controlled_sd=float(ctrl[:, :, j].std()),
                controlled_over_reference_sd=float(ratio[j]),
                symmetric_sd_ratio=float(max(ratio[j], 1 / max(ratio[j], 1e-12))),
                signed_mean_bias=float(ctrl[:, :, j].mean() - rm[j])))
frame = pd.DataFrame(rows)
frame.to_csv(root / "starting_development/scale_direction_diagnostic.csv", index=False)
summary = dict(cases=3, channel_cases=len(frame),
    controlled_sd_larger_than_reference_cases=int((frame.controlled_over_reference_sd > 1).sum()),
    overwide_above_two_cases=int((frame.controlled_over_reference_sd > 2).sum()),
    overnarrow_below_half_cases=int((frame.controlled_over_reference_sd < .5).sum()),
    median_controlled_over_reference_sd=float(frame.controlled_over_reference_sd.median()),
    mean_absolute_pooled_mean_bias=float(frame.signed_mean_bias.abs().mean()),
    worst_channel_cases=frame.sort_values("symmetric_sd_ratio", ascending=False).head(12).to_dict(orient="records"),
    no_training_or_outer_opening=True,
    note="This diagnoses development context5 only, not new outer results or selection outcomes")
base.dump(root / "starting_development/scale_direction_diagnostic.json", summary)
print(json.dumps(summary), flush=True)
