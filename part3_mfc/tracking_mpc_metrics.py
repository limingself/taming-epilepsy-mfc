"""Compute Fig. 7 and Table 1 endpoints directly from paired trajectories.

The input NPZ is private. This CLI only prints contact-aggregated results.
"""
import argparse
import json

import numpy as np
from scipy.stats import wasserstein_distance


def contact_metrics(x, ref):
    """Paths x time x contacts; the two ensembles may have different sizes."""
    if x.ndim != 3 or ref.ndim != 3 or x.shape[1:] != ref.shape[1:]:
        raise ValueError("Expected path/time/contact arrays, with matching time and contacts.")
    if not np.isfinite(x).all() or not np.isfinite(ref).all():
        raise ValueError("Missing or infinite rollout/reference values")
    time = np.array([np.mean([wasserstein_distance(x[:, t, j], ref[:, t, j])
                              for t in range(x.shape[1])]) for j in range(x.shape[2])])
    occ = np.array([wasserstein_distance(x[:, :, j].ravel(), ref[:, :, j].ravel())
                    for j in range(x.shape[2])])
    width = np.abs(x.std((0, 1), ddof=0)-ref.std((0, 1), ddof=0))
    center = np.abs(x.mean((0, 1))-ref.mean((0, 1)))
    return time, occ, width, center


def score_bank(bank):
    free, ref = bank["free"], bank["ref"]
    if free.shape != (32, 256, 36) or ref.shape != (15, 256, 36):
        raise ValueError("Not the HUP060 primary-bank shape (32 paths; 15 reference segments).")
    summaries, stats = {}, {}
    for name in ("free", "full", "mpc"):
        x = bank[name]
        u = np.zeros((32, 256, 13)) if name == "free" else bank[name+"_u"]
        if (x.shape != free.shape or u.shape != (32, 256, 13)
                or not np.isfinite(u).all() or np.abs(u).max() > 1.8+1e-12):
            raise ValueError(f"{name}: unexpected state/command shape or amplitude.")
        time, occ, width, center = stats[name] = contact_metrics(x, ref)
        summaries[name] = {
            "time_W1": float(time.mean()),
            "occupation_W1": float(occ.mean()),
            "median_absolute_SD_error": float(np.median(width)),
            "median_absolute_mean_error": float(np.median(center)),
            "command_RMS": float(np.mean(u*u)**.5),
            "both_lower_than_free": int(((time < stats["free"][0]) & (occ < stats["free"][1])).sum())
        }
    return {"scores": summaries,
            "contacts_with_Full_lower_SD_error_than_MPC": int((stats["full"][2] < stats["mpc"][2]).sum())}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("rollouts", help="Authorized paired HUP060 trajectory NPZ.")
    with np.load(p.parse_args().rollouts, allow_pickle=False) as bank:
        print(json.dumps(score_bank(bank), indent=2, allow_nan=False))
