"""Render the HUP060-style run-02 replay from a locally generated evaluation NPZ.

Example: python render.py --subject HUP065 --input eval.npz --out figures
The input and all outputs (including contact labels) are local derived data. This
script only reads the supplied frozen snapshot; it never fits or selects a model.
The default exports the all-contact density plate used for Fig. S4 or S5.
Use --diagnostics for the original supplementary local views and paginated plots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, wasserstein_distance
from threadpoolctl import threadpool_limits

CASES = {"HUP065": (64, 32), "HUP080": (96, 76)}
STEPS, PARTICLES, REFERENCES = 256, 32, 15
BANDWIDTH = 0.16  # Absolute kernel standard deviation in run-01 standardized units.
COLORS = ("#272727", "#888888", "#3C8D62", "#2166AC")
STYLES = ("-", ":", "-.", "-")
SERIES = ("Observed ictal", "Free Graph–RC", "Preictal reference", "Full (simulated)")


def load_eval(path: Path, subject: str) -> dict:
    contacts, direct = CASES[subject]
    shapes = {"channels": (contacts,), "past": (STEPS, contacts),
              "observed": (STEPS, contacts), "free": (PARTICLES, STEPS, contacts),
              "controlled": (PARTICLES, STEPS, contacts), "reference": (REFERENCES, STEPS, contacts),
              "selected_indices": (direct,), "controls": (PARTICLES, STEPS, direct)}
    with np.load(path, allow_pickle=False) as archive:
        missing = shapes.keys() - set(archive.files)
        if missing:
            raise ValueError(f"Missing evaluation arrays: {sorted(missing)}")
        data = {key: archive[key] for key in shapes}
    for key, shape in shapes.items():
        if data[key].shape != shape:
            raise ValueError(f"{subject}: expected {key} to have shape {shape}, got {data[key].shape}")
        if key not in ("channels", "selected_indices"):
            if data[key].dtype.kind not in "fiu" or not np.isfinite(data[key]).all():
                raise ValueError(f"{key} must contain finite real values")
    if data["channels"].dtype.kind not in "SU":
        raise ValueError("channels must be string labels")
    data["channels"] = data["channels"].astype(str)
    if len(set(data["channels"])) != contacts or any(not s.strip() for s in data["channels"]):
        raise ValueError("Channel labels must be nonempty and unique")
    ix = data["selected_indices"]
    if ix.dtype.kind not in "iu" or len(set(ix)) != direct or ix.min() < 0 or ix.max() >= contacts:
        raise ValueError("selected_indices must identify distinct in-range integer channels")
    if np.max(np.abs(data["controls"])) > 1.8 + 1e-8:
        raise ValueError("Stored commands exceed the replay amplitude bound")
    return data


def occupation_w1(x, y):
    """One exact empirical W1 per contact, pooling all time points and paths."""
    if x.shape[-1] != y.shape[-1]:
        raise ValueError("Occupation distances require the same contact order and count")
    return np.array([wasserstein_distance(x[..., i].ravel(), y[..., i].ravel())
                     for i in range(x.shape[-1])])


def time_w1(x, y):
    """W1 across paths at each time, averaged over time, separately by contact.

    Arguments are (paths, times, contacts); a recorded signal has one path.
    Integrating the empirical quantiles exactly avoids approximate quantile grids
    when the two path counts differ (e.g. 32 simulations, 15 references).
    """
    if x.ndim != 3 or y.ndim != 3 or x.shape[1:] != y.shape[1:] or not min(len(x), len(y)):
        raise ValueError("Expected nonempty path sets with aligned time and contact dimensions")
    a, b = np.sort(x, axis=0), np.sort(y, axis=0)
    edges = np.unique(np.r_[np.arange(len(a) + 1) / len(a), np.arange(len(b) + 1) / len(b)])
    mid = (edges[:-1] + edges[1:]) / 2
    delta = np.abs(a[(len(a) * mid).astype(int)] - b[(len(b) * mid).astype(int)])
    return np.einsum("r,rtc->c", np.diff(edges), delta) / a.shape[1]


def scores(data):
    free, controlled, observed, past, ref = (data[k] for k in
                                              ("free", "controlled", "observed", "past", "reference"))
    pred = free.mean(axis=0)
    lo, hi = np.quantile(free, (.1, .9), axis=0)
    direct = np.isin(np.arange(len(data["channels"])), data["selected_indices"])
    correlation = [np.corrcoef(p, y)[0, 1] if min(p.std(), y.std()) > 1e-12 else np.nan
                   for p, y in zip(pred.T, observed.T)]
    frame = pd.DataFrame({
        "contact": data["channels"], "direct_input": direct,
        "forecast_occup_W1": occupation_w1(free, observed),
        "prior_second_occup_W1": occupation_w1(past, observed),
        "forecast_mean_RMSE": np.sqrt(np.mean((pred-observed)**2, axis=0)),
        "persistence_RMSE": np.sqrt(np.mean((observed-past[-1])**2, axis=0)),
        "forecast_mean_trace_r": correlation,
        "empirical_10_90_coverage": ((observed >= lo) & (observed <= hi)).mean(axis=0),
    })
    finite_r = frame.forecast_mean_trace_r.dropna()
    total = {"forecast_W1": float(frame.forecast_occup_W1.mean()),
             "previous_second_W1": float(frame.prior_second_occup_W1.mean()),
             "n_forecast_W1_better": int((frame.forecast_occup_W1 < frame.prior_second_occup_W1).sum()),
             "mean_contact_RMSE": float(frame.forecast_mean_RMSE.mean()),
             "mean_contact_persistence_RMSE": float(frame.persistence_RMSE.mean()),
             "median_trace_r": None if finite_r.empty else float(finite_r.median()),
             "mean_10_90_coverage": float(frame.empirical_10_90_coverage.mean()),
             "simulated_command_rms": float(np.sqrt(np.mean(data["controls"]**2))),
             "simulated_command_peak": float(np.abs(data["controls"]).max()),
             "reference_distances": {}}
    # Use the full sample support for every distance (the plotted KDE is irrelevant).
    for name, law in (("observed", observed[None]), ("free", free), ("controlled", controlled)):
        for metric, fn in (("time_W1", time_w1), ("occupation_W1", occupation_w1)):
            key = f"{name}_{metric}_to_ref"
            frame[key] = fn(law, ref)
            total["reference_distances"][key] = float(frame[key].mean())
    for metric in ("time_W1", "occupation_W1"):
        a, b = frame[f"free_{metric}_to_ref"], frame[f"controlled_{metric}_to_ref"]
        # Keep the original per-contact column names for consumers of the case scripts.
        frame[f"free_{metric}"], frame[f"full_{metric}"] = a, b
        improved = b < a
        total[metric] = {"free": float(a.mean()), "controlled": float(b.mean()),
                         "reduction_pct": float(100 * (1-b.mean()/a.mean())) if a.mean() > 0 else None,
                         "contacts_improved": int(improved.sum()),
                         "direct_improved": int((improved & direct).sum()),
                         "indirect_improved": int((improved & ~direct).sum())}
    return frame, total


def draw(data, subject, out, *, diagnostics=False):
    """Same empirical KDE, full ordering, fixed particle, and layouts as the case figures."""
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8,
                         "legend.frameon": False, "axes.spines.right": False,
                         "axes.spines.top": False, "axes.linewidth": .55})
    names = data["channels"]
    n = len(names)
    selected = set(data["selected_indices"])
    obs, free, ref, full = (data[k] for k in ("observed", "free", "reference", "controlled"))
    curves = (obs, free, ref, full)
    paths = (obs, free[0], ref[0], full[0])
    mean = free.mean(axis=0)
    lo, hi = np.quantile(free, (.1, .9), axis=0)
    limits = np.array([np.quantile(v, (.001, .999)) for v in curves])
    grid = np.linspace(limits[:, 0].min()-3*BANDWIDTH, limits[:, 1].max()+3*BANDWIDTH, 300)
    densities = np.empty((n, len(curves), len(grid)))
    for i in range(n):
        for j, curve in enumerate(curves):
            v = curve[..., i].ravel()
            sd = v.std(ddof=1)
            densities[i, j] = (gaussian_kde(v, bw_method=BANDWIDTH/sd)(grid) if sd > 1e-10 else
                               np.exp(-.5*((grid-v.mean())/BANDWIDTH)**2)/(BANDWIDTH*np.sqrt(2*np.pi)))
    np.savez_compressed(out/"kde_source.npz", grid=grid, channels=names, series=SERIES, density=densities)
    time = np.arange(STEPS)/STEPS
    descriptions = {
        "forecast": ("Run-02 autonomous forecast", "1 s, 256 Hz   |   all forecasts open-loop"),
        "control": ("Run-02 full mean-field control", "* directly actuated   |   single-window occupation densities"),
        "control_paths": ("Paired control trajectories", "fixed particle 0 in both arms; first validation reference path shown"),
    }
    def page(kind, indices, overview):
        cols = 8 if overview else 4
        rows = (len(indices)+cols-1)//cols
        size = ((15,13.5) if n == 64 else (14,18.8)) if overview else (9.2,8.4)
        fig, axes = plt.subplots(rows, cols, figsize=size, squeeze=False)
        fig.subplots_adjust(left=.052, right=.982, top=.886, bottom=.065, hspace=.76, wspace=.34)
        title, detail = descriptions[kind]
        subtitle = (f"All {n} contacts" if overview else f"Contacts {indices[0]+1}–{indices[-1]+1} of {n}")
        fig.text(.05,.978,subject+"  /  "+title,ha="left",va="top",size=14,weight="bold")
        fig.text(.05,.952,subtitle+"   |   "+detail,ha="left",va="top",size=8,color="#333333")
        handles = ([Line2D([],[],color=COLORS[0],lw=1,label="Observed ictal"),
                    Line2D([],[],color="#AF5A12",lw=1,label="Free ensemble mean"),
                    Patch(color="#ECD7BE",label="Free particle 10th–90th percentiles")] if kind == "forecast"
                   else [Line2D([],[],color=c,ls=s,lw=1.3,label=l) for c,s,l in zip(COLORS,STYLES,SERIES)])
        fig.legend(handles=handles,loc="upper left",bbox_to_anchor=(.044,.934),ncol=len(handles))
        for ax, i in zip(axes.flat, indices):
            if kind == "forecast":
                ax.fill_between(time,lo[:,i],hi[:,i],lw=0,color="#ECD7BE")
                ax.plot(time,obs[:,i],color=COLORS[0],lw=.68)
                ax.plot(time,mean[:,i],color="#AF5A12",lw=.85)
            else:
                for j in range(len(curves)):
                    x, y = (grid,densities[i,j]) if kind == "control" else (time,paths[j][:,i])
                    ax.plot(x,y,color=COLORS[j],ls=STYLES[j],lw=1 if kind == "control" else .7)
            ax.set_title(names[i]+(" *" if kind != "forecast" and i in selected else ""),
                         loc="left",size=8,pad=2,fontweight="bold")
            if kind == "control":
                ax.set_xlim(grid[0],grid[-1]); ax.set_ylim(0,1.14*densities[i].max()); ax.set_yticks([])
            else:
                ax.set_xlim(0,1); ax.set_xticks((0,.5,1))
            ax.tick_params(labelsize=6,length=2,pad=1)
        amplitude = "Amplitude (run-01 standardized units)"
        fig.supxlabel(amplitude if kind == "control" else "Time from forecast boundary (s)",fontsize=9,y=.017)
        fig.supylabel("Density" if kind == "control" else amplitude,fontsize=9,x=.007)
        return fig

    for kind in (descriptions if diagnostics else ("control",)):
        fig = page(kind, range(n), True)
        if kind == "control":
            for ext in ("pdf", "png"):
                fig.savefig(out/f"{subject}_run02_full_all{n}.{ext}", dpi=185)
        if diagnostics:
            for ext in ("png","pdf","svg"):
                fig.savefig(out/f"{kind}_all{n}.{ext}", dpi=185)
        plt.close(fig)
        if not diagnostics:
            continue
        with PdfPages(out/f"{kind}_16_per_page.pdf") as pdf:
            for first in range(0,n,16):
                fig = page(kind,range(first,min(first+16,n)),False)
                pdf.savefig(fig); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", choices=CASES, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Frozen, combined evaluation NPZ")
    parser.add_argument("--out", type=Path, required=True, help="Local destination for derived data and figures")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--metrics-only", action="store_true", help="Skip KDE and all plots")
    mode.add_argument("--diagnostics", action="store_true",
                      help="Also draw forecasts, example paths and paginated local diagnostics")
    args = parser.parse_args(argv)
    with threadpool_limits(limits=1):
        data = load_eval(args.input, args.subject)
        frame, summary = scores(data)
        with args.input.open("rb") as stream:
            summary["evaluation_data_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        summary.update(subject=args.subject, units="run-01 standardized amplitude",
                       averaging="equal weight per contact in the preselected run-02 window",
                       forecast_W1_definition="occupation distance: free forecast versus recorded ictal future",
                       reference_distances_definition="each named law versus the same run-01 preictal reference")
        args.out.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.out/"contact_metrics.csv", index=False, encoding="utf-8-sig")
        (args.out/"metrics.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        if not args.metrics_only:
            draw(data,args.subject,args.out,diagnostics=args.diagnostics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
