"""Reproduce the red-manuscript HUP080 development-window density plate (Fig. S5)."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import gaussian_filter1d


INPUT_SHA256 = "105ca395a722e3239105e8756d58a6d81387692d8ced0530749fda5033f9096f"
PUBLISHED_SHA256 = "ce5911945cb55727644521e5022c872387d7cf590b556b2ff7bfd155863104de"
# The original Matplotlib export used a local (UTC+8) timestamp.
EXPORT_DATE = datetime(2026, 11, 25, 12, 40, 2)
SERIES = ("observed", "free", "reference", "controlled")
SHAPES = ((256, 96), (32, 256, 96), (15, 256, 96), (32, 256, 96))
LEGEND = ("Recorded ictal", "Free prediction", "Preictal reference", "Controlled")
COLORS = ("#202020", "#797979", "#298455", "#2166AC")
STYLES = ("-", (0, (3.2, 1.6)), (0, (4, 1.3, 1, 1.3)), "-")
WIDTHS = (1.13, 1.27, 1.35, 1.45)
STYLE = {"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none",
         "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
         "axes.linewidth": 0.56, "legend.frameon": False, "axes.edgecolor": "#81878b",
         "xtick.color": "#494f51"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_bundle(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = [np.asarray(data[k], dtype=np.float64) for k in SERIES]
        labels = [name.removeprefix("EEG ").removesuffix("-Ref").removesuffix("-REF").replace(" ", "")
                  for name in data["labels"].astype(str).tolist()]
        selected = data["selected"]
    if len(labels) != 96 or len(set(labels)) != 96:
        raise ValueError("Expected 96 distinct contact labels in the original order.")
    if (selected.shape != (76,) or selected.dtype.kind not in "iu" or np.unique(selected).size != 76
            or (selected < 0).any() or (selected >= 96).any()):
        raise ValueError("Expected 76 distinct direct-input indices in [0, 95].")
    for key, array, expected in zip(SERIES, arrays, SHAPES):
        if array.shape != expected or not np.isfinite(array).all():
            raise ValueError(f"{key} must be finite and have shape {expected}.")
    return arrays, labels


def density(values, bandwidth, bins=256):
    """Smooth guard bins before cropping; normalize to the full original sample."""
    dx = 2 / bins
    pad = int(np.ceil(6 * bandwidth / dx)) + 2
    edges = -1 + dx * np.arange(-pad, bins + pad + 1)
    counts = np.histogram(values, edges)[0]
    values = gaussian_filter1d(counts.astype(float), bandwidth / dx, mode="constant", cval=0.,
                               truncate=6.) / (len(values) * dx)
    crop = slice(pad, pad + bins)
    return (edges[:-1] + dx / 2)[crop], values[crop]


def contact_curves(arrays):
    observed = arrays[0]
    qlo, qhi = np.quantile(observed, [.005, .995])
    margin = .08 * max(qhi - qlo, .1)
    lo, hi = qlo - margin, qhi + margin
    scale = 2 / (hi - lo)
    observed_x = (observed - lo) * scale - 1
    samples = [a.reshape(-1, 96) for a in arrays]
    curves = np.empty((96, 4, 256))
    for j in range(96):
        bandwidth = max(np.std(observed_x[:, j], ddof=1) * len(observed) ** (-.2), 2 / 256)
        for k, array in enumerate(samples):
            grid, curves[j, k] = density((array[:, j] - lo) * scale - 1, bandwidth)
    return grid, curves


def plot_plate(grid, curves, labels, output, archive_replay=False):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(8, 12, squeeze=False, figsize=(14.5, 10.9))
        try:
            fig.subplots_adjust(left=.058, right=.99, top=.914, bottom=.065, wspace=.18, hspace=.48)
            fig.suptitle("HUP080  |  Distribution control", y=.988, fontweight="bold", fontsize=15)
            handles = [Line2D([], [], lw=1.6, color=c, ls=s, label=l)
                       for c, s, l in zip(COLORS, STYLES, LEGEND)]
            fig.legend(handles=handles, ncol=4, bbox_to_anchor=(.5, .958), loc="upper center",
                       handlelength=2.7, columnspacing=1.75, fontsize=9)
            for j, ax in enumerate(axes.flat):
                for k in (0, 1, 3, 2):
                    ax.plot(grid, curves[j, k], lw=WIDTHS[k], ls=STYLES[k], color=COLORS[k],
                            solid_capstyle="round", dash_capstyle="butt")
                ax.set_title(labels[j], fontsize=8, loc="left", pad=1.8, fontweight="bold", color="#282C30")
                ax.set(xlim=(-1, 1), ylim=(0, max(curves[j].max(), 1e-9) * 1.055), yticks=[], xticks=[-1, 0, 1])
                ax.tick_params(axis="x", length=2, labelbottom=j // 12 == 7, labelsize=7.4, pad=2)
            fig.supylabel("Probability density", x=.011, fontsize=9)
            fig.supxlabel("Normalized amplitude", y=.014, fontsize=9.5)
            fig.savefig(output, format="pdf", metadata={"CreationDate": EXPORT_DATE} if archive_replay else None)
        finally:
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Authorized private display_control.npz")
    parser.add_argument("--output", required=True, type=Path, help="Destination for the single-page vector PDF")
    parser.add_argument("--expect-input-sha256", default=INPUT_SHA256,
                        help="Override only for an independently identified private replay (not the archived figure).")
    parser.add_argument("--caption-metrics", action="store_true", help="Also calculate caption distances from all original samples.")
    args = parser.parse_args()
    digest = sha256(args.input)
    if digest != args.expect_input_sha256.lower():
        parser.error(f"Private bundle SHA-256 mismatch: {digest}; refusing to attribute it to this plate.")
    if args.output.suffix.lower() != ".pdf":
        parser.error("--output must name a PDF file.")
    arrays, labels = read_bundle(args.input)
    if args.caption_metrics:
        from development_control.exact_distances import distance, time_distance
        obs, free, ref, ctl = arrays
        f_occ, c_occ = distance(free, ref), distance(ctl, ref)
        f_time, c_time = time_distance(free, ref), time_distance(ctl, ref)
        print(json.dumps({"free_to_recorded_occupation": float(distance(free, obs).mean()),
                          "occupation_free": float(f_occ.mean()), "occupation_controlled": float(c_occ.mean()),
                          "time_free": float(f_time.mean()), "time_controlled": float(c_time.mean()),
                          "both_decreased": int(((c_occ < f_occ) & (c_time < f_time)).sum()), "contacts": len(labels)}))
    grid, curves = contact_curves(arrays)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_plate(grid, curves, labels, args.output, archive_replay=(digest == INPUT_SHA256))
    result = sha256(args.output)
    print(f"{args.output}: SHA-256 {result}")
    if digest == INPUT_SHA256:
        print("Exact published PDF." if result == PUBLISHED_SHA256 else
              "Rendering bytes differ from the archived PDF; check the graphics environment.")


if __name__ == "__main__":
    main()
