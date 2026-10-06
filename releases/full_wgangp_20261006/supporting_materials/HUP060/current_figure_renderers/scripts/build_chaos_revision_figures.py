"""Isolated manuscript figure revision; frozen source data are read only."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"D:\VS code\distribution control\output\part3\source_data")
OUTPUT = ROOT / "figures"
OUTPUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7, "axes.linewidth": .8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
})
COLORS = {"Observed ictal": "#272727", "Free RC-SDE": "#9A9A9A",
          "Interictal reference": "#3C8D62", "Actor + WGAN-GP": "#2166AC"}
LABELS = {"Observed ictal": "Observed ictal", "Free RC-SDE": "Uncontrolled Graph–RC",
          "Interictal reference": "Interictal reference", "Actor + WGAN-GP": "Full policy"}
SERIES = tuple(COLORS)
DATA_PATH = SOURCE / "figures_06_08/paired_comparison.npz"
DATA = np.load(DATA_PATH, allow_pickle=False)
CHANNELS = DATA["channels"].astype(str)
SELECTED = set(DATA["selected_indices"].astype(int).tolist())
FREE = np.asarray(DATA["uncontrolled_scaled"], dtype=float)
FULL = np.asarray(DATA["candidate_controlled_scaled"], dtype=float)
REFERENCE = np.asarray(DATA["reference_validation_scaled"], dtype=float)
FREE_W1 = np.array([wasserstein_distance(FREE[:, :, j].ravel(), REFERENCE[:, :, j].ravel()) for j in range(36)])
FULL_W1 = np.array([wasserstein_distance(FULL[:, :, j].ravel(), REFERENCE[:, :, j].ravel()) for j in range(36)])
QA = {"backend": "Python/matplotlib", "no_training_or_parameter_tuning": True,
      "density_curve_source": "frozen Source Data; existing samples plotted unchanged",
      "primary_mean_free_occupation_w1": float(FREE_W1.mean()),
      "primary_mean_full_occupation_w1": float(FULL_W1.mean()), "source_hashes": {}}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(fig, name):
    fig.savefig(ROOT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def box(ax, xy, wh, color="white", edge="#CFD5DB", rounding=.012):
    p = FancyBboxPatch(xy, *wh, boxstyle=f"round,pad=0.006,rounding_size={rounding}",
                      transform=ax.transAxes, fc=color, ec=edge, lw=.8)
    ax.add_patch(p)
    return p

def arrow(ax, start, end, color="#596775", connection="arc3,rad=0", width=1.2):
    ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes, arrowstyle="-|>",
                                mutation_scale=9, lw=width, color=color,
                                connectionstyle=connection))

def framework():
    fig = plt.figure(figsize=(183/25.4, 103/25.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    xs = [.022, .365, .708]; w=.27
    for x in xs:
        box(ax, (x, .405), (w, .535), "#F9FAFB")
    headings = ["1  Fit and select", "2  Freeze the recurrence", "3  Optimize feedback"]
    for x, heading in zip(xs, headings):
        ax.text(x+w/2, .895, heading, ha="center", va="center", fontsize=9, fontweight="bold", transform=ax.transAxes)
    box(ax, (.050, .755), (.214, .077), "#FFFFFF")
    ax.text(.157, .794, "Patient-specific iEEG", ha="center", va="center", fontsize=8.5, transform=ax.transAxes)
    arrow(ax, (.157, .75), (.157, .704))
    box(ax, (.050, .587), (.214, .105), "#EFF3F7")
    ax.text(.157, .654, "PLV graph for the mask", ha="center", fontsize=8, transform=ax.transAxes)
    ax.text(.157, .611, "Selected virtual inputs", ha="center", fontsize=8, transform=ax.transAxes)
    box(ax, (.050, .443), (.214, .105), "#EFF3F7")
    ax.text(.157, .510, "PLV graph for the model", ha="center", fontsize=8, transform=ax.transAxes)
    ax.text(.157, .467, "Features and input map", ha="center", fontsize=8, transform=ax.transAxes)
    arrow(ax, (.300, .675), (.355, .675))
    ax.text(.500, .795, "Stochastic Graph–RC", ha="center", fontsize=9, fontweight="bold", transform=ax.transAxes)
    ax.text(.500, .750, "sample-time recurrence", ha="center", fontsize=8.5, transform=ax.transAxes)
    ax.text(.500, .655, "State · delays · reservoir", ha="center", fontsize=8, transform=ax.transAxes)
    ax.text(.500, .606, "State-dependent innovations", ha="center", fontsize=8, transform=ax.transAxes)
    box(ax, (.396, .458), (.208, .089), "#EFF3F7")
    ax.text(.500, .500, "Empirical particle law", ha="center", va="center", fontsize=8.5, transform=ax.transAxes)
    arrow(ax, (.644, .675), (.698, .675))
    box(ax, (.736, .757), (.214, .074), "#EAF3EE", edge="#A1BAAE")
    ax.text(.843, .795, "Individual non-ictal reference", ha="center", va="center", fontsize=7.8, transform=ax.transAxes)
    arrow(ax, (.843, .750), (.843, .704), color="#3C8D62")
    ax.text(.843, .655, "Mean–deviation feedback", ha="center", fontsize=8.5, fontweight="bold", transform=ax.transAxes)
    ax.text(.843, .603, "Common input", ha="center", fontsize=8, transform=ax.transAxes)
    ax.text(.843, .558, "+ centered corrections", ha="center", fontsize=8, transform=ax.transAxes)
    box(ax, (.736, .442), (.214, .076), "#E8EFF7", edge="#A8BCD4")
    ax.text(.843, .480, "Bounded virtual inputs", ha="center", va="center", fontsize=8.5, transform=ax.transAxes)
    # A return arrow distinguishes feedback from a purely feed-forward workflow.
    arrow(ax, (.842, .395), (.503, .395), color="#2166AC", connection="arc3,rad=-.22")
    ax.text(.671, .287, "Assumed model-input interface", ha="center", fontsize=8, color="#2166AC", transform=ax.transAxes)
    cases = [("HUP060", "Development", "13 / 36 direct inputs"),
             ("HUP065", "Held-out seizure", "23 / 64 direct inputs"),
             ("HUP080", "Amended exploratory", "76 / 96 direct inputs")]
    for x, (patient, level, count) in zip(xs, cases):
        box(ax, (x, .026), (w, .214), "#FFFFFF")
        ax.text(x+w/2, .188, patient, ha="center", fontsize=9, fontweight="bold", transform=ax.transAxes)
        ax.text(x+w/2, .130, level, ha="center", fontsize=8.5, transform=ax.transAxes)
        ax.text(x+w/2, .074, count, ha="center", fontsize=8, transform=ax.transAxes)
    save(fig, "hup060_end_to_end_framework")

def panel_label(ax, label):
    ax.text(-.16, 1.06, label, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="left", va="bottom")

def legend_handles():
    return [Line2D([0], [0], color=COLORS[s], lw=1.2,
                   ls=":" if s=="Free RC-SDE" else "-." if s=="Interictal reference" else "-",
                   label=LABELS[s]) for s in SERIES]

def representative():
    paths = {name: SOURCE / f"figure_06/source_data_representative_{name}.csv" for name in ["trajectories", "densities", "controls"]}
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    for path in paths.values(): QA["source_hashes"][path.name] = sha(path)
    fig, axes = plt.subplots(3, 3, figsize=(183/25.4, 184/25.4), gridspec_kw={"height_ratios": [1., .95, .72]})
    for col, j in enumerate(DATA["representative_indices"].astype(int)):
        channel = CHANNELS[j]; role = ["selected SOZ", "selected non-SOZ", "unselected"][col]
        ax=axes[0,col]
        for s in SERIES:
            d=frames["trajectories"].query("node == @channel and series == @s")
            if len(d)!=256: raise RuntimeError("incomplete frozen trajectory source")
            ax.plot(d.time_s, d.standardized_amplitude, color=COLORS[s], lw=1.05 if s=="Actor + WGAN-GP" else .85, alpha=.85 if s=="Observed ictal" else 1)
        ax.set_title(f"{channel} · {role}", fontsize=8, fontweight="bold", loc="left")
        ax.set(xlim=(0,1), xticks=[0,.5,1], xlabel="Time (s)")
        if col==0: ax.set_ylabel("Standardized amplitude")
        panel_label(ax,"abc"[col])
        ax=axes[1,col]
        for s in SERIES:
            d=frames["densities"].query("node == @channel and series == @s")
            if len(d)!=300: raise RuntimeError("incomplete frozen density source")
            ax.plot(d.standardized_amplitude,d.density,color=COLORS[s],lw=1.05 if s=="Actor + WGAN-GP" else .85, ls=":" if s=="Free RC-SDE" else "-." if s=="Interictal reference" else "-")
            if s=="Interictal reference": ax.fill_between(d.standardized_amplitude,d.density,color=COLORS[s],alpha=.12)
        ax.text(.98,1.035,f"$W_1$\nUncontrolled → Reference {FREE_W1[j]:.3f}\nControlled → Reference {FULL_W1[j]:.3f}",transform=ax.transAxes,ha="right",va="bottom",fontsize=6.0)
        ax.set_xlabel("Standardized amplitude"); ax.set_yticks([])
        if col==0: ax.set_ylabel("Density")
        panel_label(ax,"def"[col])
        ax=axes[2,col]
        d=frames["controls"].query("node == @channel and series == 'Actor + WGAN-GP'")
        if len(d)!=256: raise RuntimeError("incomplete frozen input source")
        ax.plot(d.time_s,d.input_value,color=COLORS["Actor + WGAN-GP"],lw=1.05)
        ax.axhline(0,color="#C7C7C7",lw=.6,zorder=0)
        ax.set(xlim=(0,1),xticks=[0,.5,1],xlabel="Time (s)")
        if col==0: ax.set_ylabel("Control input")
        panel_label(ax,"ghi"[col])
    fig.legend(handles=legend_handles(),loc="upper center",bbox_to_anchor=(.5,.995),ncol=4,handlelength=2,columnspacing=1.25,fontsize=6.5)
    fig.suptitle("HUP060 run-02: representative empirical-law control under the Full policy",x=.08,y=1.035,ha="left",fontsize=9,fontweight="bold")
    fig.subplots_adjust(left=.08,right=.99,bottom=.075,top=.91,wspace=.32,hspace=.52)
    save(fig,"hup060_actor_wgan_mfc_preview")

def all_channels():
    path=Path(r"C:\Users\LiMing\Documents\改论文\github_release_taming_epilepsy\output\part3\source_data\figure_08\source_data_actor_wgan_all36_densities.csv")
    companion=path.parent.parent / "figures_06_08/paired_comparison.npz"
    if sha(companion)!=sha(DATA_PATH): raise RuntimeError("density source checkpoint binding differs")
    frame=pd.read_csv(path); QA["source_hashes"][path.name]=sha(path)
    fig, axes=plt.subplots(6,6,figsize=(183/25.4,190/25.4),sharex=True)
    clinical={"RPFa1","RPFa2","RPFa3","RPFb1","RPFc1"}
    for j,ax in enumerate(axes.ravel()):
        channel=CHANNELS[j]; max_density=0.
        for s in SERIES:
            source_label="Full policy" if s=="Actor + WGAN-GP" else s
            d=frame.query("channel == @channel and series == @source_label")
            if len(d)!=240: raise RuntimeError("incomplete all-contact density source")
            ax.plot(d.standardized_amplitude,d.density,color=COLORS[s],lw=.95 if s=="Actor + WGAN-GP" else .90 if s=="Interictal reference" else .75,ls=":" if s=="Free RC-SDE" else "-." if s=="Interictal reference" else "-",alpha=.78 if s=="Observed ictal" else .82 if s=="Free RC-SDE" else .92 if s=="Interictal reference" else 1)
            max_density=max(max_density,float(d.density.max()))
        color="#B64342" if channel in clinical else "#5B7FCA" if j in SELECTED else "#606060"
        ax.text(.02,.96,channel+("●" if j in SELECTED else ""),transform=ax.transAxes,ha="left",va="top",fontsize=5.8,fontweight="bold",color=color)
        ax.text(.98,.86,f"Uncontrolled → Reference {FREE_W1[j]:.3f}\nControlled → Reference {FULL_W1[j]:.3f}",transform=ax.transAxes,ha="right",va="top",fontsize=5.0)
        ax.set_ylim(0,max_density*1.50)
        ax.set_xlim(float(d.standardized_amplitude.min()),float(d.standardized_amplitude.max())); ax.set_yticks([])
        if j//6==5: ax.set_xlabel("Amplitude",fontsize=5.6)
        else: ax.tick_params(axis="x",labelbottom=False)
        ax.tick_params(axis="x",labelsize=5.2,length=2)
    fig.legend(handles=legend_handles(),loc="upper center",bbox_to_anchor=(.5,.988),ncol=4,fontsize=6,handlelength=2,columnspacing=1)
    fig.suptitle("HUP060 run-02: all-channel occupation laws under the Full policy",x=.06,y=1.015,ha="left",fontsize=8.8,fontweight="bold")
    fig.subplots_adjust(left=.045,right=.995,bottom=.06,top=.935,wspace=.12,hspace=.19)
    save(fig,"hup060_actor_wgan_all36_distribution_grid")

if __name__=="__main__":
    QA["source_hashes"][DATA_PATH.name]=sha(DATA_PATH)
    framework(); representative(); all_channels()
    (OUTPUT / "figure_revision_qa.json").write_text(json.dumps(QA,indent=2)+"\n",encoding="utf-8")
    print("Created three isolated figure revisions; frozen source arrays were not changed.")
