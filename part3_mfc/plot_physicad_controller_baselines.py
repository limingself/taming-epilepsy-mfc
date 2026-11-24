"""Reproduce Physica D Fig. 7 and Fig. S10 from released summary tables.

From the repository root:
    python part3_mfc/plot_physicad_controller_baselines.py --check-only
    python part3_mfc/plot_physicad_controller_baselines.py --outdir ./figures

This is figure reproduction; no raw iEEG, surrogate fitting or actor training
is required. "figure_09" is the historical export namespace, not the current
manuscript figure number.
"""
import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "output/part3/source_data/figure_09_trivial_baselines/physicad"
DEFAULT_OUT = ROOT / "output/part3/figure_09"
ARMS = ("Free", "Constant", "White", "P damping", "Tracking MPC", "Full")

INK="#29343D"; GRAY="#566873"; LIGHT="#D6DBDD"
FULL="#2069A1"; MPC="#A95623"; OCC="#C48548"; TIME="#446985"; REF="#287865"
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
    "font.size":8.3,"axes.labelsize":8.5,"axes.titlesize":10.25,"xtick.labelsize":8,"ytick.labelsize":8,
    "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.75,
    "axes.edgecolor":GRAY,"axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,
    "xtick.major.size":3,"ytick.major.size":3,"legend.frameon":False,"legend.fontsize":7.6,
    "pdf.fonttype":42,"svg.fonttype":"none","svg.hashsalt":"HUP060-Fig7-MPC","mathtext.fontset":"dejavusans"})


def load_tables(path):
    specs = [
        ("controller_endpoints", "arm time_w1 occupation_w1 command_rms"),
        ("pooled_width_errors", "contact free full mpc"),
        ("damping_scan", "gain control_rms mean_occupation_w1"),
        ("diffusion_scan", "kappa_sigma mean_occupation_w1 mean_occupation_w1_to_observed_ictal"),
        ("initial_mismatch", "channel free_to_reference_w1 shapley_location_contribution shapley_scale_contribution shape_residual_contribution reference_to_observed_sd_ratio reference_to_free_sd_ratio"),
    ]
    tables = []
    for name, header in specs:
        frame = pd.read_csv(path / (name + ".csv"))
        fields = header.split()
        if frame.columns.tolist() != fields or frame.empty or frame[fields[0]].duplicated().any():
            raise ValueError(f"Wrong schema, empty table or duplicate keys: {name}")
        values = frame.iloc[:, int(name in {"controller_endpoints", "pooled_width_errors", "initial_mismatch"}):]
        if frame.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Missing/non-finite data: {name}")
        tables.append(frame)
    rows, w, dc, sc, m = tables
    if tuple(rows["arm"]) != ARMS:
        raise ValueError(f"Expected six controller arms in order: {ARMS}")
    if len(w) != 36 or set(w.contact) != set(m.channel):
        raise ValueError("Width and decomposition tables must represent the same 36 contacts")
    if not dc.control_rms.is_monotonic_increasing or not sc.kappa_sigma.is_monotonic_increasing:
        raise ValueError("Sort the sweeps by command RMS and relative noise before rendering")
    rows = rows.set_index("arm")
    parts = ["shapley_location_contribution", "shapley_scale_contribution", "shape_residual_contribution"]
    np.testing.assert_allclose(m[parts].sum(axis=1), m.free_to_reference_w1, atol=1e-12, rtol=0)
    free = rows.loc["Free", "occupation_w1"]
    np.testing.assert_allclose(
        [m.free_to_reference_w1.mean(), dc.set_index("gain").loc[0, "mean_occupation_w1"],
         sc.set_index("kappa_sigma").loc[1, "mean_occupation_w1"]], free, atol=1e-12, rtol=0)
    matched = rows.loc["P damping", ["command_rms", "occupation_w1"]].to_numpy()
    sweep = dc[["control_rms", "mean_occupation_w1"]].to_numpy()
    if not np.isclose(sweep, matched, atol=1e-12, rtol=0).all(axis=1).any():
        raise ValueError("The reported matched damping point is absent from the gain sweep")
    by_contact = pd.read_csv(path / "primary_contact_endpoints.csv")
    metrics = ["time_w1", "occ_w1", "sd_gap", "mean_gap"]
    arm_names = {"free": "Free", "full": "Full", "mpc": "Tracking MPC"}
    if (by_contact.columns.tolist() != ["arm", "channel", *metrics]
            or by_contact.duplicated(["arm", "channel"]).any()
            or set(by_contact.arm) != set(arm_names)):
        raise ValueError("Expected four primary endpoints for each arm/contact pair")
    data = by_contact[metrics].to_numpy(dtype=float)
    if not np.isfinite(data).all() or (data < 0).any():
        raise ValueError("Primary distances and absolute gaps must be finite and nonnegative")
    for key, label in arm_names.items():
        sub = by_contact.loc[by_contact.arm == key].set_index("channel")
        if set(sub.index) != set(w.contact):
            raise ValueError(f"Primary endpoint and width tables disagree on contacts: {key}")
        np.testing.assert_allclose(sub[["time_w1", "occ_w1"]].mean(),
                                   rows.loc[label, ["time_w1", "occupation_w1"]], rtol=0, atol=1e-12)
        np.testing.assert_allclose(sub.loc[w.contact, "sd_gap"], w[key], rtol=0, atol=1e-12)
    return rows, w, dc, sc, m, by_contact


def numerical_summary(tables):
    rows, w, dc, sc, m, by_contact = tables
    wide = by_contact.pivot(index="channel", columns="arm", values=["time_w1", "occ_w1"])
    coverage = {
        key: int(((wide["time_w1"][key] < wide["time_w1"]["free"]) &
                  (wide["occ_w1"][key] < wide["occ_w1"]["free"])).sum())
        for key in ("full", "mpc")
    }
    return {
        "controllers": rows.loc[["Free", "Full", "Tracking MPC"]].to_dict(orient="index"),
        "contact_mean_W1_endpoints": by_contact.groupby("arm")[["time_w1", "occ_w1"]].mean().to_dict(orient="index"),
        "pooled_absolute_mean_gap_medians": by_contact.groupby("arm").mean_gap.median().to_dict(),
        "both_W1_endpoints_strictly_lower_than_free": coverage,
        "pooled_width_error_medians": w[["free", "full", "mpc"]].median().to_dict(),
        "full_closer_in_width_than_mpc": int((w.full < w.mpc).sum()),
        "contacts": len(w),
        "minimum_damping_occupation_w1": float(dc.mean_occupation_w1.min()),
        "reference_best_noise_multiplier_in_scan": float(sc.loc[sc.mean_occupation_w1.idxmin(), "kappa_sigma"]),
        "initial_scale_fraction": float(m.shapley_scale_contribution.sum() / m.free_to_reference_w1.sum()),
    }


def panel(ax,ch,title):
    ax.set_title(title,loc="left",fontweight="bold",pad=8)
    ax.text(-.14 if ch!="a" else -.072,1.055,ch,transform=ax.transAxes,
            ha="left",va="bottom",fontsize=11.2,weight="bold")
def save(fig, stem, outdir):
    for extension in ("png","svg","pdf"):
        kw={"metadata":{"CreationDate":None,"ModDate":None}} if extension=="pdf" else ({"metadata": {"Date": None}} if extension == "svg" else {})
        fig.savefig(outdir/f"{stem}.{extension}",dpi=260,**kw)
    plt.close(fig)
def figures(tables, outdir):
    """Render the original manuscript panels from validated tables."""
    rows, w, dc, sc, m, _ = tables
    outdir.mkdir(parents=True, exist_ok=True)
    fig=plt.figure(figsize=(7.17,4.80))
    a=fig.add_axes([.107,.625,.857,.279]); b=fig.add_axes([.14,.12,.30,.345]); c=fig.add_axes([.588,.12,.375,.345])
    panel(a,"a","Distribution matching on the frozen model")
    pos=np.arange(len(rows)); z=.292
    for dx,col,color,lab in ((-.5,"time_w1",TIME,r"Time-resolved $W_1$"),(.5,"occupation_w1",OCC,r"Occupation $W_1$")):
        a.bar(pos+dx*z,rows[col],z,color=color,label=lab,zorder=3)
    a.set(xticks=pos,xticklabels=rows.index,xlim=(-.5,5.5),yticks=[0,.5,1,1.5],ylim=(0,1.58),ylabel=r"Mean contact-wise $W_1$")
    a.set_axisbelow(True); a.grid(axis="y",color=LIGHT,lw=.5)
    a.legend(loc="upper right",fontsize=8.1,handlelength=1.0)
    panel(b,"b","Matching distribution width")
    maxval=.328
    b.fill_between([0,maxval],[0,0],[0,maxval],color="#EDF4F8",zorder=0)
    b.plot([0,maxval],[0,maxval],c="#96A2A9",ls="--",lw=.85)
    b.scatter(w.mpc,w.full,s=20,color=GRAY,edgecolor="white",lw=.4,zorder=3)
    b.set_aspect("equal",adjustable="box")
    b.set(xlim=(-.01,maxval),ylim=(-.01,maxval),xlabel="Tracking MPC: absolute SD error",ylabel="Full: absolute SD error",
        xticks=[0,.1,.2,.3],yticks=[0,.1,.2,.3])
    b.text(.045,.95,f"{sum(w.full<w.mpc)}/{len(w)} below the line",transform=b.transAxes,fontsize=7.5,va="top")
    b.text(.045,.84,"Median SD error",transform=b.transAxes,fontsize=7.25,va="top",color=GRAY)
    b.text(.045,.75,f"Full  {w.full.median():.3f}",transform=b.transAxes,fontsize=7.25,va="top",color=FULL)
    b.text(.045,.66,f"MPC  {w.mpc.median():.3f}",transform=b.transAxes,fontsize=7.25,va="top",color=MPC)
    panel(c,"c","Distance and input cost")
    c.plot(dc.control_rms,dc.mean_occupation_w1,color="#88939B",lw=1,marker=".",ms=3.2)
    c.set(yscale="log",xlim=(-.045,1.82),ylim=(.032,26000),xlabel="Actuator-command RMS",ylabel=r"Occupation $W_1$")
    c.set_xticks([0,.5,1,1.5]); c.yaxis.set_major_locator(FixedLocator([.1,1,10,100,1000,10000]))
    c.yaxis.set_minor_locator(NullLocator()); c.grid(axis="y",which="major",color=LIGHT,lw=.45,zorder=0)
    def marker(name,col,mk,tx,ty,word):
        x,y=rows.loc[name,["command_rms","occupation_w1"]]
        c.scatter(x,y,s=27,c=col,marker=mk,edgecolor="white",lw=.5,zorder=5)
        c.annotate(word,xy=(x,y),xytext=(tx,ty),textcoords="data",ha="center",va="center",fontsize=7.5,
            color=col,arrowprops=dict(arrowstyle="-",color=col,lw=.6,shrinkA=1,shrinkB=2))
    marker("P damping","#53636C","D",.54,8,"Matched P")
    marker("Tracking MPC",MPC,"s",.30,.051,"MPC")
    marker("Full",FULL,"o",.95,.055,"Full")
    best=dc.loc[dc.mean_occupation_w1.idxmin()]
    c.annotate("Best P",xy=(best.control_rms,best.mean_occupation_w1),xytext=(.14,.9),fontsize=7,color=GRAY,
        ha="left",arrowprops=dict(arrowstyle="-",color=GRAY,lw=.6,shrinkB=2))
    c.text(.036,.872,"Scalar damping sweep",transform=c.transAxes,ha="left",va="top",fontsize=7.45,color=GRAY)
    save(fig, "HUP060_controller_baselines_mpc", outdir)
    fig=plt.figure(figsize=(7.17,3.30))
    a=fig.add_axes([.101,.201,.36,.59]); b1=fig.add_axes([.613,.676,.35,.137]); b2=fig.add_axes([.613,.185,.35,.289])
    fig.text(.044,.895,"a",fontsize=11.2,fontweight="bold")
    fig.text(.101,.895,"Noise sensitivity of the free model",fontsize=10.25,fontweight="bold")
    for col,colr,mk,lab in (("mean_occupation_w1",REF,"o","To preictal reference"),("mean_occupation_w1_to_observed_ictal",GRAY,"s","To ictal recording")):
        a.plot(sc.kappa_sigma,sc[col],c=colr,marker=mk,ms=3.0,lw=1.15,label=lab)
    a.axvline(1,color="#96A1A7",ls="--",lw=.65,zorder=0)
    i=sc.mean_occupation_w1.idxmin(); a.axvline(sc.loc[i,"kappa_sigma"],color=REF,ls=":",lw=.75,zorder=0)
    a.set(xlim=(-.035,1.54),ylim=(0,.60),xticks=[0,.5,1,1.5],yticks=[0,.2,.4,.6],
          xlabel=r"Relative noise multiplier $\kappa_\sigma$",ylabel=r"Occupation $W_1$")
    a.legend(loc="upper left",handlelength=1.4,fontsize=7.6)
    a.set_axisbelow(True); a.grid(axis="y",color=LIGHT,lw=.5)
    fig.text(.547,.895,"b",fontsize=11.2,fontweight="bold")
    fig.text(.613,.895,"Initial mismatch",fontsize=10.25,fontweight="bold")
    parts=m[["shapley_location_contribution","shapley_scale_contribution","shape_residual_contribution"]].mean().values
    total=parts.sum();left=0
    for value,color,label in zip(parts,["#719BB6","#D6A16D","#A6AEB1"],["Location","Scale","Shape"]):
        b1.barh(0,value,left=left,height=.64,color=color,label=label);left+=value
    b1.set(xlim=(0,.325),xticks=(0,.1,.2,.3),ylim=(-.47,.47),yticks=[],xlabel=r"Contribution to free-to-reference $W_1$")
    b1.set_xlabel(r"Contribution to free-to-reference $W_1$",fontsize=7.5,labelpad=1.8)
    b1.spines["left"].set_visible(False)
    b1.text(parts[0]+.5*parts[1],0,f"Scale: {100*parts[1]/total:.1f}%",ha="center",va="center",fontsize=7.7)
    # Place a single minimal key immediately above the stack.
    b1.legend(loc="lower center",bbox_to_anchor=(.5,.96),ncol=3,fontsize=6.8,handlelength=.85,handletextpad=.3,columnspacing=.8,borderpad=.1)
    rng=np.random.default_rng(20261109)
    for k,(col,clr) in enumerate((("reference_to_observed_sd_ratio","#465965"),("reference_to_free_sd_ratio","#83949B"))):
        v=m[col].to_numpy(); j=rng.uniform(-.09,.09,len(v));b2.scatter(k+j,v,s=8,c=clr,alpha=.6,edgecolor="none")
        q1,med,q3=np.quantile(v,[.25,.5,.75]); b2.vlines(k,q1,q3,color=INK,lw=1.8);b2.hlines(med,k-.15,k+.15,color=INK,lw=1.2)
    b2.axhline(1,ls="--",color="#839097",lw=.7,zorder=0)
    b2.set(xlim=(-.45,1.45),ylim=(0,max(1.1,m[["reference_to_observed_sd_ratio","reference_to_free_sd_ratio"]].max().max()*1.07)),
       ylabel="SD ratio",xticks=[0,1],xticklabels=["Reference /\nrecorded ictal","Reference /\nfree model"],yticks=[0,.5,1])
    save(fig, "HUP060_noise_and_mismatch", outdir)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA, help="Folder containing the plot and contact-level validation tables")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT, help="Folder for the PDF, PNG and SVG pairs")
    parser.add_argument("--check-only", action="store_true", help="Validate the source tables and print their numerical summaries without plotting")
    opts = parser.parse_args(argv)
    tables = load_tables(opts.data_dir)
    print(json.dumps(numerical_summary(tables), indent=2, allow_nan=False))
    if not opts.check_only:
        stems = ("HUP060_controller_baselines_mpc", "HUP060_noise_and_mismatch")
        if any((opts.outdir / (stem + "." + ext)).exists()
               for stem in stems for ext in ("pdf", "png", "svg")):
            parser.error("An export already exists. Choose a fresh --outdir; frozen figures are not overwritten.")
        figures(tables, opts.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
