"""Redraw manuscript Fig. 3 (historical export ID 05) from two digestible CSV tables."""
from pathlib import Path
import os
import csv
import argparse

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/part2/source_data/figure_05"
OUTPUT = ROOT / "output/part2/figure_05"
for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[k] = "1"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    "font.family":"sans-serif", "font.sans-serif":["Arial","DejaVu Sans"],
    "font.size":8, "axes.spines.right":False, "axes.spines.top":False,
    "axes.linewidth":0.65, "xtick.major.width":0.65, "ytick.major.width":0.65,
    "pdf.fonttype":42, "svg.fonttype":"none", "legend.frameon":False,
})
METHODS = ("state","delay","graph","full")
STYLES = {
    "observed": ("Recorded seizure", "#202020","-",1.5),
    "state": ("State only", "#777777",(0,(4,2,1,2)),1.25),
    "delay": ("State + delay", "#378788",(0,(3,2)),1.25),
    "graph": ("State + delay + graph", "#B8662F",(0,(6,2)),1.3),
    "full": ("State + delay + graph + RC", "#225F9B","-",1.45),
}
def legend(f, y):
    handles=[Line2D([], [], color=col, ls=sty, lw=lw,label=label)
             for label,col,sty,lw in STYLES.values()]
    f.legend(handles=handles,loc="upper center",bbox_to_anchor=(0.52,y),
             ncol=3,fontsize=8,labelspacing=.45,handlelength=2.5,columnspacing=1.6)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=SOURCE)
    parser.add_argument("--out",type=Path,default=OUTPUT)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    def rows(name):
        with (args.source/name).open(encoding="utf-8-sig",newline="") as fp:
            return list(csv.DictReader(fp))
    table=rows("fixed_context_ablation_metrics.csv")
    chs=[r["channel"] for r in table]
    scores={k:[float(r[k]) for r in table] for k in METHODS}
    assert len(chs)==len(set(chs))==36
    densities=rows("fixed_context_ablation_density_curves.csv")
    curves={}
    for ch in ("RPFa3","RA3","RAFa4"):
        group=[r for r in densities if r["channel"]==ch]
        assert len(group)==450
        curves[ch]=([float(r["amplitude"]) for r in group],{k:[float(r[k]) for r in group] for k in STYLES})
    def density(ax,ch):
        x,arrays=curves[ch]
        # Recorded trace last so a near-perfect fit does not hide the reference.
        for k in (*METHODS,"observed"):
            _,col,sty,lw=STYLES[k]
            ax.plot(x,arrays[k],color=col,ls=sty,lw=lw)
        ax.set_xlim(x[0],x[-1]); ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4,steps=[1,2,2.5,5,10]))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    order=("RPFa3","RA3","RAFa4")
    fig,axes=plt.subplots(2,3,figsize=(170/25.4,150/25.4))
    fig.subplots_adjust(left=.098,right=.986,top=.808,bottom=.105,wspace=.37,hspace=.66)
    fig.text(.075,.978,"HUP060: predictive feature comparison",fontweight="bold",fontsize=11,va="top")
    legend(fig,.929)
    smax=max(scores[k][chs.index(ch)] for k in METHODS for ch in order)*1.23
    for i,ch in enumerate(order):
        j=chs.index(ch); a,b=axes[:,i]
        for r,ax in enumerate((a,b)):
            ax.set_title(ch,loc="left",fontweight="bold",fontsize=9,pad=4)
            ax.text(-.17,1.025,chr(97+r*3+i),transform=ax.transAxes,
                    fontweight="bold",fontsize=10,va="bottom")
        density(a,ch)
        a.set_xlabel("Standardized amplitude")
        if i==0: a.set_ylabel("Density")
        vals=[scores[k][j] for k in METHODS]
        b.set_axisbelow(True); b.grid(axis="y",color="#E4E8EB",lw=.55)
        b.bar(range(4),vals,width=.64,color=[STYLES[k][1] for k in METHODS])
        for ci,v in enumerate(vals):
            b.text(ci,v+smax*.025,f"{v:.3f}",ha="center",fontsize=6.8)
        b.set_xticks(range(4),["State","+Delay","+Graph","+RC"],fontsize=7.1)
        b.set_xlim(-.58,3.58); b.set_ylim(0,smax)
        b.yaxis.set_major_locator(MaxNLocator(4))
        if i==0: b.set_ylabel("Occupation $W_1$")
    for ext in ("svg","pdf","png"):
        fig.savefig(args.out/f"hup060_context3_1s_free_state_delay_graph_ablation.{ext}",dpi=240,facecolor="white")
    fig.savefig(args.out/"hup060_context3_1s_free_state_delay_graph_ablation.tiff",dpi=600,facecolor="white",pil_kwargs={"compression":"tiff_lzw"})
    plt.close(fig)

if __name__=="__main__":
    main()
