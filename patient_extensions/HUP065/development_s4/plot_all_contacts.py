"""Four distributions, paired HUP065 development prediction/control preview."""
import argparse, csv, hashlib, json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wasserstein_distance as wd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family":"DejaVu Sans","svg.fonttype":"none","pdf.fonttype":42,
                     "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.55,
                     "font.size":8,"legend.frameon":False})
SER=["observed","free","reference","controlled"]
NAMES=["Recorded ictal","Free prediction","Preictal reference","Controlled"]
COL=["#272727","#9A9A9A","#3C8D62","#2166AC"]
STY=["-",(0,(3.2,1.6)),(0,(5,1.4,1,1.4)),"-"]
LW=[1.00,1.18,1.15,1.30]

def get(z, *names):
    return np.asarray(z[next(k for k in names if k in z)])

def timeline(a,b):
    # Exact equally weighted empirical W1 at each time t/contact c for unequal
    # numbers of particles, using merged order-statistic probability intervals.
    a,b=np.sort(a,axis=0),np.sort(b,axis=0)
    n,m=len(a),len(b)
    splits=np.unique(np.r_[np.arange(n+1)/n,np.arange(m+1)/m])
    mid=(splits[1:]+splits[:-1])/2
    return np.sum(np.diff(splits)[:,None,None]*abs(a[(mid*n).astype(int)]-b[(mid*m).astype(int)]),axis=0).mean(0)

def preview(file,dest,anchor=None,metrics_dir=None):
    with np.load(file,allow_pickle=False) as z:
        channels=get(z,"channels").astype(str).tolist()
        selected=get(z,"selected","selected_indices","actuators").astype(int).ravel()
        raw={"observed":get(z,"observed","observed_scaled"),
             "free":get(z,"free","no_graph","uncontrolled","uncontrolled_scaled"),
             "reference":get(z,"reference","reference_eval","reference_validation_scaled","reference_scaled"),
             "controlled":get(z,"controlled","candidate_controlled_scaled","controlled_scaled")}
        commands=get(z,"commands","command") if any(k in z for k in ("commands","command")) else None
        controls=get(z,"controls") if "controls" in z else None
    assert len(channels)==len(set(channels))==64
    if anchor is not None:
        with np.load(anchor,allow_pickle=False) as z:
            np.testing.assert_array_equal(channels,z["channels"].astype(str))
            np.testing.assert_array_equal(raw["observed"],z["observed"])
            np.testing.assert_array_equal(raw["free"],z["no_graph"])
    assert raw["free"].shape==raw["controlled"].shape==(32,256,64)
    assert raw["reference"].ndim==3 and raw["reference"].shape[1:]==(256,64)
    assert len(selected)==len(set(selected))==32 and min(selected)>=0 and max(selected)<64
    assert all(np.isfinite(a).all() for a in raw.values())
    # Fixed historical display window. Densities are CONDITIONAL on this window.
    lo,hi=-6.309818911354839,5.201366495203709
    sc=2/(hi-lo);edges=np.linspace(-1,1,201);dx=edges[1]-edges[0];grid=np.r_[-1.,(edges[:-1]+edges[1:])/2,1.]
    tm={k:timeline(raw[k],raw["reference"]) for k in ("controlled","free")}
    # Check a time-wise channel against scipy rather than interpolated quantiles.
    np.testing.assert_allclose(tm["controlled"][0],np.mean([wd(raw["controlled"][:,t,0],raw["reference"][:,t,0])
        for t in range(256)]),rtol=1e-12,atol=1e-12)
    mat={k:v.reshape(-1,64) for k,v in raw.items()}
    curves=[];metrics=[];maxerr=0.
    for j,name in enumerate(channels):
        r={"channel":name,"selected":j in selected}
        v={k:a[:,j] for k,a in mat.items()}
        vv={k:a[(a>=lo)&(a<=hi)] for k,a in v.items()}
        u={k:(a-lo)*sc-1 for k,a in vv.items()}
        bw=max(np.std(u["observed"],ddof=1)*len(u["observed"])**(-.2),dx)
        cs=[]
        for k in SER:
            if k!="reference":
                r[k+"_W1_full_z"]=float(wd(v[k],v["reference"]))
                r[k+"_W1_visible_u"]=float(wd(u[k],u["reference"])) if len(u[k]) and len(u["reference"]) else None
            if k in tm:r[k+"_W1_time_z"]=float(tm[k][j])
            r[k+"_fraction_visible"]=len(u[k])/len(v[k])
            h=np.histogram(u[k],edges)[0]
            assert h.sum()==len(u[k])
            if h.sum():
                d=gaussian_filter1d(h.astype(float),bw/dx,mode="reflect")/(h.sum()*dx)
                y=np.r_[d[0],d,d[-1]]; maxerr=max(maxerr,abs(np.trapezoid(y,grid)-1))
            else:y=np.full_like(grid,np.nan)
            cs.append(y)
        curves.append(cs);metrics.append(r)
    assert maxerr<1e-9
    curves=np.array(curves)
    if metrics_dir is not None:
        metrics_dir.mkdir(exist_ok=True,parents=True)
        with (metrics_dir/"channel_distances.csv").open("w",newline="",encoding="utf-8-sig") as f:
            wr=csv.DictWriter(f,fieldnames=metrics[0]);wr.writeheader();wr.writerows(metrics)
    stat={"source_sha256":hashlib.sha256(file.read_bytes()).hexdigest(),"area_error":maxerr,
          "n_reference_windows":len(raw["reference"]),"input_count":len(selected)}
    for metric in ("W1_full_z","W1_time_z","W1_visible_u"):
        rows=[r for r in metrics if all(r[k+"_"+metric] is not None for k in ["free","controlled"])]
        group={}
        for key,rs in (("all",rows),("direct",[r for r in rows if r["selected"]]),("indirect",[r for r in rows if not r["selected"]])):
            if not rs: continue
            names=["free","controlled"]+(["observed"] if metric!="W1_time_z" else [])
            z={k:np.mean([r[k+"_"+metric] for r in rs]) for k in names}
            z["improved"]=sum(r["controlled_"+metric]<r["free_"+metric] for r in rs)
            if "observed" in names:
                z["controlled_closer_than_observed"]=sum(r["controlled_"+metric]<r["observed_"+metric] for r in rs)
            z["n"]=len(rs);group[key]=z
        stat[metric]=group
    for k in SER:
        stat[k+"_outside_pct"]=100*np.mean((mat[k]<lo)|(mat[k]>hi))
        stat[k+"_max_abs_z"]=float(np.abs(mat[k]).max())
    if commands is not None:stat.update(command_rms=float(np.sqrt(np.square(commands).mean())),command_absmax=float(abs(commands).max()))
    if controls is not None:
        assert controls.shape==(32,256,32) and np.isfinite(controls).all()
        peak=float(abs(controls).max()); ar=float(np.sqrt((controls**2).mean((0,1))).max())
        energy=float(np.sum(controls**2,axis=-1).mean()); saturation=float(np.mean(abs(controls)>=.95*1.8))
        stat["input_budget"]={"rms":float(np.sqrt((controls**2).mean())),"max_actuator_rms":ar,
          "energy":energy,"peak":peak,"sat_fraction":saturation,
          "pass":ar<=.405+1e-9 and energy<=3.7908+1e-9 and peak<=1.8+1e-9 and saturation<.01}
    if metrics_dir is not None: (metrics_dir/"plot_qa.json").write_text(json.dumps(stat,indent=2),encoding="utf-8")
    def draw(ids,cols,tag):
        rows=len(ids)//cols
        fig,axs=plt.subplots(rows,cols,figsize=(10.8,8.9) if cols==8 else (7.3,8.15),squeeze=False)
        fig.subplots_adjust(left=.065,right=.987,top=.895,bottom=.068,wspace=.15,hspace=.22)
        fig.suptitle("HUP065 run01 development window  |  Free prediction and control",y=.986,weight="bold",fontsize=12 if cols<8 else 15)
        fig.legend(handles=[Line2D([],[],c=c,lw=1.5,ls=sty,label=n) for c,sty,n in zip(COL,STY,NAMES)],
                   loc="upper center",bbox_to_anchor=(.52,.952),ncol=4,fontsize=8,handlelength=2.35,columnspacing=1.4)
        for i,j in enumerate(ids):
            ax=axs.flat[i]; cs=curves[j]
            for k in (0,1,2,3):
                ax.plot(grid,cs[k],color=COL[k],ls=STY[k],lw=LW[k],zorder=[2,4,5,3][k])
                if not np.isfinite(cs[k]).any():ax.text(.98,.89-.15*k,NAMES[k]+": none",transform=ax.transAxes,ha="right",fontsize=5.5,color=COL[k])
            ax.text(.035,.94,channels[j].replace("EEG ","").replace("-Ref","").replace(" ",""),ha="left",va="top",fontsize=7.3,
                    weight="bold",transform=ax.transAxes)
            ax.set(xlim=(-1,1),ylim=(0,np.nanmax(cs)*1.15),xticks=[-1,0,1],yticks=[])
            ax.tick_params(axis="x",length=2,labelbottom=i//cols==rows-1,labelsize=7,pad=2)
        fig.supylabel("Within-window probability density",x=.015,fontsize=9)
        fig.supxlabel("Normalized amplitude",y=.018,fontsize=9)
        fig.savefig(dest/(tag+".png"),dpi=175 if cols==8 else 210)
        fig.savefig(dest/(tag+".pdf"),metadata={"CreationDate":None,"ModDate":None});plt.close(fig)
    draw(range(64),8,"HUP065_run01_free_control")
    return stat

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("file",type=Path)
    ap.add_argument("--out",type=Path,required=True,help="Destination for the published PDF and PNG")
    ap.add_argument("--anchor",type=Path,help="Optional independent free-rollout array for an exact replay check")
    ap.add_argument("--metrics-dir",type=Path,help="Optional private diagnostic directory (full-range and within-window metrics)")
    args=ap.parse_args();args.out.mkdir(exist_ok=True,parents=True)
    result=preview(args.file,args.out,args.anchor,args.metrics_dir)
    print(json.dumps(result["W1_visible_u"],indent=2))
