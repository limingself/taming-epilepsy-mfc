"""Reproduce all six panels of the Physica D red manuscript HUP060 Figure 7 from public summaries.

From the repository root, use --check-only to validate numerical endpoints or
--outdir NEW_DIRECTORY to export the figure in PDF, SVG and PNG. No raw iEEG,
simulation trajectories, model checkpoints or controller training are needed.
Pipeline slot 09 is the historical artifact identifier, not the figure number.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "output/part3/source_data/figure_09_trivial_baselines"
DEFAULT_OUT = ROOT / "output/part3/figure_09"
ARMS = ("Free", "Constant", "White", "P damping", "Tracking MPC", "Full")
STEM='HUP060_controller_baselines_mpc_complete'
INK='#263746'; FAINT='#dce1e5'; FULL='#2166a1'; MPC='#a95424'; GREY='#566977'; GREEN='#287565'
TIME='#436a8a'; OCC='#c88d57'; P='#995e4b'
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7.5,
                     'axes.labelsize':7.8,'xtick.labelsize':7.2,'ytick.labelsize':7.2,'legend.fontsize':7.0,
                     'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'hup060-complete-audit-v1',
                     'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'axes.edgecolor':GREY,
                     'text.color':INK,'axes.labelcolor':INK,'xtick.color':GREY,'ytick.color':GREY,
                     'xtick.major.size':2.8,'ytick.major.size':2.8,'legend.frameon':False,'axes.axisbelow':True})


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


def render(data, out, check_only=False):
    verified = numerical_summary(load_tables(data / 'physicad'))
    files={name: data/'physicad'/(name+'.csv') for name in ('controller_endpoints','damping_scan','diffusion_scan','initial_mismatch','pooled_width_errors','primary_contact_endpoints')}
    files.update({name:data/(name+'.csv') for name in ('trivial_baseline_summary','random_input_seed_metrics')})
    dfs={name:pd.read_csv(f) for name,f in files.items()}
    rows=dfs['controller_endpoints'].set_index('arm'); old=dfs['trivial_baseline_summary'].set_index('variant'); seeds=dfs['random_input_seed_metrics']
    order=['Free','Noise 0.2','Constant','White','P damping','Tracking MPC','Full']; legacy=['free','diffusion_reference_oracle','constant_input','white_input','linear_damping_rms_matched',None,'full_actor_wgan']
    metric=['time_w1','occupation_w1','command_rms']; metric_old=['mean_time_w1','mean_occupation_w1','control_rms']
    table=pd.DataFrame([rows.loc[name,metric].to_numpy() if name!='Noise 0.2' else old.loc[key,metric_old].to_numpy() for name,key in zip(order,legacy)],columns=metric,index=order,dtype=float)
    for name,key in zip(order,legacy):
        if key: np.testing.assert_allclose(table.loc[name], old.loc[key,metric_old].to_numpy(dtype=float), atol=1e-12,rtol=0)
    for name,key in [('Constant','constant_input'),('White','white_input')]:
        subset=seeds[seeds.variant==key]; assert len(subset)==32
        np.testing.assert_allclose(table.loc[name],subset[metric_old].median(),atol=1e-12,rtol=0)
    dc,sc,m,w,by=[dfs[n] for n in ('damping_scan','diffusion_scan','initial_mismatch','pooled_width_errors','primary_contact_endpoints')]; assert len(m)==len(w)==36 and len(dc)==27
    parts_names=['shapley_location_contribution','shapley_scale_contribution','shape_residual_contribution']
    np.testing.assert_allclose(m[parts_names].sum(axis=1),m.free_to_reference_w1,atol=1e-12,rtol=0)
    for arm,label in [('free','Free'),('full','Full'),('mpc','Tracking MPC')]:
        sub=by[by.arm==arm].set_index('channel').loc[w.contact]; np.testing.assert_allclose(sub.sd_gap,w[arm],atol=1e-12,rtol=0)
        np.testing.assert_allclose(sub[['time_w1','occ_w1']].mean(),table.loc[label,metric[:2]],atol=1e-12,rtol=0)
    parts=m[parts_names].mean().values; frac=parts[1]/parts.sum(); nwin=int((w.full<w.mpc).sum()); best=dc.loc[dc.mean_occupation_w1.idxmin()]; oracle=sc.loc[sc.mean_occupation_w1.idxmin()]; assert np.isclose(oracle.kappa_sigma,old.loc['diffusion_reference_oracle','kappa_sigma'])
    np.testing.assert_allclose(oracle.mean_occupation_w1,table.loc['Noise 0.2','occupation_w1'],atol=1e-12,rtol=0)
    if check_only:
        verified.update(panel_ids=list("abcdef"), n_noise=len(sc), n_damping=len(dc),
                        n_bars=2*len(table), data_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest()
                                                         for name, path in files.items()})
        print(json.dumps(verified, indent=2, allow_nan=False))
        return
    fig=plt.figure(figsize=(183/25.4,185/25.4),facecolor='white')
    l,r=.103,.595; ww=.370
    # Explicit row/column bounds leave room for normal-size labels and the two axes in d.
    a=fig.add_axes([l,.722,ww,.190]); b=fig.add_axes([r,.722,ww,.190])
    c=fig.add_axes([l,.395,ww,.185]); d1=fig.add_axes([r,.528,ww,.042]); d2=fig.add_axes([r,.375,ww,.075]);
    e=fig.add_axes([l,.081,ww,.178]); f=fig.add_axes([r,.081,ww,.178]);
    def heading(col,y,letter,name):
        fig.text(col-.085,y,letter,va='baseline',fontsize=11.2,fontweight='bold'); fig.text(col,y,name,va='baseline',fontsize=9.35,fontweight='bold')
    for col,y,letter,name in [(l,.948,'a','Distribution matching'),(r,.948,'b','Proportional damping'),(l,.617,'c','Noise sensitivity'),(r,.617,'d','Initial distribution mismatch'),(l,.296,'e','Matching distribution width'),(r,.296,'f','Distance and input cost')]:heading(col,y,letter,name)
    x=np.arange(len(table)); width=.35
    for side,col,clr,lab in [(-.5,'time_w1',TIME,r'Time-resolved $W_1$'),(.5,'occupation_w1',OCC,r'Occupation $W_1$')]:
        bars=a.bar(x+side*width,table[col],width,color=clr,label=lab,zorder=3)
        np.testing.assert_allclose([q.get_height() for q in bars],table[col],rtol=0,atol=0)
    a.set(xticks=x,xticklabels=['Free','Noise\n0.2','Constant','White','P\ndamping','Tracking\nMPC','Full'],ylim=(0,1.88),xlim=(-.55,6.55),yticks=[0,.5,1,1.5],ylabel=r'Mean contact-wise $W_1$')
    a.tick_params(axis='x',labelsize=6.7); a.legend(loc='upper right',ncol=2,fontsize=6.65,handlelength=.9,columnspacing=.8,handletextpad=.35,borderpad=.05);a.grid(axis='y',color=FAINT,lw=.45)
    def logaxes(ax):
        ax.set(xscale='linear',yscale='log',xlim=(-.05,1.82),ylim=(.036,26000),xticks=[0,.5,1,1.5],xlabel='Actuator-command RMS',ylabel=r'Occupation $W_1$')
        ax.yaxis.set_major_locator(FixedLocator([.1,1,10,100,1000,10000]));ax.yaxis.set_minor_locator(NullLocator());ax.grid(axis='y',color=FAINT,lw=.45)
    def marks(ax,name,color,symbol,label=None):
        row=table.loc[name]; ax.scatter(row.command_rms,row.occupation_w1,s=25,color=color,marker=symbol,edgecolor='white',lw=.5,zorder=5,label=label)
    def annotate(ax,text,xy,xytext,clr=GREY):ax.annotate(text,xy=xy,xytext=xytext,textcoords='data',fontsize=6.9,color=clr,ha='left',va='center',arrowprops={'arrowstyle':'-','color':clr,'lw':.6,'shrinkA':1,'shrinkB':2})
    logaxes(b); b.plot(dc.control_rms,dc.mean_occupation_w1,c=P,marker='.',ms=3.6,lw=.9,label='Damping sweep'); marks(b,'P damping',P,'D','RMS-matched damping'); marks(b,'Full',FULL,'o','Full');b.axvline(table.loc['Full','command_rms'],c=GREY,ls='--',lw=.65,zorder=1)
    b.legend(loc='upper left',fontsize=6.8,labelspacing=.3,handlelength=1.2,borderpad=.2,frameon=True,facecolor='white',edgecolor='none',framealpha=1)
    annotate(b,f'Best gain {best.gain:.2f}\nRMS {best.control_rms:.3f}',(best.control_rms,best.mean_occupation_w1),(.075,11))
    for key,clr,marker,lab in [('mean_occupation_w1',GREEN,'o','To preictal reference'),('mean_occupation_w1_to_observed_ictal',GREY,'s','To ictal recording')]:
        c.plot(sc.kappa_sigma,sc[key],color=clr,marker=marker,ms=3,lw=1.1,label=lab)
    c.set(xlim=(-.035,1.54),ylim=(0,.60),xticks=[0,.5,1,1.5],yticks=[0,.2,.4,.6],xlabel=r'Relative noise multiplier $\kappa_\sigma$',ylabel=r'Occupation $W_1$');c.grid(axis='y',color=FAINT,lw=.45)
    c.legend(loc='upper left',fontsize=6.9,handlelength=1.1);c.axvline(1,ls='--',color=GREY,lw=.6);c.axvline(oracle.kappa_sigma,ls=':',color=GREEN,lw=.8)
    c.text(.227,.190,'Post hoc 0.2',va='bottom',fontsize=6.3,color=GREEN);c.text(1.02,.05,'Frozen 1',fontsize=6.3,color=GREY)
    left=0
    for val,color,lab in zip(parts,['#80a9c5','#d4a06b','#b4bcc0'],['Location','Scale','Residual shape']):d1.barh([0],val,left=left,height=.62,color=color,label=lab);left+=val
    d1.set(ylim=(-.4,.4),xlim=(0,.325),xticks=[0,.1,.2,.3],yticks=[],xlabel=r'Contribution to free-to-reference $W_1$');d1.xaxis.label.set_size(7.1);d1.xaxis.labelpad=.5;d1.tick_params(axis='x',labelsize=6.7,pad=1.5);d1.spines['left'].set_visible(False)
    d1.text(parts[0]+.5*parts[1],0,f'Scale: {100*frac:.1f}%',va='center',ha='center',fontsize=7.4);d1.legend(loc='lower center',bbox_to_anchor=(.50,1.01),ncol=3,fontsize=6.5,borderpad=.08,handlelength=.8,handletextpad=.3,columnspacing=.6)
    rng=np.random.default_rng(20261109)
    for i,(key,color) in enumerate([('reference_to_observed_sd_ratio',INK),('reference_to_free_sd_ratio','#85929b')]):
        v=m[key].to_numpy();d2.scatter(i+rng.uniform(-.08,.08,len(v)),v,s=9,color=color,alpha=.58,edgecolor='none');q1,med,q3=np.quantile(v,[.25,.5,.75]);d2.vlines(i,q1,q3,color=INK,lw=1.8);d2.hlines(med,i-.16,i+.16,color=INK,lw=1.2)
    d2.set(xlim=(-.45,1.45),ylim=(0,max(1.1,m[['reference_to_observed_sd_ratio','reference_to_free_sd_ratio']].max().max()*1.08)),xticks=[0,1],xticklabels=['Reference / recorded','Reference / free model'],yticks=[0,.5,1],ylabel='SD ratio');d2.tick_params(axis='x',labelsize=6.9);d2.axhline(1,color=GREY,ls='--',lw=.7,zorder=0);d2.grid(axis='y',color=FAINT,lw=.45)
    maxval=.328
    e.fill_between([0,maxval],[0,0],[0,maxval],color='#edf4f8',zorder=0);e.plot([0,maxval],[0,maxval],ls='--',lw=.8,color='#96a2a9');e.scatter(w.mpc,w.full,s=17,c=GREY,edgecolor='white',lw=.4,zorder=3)
    e.set(aspect='equal',xlim=(-.01,maxval),ylim=(-.01,maxval),xticks=[0,.1,.2,.3],yticks=[0,.1,.2,.3],xlabel='Tracking MPC: absolute SD error',ylabel='Full: absolute SD error')
    for y,label,clr in [(.96,f'{nwin}/{len(w)} below the line',INK),(.83,'Median SD error',GREY),(.73,f'Full  {w.full.median():.3f}',FULL),(.63,f'MPC  {w.mpc.median():.3f}',MPC)]:e.text(.035,y,label,color=clr,fontsize=6.5,ha='left',va='top',transform=e.transAxes)
    logaxes(f);f.plot(dc.control_rms,dc.mean_occupation_w1,color='#87959d',lw=1,marker='.',ms=3.4)
    for name,col,mk,txt,tx in [('Full',FULL,'o','Full',(.94,.082)),('Tracking MPC',MPC,'s','MPC',(.30,.055)),('P damping',GREY,'D','Matched P',(.53,9))]:
        marks(f,name,col,mk); z=table.loc[name];annotate(f,txt,(z.command_rms,z.occupation_w1),tx,col)
    annotate(f,'Best P',(best.control_rms,best.mean_occupation_w1),(.13,1));f.text(.035,.92,'Scalar damping sweep',transform=f.transAxes,color=GREY,fontsize=6.9,va='top')
    fig.canvas.draw();rend=fig.canvas.get_renderer(); boxes=[(i,ax.get_tightbbox(rend)) for i,ax in enumerate(fig.axes)];
    assert all(fig.bbox.contains(bb.x0,bb.y0) and fig.bbox.contains(bb.x1,bb.y1) for _,bb in boxes)
    for i,bi in boxes:
        for j,bj in boxes[i+1:]:
            assert Bbox.intersection(bi,bj) is None, f'Overlapping axes text or legend: {i}, {j}'
    out.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','svg','png']:
        meta={'CreationDate':None,'ModDate':None} if ext=='pdf' else ({'Date':None} if ext=='svg' else {})
        fig.savefig(out/(STEM+'.'+ext),dpi=260,metadata=meta)
    table.to_csv(out/(STEM+'_panel_a_values.csv'),index_label='arm')
    qa={'panel_ids':list('abcdef'),'n_axes':len(fig.axes),'n_contacts':len(w),'n_damping':len(dc),'n_noise':len(sc),'n_bars':len(a.patches),'scale_percent':100*frac,'width_wins':nwin,'median_full_sd_error':w.full.median(),'median_mpc_sd_error':w.mpc.median(),'data_sha256':{n:hashlib.sha256(f.read_bytes()).hexdigest() for n,f in files.items()}};
    (out/(STEM+'_qa.json')).write_text(json.dumps(qa,indent=2),encoding='utf-8');plt.close(fig);print(json.dumps(qa,indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", "--data", type=Path, default=DEFAULT_DATA,
                        help="Source-data folder, or its physicad subfolder")
    parser.add_argument("--outdir", "--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check-only", action="store_true",
                        help="Check source tables and report metrics without writing files")
    args = parser.parse_args(argv)
    data = args.data_dir.parent if args.data_dir.name == "physicad" else args.data_dir
    if not args.check_only:
        outputs = [STEM + suffix for suffix in (".pdf", ".png", ".svg", "_qa.json", "_panel_a_values.csv")]
        if any((args.outdir / name).exists() for name in outputs):
            parser.error("An output already exists. Choose a fresh --outdir; frozen figures are not overwritten.")
    render(data, args.outdir, args.check_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
