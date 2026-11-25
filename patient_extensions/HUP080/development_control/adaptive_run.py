"""Preregistered exploratory causal scale selection following the global-scale check."""
from pathlib import Path
import argparse, sys
sys.dont_write_bytecode=True
import calibrate as lib
import numpy as np, pandas as pd, torch, joblib
from threadpoolctl import threadpool_limits
import json
FRESH=[1280,4400,5504,6600]
PAST_ENDS=[512,1024,1550,1573,2048,2633,3250,3264,3694,4096,4754,5150,5815,6336,6875,7250,7424,7936]

def run(out):
    d=out/'adaptive'; fixed=out/'fixed12'
    for p in [d,fixed]:p.mkdir(exist_ok=True,parents=True)
    assert not (d/'protocol.json').exists(),'use new --out'
    assert all(min(t+256,u+256)<=max(t,u) for t in FRESH for u in PAST_ENDS)
    lib.write(d/'protocol.json',dict(script=lib.sha(Path(__file__)),base_engine=lib.sha(Path(lib.__file__)),
        subject='HUP080',exploratory_after_failed_global_increase=True,confirm_stops=FRESH,
        previous_scored_future_windows=PAST_ENDS,legacy_stops=lib.CHECK,eval_banks=[1,2],
        history=256,horizon=256,particles=32,fs=256,run04=False,controller_used=False,
        eligibility='each candidate pilot must be finite and have max abs <20 training-standardized units',
        C_grid=lib.GRID,fixed_sensitivity=1.2,
        rule='At each forecast origin, given only the preceding 256 real amplitudes and the frozen fold model, generate 32 pilot paths from the origin, with the same new pilot innovation bank for all C. Choose eligible C with smallest .6 normalized mean full occupation W1 + .4 normalized mean absolute IQR discrepancy, TOWARD the known preceding second for both metrics; denominators are the baseline C pilot metrics. Ties lower C. Lock this before independent evaluation draws. Four fresh check forecast windows are disjoint from historical scored future intervals, but from same three previously inspected development recordings; fits and input histories are not independent new seizures. Additional 4 old primary windows are illustrative only. No fitting online and no observed future refresh. Pilot ensembles beyond origin represent simulated possible futures only.',
        original_rate=lib.BASE))
    audit=json.loads((lib.PREV/'fit_audit.json').read_text())
    assert lib.sha(lib.DATA)=='8d2c6d504f314938c8794fe69d58599f1c43882b9dcc640080b976968d6ede3e'
    with np.load(lib.DATA,allow_pickle=False) as ds:
        assert not any('04' in k or 'outer' in k for k in ds.files)
        obsraw={i:ds[f'ictal_run_{i:02d}'] for i in [1,2,3]}
        labs=ds['channels'].astype(str)
    models={}
    for f in [1,2,3]:
        p=lib.PREV/f'free_fold{f}.joblib';assert lib.sha(p)==audit[f-1]['free_sha256']
        m=joblib.load(p); assert m.q==24 and not m.adjacency.any()
        world=lib.TorchGraphRCSDE(m,[0],256.,control_graph_diffusion_time=0.,preserve_physical_control_residual=True,dtype=torch.float64)
        a=lib.FrozenGraphRCMarkovAdapter(world,control_step_scale=.5,dtype=torch.float64)
        st={k:lib.FrozenIctalGraphRCBatchStepper(world,a,diffusion_scale=k) for k in lib.GRID}
        models[f]=(m,a,st)
    stages={'legacy':lib.CHECK,'fresh':FRESH}
    def pred(st,h,e):
        with torch.no_grad():return lib.uncontrolled_particle_rollout(st,h,e)[:,1:].numpy()
    # Lock all per-origin decisions separately from any scoring against actual futures.
    picks=[];pilot=[]
    for stage,stops in stages.items():
        for f,(m,a,st) in models.items():
            for t in stops:
                raw=obsraw[f][t-256:t];hist=m.transform.scaler.transform(raw)
                h=a.initial_state_from_context(raw)
                pilot_seed=26112500+101*f+137*t; e=lib.noise(pilot_seed)
                rows=[]
                for k in lib.GRID:
                    x=pred(st[k],h,e)
                    vals=lib.compute(x,hist) if np.isfinite(x).all() else None
                    rows.append(lib.row('pilot',f,f,t,-1,k,vals))
                dd=pd.DataFrame(rows)
                denom=dd.iloc[0];assert denom.C==lib.BASE and not denom.unstable
                dd['score']=.6*dd.occupation_W1/max(denom.occupation_W1,1e-12)+.4*dd.iqr_gap/max(denom.iqr_gap,1e-12)
                k=float(dd[~dd.unstable].sort_values(['score','C']).iloc[0].C)
                picks.append(dict(stage=stage,fold=f,stop=t,pilot_seed=pilot_seed,C=k,
                    history_hash=lib.digest(hist)))
                dd['stage']=stage;pilot.append(dd)
            print('pilot locked',stage,f,[p['C'] for p in picks if p['stage']==stage and p['fold']==f],flush=True)
    pd.concat(pilot).to_csv(d/'pilot_diagnostics.csv',index=False);lib.write(d/'frozen_window_C.json',picks)
    rows=[];chans=[]
    for sel in json.loads((d/'frozen_window_C.json').read_text()):
        stage,f,t,k=sel['stage'],sel['fold'],sel['stop'],sel['C'];m,a,st=models[f]
        hist=obsraw[f][t-256:t];assert lib.digest(m.transform.scaler.transform(hist))==sel['history_hash']
        y=m.transform.scaler.transform(obsraw[f][t:t+256]);h=a.initial_state_from_context(hist)
        for b in [1,2]:
            eval_seed=lib.seed('evaluation',f,f,t,b);assert eval_seed!=sel['pilot_seed']
            e=lib.noise(eval_seed)
            result={}
            for g in sorted(set([lib.BASE,1.2,k])):
                xx=pred(st[g],h,e);result[g]=(xx,lib.compute(xx,y) if np.isfinite(xx).all() else None)
            for arm,cc in [('old',lib.BASE),('fixed_1.2',1.2),('adaptive',k)]:
                x,ch=result[cc];rows.append(dict(stage=stage,**lib.row(arm,f,f,t,b,cc,ch)))
                if ch is not None:chans.append(pd.DataFrame(dict(stage=stage,arm=arm,fold=f,stop=t,bank=b,C=cc,channel=labs,**ch)))
            if (f,t,b)==(1,5150,1):
                with np.load(lib.PREV/'display.npz',allow_pickle=False) as prev:
                    np.testing.assert_array_equal(y,prev['observed']);np.testing.assert_array_equal(result[lib.BASE][0],prev['free'])
                    np.testing.assert_array_equal(labs,prev['labels'])
                for save,g in [(d,k),(fixed,1.2)]:
                    np.savez_compressed(save/'display.npz',observed=y,old=result[lib.BASE][0],new=result[g][0],labels=labs,C=g,old_C=lib.BASE,fold=f,stop=t,bank=b)
                print('DISPLAY READY',k,flush=True)
    df=pd.DataFrame(rows);assert len(df)==144
    df.to_csv(d/'evaluation.csv',index=False);pd.concat(chans).to_csv(d/'channels.csv',index=False)
    print(df.groupby(['stage','arm'])[['occupation_W1','sd_gap','iqr_gap','coverage90','rmse','unstable']].mean().round(4).to_string(),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=lib.HERE);args=ap.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):run(args.out)
