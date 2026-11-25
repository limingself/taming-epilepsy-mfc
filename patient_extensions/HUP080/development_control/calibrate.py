"""Fold-specific innovation calibration of frozen HUP080 autonomous predictors."""
from pathlib import Path
import argparse, hashlib, json, os, sys
from development_paths import CALIBRATION as HERE, FREE as PREV, PROJECT, ART
DATA=ART/'01_prepare_development/development_arrays.npz'
sys.dont_write_bytecode=True
sys.path.insert(0,str(PROJECT))
for var in ('MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[var]='1'
for var in ('MPLCONFIGDIR','_MNE_FAKE_HOME_DIR','MNE_HOME','JOBLIB_TEMP_FOLDER'):
    folder=HERE/'.runtime'/var;folder.mkdir(parents=True,exist_ok=True);os.environ[var]=str(folder)
import numpy as np
import pandas as pd
import torch, joblib
from threadpoolctl import threadpool_limits
from mfc_pipeline import part2_state_dependent_rc_sde as p2
sys.modules.setdefault('part2_state_dependent_base',p2)
from mfc_pipeline.square_wave_mfc import TorchGraphRCSDE
from mfc_pipeline.causal_ltv_riccati import FrozenGraphRCMarkovAdapter
from mfc_pipeline.causal_ltv_particle_rollout import FrozenIctalGraphRCBatchStepper
from mfc_pipeline.sequential_covariance_hjb import uncontrolled_particle_rollout
BASE=.79451175
GRID=[BASE,1.,1.2,1.4,1.6,1.8,2.,2.3]
CAL=[2048,3264,6336];CHECK=[5150,1550,3250,7250];BANKS=[1,2]

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,o):p.write_text(json.dumps(o,indent=2,allow_nan=False),encoding='utf8')
def digest(arr):return hashlib.sha256(np.ascontiguousarray(arr).view(np.uint8)).hexdigest()
def noise(seed):
    h=np.random.default_rng(seed).standard_normal((16,256,96))[:,:,:24]
    return torch.as_tensor(np.concatenate([h,-h]),dtype=torch.float64)
def seed(stage,f,run,t,bank):
    if stage=='evaluation':return 202611240+23*t+50003*bank+1000003*f
    return 25112026+13849*f+9323*run+17*t+23993*bank
def compute(z,y):
    assert z.shape==(32,256,96)
    n,h,c=z.shape
    pool=z.reshape(n*h,c); ys=np.sort(y,axis=0);zs=np.sort(pool,axis=0)
    w=np.abs(zs.reshape(h,n,c)-ys[:,None]).mean((0,1))
    qs=[.05,.25,.5,.75,.95]
    p=np.quantile(zs,qs,axis=0); obs=np.quantile(ys,qs,axis=0)
    sd=pool.std(0);sd0=y.std(0);iq=p[3]-p[1];iq0=obs[3]-obs[1]
    return dict(occupation_W1=w,sd_gap=np.abs(sd-sd0),iqr_gap=np.abs(iq-iq0),
        q05_gap=np.abs(p[0]-obs[0]),q95_gap=np.abs(p[4]-obs[4]),
        sd_ratio=sd/np.maximum(sd0,1e-12),iqr_ratio=iq/np.maximum(iq0,1e-12),
        coverage90=((y>=p[0])&(y<=p[-1])).mean(0),median_gap=np.abs(p[2]-obs[2]),
        predicted_sd=sd,observed_sd=sd0,forecast_width90=p[4]-p[0],observed_width90=obs[4]-obs[0],
        ensemble_mse=((z.mean(0)-y)**2).mean(0),absmax=np.max(np.abs(pool),axis=0))
def row(arm,f,run,t,b,k,ch):
    meta=dict(arm=arm,fold=f,run=run,stop=t,bank=b,C=k)
    if ch is None:return dict(**meta,status='nonfinite',unstable=True)
    return dict(**meta,status='finite',unstable=bool(ch['absmax'].max()>=20.),
        **{key:float(val.mean()) for key,val in ch.items() if key not in ('absmax','ensemble_mse')},
        median_sd_ratio=float(np.median(ch['sd_ratio'])),max_abs=float(ch['absmax'].max()),rmse=float(ch['ensemble_mse'].mean()**.5))

def main(out):
    out.mkdir(exist_ok=True,parents=True)
    assert not (out/'protocol.json').exists(),'use fresh --out'
    audit=json.loads((PREV/'fit_audit.json').read_text())
    assert sha(DATA)=='8d2c6d504f314938c8794fe69d58599f1c43882b9dcc640080b976968d6ede3e'
    write(out/'protocol.json',dict(stage='development cross-fitted calibration, not a new external test',subject='HUP080',calibration_stops=CAL,
         grid_absolute=GRID,baseline=BASE,horizon=256,history=256,particles=32,sfreq=256,calibration_banks=[0],
         windows_evaluation=CHECK,banks_evaluation=BANKS,display=[1,5150,1],
         selection='only fit-run forecasts and ictal observations from the two runs used to fit each frozen fold; no draw from validation run used. Candidate must have all finite paths and maximum standardized |x| < 20 over all calibration windows. Require both mean W1 and mean absolute IQR discrepancy not to increase vs baseline. Minimize (0.6*W1/W1_base + 0.4*absolute_IQR_gap/IQR_gap_base), equal weights over contacts, windows and two calibration runs. Break ties by lower C. Choices locked for all three folds before evaluation. Selection source: script and protocol, fixed prior to scoring.',
         q90='pooled predicted empirical amplitude interval Q.05--Q.95 per contact; coverage is fraction of one future recorded window falling in this interval, not time-wise coverage',
         data_sha256=sha(DATA),model_sha256=[a['free_sha256'] for a in audit],script_sha256=sha(Path(__file__)),run04=False,controls=False))
    with np.load(DATA,allow_pickle=False) as v:
        assert not any('04' in k or 'outer' in k for k in v.files)
        signals={j:v[f'ictal_run_{j:02d}'] for j in [1,2,3]};labels=v['channels'].astype(str)
        assert float(v['sfreq'])==256 and all(a.shape==(8192,96) for a in signals.values())
    mods={}
    for f in (1,2,3):
        path=PREV/f'free_fold{f}.joblib'
        assert sha(path)==audit[f-1]['free_sha256'] and audit[f-1]['training_runs']==[r for r in (1,2,3) if r!=f]
        m=joblib.load(path);assert m.q==24 and not np.any(m.adjacency)
        world=TorchGraphRCSDE(m,[0],256.,control_graph_diffusion_time=0.,preserve_physical_control_residual=True,dtype=torch.float64)
        a=FrozenGraphRCMarkovAdapter(world,control_step_scale=.5,dtype=torch.float64)
        st={k:FrozenIctalGraphRCBatchStepper(world,a,diffusion_scale=k) for k in GRID}
        mods[f]=(m,a,st)
    def rollout(stage,f,run,t,b,k):
        m,a,st=mods[f]; tr=signals[run]
        init=a.initial_state_from_context(tr[t-256:t])
        eps=noise(seed(stage,f,run,t,b))
        y=m.transform.scaler.transform(tr[t:t+256])
        with torch.no_grad():z=uncontrolled_particle_rollout(st[k],init,eps)[:,1:].numpy()
        return y,z,(compute(z,y) if np.isfinite(z).all() else None)
    cal=[]; picked={}; ranking=[]
    for f in (1,2,3):
        for k in GRID:
            for run in audit[f-1]['training_runs']:
                for t in CAL:
                    y,z,c=rollout('calibration',f,run,t,0,k)
                    cal.append(row('calibration',f,run,t,0,k,c))
        sub=pd.DataFrame([s for s in cal if s['fold']==f]).groupby('C').agg(w=('occupation_W1','mean'),iq=('iqr_gap','mean'),unsafe=('unstable','sum')).reset_index()
        n=sub[sub.C==BASE].iloc[0];assert n.unsafe==0
        sub['score']=.6*sub.w/n.w+.4*sub.iq/n.iq
        sub['eligible']=(sub.unsafe==0)&(sub.w<=n.w+1e-12)&(sub.iq<=n.iq+1e-12)
        chosen=sub[sub.eligible].sort_values(['score','C']).iloc[0];picked[f]=float(chosen.C)
        sub['fold']=f;ranking.append(sub)
        print('calibrated',f,picked[f],round(float(chosen.score),3),flush=True)
        pd.DataFrame(cal).to_csv(out/'calibration.csv',index=False);pd.concat(ranking).to_csv(out/'ranking.csv',index=False)
    write(out/'frozen_selection.json',picked)
    rows=[];channel=[]
    frozen=json.loads((out/'frozen_selection.json').read_text())
    previous=pd.read_csv(PREV/'condition_metrics.csv').query("phase=='primary' and arm=='free'").set_index(['fold','boundary','bank'])
    for f in (1,2,3):
        for t in CHECK:
            for b in BANKS:
                show=(f,t,b)==(1,5150,1); zz={}
                for arm,k in [('old',BASE),('new',frozen[str(f)])]:
                    y,z,c=rollout('evaluation',f,f,t,b,k)
                    val=row(arm,f,f,t,b,k,c);rows.append(val)
                    if c is not None:
                        if arm=='old':
                            np.testing.assert_allclose(val['occupation_W1'],previous.loc[(f,t,b),'occupation'],rtol=1e-12)
                        channel.append(pd.DataFrame(dict(arm=arm,fold=f,stop=t,bank=b,C=k,channel=labels,**c)))
                    if show:zz[arm]=z
                if show:
                    with np.load(PREV/'display.npz',allow_pickle=False) as prev:
                        np.testing.assert_array_equal(prev['observed'],y);np.testing.assert_array_equal(prev['labels'],labels)
                        np.testing.assert_array_equal(prev['free'],zz['old'])
                    np.savez_compressed(out/'display.npz',observed=y,labels=labels,C=frozen['1'],old_C=BASE,fold=f,stop=t,bank=b,**zz)
                    print('READY display',flush=True)
        pd.DataFrame(rows).to_csv(out/'evaluation.csv',index=False);pd.concat(channel).to_csv(out/'channels.csv',index=False)
        print('evaluated',f,flush=True)
    assert len(rows)==48
    df=pd.DataFrame(rows)
    vals=['occupation_W1','iqr_gap','sd_gap','coverage90','median_sd_ratio','rmse','q95_gap','q05_gap','max_abs','unstable']
    print(df.groupby('arm')[vals].mean().round(4).to_string(),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=HERE);d=ap.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):main(d.out)
