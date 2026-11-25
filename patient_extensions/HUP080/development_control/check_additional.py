"""One post-freeze, paired replay at three more fixed origins per development seizure."""
import argparse, json, sys, os
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
for v in ('MKL_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[v]='1'
sys.dont_write_bytecode=True
from development_paths import FINAL, FIRST as PREV, CODE
HERE=FINAL.parent
SP=spec_from_file_location('hup80_previous_train',CODE/'train_control.py')
p=module_from_spec(SP);sys.modules[SP.name]=p;SP.loader.exec_module(p)
import numpy as np, pandas as pd, torch
from threadpoolctl import threadpool_limits
import exact_distances as met
protocol=json.loads((CODE/'extra_check_protocol.json').read_text())
PLAN=protocol['extra_forecast_starts_samples']

def load_plant(f):
    ids=[i for i in [1,2,3] if i!=f]
    cfg=json.loads((PREV/f'fold{f}'/'protocol.json').read_text())
    assert cfg['fit_runs']==ids and cfg['tau_graph']==0
    model_f=p.old.PREV/f'free_fold{f}.joblib';assert p.old.sha(model_f)==cfg['free_model_sha']
    model=p.joblib.load(model_f); sel=np.array(cfg['actuator_indices'])
    assert len(sel)==76 and model.q==24
    world=p.old.TorchGraphRCSDE(model,sel,256.,control_graph_diffusion_time=0.,preserve_physical_control_residual=True,dtype=torch.float64)
    a=p.old.FrozenGraphRCMarkovAdapter(world,control_step_scale=.5,dtype=torch.float64)
    steps={k:p.old.FrozenIctalGraphRCBatchStepper(world,a,diffusion_scale=k) for k in p.cal.GRID}
    scaler=model.transform.scaler
    with np.load(p.old.DATA,allow_pickle=False) as d:
        assert all('04' not in k and 'outer' not in k for k in d.files)
        ref_train=np.concatenate([scaler.transform(d[f'preictal_reference_run_{i:02d}']).reshape(40,256,96)[:15] for i in ids])
        tr=d[f'ictal_run_{f:02d}'];ref=scaler.transform(d[f'preictal_reference_run_{f:02d}']).reshape(40,256,96)[15:30]
        ic=a.initial_state_from_context(d[f'ictal_run_{ids[0]:02d}'][1550-256:1550])
    s=dict(model=model,selected=sel,stepper=steps[p.cal.BASE],adapter=a,initial_train=ic,fit=torch.tensor(ref_train))
    def actor(folder):
        file=folder/'frozen_policy.pt';assert file.is_file()
        md=json.loads((folder/'frozen.json').read_text())
        assert md['sha256']==p.old.sha(file)
        ck=torch.load(file,map_location='cpu',weights_only=False)
        act=p.factory.build_actor(s,p.core)[0];act.load_state_dict(ck['actor'],strict=True);act.eval();p.core.set_requires_grad(act,False);return act
    return actor(PREV/f'fold{f}'),actor(HERE/'engine'/f'fold{f}'/'final'),s,steps,tr,ref

def check(f):
    assert all((HERE/'engine'/f'fold{i}'/'final'/'frozen.json').is_file() for i in [1,2,3]), 'freeze every final policy before evaluation'
    assert sorted(PLAN)==[2340,6076,7680]
    observed_before=json.loads((p.AD/'protocol.json').read_text());assigned=set(observed_before['previous_scored_future_windows']+observed_before['confirm_stops'])
    assert all(t+256<=k or t>=k+256 for t in PLAN for k in assigned)
    prev,cand,s,steps,tr,ref=load_plant(f)
    src_hash={'previous':p.old.sha(PREV/f'fold{f}'/'frozen_policy.pt'),'refined':p.old.sha(HERE/'engine'/f'fold{f}'/'final'/'frozen_policy.pt'),'free':'no control'}
    pths=HERE/'additional';pths.mkdir(exist_ok=True)
    direct=np.isin(np.arange(96),s['selected']);rows=[]
    for t in PLAN:
        raw=tr[t-256:t];hist=s['model'].transform.scaler.transform(raw);ctx=s['adapter'].initial_state_from_context(raw)
        eps=p.cal.noise(26112500+101*f+137*t);pick=[]
        for k,step in steps.items():
            with torch.no_grad():x=p.core.uncontrolled_particle_rollout(step,ctx,eps)[:,1:].numpy()
            ds=p.cal.compute(x,hist) if np.isfinite(x).all() else None
            pick.append(p.cal.row('pilot',f,f,t,-1,k,ds))
        dd=pd.DataFrame(pick);b=dd.iloc[0];assert b.C==p.cal.BASE and not b.unstable
        dd['score']=.6*dd.occupation_W1/max(b.occupation_W1,1e-12)+.4*dd.iqr_gap/max(b.iqr_gap,1e-12)
        kval=float(dd[~dd.unstable].sort_values(['score','C']).iloc[0].C);step=steps[kval]
        for bank in protocol['evaluation_banks']:
            e=p.cal.noise(p.cal.seed('evaluation',f,f,t,bank))
            with torch.no_grad():
                x=p.core.uncontrolled_particle_rollout(step,ctx,e)[:,1:].numpy();oo=p.core.empirical_fp_rollout(step,prev,ctx,e);cc=p.core.empirical_fp_rollout(step,cand,ctx,e)
            for arm,z,u in [('free',x,None),('previous',oo.scaled[:,1:].numpy(),oo.controls.numpy()),('refined',cc.scaled[:,1:].numpy(),cc.controls.numpy())]:
                assert np.isfinite(z).all()
                tt,od=met.time_distance(z,ref),met.distance(z,ref)
                rows.append(dict(fold=f,stop=t,bank=bank,C=kval,arm=arm,weight_hash=src_hash[arm],time_W1=tt.mean(),occup_W1=od.mean(),non_input_occ=od[~direct].mean(),direct_occ=od[direct].mean(),**(p.old.inputs(u) if u is not None else {})))
            del x,oo,cc
    out=pd.DataFrame(rows);assert len(out)==18
    out.to_csv(pths/f'fold{f}.csv',index=False);print(out.groupby('arm')[['occup_W1','time_W1']].mean().to_string())

def summary():
    data=pd.concat([pd.read_csv(HERE/'additional'/f'fold{f}.csv') for f in (1,2,3)])
    assert len(data)==54 and not data.duplicated(['fold','stop','bank','arm']).any()
    assert set(data.stop)==set(PLAN)
    w=data.groupby(['fold','stop','arm'])[['occup_W1','time_W1','direct_occ','non_input_occ','energy']].mean()
    out={}
    for m in ('occup_W1','time_W1','non_input_occ','direct_occ'):
        wide=w[m].unstack('arm');av=wide.mean();out[m]=dict(mean=av.to_dict(),higher_vs_previous=int((wide.refined>wide.previous).sum()),lower_vs_previous=int((wide.refined<wide.previous).sum()),compared_windows=len(wide),relative_vs_old=100*(1-av.refined/av.previous),relative_vs_free=100*(1-av.refined/av.free))
    for arm in ['previous','refined']:
        v=data[data.arm==arm];ok=(v.peak<=1.8+1e-12)&(v.max_rms<=.45)&(v.energy<=5.6862)&(v.sat<.01)
        out[arm+'_usage']=dict(all_pairs=len(v),pass_pairs=int(ok.sum()),max_energy=v.energy.max(),mean_energy=v.energy.mean(),max_rms=v.max_rms.max(),peak=v.peak.max())
    msg=json.dumps(out,indent=2,allow_nan=False);(HERE/'checked_additional.json').write_text(msg,encoding='utf8');print(msg)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,choices=[1,2,3]);ap.add_argument('--all',action='store_true');ap.add_argument('--summary',action='store_true');args=ap.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):
        if args.all:
            import gc
            for fold in (1,2,3):
                check(fold);gc.collect()
        if args.fold:check(args.fold)
        if args.summary:summary()
