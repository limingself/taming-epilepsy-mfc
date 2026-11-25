"""HUP080 development-only retraining with causal, per-origin noise calibration."""
from pathlib import Path
import argparse,copy,json,os,sys,time,importlib.util
from development_paths import FIRST as HERE, CODE, CALIBRATION
sys.dont_write_bytecode=True
for v in ('MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[v]='1'
def load(n,p):
    spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);sys.modules[n]=m;spec.loader.exec_module(m);return m
old=load('previous_80train',CODE/'control_common.py')
cal=load('cal_80_lib',CODE/'calibrate.py')
AD=CALIBRATION/'adaptive'
import numpy as np, pandas as pd,torch,joblib
from threadpoolctl import threadpool_limits
FRESH=[1280,4400,5504,6600]
core=old.load('hup060_control_core',old.CORE);loss=old.load('hup060_joint_objective',old.TRAIN)
factory=old.load('prior_actor_construction',old.SHELL)
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')

def setup(f,d):
    others=[i for i in (1,2,3) if i!=f];g=old.GRAPH.with_name(f'leave-run-0{f}-out.npz')
    p=old.PREV/f'free_fold{f}.joblib';audit=json.loads((old.PREV/'fit_audit.json').read_text())[f-1]
    assert old.sha(p)==audit['free_sha256'] and audit['training_runs']==others and old.sha(old.DATA)=='8d2c6d504f314938c8794fe69d58599f1c43882b9dcc640080b976968d6ede3e'
    m=joblib.load(p);assert m.q==24 and not m.adjacency.any()
    with np.load(old.DATA,allow_pickle=False) as z:
        assert not any('04' in k or 'outer' in k for k in z.files);names=z['channels'].astype(str)
        tr={i:z[f'ictal_run_0{i}'] for i in others}
        rr=[m.transform.scaler.transform(z[f'preictal_reference_run_0{i}']).reshape(40,256,96) for i in others]
    with np.load(g,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['channels'].astype(str),names)
        score=z['centrality_score'];sel=np.flatnonzero(score>np.quantile(score,.2));assert len(sel)==76
    world=old.TorchGraphRCSDE(m,sel,256.,control_graph_diffusion_time=0.,preserve_physical_control_residual=True,dtype=torch.float64)
    a=old.FrozenGraphRCMarkovAdapter(world,control_step_scale=.5,dtype=torch.float64)
    np.testing.assert_array_equal(world.control_channel_map.detach().numpy(),np.eye(96)[sel])
    steppers={k:old.FrozenIctalGraphRCBatchStepper(world,a,diffusion_scale=k) for k in cal.GRID}
    calrows=[];mapping={}
    for t in [*old.TRAIN_STOPS,old.VAL_STOP]:
        for i in others:
            raw=tr[i][t-256:t];h=m.transform.scaler.transform(raw);ctx=a.initial_state_from_context(raw)
            eps=cal.noise(26112500+101*f+137*t);rows=[]
            for k,step in steppers.items():
                with torch.no_grad():x=core.uncontrolled_particle_rollout(step,ctx,eps)[:,1:].numpy()
                ds=cal.compute(x,h) if np.isfinite(x).all() else None
                rows.append(cal.row('history_pilot',f,i,t,-1,k,ds))
            df=pd.DataFrame(rows);base=df.iloc[0];assert base.C==cal.BASE and not base.unstable
            df['score']=.6*df.occupation_W1/max(base.occupation_W1,1e-12)+.4*df.iqr_gap/max(base.iqr_gap,1e-12)
            k=float(df[~df.unstable].sort_values(['score','C']).iloc[0].C);df['selected']=(df.C==k);calrows.append(df)
            mapping[i,t]=(ctx,steppers[k])
    pd.concat(calrows).to_csv(d/'history_scale_pilots.csv',index=False)
    train=[mapping[i,t] for t in old.TRAIN_STOPS for i in others];val=[mapping[i,old.VAL_STOP] for i in others]
    s=dict(model=m,stepper=steppers[cal.BASE],adapter=a,fit=torch.tensor(np.concatenate([x[:15] for x in rr])),
           val=[torch.tensor(x[15:30]) for x in rr],selected=sel,labels=names,train=train,initial_train=train[0][0],vctx=val,
           others=others,steps=steppers,fold=f)
    write(d/'protocol.json',dict(subject='HUP080',fold=f,fit_runs=others,epochs=32,seed=old.SEED,validation_stop=old.VAL_STOP,
        training_stops=old.TRAIN_STOPS,checkpoints=[0,1,4,8,12,16,20,24,28,32],actuator_indices=sel.tolist(),tau_graph=0,
        reference="per training run segments 0:15 pooled for fit, separate 15:30 for validation, same indices in independent evaluation run after freezing",
        test_stops_legacy=old.TEST_STOPS,test_stops_additional=FRESH,banks=[1,2],rank_graph_sha=old.sha(g),
        free_model_sha=old.sha(p),training_script_sha=old.sha(Path(__file__)),old_controller_source_sha=old.sha(Path(old.__file__)),
        prior_calibration_sha=old.sha(AD/'frozen_window_C.json'),architecture_sha=old.sha(old.CORE),loss_sha=old.sha(old.TRAIN),
        selection="lowest validation score .5 relative law + .25 relative time W1 + .25 relative occupation W1, feasible on both runs, includes neutral, no evaluation feedback",
        optimizer="fresh actor+critic WGAN-GP; actor AdamW lr 0.0003 cos to 0.00001, critic Adam 0.0001, GP 10, drift .001, 24 warmups then 3:1, actor law+.5 critic",
        budgets="actuator max RMS .45, peak 1.8, summed mean-square 5.6862, saturation <.01; legacy RMS .405 also reported",
        noise="per-origin previous 256 samples, fixed causal eight-scale rule, independent pilot seeds from evaluation; use frozen earlier selections on eval origins (history hashes verified)",
        no_run04=True,mode="development preview; not an external test"))
    return s

def fit(s,d):
    SE=old.SEED;torch.manual_seed(SE);np.random.seed(SE)
    actor,critic,w,sc,dirs=factory.build_actor(s,core);ref=s['fit'];idx=tuple(range(15,256,16))
    noise=lambda seed:core.antithetic_noise(seed,24)
    def free(pair,e):ctx,step=pair;return core.uncontrolled_particle_rollout(step,ctx,e)[:,1:]
    def roll(pair,e):ctx,step=pair;return core.empirical_fp_rollout(step,actor,ctx,e)
    opt=torch.optim.AdamW(actor.parameters(),lr=.0003,weight_decay=1e-5)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=32,eta_min=.00001)
    dop=torch.optim.Adam(critic.parameters(),lr=.0001,betas=(0.,.9))
    vnoise=[noise(core.VALIDATION_SEED+i) for i in (0,1)]
    with torch.no_grad():vf=[free(p,e) for p,e in zip(s['vctx'],vnoise)]
    def validate():
        actor.eval();rows=[];bs=[]
        with torch.no_grad():
            for p,y,e,x in zip(s['vctx'],s['val'],vnoise,vf):
                r=roll(p,e);lw,_=loss.law(core,r,y,w,sc,dirs,actor,x,False);tt,oo=old.w1s(r.scaled[:,1:].numpy(),y.numpy())
                bs.append(old.inputs(r.controls.numpy()));rows.append([float(lw),tt.mean(),oo.mean()])
        return np.asarray(rows),bs
    base,_=validate()
    with torch.no_grad():
        warm=[free(s['train'][i],noise(SE+101*(i+1))) for i in range(3)]
        n=roll(s['train'][0],noise(SE+101));assert n.controls.abs().max()<1e-13
        torch.testing.assert_close(n.scaled[:,1:],warm[0],atol=1e-10,rtol=1e-10)
    def disc(x,seed):
        dop.zero_grad(set_to_none=True)
        r=core.balanced_wgan_gp_loss(critic,x,ref,idx,gradient_penalty_weight=10.,critic_drift_weight=.001,generator=torch.Generator().manual_seed(seed))
        r.loss.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),5.,error_if_nonfinite=True);dop.step();return r
    for j in range(24):disc(warm[j%3],SE+50000+j)
    def snap(ep,sc):return dict(epoch=ep,score=float(sc),actor=copy.deepcopy(actor.state_dict()),critic=copy.deepcopy(critic.state_dict()))
    best=snap(0,1);hist=[];vv=[]
    for k in range(1,33):
        pair=s['train'][(k-1)%len(s['train'])];e=noise(SE+1009*k);actor.eval()
        with torch.no_grad():x=roll(pair,e).scaled[:,1:].detach();z=free(pair,e)
        core.set_requires_grad(actor,False);core.set_requires_grad(critic,True);critic.train()
        for j in range(3):dr=disc(x,SE+100000*k+j)
        core.set_requires_grad(critic,False);core.set_requires_grad(actor,True);actor.train();opt.zero_grad(set_to_none=True)
        r=roll(pair,e);lw,_=loss.law(core,r,ref,w,sc,dirs,actor,z,True);adv,_=core.actor_wasserstein_loss(critic,r.scaled[:,1:],ref,idx)
        tot=lw+.5*adv;tot.backward();grad=torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True);opt.step();sched.step()
        hist.append(dict(update=k,law=lw.detach().item(),adv=adv.detach().item(),total=tot.detach().item(),critic=dr.loss.detach().item(),grad=float(grad),C=pair[1].diffusion_scale))
        del r,x,z,lw,adv,tot
        if k==1 or k%4==0:
            v,bs=validate();score=float(np.mean((v/base)@np.array([.5,.25,.25])));ok=all(b['budget_ok'] for b in bs)
            for i,(val,b) in enumerate(zip(v,bs)):vv.append(dict(update=k,run=s['others'][i],law=val[0],time_w1=val[1],occ_w1=val[2],score=score,**b))
            if ok and score<best['score']:best=snap(k,score)
            pd.DataFrame(vv).to_csv(d/'validation.csv',index=False)
            print(s['fold'],k,'best',best['epoch'],'eligible',ok,flush=True)
        pd.DataFrame(hist).to_csv(d/'history.csv',index=False)
    ck=d/'frozen_policy.pt';torch.save(best,ck);restored=torch.load(ck,map_location='cpu',weights_only=False)
    actor.load_state_dict(restored['actor']);actor.eval();core.set_requires_grad(actor,False)
    write(d/'frozen.json',dict(epoch=best['epoch'],score=best['score'],sha256=old.sha(ck),critic_updates=120))
    return actor

def evaluate(s,d,actor):
    f=s['fold'];assert (d/'frozen.json').is_file()
    with np.load(old.DATA,allow_pickle=False) as ds:
        tr=ds[f'ictal_run_0{f}'];rf=ds[f'preictal_reference_run_0{f}']
    scale=s['model'].transform.scaler;ref=scale.transform(rf).reshape(40,256,96)[15:30]
    picks={i['stop']:i for i in json.loads((AD/'frozen_window_C.json').read_text()) if i['fold']==f}
    direct=np.isin(np.arange(96),s['selected']);rows=[];cc=[]
    for t in [*old.TEST_STOPS,*FRESH]:
        pick=picks[t];raw=tr[t-256:t];assert cal.digest(scale.transform(raw))==pick['history_hash']
        C=pick['C'];step=s['steps'][C];ctx=s['adapter'].initial_state_from_context(raw)
        obs=scale.transform(tr[t:t+256])
        for bank in (1,2):
            e=cal.noise(cal.seed('evaluation',f,f,t,bank))
            with torch.no_grad():
                free=core.uncontrolled_particle_rollout(step,ctx,e)[:,1:].numpy()
                r=core.empirical_fp_rollout(step,actor,ctx,e);con=r.scaled[:,1:].numpy();u=r.controls.numpy()
            assert np.isfinite(con).all() and np.isfinite(u).all()
            if (f,t,bank)==(1,5150,1):
                with np.load(AD/'display.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(obs,z['observed']);np.testing.assert_array_equal(free,z['new']);np.testing.assert_array_equal(s['labels'],z['labels']);assert C==z['C']
                np.savez_compressed(HERE/'display_control.npz',observed=obs,free=free,controlled=con,reference=ref,labels=s['labels'],selected=s['selected'],controls=u,C=C)
            tf,of=old.w1s(free,ref);tc,oc=old.w1s(con,ref)
            key=dict(fold=f,stop=t,bank=bank,C=C,stage='fresh' if t in FRESH else 'legacy')
            rows.append(dict(**key,time_free=tf.mean(),time_controlled=tc.mean(),occupation_free=of.mean(),occupation_controlled=oc.mean(),
                both_improved=int(((tc<tf)&(oc<of)).sum()),occupation_improved=int((oc<of).sum()),
                indirect_occupation_improved=int(((oc<of)&~direct).sum()),**old.inputs(u)))
            cc.append(pd.DataFrame(dict(**key,channel=s['labels'],direct=direct,time_free=tf,time_controlled=tc,occupation_free=of,occupation_controlled=oc)))
            print('scored',f,t,bank,round(oc.mean(),4),rows[-1]['budget_ok'],flush=True)
    pd.DataFrame(rows).to_csv(d/'conditions.csv',index=False);pd.concat(cc).to_csv(d/'channels.csv',index=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fold',type=int,choices=(1,2,3),required=True);args=parser.parse_args()
    d=HERE/f'fold{args.fold}';d.mkdir(parents=True,exist_ok=True);assert not (d/'frozen_policy.pt').exists(),'preserve existing result'
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):
        s=setup(args.fold,d);a=fit(s,d);evaluate(s,d,a)
