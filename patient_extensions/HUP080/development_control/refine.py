"""Training-only controller refinement for the frozen HUP080 adaptive predictor."""
from pathlib import Path
import argparse,copy,importlib.util,json,os,sys
from development_paths import FINAL as HERE, FIRST as PREV, CODE
sys.dont_write_bytecode=True
for key in ('MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
sp=importlib.util.spec_from_file_location('history80',CODE/'train_control.py')
pr=importlib.util.module_from_spec(sp);sys.modules['history80']=pr;sp.loader.exec_module(pr)
pr.HERE=HERE # pr.evaluate writes the single display here; all prior artifacts remain read-only.
import numpy as np,torch,pandas as pd
from threadpoolctl import threadpool_limits
SE=202611252;CORE=pr.core;GRID=[0.,.1,.2,.35,.50,.75,1.05]
def extra_val(s):
    result=[]
    with np.load(pr.old.DATA,allow_pickle=False) as ds:
        for i in s['others']:
            raw=ds[f'ictal_run_0{i}'][1024-256:1024];h=s['model'].transform.scaler.transform(raw);cx=s['adapter'].initial_state_from_context(raw)
            eps=pr.cal.noise(26112500+101*s['fold']+137*1024);scores=[]
            for k,st in s['steps'].items():
                with torch.no_grad():x=CORE.uncontrolled_particle_rollout(st,cx,eps)[:,1:].numpy()
                ok=np.isfinite(x).all() and np.abs(x).max()<20
                if ok:
                    ds1=pr.cal.compute(x,h);scores.append((k,float(ds1['occupation_W1'].mean()),float(ds1['iqr_gap'].mean())))
            base=scores[0];assert base[0]==pr.cal.BASE
            k=min(scores,key=lambda r:(.6*r[1]/max(base[1],1e-12)+.4*r[2]/max(base[2],1e-12),r[0]))[0]
            result.append((cx,s['steps'][k]))
    return result

def fit(s,d):
    f=s['fold'];payload=PREV/f'fold{f}'/'frozen_policy.pt';assert pr.old.sha(payload)==json.loads((payload.parent/'frozen.json').read_text())['sha256']
    ck=torch.load(payload,map_location='cpu',weights_only=False)
    torch.manual_seed(SE);actor,critic,w,scale,dirs=pr.factory.build_actor(s,CORE);actor.load_state_dict(ck['actor']);critic.load_state_dict(ck['critic'])
    idx=tuple(range(15,256,16));fitref=s['fit'];nn=lambda seed:CORE.antithetic_noise(seed,24)
    validation=extra_val(s)+s['vctx']
    refs=s['val']*2
    masks=s['selected'];P=s['adapter'].components;A=P[:,masks]
    E=torch.eye(96,dtype=P.dtype)[masks]
    G=.35*E+.65*(A.T@torch.linalg.solve(A@A.T+.15*torch.eye(len(A),dtype=P.dtype),P))
    config=dict(phase='warm-start continuation; 24 actual WGAN-GP actor updates, 3 critic updates per actor, inherited actor32 and critic120',parent=pr.old.sha(payload),
         axis='0.35 E + 0.65 A^T (A A^T+0.15 I)^-1 P, added to the existing deviation_gain_delta before all limiting',
         validated_gain_grid=GRID,windows=[1024,7250],weights=[.45,.40,.10,.05],
         rule='Eligible if mean validation occupation and time are each no worse than parent, all four conditions pass original limits plus max RMS <= .4275 and energy <=5.40189. Index=.45 mean rel occ +.40 mean rel time+.10 per-contact 90th-percentile relative occ+.05 mean absolute center gap relative with .1 floor added. Condition-relative quantities averaged; original policy is fallback.',
         held_none_until_frozen=True,seed=SE,exact_state_dict_only=True,script_sha=pr.old.sha(Path(__file__)),fold=f,train_runs=s['others'])
    pr.write(d/'refinement_protocol.json',config)
    vnoise=[nn(CORE.VALIDATION_SEED+2300+j) for j in range(4)]
    def roll(p,e):cx,st=p;return CORE.empirical_fp_rollout(st,actor,cx,e)
    checkpoints=[];best=None;records=[]
    def val():
        actor.eval();vals=[];bud=[]
        with torch.no_grad():
            for p,e,yr in zip(validation,vnoise,refs):
                r=roll(p,e);x=r.scaled[:,1:].numpy();y=yr.numpy()
                assert np.isfinite(x).all()
                t,o=pr.old.w1s(x,y);ce=np.mean(abs(x.mean((0,1))-y.mean((0,1))))
                vals.append([o.mean(),t.mean(),np.quantile(o,.9),ce+.1]);bud.append(pr.old.inputs(r.controls.numpy()))
        return np.array(vals),bud
    baseline,_=val()
    def check(g,ep):
        nonlocal best
        v,b=val();rel=v/baseline;sc=rel.mean(0)@np.array([.45,.4,.1,.05])
        ok=all(y['budget_ok'] and y['max_rms']<=.4275 and y['energy']<=5.40189 for y in b) and all(v.mean(0)[:2]<=baseline.mean(0)[:2]+1e-11)
        if best is None or (ok and sc<best['score']-1e-10):
            best=dict(score=float(sc),init_gain=g,extra_steps=ep,actor=copy.deepcopy(actor.state_dict()),critic=copy.deepcopy(critic.state_dict()))
        for j,(r,bb) in enumerate(zip(v,b)):checkpoints.append(dict(gain=g,update=ep,window=[1024,7250][j//2],run=s['others'][j%2],
                 C=validation[j][1].diffusion_scale,occ=r[0],time_w1=r[1],tail_p90=r[2],center_error=r[3]-.1,score=sc,eligible=ok,**bb))
        pd.DataFrame(checkpoints).to_csv(d/'validation.csv',index=False)
        print('validation',f,g,ep,round(sc,3),ok,'best',best['init_gain'],best['extra_steps'],flush=True)
    for g in GRID:
        actor.load_state_dict(ck['actor'])
        with torch.no_grad():actor.deviation_gain_delta.add_(g*G.unsqueeze(0))
        check(g,0)
    g=best['init_gain'];actor.load_state_dict(best['actor']);critic.load_state_dict(best['critic'])
    gains=[p for n,p in actor.named_parameters() if n in ('deviation_gain_delta','mean_gain_delta','local_deviation_gain_logits')]
    neural=[p for p in actor.parameters() if all(p is not q for q in gains)]
    opt=torch.optim.AdamW([{'params':gains,'lr':.002},{'params':neural,'lr':.0004}],weight_decay=1e-5)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,24,eta_min=.00006)
    dop=torch.optim.Adam(critic.parameters(),lr=.0001,betas=(0.,.9))
    for k in range(1,25):
        p=s['train'][(k-1)%len(s['train'])];cx,st=p;e=nn(SE+1009*k);actor.eval()
        with torch.no_grad():xx=roll(p,e).scaled[:,1:].detach();ff=CORE.uncontrolled_particle_rollout(st,cx,e)[:,1:]
        CORE.set_requires_grad(critic,True);CORE.set_requires_grad(actor,False);critic.train()
        for j in range(3):
            dop.zero_grad(set_to_none=True);dr=CORE.balanced_wgan_gp_loss(critic,xx,fitref,idx,gradient_penalty_weight=10.,critic_drift_weight=.001,
                 generator=torch.Generator().manual_seed(SE+11000*k+j));dr.loss.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),5.,error_if_nonfinite=True);dop.step()
        CORE.set_requires_grad(critic,False);CORE.set_requires_grad(actor,True);actor.train();opt.zero_grad(set_to_none=True)
        r=roll(p,e);law,_=pr.loss.law(CORE,r,fitref,w,scale,dirs,actor,ff,True);adv,_=CORE.actor_wasserstein_loss(critic,r.scaled[:,1:],fitref,idx)
        obj=law+.5*adv;obj.backward();gr=torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True);opt.step();sched.step()
        records.append(dict(update=k,law=law.detach().item(),adv=adv.detach().item(),critic=dr.loss.detach().item(),grad=float(gr)))
        del r,law,adv,obj,xx,ff,dr
        pd.DataFrame(records).to_csv(d/'history.csv',index=False)
        if k in (4,8,12,16,20,24):check(g,k)
    path=d/'frozen_policy.pt';torch.save(best,path);pr.write(d/'frozen.json',{k:v for k,v in best.items() if k not in ('actor','critic')}|{'sha256':pr.old.sha(path)})
    return restore(s,d)

def restore(s,d):
    p=d/'frozen_policy.pt';assert pr.old.sha(p)==json.loads((d/'frozen.json').read_text())['sha256']
    act=pr.factory.build_actor(s,CORE)[0];act.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['actor'])
    act.eval();CORE.set_requires_grad(act,False);return act

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True,choices=(1,2,3));ap.add_argument('--evaluate',action='store_true');args=ap.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    d=HERE/f'fold{args.fold}';d.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        if args.evaluate:
            assert all((HERE/f'fold{f}'/'frozen.json').exists() for f in (1,2,3)), 'freeze all folds before any evaluation'
            s=pr.setup(args.fold,d);pr.evaluate(s,d,restore(s,d))
        else:
            assert not (d/'frozen_policy.pt').exists();s=pr.setup(args.fold,d);fit(s,d)
