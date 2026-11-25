"""Training-only amendment: compare budget-aware interpolation with frozen first-stage winner."""
from pathlib import Path
import argparse,copy,json,sys
sys.dont_write_bytecode=True
import refine as R
pr,torch,np,pd,CORE=R.pr,R.torch,R.np,R.pd,R.CORE
ALPHAS=[.25,.5,.75,.875,1.]
def replay(s,d,old_ck,previous):
    if previous['extra_steps']==24:return previous
    torch.manual_seed(R.SE);actor,critic,w,scale,dirs=pr.factory.build_actor(s,CORE)
    actor.load_state_dict(old_ck['actor']);critic.load_state_dict(old_ck['critic'])
    # Get grid-selected seed (earlier feasible grid winner), independent of later checkpoint choice.
    val=pd.read_csv(d/'validation.csv')
    g=float(val[(val['update']==0)&val.eligible].sort_values('score').iloc[0].gain)
    P=s['adapter'].components;A=P[:,s['selected']];E=torch.eye(96,dtype=P.dtype)[s['selected']]
    G=.35*E+.65*(A.T@torch.linalg.solve(A@A.T+.15*torch.eye(len(A),dtype=P.dtype),P))
    with torch.no_grad():actor.deviation_gain_delta.add_(g*G.unsqueeze(0))
    gains=[p for n,p in actor.named_parameters() if n in ('deviation_gain_delta','mean_gain_delta','local_deviation_gain_logits')]
    neural=[p for p in actor.parameters() if all(p is not q for q in gains)]
    opt=torch.optim.AdamW([{'params':gains,'lr':.002},{'params':neural,'lr':.0004}],weight_decay=1e-5)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,24,eta_min=.00006)
    dop=torch.optim.Adam(critic.parameters(),lr=.0001,betas=(0.,.9))
    idx=tuple(range(15,256,16));y=s['fit'];hist=pd.read_csv(d/'history.csv')
    for k in range(1,25):
        ctx,st=s['train'][(k-1)%len(s['train'])];e=CORE.antithetic_noise(R.SE+1009*k,24);actor.eval()
        roll=lambda:CORE.empirical_fp_rollout(st,actor,ctx,e)
        with torch.no_grad():xx=roll().scaled[:,1:].detach();ff=CORE.uncontrolled_particle_rollout(st,ctx,e)[:,1:]
        CORE.set_requires_grad(critic,True);CORE.set_requires_grad(actor,False);critic.train()
        for j in range(3):
            dop.zero_grad(set_to_none=True);dr=CORE.balanced_wgan_gp_loss(critic,xx,y,idx,gradient_penalty_weight=10.,critic_drift_weight=.001,
              generator=torch.Generator().manual_seed(R.SE+11000*k+j));dr.loss.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),5.,error_if_nonfinite=True);dop.step()
        CORE.set_requires_grad(critic,False);CORE.set_requires_grad(actor,True);actor.train()
        opt.zero_grad(set_to_none=True);r=roll();law,_=pr.loss.law(CORE,r,y,w,scale,dirs,actor,ff,True)
        adv,_=CORE.actor_wasserstein_loss(critic,r.scaled[:,1:],y,idx)
        np.testing.assert_allclose([law.detach().item(),adv.detach().item()],[hist.law.iloc[k-1],hist.adv.iloc[k-1]],rtol=1e-7,atol=1e-8)
        obj=law+.5*adv;obj.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True);opt.step();sched.step()
        del r,law,adv,obj,xx,ff,dr
    return dict(actor=actor.state_dict(),critic=critic.state_dict(),init_gain=g,extra_steps=24)

def finalize(s,d):
    f=s['fold'];end=d/'final';end.mkdir(exist_ok=True);assert not (end/'frozen.json').exists()
    start=torch.load(R.PREV/f'fold{f}'/'frozen_policy.pt',map_location='cpu',weights_only=False)
    previous=torch.load(d/'frozen_policy.pt',map_location='cpu',weights_only=False)
    pr.write(end/'protocol.json',dict(phase='internal validation-only amendment BEFORE seeing any current held-run scores',
          justification='a trained checkpoint slightly exceeded internal budget headroom in a training seizure',
          options=ALPHAS,comparator='unmodified frozen first-stage winner',
          scoring='exact same four contexts, normalization, endpoint and 5% budget margin as stage1',
          method='interpolate all trainable actor parameters between the prior 32-step parent and the 24-additional-update terminal checkpoint. No buffers/non-parametric components are interpolated. Critic used only during training.',
          original_parent=pr.old.sha(R.PREV/f'fold{f}'/'frozen_policy.pt'),phase1=pr.old.sha(d/'frozen_policy.pt'),
          script=pr.old.sha(Path(__file__))))
    last=replay(s,d,start,previous);torch.save(last,end/'terminal24.pt')
    a=pr.factory.build_actor(s,CORE)[0];names=set(dict(a.named_parameters()))
    val=R.extra_val(s)+s['vctx'];refs=s['val']*2;es=[CORE.antithetic_noise(CORE.VALIDATION_SEED+2300+j,24) for j in range(4)]
    def inspect(state):
        a.load_state_dict(state);a.eval();v=[];b=[]
        with torch.no_grad():
            for (ctx,st),e,y in zip(val,es,refs):
                r=CORE.empirical_fp_rollout(st,a,ctx,e);x=r.scaled[:,1:].numpy();yr=y.numpy();t,o=pr.old.w1s(x,yr)
                v.append([o.mean(),t.mean(),np.quantile(o,.9),np.abs(x.mean((0,1))-yr.mean((0,1))).mean()+.1]);b.append(pr.old.inputs(r.controls.numpy()))
        return np.array(v),b
    b,_=inspect(start['actor']);best=copy.deepcopy(previous);best['interpolation_alpha']=None;details=[]
    for al in ALPHAS:
        state={k:(u+al*(last['actor'][k]-u) if k in names else u) for k,u in start['actor'].items()}
        v,bb=inspect(state);sc=float((v/b).mean(0)@np.array([.45,.4,.1,.05]))
        ok=all(x['budget_ok'] and x['max_rms']<=.4275 and x['energy']<=5.40189 for x in bb) and all(v.mean(0)[:2]<=b.mean(0)[:2]+1e-11)
        for j,x in enumerate(bb):details.append(dict(alpha=al,window=[1024,7250][j//2],run=s['others'][j%2],score=sc,eligible=ok,occ=v[j,0],time_w1=v[j,1],**x))
        if ok and sc<best['score']-1e-10:
            best=dict(actor=state,critic=last['critic'],init_gain=last['init_gain'],extra_steps=24,interpolation_alpha=al,score=sc)
    pd.DataFrame(details).to_csv(end/'validation_interpolation.csv',index=False)
    p=end/'frozen_policy.pt';torch.save(best,p)
    pr.write(end/'frozen.json',{k:v for k,v in best.items() if k not in ('actor','critic')}|{'sha256':pr.old.sha(p)})
    print('FINAL FROZEN',f,{k:v for k,v in best.items() if k not in ('actor','critic')},flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,choices=[1,2,3],required=True);ap.add_argument('--evaluate',action='store_true');a=ap.parse_args();d=R.HERE/f'fold{a.fold}'
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with R.threadpool_limits(limits=1):
        assert (d/'frozen.json').exists()
        if a.evaluate:assert all((R.HERE/f'fold{f}'/'final/frozen.json').exists() for f in (1,2,3))
        # Stage-2 work never writes a first-stage protocol: use a private construction scratch directory.
        cache=d/'construction';cache.mkdir(exist_ok=True)
        s=pr.setup(a.fold,cache)
        if a.evaluate:pr.evaluate(s,d,R.restore(s,d/'final'))
        else:finalize(s,d)
