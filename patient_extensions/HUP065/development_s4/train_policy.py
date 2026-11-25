"""Development-only fresh HUP060-style 16-update WGAN-GP preview on the frozen blue predictor.

Training/selection access only the run02 context/reference interface. The
separately requested run01 display is produced after writing a frozen state.
"""
import argparse, copy, csv, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
HERE=MODEL=SOR=TRAIN=None
# Keep import side effects out of build_actor (also used by HUP080).
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from threadpoolctl import threadpool_limits
PLAN=16
SEED=20261011
def load(name,p):
    sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp)
    sys.modules[name]=m;sp.loader.exec_module(m);return m

def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def dump(name,obj):
    (HERE/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding="utf8")

def stats(fit):
    n=fit.shape[-1]
    var=fit.reshape(-1,n).var(dim=0,unbiased=False)
    scale=var.clamp_min(.1**2).sqrt()
    inv=1/var.clamp_min(.1**2)
    w=(.5+.5*inv/inv.mean()).clamp(.5,3)
    return fit.mean(dim=0),(.25*fit.var(dim=0,unbiased=False)+.75*var[None]).clamp_min(1e-5),scale,w/w.mean()

def build_actor(s,m):
    fit,sel,step=s["fit"],np.asarray(s["selected"],dtype=int),s["stepper"]
    mean,var,scale,weights=stats(fit)
    center,ss=m.build_markov_normalization(s["adapter"],s["initial_train"],fit)
    actor=m.StructuredSplineCovarianceActor(step,mean,var,scale,
        torch.zeros(len(sel),fit.shape[-1],dtype=torch.float64),sel,horizon=256,basis_count=8,
        hidden_size=64,amplitude_limit=1.8,actuator_alpha=1.,residual_scale=.15,
        local_gain_initial_fraction=0.,local_gain_maximum_fraction=.05,markov_feature_center=center,
        markov_feature_scale=ss,markov_residual_scale=.6,markov_hidden_size=96,maximum_slew=None)
    critic=m.TimeConditionedWassersteinCritic(fit.reshape(-1,fit.shape[-1]).mean(0),scale,horizon_samples=256,hidden_size=128)
    rng=np.random.default_rng(SEED)
    directions=rng.normal(size=(fit.shape[-1],16));directions/=np.linalg.norm(directions,axis=0)
    return actor,critic,weights,scale,torch.tensor(directions,dtype=torch.float64)

def w1(x,r):
    # Each of the equal-mass empirical quantile functions is piecewise constant.
    x=np.asarray(x);r=np.asarray(r)
    steps=np.lcm(len(x),len(r)); i=np.arange(steps)
    tx=np.sort(x,axis=0);tr=np.sort(r,axis=0)
    timew=np.mean([np.mean(np.abs(tx[p]-tr[q])) for p,q in zip(i*len(x)//steps,i*len(r)//steps)])
    occ=np.mean([wasserstein_distance(x[:,:,j].ravel(),r[:,:,j].ravel()) for j in range(x.shape[-1])])
    return float(timew),float(occ)

def standard(s,m,fn):
    start=time.perf_counter()
    cfg=dict(seed=SEED,epochs=PLAN,preview_only=True,horizon=256,particles=32,
        actor="StructuredSplineCovarianceActor",critic="TimeConditionedWassersteinCritic",
        arch_source_sha=sha(SOR),train_source_sha=sha(TRAIN),plant_sha=sha(MODEL),engine_sha=sha(Path(__file__).with_name("engine.py")),
        actor_lr=.0003,actor_lr_min=.00001,critic_lr=.0001,n_critic=3,critic_warmup=24,
        law="covariance, canonical non-rescaled weights",adv_weight=.5,gp=10.,drift=.001,
        actor_initialization="fresh",training="run02 only",validation="separate run02 context and reference sections",
        selection="lowest validation .5 law+.25 time W1+.25 occupation W1 relative to neutral, updates 1/4/8/12/16",
        paper_reference_training_updates=1000,heldout_run03_access=False,
        selected_indices=list(map(int,s["selected"])))
    dump("training_plan.json",cfg)
    # Byte hashes are provenance; retraining may use a repository checkout with LF or CRLF.
    torch.manual_seed(SEED);np.random.seed(SEED)
    a,d,w,sc,p=build_actor(s,m);fit,val=s["fit"],s["val"]
    st,ic,vc=s["stepper"],s["initial_train"],s["initial_val"]
    opt=torch.optim.AdamW(a.parameters(),lr=.0003,weight_decay=1e-5)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=PLAN,eta_min=1e-5)
    dopt=torch.optim.Adam(d.parameters(),lr=.0001,betas=(0.,.9))
    n=lambda seed:m.antithetic_noise(seed,st.q)
    free=lambda i,e:m.uncontrolled_particle_rollout(st,i,e)[:,1:]
    roll=lambda i,e:m.empirical_fp_rollout(st,a,i,e)
    idx=tuple(range(15,256,16));vnoise=n(m.VALIDATION_SEED)
    with torch.no_grad():
        vb=free(vc,vnoise)
    def validate():
        a.eval()
        with torch.no_grad():
            r=roll(vc,vnoise)
            l,_=fn.law(m,r,val,w,sc,p,a,vb,False)
            t,o=w1(r.scaled[:,1:],val)
            return (float(l),t,o)
    base=np.asarray(validate())
    assert np.isfinite(base).all() and np.min(base)>0
    with torch.no_grad():
        warm=[free(ic,n(SEED+101*(b+1))).detach() for b in range(3)]
        zero=roll(ic,n(SEED+101))
        assert zero.controls.abs().max().item()<1.e-13
        torch.testing.assert_close(zero.scaled[:,1:],warm[0],rtol=1e-10,atol=1e-10)
    def disc(x,gen):
        dopt.zero_grad(set_to_none=True)
        au=m.balanced_wgan_gp_loss(d,x,fit,idx,gradient_penalty_weight=10.,
            critic_drift_weight=.001,generator=torch.Generator().manual_seed(gen))
        if not torch.isfinite(au.loss):raise FloatingPointError("critic")
        au.loss.backward();torch.nn.utils.clip_grad_norm_(d.parameters(),5.,error_if_nonfinite=True);dopt.step()
        return au
    for k in range(24):disc(warm[k%3],SEED+50000+k)
    rows=[]; best_score=float("inf");best=None
    ctxs=s.get("train_contexts",[ic])
    if not isinstance(ctxs,(tuple,list)) or not len(ctxs) or not all(torch.is_tensor(ii) and ii.ndim==1 for ii in ctxs):
        ctxs=[ic]
    print("PLAN",PLAN,"warmup done",len(ctxs),"training contexts", "elapsed",round(time.perf_counter()-start,1),flush=True)
    for epoch in range(1,PLAN+1):
        ci=ctxs[(epoch-1)%len(ctxs)];eps=n(SEED+1009*epoch);a.eval()
        with torch.no_grad():
            detach=roll(ci,eps).scaled[:,1:].detach()
            bline=free(ci,eps)
        m.set_requires_grad(a,False);m.set_requires_grad(d,True);d.train()
        for j in range(3): au=disc(detach,SEED+100000*epoch+j)
        m.set_requires_grad(d,False);m.set_requires_grad(a,True);a.train()
        opt.zero_grad(set_to_none=True)
        r=roll(ci,eps)
        law,terms=fn.law(m,r,fit,w,sc,p,a,bline,True)
        adv,_=m.actor_wasserstein_loss(d,r.scaled[:,1:],fit,idx)
        total=law+.5*adv
        if not torch.isfinite(total):raise FloatingPointError("actor")
        total.backward();grad=torch.nn.utils.clip_grad_norm_(a.parameters(),1.,error_if_nonfinite=True)
        opt.step();sched.step()
        row=dict(update=epoch,law=float(law.detach()),adversarial=float(adv.detach()),total=float(total.detach()),
            critic=float(au.loss.detach()),gp=float(au.gradient_penalty.detach()),actor_preclip_norm=float(grad),
            val_law="",val_timeW1="",val_occW1="",score="",elapsed=round(time.perf_counter()-start,2))
        del r,law,terms,adv,total
        if epoch==1 or epoch%4==0:
            v=np.asarray(validate());assert np.isfinite(v).all();score=float(np.dot(v/base,[.5,.25,.25]))
            row.update(val_law=v[0],val_timeW1=v[1],val_occW1=v[2],score=score)
            if score<best_score:
                best_score=score;best=dict(update=epoch,score=score,actor=copy.deepcopy(a.state_dict()),
                    critic=copy.deepcopy(d.state_dict()),config=cfg)
        rows.append(row)
        with (HERE/"training_history.csv").open("w",newline="") as f:
            wr=csv.DictWriter(f,fieldnames=row.keys());wr.writeheader();wr.writerows(rows)
        print("UPDATE",epoch,"/",PLAN,"law",row["law"],"val",row["score"],"elapsed",row["elapsed"],flush=True)
    assert best is not None
    destination=HERE/"trained_frozen_policy.pt"
    torch.save(best,destination)
    a.load_state_dict(torch.load(destination,map_location="cpu",weights_only=False)["actor"],strict=True)
    a.eval();m.set_requires_grad(a,False)
    dump("training_summary.json",dict(trained_updates=PLAN,selected_update=best["update"],score=best_score,
            val_neutral=base.tolist(),val_frozen=validate(),elapsed_seconds=time.perf_counter()-start,
            checkpoint_sha=sha(destination),original_plant_sha=sha(MODEL),
            critic_updates=24+3*PLAN,training_context_count=len(ctxs)))
    print("FROZEN",best["update"],best_score,flush=True)
    return a

def main():
    global HERE, MODEL, SOR, TRAIN
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo",type=Path,default=ROOT,help="Repository root containing part3_mfc/ and mfc_pipeline/")
    ap.add_argument("--arrays",type=Path,required=True,help="Restricted run01/run02 development_arrays.npz")
    ap.add_argument("--graph",type=Path,required=True,help="Trusted archived fold-1 graph model for the Part III input map")
    ap.add_argument("--predictor",type=Path,required=True,help="Trusted fold-1 surrogate (from fit_predictor.py)")
    ap.add_argument("--annotations",type=Path,required=True,help="Restricted channel and direct_paper_mask CSV")
    ap.add_argument("--anchor",type=Path,help="Optional independent run01 free rollout for exact verification")
    ap.add_argument("--checkpoint",type=Path,help="Optional trusted frozen checkpoint; replay without training")
    ap.add_argument("--out",type=Path,required=True,help="PRIVATE empty output directory")
    args=ap.parse_args()
    HERE=args.out.resolve()
    if HERE.exists() and any(HERE.iterdir()): ap.error("--out must be empty; existing results are never overwritten")
    HERE.mkdir(parents=True,exist_ok=True)
    repo=args.repo.resolve()
    sys.path.insert(0,str(repo));sys.path.insert(0,str(Path(__file__).resolve().parent))
    for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[k]="1"
    for k in ("MPLCONFIGDIR","MNE_HOME","_MNE_FAKE_HOME_DIR","JOBLIB_TEMP_FOLDER"):
        os.environ.setdefault(k,str(HERE/".runtime"/k))
    SOR=repo/"part3_mfc/model_core.py";TRAIN=repo/"part3_mfc/train_full_wgangp.py";MODEL=args.predictor
    import engine
    engine.HERE,engine.DATA,engine.FOLD_GRAPH,engine.MODEL,engine.MASK,engine.ANCHOR=HERE,args.arrays,args.graph,MODEL,args.annotations,args.anchor
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):
        m=load("official_core_hup060",SOR);fn=load("official_fresh_hup060",TRAIN)
        s=engine.build()
        if args.checkpoint:
            ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
            assert ck["config"]["plant_sha"]==sha(MODEL)
            # Accept the same source under Git's two supported line endings.
            core_lf=SOR.read_bytes().replace(b"\r\n",b"\n")
            assert ck["config"]["arch_source_sha"] in (hashlib.sha256(core_lf).hexdigest(),hashlib.sha256(core_lf.replace(b"\n",b"\r\n")).hexdigest())
            a=build_actor(s,m)[0];a.load_state_dict(ck["actor"],strict=True);a.eval();m.set_requires_grad(a,False)
        else:
            a=standard(s,m,fn)
        engine.export(s,a,m)

if __name__=="__main__":main()
