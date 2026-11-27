"""Predeclared run01-only noise calibration; no input path for any other run."""
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from fit_run01 import OUT, source, p2, pipe, digest

KAPPAS=(.5,.75,1.,1.25)
C=.79451175
H=256
BOUNDARIES=tuple(range(25*H,32*H,H))
SEED=2026112701

def closed_loop(m,ctx,noise,scale):
    n,H,q=noise.shape
    state,hist,top=m._context_state(ctx)
    states=np.tile(state,(n,1)); topology=np.tile(top,(n,1))
    h=[np.tile(z,(n,1)) for z in hist[-33:]]
    paths=[]
    for t in range(H):
        feature=np.concatenate([states,h[-1],topology,*[h[-1-d] for d in m.config.delays_samples]],axis=1)
        f=m.feature_scaler.transform(feature)
        z=h[-1]+m.drift.predict(f)+(noise[:,t]@m.diffusion_correlation_root.T)*m._state_dependent_diffusion_std(f)*scale
        x=m._scaled_from_latent(z)
        topology=m._topology(x)
        h.append(z)
        drive=np.concatenate([z,topology,*[h[-1-d] for d in m.config.delays_samples]],axis=1)
        states=(1-m.config.leak_rate)*states+m.config.leak_rate*np.tanh(drive@m.w_in.T+states@m.w_res.T)
        paths.append(x)
    return np.stack(paths,axis=1)

def main():
    if (OUT/'noise_selection.json').exists(): raise FileExistsError('Selection already frozen.')
    freeze=dict(kappas=KAPPAS,global_C=C,run='run-01 only',source_sha256=digest(source),
                inner_fit_seconds=[0,22],gap_seconds=[22,24],validation_seconds=[24,32],
                absolute_boundaries_in_ictal_samples=BOUNDARIES,history=H,horizon=H,particles=32,antithetic=True,
                assessment='first validation second reserved for history; then seven nonoverlapping one-second futures, no within-window refresh',
                selection='lowest average per-window per-channel exact empirical occupation W1 in inner-training standardized units; tie favors closest to kappa=1 then lowest kappa',
                seeds=[SEED+i for i,_ in enumerate(BOUNDARIES)],selection_bank='common independent normalized innovations per window, paired across scale candidates')
    (OUT/'noise_grid_plan.json').write_text(json.dumps(freeze,indent=2),encoding='utf8')
    cfg=p2.ModelConfig(**json.loads((OUT/'fit_manifest.json').read_text(encoding='utf8'))['actual_frozen_config'])
    with np.load(source,allow_pickle=False) as f:
        x=np.array(f['data'].T,order='C',dtype='float64')
    tr=x[:22*H]
    m=p2.ResidualGraphRCSDE(cfg,pipe.compute_adjacency([tr],H)).fit([tr])
    rows=[]
    for i,b in enumerate(BOUNDARIES):
        ctx=x[b-H:b]; obs=m.transform.standardize(x[b:b+H])
        r=np.random.default_rng(SEED+i).normal(size=(16,H,m.q))
        z=np.concatenate([r,-r])
        for k in KAPPAS:
            paths=closed_loop(m,ctx,z,k*C)
            d=np.mean(np.abs(np.sort(paths.reshape(-1,x.shape[1]),axis=0)-np.repeat(np.sort(obs,axis=0),32,axis=0)),axis=0)
            rows.extend(dict(kappa=k,window=i,boundary=b,channel=j,occupation_w1=w) for j,w in enumerate(d))
    table=pd.DataFrame(rows)
    assert len(table)==len(BOUNDARIES)*len(KAPPAS)*x.shape[1] and np.isfinite(table['occupation_w1']).all()
    table.to_csv(OUT/'noise_grid_run01_channel_window.csv',index=False)
    summary=table.groupby(['kappa','window']).occupation_w1.mean().unstack()
    summary['mean']=summary.mean(axis=1)
    summary.to_csv(OUT/'noise_grid_run01_summary.csv')
    k=min(KAPPAS,key=lambda k:(summary.loc[k,'mean'],abs(k-1),k))
    (OUT/'noise_selection.json').write_text(json.dumps(dict(**freeze,selected_kappa=k,effective_multiplier=k*C,
                 canonical_score=summary.loc[1.,'mean'],selected_score=summary.loc[k,'mean'],
                 compared={str(k):summary.loc[k,'mean'] for k in KAPPAS},inner_q=m.q,
                 notes='only scalar chosen; model structure, full-run01 fitting and held-out boundary are frozen'),indent=2),encoding='utf8')
    print(summary)

if __name__=='__main__':
    with threadpool_limits(limits=1): main()
