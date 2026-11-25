"""Paired development-only HUP080 free forecasts, with matched patient-specific refits."""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
import argparse
import hashlib
import json
import os
import sys

from development_paths import FREE as ROOT, PROJECT as PAPER, ART
DATA = ART / "01_prepare_development/development_arrays.npz"
MODELS = ART / "02_loro_part1_part2"
AUDIT = os.environ.get("HUP080_HISTORICAL_AUDIT_CSV")
sys.dont_write_bytecode = True
sys.path.insert(0, str(PAPER))
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"
for var in ("MPLCONFIGDIR","MNE_HOME","_MNE_FAKE_HOME_DIR","JOBLIB_TEMP_FOLDER"):
    folder = ROOT/".runtime"/var
    folder.mkdir(parents=True,exist_ok=True)
    os.environ[var] = str(folder)

import numpy as np
import pandas as pd
import torch
import joblib
from scipy.stats import wasserstein_distance
from threadpoolctl import threadpool_limits
from mfc_pipeline import part2_state_dependent_rc_sde as p2
sys.modules.setdefault("part2_state_dependent_base",p2)
from mfc_pipeline.square_wave_mfc import TorchGraphRCSDE
from mfc_pipeline.causal_ltv_riccati import FrozenGraphRCMarkovAdapter
from mfc_pipeline.causal_ltv_particle_rollout import FrozenIctalGraphRCBatchStepper
from mfc_pipeline.sequential_covariance_hjb import uncontrolled_particle_rollout


def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf-8")


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def env(model, c):
    world=TorchGraphRCSDE(model,[0],256.,control_graph_diffusion_time=0.,
        preserve_physical_control_residual=True,dtype=torch.float64)
    adapt=FrozenGraphRCMarkovAdapter(world,control_step_scale=.5,dtype=torch.float64)
    return adapt,FrozenIctalGraphRCBatchStepper(world,adapt,diffusion_scale=c)


def metrics(z,y):
    n,h,c=z.shape
    pool=z.reshape(-1,c)
    w=np.abs(np.sort(pool,axis=0).reshape(h,n,c)-np.sort(y,axis=0)[:,None]).mean((0,1))
    # Equal-mass empirical W1, unbounded amplitudes, no KDE.
    amp=pool.std(0)
    recamp=y.std(0)
    ch=dict(occupation=w,point_to_obs=np.abs(z-y[None]).mean((0,1)),
        width_error=np.abs(amp-recamp),mean_error=np.abs(pool.mean(0)-y.mean(0)),
        median_error=np.abs(np.median(pool,axis=0)-np.median(y,axis=0)),
        predicted_sd=amp,observed_sd=recamp,
        ensemble_mean_mse=((z.mean(0)-y)**2).mean(0))
    return ch


def run(dest):
    dest.mkdir(exist_ok=True,parents=True)
    with (dest/"protocol.json").open("x",encoding="utf-8") as f:
        json.dump(dict(subject="HUP080",assessment="three leave-one-development-run-out folds",
          runs=[1,2,3],training="other two runs, standardization and PCA fitted on training only",
          models="adopted fold models vs zero-adjacency refit using those exact configs",
          control=False,teacher_forcing=False,outer_access=False,hyperparameter_search=False,
          first_display=dict(fold=1,boundary=5150,bank=1),
          primary_stops=[1550,3250,5150,7250],primary_banks=[1,2],
          secondary_stops=[2048,3264,6336],secondary_banks=[4,5],
          secondary_role="replays the previous HUP080 publication-baseline development audit; overlaps the primary in recording time, not additional independent samples",
          particles=32,history=256,horizon=256,fs=256,
          units="per-contact training-fold standard deviation; no clipping, scaling of individual curves or post-hoc model selection",
          flag="nonfinite or max absolute standardized output >=20",
          script_sha=sha(Path(__file__)),input_sha=sha(DATA),model_class_sha=sha(Path(p2.__file__))),f,indent=2)
    cfg=json.loads((MODELS/"selected_config.json").read_text())
    c=cfg["effective_diffusion_multiplier_for_part3"]
    assert cfg["subject"]=="HUP080" and c==.79451175
    old=(pd.read_csv(AUDIT).query('model=="published"').set_index(["fold","boundary","bank","channel"]) if AUDIT else None)
    with np.load(DATA,allow_pickle=False) as z:
        assert not any("04" in s or "outer" in s for s in z.files)
        runs={k:z[f"ictal_run_{k:02d}"] for k in (1,2,3)}
        labels=z["channels"].astype(str)
        assert all(seq.shape==(8192,96) for seq in runs.values()) and len(set(labels))==96
    allrows=[]; allch=[]; fits=[]
    for fold in (1,2,3):
        f=MODELS/"fold_models"/f"leave-run-{fold:02d}-out.joblib"
        digest=sha(f)
        fitruns=[k for k in runs if k!=fold]
        fc=cfg["fold_models"][fold-1]
        assert fc["model_sha256"]==digest and fc["training_runs"]==[f"run-{i:02d}" for i in fitruns]
        base=joblib.load(f)
        m=p2.ResidualGraphRCSDE(base.config,np.zeros_like(base.adjacency_input)).fit([runs[k] for k in fitruns])
        assert np.count_nonzero(base.adjacency)>0 and not np.count_nonzero(m.adjacency) and base.q==m.q==24
        assert base.training_pairs==m.training_pairs and m.config==base.config
        for a,b in [(m.transform.components_,base.transform.components_),(m.transform.scaler.mean_,base.transform.scaler.mean_),
                    (m.transform.scaler.scale_,base.transform.scaler.scale_),(m.w_res,base.w_res),(m.w_in,base.w_in)]:
            np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-13)
        j=slice(m.config.reservoir_size+m.q,m.config.reservoir_size+2*m.q)
        assert np.max(np.abs(m.feature_scaler.var_[j]))<1e-12
        assert np.max(np.abs(m.direct_coef[...,j]))<1e-10 and np.max(np.abs(m.direct_variance_coef[...,j]))<1e-10
        p=dest/f"free_fold{fold}.joblib"
        joblib.dump(m,p,compress=3)
        fits.append(dict(fold=fold,training_runs=fitruns,q=m.q,base_sha256=digest,free_sha256=sha(p),
                        zero_graph=True,graph_regressors_max_var=float(np.max(np.abs(m.feature_scaler.var_[j]))),
                        config=asdict(m.config)))
        write(dest/"fit_audit.json",fits)
        e={k:env(mm,c) for k,mm in (("baseline",base),("free",m))}
        print("fitted",fold,flush=True)
        for phase,stops,banks in (("primary",[5150,1550,3250,7250],[1,2]),("previous_080_windows",[6336,2048,3264],[4,5])):
            for t in stops:
                past=runs[fold][t-256:t]
                obs=base.transform.scaler.transform(runs[fold][t:t+256])
                initial={arm:a.initial_state_from_context(past) for arm,(a,st) in e.items()}
                assert not torch.count_nonzero(initial["free"][e["free"][0].slices.topology]).item()
                recent=base.transform.scaler.transform(past)
                for bank in banks:
                    if phase=="primary":
                        half=np.random.default_rng(202611240+23*t+50003*bank+1000003*fold).standard_normal((16,256,96))[:,:,:24]
                    else:
                        half=np.random.default_rng(25012011+349*fold+11*t+7919*bank).standard_normal((16,256,24))
                    eps=torch.as_tensor(np.concatenate((half,-half)),dtype=torch.float64)
                    show=(t==5150 and bank==1) or (t==6336 and bank==4)
                    cache={}
                    for arm,(a,st) in e.items():
                        with torch.no_grad():
                            z=uncontrolled_particle_rollout(st,initial[arm],eps)[:,1:].numpy()
                        assert z.shape==(32,256,96)
                        meta=dict(phase=phase,fold=fold,boundary=t,bank=bank,arm=arm)
                        if not np.isfinite(z).all():
                            allrows.append(dict(**meta,status="nonfinite",unstable=True))
                            if show:cache[arm]=z
                            continue
                        ch=metrics(z,obs)
                        if old is not None and phase=="previous_080_windows" and arm=="baseline":
                            prev=old.loc[(fold,t,bank)].reindex(labels)
                            for key,src in [("occupation","occupation"),("point_to_obs","time"),("width_error","width"),("mean_error","abs_bias")]:
                                np.testing.assert_allclose(ch[key],prev[src].to_numpy(),rtol=1e-9,atol=5e-11)
                        if show:
                            np.testing.assert_allclose(ch["occupation"],[wasserstein_distance(obs[:,i],z[:,:,i].ravel()) for i in range(96)],rtol=2e-13)
                            cache[arm]=z
                        metric={key:float(np.mean(vals)) for key,vals in ch.items()}
                        metric["rmse"]=float(np.sqrt(metric.pop("ensemble_mean_mse")))
                        metric["absmax"]=float(np.abs(z).max())
                        metric["median_of_median_errors"]=float(np.median(ch["median_error"]))
                        metric["persistence_occupation"]=float(np.mean(np.abs(np.sort(recent,axis=0)-np.sort(obs,axis=0))))
                        allrows.append(dict(**meta,**metric,status="finite",unstable=metric["absmax"]>=20))
                        allch.append(pd.DataFrame(dict(**meta,channel=labels,**ch)))
                    if show:
                        name="display" if (fold,t)==(1,5150) else f"display_fold{fold}_at{t}"
                        np.savez_compressed(dest/f"{name}.npz",observed=obs,labels=labels,fold=fold,boundary=t,bank=bank,fs=256,**cache)
                        print("arrays ready:",name,flush=True)
            pd.DataFrame(allrows).to_csv(dest/"condition_metrics.csv",index=False)
            pd.concat(allch).to_csv(dest/"channel_metrics.csv",index=False)
        print("finished fold",fold,flush=True)
    df=pd.DataFrame(allrows)
    assert len(df)==84
    print(df.groupby(["phase","arm"]).agg(occ=("occupation","mean"),max_abs=("absmax","max"),n=("status","size"),n_unstable=("unstable","sum")),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=ROOT)
    opt=parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):
        run(opt.out)
