"""Matched four-level feature ablation for the HUP060 fixed Fig. 3 autonomous prediction.

Provide the matched run-01 training cache and a fresh output directory.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--training-cache', type=Path, required=True, help='Matched 32-s HUP060 run-01 training cache, not a forecast window.')
parser.add_argument('--out', type=Path, required=True, help='New or empty local directory. Frozen inputs are never overwritten.')
args=parser.parse_args()
project, out = args.project.resolve(), args.out.resolve()
if out.exists() and any(out.iterdir()):
    parser.error('--out must be new or empty')
out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(project))
os.environ.update(MPLCONFIGDIR=str(out/'mplcache'), _MNE_FAKE_HOME_DIR=str(out/'runtime'),
                  MNE_LOGGING_LEVEL='WARNING', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  OMP_NUM_THREADS='1')
sys.dont_write_bytecode = True
import joblib
import numpy as np
import scipy
import sklearn
import torch
from scipy.stats import wasserstein_distance
from threadpoolctl import threadpool_limits
torch.set_num_threads(1)

from types import SimpleNamespace
from mfc_pipeline import part2_state_dependent_rc_sde as m
from mfc_pipeline.square_wave_mfc import TorchGraphRCSDE
from mfc_pipeline.causal_ltv_particle_rollout import FrozenIctalGraphRCBatchStepper, reconstruct_paired_normals
from mfc_pipeline.causal_ltv_riccati import FrozenGraphRCMarkovAdapter
sys.modules.setdefault('part2_state_dependent_base',m)
def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
art = project/'artifacts'
g=SimpleNamespace(MODEL_PATH=art/'part2_hup060_state_dependent_optimized/hup060_state_dependent_rc.joblib',
    UNIFIED_ROLLOUT_PATH=art/'part3_hup060_actor_wgan_v1/seed20261011_adv050_anchor020/run02_development_evaluation/paired_comparison.npz',
    SYNTHESIS_CONTRACT_PATH=art/'part3_hup060_causal_synthesis_contract_v1/synthesis_only_inputs.npz',
    SEALED_ROLLOUT_PATH=art/'part3_hup060_neural_hjb_fp/physical_residual_identity_run02_context3_frozen_e5_v1/test_rollout.npz',
    SOURCE=project/'output/part2/source_data/figure_05',sha256=sha256,REPEATS=32,HORIZON=256,
    DIFFUSION_SCALE=.79451175,CONTROL_STEP_SCALE=.5,GRAPH_DIFFUSION_TIME=.5,
    TorchGraphRCSDE=TorchGraphRCSDE,FrozenGraphRCMarkovAdapter=FrozenGraphRCMarkovAdapter,
    FrozenIctalGraphRCBatchStepper=FrozenIctalGraphRCBatchStepper,reconstruct_paired_normals=reconstruct_paired_normals)
original = joblib.load(g.MODEL_PATH)
expected = {
    "full_part2_model": "a2984a7f1042f2fff1231d744405f76ca8b649c8048812d7f48249a3e02e1e43",
    "unified_part3_paired_comparison": "bb5e43d4b8774d144e3fd598650ee1653e22f8391764467b13e403d2577d8541",
    "sealed_part3_zero_control_rollout": "a36b4bd5e14f7cce91e5f7fe495f0574715452e35942f314e3524f927bb8af6c",
    "part3_synthesis_contract": "4f341fd61ff3104d56dedf7a7cf9f894823299eb4f5c6499102f258ebdc17edc",
    "training_cache": "a2a378b967c35122ac78d03708f5484b2cc3e560461af4279e4651a93df7f632"
}
paths = {'full_part2_model':g.MODEL_PATH,'unified_part3_paired_comparison':g.UNIFIED_ROLLOUT_PATH,
         'sealed_part3_zero_control_rollout':g.SEALED_ROLLOUT_PATH,'part3_synthesis_contract':g.SYNTHESIS_CONTRACT_PATH}
for key,path in paths.items():
    assert g.sha256(path) == expected[key], key
with np.load(g.UNIFIED_ROLLOUT_PATH) as f:
    channels=f['channels'].astype(str); obs=f['observed_scaled'].astype(float)
    archived_full=f['uncontrolled_scaled'].astype(float)
    selected=f['selected_indices']; sf=float(f['sampling_rate_hz'].item())
with np.load(g.SYNTHESIS_CONTRACT_PATH) as f:
    context=f['past_context_scaled']; origin=int(f['forecast_origin'].item())
    assert np.array_equal(f['channels'].astype(str),channels)
with np.load(g.SEALED_ROLLOUT_PATH) as f: sealed=f['uncontrolled_scaled'][:,1:]
trainfile = args.training_cache.resolve()
assert sha256(trainfile)==expected['training_cache'], 'training cache does not match the fitted run-01 excerpt'
with np.load(trainfile) as f:
    train=f['data'].T.astype(float)
    assert np.array_equal(f['channel_names'].astype(str),channels)
assert train.shape==(8192,36) and origin==3547 and sf==256 and original.config.delays_samples==(1,8,32)


orig, past, sel, observed, archived, trainpath = original, context, selected, obs, archived_full, trainfile
@torch.no_grad()
def paired_noise():
    world=TorchGraphRCSDE(orig,sel,sf,control_graph_diffusion_time=g.GRAPH_DIFFUSION_TIME,
                          preserve_physical_control_residual=True,dtype=torch.float64,device="cpu")
    adapter=FrozenGraphRCMarkovAdapter(world,control_step_scale=g.CONTROL_STEP_SCALE,dtype=torch.float64)
    stepper=FrozenIctalGraphRCBatchStepper(world,adapter,diffusion_scale=g.DIFFUSION_SCALE)
    return reconstruct_paired_normals(stepper,adapter.initial_state_from_context(past),sealed,tolerance=2e-6)
with threadpool_limits(limits=1):
    noise, reconstruction = paired_noise()
class NoRC(m.ResidualGraphRCSDE):
    def _initialise_reservoir(self):
        assert self.config.reservoir_size==0
        self.w_res=np.zeros((0,0));self.w_in=np.zeros((0,self.q*(2+len(self.config.delays_samples))))
    def _fit_direct_head(self,*args):pass  # never used by the paired autonomous experiment

@torch.no_grad()
def rollout(model):
    world=g.TorchGraphRCSDE(model,sel,256.,control_graph_diffusion_time=g.GRAPH_DIFFUSION_TIME,preserve_physical_control_residual=True,dtype=torch.float64,device='cpu')
    adapter=g.FrozenGraphRCMarkovAdapter(world,control_step_scale=g.CONTROL_STEP_SCALE,dtype=torch.float64)
    stepper=g.FrozenIctalGraphRCBatchStepper(world,adapter,diffusion_scale=g.DIFFUSION_SCALE)
    init=adapter.initial_state_from_context(past)
    assert len(init[adapter.slices.reservoir])==model.config.reservoir_size
    rec=stepper.repeat_initial(init,len(noise)); u=torch.zeros(len(noise),stepper.actuator_dim,dtype=rec.dtype)
    err=None
    if model.config.reservoir_size==0:
        keep=1+model.maximum_delay
        changed=past.copy();changed[:-keep]+=0.873
        perturbed=adapter.initial_state_from_context(changed)
        err=float(torch.max(torch.abs(init-perturbed)))
        assert err<1e-12
    vals=[]
    for t in range(256):
        r=stepper.step(rec,u,noise[:,t]); rec=r.next_markov; vals.append(r.next_scaled.numpy())
    return np.stack(vals,axis=1),err
pred={}; specs={};parity={}
with threadpool_limits(limits=1):
    for key,delays,graph,use_rc in [('state',(),False,False),('delay',(1,8,32),False,False),
                                    ('graph',(1,8,32),True,False),('full',(1,8,32),True,True)]:
        if use_rc:
            model=orig
        else:
            cfg=replace(orig.config,reservoir_size=0,delays_samples=delays,washout_samples=64-max(delays,default=0))
            model=NoRC(cfg,orig.adjacency_input if graph else np.zeros_like(orig.adjacency_input)).fit([train])
        assert model.q==13 and model.training_pairs==8127 and model.maximum_delay+model.config.washout_samples==64
        np.testing.assert_allclose(model.transform.scaler.mean_,orig.transform.scaler.mean_,atol=1e-12,rtol=0)
        np.testing.assert_allclose(model.transform.components_,orig.transform.components_,atol=1e-10,rtol=0)
        assert (bool(np.count_nonzero(model.adjacency)))==graph
        if not graph:
            assert not np.count_nonzero(model._topology(model.transform.standardize(train)))
            # The simulator's fixed protocol carries q zero graph placeholders;
            # they are constant in training and have no readout contribution.
            assert np.max(abs(model.drift.coef_[:,13:26])) < 1e-12
            assert np.max(abs(model.diffusion_variance_model.coef_[:,13:26])) < 1e-12
        y,older_history_difference=rollout(model)
        assert y.shape==(32,256,36) and np.isfinite(y).all()
        pred[key]=y
        specs[key]=dict(delays=list(delays),graph=graph,reservoir_units=model.config.reservoir_size,
                        washout=model.config.washout_samples,training_starts_at=64,training_stops_at=8190,
                        train_pairs=model.training_pairs,stored_features=int(model.feature_scaler.n_features_in_),
                        active_feature_slots=int(model.feature_scaler.n_features_in_)-(0 if graph else 13),
                        earlier_history_perturbation_max_initial_state_difference=older_history_difference)
        print(key,'forecast done',specs[key],flush=True)
    err=float(np.abs(pred["full"]-archived).max())
    if err > 2e-6:
        raise RuntimeError(f"Full model disagrees with frozen u=0 branch: {err:.8g}")
    parity["full_archive"] = err
def score(y): return np.array([wasserstein_distance(observed[:,i],y[...,i].ravel()) for i in range(36)])
w={k:score(v) for k,v in pred.items()}
np.savez_compressed(out/'nested_runs.npz',channels=channels,observed=observed,**pred,positions=[3547])
with (out/'per_contact.csv').open('w',newline='',encoding='utf8') as f:
    writer=csv.writer(f);writer.writerow(['channel',*w]);writer.writerows(zip(channels,*w.values()))
stats={k:dict(mean=float(v.mean()),median=float(np.median(v)),min=float(v.min()),max=float(v.max())) for k,v in w.items()}
adjacent={new:dict(mean_change=float((w[new]-w[old]).mean()),n_better=int((w[new]<w[old]).sum()),n_worse=int((w[new]>w[old]).sum())) for old,new in zip(['state','delay','graph'],['delay','graph','full'])}
audit=dict(subject="HUP060",training="run-01 120-152 s; same retained covariate indices 64 through 8190",evaluation="run-02 development, context 3, forecast sample 3547; 32 paired particles for 256 samples without observation refresh; u=0", specs=specs,stats=stats,adjacent=adjacent,full_saved_rollout_parity_max_abs=parity["full_archive"],paired_diffusion_multiplier=g.DIFFUSION_SCALE,noise_reconstruction=reconstruction,noise_float64_sha256=hashlib.sha256(noise.numpy().tobytes()).hexdigest(),input_sha256=expected,code_sha256={"fit":sha256(Path(m.__file__)),"runner":sha256(__file__)},versions={"numpy":np.__version__,"torch":torch.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__})
(out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
print(json.dumps(stats),json.dumps(adjacent),parity,flush=True)
