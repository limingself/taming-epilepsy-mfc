"""HUP080 development-only 32-update mean/deviation WGAN-GP controller preview."""
from pathlib import Path
import argparse, copy, csv, hashlib, importlib.util, json, os, sys, time
from development_paths import PROJECT, ART, FREE as PREV, FIRST as HERE
DATA=ART/"01_prepare_development/development_arrays.npz"
GRAPH=ART/"02_loro_part1_part2/fold_part1_selection_networks/leave-run-01-out.npz"
SHELL=PROJECT/'patient_extensions/HUP065/development_s4/train_policy.py'
CORE=PROJECT/"part3_mfc/model_core.py"; TRAIN=PROJECT/"part3_mfc/train_full_wgangp.py"
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[key]="1"
for key in ("MPLCONFIGDIR","MNE_HOME","_MNE_FAKE_HOME_DIR","JOBLIB_TEMP_FOLDER"):
    p=HERE/".runtime"/key;p.mkdir(exist_ok=True,parents=True);os.environ[key]=str(p)
sys.dont_write_bytecode=True;sys.path.insert(0,str(PROJECT))
import numpy as np, pandas as pd, torch, joblib
from scipy.stats import wasserstein_distance
from threadpoolctl import threadpool_limits
from mfc_pipeline import part2_state_dependent_rc_sde as P2
sys.modules.setdefault("part2_state_dependent_base",P2)
from mfc_pipeline.square_wave_mfc import TorchGraphRCSDE
from mfc_pipeline.causal_ltv_riccati import FrozenGraphRCMarkovAdapter
from mfc_pipeline.causal_ltv_particle_rollout import FrozenIctalGraphRCBatchStepper

def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
N=32;SEED=20261011; TRAIN_STOPS=[1550,3250,5150]; VAL_STOP=7250; TEST_STOPS=[5150,1550,3250,7250]
C=.79451175
def w1s(x,r):
    # Integrate absolute quantile errors across the union of empirical knots.
    p,q=len(x),len(r);l=np.lcm(p,q)
    knots=np.unique(np.r_[np.arange(p+1)*(l//p),np.arange(q+1)*(l//q)])
    sx=np.sort(x,axis=0);sr=np.sort(r,axis=0)
    t=sum(dw*np.abs(sx[k*p//l]-sr[k*q//l]).mean(0) for k,dw in zip(knots[:-1],np.diff(knots)/l))
    o=np.array([wasserstein_distance(x[:,:,i].ravel(),r[:,:,i].ravel()) for i in range(96)])
    return t,o

def inputs(u):
    rms=np.sqrt(np.mean(u**2,(0,1))); peak=float(np.max(abs(u)));energy=float(np.sum(rms**2))
    sat=float(np.mean(abs(u)>=1.8*.95));mr=float(max(rms))
    valid=np.isfinite(u).all() and energy<=5.6862 and peak<=1.8+1e-12 and sat<.01
    return dict(global_rms=float(np.sqrt(np.mean(u**2))),max_rms=mr,energy=energy,peak=peak,sat=sat,
                budget_ok=bool(valid and mr<=.45),old_budget_ok=bool(valid and mr<=.405))
