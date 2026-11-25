"""Development-only HUP065 no-graph Part II, with a separately specified Part III input map."""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys
# Configured by train_policy.main(); no access to patient material at import time.
HERE=DATA=FOLD_GRAPH=MODEL=MASK=ANCHOR=None
import numpy as np
import torch
import joblib
from threadpoolctl import threadpool_limits
from scipy.stats import wasserstein_distance
from mfc_pipeline import part2_state_dependent_rc_sde as P2
sys.modules.setdefault('part2_state_dependent_base',P2)
from mfc_pipeline.square_wave_mfc import TorchGraphRCSDE
from mfc_pipeline.causal_ltv_riccati import FrozenGraphRCMarkovAdapter
from mfc_pipeline.causal_ltv_particle_rollout import FrozenIctalGraphRCBatchStepper
from mfc_pipeline.sequential_covariance_hjb import uncontrolled_particle_rollout

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def build():
    torch.set_num_threads(1)
    threadpool_limits(limits=1)
    with np.load(DATA,allow_pickle=False) as z:
        assert set(z.files)=={'ictal_run_01','ictal_run_02','preictal_reference_run_01','preictal_reference_run_02','channels','sfreq'}
        assert float(z['sfreq'])==256
        # Run01 arrays are intentionally not read at build/train time.
        tr=z['ictal_run_02']; refs=z['preictal_reference_run_02']; names=z['channels'].astype(str)
    assert tr.shape==(8192,64) and refs.shape==(10240,64)
    with MASK.open(encoding='utf-8-sig',newline='') as f: ann=list(csv.DictReader(f))
    assert [r['channel'] for r in ann]==list(names)
    sel=np.flatnonzero([r['direct_paper_mask']=='True' for r in ann])
    assert len(sel)==32
    assert sha(FOLD_GRAPH)=='a55366d14a55e89ef4993cd46f5381e4468997e9d7c8535fe51709cb37542ed4'
    graph=joblib.load(FOLD_GRAPH); m=joblib.load(MODEL)
    assert m.q==graph.q==35 and np.count_nonzero(m.adjacency)==0
    np.testing.assert_allclose(graph.transform.pca.components_,m.transform.pca.components_,rtol=1.e-11,atol=1.e-13)
    world=TorchGraphRCSDE(m,sel,256.,control_graph_diffusion_time=0.,preserve_physical_control_residual=True,dtype=torch.float64)
    # Changing the predictor should NOT change the Part III heat operator: compute
    # the source heat-map with its native implementation, never install its graph in f or Sigma.
    ctl=TorchGraphRCSDE(graph,sel,256.,control_graph_diffusion_time=.2,preserve_physical_control_residual=True,dtype=torch.float64)
    world.control_channel_map=ctl.control_channel_map.detach().clone()
    world.control_map=(world.control_channel_map@world.components.T).contiguous()
    world.control_graph_diffusion_time=.2
    a=FrozenGraphRCMarkovAdapter(world,control_step_scale=.5,dtype=torch.float64)
    b=FrozenIctalGraphRCBatchStepper(world,a,diffusion_scale=.79451175)
    assert not a.adjacency.count_nonzero().item() and torch.any(a.control_map!=0)
    rs=m.transform.scaler.transform(refs).reshape(40,256,64)
    starts=(1550,3250,5150,7250)
    contexts=[a.initial_state_from_context(tr[i-256:i]) for i in starts]
    return dict(model=m,adapter=a,world=world,stepper=b,selected=sel,channels=names,
                initial_train=contexts[0],initial_val=contexts[-1],
                train_contexts=contexts[:3],fit=torch.as_tensor(rs[:15],dtype=torch.float64),
                val=torch.as_tensor(rs[15:30],dtype=torch.float64))

def display(s):
    step=5150; bank=1; fold=1
    # This function is for use after freezing the actor only.
    with np.load(DATA,allow_pickle=False) as z:
        tr=z['ictal_run_01']; refs=z['preictal_reference_run_01']
    sc=s['model'].transform.scaler
    hist=tr[step-256:step]
    y=sc.transform(tr[step:step+256])
    initial=s['adapter'].initial_state_from_context(hist)
    half=np.random.default_rng(202611240+23*step+50003*bank+1000003*fold).standard_normal((16,256,64))[:,:,:s['stepper'].q]
    eps=torch.as_tensor(np.concatenate([half,-half]),dtype=torch.float64)
    prev=None
    if ANCHOR is not None:
        with np.load(ANCHOR,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['channels'].astype(str),s['channels'])
            np.testing.assert_array_equal(z['observed'],y)
            prev=z['no_graph']
    return dict(initial_display=initial,noise_display=eps,observed=y,blue_previous=prev,reference_eval=sc.transform(refs).reshape(40,256,64)[15:30])

def export(s,actor,core):
    actor.eval()
    d=display(s)
    with torch.no_grad():
        free=uncontrolled_particle_rollout(s['stepper'],d['initial_display'],d['noise_display'])[:,1:].numpy()
        if d['blue_previous'] is not None: np.testing.assert_array_equal(free,d['blue_previous'])
        con=core.empirical_fp_rollout(s['stepper'],actor,d['initial_display'],d['noise_display'])
    controlled=con.scaled[:,1:].detach().numpy()
    controls=con.controls.detach().numpy()
    commands=con.commands.detach().numpy()
    ref=d['reference_eval']
    assert controlled.shape==free.shape==(32,256,64) and controls.shape==(32,256,32) and commands.shape==controls.shape
    assert np.isfinite(controlled).all() and np.isfinite(controls).all() and np.abs(controls).max()<=1.8+1e-10
    np.savez_compressed(HERE/'display_control.npz',observed=d['observed'],free=free,controlled=controlled,reference_eval=ref,
                        channels=s['channels'],selected=s['selected'],controls=controls,commands=commands)
    tf=core.channelwise_time_w1(free,ref);tc=core.channelwise_time_w1(controlled,ref)
    of=core.channelwise_occupation_w1(free,ref);oc=core.channelwise_occupation_w1(controlled,ref)
    wobs=np.array([wasserstein_distance(d['observed'][:,j],ref[:,:,j].ravel()) for j in range(64)])
    rows=[dict(channel=str(c),direct=j in s['selected'],time_free=tf[j],time_controlled=tc[j],occupation_free=of[j],
               occupation_controlled=oc[j],occupation_observed_reference=wobs[j]) for j,c in enumerate(s['channels'])]
    with (HERE/'engine_channel_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    peak=float(np.abs(controls).max()); rms=np.sqrt(np.mean(controls**2,axis=(0,1))); energy=float(np.mean(np.sum(controls**2,axis=-1)))
    sat=float(np.mean(np.abs(controls)>=.95*1.8))
    stats=dict(zero_replay_bit_exact=None if d['blue_previous'] is None else bool(np.array_equal(free,d['blue_previous'])),reference_run01_segments=len(ref),
               full_time_W1=[tf.mean(),tc.mean()],full_occupation_W1=[of.mean(),oc.mean()],observed_reference_occupation_mean=wobs.mean(),
               both_improved=int(((tc<tf)&(oc<of)).sum()),controlled_closer_than_observed_occupation=int((oc<wobs).sum()),
               time_improved=int((tc<tf).sum()),occupation_improved=int((oc<of).sum()),
               max_abs_z_free=float(np.abs(free).max()),max_abs_z_controlled=float(np.abs(controlled).max()),
               population_control_rms=float(np.sqrt(np.mean(controls**2))),maximum_actuator_rms=float(rms.max()),total_energy=energy,peak=peak,saturation=sat,
               archived_hup065_budget_pass=bool(rms.max()<=.405+1e-9 and energy<=3.7908+1.e-9 and sat<.01 and peak<=1.8+1e-9),
               no_outer_access=True,model_sha256=sha(MODEL),blue_cache_sha256=sha(ANCHOR) if ANCHOR is not None else None)
    (HERE/'engine_eval.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print('Closed loop for frozen actor:',json.dumps(stats),flush=True)
    return stats
