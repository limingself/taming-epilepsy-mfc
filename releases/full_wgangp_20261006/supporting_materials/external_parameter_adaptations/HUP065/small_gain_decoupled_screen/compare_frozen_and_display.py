"""Read-only old23 versus final32 comparison; never generate another rollout."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def long_path(path):
    value=str(Path(path).absolute())
    if os.name=='nt' and not value.startswith('\\\\?\\'):value='\\\\?\\'+value
    return Path(value)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


HERE=long_path(Path(__file__).resolve().parent)
OLD=HERE/'portable_inputs'/'old23_source_run'
NEW=HERE/'terminal_once'/'final_evaluation'
results=HERE/'results'
old_report=json.loads((OLD/'outer_posthoc_amendment'/'evaluation_summary.json').read_text(encoding='utf-8'))
new_report=json.loads((NEW/'final_status.json').read_text(encoding='utf-8'))
with np.load(OLD/'outer_posthoc_amendment'/'display_context_rollout.npz',allow_pickle=False) as archive:
    old={key:np.asarray(archive[key]) for key in archive.files}
with np.load(NEW/'outer_posthoc_amendment'/'display_context_rollout.npz',allow_pickle=False) as archive:
    new={key:np.asarray(archive[key]) for key in archive.files}
unchanged={name:float(np.max(np.abs(old[name]-new[name]))) for name in
    ('observed_standardized','free_standardized','reference_standardized','standard_normal')}
if max(unchanged.values())!=0.:raise RuntimeError('Recorded/free/reference/noise display data changed')
old_selected=old['selected_indices']
new_selected=new['selected_indices']
old_mask=np.isin(new_selected,old_selected)
controls_old,controls_new=old['controls'],new['controls']
old_rms=np.sqrt(np.mean(controls_old**2,axis=(0,1)))
new_rms=np.sqrt(np.mean(controls_new**2,axis=(0,1)))
rows=[]
for slot,channel in enumerate(new_selected):
    prior_slot=np.flatnonzero(old_selected==channel)
    prior=float(old_rms[prior_slot[0]]) if len(prior_slot) else 0.
    rows.append(dict(channel_index=int(channel),channel=str(new['channels'][channel]),
        group='old23' if old_mask[slot] else 'new9',old_display_actuator_rms=prior,new_display_actuator_rms=float(new_rms[slot]),
        new_display_peak=float(np.abs(controls_new[:,:,slot]).max()),new_display_active=bool(new_rms[slot]>1e-12)))
pd.DataFrame(rows).to_csv(results/'old23_new9_display_input_comparison.csv',index=False)
baseline=torch.load(HERE.parent/'runs'/'top32_covariance_u150'/'initial_actor.pt',map_location='cpu',weights_only=False)['actor_state_dict']
frozen_path=results/'selected_pair_adaptation_u150'/'frozen_actor_wgan.pt'
final=torch.load(frozen_path,map_location='cpu',weights_only=False)['actor_state_dict']
initial_pair=torch.load(results/'mean0p05_dev0_initial_actor.pt',map_location='cpu',weights_only=False)['actor_state_dict']
selected0_exact=all(torch.equal(final[name],initial_pair[name]) for name in final)
if not selected0_exact:raise RuntimeError('Selected0 is not the exact eligible initialization')
old_slots=np.flatnonzero(old_mask)
gain_changes={name:float((final[name][:,old_slots]-baseline[name][:,old_slots]).abs().max()) for name in ('mean_gain_delta','deviation_gain_delta')}
channel_frame=pd.read_csv(NEW/'outer_posthoc_amendment'/'all_channel_context_bank_metrics.csv')
channel_means=channel_frame.groupby('channel_index').mean(numeric_only=True)
avg_both=(channel_means.time_w1_controlled<channel_means.time_w1_free)&(channel_means.occupation_w1_controlled<channel_means.occupation_w1_free)
persistent=channel_frame.groupby('channel_index').both_time_and_occupation_improved.all()
endpoint={}
for kind in ('time','occupation'):
    old_value=old_report[f'mean_{kind}_w1_controlled']
    new_value=new_report[f'mean_{kind}_w1_controlled']
    free=new_report[f'mean_{kind}_w1_free']
    endpoint[kind]=dict(old23_controlled=old_value,new32_controlled=new_value,free=free,
        old23_reduction_percent=100*(1-old_value/free),new32_reduction_percent=100*(1-new_value/free),
        relative_distance_decrease_percent=100*(1-new_value/old_value))
safety=pd.read_csv(NEW/'outer_posthoc_amendment'/'trajectory_safety_metrics.csv')
report=dict(status='read_only_completed_frozen_comparison',new_checkpoint_sha256=sha(frozen_path),
    old23_controller_sha256=old_report['checkpoint_sha256'],outer_endpoints=endpoint,
    new_all24_old_budget_pass=bool(safety.gate_c.all()),original_RMS_cap=.405,original_energy_cap=3.7908,
    maximum_outer_per_actuator_rms=float(safety.maximum_per_actuator_rms.max()),maximum_outer_energy=float(safety.total_energy.max()),
    mean_total_energy_old23=old_report['total_energy'],mean_total_energy_new32=new_report['total_energy'],
    new_full_gateB_count=new_report['gate_b_pass_count'],new_persistent_both_endpoint_count=int(persistent.sum()),
    new_cross_case_mean_both_endpoint_count=int(avg_both.sum()),
    new_nondirect_persistent_both_endpoint_count=int(persistent.loc[np.flatnonzero(~new['direct_mask'])].sum()),
    new_nondirect_cross_case_mean_both_endpoint_count=int(avg_both.loc[np.flatnonzero(~new['direct_mask'])].sum()),
    new_nondirect_count=int((~new['direct_mask']).sum()),
    predeclared_display_case_only=dict(context=7,bank=0,old23_mean_actuator_rms=float(old_rms.mean()),
        old23_mean_delivered_rms_under_new_controller=float(new_rms[old_mask].mean()),
        old23_max_delivered_rms_under_new_controller=float(new_rms[old_mask].max()),
        new9_mean_actuator_rms=float(new_rms[~old_mask].mean()),new9_max_actuator_rms=float(new_rms[~old_mask].max()),
        new9_delivered_energy=float((new_rms[~old_mask]**2).sum()),new9_active_actuator_count=int((new_rms[~old_mask]>1e-12).sum()),
        all32_active_actuator_count=int((new_rms>1e-12).sum()),
        total_energy_old23=float((old_rms**2).sum()),total_energy_new32=float((new_rms**2).sum()),
        data_parity_max_errors=unchanged),
    old23_gain_parameter_max_change=gain_changes,selected_added_update=0,
    all_selected_actor_tensors_match_initialization_bitwise=selected0_exact,
    no_additional_training_gain_claim=True,no_all_channel_restoration_claim=True,
    outer_arrays_re_evaluated=False,classification='one-time post-hoc exploratory diagnostic of a development-selected candidate',
    no_new_loss_figure=True,published_external_results_replaced=False)
(results/'old23_vs_final32_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
