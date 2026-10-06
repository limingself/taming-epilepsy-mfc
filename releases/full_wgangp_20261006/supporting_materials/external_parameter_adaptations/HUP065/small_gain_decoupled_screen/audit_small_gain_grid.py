"""Independent frozen-state/dev metric/RMS arithmetic checks for six pairs."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import optimize_hup065 as exp


def main():
    runner=exp.load_runner()
    out=HERE/'results'
    protocol=json.loads((out/'registered_six_pair_protocol.json').read_text(encoding='utf-8'))
    frame=pd.read_csv(out/'six_pair_common_dev_results.csv')
    baseline=json.loads((out/'zero_baseline_metrics.json').read_text(encoding='utf-8'))
    reference=torch.load(exp.HERE/'runs'/'top32_covariance_u150'/'initial_actor.pt',map_location='cpu',weights_only=False)['actor_state_dict']
    original=torch.load(exp.OLD_RUN/'frozen_actor_wgan.pt',map_location='cpu',weights_only=False)['actor_state_dict']
    selected=reference['actuated_channel_indices'].numpy()
    old_channels=original['actuated_channel_indices'].numpy()
    old_slots=np.flatnonzero(np.isin(selected,old_channels))
    added=[(i,int(channel)) for i,channel in enumerate(selected) if channel not in old_channels]
    reports=[]
    for _,row in frame.iterrows():
        mean,deviation=float(row.new_mean_gain),float(row.new_deviation_gain)
        tag=f'mean{mean:g}_dev{deviation:g}'.replace('.','p')
        payload=torch.load(out/(tag+'_initial_actor.pt'),map_location='cpu',weights_only=False)
        state=payload['actor_state_dict']
        checks={}
        for name,tensor in reference.items():
            desired=tensor.clone()
            if name in ('mean_gain_delta','deviation_gain_delta'):
                coefficient=mean if name=='mean_gain_delta' else deviation
                for slot,channel in added:desired[:,slot,channel]=coefficient
                checks[name+'_old23_bitwise_unchanged']=torch.equal(state[name][:,old_slots],tensor[:,old_slots])
            checks[name+'_exact_registered_parameter_value']=torch.equal(state[name],desired)
        channels=pd.read_csv(out/(tag+'_dev_channel.csv'))
        groups=pd.read_csv(out/(tag+'_actuator_groups.csv'))
        trajectory=pd.read_csv(out/(tag+'_dev_trajectory.csv'))
        checks['128_dev_channel_rows']=len(channels)==128
        checks['64_actuator_rms_rows']=len(groups)==64
        for endpoint in ('time','occupation'):
            checks[endpoint+'_mean_W1_matches_channel_rows']=abs(channels[endpoint+'_w1_controlled'].mean()-row['mean_'+endpoint+'_w1_controlled'])<1e-12
        scores=[]
        for run,c in channels.groupby('run_index'):
            per=.5*c.time_w1_controlled.to_numpy()/np.maximum(c.time_w1_free.to_numpy(),.01)
            per+=.5*c.occupation_w1_controlled.to_numpy()/np.maximum(c.occupation_w1_free.to_numpy(),.01)
            scores.append(.4*c.time_w1_controlled.mean()/c.time_w1_free.mean()+
                .4*c.occupation_w1_controlled.mean()/c.occupation_w1_free.mean()+.2*np.quantile(per,.9))
            g=groups[groups.run_index.eq(run)]
            t=trajectory[trajectory.run_index.eq(run)].iloc[0]
            checks[f'run{run}_energy_matches_rms_square_sum']=abs((g.rms**2).sum()-t.total_energy)<1e-12
            checks[f'run{run}_global_rms_matches_delivered_coordinate_rms']=abs(np.sqrt((g.rms**2).mean())-t.mean_control_rms)<1e-12
            checks[f'run{run}_max_rms_matches_raw_coordinate_max']=abs(g.rms.max()-t.maximum_per_actuator_rms)<1e-12
            checks[f'run{run}_global_peak_matches_coordinate_peak']=abs(g.peak.max()-t.control_peak)<1e-12
        checks['common_score_independently_recomputed']=abs(np.mean(scores)-row.common_selection_score)<1e-12
        new=groups[groups.group.eq('new9')]
        checks['new9_actual_nonzero_count_matches']=int((new.groupby('channel_index').rms.min()>1e-12).sum())==int(row.active_new9_count_both_runs)
        checks['new9_mean_rms_matches']=abs(new.rms.mean()-row.new9_mean_actuator_rms)<1e-12
        checks['old_budget_rms_and_energy_pass']=bool(trajectory.maximum_per_actuator_rms.max()<=.405+1e-9 and trajectory.total_energy.max()<=3.7908+1e-9)
        eligible=bool(row.budget_pass and row.common_selection_score<baseline['common_selection_score']-1e-8 and
            row.mean_time_w1_controlled<=baseline['mean_time_w1_controlled']+1e-12 and
            row.mean_occupation_w1_controlled<=baseline['mean_occupation_w1_controlled']+1e-12)
        checks['eligibility_matches_registered_two_endpoint_criterion']=eligible==bool(row.eligible_for_adaptation)
        reports.append(dict(pair=[mean,deviation],checks={k:bool(v) for k,v in checks.items()},
            passed=all(checks.values()),passed_count=int(sum(checks.values())),total_checks=len(checks)))
    report=dict(status='passed' if all(r['passed'] for r in reports) else 'failed',
        registered_pairs_exact=protocol['pairs']==[[.05,.05],[.1,.1],[.2,.2],[0.,.05],[0.,.1],[.05,0.]],
        frozen_inputs_unchanged=all(runner.sha(k)==v for k,v in protocol['input_hashes'].items()),
        outer_arrays_opened=False,ctx6_arrays_opened=False,independent_tensor_and_arithmetic_checks=True,pairs=reports)
    runner.dump(out/'six_pair_independent_audit.json',report)
    print(json.dumps(runner.ready(report)),flush=True)
    if report['status']!='passed' or not report['registered_pairs_exact'] or not report['frozen_inputs_unchanged']:
        raise RuntimeError('Six-pair independent audit failed')


if __name__=='__main__':main()
