"""Read-only independent tensor/ranking/contract audit; no outer reads."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import optimize_hup065 as exp


def main():
    runner=exp.load_runner()
    checkpoint_path=exp.OLD_RUN/'frozen_actor_wgan.pt'
    old=torch.load(checkpoint_path,map_location='cpu',weights_only=False)['actor_state_dict']
    rows=[]
    for directory in sorted((exp.HERE/'runs').iterdir()):
        initial_path=directory/'initial_actor.pt'
        if not initial_path.exists():continue
        payload=torch.load(initial_path,map_location='cpu',weights_only=False)
        new=payload['actor_state_dict']
        contract=payload['training_contract']
        old_channels=old['actuated_channel_indices'].numpy()
        new_channels=new['actuated_channel_indices'].numpy()
        if len(old_channels)!=23 or len(new_channels)!=32:raise RuntimeError('Unexpected actuator counts')
        lookup={int(c):i for i,c in enumerate(new_channels)}
        slots=torch.tensor([lookup[int(c)] for c in old_channels])
        added=torch.tensor([i for i,c in enumerate(new_channels) if c not in old_channels])
        mask_path=runner.paths('HUP065')['network']
        with np.load(mask_path,allow_pickle=False) as z:
            score=z['centrality_score']
        order=np.lexsort((np.arange(64),-score))
        checks={
            'top32_exact':np.array_equal(new_channels,np.sort(order[:32])),
            'top23_original_exact':np.array_equal(old_channels,np.sort(order[:23])),
            'original_model_hash':contract['model_sha256']==runner.MODEL_HASHES['HUP065'],
            'not_from_scratch':contract['initialization']['initialized_from_scratch'] is False,
            'original_rms_cap':contract['per_actuator_rms_cap']==.405,
            'original_energy_cap':contract['input_energy_cap']==3.7908,
            'original_peak_cap':contract['amplitude_limit']==1.8,
            'original_noise':contract['diffusion_scale']==.79451175,
            'original_step_scale':contract['control_step_scale']==.5,
            'outer_not_used':contract['outer_used_for_training_or_selection'] is False,
            'no_runtime_projection':contract['no_runtime_projection_or_rescaling'] is True,
        }
        for name in ('mean_gain_delta','deviation_gain_delta'):
            checks[name+'_old_rows_exact']=torch.equal(new[name][:,slots],old[name])
            checks[name+'_new_rows_zero']=torch.count_nonzero(new[name][:,added]).item()==0
        for prefix in ('common_residual','deviation_residual','common_markov_residual','deviation_markov_residual'):
            name=prefix+'.0.weight'
            prefix_dim=old[name].shape[1]-23
            checks[name+'_shared_features_exact']=torch.equal(new[name][:,:prefix_dim],old[name][:,:prefix_dim])
            checks[name+'_old_control_slots_exact']=torch.equal(new[name][:,prefix_dim+slots],old[name][:,prefix_dim:])
            checks[name+'_new_control_slots_zero']=torch.count_nonzero(new[name][:,prefix_dim+added]).item()==0
            for suffix in ('.5.weight','.5.bias'):
                name=prefix+suffix
                checks[name+'_old_output_rows_exact']=torch.equal(new[name][slots],old[name])
                checks[name+'_new_output_rows_zero']=torch.count_nonzero(new[name][added]).item()==0
        checks['all_input_hashes_unchanged']=all(runner.sha(k)==v for k,v in contract['input_paths'].items())
        rows.append({'run':directory.name,'checks':{k:bool(v) for k,v in checks.items()},
            'passed_count':sum(checks.values()),'total_checks':len(checks),'passed':all(checks.values())})
    if not rows or not all(r['passed'] for r in rows):raise RuntimeError('Parameter setup audit failed')
    report={'status':'passed','outer_arrays_opened':False,'training_run_started_by_audit':False,
        'independent_direct_tensor_equality_checks':True,'runs':rows}
    runner.dump(exp.HERE/'parameter_setup_independent_audit.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
