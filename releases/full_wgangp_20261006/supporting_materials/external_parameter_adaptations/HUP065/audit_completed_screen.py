"""Independent dev score arithmetic/checkpoint choice audit; no outer reads."""
import json

import numpy as np
import pandas as pd

import optimize_hup065 as exp


def main():
    runner=exp.load_runner()
    checks=[]
    for name in ('top32_covariance_u150','top32_occupation_worst_u150','top32_tau1_covariance_u150'):
        path=exp.HERE/'runs'/name
        summary=json.loads((path/'training_summary.json').read_text(encoding='utf-8'))
        history=pd.read_csv(path/'training_history.csv')
        channels=pd.read_csv(path/'selected_validation_channel_metrics.csv')
        trajectories=pd.read_csv(path/'selected_validation_trajectory_metrics.csv')
        contract=json.loads((path/'training_contract.json').read_text(encoding='utf-8'))
        name_checks={
            '151_rows_including_initial':len(history)==151,
            'updates_all0_through150':np.array_equal(history['update'].to_numpy(),np.arange(151)),
            'finite_training_totals':np.isfinite(history.loc[history['update']>0,'train_total']).all(),
            'outer_not_opened':summary['outer_opened'] is False,
            'same_frozen_predictive_model':contract['model_sha256']==runner.MODEL_HASHES['HUP065'],
            'same_frozen_input_hashes':all(runner.sha(k)==v for k,v in contract['input_paths'].items()),
            'fixed_nodes32':contract['nodes']==32,
            'existing_architecture_dimensions':contract['particles']==32 and contract['horizon']==256,
            'original_budgets_retained':contract['input_energy_cap']==3.7908 and contract['per_actuator_rms_cap']==.405 and contract['amplitude_limit']==1.8,
            'ctx5_only_selection':contract['selection_context_index']==5,
            'sixteen_covariance_error_terms_unchanged':contract['source_sha256']=='57da573dbe300ad6bd585fbd69e46102d9f315dba8862e63c7d3e8320b6943e1',
        }
        eligible=history.dropna(subset=['common_selection_score'])
        eligible=eligible[eligible['budget_pass'].astype(str).str.lower().eq('true')]
        chosen=eligible.sort_values(['common_selection_score','update']).iloc[0]
        name_checks['selected_is_global_eligible_dev_min']=int(chosen['update'])==summary['selected_update']
        for key in ('time','occupation'):
            raw=channels[f'{key}_w1_controlled'].mean()
            name_checks[f'{key}_w1_summary_matches128_raw_channel_rows']=abs(raw-summary['selected_validation'][f'mean_{key}_w1_controlled'])<1e-12
        scores=[]
        for run,frame in channels.groupby('run_index'):
            per=.5*frame.time_w1_controlled.to_numpy()/np.maximum(frame.time_w1_free.to_numpy(),.01)
            per+=.5*frame.occupation_w1_controlled.to_numpy()/np.maximum(frame.occupation_w1_free.to_numpy(),.01)
            score=.4*frame.time_w1_controlled.mean()/frame.time_w1_free.mean()
            score+=.4*frame.occupation_w1_controlled.mean()/frame.occupation_w1_free.mean()+.2*np.quantile(per,.9)
            scores.append(score)
        name_checks['common_score_independently_recomputed']=abs(np.mean(scores)-summary['selected_validation']['common_selection_score'])<1e-12
        name_checks['dev_budget_flag_matches_saved_measurements']=bool(trajectories.budget_pass.all())==summary['selected_validation']['budget_pass']
        name_checks['checkpoint_hash_verified']=runner.sha(path/'frozen_actor_wgan.pt')==summary['checkpoint_sha256']
        checks.append(dict(run=name,checks={k:bool(v) for k,v in name_checks.items()},
            passed=all(name_checks.values()),passed_count=int(sum(name_checks.values())),total_checks=len(name_checks)))
    report=dict(status='passed' if all(r['passed'] for r in checks) else 'failed',
        outer_arrays_opened=False,context6_arrays_opened=False,runs=checks)
    runner.dump(exp.HERE/'completed_screen_independent_audit.json',report)
    print(json.dumps(runner.ready(report)),flush=True)
    if report['status']!='passed':raise RuntimeError('Completed dev audit failed')


if __name__=='__main__':main()
