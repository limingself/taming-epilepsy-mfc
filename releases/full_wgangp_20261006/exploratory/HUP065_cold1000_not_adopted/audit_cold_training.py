"""Recompute development selection from saved CSVs without opening outer data."""
from __future__ import annotations
import argparse
import json

import numpy as np
import pandas as pd

import run_cold_top32 as experiment
from finalize_cold_top32 import verify_frozen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True)
    args=parser.parse_args()
    run=experiment.HERE/'runs'/args.tag
    destination=run/'development_audit.json'
    if destination.exists():raise RuntimeError('Refuse replacing a completed audit receipt')
    summary,receipt,protocol,checkpoint=verify_frozen(run)
    history=pd.read_csv(run/'training_history.csv')
    trained=history.iloc[1:]
    baseline,_=experiment.adopted_baseline()
    checks=[]
    def check(name,passed,detail=None):
        checks.append(dict(check=name,passed=bool(passed),detail=detail))
        if not passed:raise RuntimeError(f'Audit failed: {name}: {detail}')
    check('neutral_teacher_free_complete1000',summary['actor_and_critic_checkpoint_load_calls']==[] and
        summary['actor_teacher_or_critic_warmstart'] is False and summary['trained_updates']==1000)
    check('actor_update_not_epoch_labels',np.array_equal(trained['update'].to_numpy(),np.arange(1,1001)))
    check('all_fitting_slots_match_context0_4_only',np.array_equal(trained.fit_context_slot.to_numpy(),np.arange(1000)%10))
    check('critic_updates24_warmup_plus3peractor',np.array_equal(trained.cumulative_critic_updates.to_numpy(),24+3*np.arange(1,1001)) and summary['critic_updates']==3024)
    warm=pd.read_csv(run/'critic_warmup.csv')
    check('24_actual_GP_warmup_rows',len(warm)==24 and np.isfinite(warm.select_dtypes(include='number').to_numpy()).all())
    check('all_training_terms_finite',np.isfinite(trained.filter(regex='^(train_|critic_|actor_lr)').to_numpy()).all())
    loss_error=float(np.abs(trained.train_law-(trained.train_law_loss+4*trained.train_law_occupancy+3*trained.train_law_worst_quantile)).max())
    check('only_existing_occupation_and_worst_coefficients_reweighted',loss_error<1e-9,loss_error)
    total_error=float(np.abs(trained.train_total-(trained.train_law+.5*trained.train_adversarial)).max())
    check('WGAN_in_every_actor_update',total_error<1e-9,total_error)
    preset=protocol['preset']
    check('predeclared_parameters_preserved',preset==experiment.PRESET)
    lr=1e-5+(3e-4-1e-5)*(1+np.cos(np.pi*np.arange(1000)/1000))/2
    lr_error=float(np.abs(trained.actor_lr-lr).max())
    check('cosine_learning_rate_predeclared',lr_error<1e-15,lr_error)
    expected=[0,1]+list(range(25,1001,25))
    rows=[]
    endpoint_errors=[]
    for update in expected:
        folder=run/'validation'/f'u{update:04d}'
        metrics=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        trajectory=pd.read_csv(folder/'trajectory_metrics.csv')
        channel=pd.read_csv(folder/'channel_metrics.csv')
        check(f'u{update:04d}_complete64x2_dev_channel_rows',len(channel)==128 and len(trajectory)==2 and
            set(trajectory.context_index)=={5} and set(trajectory.run_index)=={0,1})
        score=[]
        sd_mean=[]; sd_p90=[]
        for index in (0,1):
            c=channel[channel.run_index==index].sort_values('channel_index')
            t=trajectory[trajectory.run_index==index].iloc[0]
            check(f'u{update:04d}_run{index}_weighted_top32',np.array_equal(c.channel_index.to_numpy(),np.arange(64)) and
                np.array_equal(np.flatnonzero(c.direct_actuated.to_numpy(dtype=bool)),np.asarray(protocol['selected_indices'])))
            both=.5*c.time_w1_controlled/np.maximum(c.time_w1_free,.01)+.5*c.occupation_w1_controlled/np.maximum(c.occupation_w1_free,.01)
            common=.4*c.time_w1_controlled.mean()/c.time_w1_free.mean()+.4*c.occupation_w1_controlled.mean()/c.occupation_w1_free.mean()+.2*np.quantile(both,.9)
            score.append(common)
            sd_mean.append(float(np.log(c.symmetric_sd_ratio).mean()))
            sd_p90.append(float(np.quantile(c.symmetric_sd_ratio,.9)))
            endpoint_errors.extend(abs(c[col].mean()-float(t['mean_'+col])) for col in
                ('time_w1_free','time_w1_controlled','occupation_w1_free','occupation_w1_controlled'))
        score_error=abs(float(np.mean(score))-metrics['common_selection_score'])
        check(f'u{update:04d}_common_score_recalculated_from_channels',score_error<1e-12,score_error)
        sd_error=max(abs(float(np.mean(sd_mean))-metrics['mean_abs_log_sd_ratio']),
            abs(float(np.mean(sd_p90))-metrics['p90_sd_ratio']))
        check(f'u{update:04d}_SD_tiebreaker_recalculated_from_channels',sd_error<1e-12,sd_error)
        budget=bool(trajectory.finite.all() and trajectory.maximum_per_actuator_rms.max()<=.405+1e-9 and
            trajectory.total_energy.max()<=3.7908+1e-9 and trajectory.control_peak.max()<=1.8+1e-9 and
            trajectory.saturation_fraction.max()<.01)
        endpoint_ok=bool(trajectory.mean_time_w1_controlled.mean()<=baseline['mean_time_w1_controlled']+1e-12 and
            trajectory.mean_occupation_w1_controlled.mean()<=baseline['mean_occupation_w1_controlled']+1e-12)
        eligible=bool(budget and endpoint_ok and np.mean(score)<baseline['common_selection_score']-1e-8)
        check(f'u{update:04d}_budget_and_eligibility_recomputed',budget==metrics['budget_pass'] and eligible==metrics['terminal_candidate_eligible'])
        checkpoint_path=run/'checkpoints'/f'actor_critic_u{update:04d}.pt'
        check(f'u{update:04d}_all_ineligible_checkpoints_retained',checkpoint_path.is_file())
        rows.append(dict(update=update,budget=budget,eligible=eligible,score=metrics['common_selection_score'],
            sd_error=metrics['mean_abs_log_sd_ratio'],time_w1=metrics['mean_time_w1_controlled'],
            occupation_w1=metrics['mean_occupation_w1_controlled'],checkpoint_sha256=experiment.sha(checkpoint_path)))
    check('all_W1_means_bound_to_channel_CSV',max(endpoint_errors)<1e-12,max(endpoint_errors))
    budget_rows=[row for row in rows if row['budget']]
    eligible_rows=[row for row in rows if row['eligible']]
    best_budget=min(budget_rows,key=lambda row:(row['score'],row['sd_error']))
    selected=min(eligible_rows,key=lambda row:(row['score'],row['sd_error'])) if eligible_rows else best_budget
    check('budget_best_selection_recomputed',best_budget['update']==receipt['best_budget_update'])
    check('final_frozen_selected_development_checkpoint_recomputed',selected['update']==receipt['selected_update'] and
        selected['checkpoint_sha256']==experiment.sha(checkpoint))
    check('terminal_eligibility_recomputed_without_outer',bool(eligible_rows)==receipt['terminal_eligible'])
    check('fit_validation_reference_shapes',protocol['fit_reference_shape']==[30,256,64] and
        protocol['validation_reference_shape']==[30,256,64])
    check('outer_unopened_during_training',summary['outer_opened'] is False and summary['ctx6_opened'] is False)
    output=dict(status='passed',checks_count=len(checks),checks=checks,all42_candidates=rows,
        budget_feasible_candidates=len(budget_rows),dual_endpoint_eligible_candidates=len(eligible_rows),
        selected=selected,baseline=baseline,best_budget=best_budget,
        outer_data_opened_by_audit=False,training_history_sha256=experiment.sha(run/'training_history.csv'),
        source_sha256=experiment.sha(__file__),selected_checkpoint_sha256=experiment.sha(checkpoint),
        classification='independent CSV recalculation; no training or terminal selection changes')
    destination.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({key:output[key] for key in ('status','checks_count','budget_feasible_candidates',
        'dual_endpoint_eligible_candidates','selected')}),flush=True)


if __name__=='__main__':main()
