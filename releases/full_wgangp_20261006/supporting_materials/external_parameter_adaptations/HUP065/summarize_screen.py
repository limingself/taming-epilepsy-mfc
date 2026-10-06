"""Seal a common-scale, budget-first development decision; no outer reads."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import optimize_hup065 as exp


def main():
    runner=exp.load_runner()
    names=('top32_covariance_u150','top32_occupation_worst_u150','top32_tau1_covariance_u150')
    rows=[]
    completed=[]
    for name in names:
        path=exp.HERE/'runs'/name/'training_summary.json'
        if not path.exists():raise RuntimeError('Wait until all pre-registered 150-update arms finish')
        summary=json.loads(path.read_text(encoding='utf-8'))
        if summary['outer_opened'] is not False:raise RuntimeError('Outer-informed candidate cannot enter selection')
        if summary['trained_updates']!=150:raise RuntimeError('Unequal fixed update budget')
        completed.append(summary)
        checkpoint=path.parent/'frozen_actor_wgan.pt'
        if runner.sha(checkpoint)!=summary['checkpoint_sha256']:raise RuntimeError('Checkpoint receipt mismatch')
        metrics=summary['selected_validation']
        channels=pd.read_csv(path.parent/'selected_validation_channel_metrics.csv')
        vector=channels.groupby('channel_index').both_improved.all()
        rows.append(dict(arm=name,configured_actuators=32,selected_update=summary['selected_update'],
            effective_new_actuators_at_selected_checkpoint='0 exactly' if summary['selected_update']==0 and 'tau1' not in name else 'requires delivered-control measurement',
            both_improved_in_both_dev_runs=int(vector.sum()),
            **metrics,time_reduction_percent=100*(1-metrics['mean_time_w1_controlled']/metrics['mean_time_w1_free']),
            occupation_reduction_percent=100*(1-metrics['mean_occupation_w1_controlled']/metrics['mean_occupation_w1_free'])))
    baseline=dict(completed[0]['initial_validation'])
    rows.insert(0,dict(arm='old23_selected550_reference_baseline',configured_actuators=23,selected_update=550,
        effective_new_actuators_at_selected_checkpoint='not applicable',**baseline,
        time_reduction_percent=100*(1-baseline['mean_time_w1_controlled']/baseline['mean_time_w1_free']),
        occupation_reduction_percent=100*(1-baseline['mean_occupation_w1_controlled']/baseline['mean_occupation_w1_free'])))
    # RMS over actuator coordinates depends on resource count; restore old23
    # coordinate denominator only for the historical reference baseline.
    rows[0]['mean_control_rms']=rows[0]['mean_control_rms']*np.sqrt(32/23)
    grid=json.loads((exp.HERE/'new_node_gain_grid_top32'/'selection_summary.json').read_text(encoding='utf-8'))
    for row in grid['candidates']:
        if row['coefficient'] in (.35,.8):
            rows.append(dict(arm=f"top32_new_node_diagonal_gain_{row['coefficient']}",configured_actuators=32,
                selected_update=0,effective_new_actuators_at_selected_checkpoint='9 nonzero feedback rows; no gradient training',**row,
                time_reduction_percent=100*(1-row['mean_time_w1_controlled']/row['mean_time_w1_free']),
                occupation_reduction_percent=100*(1-row['mean_occupation_w1_controlled']/row['mean_occupation_w1_free'])))
    frame=pd.DataFrame(rows)
    frame.to_csv(exp.HERE/'development_screen_comparison.csv',index=False)
    eligible=frame[frame.budget_pass].sort_values('common_selection_score',kind='stable')
    winner=eligible.iloc[0]
    report=dict(status='sealed_development_parameter_screen',registered_gradient_arms=3,updates_per_arm=150,
        finite_eligible_gain_initializations=[.35,.8],additional_gain_1p2_excluded=True,
        selected_arm=winner['arm'],selected_common_score=float(winner['common_selection_score']),
        baseline_common_score=baseline['common_selection_score'],
        improvement_over_baseline_found=bool(winner['common_selection_score']<baseline['common_selection_score']-1e-12),
        unchanged_original_budgets=True,ranking_only_from_frozen_part1=True,
        core_architecture_and_existing_error_terms_unchanged=True,explicit_warm_start=True,
        context6_opened=False,outer_opened=False,ctx6_used_for_selection=False,outer_used_for_selection=False,
        loss_figures_generated=False,not_adopted_in_paper=True,
        rms_denominator_caution='Mean actuator RMS across32 coordinates includes9 neutral slots for selected0; not an energy reduction. Baseline RMS denominator restored to23.',
        candidate_results=rows)
    runner.dump(exp.HERE/'development_selection_report.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
