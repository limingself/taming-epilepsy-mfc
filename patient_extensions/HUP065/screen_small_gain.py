"""Registered six small new-node gain pairs, dev-only then eligible adaptation.

No old23 parameter is changed by initialization. The same covariance error
definitions/coefficients, frozen plant/reference/noise, actuator bounds, and
common dev score are used. If a pair improves the score without worsening either
mean W1 endpoint, all existing actor parameters may adapt for 150 updates.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import optimize_hup065 as exp

PAIRS=((.05,.05),(.1,.1),(.2,.2),(0.,.05),(0.,.1),(.05,0.))


def control_groups(s,cache,old_channels):
    old_mask=np.isin(s['selected'],old_channels)
    rows=[]
    with torch.no_grad():
        for run,(initial,noise,_) in enumerate(cache):
            rollout=s['core'].empirical_fp_rollout(s['stepper'],s['actor'],initial,noise)
            controls=rollout.controls.numpy()
            per=np.sqrt(np.mean(controls**2,axis=(0,1)))
            for slot,channel in enumerate(s['selected']):
                rows.append(dict(run_index=run,channel_index=int(channel),actuator_slot=slot,
                    group='old23' if old_mask[slot] else 'new9',rms=float(per[slot]),
                    peak=float(np.abs(controls[:,:,slot]).max())))
    frame=pd.DataFrame(rows)
    new=frame[frame.group.eq('new9')]
    old=frame[frame.group.eq('old23')]
    return dict(new9_mean_actuator_rms=float(new.rms.mean()),new9_max_actuator_rms=float(new.rms.max()),
        old23_mean_actuator_rms=float(old.rms.mean()),old23_max_actuator_rms=float(old.rms.max()),
        active_new9_count_both_runs=int((new.groupby('channel_index').rms.min()>1e-12).sum())),frame


def label(mean,deviation):
    return f"mean{mean:g}_dev{deviation:g}".replace('.','p')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    runner=exp.load_runner()
    out=HERE/'results'
    if out.exists():raise RuntimeError('Refusing existing six-pair screen overwrite')
    out.mkdir(parents=True)
    source_snapshot=out/'source_snapshot'
    source_snapshot.mkdir()
    shutil.copy2(exp.__file__,source_snapshot/'optimize_hup065.py')
    shutil.copy2(__file__,source_snapshot/'screen_small_gain.py')
    setup_args=SimpleNamespace(seed=20261011,nodes=32,tau=.2,init_checkpoint=None)
    s=exp.setup(runner,setup_args)
    baseline_state=copy.deepcopy(s['actor'].state_dict())
    old_channels=s['migration']['source_actuators']
    old_slots=torch.as_tensor([j for j,c in enumerate(s['selected']) if int(c) in old_channels],dtype=torch.long)
    added=[(j,int(c)) for j,c in enumerate(s['selected']) if int(c) not in old_channels]
    s['ranking'].to_csv(out/'frozen_part1_top32_ranking.csv',index=False)
    registration=dict(pairs=[list(pair) for pair in PAIRS],configured_nodes=32,new_nodes=9,tau=.2,
        model_sha256=runner.MODEL_HASHES['HUP065'],input_hashes=s['hashes'],
        source_checkpoint_sha256=s['migration']['source_checkpoint_sha256'],
        explicit_warm_start=True,old23_parameters_fixed_for_initialization=True,
        canonical_covariance_coefficients=dict(mean=10.,log_variance=10.,worst_log_variance=5.,quantile=8.,
            worst_quantile=5.,no_harm_mean=20.,no_harm_quantile=20.,occupancy=2.,tube=2.,joint=.5,
            energy=.01,smoothness=.2,curvature=.1,saturation=1.,gain_regularization=.001),
        fit_contexts=s['fit_ledger'],selection_context_index=5,particles=32,horizon=256,
        diffusion_scale=.79451175,control_step_scale=.5,input_energy_cap=3.7908,per_actuator_rms_cap=.405,
        amplitude_limit=1.8,existing_actor_formula_unchanged=True,
        candidate_eligibility='unchanged budgets; common score lower than zero baseline; both mean W1 endpoints not worsened',
        strict_score_improvement_epsilon=1e-8,endpoint_numerical_tolerance=1e-12,
        eligible_best_adaptation_updates=150,all_actor_and_critic_parameters_may_update_during_adaptation=True,
        no_outer_or_ctx6_used_for_selection=True,source_wrapper_sha256=runner.sha(__file__),
        source_optimizer_sha256=runner.sha(exp.__file__),not_adopted_in_paper=True)
    runner.dump(out/'registered_six_pair_protocol.json',registration)
    cache=runner.validation_cache(s)
    baseline,bt,bc=exp.validate(runner,s,cache)
    base_groups,bg=control_groups(s,cache,old_channels)
    bt.to_csv(out/'zero_baseline_dev_trajectory.csv',index=False)
    bc.to_csv(out/'zero_baseline_dev_channel.csv',index=False)
    bg.to_csv(out/'zero_baseline_actuator_groups.csv',index=False)
    runner.dump(out/'zero_baseline_metrics.json',dict(**baseline,**base_groups))
    rows=[]
    for mean,deviation in PAIRS:
        s['actor'].load_state_dict(baseline_state,strict=True)
        with torch.no_grad():
            for slot,channel in added:
                s['actor'].mean_gain_delta[:,slot,channel]=mean
                s['actor'].deviation_gain_delta[:,slot,channel]=deviation
        state=s['actor'].state_dict()
        # Direct tensor equality establishes old23/hidden/MLP parameters intact.
        unchanged=[]
        for name,tensor in baseline_state.items():
            if name in ('mean_gain_delta','deviation_gain_delta'):
                unchanged.append(torch.equal(state[name][:,old_slots],tensor[:,old_slots]))
            else:
                unchanged.append(torch.equal(state[name],tensor))
        if not all(unchanged):raise RuntimeError('A non-registered old/shared parameter changed during initialization')
        metrics,trajectory,channels=exp.validate(runner,s,cache)
        groups,frame=control_groups(s,cache,old_channels)
        eligible=bool(metrics['budget_pass'] and
            metrics['common_selection_score']<baseline['common_selection_score']-1e-8 and
            metrics['mean_time_w1_controlled']<=baseline['mean_time_w1_controlled']+1e-12 and
            metrics['mean_occupation_w1_controlled']<=baseline['mean_occupation_w1_controlled']+1e-12)
        row=dict(new_mean_gain=mean,new_deviation_gain=deviation,eligible_for_adaptation=eligible,
            old23_and_shared_parameters_bitwise_unchanged=True,**metrics,**groups)
        rows.append(row)
        tag=label(mean,deviation)
        trajectory.to_csv(out/(tag+'_dev_trajectory.csv'),index=False)
        channels.to_csv(out/(tag+'_dev_channel.csv'),index=False)
        frame.to_csv(out/(tag+'_actuator_groups.csv'),index=False)
        torch.save(dict(actor_state_dict=copy.deepcopy(state),critic_state_dict=copy.deepcopy(s['critic'].state_dict()),
            best_update=0,training_contract=dict(subject='HUP065',nodes=32,graph_spread=.2,seed=20261011,
                model_sha256=runner.MODEL_HASHES['HUP065'],input_paths=s['hashes'],
                new_mean_gain=mean,new_deviation_gain=deviation,explicit_warm_start=True,
                original_covariance_objective=True,source_code_sha256=runner.sha(__file__)),
            selected_validation=metrics),out/(tag+'_initial_actor.pt'))
        print(json.dumps(row),flush=True)
    result=pd.DataFrame(rows)
    result.to_csv(out/'six_pair_common_dev_results.csv',index=False)
    allowed=result[result.eligible_for_adaptation].sort_values('common_selection_score')
    report=dict(status='six_pair_initialization_screen_completed',zero_baseline=baseline,candidates=rows,
        eligible_count=len(allowed),outer_opened=False,context6_opened=False,not_adopted_in_paper=True)
    if not len(allowed):
        report['selected']='retain_previous_old23_selected550_controller'
        report['training_started']=False
        runner.dump(out/'selection_report.json',report)
        print(json.dumps(dict(status='no_eligible_small_gain_pair',baseline_common_score=baseline['common_selection_score'])),flush=True)
        return
    best=allowed.iloc[0]
    mean,deviation=float(best.new_mean_gain),float(best.new_deviation_gain)
    tag=label(mean,deviation)
    initialization=out/(tag+'_initial_actor.pt')
    report.update(selected_pair=[mean,deviation],selected_initialization_sha256=runner.sha(initialization),
        selected_initialization=str(initialization),training_started=True)
    runner.dump(out/'selection_report.json',report)
    training_args=SimpleNamespace(seed=20261011,nodes=32,tau=.2,arm='covariance',updates=150,lr=1e-4,
        adv=.5,init_checkpoint=str(initialization))
    training_out=out/'selected_pair_adaptation_u150'
    exp.train(runner,training_args,training_out)
    summary=json.loads((training_out/'training_summary.json').read_text(encoding='utf-8'))
    final=summary['selected_validation']
    # Original baseline remains the common comparator; outer is not rescued.
    qualifies=bool(final['budget_pass'] and final['common_selection_score']<baseline['common_selection_score']-1e-8
        and final['mean_time_w1_controlled']<=baseline['mean_time_w1_controlled']+1e-12
        and final['mean_occupation_w1_controlled']<=baseline['mean_occupation_w1_controlled']+1e-12)
    report.update(adaptation_completed=True,adaptation_selected_update=summary['selected_update'],
        adaptation_final_qualifies=qualifies,adaptation_final=final,
        selected_checkpoint=str(training_out/'frozen_actor_wgan.pt'),selected_checkpoint_sha256=summary['checkpoint_sha256'])
    # Raw delivered RMS on old23/new9 groups and changes in old parameter rows.
    final_setup=SimpleNamespace(seed=20261011,nodes=32,tau=.2,init_checkpoint=report['selected_checkpoint'])
    final_s=exp.setup(runner,final_setup)
    final_groups,final_frame=control_groups(final_s,runner.validation_cache(final_s),old_channels)
    final_frame.to_csv(out/'adapted_selected_actuator_groups.csv',index=False)
    final_state=final_s['actor'].state_dict()
    report['final_actuator_groups']=final_groups
    report['old23_gain_row_max_change']={name:float((final_state[name][:,old_slots]-baseline_state[name][:,old_slots]).abs().max())
        for name in ('mean_gain_delta','deviation_gain_delta')}
    runner.dump(out/'selection_report.json',report)
    runner.dump(out/'frozen_dev_selection_receipt.json',dict(selected_checkpoint=report['selected_checkpoint'],
        selected_checkpoint_sha256=report['selected_checkpoint_sha256'],outer_used_for_selection=False,ctx6_used_for_selection=False,
        eligible_for_terminal_veto=qualifies,registered_source_sha256=registration['source_wrapper_sha256'],
        classification='post-hoc exploratory parameter amendment; previously revealed outer run',not_adopted_in_paper=True))
    print(json.dumps(dict(status='small_gain_adaptation_completed',qualifies=qualifies,**final_groups)),flush=True)


if __name__=='__main__':main()
