"""One predeclared neutral32-node HUP065 WGAN-GP1000-update parameter screen.

Frozen predictor/noise/scaler/reference and canonical error functions are retained.
No teacher/controller/critic checkpoint is loaded during preflight or training.
All development checkpoints, including budget/endpoint failures, are preserved.
No loss figure or manuscript mutation is performed by this experiment.
"""
from __future__ import annotations
import argparse
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd
import torch


def long_path(path):
    value=str(Path(path).absolute())
    if os.name=='nt' and not value.startswith('\\\\?\\'):value='\\\\?\\'+value
    return Path(value)


HERE=long_path(Path(__file__).resolve().parent)
BUNDLE=HERE/'frozen_inputs'/'external_bundle'
BASELINE=HERE/'frozen_inputs'/'adopted32_development_baseline'
MODEL_SHA='278591950c0938458fca1fc9dfdfb943eb4d413d1ad95a0f13fa59630b39fa89'
CANONICAL_SHA='57da573dbe300ad6bd585fbd69e46102d9f315dba8862e63c7d3e8320b6943e1'
PRESET=dict(actor_updates=1000,actor_lr=3e-4,actor_min_lr=1e-5,critic_lr=1e-4,
    n_critic=3,critic_warmup=24,adv_weight=.5,gp_weight=10.,critic_drift=.001,
    particles=32,horizon=256,tau=.2,noise_scale=.79451175,input_step_scale=.5,
    amplitude=1.8,actuator_rms_cap=.405,total_energy_cap=3.7908,
    actor_gradient_clip=1.,critic_gradient_clip=5.,actor_weight_decay=1e-5,
    actor_betas=[.9,.999],critic_betas=[0.,.9],optimizer_epsilon=1e-8,
    covariance_coefficients=dict(mean=10.,log_variance=10.,worst_log_variance=5.,quantile=8.,
        worst_quantile=8.,no_harm_mean=20.,no_harm_quantile=20.,occupancy=6.,tube=2.,joint=.5,
        energy=.01,smoothness=.2,curvature=.1,saturation=1.,gain_regularization=.001),
    validation_every=25,validation_noise_seed=20260922)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def forbid_controller_loads():
    original=torch.load
    calls=[]
    def forbidden(*args,**kwargs):
        calls.append(str(args[0]) if args else 'unknown')
        raise RuntimeError('Neutral experiment forbids all torch/controller/teacher checkpoint loads during setup/train')
    torch.load=forbidden
    try:yield calls
    finally:torch.load=original


def load_runner(out):
    source=BUNDLE/'reproduce_external.py'
    spec=importlib.util.spec_from_file_location('cold32_frozen_portable_routes',source)
    portable=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=portable
    spec.loader.exec_module(portable)
    manifest=portable.read_manifest()
    count=portable.verify_hashes(manifest)
    runner=portable.load_runner('HUP065',manifest,HERE/'.runtime',out.parent)
    if runner.sha(runner.SOURCE)!=CANONICAL_SHA:raise RuntimeError('Canonical function source changed')
    return runner,count


def setup_neutral(runner,args):
    # Existing setup performs official frozen data/preprocessing/Part-II checks.
    # Its temporary23 actor is neutral and never retained/loaded from a file.
    s=runner.setup('HUP065',args.seed)
    if runner.sha(s['p']['model'])!=MODEL_SHA:raise RuntimeError('Frozen predictive plant hash mismatch')
    with np.load(s['p']['network'],allow_pickle=False) as archive:
        score=np.asarray(archive['centrality_score'])
        channels=np.asarray(archive['channels'])
        official_mask=np.asarray(archive['target_mask'],dtype=bool)
    ranking=np.lexsort((np.arange(64),-score))
    if not np.array_equal(np.flatnonzero(official_mask),np.sort(ranking[:23])):
        raise RuntimeError('Official source mask does not match frozen weighted top23')
    selected=np.sort(ranking[:32])
    core=s['core']
    world=core.TorchGraphRCSDE(s['model'],selected,256.,control_graph_diffusion_time=.2,
        preserve_physical_control_residual=True,dtype=torch.float64,device='cpu')
    if not np.array_equal(world.adjacency.numpy(),np.asarray(s['model'].adjacency)):
        raise RuntimeError('Input heat time substituted the frozen dynamics graph')
    adapter=core.FrozenGraphRCMarkovAdapter(world,control_step_scale=.5,dtype=torch.float64)
    stepper=core.FrozenIctalGraphRCBatchStepper(world,adapter,diffusion_scale=.79451175)
    mean,variance,scale,weights=runner.reference_statistics(s['fit'])
    center,state_scale=runner.markov_normalization(adapter,s['initials'],s['fit'])
    # Reset RNG before the actual32 actor/critic. No23/32 parameter transfer.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    actor=core.StructuredSplineCovarianceActor(stepper,mean,variance,scale,
        torch.zeros(32,64,dtype=torch.float64),selected,horizon=256,basis_count=8,
        hidden_size=64,amplitude_limit=1.8,actuator_alpha=1.,residual_scale=.15,
        local_gain_initial_fraction=0.,local_gain_maximum_fraction=.05,
        markov_feature_center=center,markov_feature_scale=state_scale,
        markov_residual_scale=.6,markov_hidden_size=96,maximum_slew=None)
    critic=core.TimeConditionedWassersteinCritic(s['fit'].reshape(-1,64).mean(dim=0),scale,
        horizon_samples=256,hidden_size=128)
    mask=np.isin(np.arange(64),selected)
    s.update(p=dict(s['p'],m=32,tau=.2),world=world,adapter=adapter,stepper=stepper,
        actor=actor,critic=critic,selected=selected,mask=mask,scale=scale,weights=weights)
    s['ranking']=pd.DataFrame(dict(rank=np.arange(1,65),channel_index=ranking,
        channel=channels[ranking],weighted_centrality_score=score[ranking],selected=np.arange(64)<32))
    return s


def law(runner,s,rollout,reference,baseline,regularize):
    original,terms=runner.law(s,rollout,reference,baseline,regularize)
    # Reweight only existing returned terms; canonical functions remain bytes-identical.
    value=original+4.*terms['occupancy']+3.*terms['worst_quantile']
    return value,terms


def adopted_baseline():
    report=json.loads((BASELINE/'selection_report.json').read_text(encoding='utf-8'))
    metrics=dict(report['adaptation_final'])
    if report['adaptation_final_qualifies'] is not True or report['selected_checkpoint_sha256']!='2796775c4feb4534e9a4a294b2ab95687d77559b60bfab24cb383a42f1f25fdf':
        raise RuntimeError('Adopted32 baseline identity differs from authorized comparison')
    channels=pd.read_csv(BASELINE/'selected_validation_channel_metrics.csv')
    metrics['mean_abs_log_sd_ratio']=float(np.log(channels.symmetric_sd_ratio).mean())
    metrics['p90_sd_ratio']=float(np.mean([np.quantile(group.symmetric_sd_ratio,.9) for _,group in channels.groupby('run_index')]))
    return metrics,report


def validate(runner,s,cache,baseline):
    s['actor'].eval()
    trajectory_rows=[]
    channel_frames=[]
    with torch.no_grad():
        for run,(initial,noise,free_tensor) in enumerate(cache):
            r=s['core'].empirical_fp_rollout(s['stepper'],s['actor'],initial,noise)
            controlled,free,reference=r.scaled[:,1:].numpy(),free_tensor.numpy(),s['validation'].numpy()
            tf,of=runner.distances(free,reference)
            tc,oc=runner.distances(controlled,reference)
            controls=r.controls.numpy()
            rms=np.sqrt(np.mean(controls**2,axis=(0,1)))
            energy=float(np.mean(np.sum(controls**2,axis=-1)))
            peak=float(np.abs(controls).max())
            saturation=float(np.mean(np.abs(controls)>=.95*1.8))
            finite=bool(np.isfinite(controlled).all() and np.isfinite(controls).all())
            budget=bool(finite and rms.max()<=.405+1e-9 and energy<=3.7908+1e-9 and peak<=1.8+1e-9 and saturation<.01)
            ref_sd=reference.std(axis=(0,1))
            control_sd=controlled.std(axis=(0,1))
            rho=np.maximum(control_sd/np.maximum(ref_sd,1e-12),ref_sd/np.maximum(control_sd,1e-12))
            per=.5*tc/np.maximum(tf,.01)+.5*oc/np.maximum(of,.01)
            p90=float(np.quantile(per,.9))
            score=.4*tc.mean()/tf.mean()+.4*oc.mean()/of.mean()+.2*p90
            trajectory_rows.append(dict(run_index=run,context_index=5,
                mean_time_w1_free=tf.mean(),mean_time_w1_controlled=tc.mean(),
                mean_occupation_w1_free=of.mean(),mean_occupation_w1_controlled=oc.mean(),
                common_selection_score=score,p90_channel_distance_ratio=p90,
                mean_abs_log_sd_ratio=float(np.log(rho).mean()),p90_sd_ratio=float(np.quantile(rho,.9)),
                sd_ratio_le2_count=int((rho<=2).sum()),both_improved_count=int(((tc<tf)&(oc<of)).sum()),
                mean_control_rms=np.sqrt(np.mean(controls**2)),maximum_per_actuator_rms=float(rms.max()),
                total_energy=energy,control_peak=peak,saturation_fraction=saturation,budget_pass=budget,finite=finite))
            channel_frames.append(pd.DataFrame(dict(run_index=run,channel_index=np.arange(64),
                channel=s['arrays']['channels'],direct_actuated=s['mask'],time_w1_free=tf,time_w1_controlled=tc,
                occupation_w1_free=of,occupation_w1_controlled=oc,symmetric_sd_ratio=rho,
                mean_abs_error=np.abs(controlled.mean(axis=(0,1))-reference.mean(axis=(0,1))),
                both_improved=(tc<tf)&(oc<of))))
    trajectory=pd.DataFrame(trajectory_rows)
    result={name:float(trajectory[name].mean()) for name in trajectory.columns if name not in ('run_index','context_index','budget_pass','finite')}
    result['maximum_per_actuator_rms']=float(trajectory.maximum_per_actuator_rms.max())
    result['maximum_total_energy']=float(trajectory.total_energy.max())
    result['budget_pass']=bool(trajectory.budget_pass.all())
    result['finite']=bool(trajectory.finite.all())
    result['both_mean_endpoints_not_worse_than_adopted32']=bool(
        result['mean_time_w1_controlled']<=baseline['mean_time_w1_controlled']+1e-12 and
        result['mean_occupation_w1_controlled']<=baseline['mean_occupation_w1_controlled']+1e-12)
    result['terminal_candidate_eligible']=bool(result['budget_pass'] and result['both_mean_endpoints_not_worse_than_adopted32'] and
        result['common_selection_score']<baseline['common_selection_score']-1e-8)
    return result,trajectory,pd.concat(channel_frames,ignore_index=True)


def protocol(runner,s,args,count,baseline,baseline_report):
    imports={}
    for name,module in sys.modules.items():
        if name=='mfc_pipeline' or name.startswith('mfc_pipeline.'):
            source=Path(module.__file__).resolve()
            if BUNDLE.resolve() not in source.parents:raise RuntimeError('Canonical source import escaped isolated bundle')
            imports[name]=str(source.relative_to(HERE))
    return dict(status='predeclared_single_seed_neutral1000_update_parameter_screen',subject='HUP065',seed=args.seed,
        configured_actuators=32,total_channels=64,actuation_fraction=.5,
        selected_indices=s['selected'],preset=PRESET,canonical_function_sha256=CANONICAL_SHA,
        runner_source_sha256=sha(__file__),base_fresh_runner_sha256=sha(runner.__file__),
        predictive_model_sha256=MODEL_SHA,frozen_input_identities=s['hashes'],frozen_bundle_files_verified=count,
        canonical_dependency_imports=imports,baseline_source_hashes={name:sha(BASELINE/name) for name in
            ('selection_report.json','selected_validation_channel_metrics.csv','selected_validation_trajectory_metrics.csv')},
        adopted32_development_endpoints=baseline,adopted32_actor_identity_only=baseline_report['selected_checkpoint_sha256'],
        adopted32_actor_weights_opened=False,actor_critic_teacher_checkpoint_loaded=False,
        neutral_actor_initializer=True,actor_hidden_weights_fresh_random_neutral_final_layers=True,
        model_noise_scaler_reference_and_dynamics_graph_frozen=True,tau_only_control_input_spreading=True,
        framework_error_function_forms_unchanged=True,only_existing_term_coefficient_overrides={'occupancy':6.,'worst_quantile':8.},
        fitting_ledger=s['fit_ledger'],fit_reference_shape=list(s['fit'].shape),validation_reference_shape=list(s['validation'].shape),
        fit_context_indices=list(range(5)),selection_context_index=5,terminal_veto_context_index=6,
        common_selection_score='.4 mean(timeW1)/free+.4 mean(occupationW1)/free+.2 p90(.5 per-channel time/free+.5 occupation/free); per-channel free denominator floor.01',
        SD_selection_rule='mean absolute logSD ratio is only a secondary tie-breaker; full SD summaries retained; no new loss/error functional',
        checkpoint_selection='budget first, fixed common score then SD tie-breaker; separate terminal-eligible track additionally requires both mean endpoints not worse than adopted32 and score improvement',
        validation_checkpoint_retention='all initial/step1/every25 actor+critic snapshots including ineligible candidates',
        ctx6_failure_stops_outer=True,ctx6_or_outer_reselection=False,outer_arrays_opened_during_training=False,
        classification='post-hoc exploratory cold-initialization amendment; historical outer previously revealed; not new locked validation',
        no_projection_rescaling_or_new_loss_plot=True,no_global_optimality_or_all_channels_restoration_claim=True,
        original_code_history_and_overleaf_modified=False,not_automatically_adopted=True)


def preflight(runner,args,out,count):
    if out.exists():raise RuntimeError('Use a new run tag; refusing preflight overwrite')
    out.mkdir(parents=True)
    with forbid_controller_loads() as forbidden:
        s=setup_neutral(runner,args)
        baseline,baseline_report=adopted_baseline()
        audit=protocol(runner,s,args,count,baseline,baseline_report)
        runner.dump(out/'protocol.json',audit)
        s['ranking'].to_csv(out/'frozen_weighted_top32_ranking.csv',index=False)
        error,peak=0.,0.
        noise=s['core'].antithetic_noise(args.seed,s['stepper'].q)
        with torch.no_grad():
            for initial in s['initials']+s['selections']:
                free=s['core'].uncontrolled_particle_rollout(s['stepper'],initial,noise)
                neutral=s['core'].empirical_fp_rollout(s['stepper'],s['actor'],initial,noise)
                error=max(error,float((neutral.scaled-free).abs().max()))
                peak=max(peak,float(neutral.controls.abs().max()))
        # Input heat0 and heat.2 must have exactly the same zero-input predictor.
        core=s['core']
        world0=core.TorchGraphRCSDE(s['model'],s['selected'],256.,control_graph_diffusion_time=0.,
            preserve_physical_control_residual=True,dtype=torch.float64,device='cpu')
        adapter0=core.FrozenGraphRCMarkovAdapter(world0,control_step_scale=.5,dtype=torch.float64)
        stepper0=core.FrozenIctalGraphRCBatchStepper(world0,adapter0,diffusion_scale=.79451175)
        with torch.no_grad():
            noheat=core.uncontrolled_particle_rollout(stepper0,s['selections'][0],noise)
            heat=core.uncontrolled_particle_rollout(s['stepper'],s['selections'][0],noise)
            heat_error=float((noheat-heat).abs().max())
        zero_gains=all(torch.count_nonzero(parameter).item()==0 for parameter in
            (s['actor'].mean_gain_delta,s['actor'].deviation_gain_delta,s['actor'].local_deviation_gain_logits))
        zero_final=all(torch.count_nonzero(module[-1].weight).item()==0 and torch.count_nonzero(module[-1].bias).item()==0
            for module in (s['actor'].common_residual,s['actor'].deviation_residual,s['actor'].common_markov_residual,s['actor'].deviation_markov_residual))
        cache=runner.validation_cache(s)
        initial,trajectory,channels=validate(runner,s,cache,baseline)
        adopted_trajectory=pd.read_csv(BASELINE/'selected_validation_trajectory_metrics.csv')
        parity=max(abs(initial['mean_time_w1_free']-adopted_trajectory.mean_time_w1_free.mean()),
            abs(initial['mean_occupation_w1_free']-adopted_trajectory.mean_occupation_w1_free.mean()))
        if max(error,peak,heat_error,parity)>1e-11 or not zero_gains or not zero_final or forbidden:
            raise RuntimeError('Neutral/input-heat/frozen-baseline preflight failed')
        trajectory.to_csv(out/'neutral_validation_trajectory.csv',index=False)
        channels.to_csv(out/'neutral_validation_channels.csv',index=False)
        check=dict(status='passed',model_sha256=MODEL_SHA,canonical_source_sha256=CANONICAL_SHA,
            source_sha256=sha(__file__),direct_nodes=32,selected_indices=s['selected'],
            zero_gain_and_neutral_output_layer_checks=True,neutral_free_max_error=error,neutral_control_peak=peak,
            tau_only_input_map_zero_control_parity_error=heat_error,unchanged_adopted32_free_metrics_error=parity,
            no_teacher_actor_or_critic_weights_loaded=True,forbidden_torch_load_calls=forbidden,
            no_outer_arrays_opened=True,fitting_contexts=10,selection_contexts=2,
            actor_parameter_count=sum(p.numel() for p in s['actor'].parameters()),
            initial_validation=initial,all_frozen_inputs_unchanged=all(runner.sha(p)==d for p,d in s['hashes'].items()))
        runner.dump(out/'preflight.json',check)
        print(json.dumps(runner.ready(check)),flush=True)


def train(runner,args,out):
    check=json.loads((out/'preflight.json').read_text(encoding='utf-8'))
    if check['status']!='passed' or check['source_sha256']!=sha(__file__):raise RuntimeError('Exact-source successful preflight required')
    if (out/'training_history.csv').exists():raise RuntimeError('No partial resume or run overwrite')
    with forbid_controller_loads() as forbidden:
        s=setup_neutral(runner,args)
        baseline,_=adopted_baseline()
        core,actor,critic=s['core'],s['actor'],s['critic']
        cache=runner.validation_cache(s)
        initial,trajectory,channels=validate(runner,s,cache,baseline)
        contract=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
        if contract['runner_source_sha256']!=sha(__file__) or contract['preset']!=PRESET:
            raise RuntimeError('Declared preset/source changed after preflight')
        checkpoint_dir=out/'checkpoints'
        checkpoint_dir.mkdir()
        validation_dir=out/'validation'
        validation_dir.mkdir()
        def save_checkpoint(update,metrics):
            path=checkpoint_dir/f'actor_critic_u{update:04d}.pt'
            torch.save(dict(actor_state_dict=copy.deepcopy(actor.state_dict()),critic_state_dict=copy.deepcopy(critic.state_dict()),
                update=update,validation=metrics,training_contract=contract),path)
            return path
        def save_validation(update,metrics,trajectories,channel_values):
            folder=validation_dir/f'u{update:04d}'
            folder.mkdir()
            runner.dump(folder/'summary.json',metrics)
            trajectories.to_csv(folder/'trajectory_metrics.csv',index=False)
            channel_values.to_csv(folder/'channel_metrics.csv',index=False)
        save_validation(0,initial,trajectory,channels)
        first_checkpoint=save_checkpoint(0,initial)
        best_budget_tuple=(initial['common_selection_score'],initial['mean_abs_log_sd_ratio'])
        best_budget_update=0
        best_budget_path=first_checkpoint
        best_eligible_tuple=(float('inf'),float('inf'))
        best_eligible_update=None
        best_eligible_path=None
        history=[dict(update=0,**initial)]
        actor_opt=torch.optim.AdamW(actor.parameters(),lr=3e-4,weight_decay=1e-5)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(actor_opt,T_max=1000,eta_min=1e-5)
        critic_opt=torch.optim.Adam(critic.parameters(),lr=1e-4,betas=(0.,.9))
        indices=tuple(range(15,256,16))
        started=time.perf_counter()
        with torch.no_grad():
            warm=[core.empirical_fp_rollout(s['stepper'],actor,s['initials'][i],core.antithetic_noise(args.seed+101*(i+1),s['stepper'].q)).scaled[:,1:].detach() for i in range(3)]
        warm_rows=[]
        for i in range(24):
            critic_opt.zero_grad(set_to_none=True)
            audit=core.balanced_wgan_gp_loss(critic,warm[i%3],s['fit'],indices,
                gradient_penalty_weight=10.,critic_drift_weight=.001,
                generator=torch.Generator().manual_seed(args.seed+50000+i))
            if not torch.isfinite(audit.loss):raise RuntimeError('Nonfinite critic warmup')
            audit.loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(critic.parameters(),5.)
            critic_opt.step()
            warm_rows.append(dict(critic_warmup_update=i+1,critic_loss=float(audit.loss.detach()),
                gradient_penalty=float(audit.gradient_penalty.detach()),critic_input_gradient_norm=float(audit.mean_gradient_norm.detach()),
                critic_gradient_norm_before_clip=float(norm)))
        pd.DataFrame(warm_rows).to_csv(out/'critic_warmup.csv',index=False)
        for update in range(1,1001):
            slot=(update-1)%len(s['initials'])
            initial_state=s['initials'][slot]
            noise=core.antithetic_noise(args.seed+1009*update,s['stepper'].q)
            actor.eval()
            with torch.no_grad():
                detached=core.empirical_fp_rollout(s['stepper'],actor,initial_state,noise).scaled[:,1:].detach()
                free=core.uncontrolled_particle_rollout(s['stepper'],initial_state,noise)[:,1:]
            core.set_requires_grad(actor,False)
            core.set_requires_grad(critic,True)
            critic.train()
            for j in range(3):
                critic_opt.zero_grad(set_to_none=True)
                audit=core.balanced_wgan_gp_loss(critic,detached,s['fit'],indices,
                    gradient_penalty_weight=10.,critic_drift_weight=.001,
                    generator=torch.Generator().manual_seed(args.seed+100000*update+j))
                if not torch.isfinite(audit.loss):raise RuntimeError('Nonfinite critic')
                audit.loss.backward()
                dnorm=torch.nn.utils.clip_grad_norm_(critic.parameters(),5.)
                critic_opt.step()
            core.set_requires_grad(critic,False)
            core.set_requires_grad(actor,True)
            actor.train()
            actor_opt.zero_grad(set_to_none=True)
            rollout=core.empirical_fp_rollout(s['stepper'],actor,initial_state,noise)
            law_loss,terms=law(runner,s,rollout,s['fit'],free,True)
            adv,_=core.actor_wasserstein_loss(critic,rollout.scaled[:,1:],s['fit'],indices)
            total=law_loss+.5*adv
            if not torch.isfinite(total):raise RuntimeError('Nonfinite actor')
            total.backward()
            anorm=torch.nn.utils.clip_grad_norm_(actor.parameters(),1.)
            actor_opt.step()
            used_lr=actor_opt.param_groups[0]['lr']
            scheduler.step()
            row=dict(update=update,fit_context_slot=slot,actor_lr=used_lr,
                train_total=float(total.detach()),train_law=float(law_loss.detach()),train_adversarial=float(adv.detach()),
                critic_loss=float(audit.loss.detach()),critic_estimate=float(audit.estimate.detach()),
                critic_gp=float(audit.gradient_penalty.detach()),critic_input_gradient_norm=float(audit.mean_gradient_norm.detach()),
                actor_gradient_norm_before_clip=float(anorm),critic_gradient_norm_before_clip=float(dnorm),
                cumulative_critic_updates=24+3*update)
            row.update({'train_law_'+name:float(value.detach()) for name,value in terms.items()})
            if update==1 or update%25==0:
                metrics,trajectories,channel_values=validate(runner,s,cache,baseline)
                row.update(metrics)
                save_validation(update,metrics,trajectories,channel_values)
                checkpoint=save_checkpoint(update,metrics)
                criterion=(metrics['common_selection_score'],metrics['mean_abs_log_sd_ratio'])
                if metrics['budget_pass'] and criterion<best_budget_tuple:
                    best_budget_tuple,best_budget_update,best_budget_path=criterion,update,checkpoint
                if metrics['terminal_candidate_eligible'] and criterion<best_eligible_tuple:
                    best_eligible_tuple,best_eligible_update,best_eligible_path=criterion,update,checkpoint
            history.append(row)
            if update==1 or update%10==0:
                pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
                elapsed=time.perf_counter()-started
                progress=dict(status='running_neutral_cold_top32',subject='HUP065',seed=args.seed,
                    update=update,planned_updates=1000,elapsed_seconds=elapsed,seconds_per_actor_update=elapsed/update,
                    remaining_seconds_linear_estimate=elapsed/update*(1000-update),
                    best_budget_update=best_budget_update,best_budget_common_score=best_budget_tuple[0],
                    best_terminal_eligible_update=best_eligible_update,
                    best_terminal_eligible_common_score=best_eligible_tuple[0] if best_eligible_path else None,
                    adopted32_common_score=baseline['common_selection_score'],
                    adopted32_time_w1=baseline['mean_time_w1_controlled'],adopted32_occupation_w1=baseline['mean_occupation_w1_controlled'],
                    cumulative_critic_updates=24+3*update,checkpoint_weights_loaded=False,outer_opened=False)
                runner.dump(out/'progress.json',progress)
                print(json.dumps(progress),flush=True)
        # No checkpoint is loaded inside guarded training. Frozen files copy the
        # already saved selected immutable state; evaluation is a separate mode.
        import shutil
        chosen_path=best_eligible_path or best_budget_path
        shutil.copy2(chosen_path,out/'frozen_actor_wgan.pt')
        runner.dump(out/'selection_receipt.json',dict(
            status='eligible_cold_candidate_frozen' if best_eligible_path else 'completed_without_dual_endpoint_improvement',
            selected_update=best_eligible_update if best_eligible_path else best_budget_update,
            selected_checkpoint=str(chosen_path),frozen_checkpoint_sha256=sha(out/'frozen_actor_wgan.pt'),
            best_budget_update=best_budget_update,best_budget_checkpoint=str(best_budget_path),
            terminal_eligible=bool(best_eligible_path),best_terminal_eligible_update=best_eligible_update,
            adopted32_development_baseline=baseline,ctx6_used_for_reselection=False,outer_used_for_selection=False,
            neutral_initialization=True,all_validation_candidates_retained=True,not_automatically_adopted=True))
        summary=dict(status='completed_neutral_cold1000_parameter_screen',subject='HUP065',trained_updates=1000,
            selected_update=best_eligible_update if best_eligible_path else best_budget_update,terminal_eligible=bool(best_eligible_path),
            best_budget_update=best_budget_update,best_budget_common_score=best_budget_tuple[0],
            best_terminal_eligible_common_score=best_eligible_tuple[0] if best_eligible_path else None,
            elapsed_seconds=time.perf_counter()-started,actor_and_critic_checkpoint_load_calls=forbidden,
            actor_teacher_or_critic_warmstart=False,critic_updates=3024,
            frozen_checkpoint_sha256=sha(out/'frozen_actor_wgan.pt'),outer_opened=False,ctx6_opened=False,
            all_inputs_unchanged=all(runner.sha(path)==digest for path,digest in s['hashes'].items()),
            no_new_loss_figure=True,not_automatically_adopted=True)
        if not summary['all_inputs_unchanged'] or forbidden:raise RuntimeError('Frozen input mutation/checkpoint load')
        runner.dump(out/'training_summary.json',summary)
        runner.dump(out/'progress.json',summary)
        print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('preflight','train'))
    parser.add_argument('--seed',type=int,default=20261011)
    parser.add_argument('--threads',type=int,choices=(2,),default=2)
    parser.add_argument('--tag',required=True)
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,79}',args.tag):raise RuntimeError('Use a unique simple tag')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    out=HERE/'runs'/args.tag
    runner,count=load_runner(out)
    if args.mode=='preflight':preflight(runner,args,out,count)
    else:train(runner,args,out)


if __name__=='__main__':main()
