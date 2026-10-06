"""Parameter-only HUP065 development tuning; sealed/outer data never in train.

The frozen Graph-RC, references, normalization, noise, actor/critic architecture
and empirical-law terms are unchanged. Actuator resources follow frozen Part-I
weighted-centrality ranking. This is explicit checkpoint warm-start adaptation,
NOT a from-scratch controller comparison. No runtime clipping/rescaling is added.
"""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[2]
HERE = PROJECT / 'patient_results/HUP065_sparse_control_final_v1/current_full_wgangp'
BUNDLE = HERE / 'small_gain_decoupled_screen/portable_inputs/external_bundle'
OLD_RUN = HERE / 'small_gain_decoupled_screen/portable_inputs/old23_source_run'
ARMS = {
    'covariance': {'occupancy': 2., 'worst_quantile': 5., 'no_harm_mean': 20., 'no_harm_quantile': 20.},
    'occupation_worst': {'occupancy': 8., 'worst_quantile': 8., 'no_harm_mean': 30., 'no_harm_quantile': 30.},
}
BASE_COEFFICIENTS = {'occupancy': 2., 'worst_quantile': 5., 'no_harm_mean': 20., 'no_harm_quantile': 20.}


def load_runner():
    spec = importlib.util.spec_from_file_location('hup065_portable_route', BUNDLE / 'reproduce_external.py')
    portable = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = portable
    spec.loader.exec_module(portable)
    manifest = portable.read_manifest()
    portable.verify_hashes(manifest)
    runtime = HERE / '.runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    return portable.load_runner('HUP065', manifest, runtime, HERE)


def transplant(actor, old_state, old_selected, selected):
    """Exact functional expansion: old input slots/actuator rows are mapped.

    New input columns and actuator rows are zero. Shared MLP hidden parameters
    retain old values; new actuator-index/local-inverse-effect buffers remain
    those of the correctly expanded frozen plant.
    """
    lookup = {int(channel): i for i, channel in enumerate(selected)}
    slots = torch.as_tensor([lookup[int(c)] for c in old_selected], dtype=torch.long)
    old_m, new_m = len(old_selected), len(selected)
    state = actor.state_dict()
    audit = []
    semantic_buffers = {'actuated_channel_indices', 'local_inverse_effect'}
    for name, target in state.items():
        source = old_state[name]
        if name in semantic_buffers:
            audit.append({'name': name, 'operation': 'retain_correct_new_plant_buffer'})
        elif name in {'mean_gain_delta', 'deviation_gain_delta', 'base_gain'}:
            target.zero_()
            axis = 1 if source.ndim == 3 else 0
            target.index_copy_(axis, slots, source)
            audit.append({'name': name, 'operation': 'map_actuator_axis_zero_new'})
        elif name == 'local_deviation_gain_logits':
            target.zero_()
            target.index_copy_(1, slots, source)
            audit.append({'name': name, 'operation': 'map_actuator_axis_zero_new'})
        elif name.endswith('.0.weight') and source.shape[1] + new_m - old_m == target.shape[1]:
            prefix = source.shape[1] - old_m
            target.zero_()
            target[:, :prefix] = source[:, :prefix]
            target[:, prefix + slots] = source[:, prefix:]
            audit.append({'name': name, 'operation': 'map_previous_control_feature_slots_zero_new'})
        elif name.endswith('.5.weight') or name.endswith('.5.bias'):
            target.zero_()
            target.index_copy_(0, slots, source)
            audit.append({'name': name, 'operation': 'map_output_rows_zero_new'})
        elif source.shape == target.shape:
            target.copy_(source)
            audit.append({'name': name, 'operation': 'unchanged_shared_parameter_or_buffer'})
        else:
            raise RuntimeError(f'Unaudited state tensor: {name}: {source.shape} -> {target.shape}')
    actor.load_state_dict(state, strict=True)
    return audit, slots


def setup(runner, args):
    s = runner.setup('HUP065', args.seed)
    old_selected, old_stepper, old_actor = s['selected'].copy(), s['stepper'], s['actor']
    network = np.load(s['p']['network'], allow_pickle=False)
    score = np.asarray(network['centrality_score'], dtype=np.float64)
    # Stable original-index tie break; all current centrality values distinct.
    ranking = np.lexsort((np.arange(len(score)), -score))
    if not np.array_equal(np.sort(ranking[:23]), old_selected):
        raise RuntimeError('Official 23 mask is not the weighted Part-I top23')
    selected = np.sort(ranking[:args.nodes])
    if not set(old_selected).issubset(set(selected)):
        raise RuntimeError('Expanded ranked mask must contain all original actuators')
    core = s['core']
    p = dict(s['p'], m=args.nodes, tau=args.tau)
    world = core.TorchGraphRCSDE(s['model'], selected, 256., control_graph_diffusion_time=args.tau,
        preserve_physical_control_residual=True, dtype=torch.float64, device='cpu')
    adapter = core.FrozenGraphRCMarkovAdapter(world, control_step_scale=.5, dtype=torch.float64)
    stepper = core.FrozenIctalGraphRCBatchStepper(world, adapter, diffusion_scale=.79451175)
    mean, variance, scale, weights = runner.reference_statistics(s['fit'])
    actor = core.StructuredSplineCovarianceActor(stepper, mean, variance, scale,
        torch.zeros(args.nodes, 64, dtype=torch.float64), selected, horizon=256,
        basis_count=8, hidden_size=64, amplitude_limit=1.8, actuator_alpha=1., residual_scale=.15,
        local_gain_initial_fraction=0., local_gain_maximum_fraction=.05,
        markov_feature_center=old_actor.markov_feature_center,
        markov_feature_scale=old_actor.markov_feature_scale, markov_residual_scale=.6,
        markov_hidden_size=96, maximum_slew=None)
    checkpoint_path = Path(args.init_checkpoint) if args.init_checkpoint else OLD_RUN / 'frozen_actor_wgan.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint['training_contract']['model_sha256'] != runner.MODEL_HASHES['HUP065']:
        raise RuntimeError('Initialization checkpoint predictive-plant mismatch')
    incoming = checkpoint['actor_state_dict']
    previous_selected = incoming['actuated_channel_indices'].numpy()
    if len(previous_selected) == 23:
        old_actor.load_state_dict(incoming, strict=True)
        migration_audit, slots = transplant(actor, incoming, previous_selected, selected)
        # With unchanged tau, expansion must be a function-preserving warm start.
        with torch.no_grad():
            noise = core.antithetic_noise(20260922, old_stepper.q)
            a = core.empirical_fp_rollout(old_stepper, old_actor, s['initials'][0], noise)
            b = core.empirical_fp_rollout(stepper, actor, s['initials'][0], noise)
            parity_output = float((a.scaled - b.scaled).abs().max())
            parity_old_control = float((a.controls - b.controls[:, :, slots]).abs().max())
            new_slots = np.setdiff1d(np.arange(args.nodes), slots.numpy())
            neutral_new = float(b.controls[:, :, new_slots].abs().max()) if len(new_slots) else 0.
        if args.tau == .2 and max(parity_output, parity_old_control, neutral_new) > 1e-11:
            raise RuntimeError(f'Functional warm-start expansion failed: {parity_output}, {parity_old_control}, {neutral_new}')
    elif np.array_equal(previous_selected, selected):
        actor.load_state_dict(incoming, strict=True)
        migration_audit, parity_output, parity_old_control, neutral_new = [], None, None, None
    else:
        raise RuntimeError('Continuation selected-channel identity differs')
    s.update(p=p, selected=selected, mask=np.isin(np.arange(64), selected), world=world,
        adapter=adapter, stepper=stepper, actor=actor, scale=scale, weights=weights)
    s['critic'].load_state_dict(checkpoint['critic_state_dict'], strict=True)
    s['migration'] = dict(source_checkpoint=str(checkpoint_path), source_checkpoint_sha256=runner.sha(checkpoint_path),
        source_selected_update=checkpoint.get('best_update'), source_actuators=previous_selected.tolist(),
        target_actuators=selected.tolist(), explicit_warm_start=True, initialized_from_scratch=False,
        tensor_operations=migration_audit, initial_output_max_error=parity_output,
        initial_old_control_max_error=parity_old_control, new_actuator_initial_peak=neutral_new,
        source_critic_reused=True, optimizer_states_reused=False)
    s['ranking'] = pd.DataFrame({'rank': np.arange(1,65), 'channel_index': ranking,
        'channel': np.asarray(network['channels'])[ranking], 'weighted_centrality_score': score[ranking],
        'direct_actuated': np.arange(64) < args.nodes})
    return s


def law(runner, s, rollout, reference, baseline, regularize, arm):
    original, terms = runner.law(s, rollout, reference, baseline, regularize)
    loss = original
    for name, coefficient in ARMS[arm].items():
        loss = loss + (coefficient - BASE_COEFFICIENTS[name]) * terms[name]
    return loss, terms


def validate(runner, s, cache):
    s['actor'].eval()
    rows, channels = [], []
    with torch.no_grad():
        for index, (initial, noise, baseline) in enumerate(cache):
            r = s['core'].empirical_fp_rollout(s['stepper'], s['actor'], initial, noise)
            controlled, free, reference = r.scaled[:,1:].numpy(), baseline.numpy(), s['validation'].numpy()
            tf, of = runner.distances(free, reference)
            tc, oc = runner.distances(controlled, reference)
            controls = r.controls.numpy()
            rms = np.sqrt(np.mean(controls**2, axis=(0,1)))
            energy = float(np.mean(np.sum(controls**2, axis=-1)))
            peak = float(np.abs(controls).max())
            sat = float(np.mean(np.abs(controls) >= .95 * 1.8))
            finite = bool(np.isfinite(controlled).all() and np.isfinite(controls).all())
            budget = bool(finite and rms.max() <= .405+1e-9 and energy <= 3.7908+1e-9 and peak <= 1.8+1e-9 and sat < .01)
            rho = np.maximum(controlled.std(axis=(0,1))/np.maximum(reference.std(axis=(0,1)),1e-12),
                reference.std(axis=(0,1))/np.maximum(controlled.std(axis=(0,1)),1e-12))
            mean_error = np.abs(controlled.mean(axis=(0,1))-reference.mean(axis=(0,1)))
            per = .5 * tc / np.maximum(tf,.01) + .5 * oc / np.maximum(of,.01)
            score = .4*tc.mean()/tf.mean()+.4*oc.mean()/of.mean()+.2*np.quantile(per,.90)
            rows.append(dict(context_index=5, run_index=index, mean_time_w1_free=tf.mean(), mean_time_w1_controlled=tc.mean(),
                mean_occupation_w1_free=of.mean(), mean_occupation_w1_controlled=oc.mean(), mean_control_rms=np.sqrt(np.mean(controls**2)),
                maximum_per_actuator_rms=rms.max(), total_energy=energy, control_peak=peak, saturation_fraction=sat,
                budget_pass=budget, common_selection_score=score, p90_channel_distance_ratio=np.quantile(per,.90),
                both_improved_count=np.sum((tc<tf)&(oc<of)), sd_ratio_le2_count=np.sum(rho<=2)))
            channels.append(pd.DataFrame(dict(run_index=index, channel_index=np.arange(64), time_w1_free=tf,
                time_w1_controlled=tc, occupation_w1_free=of, occupation_w1_controlled=oc,
                mean_abs_error=mean_error, symmetric_sd_ratio=rho, both_improved=(tc<tf)&(oc<of))))
    frame = pd.DataFrame(rows)
    summary = {key:float(frame[key].mean()) for key in frame.columns if key not in ('budget_pass','context_index','run_index')}
    summary['budget_pass'] = bool(frame.budget_pass.all())
    summary['maximum_per_actuator_rms'] = float(frame.maximum_per_actuator_rms.max())
    summary['maximum_total_energy'] = float(frame.total_energy.max())
    return summary, frame, pd.concat(channels,ignore_index=True)


def train(runner, args, out):
    if out.exists():
        raise RuntimeError('Refusing existing arm/run overwrite')
    out.mkdir(parents=True)
    s = setup(runner,args)
    s['ranking'].to_csv(out/'part1_weighted_ranking.csv',index=False)
    runner.dump(out/'warm_start_migration_audit.json',s['migration'])
    contract = dict(subject='HUP065',seed=args.seed,nodes=args.nodes,actuation_fraction=args.nodes/64,
        graph_spread=args.tau,model_sha256=runner.MODEL_HASHES['HUP065'],input_paths=s['hashes'],
        source_sha256=runner.sha(runner.SOURCE), source_code_sha256=runner.sha(__file__),
        actor_updates=args.updates,actor_lr=args.lr,actor_min_lr=1e-5,critic_lr=1e-4,n_critic=3,
        adversarial_weight=args.adv,law_existing_coefficient_overrides=ARMS[args.arm],
        initialization=s['migration'],particles=32,horizon=256,amplitude_limit=1.8,per_actuator_rms_cap=.405,
        input_energy_cap=3.7908,diffusion_scale=.79451175,control_step_scale=.5,
        fit_contexts=s['fit_ledger'],selection_context_index=5,terminal_veto_context_index=6,
        outer_used_for_training_or_selection=False,no_runtime_projection_or_rescaling=True,
        fixed_common_selection_score='.4*mean(time W1)/free+.4*mean(occupation W1)/free+.2*p90(.5*time/free+.5*occupation/free), per-channel free denominator floor .01',
        budget_first=True,validation_every=25,all_reference_paths_used=True,not_adopted_in_paper=True)
    runner.dump(out/'training_contract.json',contract)
    core,actor,critic=s['core'],s['actor'],s['critic']
    actor_opt=torch.optim.AdamW(actor.parameters(),lr=args.lr,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(actor_opt,T_max=args.updates,eta_min=1e-5)
    critic_opt=torch.optim.Adam(critic.parameters(),lr=1e-4,betas=(0.,.9))
    cache=runner.validation_cache(s)
    initial,_,_=validate(runner,s,cache)
    history=[dict(update=0,**initial)]
    best_score = initial['common_selection_score'] if initial['budget_pass'] else float('inf')
    best_update=0
    best_actor,best_critic=copy.deepcopy(actor.state_dict()),copy.deepcopy(critic.state_dict())
    torch.save(dict(actor_state_dict=best_actor,critic_state_dict=best_critic,best_update=0,training_contract=contract),out/'initial_actor.pt')
    runner.dump(out/'initial_validation.json',initial)
    indices=tuple(range(15,256,16))
    started=time.perf_counter()
    for update in range(1,args.updates+1):
        slot=(update-1)%len(s['initials'])
        initial_state=s['initials'][slot]
        noise=core.antithetic_noise(args.seed+1009*update,s['stepper'].q)
        actor.eval()
        with torch.no_grad():
            detached=core.empirical_fp_rollout(s['stepper'],actor,initial_state,noise).scaled[:,1:].detach()
            baseline=core.uncontrolled_particle_rollout(s['stepper'],initial_state,noise)[:,1:]
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
            torch.nn.utils.clip_grad_norm_(critic.parameters(),5.)
            critic_opt.step()
        core.set_requires_grad(critic,False)
        core.set_requires_grad(actor,True)
        actor.train()
        actor_opt.zero_grad(set_to_none=True)
        rollout=core.empirical_fp_rollout(s['stepper'],actor,initial_state,noise)
        loss,terms=law(runner,s,rollout,s['fit'],baseline,True,args.arm)
        adv,_=core.actor_wasserstein_loss(critic,rollout.scaled[:,1:],s['fit'],indices)
        total=loss+args.adv*adv
        if not torch.isfinite(total):raise RuntimeError('Nonfinite actor')
        total.backward()
        grad=torch.nn.utils.clip_grad_norm_(actor.parameters(),1.)
        actor_opt.step()
        scheduler.step()
        row=dict(update=update,train_total=float(total.detach()),train_law=float(loss.detach()),train_adv=float(adv.detach()),
            actor_gradient_norm_before_clip=float(grad),fit_context_slot=slot)
        row.update({'train_'+k:float(v.detach()) for k,v in terms.items()})
        if update==1 or update%25==0 or update==args.updates:
            result,_,_=validate(runner,s,cache)
            row.update(result)
            if result['budget_pass'] and result['common_selection_score']<best_score:
                best_score,best_update=result['common_selection_score'],update
                best_actor,best_critic=copy.deepcopy(actor.state_dict()),copy.deepcopy(critic.state_dict())
                torch.save(dict(actor_state_dict=best_actor,critic_state_dict=best_critic,best_update=best_update,
                    training_contract=contract),out/'best_so_far.pt')
        history.append(row)
        if update==1 or update%10==0 or update==args.updates:
            pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
            elapsed=time.perf_counter()-started
            progress=dict(status='running',subject='HUP065',arm=args.arm,nodes=args.nodes,update=update,
                planned_updates=args.updates,best_update=best_update,best_common_selection_score=best_score,
                elapsed_seconds=elapsed,remaining_seconds_linear_estimate=elapsed/update*(args.updates-update),outer_opened=False)
            runner.dump(out/'progress.json',progress)
            print(json.dumps(progress),flush=True)
    torch.save(dict(actor_state_dict=actor.state_dict(),critic_state_dict=critic.state_dict(),update=args.updates,
        training_contract=contract),out/'last_actor_wgan.pt')
    actor.load_state_dict(best_actor)
    critic.load_state_dict(best_critic)
    selected,trajectories,channels=validate(runner,s,cache)
    trajectories.to_csv(out/'selected_validation_trajectory_metrics.csv',index=False)
    channels.to_csv(out/'selected_validation_channel_metrics.csv',index=False)
    torch.save(dict(actor_state_dict=best_actor,critic_state_dict=best_critic,best_update=best_update,
        training_contract=contract,selected_validation=selected),out/'frozen_actor_wgan.pt')
    if any(runner.sha(path)!=digest for path,digest in s['hashes'].items()):
        raise RuntimeError('Frozen inputs changed')
    summary=dict(status='completed_parameter_screen',subject='HUP065',arm=args.arm,nodes=args.nodes,
        trained_updates=args.updates,selected_update=best_update,selected_validation=selected,initial_validation=initial,
        checkpoint_sha256=runner.sha(out/'frozen_actor_wgan.pt'),warm_start=True,outer_opened=False,
        elapsed_seconds=time.perf_counter()-started,not_adopted_in_paper=True)
    runner.dump(out/'training_summary.json',summary)
    runner.dump(out/'progress.json',summary)
    print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=tuple(ARMS),required=True)
    parser.add_argument('--nodes',type=int,choices=(32,40),default=32)
    parser.add_argument('--tau',type=float,default=.2)
    parser.add_argument('--seed',type=int,default=20261011)
    parser.add_argument('--updates',type=int,default=150)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--adv',type=float,default=.5)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--init-checkpoint')
    parser.add_argument('--tag',required=True)
    args=parser.parse_args()
    if not args.tag.replace('_','').replace('-','').isalnum():raise RuntimeError('Simple unique tag required')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    runner=load_runner()
    train(runner,args,HERE/'runs'/args.tag)


if __name__=='__main__':main()
