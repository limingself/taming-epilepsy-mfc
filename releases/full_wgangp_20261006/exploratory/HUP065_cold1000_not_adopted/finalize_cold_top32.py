"""Once-only veto and, only after passing it, one post-hoc outer diagnostic.

This never trains, chooses a replacement, or bypasses the predeclared two-dev-
endpoint comparison. No historical actor/critic/teacher checkpoint is loaded.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import run_cold_top32 as experiment


def verify_frozen(run):
    summary = json.loads((run/'training_summary.json').read_text(encoding='utf-8'))
    receipt = json.loads((run/'selection_receipt.json').read_text(encoding='utf-8'))
    protocol = json.loads((run/'protocol.json').read_text(encoding='utf-8'))
    checkpoint = run/'frozen_actor_wgan.pt'
    history = pd.read_csv(run/'training_history.csv')
    if summary['status'] != 'completed_neutral_cold1000_parameter_screen' or summary['trained_updates'] != 1000:
        raise RuntimeError('Wait for the full predeclared1000 updates')
    if not np.array_equal(history['update'].to_numpy(),np.arange(1001)):
        raise RuntimeError('Incomplete, missing or duplicated training updates')
    if summary['actor_and_critic_checkpoint_load_calls'] or summary['actor_teacher_or_critic_warmstart']:
        raise RuntimeError('Expected actual neutral training without teacher/actor/critic loads')
    if summary['outer_opened'] or summary['ctx6_opened']:
        raise RuntimeError('Terminal data already opened during training')
    digest = experiment.sha(checkpoint)
    if digest != summary['frozen_checkpoint_sha256'] or digest != receipt['frozen_checkpoint_sha256']:
        raise RuntimeError('Frozen policy identity changed after development selection')
    if protocol['runner_source_sha256'] != experiment.sha(experiment.__file__):
        raise RuntimeError('Scientific training source changed after preflight')
    if protocol['runner_source_sha256'] != experiment.sha(run/'run_cold_top32_source_snapshot.py'):
        raise RuntimeError('Training-source snapshot differs from the declared source')
    for name,digest in protocol['baseline_source_hashes'].items():
        if experiment.sha(experiment.BASELINE/name) != digest:
            raise RuntimeError('Adopted32 dev comparison bytes changed after predeclaration')
    if receipt['outer_used_for_selection'] or receipt['ctx6_used_for_reselection']:
        raise RuntimeError('Development-only checkpoint selection required')
    if receipt['terminal_eligible'] != summary['terminal_eligible']:
        raise RuntimeError('Development eligibility receipts disagree')
    return summary,receipt,protocol,checkpoint


def immutable_input_digest_map(s):
    # Digest-multiset parity is relocation-safe; the portable routing layer also
    # verifies every individual frozen file against its relative manifest name.
    return sorted(s['hashes'].values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--threads',type=int,choices=(2,),default=2)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    run = experiment.HERE/'runs'/args.tag
    out = run/'terminal_once'
    summary,receipt,protocol,checkpoint = verify_frozen(run)
    audit = json.loads((run/'development_audit.json').read_text(encoding='utf-8'))
    if audit['status'] != 'passed' or audit['selected_checkpoint_sha256'] != experiment.sha(checkpoint):
        raise RuntimeError('Complete independently recalculated development audit before once-only finalization')
    if out.exists():
        raise RuntimeError('Once-only terminal directory already exists; no rerun/rescue allowed')
    out.mkdir()
    runner,count = experiment.load_runner(run)
    runner.dump(out/'frozen_development_selection_receipt.json',dict(receipt,
        completed_training_summary_sha256=experiment.sha(run/'training_summary.json'),
        full_training_history_sha256=experiment.sha(run/'training_history.csv'),
        finalization_source_sha256=experiment.sha(__file__)))
    if receipt['terminal_eligible'] is not True:
        runner.dump(out/'final_status.json',dict(status='development_dual_endpoint_comparison_failed',
            selected_update=receipt['selected_update'],outer_arrays_opened=False,ctx6_opened=False,
            selection_track='best_budget_feasible_development_preview_not_terminal_eligible',
            terminal_eligible=False,adopted32_development_baseline=receipt['adopted32_development_baseline'],
            selected_development_metrics=audit['selected'],
            no_terminal_checkpoint_rescue=True,not_automatically_adopted=True,
            frozen_checkpoint_sha256=experiment.sha(checkpoint)))
        print('No eligible cold candidate: terminal and outer remain unopened.',flush=True)
        return
    setup_args = SimpleNamespace(seed=protocol['seed'])
    with experiment.forbid_controller_loads() as forbidden:
        s = experiment.setup_neutral(runner,setup_args)
    if forbidden or immutable_input_digest_map(s) != sorted(protocol['frozen_input_identities'].values()):
        raise RuntimeError('Frozen predictor/data identity mismatch or historical controller load')
    # The sole controller load is this completed experiment's already-frozen
    # policy, in evaluation mode. It is not a training warm start.
    payload = torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload['training_contract'] != protocol or payload['update'] != receipt['selected_update']:
        raise RuntimeError('Frozen selected checkpoint contract/update mismatch')
    s['actor'].load_state_dict(payload['actor_state_dict'],strict=True)
    s['actor'].eval()
    baseline,_ = experiment.adopted_baseline()
    actual,_,_ = experiment.validate(runner,s,runner.validation_cache(s),baseline)
    cached = json.loads((run/'validation'/f"u{receipt['selected_update']:04d}"/'summary.json').read_text(encoding='utf-8'))
    for key in ('common_selection_score','mean_time_w1_controlled','mean_occupation_w1_controlled',
                'maximum_per_actuator_rms','maximum_total_energy','mean_abs_log_sd_ratio'):
        if abs(actual[key]-cached[key]) > 1e-12:
            raise RuntimeError(f'Frozen dev replay mismatch: {key}')
    if actual['terminal_candidate_eligible'] is not True:
        raise RuntimeError('Frozen dev replay does not satisfy the predeclared eligibility rule')
    runner.dump(out/'frozen_development_replay.json',actual)
    eval_args = SimpleNamespace(subject='HUP065',seed=protocol['seed'])
    initials,observed = [],[]
    for name in s['p']['development_runs']:
        ictal = s['arrays']['ictal_'+name.replace('-','_')]
        stop = runner.boundary(ictal,6)
        initials.append(s['adapter'].initial_state_from_context(ictal[stop-256:stop]))
        observed.append(s['model'].transform.scaler.transform(ictal[stop:stop+256]))
    veto = runner.evaluate_set(eval_args,s,initials,s['validation'].numpy(),observed,
        'all-development','ctx6_terminal_veto',out/'ctx6_terminal_veto')
    runner.dump(out/'terminal_veto_receipt.json',dict(veto,
        checkpoint_sha256=experiment.sha(checkpoint),ctx6_used_for_reselection=False,
        outer_arrays_opened=False,original_budget_retained=True))
    if veto['terminal_veto_pass'] is not True:
        runner.dump(out/'final_status.json',dict(status='ctx6_terminal_veto_failed',
            outer_arrays_opened=False,ctx6_opened=True,checkpoint_reselection_after_veto=False,
            not_automatically_adopted=True,frozen_checkpoint_sha256=experiment.sha(checkpoint)))
        print('Ctx6 veto failed: stop before any outer array opening.',flush=True)
        return
    # Exactly one already-revealed outer diagnostic; no alternate checkpoint.
    with np.load(s['p']['outer'],allow_pickle=False) as archive:
        outer = {key:np.asarray(archive[key]) for key in archive.files}
    if not np.array_equal(outer['channels'].astype(str),s['arrays']['channels'].astype(str)):
        raise RuntimeError('Outer channel identity/order changed')
    reference = s['model'].transform.scaler.transform(outer['preictal_reference']).reshape(-1,256,64)
    ictal = outer['ictal']
    initials,observed = [],[]
    for index in range(8):
        stop = runner.boundary(ictal,index)
        initials.append(s['adapter'].initial_state_from_context(ictal[stop-256:stop]))
        observed.append(s['model'].transform.scaler.transform(ictal[stop:stop+256]))
    result = runner.evaluate_set(eval_args,s,initials,reference,observed,s['p']['outer_fold'],
        s['p']['outer_stage'],out/'outer_posthoc_amendment',display=7)
    old_path = experiment.HERE/'frozen_inputs'/'adopted32_outer_comparison'/'evaluation_summary.json'
    old = json.loads(old_path.read_text(encoding='utf-8'))
    if old['checkpoint_sha256'] != '2796775c4feb4534e9a4a294b2ab95687d77559b60bfab24cb383a42f1f25fdf':
        raise RuntimeError('Comparison is not the presently adopted32 variant')
    parity = max(abs(result[key]-old[key]) for key in ('mean_time_w1_free','mean_occupation_w1_free'))
    if parity > 1e-7:
        raise RuntimeError('Zero-control outer parity failed; report diagnostic error, never reselect')
    result.update(status='completed_once_posthoc_outer_diagnostic',
        classification='post-hoc exploratory cold-initialization amendment; outer had previously been revealed',
        checkpoint_sha256=experiment.sha(checkpoint),selected_update=payload['update'],
        trained_updates=1000,neutral_initialization=True,warm_start=False,
        direct_actuation_fraction=.5,ctx6_veto_pass=True,ctx6_used_for_reselection=False,
        outer_used_for_training_or_checkpoint_selection=False,not_automatically_adopted=True,
        original_per_actuator_rms_cap_retained=True,original_total_energy_cap_retained=True,
        frozen_bundle_files_verified=count,old_zero_control_metric_parity_max_abs_error=parity,
        current32_baseline_outer_summary_sha256=experiment.sha(old_path),
        mean_time_endpoint_better_than_adopted32=result['mean_time_w1_controlled']<old['mean_time_w1_controlled'],
        mean_occupation_endpoint_better_than_adopted32=result['mean_occupation_w1_controlled']<old['mean_occupation_w1_controlled'])
    runner.dump(out/'outer_posthoc_amendment'/'evaluation_summary.json',result)
    if not all(runner.sha(path)==digest for path,digest in s['hashes'].items()):
        raise RuntimeError('Frozen input mutation during evaluation')
    runner.dump(out/'final_status.json',result)
    print(json.dumps(runner.ready(result)),flush=True)


if __name__ == '__main__':
    main()
