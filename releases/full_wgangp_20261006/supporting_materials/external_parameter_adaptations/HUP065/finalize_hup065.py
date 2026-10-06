"""Once-only terminal veto and post-hoc outer diagnostics for a dev-selected arm.

Requires an immutable development-selection receipt containing checkpoint SHA.
It never changes any training/selection state or chooses a replacement after
context-6/outer diagnostics. A failed context-6 veto stops before opening outer.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import optimize_hup065 as experiment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-receipt',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    receipt_path=Path(args.selection_receipt)
    receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    out=Path(args.output)
    if out.exists():raise RuntimeError('Once-only final evaluation output must not already exist')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    runner=experiment.load_runner()
    checkpoint_path=Path(receipt['selected_checkpoint'])
    if runner.sha(checkpoint_path)!=receipt['selected_checkpoint_sha256']:
        raise RuntimeError('Selected checkpoint changed after development sealing')
    if receipt.get('outer_used_for_selection') is not False or receipt.get('ctx6_used_for_selection') is not False:
        raise RuntimeError('Development-only selection receipt required')
    payload=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    contract=payload['training_contract']
    setup_args=SimpleNamespace(seed=contract['seed'],nodes=contract['nodes'],tau=contract['graph_spread'],init_checkpoint=str(checkpoint_path))
    s=experiment.setup(runner,setup_args)
    eval_args=SimpleNamespace(subject='HUP065',seed=contract['seed'])
    s['actor'].eval()
    out.mkdir(parents=True)
    runner.dump(out/'frozen_development_selection_receipt.json',receipt)
    initials,observed=[],[]
    for run in s['p']['development_runs']:
        ictal=s['arrays']['ictal_'+run.replace('-','_')]
        stop=runner.boundary(ictal,6)
        initials.append(s['adapter'].initial_state_from_context(ictal[stop-256:stop]))
        observed.append(s['model'].transform.scaler.transform(ictal[stop:stop+256]))
    veto=runner.evaluate_set(eval_args,s,initials,s['validation'].numpy(),observed,
        'all-development','ctx6_terminal_veto',out/'ctx6_terminal_veto')
    runner.dump(out/'terminal_veto_receipt.json',dict(**veto,
        checkpoint_sha256=runner.sha(checkpoint_path),ctx6_used_for_reselection=False,
        outer_arrays_opened=False))
    if not veto['terminal_veto_pass']:
        runner.dump(out/'final_status.json',dict(status='ctx6_terminal_veto_failed',
            outer_arrays_opened=False,checkpoint_reselection_after_veto=False,not_adopted_in_paper=True))
        print('Terminal veto failed; no outer diagnostics or checkpoint rescue.',flush=True)
        return
    with np.load(s['p']['outer'],allow_pickle=False) as archive:
        outer={k:np.asarray(archive[k]) for k in archive.files}
    if not np.array_equal(outer['channels'].astype(str),s['arrays']['channels'].astype(str)):
        raise RuntimeError('Outer channel identity/order changed')
    ictal=outer['ictal']
    reference=s['model'].transform.scaler.transform(outer['preictal_reference']).reshape(-1,256,64)
    initials,observed=[],[]
    for index in range(8):
        stop=runner.boundary(ictal,index)
        initials.append(s['adapter'].initial_state_from_context(ictal[stop-256:stop]))
        observed.append(s['model'].transform.scaler.transform(ictal[stop:stop+256]))
    summary=runner.evaluate_set(eval_args,s,initials,reference,observed,s['p']['outer_fold'],
        s['p']['outer_stage'],out/'outer_posthoc_amendment',display=7)
    old_report=json.loads(s['p']['old_outer_report'].read_text(encoding='utf-8'))
    old=old_report.get('metrics',old_report.get('summary',old_report))
    parity=max(abs(summary['mean_time_w1_free']-old['mean_time_w1_free']),
        abs(summary['mean_occupation_w1_free']-old['mean_occupation_w1_free']))
    if parity>1e-7:raise RuntimeError('Old frozen zero-control outer parity failed')
    summary.update(classification='post-hoc exploratory parameter amendment; previously revealed outer run',
        checkpoint_sha256=runner.sha(checkpoint_path),selected_update=payload['best_update'],
        selection_receipt_sha256=runner.sha(receipt_path),warm_start=True,
        direct_actuation_fraction=contract['nodes']/64,original_per_actuator_rms_cap_retained=True,
        original_total_energy_cap_retained=True,old_zero_control_metric_parity_max_abs_error=parity,
        ctx6_veto_pass=True,ctx6_used_for_reselection=False,outer_used_for_training_or_checkpoint_selection=False,
        not_adopted_in_paper=True)
    runner.dump(out/'outer_posthoc_amendment/evaluation_summary.json',summary)
    runner.dump(out/'final_status.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
