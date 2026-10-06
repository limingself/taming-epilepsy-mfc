"""Read-only portable hash/import/plant/frozen-policy preflight; never retrain."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import run_cold_top32 as experiment
from finalize_cold_top32 import verify_frozen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    output=experiment.long_path(Path(args.output))
    if output.exists():raise RuntimeError('Refuse verification receipt overwrite')
    run=experiment.HERE/'runs'/args.tag
    summary,receipt,protocol,checkpoint=verify_frozen(run)
    runner,count=experiment.load_runner(run)
    with experiment.forbid_controller_loads() as forbidden:
        s=experiment.setup_neutral(runner,SimpleNamespace(seed=protocol['seed']))
    if forbidden or sorted(s['hashes'].values())!=sorted(protocol['frozen_input_identities'].values()):
        raise RuntimeError('Portable frozen-input set mismatch or historical actor load')
    for name,digest in protocol['baseline_source_hashes'].items():
        if experiment.sha(experiment.BASELINE/name)!=digest:raise RuntimeError('Portable dev baseline snapshot mismatch')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload['training_contract']!=protocol or payload['update']!=receipt['selected_update']:
        raise RuntimeError('Portable frozen policy contract mismatch')
    s['actor'].load_state_dict(payload['actor_state_dict'],strict=True)
    s['critic'].load_state_dict(payload['critic_state_dict'],strict=True)
    if not np.array_equal(s['selected'],protocol['selected_indices']):
        raise RuntimeError('Portable weighted mask identity changed')
    imports={}
    for name,module in list(__import__('sys').modules.items()):
        if name=='mfc_pipeline' or name.startswith('mfc_pipeline.'):
            path=experiment.long_path(Path(module.__file__))
            if experiment.BUNDLE not in path.parents:raise RuntimeError('Scientific import escaped portable inputs')
            imports[name]=str(path.relative_to(experiment.HERE))
    if not all(runner.sha(path)==digest for path,digest in s['hashes'].items()):
        raise RuntimeError('Frozen input changed during verification')
    result=dict(status='passed_portable_no_retraining_preflight',frozen_bundle_files_verified=count,
        predictive_plant_sha256=experiment.sha(s['p']['model']),
        canonical_function_sha256=experiment.CANONICAL_SHA,scientific_imports=imports,
        configured_actuators=32,selected_indices=s['selected'].tolist(),
        trained_updates=1000,selected_update=receipt['selected_update'],
        frozen_checkpoint_sha256=experiment.sha(checkpoint),controller_initialization='neutral; no teacher or warm-start actor/critic loaded during training',
        current_completed_policy_loaded_for_shape_verification_only=True,
        historical_actor_or_critic_weights_opened=False,no_rollout_evaluation_or_retraining=True,
        no_ctx6_or_outer_arrays_opened=True,dev_eligibility=receipt['terminal_eligible'],
        all_frozen_inputs_unchanged=True,not_automatically_adopted=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
