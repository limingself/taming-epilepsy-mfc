"""Honest32-node HUP065 bindings on an unchanged original64-contact layout.

Only the completed-run loader/resource metadata and filesystem paths are adapted.
Original65 grid, labels, palette, four curves, bandwidth and layout are unchanged.
No loss plot is made and no old/public manuscript figure is overwritten.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def long_path(path):
    value=str(Path(path).absolute())
    if os.name=='nt' and not value.startswith('\\\\?\\'):value='\\\\?\\'+value
    return Path(value)


HERE=long_path(Path(__file__).resolve().parent)
layout_path=HERE/'plot_external_layout_snapshot.py'
spec=importlib.util.spec_from_file_location('hup065_original_layout_snapshot',layout_path)
plot=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=plot
spec.loader.exec_module(plot)
if plot.sha(layout_path)!='10391ffc261ff08ecebc30424f23ff76367f3de14dbb7a52bf9b62131b6f80e3':
    raise RuntimeError('Unchanged original rendering layout snapshot hash mismatch')
plot.SUBJECTS['HUP065']['m']=32


def load_completed(subject,run):
    if subject!='HUP065':raise RuntimeError('Only patient65 original grids can be used by this wrapper')
    training_path=HERE/'results'/'selected_pair_adaptation_u150'/'training_summary.json'
    decision_path=HERE/'results'/'selection_report.json'
    receipt_path=HERE/'terminal_once'/'relocated_development_selection_receipt.json'
    evaluation_path=run/'outer_posthoc_amendment'/'evaluation_summary.json'
    for path in (training_path,decision_path,receipt_path,evaluation_path):
        if not path.is_file():raise RuntimeError('Complete freeze, veto and all24 diagnostics before plotting')
    training=json.loads(training_path.read_text(encoding='utf-8'))
    decision=json.loads(decision_path.read_text(encoding='utf-8'))
    receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    evaluation=json.loads(evaluation_path.read_text(encoding='utf-8'))
    if training['status']!='completed_parameter_screen' or training['trained_updates']!=150 or training['outer_opened'] is not False:
        raise RuntimeError('Completed development-only150-update adaptation required')
    if decision.get('adaptation_completed') is not True or decision.get('adaptation_final_qualifies') is not True:
        raise RuntimeError('Registered two-endpoint development eligibility failed')
    if receipt['outer_used_for_selection'] is not False or receipt['ctx6_used_for_selection'] is not False:
        raise RuntimeError('No outer/context6 selection is allowed')
    if evaluation['context_bank_evaluations']!=24 or evaluation['channels']!=64 or evaluation['direct_actuators']!=32:
        raise RuntimeError('All8 contexts x3banks /64contacts /32configured actuators required')
    if evaluation['ctx6_veto_pass'] is not True or evaluation['ctx6_used_for_reselection'] is not False:
        raise RuntimeError('Once-only successful terminal veto required')
    if evaluation['outer_used_for_training_or_checkpoint_selection'] is not False or evaluation['classification'].startswith('post-hoc') is not True:
        raise RuntimeError('Honest already-revealed post-hoc diagnostic metadata required')
    checkpoint=HERE/'results'/'selected_pair_adaptation_u150'/'frozen_actor_wgan.pt'
    digest=plot.sha(checkpoint)
    if any(value!=digest for value in (training['checkpoint_sha256'],evaluation['checkpoint_sha256'],receipt['selected_checkpoint_sha256'])):
        raise RuntimeError('Selected/frozen/diagnosed checkpoint identities differ')
    display_path=run/'outer_posthoc_amendment'/'display_context_rollout.npz'
    with np.load(display_path,allow_pickle=False) as archive:
        arrays={key:np.asarray(archive[key]) for key in archive.files}
    if int(arrays['display_context_index'])!=7 or int(arrays['display_crn_bank'])!=0:
        raise RuntimeError('Do not select a new prettier display case')
    if arrays['observed_standardized'].shape!=(256,64) or arrays['free_standardized'].shape!=(32,256,64) or arrays['controlled_standardized'].shape!=(32,256,64):
        raise RuntimeError('Unchanged original observation/particle/horizon shapes required')
    if arrays['reference_standardized'].ndim!=3 or arrays['reference_standardized'].shape[1:]!=(256,64):
        raise RuntimeError('Frozen patient-reference shape changed')
    for style in plot.SERIES.values():
        if not np.isfinite(arrays[style[0]]).all():raise RuntimeError('Nonfinite source law')
    network=HERE/'portable_inputs'/'external_bundle'/'snapshot'/'project'/'patient_results'/'HUP065_sparse_control_final_v1'/'frozen_run'/'science_run'/'p'/'ALLDEV_REFIT'/'network.npz'
    with np.load(network,allow_pickle=False) as archive:
        scores=archive['centrality_score']
    rank=np.lexsort((np.arange(64),-scores))
    if arrays['direct_mask'].astype(bool).sum()!=32 or not np.array_equal(arrays['selected_indices'],np.sort(rank[:32])):
        raise RuntimeError('Displayed mask must be the frozen weighted-centrality top32')
    if not np.array_equal(np.flatnonzero(arrays['direct_mask']),arrays['selected_indices']):raise RuntimeError('Mask/index mismatch')
    if len(set(arrays['channels'].astype(str)))!=64:raise RuntimeError('Duplicate contacts')
    return arrays,dict(status='completed_parameter_adaptation',subject=subject,trained_updates=150,
        selected_update=training['selected_update'],warm_start=True,starting_fresh_selected_update=550,
        previous_fresh_training_budget=1000),evaluation,dict(checkpoint_sha256=digest,
        display_npz_sha256=plot.sha(display_path),training_summary_sha256=plot.sha(training_path),
        evaluation_summary_sha256=plot.sha(evaluation_path),development_decision_sha256=plot.sha(decision_path),
        frozen_development_receipt_sha256=plot.sha(receipt_path),
        rendering_layout_snapshot_sha256=plot.sha(layout_path))


plot.load_completed=load_completed

if __name__=='__main__':
    for flag in ('--run-directory','--original-density-source','--output'):
        slot=sys.argv.index(flag)+1
        sys.argv[slot]=str(long_path(sys.argv[slot]))
    result=plot.main()
    output=long_path(sys.argv[sys.argv.index('--output')+1])
    path=output/'external_original_style_numeric_qa.json'
    qa=json.loads(path.read_text(encoding='utf-8'))
    qa.update(training_classification='explicit warm-start parameter adaptation, not from-scratch training',
        warm_start=True,starting_fresh_selected_update=550,previous_fresh_training_budget=1000,
        added_actor_updates_trained=150,selected_added_update=0,new_node_mean_gain=.05,new_node_deviation_gain=0.,
        original_RMS_cap=.405,original_energy_cap=3.7908,no_new_loss_figure=True,
        published_external_results_replaced=False,human_visual_review=False,
        small_global_improvement_not_uniform_restoration=True)
    path.write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
    raise SystemExit(result)
