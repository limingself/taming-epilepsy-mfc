"""Original65 visual geometry; honest cold32 development/outer display binding.

Failed selection/veto never opens outer arrays just to produce a figure.
All curves use the original fixed KDE bandwidth and standardized x grid.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance

import run_cold_top32 as experiment
from finalize_cold_top32 import verify_frozen


LAYOUT=experiment.HERE/'frozen_inputs'/'plot_original65_layout_snapshot.py'
spec=importlib.util.spec_from_file_location('hup065_original_cold_display_layout',LAYOUT)
plot=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=plot
spec.loader.exec_module(plot)
if experiment.sha(LAYOUT)!='10391ffc261ff08ecebc30424f23ff76367f3de14dbb7a52bf9b62131b6f80e3':
    raise RuntimeError('Original65 rendering source snapshot mismatch')


def bound_display(run):
    training,receipt,protocol,checkpoint=verify_frozen(run)
    terminal=run/'terminal_once'
    status=json.loads((terminal/'final_status.json').read_text(encoding='utf-8'))
    if status['status']=='completed_once_posthoc_outer_diagnostic':
        summary=json.loads((terminal/'outer_posthoc_amendment'/'evaluation_summary.json').read_text(encoding='utf-8'))
        if summary['checkpoint_sha256']!=experiment.sha(checkpoint) or summary['ctx6_veto_pass'] is not True or summary['context_bank_evaluations']!=24:
            raise RuntimeError('Incomplete or differently bound once-only outer evaluation')
        data=terminal/'outer_posthoc_amendment'/'display_context_rollout.npz'
        with np.load(data,allow_pickle=False) as archive:
            arrays={name:np.asarray(archive[name]) for name in archive.files}
        if int(arrays['display_context_index'])!=7 or int(arrays['display_crn_bank'])!=0:
            raise RuntimeError('Wrong predeclared outer display')
        frame=pd.read_csv(terminal/'outer_posthoc_amendment'/'all_channel_context_bank_metrics.csv')
        expected=frame[(frame.context_index==7)&(frame.crn_bank==0)].sort_values('channel_index')
        return arrays,expected,dict(role='already_revealed_outer_posthoc_preview',context=7,bank=0,
            display_source_sha256=experiment.sha(data),outer_arrays_opened_by_plot=False,
            outer_summary=summary),training,receipt
    if status['status'] not in ('development_dual_endpoint_comparison_failed','ctx6_terminal_veto_failed'):
        raise RuntimeError('Complete terminal decision receipt required')
    target=run/'development_preview_context05_run01'
    if target.exists():raise RuntimeError('Refuse repeated preview rollout/overwrite')
    runner,_=experiment.load_runner(run)
    with experiment.forbid_controller_loads() as forbidden:
        s=experiment.setup_neutral(runner,SimpleNamespace(seed=protocol['seed']))
    if forbidden:raise RuntimeError('No historical controller load is allowed')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload['training_contract']!=protocol or payload['update']!=receipt['selected_update']:
        raise RuntimeError('Cold preview policy binding mismatch')
    s['actor'].load_state_dict(payload['actor_state_dict'],strict=True)
    s['actor'].eval()
    noise=s['core'].antithetic_noise(20260922,s['stepper'].q)
    with torch.no_grad():
        free=s['core'].uncontrolled_particle_rollout(s['stepper'],s['selections'][0],noise)[:,1:].numpy()
        rollout=s['core'].empirical_fp_rollout(s['stepper'],s['actor'],s['selections'][0],noise)
    ictal=s['arrays']['ictal_run_01']
    stop=runner.boundary(ictal,5)
    arrays=dict(free_standardized=free,controlled_standardized=rollout.scaled[:,1:].numpy(),
        reference_standardized=s['validation'].numpy(),controls=rollout.controls.numpy(),
        commands=rollout.commands.numpy(),standard_normal=noise.numpy(),
        observed_standardized=s['model'].transform.scaler.transform(ictal[stop:stop+256]),
        channels=s['arrays']['channels'],direct_mask=s['mask'],selected_indices=s['selected'],
        display_context_index=np.asarray(5),display_crn_bank=np.asarray(0))
    path=run/'validation'/f"u{receipt['selected_update']:04d}"/'channel_metrics.csv'
    frame=pd.read_csv(path)
    expected=frame[frame.run_index==0].sort_values('channel_index')
    target.mkdir()
    data=target/'display_context_rollout.npz'
    np.savez_compressed(data,**arrays)
    scope=dict(role='development_only_failed_improvement_preview',run='run-01',context=5,
        bank=0,noise_seed=20260922,terminal_status=status['status'],
        display_source_sha256=experiment.sha(data),source_validation_metrics_sha256=experiment.sha(path),
        outer_arrays_opened_by_plot=False,not_external_improvement_evidence=True)
    runner.dump(target/'scope.json',scope)
    return arrays,expected,scope,training,receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    run=experiment.HERE/'runs'/args.tag
    out=run/'figures_original_style'
    if out.exists():raise RuntimeError('Refuse historical preview replacement')
    arrays,expected,scope,training,receipt=bound_display(run)
    if arrays['observed_standardized'].shape!=(256,64) or arrays['controlled_standardized'].shape!=(32,256,64) or arrays['free_standardized'].shape!=(32,256,64):
        raise RuntimeError('Original particles/step/channel dimensions differ')
    if not np.array_equal(np.flatnonzero(arrays['direct_mask']),arrays['selected_indices']) or int(np.sum(arrays['direct_mask']))!=32:
        raise RuntimeError('Mask/index or configured-actuator mismatch')
    protocol=json.loads((run/'protocol.json').read_text(encoding='utf-8'))
    if not np.array_equal(arrays['selected_indices'],protocol['selected_indices']):
        raise RuntimeError('Plot must use frozen weighted top32')
    for style in plot.SERIES.values():
        if not np.isfinite(arrays[style[0]]).all():raise RuntimeError('Nonfinite displayed sample')
    original=experiment.HERE/'frozen_inputs'/'original65_density_grid.csv'
    old=pd.read_csv(original,encoding='utf-8-sig')
    if set(old.subject_id.astype(str))!={'HUP065'} or set(old.series_code.astype(str))!=set(plot.SERIES):
        raise RuntimeError('Original65 grid belongs to another subject or curve contract')
    labels=old[['channel_index','channel']].drop_duplicates().sort_values('channel_index')
    if not np.array_equal(labels.channel_index.to_numpy(),np.arange(64)) or not np.array_equal(labels.channel.astype(str).to_numpy(),arrays['channels'].astype(str)):
        raise RuntimeError('Original channel labels/order changed')
    grids={}
    for index in range(64):
        frame=old[(old.channel_index==index)&(old.series_code=='O')]
        grids[index]=frame.standardized_amplitude.to_numpy(dtype=float)
        if len(grids[index])!=240 or np.any(np.diff(grids[index])<=0):raise RuntimeError('Original x grid changed')
        for code in plot.SERIES:
            other=old[(old.channel_index==index)&(old.series_code==code)]
            if not np.array_equal(other.standardized_amplitude.to_numpy(dtype=float),grids[index]):
                raise RuntimeError('The original curves did not share one grid')
    minimum=float(old.standardized_amplitude.min()); maximum=float(old.standardized_amplitude.max())
    curves={(c,code):plot.density(arrays[style[0]][...,c],grids[c]) for c in range(64) for code,style in plot.SERIES.items()}
    rows=[]; metrics=[]
    reference=arrays['reference_standardized']
    for c in range(64):
        free=arrays['free_standardized'][...,c]; controlled=arrays['controlled_standardized'][...,c]; ref=reference[...,c]
        values=dict(subject='HUP065',channel_index=c,channel=str(arrays['channels'][c]),
            direct_actuated=bool(arrays['direct_mask'][c]),display_context=scope['context'],
            crn_bank=scope['bank'],display_role=scope['role'],
            time_w1_free=plot.time_w1(free,ref),time_w1_controlled=plot.time_w1(controlled,ref),
            occupation_w1_free=float(wasserstein_distance(free.ravel(),ref.ravel())),
            occupation_w1_controlled=float(wasserstein_distance(controlled.ravel(),ref.ravel())),
            reference_mean=float(ref.mean()),reference_sd=float(ref.std()),
            free_mean=float(free.mean()),free_sd=float(free.std()),
            controlled_mean=float(controlled.mean()),controlled_sd=float(controlled.std()))
        metrics.append(values)
        for code in plot.SERIES:
            rows.extend(dict(subject_id='HUP065',page=c//36+1,channel_index=c,channel=str(arrays['channels'][c]),
                series_code=code,series_label=plot.SERIES[code][1],standardized_amplitude=float(x),density=float(y))
                for x,y in zip(grids[c],curves[(c,code)]))
    calculated=pd.DataFrame(metrics).sort_values('channel_index')
    discrepancies=[float(np.abs(calculated[key].to_numpy()-expected[key].to_numpy()).max()) for key in
        ('time_w1_free','time_w1_controlled','occupation_w1_free','occupation_w1_controlled')]
    if len(expected)!=64 or max(discrepancies)>1e-12:raise RuntimeError('Displayed W1 differs from the frozen evaluation/validation endpoint')
    out.mkdir()
    outputs=[]; offset=0
    for page,count in enumerate((36,28),1):
        height=180-max(0,6-int(math.ceil(count/6)))*24.66
        fig,axes=plot.plt.subplots(6,6,figsize=(170/25.4,height/25.4),sharex=True,squeeze=False)
        for i,ax in enumerate(axes.flat):
            if i>=count:ax.set_visible(False); continue
            c=offset+i; row,col=divmod(i,6)
            ax.set_position([.030+col*.160,(height-29.7-row*24.66-17.1)/height,.145,17.1/height])
            ymax=0.
            for code,style in plot.SERIES.items():
                y=curves[(c,code)]
                ax.plot(grids[c],y,color=style[2],lw=style[3],ls=style[4],alpha=style[5])
                ymax=max(ymax,float(y.max()))
            ax.set_xlim(minimum,maximum); ax.set_ylim(0,ymax*1.10); ax.set_yticks([])
            ax.set_title(plot.clean_contact(str(arrays['channels'][c])),loc='left',fontsize=9,
                fontweight='bold',color='#4D4D4D',pad=2)
            ax.set_xticks([value for value in (-1,0,1) if minimum<=value<=maximum])
            has_below=any(j<count for j in range(i+6,count,6))
            ax.tick_params(axis='x',labelbottom=not has_below,labelsize=9,length=2)
        fig.text(.09,.984,f'HUP065: channel occupation laws ({page}/2)',va='top',ha='left',fontsize=10,fontweight='bold')
        legend=[plot.Line2D([0],[0],color=style[2],lw=1.2,ls=style[4],label=style[1]) for style in plot.SERIES.values()]
        fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.52,1-9/height),ncol=4,
            fontsize=9,handlelength=1.2,columnspacing=.5)
        fig.supxlabel('Standardized amplitude',fontsize=9,y=2.16/height)
        fig.canvas.draw()
        axes_visible=[ax for ax in fig.axes if ax.get_visible()]
        if len(axes_visible)!=count or any(len(ax.lines)!=4 for ax in axes_visible):raise RuntimeError('All displayed contacts require exactly four curves')
        base=out/f'hup065_ofrc_all_channels_page_{page:02d}'
        for extension in ('pdf','png','svg'):
            kwargs={'metadata':{'CreationDate':None,'ModDate':None}} if extension=='pdf' else {}
            fig.savefig(base.with_suffix('.'+extension),dpi=300,**kwargs)
        outputs.append(dict(page=page,channel_count=count,width_mm=170,height_mm=height,
            occupied_axis_count=len(axes_visible),curves_per_channel=4,
            artifact_sha256={extension:experiment.sha(base.with_suffix('.'+extension)) for extension in ('pdf','png','svg')}))
        plot.plt.close(fig); offset+=count
    pd.DataFrame(rows).to_csv(out/'source_data_channel_density_long.csv',index=False,encoding='utf-8-sig')
    calculated.to_csv(out/'source_data_display_metrics.csv',index=False,encoding='utf-8-sig')
    controls=arrays['controls']
    qa=dict(subject='HUP065',backend='Python/matplotlib',archetype='quantitative grid',
        scope=scope,pages=outputs,page_occupancy=[36,28],channel_count=64,selected_actuator_count=32,
        trained_updates=1000,selected_update=receipt['selected_update'],warm_start=False,neutral_initialization=True,
        original_grid_points=240,original_x_limits=[minimum,maximum],kde_absolute_bandwidth=.16,
        empirical_endpoint_binding_maximum_error=max(discrepancies),
        no_new_training_rollout_normalization_or_rescaling=True,no_distance_annotation=True,
        no_new_training_by_plot=True,no_sample_or_density_rescaling=True,
        new_development_preview_replay=(scope['role']=='development_only_failed_improvement_preview'),
        no_gate_annotation=True,no_original_actor_comparator=True,no_new_loss_figure=True,
        all_contacts_and_negative_cases_retained=True,human_visual_review=False,
        original_montage_labels_preserved_except_display_prefix_suffix=True,
        maximum_per_actuator_rms=float(np.sqrt(np.mean(controls**2,axis=(0,1))).max()),
        total_energy=float(np.mean(np.sum(controls**2,axis=-1))),control_peak=float(np.abs(controls).max()),
        source_sha256=dict(checkpoint=experiment.sha(run/'frozen_actor_wgan.pt'),
            original_density_source=experiment.sha(original),rendering_snapshot=experiment.sha(LAYOUT),
            wrapper=experiment.sha(__file__),figure_contract=experiment.sha(experiment.HERE/'figure_contract.json')),
        outside_axis_sample_fractions={code:float(np.mean((arrays[style[0]]<minimum)|(arrays[style[0]]>maximum))) for code,style in plot.SERIES.items()},
        plot_window_note='Original window is unchanged; empirical distances use every sample, including tails outside the viewing window.',
        overleaf_modified=False,historical_source_modified=False,not_automatically_adopted=True)
    (out/'external_original_style_numeric_qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='original_style_cold_preview_created',scope=scope,pages=outputs,
        empirical_endpoint_binding_maximum_error=max(discrepancies))),flush=True)


if __name__=='__main__':main()
