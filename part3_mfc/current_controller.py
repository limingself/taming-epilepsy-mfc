"""Current in-place Full WGAN-GP commands; public checks never open signals."""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
CANONICAL='57da573dbe300ad6bd585fbd69e46102d9f315dba8862e63c7d3e8320b6943e1'
RESULTS=ROOT/'output/part3/source_data/figures_06_08/current_results.json'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))
def boolean(value):return str(value).lower() in ('true','1')

def verify_current(subject=None, *, check_manifest=True):
    current=read(RESULTS)
    if sha(ROOT/'part3_mfc/model_core.py')!=CANONICAL:raise RuntimeError('Canonical scientific source bytes changed')
    checked=[]
    for patient,case in current['case_results'].items():
        if subject and patient!=subject:continue
        table=rows(ROOT/case['metrics_csv'])
        if len(table)!=case['expected_rows']:raise RuntimeError('Missing current channel/context/bank rows')
        channels={r['channel'] for r in table}
        direct={r['channel'] for r in table if boolean(r[case['columns']['direct']])}
        if (len(channels),len(direct))!=(case['channels'],case['direct_nodes']):raise RuntimeError('Current mask count changed')
        for key,column in (('time_w1_free','time_w1_free'),('occupation_w1_free','occupation_w1_free'),
                           ('time_w1_controlled',case['columns']['time_controlled']),
                           ('occupation_w1_controlled',case['columns']['occupation_controlled'])):
            mean=math.fsum(float(r[column]) for r in table)/len(table)
            if not math.isclose(mean,case[key],rel_tol=1e-11,abs_tol=1e-12):raise RuntimeError('Current mean endpoint mismatch: '+patient+'/'+key)
        if not re.fullmatch('[0-9a-f]{64}',case['checkpoint_sha256']):raise RuntimeError('Missing exact frozen controller SHA')
        if patient!='HUP060':
            if len({(r['context_index'],r['crn_bank']) for r in table})!=24:raise RuntimeError('Not all8contexts/3banks')
            count=sum(all(boolean(r['full_gate_b_pass']) for r in table if r['channel']==channel) for channel in channels)
            if count!=case['full_gate_b_all_conditions'] or count!={'HUP065':21,'HUP080':57}[patient]:raise RuntimeError('Current all-condition Gate-B count mismatch')
        checked.append(dict(patient=patient,channels=len(channels),direct_nodes=len(direct),rows=len(table),checkpoint_sha256=case['checkpoint_sha256']))
    if current['case_results']['HUP060']['checkpoint_sha256']!='980952488c536b76cfc51fa5d979243e0a69be4cba1cbf36aacdc094e7b29ab0':raise RuntimeError('Not current selected Full700')
    if current['case_results']['HUP065']['selected_added_update']!=0 or current['case_results']['HUP080']['selected_added_update']!=125:raise RuntimeError('External selected lineage changed')
    expected_sha={'HUP065':'2796775c4feb4534e9a4a294b2ab95687d77559b60bfab24cb383a42f1f25fdf','HUP080':'c25d86d1a4dd9d0860106f10e800657e8be291ed3350313f0d9d05a410ab062c'}
    if any(current['case_results'][p]['checkpoint_sha256']!=h for p,h in expected_sha.items()):raise RuntimeError('Wrong adopted external checkpoint')
    audit=read(ROOT/'output/part3/source_data/figure_07/ablation_aggregation_qa.json')
    if audit['checkpoint_sha256']['full']!=current['case_results']['HUP060']['checkpoint_sha256']:raise RuntimeError('Component comparisons do not share current Full700')
    baseline=next(r for r in rows(ROOT/'output/part3/source_data/figure_09_trivial_baselines/trivial_baseline_summary.csv') if r['variant']=='full_actor_wgan')
    if not math.isclose(float(baseline['control_rms']),.6730960629447963,abs_tol=1e-12):raise RuntimeError('Baseline not current Full RMS')
    for name,key in [('mean_time_w1','time_w1_controlled'),('mean_occupation_w1','occupation_w1_controlled')]:
        if not math.isclose(float(baseline[name]),current['case_results']['HUP060'][key],abs_tol=1e-12):raise RuntimeError('Baseline Full endpoint mismatch')
    loss=rows(ROOT/'output/part3/source_data/figure_10_loss/HUP060_WGANGP_loss_source.csv')
    contract=read(ROOT/'output/part3/source_data/figure_10_loss/training_contract.json')
    if len(loss)!=1000 or int(loss[-1]['epoch'])!=1000 or contract['teacher_proximity_weight']!=0 or contract['gradient_penalty']!=10:raise RuntimeError('Loss is not actual Full1000/GP10/no-teacher')
    safety=rows(ROOT/'patient_extensions/HUP080/results/trajectory_safety_metrics.csv')
    old=sum(boolean(r['gate_c']) for r in safety)
    revised=sum(boolean(r['finite']) and float(r['maximum_per_actuator_rms'])<=.45 and float(r['total_energy'])<=5.6862 and float(r['control_peak'])<=1.8 and float(r['saturation_fraction'])<=.01 for r in safety)
    if len(safety)!=24 or old!=0 or revised!=24:raise RuntimeError('HUP080 old/revised budget evidence changed')
    if check_manifest:
        manifest=read(ROOT/'CURRENT_PROJECT_MANIFEST.json')
        for entry in manifest['files']:
            path=ROOT/entry['path']
            if not path.is_file() or sha(path)!=entry['sha256']:raise RuntimeError('Current project byte mismatch: '+entry['path'])
    return dict(status='pass',scope='current in-place public table/source/figure verification',cases=checked,
                canonical_part3_sha256=CANONICAL,private_arrays_loaded=False,models_deserialized=False,
                training_started=False,terminal_or_outer_evaluation_run=False)

def verify_figure(number):
    contract=read(ROOT/'paper_figures.json')
    figure=next(f for f in contract['figures'] if f['number']==number)
    manifest=read(ROOT/'CURRENT_PROJECT_MANIFEST.json')
    ledger={r['path']:r['sha256'] for r in manifest['files']}
    paths=[]
    for extension in figure.get('formats',['pdf','png','svg']):
        relative=figure['output_directory']+'/'+figure['stem']+'.'+extension
        if relative not in ledger or sha(ROOT/relative)!=ledger[relative]:raise RuntimeError('Wrong frozen current figure '+relative)
        paths.append(relative)
    return dict(status='pass',figure=number,mode='verified_current_frozen_public_exports_not_regenerated',files=paths,
                selected_full_update=700,training_started=False,signal_arrays_opened=False)

def figure_main(number):
    parser=argparse.ArgumentParser(description='Current Full WGAN-GP figure. Default verifies the approved frozen exports; --redraw uses public CSVs only.')
    parser.add_argument('--redraw',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    if args.redraw:
        if not args.output_dir:parser.error('--redraw requires a new --output-dir; approved frozen exports are not silently overwritten')
        if __package__:
            from .plot_current_source_data import draw
        else:
            from plot_current_source_data import draw
        return draw(number,args.output_dir)
    print(json.dumps(verify_figure(number),indent=2))
    return 0

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('verify','source-data','train','evaluate','evaluate-ablations'))
    parser.add_argument('--new-tag')
    parser.add_argument('--allow-evaluation',action='store_true')
    parser.add_argument('--epochs',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=20261011)
    parser.add_argument('--threads',type=int,default=4)
    args,extra=parser.parse_known_args(argv)
    if args.command in ('verify','source-data'):
        if extra:parser.error('Unexpected verification arguments')
        print(json.dumps(verify_current(),indent=2));return 0
    if not args.new_tag or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}',args.new_tag):parser.error('Private execution requires an explicit new --new-tag')
    if args.new_tag=='fresh_seed20261011_e1000':parser.error('Adopted run tag cannot be reused')
    if args.command.startswith('evaluate') and not args.allow_evaluation:parser.error('Evaluation is not a synchronization step; explicit --allow-evaluation required')
    target=ROOT/'part3_mfc'/('run_component_ablation.py' if args.command=='evaluate-ablations' else 'train_full_wgangp.py')
    mode='train' if args.command=='train' else 'evaluate'
    command=[sys.executable,'-B',str(target),mode,'--tag',args.new_tag,'--threads',str(args.threads)]
    if mode=='train':command+=['--epochs',str(args.epochs),'--seed',str(args.seed)]
    return subprocess.run([*command,*extra],cwd=ROOT,check=False).returncode

if __name__=='__main__':raise SystemExit(main())
