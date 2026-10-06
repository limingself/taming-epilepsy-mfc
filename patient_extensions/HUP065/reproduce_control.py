"""Self-contained relocation-only replay of exact six-pair HUP065 sources.

Math/selection/optimizer function bodies are never rewritten. Frozen source and
predictive inputs plus old23 warm-start checkpoint are included in portable_inputs.
Only three optimizer path bindings and the new-output directory are redirected.
Use trusted local/archive PyTorch inputs; hashes are checked before deserialization.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

def long_path(path):
    value=str(Path(path).absolute())
    if os.name=='nt' and not value.startswith('\\\\?\\'):
        value='\\\\?\\'+value
    return Path(value)


PROJECT=Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE=PROJECT/'patient_results/HUP065_sparse_control_final_v1/current_full_wgangp'
BUNDLE_ARGUMENT=Path(sys.argv[sys.argv.index('--bundle-root')+1]) if '--bundle-root' in sys.argv else DEFAULT_BUNDLE
BASE=long_path(BUNDLE_ARGUMENT/'small_gain_decoupled_screen')
INPUTS=BASE/'portable_inputs'
SOURCES=INPUTS/'optimizer_sources'
BUNDLE=INPUTS/'external_bundle'
OLD=INPUTS/'old23_source_run'
EXPECTED={
    'optimize_hup065.py':'8e52540c3aa7ca0a798e702b4beddc6d2f63232a0bd14192f3d33bb8398c688f',
    'screen_small_gain.py':'4defa4e8c6fea104a983ea6f2ed1374b630dee8f6e1414a8c84df269c46aace0',
    'finalize_hup065.py':'57b32e5bb09f7a139ef442d2f806f64fa43e73ba2763e32d29833bd17d1016ad',
}
OLD_SHA='9014e2ce416021387515338ddf1ffc3ca435be085d1ea3cd66147f9a78f4b475'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def verified_optimizer(runtime):
    for filename,digest in EXPECTED.items():
        if sha(SOURCES/filename)!=digest:raise RuntimeError('Exact parameter-screen source snapshot changed')
    if sha(OLD/'frozen_actor_wgan.pt')!=OLD_SHA:raise RuntimeError('Old23 warm-start checkpoint changed')
    portable=load('selfcontained_external_route',BUNDLE/'reproduce_external.py')
    manifest=portable.read_manifest()
    verified=portable.verify_hashes(manifest)
    optimizer=load('optimize_hup065',SOURCES/'optimize_hup065.py')
    optimizer.BUNDLE=BUNDLE
    optimizer.OLD_RUN=OLD
    optimizer.HERE=runtime
    return optimizer,verified


def fresh_output(args):
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}',args.tag):raise RuntimeError('Use a new simple tag')
    root=long_path(args.output_root).resolve() if args.output_root else BASE/'portable_generated'
    output=(root/args.tag).resolve()
    if output.parent!=root.resolve() or INPUTS in output.parents:raise RuntimeError('Unsafe replay output')
    if output.exists():raise RuntimeError('Refusing an existing replay output')
    output.mkdir(parents=True)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('verify-bundle','screen-train','terminal-evaluate'))
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--bundle-root',type=Path,default=DEFAULT_BUNDLE)
    parser.add_argument('--output-root')
    parser.add_argument('--tag',required=True)
    parser.add_argument('--run-dir')
    args=parser.parse_args()
    output=fresh_output(args)
    optimizer,verified=verified_optimizer(output/'.runtime')
    if args.mode=='verify-bundle':
        optimizer.torch.set_num_threads(args.threads)
        optimizer.torch.set_num_interop_threads(1)
        runner=optimizer.load_runner()
        config=SimpleNamespace(seed=20261011,nodes=32,tau=.2,init_checkpoint=None)
        s=optimizer.setup(runner,config)
        imports={}
        for name,module in list(sys.modules.items()):
            if name=='mfc_pipeline' or name.startswith('mfc_pipeline.'):
                path=Path(module.__file__).resolve()
                if BUNDLE.resolve() not in path.parents:raise RuntimeError('Canonical import escaped portable input snapshot')
                imports[name]=str(path.relative_to(BASE))
        report=dict(status='verified_selfcontained_hashes_imports_plant_and_functional_warmstart',
            external_snapshot_files_verified=verified,source_snapshots_verified=EXPECTED,
            old23_checkpoint_sha256=OLD_SHA,predictive_model_sha256=runner.MODEL_HASHES['HUP065'],
            channels=64,configured_actuators=32,source_actuators=23,
            warm_start_output_max_error=s['migration']['initial_output_max_error'],
            new_actuator_initial_peak=s['migration']['new_actuator_initial_peak'],
            math_function_bodies_unchanged=True,filesystem_routing_only=True,
            original_D_absolute_paths_not_required=True,training_started=False,outer_arrays_opened=False,
            context6_opened=False,canonical_dependency_imports=imports)
        runner.dump(output/'verification_report.json',report)
        print(json.dumps(report),flush=True)
        return
    if args.mode=='screen-train':
        screen=load('selfcontained_small_gain_screen',SOURCES/'screen_small_gain.py')
        screen.HERE=output
        sys.argv=[str(screen.__file__),'--threads',str(args.threads)]
        screen.main()
        return
    if not args.run_dir:raise RuntimeError('terminal-evaluate requires completed results --run-dir')
    run=long_path(args.run_dir)
    decision=json.loads((run/'selection_report.json').read_text(encoding='utf-8'))
    original_receipt=run/'frozen_dev_selection_receipt.json'
    receipt=json.loads(original_receipt.read_text(encoding='utf-8'))
    if decision.get('adaptation_completed') is not True or decision.get('adaptation_final_qualifies') is not True:
        raise RuntimeError('Completed eligible150-update development receipt required')
    checkpoint=run/'selected_pair_adaptation_u150'/'frozen_actor_wgan.pt'
    if sha(checkpoint)!=receipt['selected_checkpoint_sha256']:raise RuntimeError('Frozen selected controller hash changed')
    receipt['selected_checkpoint']=str(checkpoint.resolve())
    receipt['replay_source_receipt_sha256']=sha(original_receipt)
    runner=optimizer.load_runner()
    relocated_receipt=output/'relocated_development_selection_receipt.json'
    runner.dump(relocated_receipt,receipt)
    finalizer=load('selfcontained_small_gain_finalizer',SOURCES/'finalize_hup065.py')
    sys.argv=[str(finalizer.__file__),'--selection-receipt',str(relocated_receipt),
        '--output',str(output/'final_evaluation'),'--threads',str(args.threads)]
    finalizer.main()


if __name__=='__main__':main()
