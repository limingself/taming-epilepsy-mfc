"""Portable routing wrapper for the exact isolated HUP065/HUP080 runners.

No predictive dynamics, actor algebra, law coefficients, optimization settings,
reference sampler, selection score, veto, or evaluation gates are rewritten.
The runner snapshots retain their original bytes. Only filesystem bindings and
historical path identities are relocated in the loaded module.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import sys
import types
import uuid

BUNDLE = Path(__file__).resolve().parent
PROJECT = BUNDLE / 'snapshot' / 'project'
MANIFEST = BUNDLE / 'bundle_manifest.json'
INPUT_ROLES = {'official_training_input'}
EXPECTED_PACKAGES = ('numpy', 'scipy', 'pandas', 'scikit-learn', 'joblib',
                     'torch', 'matplotlib', 'networkx', 'PyYAML')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_manifest():
    return json.loads(MANIFEST.read_text(encoding='utf-8-sig'))


def bundled_path(relative):
    result = (BUNDLE / relative).resolve()
    if BUNDLE not in result.parents:
        raise RuntimeError(f'Unsafe manifest path: {relative}')
    return result


def verify_hashes(manifest):
    for entry in manifest['files']:
        path = bundled_path(entry['relative_path'])
        if not path.is_file() or sha(path) != entry['sha256']:
            raise RuntimeError(f'Frozen snapshot mismatch: {entry["relative_path"]}')
    return len(manifest['files'])


def path_identity(path):
    return str(path).replace('\\', '/').casefold()


def load_runner(subject, manifest, runtime, output_root):
    """Retain all function ASTs; redirect only the four top-level path bindings."""
    source = bundled_path(manifest['runners'][subject])
    tree = ast.parse(source.read_text(encoding='utf-8-sig'), filename=str(source))
    redirected = {'ROOT', 'RUNTIME', 'SOURCE', 'PATIENT_BASE'}
    replacements = {
        'ROOT': ast.Name(id='_portable_output_root', ctx=ast.Load()),
        'RUNTIME': ast.Name(id='_portable_runtime', ctx=ast.Load()),
        'SOURCE': ast.Name(id='_portable_part3_source', ctx=ast.Load()),
        'PATIENT_BASE': ast.Name(id='_portable_patient_base', ctx=ast.Load()),
    }
    counts = {name: 0 for name in redirected}
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name) and target.id in redirected:
                statement.value = ast.copy_location(replacements[target.id], statement.value)
                counts[target.id] += 1
    if counts != {name: 1 for name in redirected}:
        raise RuntimeError(f'Runner routing bindings changed: {counts}')
    ast.fix_missing_locations(tree)
    module = types.ModuleType(f'portable_external_{subject.lower()}')
    module.__file__ = str(source)
    module.__dict__.update(_portable_output_root=Path(output_root),
                           _portable_runtime=Path(runtime),
                           _portable_part3_source=PROJECT / 'part3_mfc' / 'part3_model.py',
                           _portable_patient_base=PROJECT / 'patient_results')
    sys.modules[module.__name__] = module
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    exec(compile(tree, str(source), 'exec'), module.__dict__)

    aliases = {path_identity(entry['original_path']): bundled_path(entry['relative_path'])
               for entry in manifest['files']}
    exact_sha = module.sha

    def relocated_sha(path):
        return exact_sha(aliases.get(path_identity(path), Path(path)))

    # Historical absolute paths remain identity labels in saved training
    # contracts. The SHA resolver reads the corresponding relocated bytes.
    module.sha = relocated_sha
    original_setup = module.setup
    expected_inputs = {entry['original_path']: entry['sha256']
                       for entry in manifest['files']
                       if entry['subject'] == subject and entry['role'] in INPUT_ROLES}

    def relocated_setup(requested_subject, seed):
        if requested_subject != subject:
            raise RuntimeError('A loaded runner cannot switch patient')
        result = original_setup(requested_subject, seed)
        if sorted(result['hashes'].values()) != sorted(expected_inputs.values()):
            raise RuntimeError('Relocated inputs differ from frozen training identities')
        result['hashes'] = dict(expected_inputs)
        return result

    module.setup = relocated_setup
    return module


def verify_bundle(args, manifest, runtime):
    count = verify_hashes(manifest)
    subjects = [args.subject] if args.subject else ['HUP065', 'HUP080']
    report = {'status': 'verified_hashes_imports_and_predictive_model_deserialization',
              'files_verified': count,
              'outer_arrays_unpacked': False, 'controller_checkpoints_loaded': False,
              'training_or_evaluation_run': False,
              'bundle_inputs_modified': False,
              'python': sys.version,
              'dependency_versions': {name: importlib.metadata.version(name)
                                      for name in EXPECTED_PACKAGES},
              'subjects': {}}
    for subject in subjects:
        module = load_runner(subject, manifest, runtime, runtime)
        core = module.load_core()
        config = module.paths(subject)
        model = module.joblib.load(config['model'])
        if module.sha(config['model']) != module.MODEL_HASHES[subject]:
            raise RuntimeError('Predictive model hash mismatch')
        report['subjects'][subject] = {
            'model_sha256': module.MODEL_HASHES[subject],
            'model_type': f'{type(model).__module__}.{type(model).__name__}',
            'channels': int(model.adjacency.shape[0]),
            'direct_actuator_count_contract': config['m'],
            'canonical_part3': str(Path(core.__file__).relative_to(BUNDLE)),
            'function_bodies_unchanged': True,
            'reference_sampler': 'uniform_without_replacement_per_critic_update' if subject == 'HUP080' else 'original_frozen_HUP065_sampler',
        }
    imports = {}
    for name, module in list(sys.modules.items()):
        if name == 'mfc_pipeline' or name.startswith('mfc_pipeline.'):
            path = Path(module.__file__).resolve()
            if PROJECT not in path.parents:
                raise RuntimeError(f'Dependency imported outside snapshot: {name}: {path}')
            imports[name] = str(path.relative_to(BUNDLE))
    report['canonical_dependency_imports'] = imports
    # Hash recheck establishes that importing/deserializing did not mutate any
    # frozen source/input, including the still-opaque sealed NPZs.
    verify_hashes(manifest)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def safe_output(args):
    if not args.tag or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_.-]{0,79}', args.tag):
        raise RuntimeError('Provide a new simple tag, starting with a letter')
    if args.tag.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL', *[f'COM{i}' for i in range(1, 10)], *[f'LPT{i}' for i in range(1, 10)]}:
        raise RuntimeError('Reserved Windows output tag')
    root = Path(args.output_root).resolve() if args.output_root else BUNDLE / 'generated'
    out = (root / args.tag).resolve()
    if out.parent != root.resolve() or BUNDLE / 'snapshot' in out.parents:
        raise RuntimeError('Unsafe reproduction output path')
    return out


def import_checkpoint_for_replay(args, out, module):
    if not args.checkpoint_run:
        raise RuntimeError('evaluate requires --checkpoint-run containing a frozen checkpoint and training_summary.json')
    existing = Path(args.checkpoint_run).resolve()
    if existing == out or existing in out.parents or out in existing.parents:
        raise RuntimeError('Replay output must be separate from the frozen source run')
    checkpoint = existing / 'frozen_actor_wgan.pt'
    summary_path = existing / 'training_summary.json'
    if not checkpoint.is_file() or not summary_path.is_file():
        raise RuntimeError('The source run is not frozen and complete')
    summary = json.loads(summary_path.read_text(encoding='utf-8-sig'))
    if summary.get('subject') != args.subject or summary.get('trained_updates') != 1000:
        raise RuntimeError('Frozen run patient or budget mismatch')
    if summary.get('checkpoint_sha256') != module.sha(checkpoint):
        raise RuntimeError('Frozen checkpoint differs from training receipt')
    if out.exists():
        raise RuntimeError('Refusing to overwrite an existing replay tag')
    # Only trusted local/archive PyTorch checkpoints should be passed here.
    payload = module.torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload['training_contract']['seed'] != args.seed:
        raise RuntimeError('Use the frozen run seed for exact bank reproduction')
    if payload['training_contract']['source_code_sha256'] != module.sha(module.__file__):
        raise RuntimeError('Frozen controller belongs to a different runner snapshot')
    out.mkdir(parents=True)
    shutil.copy2(checkpoint, out / checkpoint.name)
    shutil.copy2(summary_path, out / summary_path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('verify-bundle', 'preflight', 'train', 'evaluate'))
    parser.add_argument('--subject', choices=('HUP065', 'HUP080'))
    parser.add_argument('--seed', type=int, default=20261011)
    parser.add_argument('--updates', type=int, choices=(1000,), default=1000)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--tag')
    parser.add_argument('--output-root')
    parser.add_argument('--checkpoint-run')
    args = parser.parse_args()
    if args.mode != 'verify-bundle' and not args.subject:
        parser.error('--subject is required outside verify-bundle')
    manifest = read_manifest()
    verify_hashes(manifest)
    # A normal-inheritance cache directory avoids Windows Temp mode-0700 ACL
    # failures in restricted desktop processes. It sits outside the bundle.
    # No snapshot/source input is written and no caches are needed for replay.
    runtime = BUNDLE.parent / '.portable_runtime' / ('import_' + uuid.uuid4().hex)
    runtime.mkdir(parents=True)
    if args.mode == 'verify-bundle':
        verify_bundle(args, manifest, runtime)
        return
    out = safe_output(args)
    module = load_runner(args.subject, manifest, runtime, out.parent)
    module.torch.set_num_threads(args.threads)
    module.torch.set_num_interop_threads(1)
    if args.mode == 'preflight':
        if out.exists():
            raise RuntimeError('Refusing an existing preflight tag')
        out.mkdir(parents=True)
        module.preflight(args, out)
    elif args.mode == 'train':
        if (out / 'training_history.csv').exists() or (out / 'training_contract.json').exists():
            raise RuntimeError('Training is from scratch only; use a new tag, not a partial run')
        module.train(args, out)
    else:
        import_checkpoint_for_replay(args, out, module)
        module.evaluate(args, out)
    verify_hashes(manifest)


if __name__ == '__main__':
    main()
