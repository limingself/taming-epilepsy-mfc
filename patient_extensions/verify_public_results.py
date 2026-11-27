"""Verify manuscript S4/S5 (run-02); use --archive for the older outer analyses."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'part3_mfc'))
from single_window_replay.verify_publication import verify as verify_publication

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh',action='store_true',help='Compatibility flag; current manifests are not silently regenerated')
    parser.add_argument('--archive',action='store_true',help='Also verify the archived multi-window results in a full checkout')
    args=parser.parse_args()
    figures=verify_publication()
    if not args.archive:
        print(json.dumps(dict(status='PASS',figures=figures,training_started=False),indent=2))
        return 0
    from current_controller import verify_current
    manifest=json.loads((ROOT/'PUBLIC_MANIFEST.json').read_text(encoding='utf-8-sig'))
    for entry in manifest['files']:
        path=ROOT/entry['path']
        with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=entry['sha256']:raise RuntimeError('Current patient file SHA mismatch: '+entry['path'])
    reports=[verify_current(subject) for subject in ('HUP065','HUP080')]
    print(json.dumps(dict(status='PASS',figures=figures,archived_patients=reports,current_patient_files_verified=len(manifest['files']),signal_arrays_opened=False,model_loaded=False,training_started=False,outer_evaluated=False),indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
