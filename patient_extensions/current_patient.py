"""Current32/76 external-control commands without implicit signal access."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'part3_mfc'))
from current_controller import verify_current

def main(subject,argv=None):
    parser=argparse.ArgumentParser(description=f'{subject} current Full WGAN-GP; default public verification. Private execution requires an explicitly selected current bundle and a new tag.')
    parser.add_argument('mode',nargs='?',choices=('verify','figures','verify-private','screen-train'),default='verify')
    parser.add_argument('--bundle-root',type=Path)
    parser.add_argument('--new-tag')
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args(argv)
    if args.mode in ('verify','figures'):
        print(json.dumps(verify_current(subject),indent=2));return 0
    directory='HUP065_sparse_control_final_v1' if subject=='HUP065' else 'HUP080_sparse_control_final_v2'
    private=args.bundle_root or ROOT/'patient_results'/directory/'current_full_wgangp'
    if not private.is_dir():parser.error('The complete current private frozen-input bundle is absent. Supply --bundle-root; no historical controller fallback is permitted.')
    if not args.new_tag:parser.error('Private checks/training require an explicit new --new-tag')
    script=ROOT/'patient_extensions'/subject/'reproduce_control.py'
    mode='verify-bundle' if args.mode=='verify-private' else ('screen-train' if subject=='HUP065' else 'preflight')
    # HUP080 screen-evaluate would also open terminal/outer data. It is not
    # exposed as a synchronization/training command by this default entry.
    if args.mode=='screen-train' and subject=='HUP080':parser.error('This executed80 wrapper couples screening to evaluation. Only private preflight is exposed here; a separate explicit scientific evaluation decision is required.')
    command=[sys.executable,'-B','-X','utf8',str(script),mode,'--bundle-root',str(private),'--tag',args.new_tag,'--threads',str(args.threads)]
    return subprocess.run(command,cwd=ROOT,check=False).returncode

if __name__=='__main__':raise SystemExit('Use the existing HUP065/runner.py or HUP080/runner.py subject entry.')
