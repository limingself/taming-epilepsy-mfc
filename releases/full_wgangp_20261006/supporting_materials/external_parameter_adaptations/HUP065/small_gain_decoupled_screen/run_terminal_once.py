"""Guarded once-only terminal evaluation after this six-pair adaptation freezes."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    results=HERE/'results'
    report=json.loads((results/'selection_report.json').read_text(encoding='utf-8'))
    if report.get('adaptation_completed') is not True:
        raise RuntimeError('Wait for all150 adaptation updates to complete and policy to freeze')
    if report.get('adaptation_final_qualifies') is not True:
        raise RuntimeError('No terminal/outer opening for a candidate that fails development eligibility')
    receipt=results/'frozen_dev_selection_receipt.json'
    contract=json.loads(receipt.read_text(encoding='utf-8'))
    if contract.get('eligible_for_terminal_veto') is not True:
        raise RuntimeError('Eligible frozen development receipt required')
    destination=results/'final_evaluation'
    if destination.exists():raise RuntimeError('Terminal evaluation is once-only; output already exists')
    original=HERE.parent/'finalize_hup065.py'
    source=results/'source_snapshot'/'finalize_hup065.py'
    if source.exists():raise RuntimeError('Do not overwrite the final-evaluation source snapshot')
    shutil.copy2(original,source)
    # Snapshot bytes are preserved separately; original companion locates its
    # unchanged optimizer/portable bundle using the validated workspace routes.
    subprocess.run([sys.executable,'-B','-X','utf8',str(original),
        '--selection-receipt',str(receipt),'--output',str(destination),
        '--threads',str(args.threads)],check=True)


if __name__=='__main__':main()
