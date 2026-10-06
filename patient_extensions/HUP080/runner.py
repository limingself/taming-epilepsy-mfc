#!/usr/bin/env python
from __future__ import annotations

import argparse
import hmac
import json
import os
from pathlib import Path
import sys

# This must precede every import that can reach the D canonical tree.
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from h080.contracts import (  # noqa: E402
    assert_bytecode_guard, configure_runtime, load_config, phase_specs, sha256_file,
    verify_external_locks,
)
from h080.staging import PhaseStore  # noqa: E402


SCIENCE_ENV = "HUP080_FRESH_SPARSE_SCIENCE_AUTHORIZATION"
EXPECTED_SCIENCE_ENV = "HUP080_EXPECTED_SCIENCE_AUTHORIZATION"


def _authorized() -> None:
    supplied = os.environ.get(SCIENCE_ENV, "")
    expected = os.environ.get(EXPECTED_SCIENCE_ENV, "")
    if not supplied or not expected or not hmac.compare_digest(supplied, expected):
        raise PermissionError(
            f"science execution requires matching non-empty values in "
            f"{SCIENCE_ENV} and {EXPECTED_SCIENCE_ENV}; static self-test/status do not"
        )


def _store(config: dict) -> PhaseStore:
    return PhaseStore(
        Path(config["science_root"]),
        sha256_file(Path(config["project_root"]) / "config.json"),
    )


def historical_main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed, resumable HUP080 exploratory partial-actuation extension"
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test", help="static/synthetic only; no D drive or patient data")
    sub.add_parser("status", help="validate and show atomically published phases")
    one = sub.add_parser("run-phase", help="execute one real science phase")
    one.add_argument("phase", choices=[item.name for item in phase_specs(load_config())])
    one.add_argument("--run04-go-file", type=Path)
    through = sub.add_parser("run-through", help="execute/resume phases through a declared stop")
    through.add_argument("--through", required=True, choices=[item.name for item in phase_specs(load_config())])
    through.add_argument("--run04-go-file", type=Path)
    args = parser.parse_args()

    config = load_config()
    configure_runtime(config)
    assert_bytecode_guard()
    if args.command == "self-test":
        from selftest import main as selftest_main

        return int(selftest_main())
    store = _store(config)
    if args.command == "status":
        from h080.pipeline import status

        print(json.dumps(status(config, store), ensure_ascii=False, indent=2))
        return 0
    _authorized()
    # Revalidate every source and this formal implementation on every science
    # invocation, including resumed phases; the raw archive itself is never
    # hashed or opened here.
    verify_external_locks(config, include_raw_zip=False)
    store.initialize_root()
    from h080.pipeline import run_phase

    if args.command == "run-phase":
        run_phase(config, store, args.phase, run04_go_file=args.run04_go_file)
        return 0
    for spec in phase_specs(config):
        run_phase(config, store, spec.name, run04_go_file=args.run04_go_file)
        if spec.name == args.through:
            break
    return 0



def main(argv=None):
    """Current HUP080 Full WGAN-GP main with preserved historical input-preparation helpers.

The existing helper definitions below remain for import compatibility; only
main dispatch is current. Historical full-pipeline/control defaults are not
the current CLI.

Current Full WGAN-GP command; original preparation helpers stay available."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from current_patient import main as current_main
    return current_main('HUP080', argv)


if __name__ == "__main__":
    raise SystemExit(main())

