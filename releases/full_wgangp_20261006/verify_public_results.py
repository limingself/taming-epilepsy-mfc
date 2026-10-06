"""Signal-free, standard-library-only verification of the additive public release.

No scientific module imports, weights/arrays, training, private path access,
authorisation receipts, or seizure evaluation are used by this verifier.
"""
from __future__ import annotations
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
if os.name == "nt" and not str(ROOT).startswith("\\\\?\\"):
    ROOT = Path("\\\\?\\" + str(ROOT))
CANONICAL = "57da573dbe300ad6bd585fbd69e46102d9f315dba8862e63c7d3e8320b6943e1"
FORBIDDEN_EXTENSIONS = {".pt", ".joblib", ".npy", ".npz", ".edf", ".zip", ".pkl", ".pickle", ".gz"}
SECRET_PATTERNS = (
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{32,}\b"),
)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def table(relative):
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def true(value):
    return str(value).strip().lower() in {"true", "1"}


def checked(condition, message, checks):
    if not condition:
        raise ValueError(message)
    checks.append(message)


def close(actual, expected):
    return math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12)


def main() -> int:
    checks = []
    manifest = load_json(ROOT / "public_manifest.json")
    declared = {entry["path"]: entry for entry in manifest["files"]}
    actual_files = {path.relative_to(ROOT).as_posix(): path for path in ROOT.rglob("*") if path.is_file()
                    and "__pycache__" not in path.parts and path.name != "public_manifest.json"}
    checked(set(declared) == set(actual_files), "All public files are listed; no missing or unlisted payload", checks)
    python_files = 0
    for relative, entry in declared.items():
        path = actual_files[relative]
        checked(path.suffix.lower() not in FORBIDDEN_EXTENSIONS, f"No serialized/raw/signal payload: {relative}", checks)
        checked(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], f"Public byte SHA: {relative}", checks)
        checked(not re.search(r"(training_history|critic_warmup|authorization_receipt|access_receipt|one_time_receipt)", relative, re.I),
                f"No training/access receipt file: {relative}", checks)
        if path.suffix.lower() in {".py", ".ps1", ".mjs", ".js", ".json", ".yaml", ".yml", ".toml", ".md", ".txt", ".csv"}:
            text = path.read_text(encoding="utf-8-sig")
            checked(not any(pattern.search(text) for pattern in SECRET_PATTERNS), f"Credential signature scan: {relative}", checks)
            if path.suffix.lower() == ".py":
                ast.parse(text, filename=relative)
                python_files += 1
        external = relative.startswith("exploratory/") or relative.startswith("supporting_materials/external_")
        checked(not (external and path.suffix.lower() == ".csv" and re.search(r"density|densities", path.name, re.I)),
                f"No external long-density table: {relative}", checks)

    provenance = load_json(ROOT / "source_provenance.json")
    for entry in provenance:
        checked(entry["path"] in declared and declared[entry["path"]]["sha256"] == entry["sha256"],
                f"Frozen archive-to-public byte binding: {entry['path']}", checks)
        if entry["path"].endswith("/part3_mfc/part3_model.py"):
            checked(entry["sha256"] == CANONICAL, f"Shared canonical Part-III source: {entry['path']}", checks)

    current = load_json(ROOT / "results/current_results.json")
    checked(current["canonical_part3_sha256"] == CANONICAL, "Current controller framework/source SHA", checks)
    checked(current["raw_or_serialized_inputs_public"] is False, "Public/private frozen-input boundary disclosed", checks)
    expected_nodes = {"HUP060": (36, 13), "HUP065": (64, 32), "HUP080": (96, 76)}
    case_reports = {}
    for subject, case in current["case_results"].items():
        checked((case["channels"], case["direct_nodes"]) == expected_nodes[subject], f"{subject} configured channel/direct count", checks)
        checked(bool(re.fullmatch(r"[0-9a-f]{64}", case["checkpoint_sha256"])), f"{subject} full frozen controller SHA metadata", checks)
        checked(bool(re.fullmatch(r"[0-9a-f]{64}", case["surrogate_sha256"])), f"{subject} frozen surrogate SHA metadata", checks)
        rows = table(case["metrics_csv"])
        checked(len(rows) == case["expected_rows"], f"{subject} all channel-condition rows retained", checks)
        channels = {row["channel"] for row in rows}
        direct = {row["channel"] for row in rows if true(row[case["columns"]["direct"]])}
        checked(len(channels) == case["channels"] and len(direct) == case["direct_nodes"], f"{subject} mask count from published table", checks)
        for field, column in (("time_w1_free", "time_w1_free"), ("occupation_w1_free", "occupation_w1_free"),
                              ("time_w1_controlled", case["columns"]["time_controlled"]),
                              ("occupation_w1_controlled", case["columns"]["occupation_controlled"])):
            mean = math.fsum(float(row[column]) for row in rows) / len(rows)
            checked(close(mean, case[field]), f"{subject} unrounded mean {field}", checks)
        for endpoint in ("time", "occupation"):
            reduction = 100 * (1 - case[f"{endpoint}_w1_controlled"] / case[f"{endpoint}_w1_free"])
            checked(close(reduction, case[f"{endpoint}_reduction_percent"]), f"{subject} ratio-of-means reduction {endpoint}", checks)
        if subject != "HUP060":
            conditions = {(row["context_index"], row["crn_bank"]) for row in rows}
            checked(len(conditions) == 24, f"{subject} 8 contexts x 3 CRN banks, not patients", checks)
            checked(all(sum(row["channel"] == channel for row in rows) == 24 for channel in channels), f"{subject} every channel has all 24 rows", checks)
            full_b = sum(all(true(row["full_gate_b_pass"]) for row in rows if row["channel"] == channel) for channel in channels)
            checked(full_b == case["full_gate_b_all_conditions"], f"{subject} all-condition Gate-B count", checks)
            checked(case["status"] == "post_hoc_amended_reanalysis", f"{subject} amended-analysis role retained", checks)
        case_reports[subject] = {"channels": len(channels), "direct": len(direct), "rows": len(rows),
                                 "time_reduction_percent": case["time_reduction_percent"],
                                 "occupation_reduction_percent": case["occupation_reduction_percent"]}

    trajectory80 = table("supporting_materials/external_parameter_adaptations/HUP080/selected/outer_posthoc_amendment/trajectory_safety_metrics.csv")
    old_pass = sum(true(row["gate_c"]) for row in trajectory80)
    revised_pass = sum(true(row["finite"]) and float(row["maximum_per_actuator_rms"]) <= .45
                       and float(row["total_energy"]) <= 5.6862 and float(row["control_peak"]) <= 1.8
                       and float(row["saturation_fraction"]) <= .01 for row in trajectory80)
    checked(len(trajectory80) == 24 and old_pass == 0 and revised_pass == 24,
            "HUP080 original budget failures retained separately from revised 0.45 cap", checks)

    cold_path = "exploratory/HUP065_cold1000_not_adopted/runs/cold_top32_seed20261011_u1000_occ6_worst8/FINAL_RESULTS.json"
    cold = load_json(ROOT / cold_path)
    checked(cold["trained_actor_updates"] == 1000 and cold["selected_actor_update"] == 900,
            "Cold32 exploratory update1000/selected900 retained", checks)
    checked(cold["not_automatically_adopted"] is True and cold["dual_endpoint_terminal_eligibility"] is False,
            "Cold32 negative result is not adopted", checks)
    checked(cold["terminal_status"]["ctx6_opened"] is False and cold["terminal_status"]["outer_arrays_opened"] is False,
            "Cold32 failed development gate stopped before ctx6/outer", checks)

    mapping = load_json(ROOT / "code_to_figure.json")
    checked(len(mapping["current_figures"]) == 16, "Exactly 16 adopted figure assets mapped", checks)
    for figure in mapping["current_figures"]:
        checked(figure["path"] in declared and figure["sha256"] == declared[figure["path"]]["sha256"],
                f"Current compiled figure source SHA: {figure['path']}", checks)
        for source in figure["scientific_sources"]:
            checked(source in declared, f"Figure code source exists: {source}", checks)

    print(json.dumps({"status": "pass", "scope": "signal-free public bytes, metrics, source lineage, syntax, exclusions",
                      "files_verified": len(declared), "source_bindings_verified": len(provenance),
                      "python_files_syntax_checked": python_files, "check_count": len(checks),
                      "cases": case_reports, "current_figures": 16,
                      "private_bundle_opened": False, "serialized_model_loaded": False,
                      "training_started": False, "terminal_or_outer_evaluation_run": False}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, SyntaxError) as error:
        print(json.dumps({"status": "fail", "error": str(error)}, indent=2), file=sys.stderr)
        raise SystemExit(1)
