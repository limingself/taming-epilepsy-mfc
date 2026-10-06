#!/usr/bin/env python3
"""Verify archived source arrays and tables; never train or run an outer evaluator.

Requires Python 3.10+ and NumPy. All inputs resolve relative to this script.
The checksum manifest is immutable for normal verification. --create-manifest
is an explicit packaging operation, not an alternative verification pathway.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SUBJECTS = {"HUP065": (64, 23, 19), "HUP080": (96, 76, 24)}
COMPONENTS = (
    "gate_time_w1_relative_reduction_pass",
    "gate_occupation_w1_relative_reduction_pass",
    "gate_time_w1_absolute_pass",
    "gate_occupation_w1_absolute_pass",
    "gate_mean_error_pass",
    "gate_sd_ratio_pass",
)
COMPONENT_KEYS_080 = (
    "time_relative_reduction_pass", "occupation_relative_reduction_pass",
    "time_absolute_pass", "occupation_absolute_pass",
    "mean_absolute_error_pass", "symmetric_sd_ratio_pass",
)
FIDELITY_KEYS = (
    "mean_time_w1_free_observed", "mean_time_w1_free_reference_plant",
    "mean_time_w1_observed_reference", "mean_occupation_w1_free_observed",
    "mean_occupation_w1_free_reference_plant", "mean_occupation_w1_observed_reference",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def boolean(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    require(str(value).lower() in {"true", "false", "1", "0"}, f"Invalid Boolean: {value}")
    return str(value).lower() in {"true", "1"}


def close(actual, expected, label):
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    require(a.shape == e.shape, f"{label}: shape mismatch {a.shape} != {e.shape}")
    require(np.isfinite(a).all() and np.isfinite(e).all(), f"{label}: nonfinite value")
    error = float(np.max(np.abs(a - e))) if a.size else 0.0
    require(np.allclose(a, e, rtol=1e-10, atol=1e-11), f"{label}: numeric mismatch, max abs error={error}")
    return error


def empirical_w1(lhs, rhs):
    """Exact 1D equal-weight empirical W1, vectorized over remaining axes."""
    lhs, rhs = np.asarray(lhs), np.asarray(rhs)
    require(lhs.shape[1:] == rhs.shape[1:], "W1 comparison axes differ")
    n, m = len(lhs), len(rhs)
    require(n > 0 and m > 0, "W1 input is empty")
    edges = np.unique(np.concatenate((np.arange(n + 1) / n, np.arange(m + 1) / m)))
    weights = np.diff(edges)
    mid = 0.5 * (edges[:-1] + edges[1:])
    li = np.minimum((mid * n).astype(int), n - 1)
    ri = np.minimum((mid * m).astype(int), m - 1)
    diff = np.abs(np.sort(lhs, axis=0)[li] - np.sort(rhs, axis=0)[ri])
    return np.sum(diff * weights.reshape((-1,) + (1,) * (diff.ndim - 1)), axis=0)


def distances(a, b):
    require(a.ndim == b.ndim == 3, "Expected sample x time x channel laws")
    return (
        empirical_w1(a, b).mean(axis=0),
        empirical_w1(a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])),
    )


def create_manifest():
    target = ROOT / "SHA256SUMS.json"
    require(not target.exists(), "Refusing to overwrite an existing checksum manifest")
    excluded = {"SHA256SUMS.json", "verification_report.json"}
    entries = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name not in excluded and "__pycache__" not in path.parts:
            entries.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                            "sha256": sha256(path)})
    value = {"schema": "external-tierA-source-checksums-v1", "files": entries,
             "notes": "Local author verification package; no open redistribution licence is granted."}
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(entries)


def verify_manifest():
    manifest = read_json(ROOT / "SHA256SUMS.json")
    entries = manifest["files"]
    paths = [entry["path"] for entry in entries]
    require(len(paths) == len(set(paths)), "Duplicate checksum-manifest path")
    total_bytes = 0
    for entry in entries:
        p = (ROOT / entry["path"]).resolve()
        require(p.is_relative_to(ROOT), "Manifest attempts a path outside the bundle")
        require(p.is_file(), f"Missing source file: {entry['path']}")
        require(p.stat().st_size == entry["bytes"], f"Size mismatch: {entry['path']}")
        require(sha256(p) == entry["sha256"], f"SHA256 mismatch: {entry['path']}")
        total_bytes += entry["bytes"]
    present = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()
               and p.name not in {"SHA256SUMS.json", "verification_report.json"}
               and "__pycache__" not in p.parts}
    require(present == set(paths), "Unmanifested or omitted bundle file")
    return {"verified_file_count": len(entries), "verified_bytes": total_bytes}


def verify_subject(subject):
    n, d, q = SUBJECTS[subject]
    base = ROOT / subject
    result = {"subject": subject, "checks": []}
    checks = result["checks"]
    c = np.load(base / "canonical_ofrc_input.npz", allow_pickle=False)
    z = np.load(base / "display_context7_bank0.npz", allow_pickle=False)
    f, u, r = z["free_standardized"], z["controlled_standardized"], z["reference_standardized"]
    o, commands, noise = z["observed_standardized"], z["controls"], z["standard_normal"]
    expected = {"free_standardized": (32, 256, n), "controlled_standardized": (32, 256, n),
                "reference_standardized": (40, 256, n), "observed_standardized": (256, n),
                "controls": (32, 256, d), "standard_normal": (32, 256, q)}
    for key, shape in expected.items():
        require(z[key].shape == shape and np.isfinite(z[key]).all(), f"{subject}/{key}: invalid shape or values")
    checks.append("Official display NPZ shapes and finite arrays verified (32 particles; 40 reference paths)")
    for ck, dk in [("free_scaled", "free_standardized"), ("controlled_scaled", "controlled_standardized"),
                   ("reference_scaled", "reference_standardized"), ("observed_scaled", "observed_standardized")]:
        require(np.array_equal(c[ck], z[dk].reshape(c[ck].shape)), f"{subject}: canonical/display identity failed for {ck}")
    for key in ["channels", "direct_mask"]:
        require(np.array_equal(c[key], z[key]), f"{subject}: {key} identity failed")
    require(str(c["subject_id"].item()) == subject, f"{subject}: subject identifier mismatch")
    require(str(z["display_window_id"].item()) == "outer-context-07" and int(z["display_crn_bank"].item()) == 0,
            f"{subject}: display identity mismatch")
    require(np.array_equal(noise[:16], -noise[16:]), f"{subject}: innovations are not 16 antithetic pairs")
    checks.append("Canonical/display four-law arrays, channel labels, masks and antithetic pairs identical")
    network = np.load(base / "part1_network.npz", allow_pickle=False)
    require(np.array_equal(network["target_mask"], c["direct_mask"]), f"{subject}: network mask differs")
    require(np.array_equal(network["channels"], c["channels"]), f"{subject}: network channel order differs")
    require(int(c["direct_mask"].sum()) == d, f"{subject}: direct actuator count differs")
    masks = sorted(read_csv(base / "frozen_mask.csv"), key=lambda x: int(x["channel_index"]))
    require([x["channel"] for x in masks] == c["channels"].tolist(), f"{subject}: CSV channel order differs")
    require(np.array_equal([boolean(x["direct_actuated"]) for x in masks], c["direct_mask"]),
            f"{subject}: CSV mask differs")
    checks.append("Part-I network and frozen mask identity verified")
    binding = read_json(base / "final_candidate_binding.json")
    for key in ["subject_id", "candidate_id", "shared_protocol_sha256", "frozen_evaluator_manifest_sha256",
                "gate_vector_source_sha256", "safety_vector_source_sha256"]:
        require(str(c[key].item()) == str(binding[key]), f"{subject}: candidate binding mismatch for {key}")
    require(binding["display_window_id"] == "outer-context-07" and binding["display_crn_bank"] == 0,
            f"{subject}: binding display mismatch")
    checks.append("Semantic frozen candidate/protocol/evaluator binding verified (canonical file has its own manifest hash)")
    rows = read_csv(base / "all_context_channel_metrics.csv")
    require(len(rows) == 24 * n, f"{subject}: incorrect all-context row count")
    expected_keys = {(i, b, j) for i in range(8) for b in range(3) for j in range(n)}
    keys = [(int(x["context_index"]), int(x["crn_bank"]), int(x["channel_index"])) for x in rows]
    require(set(keys) == expected_keys and len(keys) == len(set(keys)), f"{subject}: missing/duplicate Cartesian rows")
    boundaries = [512, 1573, 2633, 3694, 4754, 5815, 6875, 7936]
    for x in rows:
        require(int(x["boundary_sample"]) == boundaries[int(x["context_index"])], f"{subject}: boundary mismatch")
        require(boolean(x["direct_actuated"]) == bool(c["direct_mask"][int(x["channel_index"])]),
                f"{subject}: row direct-actuation mismatch")
    checks.append(f"Complete unique 8 contexts x 3 banks x {n} channels Cartesian ledger verified")
    fixed_rows = sorted([x for x in rows if int(x["context_index"]) == 7 and int(x["crn_bank"]) == 0],
                        key=lambda x: int(x["channel_index"]))
    tf, of = distances(f, r)
    tc, oc = distances(u, r)
    maximum_error = 0.0
    for values, name in [(tf, "time_w1_free"), (tc, "time_w1_controlled"),
                         (of, "occupation_w1_free"), (oc, "occupation_w1_controlled")]:
        maximum_error = max(maximum_error, close(values, [float(x[name]) for x in fixed_rows], f"{subject}/{name}"))
    mean_error = np.abs(u.mean(axis=(0, 1)) - r.mean(axis=(0, 1)))
    su, sr = u.std(axis=(0, 1)), r.std(axis=(0, 1))
    sd_ratio = np.maximum(su / np.maximum(sr, 1e-12), sr / np.maximum(su, 1e-12))
    maximum_error = max(maximum_error, close(mean_error, [float(x["mean_absolute_error"]) for x in fixed_rows], subject + "/mean error"),
                        close(sd_ratio, [float(x["symmetric_sd_ratio"]) for x in fixed_rows], subject + "/sd ratio"))
    metadata = sorted(read_csv(base / "channel_metadata.csv"), key=lambda x: int(x["channel_index"]))
    to, oo = distances(o[None], r)
    for values, key in [(oo, "w1_observed_to_reference"), (of, "w1_free_to_reference"),
                        (oc, "w1_controlled_to_reference")]:
        maximum_error = max(maximum_error, close(values, [float(x[key]) for x in metadata], subject + "/metadata " + key))
    checks.append("All display-channel W1, pooled-mean error, symmetric SD ratio and figure source metadata recomputed from arrays")
    report = read_json(base / "retrospective_report.json")["metrics"]
    endpoint_means = {}
    for field in ["time_w1_free", "time_w1_controlled", "occupation_w1_free", "occupation_w1_controlled"]:
        endpoint_means["mean_" + field] = float(np.mean([float(x[field]) for x in rows]))
        maximum_error = max(maximum_error, close(endpoint_means["mean_" + field], report["mean_" + field], subject + "/aggregate " + field))
    component_keys = COMPONENTS if subject == "HUP065" else COMPONENT_KEYS_080
    aggregate_components = {name: np.ones(n, dtype=bool) for name in COMPONENTS}
    all_gate_b, all_improved = np.ones(n, dtype=bool), np.ones(n, dtype=bool)
    for row in rows:
        j = int(row["channel_index"])
        a, b, e, g = [float(row[x]) for x in ["time_w1_free", "time_w1_controlled", "occupation_w1_free", "occupation_w1_controlled"]]
        numeric_components = ((a - b) / max(a, 1e-12) >= .10, (e - g) / max(e, 1e-12) >= .10,
                              b <= .35, g <= .25, float(row["mean_absolute_error"]) <= .10,
                              float(row["symmetric_sd_ratio"]) <= 2.0)
        for canonical, key, actual in zip(COMPONENTS, component_keys, numeric_components):
            require(actual == boolean(row[key]), subject + "/stored contact criterion differs from numeric rule")
            aggregate_components[canonical][j] &= actual
        row_pass = all(numeric_components)
        require(row_pass == boolean(row["gate_b_pass"]), subject + "/stored contact conjunction differs")
        all_gate_b[j] &= row_pass
        all_improved[j] &= (b < a and g < e)
    for name, values in aggregate_components.items():
        require(np.array_equal(values, c[name]), subject + "/all-context component mask differs")
    require(np.array_equal(all_gate_b, c["full_gate_b_pass"]), subject + "/all-context Gate-B mask differs")
    require(int(all_gate_b.sum()) == int(report["gate_b_pass_count"]), subject + "/Gate-B count differs")
    improved_key = "both_improved_all_context_count" if subject == "HUP065" else "both_improved_all_context_bank_count"
    require(int(all_improved.sum()) == int(report[improved_key]), subject + "/improved channel count differs")
    if subject == "HUP065":
        aggregate_source = read_json(base / "aggregate_gate_vectors.json")
        require(np.array_equal(aggregate_source["full_gate_b_pass"], all_gate_b), subject + "/aggregate JSON mask differs")
    else:
        aggregate_source = sorted(read_csv(base / "aggregate_gate_vectors.csv"), key=lambda x: int(x["channel_index"]))
        require(np.array_equal([boolean(x["full_gate_b_pass"]) for x in aggregate_source], all_gate_b), subject + "/aggregate CSV mask differs")
    checks.append("All 24-condition endpoint means, contact thresholds, strict cross-condition conjunctions and success counts reconstructed from CSV")
    safety = read_json(base / "aggregate_safety.json")
    if subject == "HUP065":
        by_pair = {}
        for row in rows:
            pair = (int(row["context_index"]), int(row["crn_bank"]))
            operands = tuple(float(row[x]) for x in ["trajectory_total_energy", "trajectory_maximum_per_actuator_rms",
                                                    "trajectory_control_peak", "trajectory_saturation_fraction"])
            if pair in by_pair:
                close(operands, by_pair[pair], subject + "/repeated trajectory safety operands")
            by_pair[pair] = operands
        safety_rows = {pair: dict(zip(["total_energy", "maximum_per_actuator_rms", "control_peak", "saturation_fraction"], values))
                       for pair, values in by_pair.items()}
        require(boolean(safety["finite_and_energy_peak_saturation_all_window_banks"]), subject + "/aggregate safety false")
    else:
        source = read_csv(base / "all_context_safety_metrics.csv")
        safety_rows = {(int(x["context_index"]), int(x["crn_bank"])): x for x in source}
        require(len(source) == len(safety_rows) == 24, subject + "/missing/duplicate safety rows")
        require(boolean(safety["all_predeclared_windows_x_banks_gate_c_pass"]), subject + "/aggregate safety false")
    require(set(safety_rows) == {(i, b) for i in range(8) for b in range(3)}, subject + "/safety ledger incomplete")
    energy_cap = 3.7908 if subject == "HUP065" else 5.6862
    for row in safety_rows.values():
        require(float(row["total_energy"]) <= energy_cap + 1e-9 and float(row["maximum_per_actuator_rms"]) <= .405 + 1e-9
                and float(row["control_peak"]) <= 1.8 + 1e-9 and float(row["saturation_fraction"]) < .01,
                subject + "/trajectory feasibility failure")
    energy = float(np.square(commands).sum(axis=(1, 2)).mean() / 256)
    rms = np.sqrt(np.square(commands).mean(axis=(0, 1)))
    peak, saturation = float(np.abs(commands).max()), float((np.abs(commands) >= .95 * 1.8).mean())
    srow = safety_rows[(7, 0)]
    for value, key in [(energy, "total_energy"), (rms.max(), "maximum_per_actuator_rms"),
                       (peak, "control_peak"), (saturation, "saturation_fraction")]:
        maximum_error = max(maximum_error, close(value, float(srow[key]), subject + "/display safety " + key))
    worst = {key: max(float(x[key]) for x in safety_rows.values()) for key in
             ["total_energy", "maximum_per_actuator_rms", "control_peak", "saturation_fraction"]}
    if subject == "HUP080":
        for key, value in worst.items():
            maximum_error = max(maximum_error, close(value, safety["worst_case_" + key], subject + "/worst safety " + key))
    checks.append("24-condition safety operands and display commands independently checked; no control regeneration")
    tfo, ofo = distances(f, o[None])
    fixed_fidelity = dict(zip(FIDELITY_KEYS, map(float, [tfo.mean(), tf.mean(), to.mean(), ofo.mean(), of.mean(), oo.mean()])))
    if subject == "HUP080":
        for key in FIDELITY_KEYS:
            maximum_error = max(maximum_error, close(fixed_fidelity[key], float(srow[key]), subject + "/display fidelity " + key))
            mean_from_rows = float(np.mean([float(x[key]) for x in safety_rows.values()]))
            maximum_error = max(maximum_error, close(mean_from_rows, report[key], subject + "/aggregate fidelity " + key))
        checks.append("Display six fidelity distances recomputed from arrays; 24-condition fidelity mean independently aggregated from archived safety CSV")
    else:
        checks.append("Display six fidelity distances recomputed from arrays; 24-condition fidelity means are available only as original report values")
    fv = [float(report[key]) for key in FIDELITY_KEYS]
    comparisons = [fv[0] < fv[1], fv[0] < fv[2], fv[3] < fv[4], fv[3] < fv[5]]
    reported_diagnostic = report["plant_fidelity_diagnostic_pass"] if subject == "HUP065" else report["plant_four_inequality_diagnostic_pass"]
    require(all(comparisons) == boolean(reported_diagnostic), subject + "/fidelity report-only Boolean differs")
    result.update({"channels": n, "direct_actuators": d, "particles": 32, "antithetic_pairs_per_bank": 16,
                   "contexts": 8, "banks": 3, "CSV_rows": len(rows), "all_context_endpoint_means_from_CSV": endpoint_means,
                   "all_context_GateB_count": int(all_gate_b.sum()), "all_context_both_improved_count": int(all_improved.sum()),
                   "all_context_GateB_count_without_dedicated_outputs": int((all_gate_b & ~c["direct_mask"]).sum()),
                   "display_fidelity_distances_recomputed_from_arrays": fixed_fidelity,
                   "all_context_fidelity_distances_archived": dict(zip(FIDELITY_KEYS, fv)),
                   "fidelity_inequalities_FO_lt_FR_time_then_FO_lt_OR_time_then_occupation": comparisons,
                   "worst_safety_operands_from_CSV": worst, "maximum_absolute_comparison_error": maximum_error,
                   "full_24_condition_trajectory_replay_performed": False,
                   "all_context_fidelity_reaggregation_from_CSV_available": subject == "HUP080",
                   "archived_access_route_author_confirmation_still_required": True})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create-manifest", action="store_true", help="Packaging only: create a new manifest, refusing overwrite")
    parser.add_argument("--output", type=Path, help="Optional JSON verification output (no output file by default)")
    args = parser.parse_args()
    if args.create_manifest:
        print(json.dumps({"manifest_created_file_count": create_manifest()}))
        return 0
    report = {"schema": "external-tierA-verified-sources-v1", "status": "PASS", "checksums": verify_manifest(),
              "software": {"python": sys.version.split()[0], "numpy": np.__version__},
              "scope": "Saved-array and source-table verification only; no model fitting, controller changes, outer replay or public upload",
              "patients": [verify_subject(subject) for subject in SUBJECTS]}
    output = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        destination = args.output.resolve()
        require(destination.is_relative_to(ROOT), "Verification output must stay within the bundle")
        require(destination.name == "verification_report.json", "Use the reserved report filename verification_report.json")
        destination.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
