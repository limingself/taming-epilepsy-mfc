"""Verify the published run-02 density figures and scores; no private inputs needed."""
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import fmean, median


ROOT = Path(__file__).resolve().parents[2]
REPLAY = Path("patient_extensions/single_window_replay")
# Figure order, contact count, actuators and the two percentage reductions in
# the manuscript. Improvement is strict and computed contact by contact.
CASES = (("S4", "HUP065", 64, 32, (42.4, 61.6)),
         ("S5", "HUP080", 96, 76, (68.6, 88.4)))


def require(condition, description):
    if not condition:
        raise ValueError(description)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_checked(root, expected, registry_path, path_index):
    require(registry_path == expected.as_posix(), f"Unexpected figure registry path: {registry_path}")
    entry = path_index[registry_path]
    data = (root / expected).read_bytes()
    require(len(data) == entry["bytes"], f"Size mismatch: {registry_path}")
    require(hashlib.sha256(data).hexdigest() == entry["sha256"].lower(), f"SHA-256 mismatch: {registry_path}")
    return data


def numbers(rows, field):
    values = [float(row[field]) for row in rows]
    require(all(map(math.isfinite, values)), f"Nonfinite {field}")
    return values


def close(actual, reported, label):
    # Aggregate values are reported to six decimal places.
    require(math.isclose(actual, reported, rel_tol=0, abs_tol=0.500001e-6),
            f"{label}: from CSV {actual}, reported {reported}")


def verify(root=ROOT):
    root = Path(root)
    registry = read_json(root / "paper_figures.json")
    require(registry["manuscript_patient_order"] == ["HUP060", "HUP065", "HUP080"],
            "Unexpected manuscript patient order")
    published = registry["red_manuscript_figures"]
    require([key for key in published if key in ("S4", "S5")] == ["S4", "S5"],
            "Expected figures S4, S5 in manuscript order")
    reports = read_json(root / REPLAY / "reported_results.json")
    entries = read_json(root / "CURRENT_PROJECT_MANIFEST.json")["files"]
    path_index = {entry["path"]: entry for entry in entries}
    require(len(entries) == len(path_index), "Duplicate path in public manifest")

    results = []
    for figure, subject, n_contacts, n_actuators, reductions in CASES:
        record, report = published[figure], reports[subject]
        require(record["patient"] == subject and report["figure"] == figure, f"Unexpected patient for {figure}")
        require(record["contacts"] == n_contacts and record["actuators"] == n_actuators,
                f"Unexpected coverage in registry: {subject}")
        require(record["fit_run"] == report["fit_run"] == "run-01" and
                record["evaluation_run"] == report["evaluation_run"] == "run-02" and record["window_seconds"] == 1,
                f"Unexpected protocol for {figure}")
        directory = REPLAY / "publication" / subject
        stem = f"{subject}_run02_full_all{n_contacts}"
        for ext, magic in (("pdf", b"%PDF-"), ("png", b"\x89PNG\r\n\x1a\n")):
            data = read_checked(root, directory / f"{stem}.{ext}", record[ext], path_index)
            require(data.startswith(magic), f"Invalid {ext} format: {figure}")
        csv_path = directory / "contact_metrics.csv"
        read_checked(root, csv_path, record["metrics"], path_index)
        with (root / csv_path).open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == report["contacts"] == n_contacts, f"Contact count: {subject}")
        labels = [row["contact"].strip() for row in rows]
        require(all(labels) and len(set(labels)) == n_contacts, f"Missing/duplicate contacts: {subject}")
        flags = [row["direct_input"] for row in rows]
        require(set(flags) <= {"True", "False"} and flags.count("True") == report["direct_inputs"]
                == n_actuators, f"Actuator count: {subject}")

        scores = []
        for kind, expected_pct in zip(("time", "occupation"), reductions):
            key = f"{kind}_W1"
            summary = report[f"{key}_to_reference"]
            free = numbers(rows, f"free_{key}")
            controlled = numbers(rows, f"full_{key}")
            observed = numbers(rows, f"observed_{key}_to_ref")
            for role, vector in (("free", free), ("controlled", controlled), ("observed", observed)):
                require(min(vector) >= 0, f"Negative {key} for {subject}")
                close(fmean(vector), summary[role], f"{subject}/{key}/{role}")
            count = sum(c < f for c, f in zip(controlled, free))
            require(count == summary["contacts_improved_vs_free"], f"Improvement count: {subject}/{key}")
            pct = round(100 * (1 - fmean(controlled) / fmean(free)), 1)
            require(pct == expected_pct, f"Reduction: {subject}/{key}: {pct}%")
            scores.append(f"{kind} W1 -{pct:.1f}% ({count}/{n_contacts})")

        forecast = report["forecast_vs_observed"]
        for output, source, aggregate in (("RMSE", "forecast_mean_RMSE", fmean),
                                          ("baseline_RMSE", "persistence_RMSE", fmean),
                                          ("occupation_W1", "forecast_occup_W1", fmean),
                                          ("baseline_occupation_W1", "prior_second_occup_W1", fmean),
                                          ("median_channel_correlation", "forecast_mean_trace_r", median)):
            close(aggregate(numbers(rows, source)), forecast[output], f"{subject}/forecast/{output}")
        results.append(f"{figure} {subject}: " + "; ".join(scores))
    return results


if __name__ == "__main__":
    try:
        print("\n".join(verify()))
    except (OSError, KeyError, ValueError, csv.Error) as err:
        sys.exit(f"Publication verification failed: {err}")
