"""Build HUP080 Table S17 from contact-averaged, per-window-bank development scores."""

import argparse
import csv
import itertools
import math
import sys
from pathlib import Path
from statistics import fmean


HERE = Path(__file__).resolve().parent
PUBLIC_BANKS = HERE / "results" / "table_s17_window_banks.csv"
ORIGINS = {"Original": (1550, 3250, 5150, 7250), "Comparison": (1280, 4400, 5504, 6600),
           "Additional": (2340, 6076, 7680)}
KEYS = ("group", "fold", "start", "bank")
DISTANCES = ("time_free", "time_controlled", "occupation_free", "occupation_controlled")
LIMITS = ("peak", "max_rms", "energy", "sat")
FIELDS = (*KEYS, *DISTANCES, *LIMITS)
SUMMARY = ("group", "windows", *DISTANCES, "passes", "evaluations")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def within_limits(row):
    return (float(row["peak"]) <= 1.8 + 1e-12 and float(row["max_rms"]) <= .45 and
            float(row["energy"]) <= 5.6862 and float(row["sat"]) < .01)


def extract(root):
    """Read the original six scalar evaluation CSVs; never load any signal array."""
    rows = []
    for f in (1, 2, 3):
        for v in read_csv(root / "engine" / f"fold{f}" / "conditions.csv"):
            g = {"legacy": "Original", "fresh": "Comparison"}[v["stage"]]
            if int(v["fold"]) != f or within_limits(v) != (v["budget_ok"].lower() == "true"):
                raise ValueError(f"Fold or recorded input check disagrees in conditions for fold {f}.")
            rows.append(dict(group=g, fold=f, start=v["stop"], bank=v["bank"],
                             **{key: v[key] for key in (*DISTANCES, *LIMITS)}))
        arms = {}
        for v in read_csv(root / "additional" / f"fold{f}.csv"):
            key = (v["stop"], v["bank"])
            d = arms.setdefault(key, {})
            if int(v["fold"]) != f or v["arm"] in d:
                raise ValueError(f"Fold mismatch or repeated additional arm: {f}, {key}.")
            d[v["arm"]] = v
        for (start, bank), d in arms.items():
            free, refined = d["free"], d["refined"]
            if within_limits(refined) != (refined["budget_ok"].lower() == "true"):
                raise ValueError(f"Recorded budget mismatch: {f}, {start}, {bank}.")
            rows.append(dict(group="Additional", fold=f, start=start, bank=bank, time_free=free["time_W1"],
                             time_controlled=refined["time_W1"], occupation_free=free["occup_W1"],
                             occupation_controlled=refined["occup_W1"], **{k: refined[k] for k in LIMITS}))
    return rows


def summarize(rows):
    indexed = {}
    expected = {(g, f, start, b) for g, starts in ORIGINS.items()
                for f, start, b in itertools.product((1, 2, 3), starts, (1, 2))}
    for row in rows:
        key = (row["group"], int(row["fold"]), int(row["start"]), int(row["bank"]))
        if key in indexed or key not in expected:
            raise ValueError(f"Unexpected or repeated window-bank key: {key}.")
        if not all(math.isfinite(float(row[c])) and float(row[c]) >= 0 for c in (*DISTANCES, *LIMITS)):
            raise ValueError(f"Non-finite or negative metric: {key}.")
        if float(row["sat"]) > 1:
            raise ValueError(f"Invalid saturation fraction: {key}.")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError(f"Missing window-bank keys: {sorted(expected - set(indexed))}.")
    results = []
    for g, origins in ORIGINS.items():
        windows = [[indexed[g, f, start, b] for b in (1, 2)]
                   for f, start in itertools.product((1, 2, 3), origins)]
        # Average banks within windows, then weight the distinct windows equally.
        dist = {c: f"{fmean(fmean(float(v[c]) for v in window) for window in windows):.3f}" for c in DISTANCES}
        results.append(dict(group=g, windows=len(windows), **dist,
                            passes=sum(within_limits(v) for window in windows for v in window),
                            evaluations=sum(map(len, windows))))
    return results


def emit(handle, keys, rows):
    writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def save_csv(path, keys, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        emit(handle, keys, rows)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    source = cli.add_mutually_exclusive_group()
    source.add_argument("--window-banks", type=Path, default=PUBLIC_BANKS, help="Portable, signal-free scalar CSV")
    source.add_argument("--evaluation-root", type=Path, help="Original directory with engine/fold*/conditions.csv and additional/fold*.csv")
    cli.add_argument("--export-window-banks", type=Path, help="Write only the permitted scalar columns from the six input CSVs")
    cli.add_argument("--output", type=Path, help="Write the three-row table as CSV (default: stdout)")
    args = cli.parse_args()
    if args.export_window_banks and not args.evaluation_root:
        cli.error("--export-window-banks requires --evaluation-root")
    if args.output and args.output.resolve() == args.window_banks.resolve():
        cli.error("--output cannot replace the per-bank source CSV.")
    if args.output and args.export_window_banks and args.output.resolve() == args.export_window_banks.resolve():
        cli.error("Supply different outputs for the input rows and the manuscript table.")
    rows = extract(args.evaluation_root) if args.evaluation_root else read_csv(args.window_banks)
    results = summarize(rows)
    if args.export_window_banks:
        rows.sort(key=lambda r: (list(ORIGINS).index(r["group"]), int(r["fold"]), int(r["start"]), int(r["bank"])))
        save_csv(args.export_window_banks, FIELDS, rows)
    if args.output:
        save_csv(args.output, SUMMARY, results)
    else:
        emit(sys.stdout, SUMMARY, results)


if __name__ == "__main__":
    main()
