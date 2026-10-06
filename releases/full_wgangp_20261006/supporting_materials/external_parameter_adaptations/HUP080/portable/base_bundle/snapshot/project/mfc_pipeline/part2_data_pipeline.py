#!/usr/bin/env python
"""Run the ictal-only, nested-validated V2 multihorizon prediction study.

The configured final seizure is never used for model or hyperparameter
selection.  HUP064 is handled as a prespecified within-record temporal
generalization case and is never counted as cross-seizure evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import logging
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_PROPOSAL = PROJECT_ROOT / "artifacts" / "v2" / "qc" / "patient_split_proposal.csv"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mfc_pipeline.data import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    BIDSZipDataset,
    EEGSegmentLoader,
)
from mfc_pipeline.metrics import (  # noqa: E402
    fixed_random_projections,
    normalized_rmse,
    pearson_correlation,
    prediction_metrics,
    sliced_wasserstein_distance,
)
from mfc_pipeline.network import mst_proportional_graph, windowed_plv_median  # noqa: E402
from mfc_pipeline.rc_v2 import (  # noqa: E402
    RC_V2_SCHEMA_VERSION,
    RCV2Config,
    RCV2Forecaster,
    forecast_positions,
    temporal_block_split,
)
from mfc_pipeline.statistics import (  # noqa: E402
    bootstrap_confidence_interval,
    exact_paired_sign_flip,
)


LOGGER = logging.getLogger("prediction_v2")
SCHEMA_VERSION = "2.1.0"
NETWORK_DENSITY = 0.10
PLV_WINDOW_S = 2.0
PLV_OVERLAP = 0.50
VALIDATION_WINDOWS = 4
INDEPENDENT_MODEL_LABELS = (
    "persistence",
    "ridge_var_recursive",
    "delay_linear_direct",
    "esn_plain_recursive",
    "esn_topology_recursive",
    "esn_delay_recursive",
    "esn_delay_direct",
    "rc_topology_delay_recursive",
    "rc_topology_delay_direct",
)
MATCHED_TOPOLOGY_ABLATION_LABEL = "esn_delay_direct_matched_to_primary"
MODEL_LABELS = (*INDEPENDENT_MODEL_LABELS, MATCHED_TOPOLOGY_ABLATION_LABEL)


def safe_prediction_metrics(truth: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    """Evaluate a path, including the single-sample horizon edge case.

    Channel-wise temporal scores are undefined for a one-sample path.  The
    pooled NRMSE/correlation remain well-defined across the channel vector and
    are retained; the channel-balanced NRMSE is explicitly recorded as NaN.
    """

    if len(truth) >= 2:
        return prediction_metrics(truth, estimate)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(estimate - truth)))),
        "nrmse": float(normalized_rmse(truth, estimate)),
        "correlation": float(pearson_correlation(truth, estimate)),
        "mean_channel_nrmse": float("nan"),
        "mean_channel_correlation": float("nan"),
    }


def safe_sliced_wasserstein(
    first: np.ndarray,
    second: np.ndarray,
    projections: np.ndarray,
) -> float:
    """Sliced W1 including the mathematically valid one-point empirical case."""

    if len(first) >= 2 and len(second) >= 2:
        return float(
            sliced_wasserstein_distance(
                first, second, projections=projections, time_axis=0
            )
        )
    directions = np.asarray(projections, dtype=float)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    projected_first = np.asarray(first, dtype=float) @ directions.T
    projected_second = np.asarray(second, dtype=float) @ directions.T
    return float(np.mean(np.abs(projected_first[0] - projected_second[0])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config_v2.yaml")
    parser.add_argument(
        "--split-proposal",
        type=Path,
        default=DEFAULT_SPLIT_PROPOSAL,
        help="mandatory event-QC split proposal; its run roles are enforced and hashed",
    )
    parser.add_argument("--subjects", nargs="+", help="optional configured subject subset")
    parser.add_argument("--seed", type=int, default=11, help="technical seed; compact main run uses 11")
    parser.add_argument("--quick", action="store_true", help="one reservoir profile and two windows")
    parser.add_argument("--force", action="store_true", help="replace selected subject outputs")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.resolve().open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _semicolon_runs(value: Any) -> list[str]:
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in text.split(";") if item.strip()]


def _proposal_json(row: pd.Series, field: str, expected_type: type) -> Any:
    try:
        value = json.loads(str(row[field]))
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in split proposal field {field}") from error
    if not isinstance(value, expected_type):
        raise ValueError(
            f"split proposal field {field} must decode to {expected_type.__name__}"
        )
    return value


def _validate_proposal_window(
    subject: str,
    window: dict[str, Any],
    eligible_runs: list[str],
) -> tuple[str, float, float]:
    required = {"run", "start_offset_from_onset_s", "stop_offset_from_onset_s"}
    missing = sorted(required - set(window))
    if missing:
        raise ValueError(f"{subject}: split window lacks fields {missing}")
    run = str(window["run"])
    start = float(window["start_offset_from_onset_s"])
    stop = float(window["stop_offset_from_onset_s"])
    if run not in eligible_runs or not (0.0 <= start < stop):
        raise ValueError(f"{subject}: invalid split window {run}[{start},{stop})")
    if "duration_s" in window and not np.isclose(float(window["duration_s"]), stop - start):
        raise ValueError(f"{subject}: split-window duration does not match its offsets")
    return run, start, stop


def load_split_proposal(
    path: Path,
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Load and strictly validate the event-QC run split contract."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"mandatory QC split proposal is absent: {resolved}")
    frame = pd.read_csv(resolved, dtype=str, keep_default_na=False)
    required = {
        "schema_version",
        "subject",
        "split_status",
        "split_scheme",
        "all_ictal_runs",
        "eligible_ictal_runs",
        "excluded_event_qc_runs",
        "ictal_train_runs",
        "ictal_test_run",
        "inner_validation_scheme",
        "ictal_inner_validation_folds_json",
        "ictal_final_refit_windows_json",
        "ictal_test_window_json",
        "inner_validation_embargo_s",
        "final_test_embargo_s",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"QC split proposal lacks required columns: {missing}")
    if frame["subject"].duplicated().any():
        duplicated = frame.loc[frame["subject"].duplicated(False), "subject"].tolist()
        raise ValueError(f"QC split proposal has duplicate subjects: {duplicated}")

    configured = [str(value) for value in config["data"]["subjects"]]
    observed = set(frame["subject"].astype(str))
    absent = sorted(set(configured) - observed)
    if absent:
        raise ValueError(f"QC split proposal is missing configured subjects: {absent}")

    proposals: dict[str, dict[str, Any]] = {}
    for subject in configured:
        row = frame.loc[frame["subject"].astype(str) == subject].iloc[0]
        if int(row["schema_version"]) != 2:
            raise ValueError(f"{subject}: V2 runner requires split schema_version=2")
        if str(row["split_status"]).strip().lower() != "ready":
            raise ValueError(f"{subject}: QC split status is not ready")
        all_runs = _semicolon_runs(row["all_ictal_runs"])
        eligible = _semicolon_runs(row["eligible_ictal_runs"])
        excluded = _semicolon_runs(row["excluded_event_qc_runs"])
        train_runs = _semicolon_runs(row["ictal_train_runs"])
        test_runs = _semicolon_runs(row["ictal_test_run"])
        scheme = str(row["split_scheme"]).strip()
        inner_scheme = str(row["inner_validation_scheme"]).strip()
        folds = _proposal_json(row, "ictal_inner_validation_folds_json", list)
        final_refit_windows = _proposal_json(
            row, "ictal_final_refit_windows_json", list
        )
        test_window = _proposal_json(row, "ictal_test_window_json", dict)
        if len(test_runs) != 1:
            raise ValueError(f"{subject}: proposal must define exactly one ictal test run")
        test_run = test_runs[0]
        if not all_runs or not eligible or not train_runs:
            raise ValueError(f"{subject}: proposal contains an empty required run set")
        if len(set(all_runs)) != len(all_runs) or len(set(eligible)) != len(eligible):
            raise ValueError(f"{subject}: proposal run lists contain duplicates")
        if set(eligible) & set(excluded):
            raise ValueError(f"{subject}: eligible and event-QC-excluded runs overlap")
        if set(all_runs) != set(eligible) | set(excluded):
            raise ValueError(f"{subject}: all runs do not equal eligible plus excluded runs")
        if set(train_runs) | {test_run} != set(eligible):
            raise ValueError(f"{subject}: train/test roles do not exactly cover eligible runs")
        configured_test = str(config["data"]["analysis_test_runs"][subject])
        if test_run != configured_test:
            raise ValueError(
                f"{subject}: config test run {configured_test} differs from QC proposal {test_run}"
            )
        if scheme == "held_out_seizure":
            if test_run in train_runs or len(eligible) < 2:
                raise ValueError(f"{subject}: held-out-seizure roles are not disjoint")
        elif scheme == "blocked_single_seizure":
            if subject != "HUP064" or eligible != [test_run] or train_runs != [test_run]:
                raise ValueError(f"{subject}: invalid single-seizure blocked role assignment")
        else:
            raise ValueError(f"{subject}: unsupported split scheme {scheme!r}")

        if not folds or not final_refit_windows:
            raise ValueError(f"{subject}: split JSON contains no validation/refit windows")
        refit_runs: list[str] = []
        for window in final_refit_windows:
            if not isinstance(window, dict):
                raise ValueError(f"{subject}: final-refit window must be an object")
            run, _start, _stop = _validate_proposal_window(subject, window, eligible)
            refit_runs.append(run)
        test_window_run, test_start, test_stop = _validate_proposal_window(
            subject, test_window, eligible
        )
        if set(refit_runs) != set(train_runs) or test_window_run != test_run:
            raise ValueError(f"{subject}: JSON refit/test windows disagree with run roles")
        for fold in folds:
            if not isinstance(fold, dict):
                raise ValueError(f"{subject}: inner-validation fold must be an object")
            training_windows = fold.get("training_windows")
            validation_window = fold.get("validation_window")
            if not isinstance(training_windows, list) or not training_windows:
                raise ValueError(f"{subject}: fold has no training windows")
            if not isinstance(validation_window, dict):
                raise ValueError(f"{subject}: fold has no validation window")
            fold_train_runs = []
            for window in training_windows:
                if not isinstance(window, dict):
                    raise ValueError(f"{subject}: fold training window must be an object")
                run, _start, _stop = _validate_proposal_window(
                    subject, window, eligible
                )
                fold_train_runs.append(run)
            validation_run, validation_start, _validation_stop = (
                _validate_proposal_window(subject, validation_window, eligible)
            )
            declared_train_runs = [str(value) for value in fold.get("training_runs", [])]
            if set(fold_train_runs) != set(declared_train_runs):
                raise ValueError(f"{subject}: fold JSON training runs disagree with windows")
            if not set(fold_train_runs) <= set(train_runs) or validation_run not in train_runs:
                raise ValueError(f"{subject}: inner fold uses a non-training run")
            if validation_run in fold_train_runs and len(train_runs) > 1:
                raise ValueError(f"{subject}: seizure-blocked fold leaks validation run")
            for window in training_windows:
                run, _start, stop = _validate_proposal_window(subject, window, eligible)
                if run == validation_run and stop > validation_start:
                    raise ValueError(f"{subject}: temporal fold training overlaps validation")

        if scheme == "blocked_single_seizure":
            fold = folds[0]
            train_window = fold["training_windows"][0]
            validation_window = fold["validation_window"]
            refit_window = final_refit_windows[0]
            observed_protocol = (
                float(train_window["start_offset_from_onset_s"]),
                float(train_window["stop_offset_from_onset_s"]),
                float(validation_window["start_offset_from_onset_s"]),
                float(validation_window["stop_offset_from_onset_s"]),
                float(refit_window["start_offset_from_onset_s"]),
                float(refit_window["stop_offset_from_onset_s"]),
                test_start,
                test_stop,
            )
            expected_protocol = (0.0, 12.0, 14.0, 19.0, 0.0, 19.0, 21.0, 32.0)
            if len(folds) != 1 or not np.allclose(observed_protocol, expected_protocol):
                raise ValueError(f"{subject}: schema-2 single-seizure protocol changed")
            if not np.isclose(
                float(row["inner_validation_embargo_s"]),
                observed_protocol[2] - observed_protocol[1],
            ) or not np.isclose(
                float(row["final_test_embargo_s"]),
                observed_protocol[6] - observed_protocol[5],
            ):
                raise ValueError(f"{subject}: declared temporal embargo is inconsistent")
        proposals[subject] = {
            "subject": subject,
            "split_scheme": scheme,
            "all_ictal_runs": all_runs,
            "eligible_ictal_runs": eligible,
            "excluded_event_qc_runs": excluded,
            "ictal_train_runs": train_runs,
            "ictal_test_run": test_run,
            "inner_validation_scheme": inner_scheme,
            "inner_validation_folds": folds,
            "final_refit_windows": final_refit_windows,
            "test_window": test_window,
            "inner_validation_embargo_s": str(row["inner_validation_embargo_s"]),
            "final_test_embargo_s": str(row["final_test_embargo_s"]),
        }
    return proposals, {"path": str(resolved), "sha256": _sha256_file(resolved)}


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_safe_json(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def merge_subject_frame(path: Path, subject: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError(f"refusing to save empty rows for {subject}")
    existing = _read_frame(path)
    if not existing.empty and "subject" in existing:
        existing = existing[existing["subject"].astype(str) != str(subject)]
    combined = pd.concat([existing, frame], ignore_index=True, sort=False)
    sort_columns = [
        column
        for column in (
            "subject",
            "model_label",
            "candidate_id",
            "fold_id",
            "window",
            "lead_end_samples",
        )
        if column in combined
    ]
    if sort_columns:
        combined = combined.sort_values(sort_columns, kind="mergesort")
    _atomic_frame(path, combined)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    return sha256(
        json.dumps(_safe_json(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def analysis_fingerprints(
    config_path: Path,
    split_proposal_metadata: dict[str, str],
) -> dict[str, str]:
    """Hashes that make resume decisions code/config/split specific."""

    return {
        "analysis_schema_version": SCHEMA_VERSION,
        "analysis_config_sha256": _sha256_file(config_path.resolve()),
        "analysis_split_proposal_sha256": str(split_proposal_metadata["sha256"]),
        "analysis_runner_sha256": _sha256_file(Path(__file__).resolve()),
        "analysis_rc_module_sha256": _sha256_file(
            PROJECT_ROOT / "mfc_pipeline" / "rc_v2.py"
        ),
        "analysis_data_loader_sha256": _sha256_file(
            PROJECT_ROOT / "mfc_pipeline" / "data.py"
        ),
    }


def setup_logging(output_root: Path) -> None:
    log_path = output_root / "logs" / "prediction_v2.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )


def build_loader(config: dict[str, Any]) -> tuple[BIDSZipDataset, EEGSegmentLoader]:
    data = config["data"]
    resampling_mode = str(data["resampling_mode"])
    causal_half_length = int(data["causal_fir_half_length_factor"])
    forecast_guard = int(data["forecast_guard_samples"])
    if resampling_mode != "causal_upfirdn":
        raise ValueError("V2 prediction requires data.resampling_mode='causal_upfirdn'")
    if forecast_guard != 0:
        raise ValueError("strictly causal V2 prediction requires forecast_guard_samples=0")
    cache = resolve_project_path(config["project"]["cache_dir"])
    dataset = BIDSZipDataset(
        data["zip_root"],
        subjects=data["subjects"],
        offset_qc_overrides_s=data.get("offset_qc_overrides_s", {}),
    )
    loader = EEGSegmentLoader(
        dataset,
        cache_dir=cache / "segments",
        temp_dir=cache / "_edf_tmp",
        target_sfreq=float(data["target_sampling_rate_hz"]),
        bandpass_hz=tuple(data["bandpass_hz"]),
        burn_in_s=float(data["filter_burn_in_s"]),
        reference=str(data["reference"]),
        resampling_mode=resampling_mode,
        causal_fir_half_length_factor=causal_half_length,
    )
    return dataset, loader


def compute_adjacency(sequences: list[np.ndarray], sfreq: float) -> np.ndarray:
    matrices = [
        windowed_plv_median(
            sequence,
            sfreq,
            window_s=PLV_WINDOW_S,
            overlap=PLV_OVERLAP,
            time_axis=0,
        )
        for sequence in sequences
    ]
    plv = np.median(np.stack(matrices), axis=0)
    return mst_proportional_graph(plv, density=NETWORK_DENSITY)


def _slice_proposal_window(
    values: dict[str, np.ndarray],
    window: dict[str, Any],
    sfreq: float,
    subject: str,
) -> tuple[np.ndarray, str, int, int]:
    run = str(window["run"])
    start_s = float(window["start_offset_from_onset_s"])
    stop_s = float(window["stop_offset_from_onset_s"])
    start = int(round(start_s * sfreq))
    stop = int(round(stop_s * sfreq))
    if run not in values or not (0 <= start < stop <= len(values[run])):
        raise ValueError(f"{subject}: proposal window {run}[{start_s},{stop_s}) is invalid")
    label = f"{run}[{start_s:g},{stop_s:g}s)"
    return values[run][start:stop].copy(), label, start, stop


def load_ictal_plan(
    config: dict[str, Any],
    dataset: BIDSZipDataset,
    loader: EEGSegmentLoader,
    subject: str,
    split_proposal: dict[str, Any],
    split_proposal_sha256: str,
    fingerprints: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Load only event-QC-eligible ictal signals and enforce frozen run roles."""

    if str(split_proposal["subject"]) != str(subject):
        raise ValueError(f"{subject}: mismatched QC split proposal row")
    manifest_records = list(dataset.records(subject, task="ictal"))
    manifest_runs = [str(record.run) for record in manifest_records]
    proposed_all = [str(value) for value in split_proposal["all_ictal_runs"]]
    if not manifest_runs or len(set(manifest_runs)) != len(manifest_runs):
        raise ValueError(f"{subject} has an invalid ictal run manifest: {manifest_runs}")
    if set(manifest_runs) != set(proposed_all):
        raise ValueError(
            f"{subject}: dataset ictal manifest {manifest_runs} differs from QC proposal "
            f"all_ictal_runs {proposed_all}"
        )
    runs = [str(value) for value in split_proposal["eligible_ictal_runs"]]
    excluded_runs = [str(value) for value in split_proposal["excluded_event_qc_runs"]]
    if set(runs) & set(excluded_runs):
        raise AssertionError(f"{subject}: excluded event-QC run entered eligible run set")
    records_by_run = {str(record.run): record for record in manifest_records}
    records = [records_by_run[run] for run in runs]
    selected_records = dataset.selected_records(subject, [("ictal", run) for run in runs])
    channels = dataset.common_good_channels(subject, records=selected_records)
    duration = float(config["data"]["ictal_segment_s"])
    segments = {
        run: loader.load_ictal(
            subject,
            run,
            duration_s=duration,
            channel_names=channels,
            use_cache=True,
        )
        for run in runs
    }
    for run, segment in segments.items():
        provenance = segment.provenance
        if not bool(provenance.get("preprocessing_fully_causal", False)):
            raise AssertionError(f"{subject} {run}: V2 preprocessing is not fully causal")
        if not bool(provenance.get("resampling_causal", False)):
            raise AssertionError(f"{subject} {run}: V2 resampling is not causal")
        if bool(provenance.get("forecast_boundary_guard_required", True)):
            raise AssertionError(f"{subject} {run}: causal preprocessing still requests a guard")
    values = {run: segments[run].data.T.astype(np.float64) for run in runs}
    sfreq = float(segments[runs[0]].sfreq)
    if any(not np.isclose(segment.sfreq, sfreq) for segment in segments.values()):
        raise ValueError(f"{subject} ictal runs do not share a sampling rate")
    test_run = str(split_proposal["ictal_test_run"])
    training_runs = [str(value) for value in split_proposal["ictal_train_runs"]]
    if test_run != str(config["data"]["analysis_test_runs"][subject]):
        raise AssertionError(f"{subject}: proposal/config final test run mismatch")
    if set(training_runs) | {test_run} != set(runs):
        raise AssertionError(f"{subject}: proposal train/test roles do not match eligible runs")
    folds: list[dict[str, Any]] = []
    for fold_definition in split_proposal["inner_validation_folds"]:
        train_sequences: list[np.ndarray] = []
        train_labels: list[str] = []
        for window in fold_definition["training_windows"]:
            sequence, label, _start, _stop = _slice_proposal_window(
                values, window, sfreq, subject
            )
            train_sequences.append(sequence)
            train_labels.append(label)
        validation, validation_label, _start, _stop = _slice_proposal_window(
            values, fold_definition["validation_window"], sfreq, subject
        )
        validation_embargo = fold_definition.get(
            "embargo_s", fold_definition.get("embargo", "unspecified")
        )
        folds.append(
            {
                "fold_id": str(fold_definition["fold_id"]),
                "train_sequences": train_sequences,
                "validation_sequence": validation,
                "training_labels": train_labels,
                "validation_label": validation_label,
                "validation_embargo_s": validation_embargo,
            }
        )

    final_train_sequences: list[np.ndarray] = []
    final_train_labels: list[str] = []
    for window in split_proposal["final_refit_windows"]:
        sequence, label, _start, _stop = _slice_proposal_window(
            values, window, sfreq, subject
        )
        final_train_sequences.append(sequence)
        final_train_labels.append(label)
    test_sequence, test_label, test_start_sample, _test_stop = _slice_proposal_window(
        values, split_proposal["test_window"], sfreq, subject
    )
    if split_proposal["split_scheme"] == "blocked_single_seizure":
        generalization_scope = "within_record_temporal_generalization"
        split_type = "single_seizure_nested_temporal"
    elif split_proposal["split_scheme"] == "held_out_seizure":
        generalization_scope = "held_out_seizure_generalization"
        split_type = "held_out_final_seizure"
    else:
        raise AssertionError(f"{subject}: unsupported validated split scheme")
    plan = {
        "generalization_scope": generalization_scope,
        "split_type": split_type,
        "inner_validation_scope": split_proposal["inner_validation_scheme"],
        "final_train_sequences": final_train_sequences,
        "final_train_labels": final_train_labels,
        "test_sequence": test_sequence,
        "test_offset_samples": test_start_sample,
        "test_label": test_label,
        "final_test_embargo_s": split_proposal["final_test_embargo_s"],
        "split_windows_driven_by_schema2_json": True,
    }

    return {
        "subject": subject,
        "ictal_runs": runs,
        "all_ictal_runs": proposed_all,
        "eligible_ictal_runs": runs,
        "excluded_event_qc_runs": excluded_runs,
        "proposal_training_runs": training_runs,
        "split_proposal_sha256": str(split_proposal_sha256),
        "analysis_fingerprints": dict(fingerprints or {}),
        "preprocessing_fully_causal": True,
        "resampling_mode": str(config["data"]["resampling_mode"]),
        "causal_fir_half_length_factor": int(
            config["data"]["causal_fir_half_length_factor"]
        ),
        "forecast_guard_samples": int(config["data"]["forecast_guard_samples"]),
        "source_sampling_rates_hz": {
            run: float(segments[run].source_sfreq) for run in runs
        },
        "channels": channels,
        "sfreq": sfreq,
        "test_run": test_run,
        "folds": folds,
        "preictal_signal_loaded": False,
        **plan,
    }


def _tag(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}".replace("-", "m").replace(".", "p")
    if isinstance(value, (list, tuple)):
        return "-".join(str(int(item)) for item in value)
    return str(value)


def compact_profiles(prediction: dict[str, Any], quick: bool) -> list[dict[str, Any]]:
    ridge_values = [float(value) for value in prediction["ridge_alphas"]]
    middle_ridge = ridge_values[len(ridge_values) // 2]
    profiles = [
        {
            "profile_id": "compact_conservative",
            "reservoir_size": int(prediction["reservoir_sizes"][0]),
            "spectral_radius": float(prediction["spectral_radii"][0]),
            "leak_rate": float(prediction["leak_rates"][0]),
            "input_scale": float(prediction["input_scales"][0]),
            "ridge_alpha": middle_ridge,
            "delays_samples": tuple(int(value) for value in prediction["delay_sets_samples"][0]),
        },
        {
            "profile_id": "compact_alternative",
            "reservoir_size": int(prediction["reservoir_sizes"][-1]),
            "spectral_radius": float(prediction["spectral_radii"][-1]),
            "leak_rate": float(prediction["leak_rates"][-1]),
            "input_scale": float(prediction["input_scales"][-1]),
            "ridge_alpha": ridge_values[-1],
            "delays_samples": tuple(int(value) for value in prediction["delay_sets_samples"][-1]),
        },
    ]
    return profiles[:1] if quick else profiles


def candidate_specs(config: dict[str, Any], seed: int, quick: bool) -> list[dict[str, Any]]:
    prediction = config["prediction"]
    maximum_direct = max(int(value) for value in prediction["horizons_samples"])
    common = {
        "variance_threshold": float(prediction["latent_variance_threshold"]),
        "latent_components_min": int(prediction["latent_components_min"]),
        "latent_components_max": int(prediction["latent_components_max"]),
        "maximum_direct_horizon": maximum_direct,
        "random_seed": int(seed),
    }
    specs: list[dict[str, Any]] = []

    def add(model_label: str, candidate_id: str, method: str, fit_form: str, evaluation_form: str, **kwargs: Any) -> None:
        numerical = {**common, **kwargs}
        fit_key_payload = {"method": method, "fit_form": fit_form, **numerical}
        specs.append(
            {
                "model_label": model_label,
                "candidate_id": candidate_id,
                "method": method,
                "fit_readout_form": fit_form,
                "evaluation_readout_form": evaluation_form,
                "fit_key": _sha256_json(fit_key_payload)[:16],
                **numerical,
            }
        )

    add("persistence", "persistence", "persistence", "persistence", "persistence")
    ridge_values = [float(value) for value in prediction["ridge_alphas"]]
    for alpha in ridge_values:
        add(
            "ridge_var_recursive",
            f"ridge_a{_tag(alpha)}",
            "ridge_var",
            "recursive",
            "recursive",
            ridge_alpha=alpha,
        )
    for delays in prediction["delay_sets_samples"]:
        delay_tuple = tuple(int(value) for value in delays)
        for alpha in ridge_values:
            add(
                "delay_linear_direct",
                f"delay_{_tag(delay_tuple)}_a{_tag(alpha)}",
                "delay_linear",
                "direct_horizon",
                "direct_horizon",
                ridge_alpha=alpha,
                delays_samples=delay_tuple,
            )
    for profile in compact_profiles(prediction, quick):
        profile_values = {key: value for key, value in profile.items() if key != "profile_id"}
        profile_id = str(profile["profile_id"])
        for method, label in (
            ("esn_plain", "esn_plain_recursive"),
            ("esn_topology", "esn_topology_recursive"),
        ):
            add(label, profile_id, method, "recursive", "recursive", **profile_values)
        for method, stem in (
            ("esn_delay", "esn_delay"),
            ("rc_topology_delay", "rc_topology_delay"),
        ):
            add(
                f"{stem}_recursive",
                profile_id,
                method,
                "direct_horizon",
                "recursive",
                **profile_values,
            )
            add(
                f"{stem}_direct",
                profile_id,
                method,
                "direct_horizon",
                "direct_horizon",
                **profile_values,
            )
    return specs


def spec_to_model(spec: dict[str, Any], adjacency: np.ndarray) -> RCV2Forecaster:
    defaults = {
        "reservoir_size": 96,
        "spectral_radius": 0.75,
        "leak_rate": 0.25,
        "input_scale": 0.10,
        "ridge_alpha": 1e-3,
        "delays_samples": (16, 64),
    }
    values = {**defaults, **spec}
    model_config = RCV2Config(
        method=str(values["method"]),
        readout_form=str(values["fit_readout_form"]),
        variance_threshold=float(values["variance_threshold"]),
        latent_components_min=int(values["latent_components_min"]),
        latent_components_max=int(values["latent_components_max"]),
        reservoir_size=int(values["reservoir_size"]),
        spectral_radius=float(values["spectral_radius"]),
        leak_rate=float(values["leak_rate"]),
        input_scale=float(values["input_scale"]),
        ridge_alpha=float(values["ridge_alpha"]),
        delays_samples=tuple(int(item) for item in values["delays_samples"]),
        maximum_direct_horizon=int(values["maximum_direct_horizon"]),
        random_seed=int(values["random_seed"]),
    )
    return RCV2Forecaster(model_config, adjacency=adjacency)


def score_model(
    model: RCV2Forecaster,
    sequence: np.ndarray,
    *,
    readout_form: str,
    horizons: list[int],
    context_samples: int,
    window_count: int,
    projection_seed_offset: int,
) -> list[dict[str, float | int]]:
    maximum = max(horizons)
    positions = forecast_positions(
        len(sequence),
        context_samples=context_samples,
        maximum_horizon=maximum,
        guard_samples=0,
        count=window_count,
    )
    rows: list[dict[str, float | int]] = []
    for window, position in enumerate(positions):
        context = sequence[position - context_samples : position]
        full_horizon = maximum
        forecast = model.forecast(
            context,
            full_horizon,
            readout_form=readout_form,
        )
        persistence = np.repeat(context[-1:], full_horizon, axis=0)
        for horizon in horizons:
            truth = sequence[position : position + horizon]
            estimate = forecast[:horizon]
            baseline = persistence[:horizon]
            metrics = safe_prediction_metrics(truth, estimate)
            baseline_metrics = safe_prediction_metrics(truth, baseline)
            exact_metrics = safe_prediction_metrics(truth[-1:], estimate[-1:])
            exact_baseline_metrics = safe_prediction_metrics(truth[-1:], baseline[-1:])
            exact_truth = np.asarray(truth[-1], dtype=float).ravel()
            exact_estimate = np.asarray(estimate[-1], dtype=float).ravel()
            exact_persistence = np.asarray(baseline[-1], dtype=float).ravel()
            projections = fixed_random_projections(
                truth.shape[1],
                n_projections=128,
                seed=projection_seed_offset + int(horizon),
            )
            swd = safe_sliced_wasserstein(estimate, truth, projections)
            persistence_swd = safe_sliced_wasserstein(baseline, truth, projections)
            rows.append(
                {
                    "window": int(window),
                    "forecast_boundary_sample": int(position),
                    "lead_end_samples": int(horizon),
                    "path_start_lead_samples": 1,
                    "path_nrmse": float(metrics["nrmse"]),
                    "path_rmse": float(metrics["rmse"]),
                    "path_mae": float(np.mean(np.abs(estimate - truth))),
                    "path_correlation": float(metrics["correlation"]),
                    "path_mean_channel_nrmse": float(metrics["mean_channel_nrmse"]),
                    "path_sliced_wasserstein": swd,
                    "persistence_path_nrmse": float(baseline_metrics["nrmse"]),
                    "persistence_path_mae": float(np.mean(np.abs(baseline - truth))),
                    "persistence_path_correlation": float(baseline_metrics["correlation"]),
                    "persistence_path_sliced_wasserstein": persistence_swd,
                    "relative_path_nrmse_reduction_vs_persistence": float(
                        1.0 - metrics["nrmse"] / baseline_metrics["nrmse"]
                    )
                    if baseline_metrics["nrmse"] > 0
                    else float("nan"),
                    "relative_path_swd_reduction_vs_persistence": float(
                        1.0 - swd / persistence_swd
                    )
                    if persistence_swd > 0
                    else float("nan"),
                    "exact_lead_nrmse": float(exact_metrics["nrmse"]),
                    "exact_lead_rmse": float(exact_metrics["rmse"]),
                    "exact_lead_mae": float(np.mean(np.abs(estimate[-1:] - truth[-1:]))),
                    "exact_lead_correlation": float(exact_metrics["correlation"]),
                    "persistence_exact_lead_nrmse": float(
                        exact_baseline_metrics["nrmse"]
                    ),
                    "persistence_exact_lead_mae": float(
                        np.mean(np.abs(baseline[-1:] - truth[-1:]))
                    ),
                    "persistence_exact_lead_correlation": float(
                        exact_baseline_metrics["correlation"]
                    ),
                    "relative_exact_lead_nrmse_reduction_vs_persistence": float(
                        1.0
                        - exact_metrics["nrmse"] / exact_baseline_metrics["nrmse"]
                    )
                    if exact_baseline_metrics["nrmse"] > 0
                    else float("nan"),
                    # Sufficient statistics allow patient-level exact-lead
                    # NRMSE/MAE to be pooled across all test windows/channels.
                    "exact_lead_value_count": int(exact_truth.size),
                    "exact_lead_truth_sum": float(np.sum(exact_truth)),
                    "exact_lead_truth_squared_sum": float(
                        np.sum(np.square(exact_truth))
                    ),
                    "exact_lead_squared_error_sum": float(
                        np.sum(np.square(exact_estimate - exact_truth))
                    ),
                    "exact_lead_absolute_error_sum": float(
                        np.sum(np.abs(exact_estimate - exact_truth))
                    ),
                    "persistence_exact_lead_squared_error_sum": float(
                        np.sum(np.square(exact_persistence - exact_truth))
                    ),
                    "persistence_exact_lead_absolute_error_sum": float(
                        np.sum(np.abs(exact_persistence - exact_truth))
                    ),
                }
            )
    return rows


def validate_candidates(
    plan: dict[str, Any],
    specs: list[dict[str, Any]],
    prediction: dict[str, Any],
    quick: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], dict[str, Any]]]:
    horizons = [int(value) for value in prediction["horizons_samples"]]
    primary = int(prediction["primary_horizon_samples"])
    context = int(prediction["context_samples"])
    weights = prediction["validation_objective_weights"]
    validation_rows: list[dict[str, Any]] = []
    spec_lookup = {(str(spec["model_label"]), str(spec["candidate_id"])): spec for spec in specs}

    for fold_number, fold in enumerate(plan["folds"]):
        adjacency = compute_adjacency(fold["train_sequences"], plan["sfreq"])
        cache: dict[str, tuple[RCV2Forecaster, float]] = {}
        for spec in specs:
            fit_key = str(spec["fit_key"])
            if fit_key not in cache:
                started = time.perf_counter()
                model = spec_to_model(spec, adjacency)
                model.fit_sequences(fold["train_sequences"])
                cache[fit_key] = (model, time.perf_counter() - started)
            model, fit_seconds = cache[fit_key]
            scored = score_model(
                model,
                fold["validation_sequence"],
                readout_form=str(spec["evaluation_readout_form"]),
                horizons=[primary],
                context_samples=context,
                window_count=2 if quick else VALIDATION_WINDOWS,
                projection_seed_offset=10260716 + 1000 * fold_number,
            )
            frame = pd.DataFrame(scored)
            mean_nrmse = float(frame["path_nrmse"].mean())
            mean_swd = float(frame["path_sliced_wasserstein"].mean())
            persistence_swd = float(
                frame["persistence_path_sliced_wasserstein"].mean()
            )
            swd_ratio = mean_swd / persistence_swd if persistence_swd > 0 else float("inf")
            objective = (
                float(weights["normalized_rmse"]) * mean_nrmse
                + float(weights["sliced_wasserstein"]) * swd_ratio
            )
            validation_rows.append(
                {
                    "subject": plan["subject"],
                    "generalization_scope": plan["generalization_scope"],
                    "eligible_ictal_runs": ";".join(plan["eligible_ictal_runs"]),
                    "excluded_event_qc_runs": ";".join(
                        plan["excluded_event_qc_runs"]
                    ),
                    "split_proposal_sha256": plan["split_proposal_sha256"],
                    "split_windows_driven_by_schema2_json": plan[
                        "split_windows_driven_by_schema2_json"
                    ],
                    **plan["analysis_fingerprints"],
                    "fold_id": fold["fold_id"],
                    "fold_training_labels": ";".join(fold["training_labels"]),
                    "fold_validation_label": fold["validation_label"],
                    "validation_embargo_s": fold["validation_embargo_s"],
                    "model_label": spec["model_label"],
                    "candidate_id": spec["candidate_id"],
                    "fit_key": fit_key,
                    "method": spec["method"],
                    "readout_form": spec["evaluation_readout_form"],
                    "seed": int(spec["random_seed"]),
                    "validation_lead_end_samples": primary,
                    "validation_windows": int(len(frame)),
                    "validation_path_nrmse": mean_nrmse,
                    "validation_path_mae": float(frame["path_mae"].mean()),
                    "validation_path_correlation": float(
                        frame["path_correlation"].mean()
                    ),
                    "validation_exact_lead_nrmse": float(
                        frame["exact_lead_nrmse"].mean()
                    ),
                    "validation_exact_lead_mae": float(
                        frame["exact_lead_mae"].mean()
                    ),
                    "validation_path_swd": mean_swd,
                    "validation_persistence_path_nrmse": float(
                        frame["persistence_path_nrmse"].mean()
                    ),
                    "validation_persistence_path_swd": persistence_swd,
                    "validation_path_swd_ratio_to_persistence": swd_ratio,
                    "validation_objective": objective,
                    "fit_seconds": fit_seconds,
                    "latent_components": model.latent_dim_,
                    "latent_explained_variance": model.latent_selection_[
                        "explained_variance_at_q"
                    ],
                    "variance_threshold_reached": model.latent_selection_[
                        "variance_threshold_reached"
                    ],
                    "test_run_used_for_selection": False,
                    "preictal_signal_loaded": False,
                }
            )
        LOGGER.info("%s validation fold %s complete", plan["subject"], fold["fold_id"])

    validation = pd.DataFrame(validation_rows)
    summaries = (
        validation.groupby(["subject", "model_label", "candidate_id"], sort=True)
        .agg(
            validation_objective=("validation_objective", "mean"),
            validation_objective_sd=("validation_objective", "std"),
            validation_path_nrmse=("validation_path_nrmse", "mean"),
            validation_path_swd=("validation_path_swd", "mean"),
            n_validation_folds=("fold_id", "nunique"),
            validation_fit_seconds=("fit_seconds", "sum"),
        )
        .reset_index()
    )
    summaries["validation_objective_sd"] = summaries["validation_objective_sd"].fillna(0.0)
    selected_rows: list[dict[str, Any]] = []
    selected_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for model_label in INDEPENDENT_MODEL_LABELS:
        options = summaries[summaries["model_label"] == model_label].sort_values(
            ["validation_objective", "candidate_id"], kind="mergesort"
        )
        if options.empty:
            raise RuntimeError(f"no validation candidate for {model_label}")
        best = options.iloc[0].to_dict()
        key = (model_label, str(best["candidate_id"]))
        spec = dict(spec_lookup[key])
        selected_lookup[key] = spec
        selected_rows.append(
            {
                **best,
                "seed": int(spec["random_seed"]),
                "generalization_scope": plan["generalization_scope"],
                "method": spec["method"],
                "readout_form": spec["evaluation_readout_form"],
                "fit_readout_form": spec["fit_readout_form"],
                "fit_key": spec["fit_key"],
                "matched_ablation": False,
                "matched_to_primary_model_label": "",
                "matched_to_primary_candidate_id": "",
                "selected_without_final_test": True,
                "selection_mode": "independent_validation_optimization",
                "selection_rule": (
                    "minimum mean blocked-validation objective; deterministic candidate-id tie break"
                ),
                "validation_objective_formula": (
                    "0.65*path_nrmse + 0.35*(path_swd/persistence_path_swd)"
                ),
                "candidate_config_json": json.dumps(_safe_json(spec), sort_keys=True),
                **plan["analysis_fingerprints"],
            }
        )
    selected_frame = pd.DataFrame(selected_rows)
    primary = selected_frame[
        selected_frame["model_label"] == "rc_topology_delay_direct"
    ]
    if len(primary) != 1:
        raise RuntimeError("primary topology+delay direct model was not selected exactly once")
    anchor_candidate = str(primary.iloc[0]["candidate_id"])
    matched_source_key = ("esn_delay_direct", anchor_candidate)
    if matched_source_key not in spec_lookup:
        raise RuntimeError(
            "no topology-removed candidate exactly matches the selected primary profile"
        )
    matched_options = summaries[
        (summaries["model_label"] == "esn_delay_direct")
        & (summaries["candidate_id"].astype(str) == anchor_candidate)
    ]
    if len(matched_options) != 1:
        raise RuntimeError("matched-profile validation summary is not unique")
    matched_best = matched_options.iloc[0].to_dict()
    matched_spec = dict(spec_lookup[matched_source_key])
    matched_spec["model_label"] = MATCHED_TOPOLOGY_ABLATION_LABEL
    matched_key = (MATCHED_TOPOLOGY_ABLATION_LABEL, anchor_candidate)
    selected_lookup[matched_key] = matched_spec
    selected_rows.append(
        {
            **matched_best,
            "model_label": MATCHED_TOPOLOGY_ABLATION_LABEL,
            "candidate_id": anchor_candidate,
            "seed": int(matched_spec["random_seed"]),
            "generalization_scope": plan["generalization_scope"],
            "method": matched_spec["method"],
            "readout_form": matched_spec["evaluation_readout_form"],
            "fit_readout_form": matched_spec["fit_readout_form"],
            "fit_key": matched_spec["fit_key"],
            "matched_ablation": True,
            "matched_to_primary_model_label": "rc_topology_delay_direct",
            "matched_to_primary_candidate_id": anchor_candidate,
            "selected_without_final_test": True,
            "selection_mode": "profile_anchored_to_selected_primary_full_model",
            "selection_rule": (
                "copy selected primary reservoir/delay/ridge/seed profile; remove only "
                "topology input; no test-based retuning"
            ),
            "validation_objective_formula": "0.65*path_nrmse + 0.35*(path_swd/persistence_path_swd)",
            "candidate_config_json": json.dumps(
                _safe_json(matched_spec), sort_keys=True
            ),
            **plan["analysis_fingerprints"],
        }
    )
    return validation, pd.DataFrame(selected_rows), selected_lookup


def final_fit_and_test(
    plan: dict[str, Any],
    selected: pd.DataFrame,
    selected_lookup: dict[tuple[str, str], dict[str, Any]],
    prediction: dict[str, Any],
    output_root: Path,
    quick: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    adjacency = compute_adjacency(plan["final_train_sequences"], plan["sfreq"])
    horizons = [int(value) for value in prediction["horizons_samples"]]
    context = int(prediction["context_samples"])
    models: dict[str, tuple[RCV2Forecaster, float, Path, str]] = {}
    selected_records: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    model_directory = output_root / "models" / plan["subject"]
    model_directory.mkdir(parents=True, exist_ok=True)

    for selected_row in selected.to_dict(orient="records"):
        model_label = str(selected_row["model_label"])
        candidate_id = str(selected_row["candidate_id"])
        spec = selected_lookup[(model_label, candidate_id)]
        fit_key = str(spec["fit_key"])
        if fit_key not in models:
            started = time.perf_counter()
            model = spec_to_model(spec, adjacency)
            model.fit_sequences(plan["final_train_sequences"])
            fit_seconds = time.perf_counter() - started
            model_path = model_directory / f"fit-{fit_key}.joblib"
            temporary = model_path.with_suffix(".joblib.tmp")
            joblib.dump(model, temporary, compress=3)
            temporary.replace(model_path)
            models[fit_key] = (model, fit_seconds, model_path, _sha256_file(model_path))
        model, fit_seconds, model_path, model_hash = models[fit_key]
        selected_records.append(
            {
                **selected_row,
                "final_training_labels": ";".join(plan["final_train_labels"]),
                "eligible_ictal_runs": ";".join(plan["eligible_ictal_runs"]),
                "excluded_event_qc_runs": ";".join(plan["excluded_event_qc_runs"]),
                "split_proposal_sha256": plan["split_proposal_sha256"],
                "split_windows_driven_by_schema2_json": plan[
                    "split_windows_driven_by_schema2_json"
                ],
                "final_test_run": plan["test_run"],
                "final_test_label": plan["test_label"],
                "final_test_embargo_s": plan["final_test_embargo_s"],
                "model_path": str(model_path.resolve()),
                "model_sha256": model_hash,
                "final_fit_seconds": fit_seconds,
                "final_latent_components": model.latent_dim_,
                "final_latent_explained_variance": model.latent_selection_[
                    "explained_variance_at_q"
                ],
                "final_variance_threshold_reached": model.latent_selection_[
                    "variance_threshold_reached"
                ],
                "test_metrics_present_in_selection_table": False,
                "preictal_signal_loaded": False,
            }
        )
        scored = score_model(
            model,
            plan["test_sequence"],
            readout_form=str(spec["evaluation_readout_form"]),
            horizons=horizons,
            context_samples=context,
            window_count=2 if quick else int(prediction["windows_per_test_run"]),
            projection_seed_offset=20260716,
        )
        for row in scored:
            window_rows.append(
                {
                    "subject": plan["subject"],
                    "generalization_scope": plan["generalization_scope"],
                    "split_type": plan["split_type"],
                    "test_run": plan["test_run"],
                    "test_label": plan["test_label"],
                    "model_label": model_label,
                    "matched_ablation": bool(selected_row["matched_ablation"]),
                    "matched_to_primary_model_label": selected_row[
                        "matched_to_primary_model_label"
                    ],
                    "eligible_ictal_runs": ";".join(plan["eligible_ictal_runs"]),
                    "excluded_event_qc_runs": ";".join(
                        plan["excluded_event_qc_runs"]
                    ),
                    "split_proposal_sha256": plan["split_proposal_sha256"],
                    "split_windows_driven_by_schema2_json": plan[
                        "split_windows_driven_by_schema2_json"
                    ],
                    **plan["analysis_fingerprints"],
                    "candidate_id": candidate_id,
                    "method": spec["method"],
                    "readout_form": spec["evaluation_readout_form"],
                    "seed": int(spec["random_seed"]),
                    "sampling_rate_hz": plan["sfreq"],
                    "lead_end_s": int(row["lead_end_samples"]) / plan["sfreq"],
                    "context_samples": context,
                    "forecast_guard_samples": plan["forecast_guard_samples"],
                    "preprocessing_fully_causal": True,
                    "resampling_mode": plan["resampling_mode"],
                    "forecast_boundary_sample_within_test_block": row[
                        "forecast_boundary_sample"
                    ],
                    "forecast_boundary_sample_from_loaded_ictal_start": int(
                        plan["test_offset_samples"] + int(row["forecast_boundary_sample"])
                    ),
                    "final_latent_components": model.latent_dim_,
                    "final_latent_explained_variance": model.latent_selection_[
                        "explained_variance_at_q"
                    ],
                    "selection_validation_objective": selected_row[
                        "validation_objective"
                    ],
                    "selected_without_final_test": True,
                    "preictal_signal_loaded": False,
                    **row,
                }
            )
        LOGGER.info("%s final test %s complete", plan["subject"], model_label)

    return pd.DataFrame(selected_records), pd.DataFrame(window_rows)


def patient_and_calibration(window: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patient = (
        window.groupby(
            [
                "subject",
                "generalization_scope",
                "model_label",
                "method",
                "readout_form",
                "matched_ablation",
                "seed",
                "lead_end_samples",
                "lead_end_s",
            ],
            sort=True,
        )
        .agg(
            path_nrmse=("path_nrmse", "mean"),
            path_nrmse_sd=("path_nrmse", "std"),
            path_mae=("path_mae", "mean"),
            path_correlation=("path_correlation", "mean"),
            path_sliced_wasserstein=("path_sliced_wasserstein", "mean"),
            persistence_path_nrmse=("persistence_path_nrmse", "mean"),
            persistence_path_mae=("persistence_path_mae", "mean"),
            persistence_path_correlation=("persistence_path_correlation", "mean"),
            persistence_path_sliced_wasserstein=(
                "persistence_path_sliced_wasserstein",
                "mean",
            ),
            window_exact_lead_nrmse_mean=("exact_lead_nrmse", "mean"),
            window_exact_lead_nrmse_sd=("exact_lead_nrmse", "std"),
            window_exact_lead_correlation_mean=("exact_lead_correlation", "mean"),
            window_persistence_exact_lead_correlation_mean=(
                "persistence_exact_lead_correlation",
                "mean",
            ),
            exact_lead_value_count=("exact_lead_value_count", "sum"),
            exact_lead_truth_sum=("exact_lead_truth_sum", "sum"),
            exact_lead_truth_squared_sum=("exact_lead_truth_squared_sum", "sum"),
            exact_lead_squared_error_sum=("exact_lead_squared_error_sum", "sum"),
            exact_lead_absolute_error_sum=("exact_lead_absolute_error_sum", "sum"),
            persistence_exact_lead_squared_error_sum=(
                "persistence_exact_lead_squared_error_sum",
                "sum",
            ),
            persistence_exact_lead_absolute_error_sum=(
                "persistence_exact_lead_absolute_error_sum",
                "sum",
            ),
            n_windows=("window", "nunique"),
        )
        .reset_index()
    )
    patient["relative_path_nrmse_reduction_vs_persistence"] = 1.0 - (
        patient["path_nrmse"] / patient["persistence_path_nrmse"]
    )
    patient["relative_path_swd_reduction_vs_persistence"] = 1.0 - (
        patient["path_sliced_wasserstein"]
        / patient["persistence_path_sliced_wasserstein"]
    )
    exact_count = patient["exact_lead_value_count"].to_numpy(dtype=float)
    exact_mean = patient["exact_lead_truth_sum"].to_numpy(dtype=float) / exact_count
    exact_variance = (
        patient["exact_lead_truth_squared_sum"].to_numpy(dtype=float) / exact_count
        - np.square(exact_mean)
    )
    exact_scale = np.sqrt(np.maximum(exact_variance, 0.0))
    patient["exact_lead_rmse"] = np.sqrt(
        patient["exact_lead_squared_error_sum"] / exact_count
    )
    patient["persistence_exact_lead_rmse"] = np.sqrt(
        patient["persistence_exact_lead_squared_error_sum"] / exact_count
    )
    patient["exact_lead_mae"] = patient["exact_lead_absolute_error_sum"] / exact_count
    patient["persistence_exact_lead_mae"] = (
        patient["persistence_exact_lead_absolute_error_sum"] / exact_count
    )
    patient["exact_lead_nrmse"] = np.where(
        exact_scale > np.finfo(float).eps,
        patient["exact_lead_rmse"] / exact_scale,
        np.nan,
    )
    patient["persistence_exact_lead_nrmse"] = np.where(
        exact_scale > np.finfo(float).eps,
        patient["persistence_exact_lead_rmse"] / exact_scale,
        np.nan,
    )
    patient["relative_exact_lead_nrmse_reduction_vs_persistence"] = 1.0 - (
        patient["exact_lead_nrmse"] / patient["persistence_exact_lead_nrmse"]
    )
    calibration = patient.copy()
    calibration["path_nrmse_not_worse_than_persistence"] = (
        calibration["path_nrmse"] <= calibration["persistence_path_nrmse"]
    )
    calibration["path_swd_not_worse_than_persistence"] = (
        calibration["path_sliced_wasserstein"]
        <= calibration["persistence_path_sliced_wasserstein"]
    )
    calibration["exact_lead_nrmse_not_worse_than_persistence"] = (
        calibration["exact_lead_nrmse"]
        <= calibration["persistence_exact_lead_nrmse"]
    )
    calibration["calibration_pass"] = (
        calibration["path_nrmse_not_worse_than_persistence"]
        & calibration["path_swd_not_worse_than_persistence"]
    )
    calibration["gate_interpretation"] = np.where(
        calibration["calibration_pass"],
        "eligible_for_distributional_prediction_claim_at_this_horizon",
        "fails_persistence_calibration_gate",
    )

    cohort_rows: list[dict[str, Any]] = []
    scopes = (
        (
            "primary_n7",
            patient[
                patient["generalization_scope"] == "held_out_seizure_generalization"
            ],
            7,
            "primary inference: held-out seizure patients only",
        ),
        (
            "all8_sensitivity",
            patient,
            8,
            "secondary sensitivity: includes HUP064 within-record temporal generalization",
        ),
    )
    for analysis_scope, scoped, target_n, interpretation in scopes:
        for (model_label, lead_end), group in scoped.groupby(
            ["model_label", "lead_end_samples"], sort=True
        ):
            effects = group[
                "relative_path_nrmse_reduction_vs_persistence"
            ].to_numpy(dtype=float)
            finite = effects[np.isfinite(effects)]
            exact_effects = group[
                "relative_exact_lead_nrmse_reduction_vs_persistence"
            ].to_numpy(dtype=float)
            exact_finite = exact_effects[np.isfinite(exact_effects)]
            if len(finite) == 0:
                continue
            seed_offset = 0 if analysis_scope == "primary_n7" else 100_000
            ci = bootstrap_confidence_interval(
                finite, n_resamples=10_000, seed=20260716 + seed_offset + int(lead_end)
            )
            sign = exact_paired_sign_flip(finite, alternative="two-sided")
            exact_ci = (
                bootstrap_confidence_interval(
                    exact_finite,
                    n_resamples=10_000,
                    seed=20260717 + seed_offset + int(lead_end),
                )
                if len(exact_finite)
                else None
            )
            cohort_rows.append(
                {
                    "analysis_scope": analysis_scope,
                    "scope_interpretation": interpretation,
                    "scope_target_n": target_n,
                    "model_label": model_label,
                    "matched_ablation": bool(group["matched_ablation"].iloc[0]),
                    "lead_end_samples": int(lead_end),
                    "lead_end_s": float(group["lead_end_s"].iloc[0]),
                    "n_patients": int(len(finite)),
                    "mean_relative_path_nrmse_reduction_vs_persistence": float(
                        np.mean(finite)
                    ),
                    "median_relative_path_nrmse_reduction_vs_persistence": float(
                        np.median(finite)
                    ),
                    "path_effect_ci_low": float(ci.low),
                    "path_effect_ci_high": float(ci.high),
                    "n_patients_path_improved": int(np.sum(finite > 0.0)),
                    "path_effect_p_exact_sign_flip": float(sign.pvalue),
                    "n_patients_exact_lead": int(len(exact_finite)),
                    "mean_relative_exact_lead_nrmse_reduction_vs_persistence": (
                        float(np.mean(exact_finite)) if len(exact_finite) else float("nan")
                    ),
                    "exact_lead_effect_ci_low": (
                        float(exact_ci.low) if exact_ci is not None else float("nan")
                    ),
                    "exact_lead_effect_ci_high": (
                        float(exact_ci.high) if exact_ci is not None else float("nan")
                    ),
                    "cross_seizure_only": analysis_scope == "primary_n7",
                    "hup064_within_record_included": bool(
                        (
                            group["generalization_scope"]
                            == "within_record_temporal_generalization"
                        ).any()
                    ),
                }
            )
    return patient, calibration, pd.DataFrame(cohort_rows)


def paired_ablation_summary(patient: pd.DataFrame) -> pd.DataFrame:
    """Patient-paired model contrasts, with matched topology removal primary."""

    comparisons = (
        (
            "matched_topology_increment",
            "rc_topology_delay_direct",
            MATCHED_TOPOLOGY_ABLATION_LABEL,
            "matched profile: reservoir/delay/ridge/seed fixed; only topology input removed",
        ),
        (
            "full_direct_vs_full_recursive_optimized",
            "rc_topology_delay_direct",
            "rc_topology_delay_recursive",
            "independently validation-optimized readout forms",
        ),
        (
            "delay_direct_vs_delay_recursive_optimized",
            "esn_delay_direct",
            "esn_delay_recursive",
            "independently validation-optimized readout forms",
        ),
        (
            "full_direct_vs_delay_direct_optimized",
            "rc_topology_delay_direct",
            "esn_delay_direct",
            "independently validation-optimized full versus delay-only models",
        ),
        (
            "full_direct_vs_delay_linear",
            "rc_topology_delay_direct",
            "delay_linear_direct",
            "independently validation-optimized nonlinear versus linear direct models",
        ),
    )
    scopes = (
        (
            "primary_n7",
            patient[
                patient["generalization_scope"] == "held_out_seizure_generalization"
            ],
            7,
        ),
        ("all8_sensitivity", patient, 8),
    )
    rows: list[dict[str, Any]] = []
    for scope, scoped, target_n in scopes:
        for comparison, target_label, reference_label, design in comparisons:
            target = scoped[scoped["model_label"] == target_label]
            reference = scoped[scoped["model_label"] == reference_label]
            paired = target.merge(
                reference,
                on=["subject", "lead_end_samples"],
                suffixes=("_target", "_reference"),
                validate="one_to_one",
            )
            if paired.empty:
                continue
            paired["path_effect"] = 1.0 - (
                paired["path_nrmse_target"] / paired["path_nrmse_reference"]
            )
            paired["exact_effect"] = 1.0 - (
                paired["exact_lead_nrmse_target"]
                / paired["exact_lead_nrmse_reference"]
            )
            paired["swd_effect"] = 1.0 - (
                paired["path_sliced_wasserstein_target"]
                / paired["path_sliced_wasserstein_reference"]
            )
            for lead_end, group in paired.groupby("lead_end_samples", sort=True):
                path_effect = group["path_effect"].to_numpy(dtype=float)
                exact_effect = group["exact_effect"].to_numpy(dtype=float)
                swd_effect = group["swd_effect"].to_numpy(dtype=float)
                path_effect = path_effect[np.isfinite(path_effect)]
                exact_effect = exact_effect[np.isfinite(exact_effect)]
                swd_effect = swd_effect[np.isfinite(swd_effect)]
                if not len(path_effect):
                    continue
                seed_offset = int(
                    _sha256_json([scope, comparison, int(lead_end)])[:8], 16
                )
                path_ci = bootstrap_confidence_interval(
                    path_effect, n_resamples=10_000, seed=seed_offset
                )
                path_sign = exact_paired_sign_flip(path_effect, alternative="two-sided")
                exact_ci = (
                    bootstrap_confidence_interval(
                        exact_effect, n_resamples=10_000, seed=seed_offset + 1
                    )
                    if len(exact_effect)
                    else None
                )
                swd_ci = (
                    bootstrap_confidence_interval(
                        swd_effect, n_resamples=10_000, seed=seed_offset + 2
                    )
                    if len(swd_effect)
                    else None
                )
                rows.append(
                    {
                        "analysis_scope": scope,
                        "scope_target_n": target_n,
                        "comparison": comparison,
                        "target_model": target_label,
                        "reference_model": reference_label,
                        "comparison_design": design,
                        "matched_ablation": comparison == "matched_topology_increment",
                        "lead_end_samples": int(lead_end),
                        "lead_end_s": float(group["lead_end_s_target"].iloc[0]),
                        "n_patients": int(len(path_effect)),
                        "mean_relative_path_nrmse_improvement_target_vs_reference": float(
                            np.mean(path_effect)
                        ),
                        "path_effect_ci_low": float(path_ci.low),
                        "path_effect_ci_high": float(path_ci.high),
                        "n_patients_path_target_better": int(np.sum(path_effect > 0)),
                        "path_effect_p_exact_sign_flip": float(path_sign.pvalue),
                        "mean_relative_exact_lead_nrmse_improvement_target_vs_reference": (
                            float(np.mean(exact_effect))
                            if len(exact_effect)
                            else float("nan")
                        ),
                        "exact_effect_ci_low": (
                            float(exact_ci.low) if exact_ci is not None else float("nan")
                        ),
                        "exact_effect_ci_high": (
                            float(exact_ci.high) if exact_ci is not None else float("nan")
                        ),
                        "mean_relative_path_swd_improvement_target_vs_reference": (
                            float(np.mean(swd_effect))
                            if len(swd_effect)
                            else float("nan")
                        ),
                        "swd_effect_ci_low": (
                            float(swd_ci.low) if swd_ci is not None else float("nan")
                        ),
                        "swd_effect_ci_high": (
                            float(swd_ci.high) if swd_ci is not None else float("nan")
                        ),
                        "exploratory_multiple_comparison_status": (
                            "exploratory; p-values are descriptive and unadjusted"
                        ),
                        "hup064_within_record_included": scope == "all8_sensitivity",
                    }
                )
    return pd.DataFrame(rows)


def calibration_scope_summary(calibration: pd.DataFrame) -> pd.DataFrame:
    """Count patient-level persistence gates separately for n=7 and all-eight."""

    rows: list[dict[str, Any]] = []
    scopes = (
        (
            "primary_n7",
            calibration[
                calibration["generalization_scope"]
                == "held_out_seizure_generalization"
            ],
            7,
        ),
        ("all8_sensitivity", calibration, 8),
    )
    for scope, scoped, target_n in scopes:
        for (model_label, lead_end), group in scoped.groupby(
            ["model_label", "lead_end_samples"], sort=True
        ):
            n_patients = int(group["subject"].nunique())
            rows.append(
                {
                    "analysis_scope": scope,
                    "scope_target_n": target_n,
                    "model_label": model_label,
                    "lead_end_samples": int(lead_end),
                    "lead_end_s": float(group["lead_end_s"].iloc[0]),
                    "n_patients": n_patients,
                    "calibration_pass_count": int(group["calibration_pass"].sum()),
                    "calibration_pass_fraction": float(group["calibration_pass"].mean()),
                    "path_nrmse_not_worse_count": int(
                        group["path_nrmse_not_worse_than_persistence"].sum()
                    ),
                    "path_swd_not_worse_count": int(
                        group["path_swd_not_worse_than_persistence"].sum()
                    ),
                    "exact_lead_nrmse_not_worse_count": int(
                        group["exact_lead_nrmse_not_worse_than_persistence"].sum()
                    ),
                    "hup064_within_record_included": scope == "all8_sensitivity",
                }
            )
    return pd.DataFrame(rows)


def expected_window_rows(config: dict[str, Any], quick: bool) -> int:
    windows = 2 if quick else int(config["prediction"]["windows_per_test_run"])
    return len(MODEL_LABELS) * len(config["prediction"]["horizons_samples"]) * windows


def subject_complete(
    window_path: Path,
    selected_config_path: Path,
    subject: str,
    config: dict[str, Any],
    quick: bool,
    seed: int,
    fingerprints: dict[str, str],
) -> bool:
    """Validate the full Cartesian output, fingerprints, and saved model hashes."""

    frame = _read_frame(window_path)
    selected_configs = _read_frame(selected_config_path)
    if frame.empty or selected_configs.empty:
        return False
    # Legacy/partial outputs must be treated as incomplete rather than raising
    # during provenance generation.  Both tables carry the explicit technical
    # seed so a resumed run cannot silently mix analyses.
    if not {"subject", "seed"} <= set(frame.columns) or not {
        "subject",
        "seed",
    } <= set(selected_configs.columns):
        return False
    rows = frame[
        (frame["subject"].astype(str) == str(subject))
        & (pd.to_numeric(frame["seed"], errors="coerce") == int(seed))
    ]
    if len(rows) != expected_window_rows(config, quick):
        return False
    required_window_columns = {
        "model_label",
        "lead_end_samples",
        "window",
        "split_windows_driven_by_schema2_json",
        *fingerprints.keys(),
    }
    if not required_window_columns <= set(rows.columns):
        return False
    window_count = 2 if quick else int(config["prediction"]["windows_per_test_run"])
    expected_cartesian = {
        (model_label, int(lead), window)
        for model_label in MODEL_LABELS
        for lead in config["prediction"]["horizons_samples"]
        for window in range(window_count)
    }
    observed_cartesian = {
        (str(row.model_label), int(row.lead_end_samples), int(row.window))
        for row in rows.itertuples(index=False)
    }
    if (
        rows.duplicated(["model_label", "lead_end_samples", "window"]).any()
        or observed_cartesian != expected_cartesian
    ):
        return False
    split_driven = rows["split_windows_driven_by_schema2_json"].astype(str).str.lower()
    if not split_driven.isin({"true", "1"}).all():
        return False
    for column, expected in fingerprints.items():
        if not rows[column].astype(str).eq(str(expected)).all():
            return False

    models = selected_configs[
        (selected_configs["subject"].astype(str) == str(subject))
        & (pd.to_numeric(selected_configs["seed"], errors="coerce") == int(seed))
    ]
    required_model_columns = {
        "model_label",
        "seed",
        "model_path",
        "model_sha256",
        "selected_without_final_test",
        "test_metrics_present_in_selection_table",
        *fingerprints.keys(),
    }
    if (
        len(models) != len(MODEL_LABELS)
        or not required_model_columns <= set(models.columns)
        or models["model_label"].duplicated().any()
        or set(models["model_label"].astype(str)) != set(MODEL_LABELS)
    ):
        return False
    if not models["selected_without_final_test"].astype(str).str.lower().isin(
        {"true", "1"}
    ).all():
        return False
    if models["test_metrics_present_in_selection_table"].astype(str).str.lower().isin(
        {"true", "1"}
    ).any():
        return False
    for column, expected in fingerprints.items():
        if not models[column].astype(str).eq(str(expected)).all():
            return False
    for row in models.itertuples(index=False):
        model_path = Path(str(row.model_path))
        expected_hash = str(row.model_sha256).lower()
        if (
            len(expected_hash) != 64
            or not model_path.is_file()
            or _sha256_file(model_path).lower() != expected_hash
        ):
            return False
    return True


def write_provenance(
    output_root: Path,
    config_path: Path,
    config: dict[str, Any],
    split_proposal_metadata: dict[str, str],
    requested_subjects: list[str],
    seed: int,
    quick: bool,
    started: float,
    fingerprints: dict[str, str],
) -> None:
    window_path = output_root / "window_metrics.csv"
    selected_path = output_root / "selected_configs.csv"
    completed = [
        subject
        for subject in requested_subjects
        if subject_complete(
            window_path,
            selected_path,
            subject,
            config,
            quick,
            seed,
            fingerprints,
        )
    ]
    outputs = []
    for path in sorted(output_root.glob("*.csv")):
        outputs.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "rc_v2_schema_version": RC_V2_SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_status": "complete" if set(completed) == set(requested_subjects) else "partial",
        "subjects_requested": requested_subjects,
        "subjects_completed": completed,
        "seed": int(seed),
        "quick": bool(quick),
        "analysis_fingerprints": dict(fingerprints),
        "elapsed_seconds_this_invocation": float(time.perf_counter() - started),
        "claim_boundary": (
            "Primary inference is ictal-only held-out-seizure prediction in seven patients. "
            "HUP064 is reported separately and only in the all-eight within-record sensitivity; "
            "no result is clinical efficacy or cross-patient evidence."
        ),
        "leakage_barriers": {
            "preictal_signal_loaded": False,
            "interictal_signal_loaded": False,
            "final_test_used_for_candidate_selection": False,
            "selection": "seizure-blocked or prespecified chronological blocked validation",
            "event_qc_split_proposal_enforced": True,
            "split_windows_driven_by_schema2_json": True,
            "only_eligible_ictal_runs_loaded": True,
            "preprocessing_fully_causal": True,
            "resampling_mode": str(config["data"]["resampling_mode"]),
            "causal_fir_half_length_factor": int(
                config["data"]["causal_fir_half_length_factor"]
            ),
            "forecast_guard_samples": int(config["data"]["forecast_guard_samples"]),
            "final_seizure_isolated_for_multiseizure_patients": True,
        },
        "compact_candidate_policy": {
            "description": (
                "Two deterministic diagonal reservoir profiles plus all configured ridge/delay "
                "linear candidates; seed 11 main analysis."
            ),
            "network_density": NETWORK_DENSITY,
            "plv_window_s": PLV_WINDOW_S,
            "plv_overlap": PLV_OVERLAP,
            "validation_windows": 2 if quick else VALIDATION_WINDOWS,
        },
        "inputs": {
            "config": {"path": str(config_path.resolve()), "sha256": _sha256_file(config_path)},
            "patient_split_proposal": dict(split_proposal_metadata),
            "module": {
                "path": str((PROJECT_ROOT / "mfc_pipeline" / "rc_v2.py").resolve()),
                "sha256": _sha256_file(PROJECT_ROOT / "mfc_pipeline" / "rc_v2.py"),
            },
            "data_loader": {
                "path": str((PROJECT_ROOT / "mfc_pipeline" / "data.py").resolve()),
                "sha256": _sha256_file(PROJECT_ROOT / "mfc_pipeline" / "data.py"),
                "cache_schema_version": CACHE_SCHEMA_VERSION,
            },
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__)),
            },
            "zip_root": str(config["data"]["zip_root"]),
        },
        "outputs": outputs,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "pyyaml": getattr(yaml, "__version__", "unknown"),
            "platform": platform.platform(),
        },
    }
    _atomic_json(output_root / "provenance.json", payload)


def main() -> int:
    args = parse_args()
    invocation_started = time.perf_counter()
    config_path = args.config.resolve()
    config = load_config(config_path)
    split_proposals, split_proposal_metadata = load_split_proposal(
        args.split_proposal.resolve(), config
    )
    fingerprints = analysis_fingerprints(config_path, split_proposal_metadata)
    configured = [str(value) for value in config["data"]["subjects"]]
    subjects = configured if args.subjects is None else [str(value) for value in args.subjects]
    unknown = sorted(set(subjects) - set(configured))
    if unknown:
        raise ValueError(f"subjects are not configured: {unknown}")
    if int(args.seed) not in [int(value) for value in config["project"]["random_seeds"]]:
        raise ValueError("seed is not pre-specified in config_v2.yaml")
    output_root = resolve_project_path(config["project"]["output_dir"]) / "prediction"
    if args.quick:
        output_root = output_root / "quick"
    output_root.mkdir(parents=True, exist_ok=True)
    setup_logging(output_root)
    dataset, loader = build_loader(config)
    specs = candidate_specs(config, int(args.seed), bool(args.quick))
    window_path = output_root / "window_metrics.csv"
    selected_config_path = output_root / "selected_configs.csv"

    write_provenance(
        output_root,
        config_path,
        config,
        split_proposal_metadata,
        subjects,
        int(args.seed),
        bool(args.quick),
        invocation_started,
        fingerprints,
    )
    for subject in subjects:
        if not args.force and subject_complete(
            window_path,
            selected_config_path,
            subject,
            config,
            bool(args.quick),
            int(args.seed),
            fingerprints,
        ):
            LOGGER.info("%s complete/current, skipping", subject)
            continue
        subject_started = time.perf_counter()
        plan = load_ictal_plan(
            config,
            dataset,
            loader,
            subject,
            split_proposals[subject],
            split_proposal_metadata["sha256"],
            fingerprints,
        )
        validation, selected, selected_lookup = validate_candidates(
            plan,
            specs,
            config["prediction"],
            bool(args.quick),
        )
        selected_final, windows = final_fit_and_test(
            plan,
            selected,
            selected_lookup,
            config["prediction"],
            output_root,
            bool(args.quick),
        )
        if len(windows) != expected_window_rows(config, bool(args.quick)):
            raise RuntimeError(
                f"{subject}: expected {expected_window_rows(config, bool(args.quick))} "
                f"test rows, observed {len(windows)}"
            )
        patient, calibration, cohort_unused = patient_and_calibration(windows)
        merge_subject_frame(output_root / "candidate_validation.csv", subject, validation)
        merge_subject_frame(output_root / "selected_configs.csv", subject, selected_final)
        merge_subject_frame(window_path, subject, windows)
        merge_subject_frame(output_root / "patient_summary.csv", subject, patient)
        merge_subject_frame(output_root / "calibration.csv", subject, calibration)
        LOGGER.info(
            "%s complete in %.2fs (%s)",
            subject,
            time.perf_counter() - subject_started,
            plan["generalization_scope"],
        )
        all_windows = _read_frame(window_path)
        all_patient, all_calibration, all_cohort = patient_and_calibration(all_windows)
        _atomic_frame(output_root / "patient_summary.csv", all_patient)
        _atomic_frame(output_root / "calibration.csv", all_calibration)
        _atomic_frame(output_root / "cohort_summary.csv", all_cohort)
        _atomic_frame(
            output_root / "paired_ablation_summary.csv",
            paired_ablation_summary(all_patient),
        )
        _atomic_frame(
            output_root / "calibration_scope_summary.csv",
            calibration_scope_summary(all_calibration),
        )
        write_provenance(
            output_root,
            config_path,
            config,
            split_proposal_metadata,
            subjects,
            int(args.seed),
            bool(args.quick),
            invocation_started,
            fingerprints,
        )

    incomplete = [
        subject
        for subject in subjects
        if not subject_complete(
            window_path,
            selected_config_path,
            subject,
            config,
            bool(args.quick),
            int(args.seed),
            fingerprints,
        )
    ]
    all_windows = _read_frame(window_path)
    if not all_windows.empty:
        all_patient, all_calibration, all_cohort = patient_and_calibration(all_windows)
        _atomic_frame(output_root / "patient_summary.csv", all_patient)
        _atomic_frame(output_root / "calibration.csv", all_calibration)
        _atomic_frame(output_root / "cohort_summary.csv", all_cohort)
        _atomic_frame(
            output_root / "paired_ablation_summary.csv",
            paired_ablation_summary(all_patient),
        )
        _atomic_frame(
            output_root / "calibration_scope_summary.csv",
            calibration_scope_summary(all_calibration),
        )
    write_provenance(
        output_root,
        config_path,
        config,
        split_proposal_metadata,
        subjects,
        int(args.seed),
        bool(args.quick),
        invocation_started,
        fingerprints,
    )
    if incomplete:
        raise RuntimeError(f"V2 prediction output is incomplete for {incomplete}")
    LOGGER.info("V2 prediction complete: %s", output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
