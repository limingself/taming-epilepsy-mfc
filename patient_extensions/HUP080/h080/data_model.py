from __future__ import annotations

import csv
from dataclasses import asdict
from hashlib import sha256
import importlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

from .contracts import (
    context_positions,
    deterministic_seed,
    sha256_file,
    sha256_json,
    validate_zip_central_directory_contract,
    zip_central_directory_receipt,
)
from .staging import atomic_json


def _purge_mfc_pipeline() -> None:
    for name in tuple(sys.modules):
        if name == "mfc_pipeline" or name.startswith("mfc_pipeline."):
            del sys.modules[name]


def activate_canonical(config: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    """Idempotently import only the hash-pinned D canonical module identities."""

    root = Path(str(config["canonical_locks"]["part2_core"][0])).parents[1]
    bindings = {
        "mfc_pipeline.data": config["canonical_locks"]["data_core"],
        "mfc_pipeline.network": config["canonical_locks"]["network_core"],
        "mfc_pipeline.part2_state_dependent_rc_sde": config["canonical_locks"]["part2_core"],
        "mfc_pipeline.part2_data_pipeline": config["canonical_locks"]["part2_data_pipeline"],
    }

    def valid(module_name: str) -> bool:
        module = sys.modules.get(module_name)
        if module is None or getattr(module, "__file__", None) is None:
            return False
        expected_path, expected_sha256 = bindings[module_name]
        observed = Path(str(module.__file__)).resolve()
        return (
            os.path.normcase(str(observed))
            == os.path.normcase(str(Path(expected_path).resolve()))
            and sha256_file(observed) == str(expected_sha256)
        )

    loaded_required = [name for name in bindings if name in sys.modules]
    invalid_loaded = [name for name in loaded_required if not valid(name)]
    if invalid_loaded:
        raise PermissionError(
            f"loaded canonical module has a foreign path/hash: {invalid_loaded}"
        )
    parent = sys.modules.get("mfc_pipeline")
    if parent is not None and getattr(parent, "__file__", None) is not None:
        parent_file = Path(str(parent.__file__)).resolve()
        expected_package = (root / "mfc_pipeline").resolve()
        try:
            parent_file.relative_to(expected_package)
        except ValueError:
            if loaded_required:
                raise PermissionError(
                    "cannot replace a foreign mfc_pipeline while live canonical objects may exist"
                )
            _purge_mfc_pipeline()
    sys.path[:] = [
        item for item in sys.path
        if os.path.normcase(item) != os.path.normcase(str(root))
    ]
    sys.path.insert(0, str(root))
    for module_name in bindings:
        if module_name not in sys.modules:
            importlib.import_module(module_name)
    if not all(valid(name) for name in bindings):
        raise ImportError("D-canonical data/network/Part-II module binding failed")
    data = sys.modules["mfc_pipeline.data"]
    network = sys.modules["mfc_pipeline.network"]
    part2 = sys.modules["mfc_pipeline.part2_state_dependent_rc_sde"]
    part2_pipeline = sys.modules["mfc_pipeline.part2_data_pipeline"]
    return data, network, part2, part2_pipeline


def assert_live_part2_model_identity(
    model: Any, config: Mapping[str, Any] | None = None,
) -> None:
    """Reject stale model instances after any mfc_pipeline module replacement."""

    module_name = "mfc_pipeline.part2_state_dependent_rc_sde"
    module = sys.modules.get(module_name)
    if module is None or model.__class__ is not getattr(
        module, "ResidualGraphRCSDE", None
    ):
        raise PermissionError("Part-II model has a stale or foreign class identity")
    if config is not None:
        expected_path, expected_sha256 = config["canonical_locks"]["part2_core"]
        observed = Path(str(module.__file__)).resolve()
        if os.path.normcase(str(observed)) != os.path.normcase(
            str(Path(expected_path).resolve())
        ) or sha256_file(observed) != str(expected_sha256):
            raise PermissionError("live Part-II class is not the pinned D canonical source")


def _run_key(prefix: str, run: str) -> str:
    return f"{prefix}_{str(run).replace('-', '_')}"


def parse_hup080_split(config: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(config["source_data"]["split_proposal"]))
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("subject") == "HUP080"]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one HUP080 split row, observed {len(rows)}")
    row = rows[0]
    if row.get("schema_version") != "2" or row.get("split_status") != "ready":
        raise ValueError("HUP080 split row is not ready schema-v2")
    development = tuple(item for item in row["ictal_train_runs"].split(";") if item)
    if development != tuple(config["source_data"]["development_runs"]):
        raise PermissionError("split-proposal development runs changed")
    if row["ictal_test_run"] != config["source_data"]["sealed_run"]:
        raise PermissionError("split-proposal sealed run changed")
    folds = json.loads(row["ictal_inner_validation_folds_json"])
    refit = json.loads(row["ictal_final_refit_windows_json"])
    ictal_test = json.loads(row["ictal_test_window_json"])
    references = json.loads(row["reference_train_windows_json"])
    reference_test = json.loads(row["reference_test_window_json"])
    if [item["fold_id"] for item in folds] != [
        "leave-run-01-out", "leave-run-02-out", "leave-run-03-out"
    ]:
        raise PermissionError("HUP080 LORO fold identities changed")
    for fold in folds:
        validation = fold["validation_window"]
        if [float(validation["start_s"]), float(validation["stop_s"])] != [120.0, 152.0]:
            raise PermissionError("LORO validation ictal window changed")
        for window in fold["training_windows"]:
            if [float(window["start_s"]), float(window["stop_s"])] != [120.0, 152.0]:
                raise PermissionError("LORO training ictal window changed")
    if [item["run"] for item in references] != list(development):
        raise PermissionError("development reference run order changed")
    for item in references:
        if [float(item["start_s"]), float(item["stop_s"])] != [30.0, 70.0]:
            raise PermissionError("development reference window changed")
    if reference_test != {
        "duration_s": 40.0,
        "role": "reference_evaluation_only_after_model_freeze",
        "run": "run-04",
        "start_s": 75.0,
        "stop_s": 115.0,
    }:
        raise PermissionError("sealed reference-test role changed")
    if ictal_test.get("run") != "run-04" or [
        float(ictal_test["start_s"]), float(ictal_test["stop_s"])
    ] != [120.0, 152.0]:
        raise PermissionError("sealed ictal-test role changed")
    if row.get("reference_label") != "preictal_reference":
        raise PermissionError("reference label is not seizure-pre")
    if row.get("reference_isolation_level") != "seizure-level":
        raise PermissionError("reference isolation is not seizure-level")
    if "never use reference_test_window" not in row.get("controller_target_rule", ""):
        raise PermissionError("controller target freeze rule changed")
    return {
        "row": row,
        "folds": folds,
        "final_refit_windows": refit,
        "ictal_test_window": ictal_test,
        "reference_train_windows": references,
        "reference_test_window": reference_test,
    }


def prepare_development(config: Mapping[str, Any], output: Path) -> None:
    """Read only run-01..03 ictal EDFs and their seizure-pre blocks."""

    import numpy as np

    # Re-check metadata immediately before any selected development member is
    # opened. This never hashes the archive or opens a ZIP member payload.
    zip_receipt = zip_central_directory_receipt(
        Path(str(config["source_data"]["raw_zip"]))
    )
    validate_zip_central_directory_contract(config, zip_receipt)
    data_module, _network_module, _part2, _part2_pipeline = activate_canonical(config)
    split = parse_hup080_split(config)
    data_cfg = config["source_data"]
    prep = config["preprocessing"]
    runs = tuple(data_cfg["development_runs"])
    dataset = data_module.BIDSZipDataset(data_cfg["zip_root"], subjects=["HUP080"])
    records = dataset.selected_records("HUP080", [("ictal", run) for run in runs])
    if tuple(record.run for record in records) != runs:
        raise PermissionError("selected development EDF order differs from the frozen role ledger")
    if any(record.task.casefold() != "ictal" or record.run == "run-04" for record in records):
        raise PermissionError("non-development or non-ictal record entered preparation")
    channels = tuple(dataset.common_good_channels("HUP080", records=records))
    if len(channels) != int(data_cfg["expected_channels"]):
        raise ValueError(f"expected 96 common-good channels, observed {len(channels)}")
    runtime = Path(str(config["runtime_root"]))
    loader = data_module.EEGSegmentLoader(
        dataset,
        cache_dir=runtime / "fresh_cache_never_reused",
        temp_dir=runtime / "edf_tmp",
        target_sfreq=float(prep["target_sampling_rate_hz"]),
        bandpass_hz=tuple(prep["bandpass_hz"]),
        filter_order=int(prep["filter_order"]),
        burn_in_s=float(prep["filter_burn_in_s"]),
        reference=str(prep["reference"]),
        resampling_mode=str(prep["resampling_mode"]),
        causal_fir_half_length_factor=int(prep["causal_fir_half_length_factor"]),
    )
    arrays: dict[str, Any] = {
        "channels": np.asarray(channels, dtype=str),
        "sfreq": np.asarray(float(prep["target_sampling_rate_hz"]), dtype=np.float64),
    }
    run_rows: list[dict[str, Any]] = []
    for record in records:
        metadata = dataset.metadata(record)
        if not np.isclose(float(metadata.onset_s), 120.0):
            raise PermissionError(f"{record.run}: onset changed from 120 s")
        ictal = loader.load(
            record, start_s=120.0, duration_s=32.0,
            channel_names=channels, use_cache=False,
        )
        reference = loader.load(
            record, start_s=30.0, duration_s=40.0,
            channel_names=channels, use_cache=False,
        )
        for role, segment, samples in (
            ("ictal", ictal, 32 * 256), ("preictal_reference", reference, 40 * 256)
        ):
            values = np.asarray(segment.data.T, dtype=np.float64)
            if values.shape != (samples, 96) or not np.isfinite(values).all():
                raise ValueError(f"{record.run}/{role}: invalid processed shape or values")
            provenance = dict(segment.provenance)
            if not provenance.get("preprocessing_fully_causal") or not provenance.get("resampling_causal"):
                raise PermissionError(f"{record.run}/{role}: preprocessing is not fully causal")
            arrays[_run_key(role, record.run)] = values
            run_rows.append(
                {
                    "run": record.run,
                    "role": role,
                    "task": record.task,
                    "absolute_window_s": [float(segment.start_s), float(segment.start_s + segment.duration_s)],
                    "shape": list(values.shape),
                    "edf_member": record.edf_member,
                    "edf_crc": int(record.edf_crc),
                    "edf_size_bytes": int(record.edf_size_bytes),
                    "array_sha256": array_sha256(values),
                    "preprocessing": provenance,
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "development_arrays.npz", **arrays)
    positions = context_positions(32 * 256)
    atomic_json(
        output / "role_ledger.json",
        {
            "subject": "HUP080",
            "opened_signal_runs": list(runs),
            "sealed_signal_run": "run-04",
            "run04_signal_opened": False,
            "task_interictal_opened": False,
            "reference_label": "preictal_reference",
            "development_reference_windows": {run: [30.0, 70.0] for run in runs},
            "development_ictal_windows": {run: [120.0, 152.0] for run in runs},
            "context_positions_samples": list(positions),
            "context_roles": config["contexts"],
            "fold_reference_rule": "each fold fits its target only from the other two seizures",
            "split_proposal_sha256": config["source_data"]["split_proposal_sha256"],
        },
    )
    atomic_json(
        output / "provenance.json",
        {
            "source_archive": str(Path(data_cfg["raw_zip"]).resolve()),
            "phase01_central_directory_metadata_sha256": zip_receipt[
                "central_directory_metadata_sha256"
            ],
            "phase01_member_payload_opened_by_metadata_check": False,
            "historical_whole_zip_sha256_nonselection_receipt": data_cfg["historical_whole_zip_sha256_nonselection_receipt"],
            "whole_archive_sha256_verified": False,
            "reason": "whole-ZIP hashing is never an executable gate; central-directory metadata and selected development member CRC/size are pinned",
            "channels": list(channels),
            "channel_count": len(channels),
            "analysis_sampling_rate_hz": 256.0,
            "records": run_rows,
            "split_contract": {
                "folds": split["folds"],
                "final_refit_windows": split["final_refit_windows"],
                "reference_train_windows": split["reference_train_windows"],
                "sealed_windows_parsed_but_signal_not_opened": True,
            },
        },
    )


def array_sha256(values: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(values)
    digest = sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def load_development(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(Path(path), allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def get_run(data: Mapping[str, Any], role: str, run: str) -> Any:
    return data[_run_key(role, run)]


def bandpass_for_plv(values: Any, sfreq: float, band: Sequence[float]) -> Any:
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    low, high = map(float, band)
    nyquist = 0.5 * float(sfreq)
    high = min(high, 0.98 * nyquist)
    sos = butter(4, [low / nyquist, high / nyquist], btype="band", output="sos")
    return np.asarray(sosfiltfilt(sos, values, axis=0), dtype=np.float64)


def build_part1_selection_network(
    config: Mapping[str, Any], sequences: Sequence[Any], sfreq: float,
    *, canonical_network_module: Any | None = None,
) -> dict[str, Any]:
    """Part-I multiband 4-s graph used only for centrality and sparse mask."""
    import numpy as np

    canonical = canonical_network_module
    if canonical is None:
        _data, canonical, _part2, _part2_pipeline = activate_canonical(config)
    spec = config["part1"]
    by_band: dict[str, Any] = {}
    for name, band in spec["bands_hz"].items():
        matrices = [
            canonical.windowed_plv_median(
                bandpass_for_plv(np.asarray(sequence), sfreq, band),
                sfreq,
                window_s=float(spec["plv_window_s"]),
                overlap=float(spec["plv_overlap"]),
                time_axis=0,
            )
            for sequence in sequences
        ]
        by_band[name] = np.median(np.stack(matrices), axis=0)
    similarity = np.mean(np.stack(list(by_band.values())), axis=0)
    adjacency = canonical.mst_proportional_graph(
        similarity, density=float(spec["network_density"])
    )
    centrality = canonical.centrality_composite(
        adjacency, weights={key: float(value) for key, value in spec["centrality_weights"].items()}
    )
    score = np.asarray(centrality["composite"], dtype=np.float64)
    threshold = float(np.quantile(score, 0.65, method="linear"))
    degree = adjacency.sum(axis=1)
    result = {
        "network_role": np.asarray("part1_selection_only"),
        "adjacency": np.asarray(adjacency, dtype=np.float64),
        "laplacian": np.diag(degree) - adjacency,
        "plv_multiband_equal": similarity,
        "centrality_score": score,
        "degree_centrality": np.asarray(centrality["degree"]),
        "betweenness_centrality": np.asarray(centrality["betweenness"]),
        "eigenvector_centrality": np.asarray(centrality["eigenvector"]),
        "target_mask": score > threshold,
        "centrality_threshold_q065": np.asarray(threshold),
    }
    result.update({f"plv_{name}": matrix for name, matrix in by_band.items()})
    return result


def build_part2_plant_network(
    config: Mapping[str, Any], sequences: Sequence[Any], sfreq: float,
    *, canonical_part2_pipeline_module: Any | None = None,
) -> dict[str, Any]:
    """D Part-II broadband 2-s PLV graph used only by the Graph-RC plant."""

    import numpy as np

    canonical_pipeline = canonical_part2_pipeline_module
    if canonical_pipeline is None:
        _data, _network, _part2, canonical_pipeline = activate_canonical(config)
    spec = config["part2"]
    if bool(spec["part1_graph_may_be_reused_as_plant_graph"]):
        raise PermissionError("Part-I selection graph cannot be reused by Part-II")
    adjacency = np.asarray(
        canonical_pipeline.compute_adjacency(
            [np.asarray(sequence, dtype=np.float64) for sequence in sequences],
            float(sfreq),
        ),
        dtype=np.float64,
    )
    expected_channels = int(np.asarray(sequences[0]).shape[1])
    if adjacency.shape != (expected_channels, expected_channels):
        raise RuntimeError("D Part-II plant adjacency has an invalid channel shape")
    if not np.isfinite(adjacency).all():
        raise RuntimeError("D Part-II plant adjacency contains non-finite values")
    degree = adjacency.sum(axis=1)
    return {
        "network_role": np.asarray("part2_plant_only"),
        "adjacency": np.asarray(adjacency, dtype=np.float64),
        "laplacian": np.diag(degree) - adjacency,
        "plv_window_s": np.asarray(float(spec["plant_plv_window_s"])),
        "plv_overlap": np.asarray(float(spec["plant_plv_overlap"])),
    }


def save_network(path: Path, network: Mapping[str, Any], channels: Sequence[str]) -> None:
    import numpy as np

    payload = {key: np.asarray(value) for key, value in network.items()}
    payload["channels"] = np.asarray(channels, dtype=str)
    payload["clinical_labels_used_for_selection"] = np.asarray(False)
    np.savez_compressed(path, **payload)


def load_network(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def part3_context_block(sequence: Any, index: int) -> tuple[Any, Any, int]:
    """Frozen Part-III 256-sample history and 256-sample control horizon."""

    positions = context_positions(len(sequence))
    boundary = positions[int(index)]
    return sequence[boundary - 256 : boundary].copy(), sequence[boundary : boundary + 256].copy(), boundary


def part2_context_block(
    sequence: Any, index: int, forecast_positions_fn: Any,
) -> tuple[Any, Any, int]:
    """D-canonical Part-II 512-sample history and 128-sample validation future."""

    positions = tuple(int(value) for value in forecast_positions_fn(
        len(sequence), context_samples=512, maximum_horizon=128,
        guard_samples=0, count=6,
    ))
    if len(positions) != 6:
        raise RuntimeError("Part-II forecast position construction lost uniqueness")
    boundary = positions[int(index)]
    context = sequence[boundary - 512 : boundary].copy()
    future = sequence[boundary : boundary + 128].copy()
    if context.shape[0] != 512 or future.shape[0] != 128:
        raise ValueError("sequence cannot supply canonical Part-II context/future")
    return context, future, boundary


def reference_paths(reference: Any) -> Any:
    import numpy as np

    values = np.asarray(reference, dtype=np.float64)
    if values.shape[0] % 256:
        raise ValueError("reference length is not an integer number of 1-s paths")
    return values.reshape(values.shape[0] // 256, 256, values.shape[1])


def part2_grid(config: Mapping[str, Any], part2_module: Any) -> list[Any]:
    spec = config["part2"]
    grid = spec["grid"]
    common = {
        "reservoir_sparsity": float(spec["reservoir_sparsity"]),
        "washout_samples": int(spec["washout_samples"]),
        "random_seed": int(spec["random_seed"]),
        "latent_variance_threshold": float(spec["latent_variance_threshold"]),
        "latent_components_min": int(spec["latent_components_min"]),
        "latent_components_max": int(spec["latent_components_max"]),
        "diffusion_shrinkage": float(spec["diffusion_shrinkage"]),
        "diffusion_relative_jitter": float(spec["diffusion_relative_jitter"]),
        "diffusion_mode": str(spec["diffusion_mode"]),
        "diffusion_ridge_alpha": 10.0,
    }
    candidates = [
        part2_module.ModelConfig(
            reservoir_size=int(size), spectral_radius=float(radius),
            leak_rate=float(leak), input_scale=float(scale),
            delays_samples=tuple(int(value) for value in delays),
            ridge_alpha=float(ridge), **common,
        )
        for size, radius, leak, scale, delays, ridge in itertools.product(
            grid["reservoir_sizes"], grid["spectral_radii"], grid["leak_rates"],
            grid["input_scales"], grid["delay_sets_samples"], grid["ridge_alphas"],
        )
    ]
    if len(candidates) != 144:
        raise AssertionError("canonical Part-II grid is not 144 candidates")
    return candidates


def _score_part2_candidate(
    model: Any,
    validation: Any,
    context_indices: Sequence[int],
    part2_module: Any,
    seed: int,
) -> tuple[float, list[dict[str, Any]]]:
    import numpy as np

    rows: list[dict[str, Any]] = []
    projections = part2_module.fixed_random_projections(
        validation.shape[1], n_projections=128, seed=int(seed)
    )
    for context_index in context_indices:
        context, future, boundary = part2_context_block(
            validation, int(context_index), part2_module.forecast_positions,
        )
        forecast = model.forecast_direct(context, 128)
        persistence = np.repeat(context[-1:], 128, axis=0)
        for horizon in (16, 32, 64, 128):
            truth = future[:horizon]
            pred = forecast[:horizon]
            base = persistence[:horizon]
            metrics = part2_module.trajectory_metrics(truth, pred, model.transform.scaler)
            base_metrics = part2_module.trajectory_metrics(truth, base, model.transform.scaler)
            truth_scaled = model.transform.scaler.transform(truth)
            pred_scaled = model.transform.scaler.transform(pred)
            base_scaled = model.transform.scaler.transform(base)
            swd = part2_module.safe_swd(pred_scaled, truth_scaled, projections)
            persistence_swd = part2_module.safe_swd(base_scaled, truth_scaled, projections)
            rows.append(
                {
                    "context_index": int(context_index),
                    "boundary_sample": int(boundary),
                    "horizon_samples": int(horizon),
                    "nrmse": float(metrics["nrmse"]),
                    "persistence_nrmse": float(base_metrics["nrmse"]),
                    "swd": float(swd),
                    "persistence_swd": float(persistence_swd),
                    "swd_ratio_to_persistence": float(swd / max(persistence_swd, 1e-12)),
                }
            )
    weights = {16: 0.40, 32: 0.35, 64: 0.15}
    means = {
        horizon: sum(row["nrmse"] for row in rows if row["horizon_samples"] == horizon)
        / sum(1 for row in rows if row["horizon_samples"] == horizon)
        for horizon in (16, 32, 64)
    }
    swd128 = [row["swd_ratio_to_persistence"] for row in rows if row["horizon_samples"] == 128]
    objective = sum(weights[h] * means[h] for h in weights) + 0.10 * float(np.mean(swd128))
    return float(objective), rows


def _rolling_block_metrics(
    model: Any, sequence: Any, positions: Sequence[int], *,
    context_samples: int, block_samples: int, total_samples: int,
    channels: Sequence[str],
) -> list[dict[str, Any]]:
    """D Part-II rolling direct forecast, reduced to independent window/channel rows."""

    import numpy as np

    if model.transform is None:
        raise RuntimeError("rolling screen received an unfitted Part-II model")
    rows: list[dict[str, Any]] = []
    for window, position in enumerate(positions):
        blocks = []
        for start in range(0, int(total_samples), int(block_samples)):
            boundary = int(position) + start
            context = sequence[boundary - int(context_samples) : boundary]
            steps = min(int(block_samples), int(total_samples) - start)
            blocks.append(model.forecast_direct(context, steps))
        prediction = np.vstack(blocks)
        truth = sequence[int(position) : int(position) + int(total_samples)]
        truth_scaled = model.transform.scaler.transform(truth)
        prediction_scaled = model.transform.scaler.transform(prediction)
        for channel_index, channel in enumerate(channels):
            observed = truth_scaled[:, channel_index]
            estimated = prediction_scaled[:, channel_index]
            observed_sd = float(np.std(observed))
            estimated_sd = float(np.std(estimated))
            nrmse = float(
                np.sqrt(np.mean(np.square(estimated - observed)))
                / max(observed_sd, 1.0e-8)
            )
            correlation = (
                float(np.corrcoef(observed, estimated)[0, 1])
                if min(observed_sd, estimated_sd) > 1.0e-8
                else float("nan")
            )
            rows.append({
                "window": int(window),
                "forecast_boundary_sample": int(position),
                "channel_index": int(channel_index),
                "channel": str(channel),
                "block_samples": int(block_samples),
                "channel_window_nrmse": nrmse,
                "channel_window_correlation": correlation,
            })
    return rows


def select_loro_rolling_block(
    rows: Sequence[Mapping[str, Any]], *, fold_ids: Sequence[str],
    candidates: Sequence[int], minimum_correlation: float,
    maximum_nrmse: float,
) -> tuple[int, list[dict[str, Any]]]:
    """Select the largest block passing the frozen gates in every LORO fold."""

    import math

    expected = {(str(fold), int(block)) for fold in fold_ids for block in candidates}
    observed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["fold_id"]), int(row["block_samples"]))
        if key in observed:
            raise PermissionError(f"duplicate rolling fold/block evaluation: {key}")
        observed[key] = row
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        extra = sorted(set(observed) - expected)
        raise PermissionError(
            f"incomplete rolling LORO Cartesian product; missing={missing}, extra={extra}"
        )
    summary = []
    eligible = []
    for block in map(int, candidates):
        group = [observed[(str(fold), block)] for fold in fold_ids]
        fold_passes = []
        for row in group:
            correlation = float(row["median_correlation"])
            nrmse = float(row["median_nrmse"])
            fold_passes.append(bool(
                math.isfinite(correlation) and math.isfinite(nrmse)
                and correlation >= float(minimum_correlation)
                and nrmse <= float(maximum_nrmse)
                and bool(row["rolling_gate_pass"])
            ))
        all_folds_pass = bool(all(fold_passes))
        summary.append({
            "block_samples": block,
            "all_folds_pass": all_folds_pass,
            "worst_fold_median_nrmse": max(
                float(row["median_nrmse"]) for row in group
            ),
            "worst_fold_median_correlation": min(
                float(row["median_correlation"]) for row in group
            ),
            "fold_count": len(group),
        })
        if all_folds_pass:
            eligible.append(block)
    if not eligible:
        raise RuntimeError("no rolling block passed every development LORO fold")
    selected = max(eligible)
    for row in summary:
        row["selected_largest_jointly_eligible"] = bool(
            int(row["block_samples"]) == selected
        )
    return selected, summary


def loro_part1_part2(config: Mapping[str, Any], development_npz: Path, output: Path) -> None:
    import joblib
    import numpy as np
    import pandas as pd
    from dataclasses import replace

    _data, canonical_network, part2_module, part2_pipeline = activate_canonical(config)
    arrays = load_development(development_npz)
    channels = arrays["channels"].astype(str).tolist()
    sfreq = float(arrays["sfreq"])
    runs = list(config["source_data"]["development_runs"])
    context_indices = list(config["part2"]["selection_context_indices"])
    candidates = part2_grid(config, part2_module)
    fold_part1_networks: dict[str, dict[str, Any]] = {}
    fold_part2_networks: dict[str, dict[str, Any]] = {}
    fold_definitions: list[tuple[str, list[str], str]] = []
    for validation_run in runs:
        training_runs = [run for run in runs if run != validation_run]
        fold_id = f"leave-{validation_run}-out"
        fold_definitions.append((fold_id, training_runs, validation_run))
        training_sequences = [get_run(arrays, "ictal", run) for run in training_runs]
        fold_part1_networks[fold_id] = build_part1_selection_network(
            config, training_sequences, sfreq,
            canonical_network_module=canonical_network,
        )
        fold_part2_networks[fold_id] = build_part2_plant_network(
            config, training_sequences, sfreq,
            canonical_part2_pipeline_module=part2_pipeline,
        )

    output.mkdir(parents=True, exist_ok=True)
    (output / "fold_part1_selection_networks").mkdir()
    (output / "fold_part2_plant_networks").mkdir()
    for fold_id, _training, _validation in fold_definitions:
        save_network(
            output / "fold_part1_selection_networks" / f"{fold_id}.npz",
            fold_part1_networks[fold_id], channels,
        )
        save_network(
            output / "fold_part2_plant_networks" / f"{fold_id}.npz",
            fold_part2_networks[fold_id], channels,
        )

    candidate_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        per_fold: list[float] = []
        for fold_index, (fold_id, training_runs, validation_run) in enumerate(fold_definitions):
            started = time.perf_counter()
            status = "ok"
            error = ""
            try:
                model = part2_module.ResidualGraphRCSDE(
                    candidate, fold_part2_networks[fold_id]["adjacency"]
                ).fit([get_run(arrays, "ictal", run) for run in training_runs])
                objective, details = _score_part2_candidate(
                    model,
                    get_run(arrays, "ictal", validation_run),
                    context_indices,
                    part2_module,
                    deterministic_seed(
                        "HUP060_consistent_sparse_rerun_common_v1", "HUP080", fold_id,
                        "part2_grid", 0, int(config["controller"]["validation_seed"]),
                    ),
                )
                for row in details:
                    context_rows.append(
                        {"candidate_index": candidate_index, "fold_id": fold_id, **row}
                    )
            except Exception as exc:
                objective = 1e12
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
            per_fold.append(float(objective))
            candidate_rows.append(
                {
                    "candidate_index": candidate_index,
                    "candidate_id": f"part2-candidate-{candidate_index:03d}",
                    "fold_id": fold_id,
                    "validation_run": validation_run,
                    "training_runs": ";".join(training_runs),
                    "objective": float(objective),
                    "status": status,
                    "error": error,
                    "elapsed_seconds": time.perf_counter() - started,
                    **asdict(candidate),
                }
            )
        print(f"Part-II candidate {candidate_index + 1}/144 complete", flush=True)
    frame = pd.DataFrame(candidate_rows)
    frame.to_csv(output / "part2_grid_fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(context_rows).to_csv(
        output / "part2_grid_context_metrics.csv", index=False, encoding="utf-8-sig"
    )
    aggregate = frame.groupby("candidate_index", as_index=False).agg(
        candidate_id=("candidate_id", "first"),
        mean_objective=("objective", "mean"),
        worst_fold_objective=("objective", "max"),
        successful_folds=("status", lambda values: int((values == "ok").sum())),
    )
    eligible = aggregate[
        aggregate["successful_folds"].eq(3) & np.isfinite(aggregate["mean_objective"])
    ].sort_values(
        ["worst_fold_objective", "mean_objective", "candidate_id"],
        kind="mergesort",
    )
    if eligible.empty:
        raise RuntimeError("all Part-II candidates failed LORO")
    selected_index = int(eligible.iloc[0]["candidate_index"])
    selected = candidates[selected_index]

    alpha_rows: list[dict[str, Any]] = []
    for alpha in config["part2"]["diffusion_ridge_alphas"]:
        for fold_index, (fold_id, training_runs, validation_run) in enumerate(fold_definitions):
            alpha_config = replace(selected, diffusion_ridge_alpha=float(alpha))
            model = part2_module.ResidualGraphRCSDE(
                alpha_config, fold_part2_networks[fold_id]["adjacency"]
            ).fit([get_run(arrays, "ictal", run) for run in training_runs])
            projections = part2_module.fixed_random_projections(96, n_projections=128, seed=20261002)
            for context_index in context_indices:
                context, future, _ = part2_context_block(
                    get_run(arrays, "ictal", validation_run), context_index,
                    part2_module.forecast_positions,
                )
                samples = model.sample_direct_scaled(
                    context, 128, 32,
                    deterministic_seed(
                        "HUP060_consistent_sparse_rerun_common_v1", "HUP080", fold_id,
                        "part2_diffusion", context_index, 20261002,
                    ),
                    diffusion_scale=1.0,
                )
                truth = model.transform.scaler.transform(future[:128])
                persistence = np.repeat(
                    model.transform.scaler.transform(context[-1:]), 128, axis=0
                )
                swd = part2_module.safe_swd(samples.reshape(-1, 96), truth, projections)
                baseline = part2_module.safe_swd(persistence, truth, projections)
                alpha_rows.append(
                    {
                        "diffusion_ridge_alpha": float(alpha), "fold_id": fold_id,
                        "context_index": int(context_index), "swd": float(swd),
                        "persistence_swd": float(baseline),
                        "swd_ratio_to_persistence": float(swd / max(baseline, 1e-12)),
                    }
                )
    alpha_frame = pd.DataFrame(alpha_rows)
    alpha_frame.to_csv(output / "diffusion_alpha_loro.csv", index=False, encoding="utf-8-sig")
    alpha_fold_summary = alpha_frame.groupby(
        ["diffusion_ridge_alpha", "fold_id"], as_index=False
    ).agg(
        fold_mean_swd_ratio=("swd_ratio_to_persistence", "mean"),
        validation_window_rows=("context_index", "size"),
        unique_validation_windows=("context_index", "nunique"),
    )
    if not (
        alpha_fold_summary["validation_window_rows"].eq(6).all()
        and alpha_fold_summary["unique_validation_windows"].eq(6).all()
    ):
        raise RuntimeError(
            "diffusion-alpha LORO aggregation requires six unique windows per fold"
        )
    expected_alpha_fold_cells = int(
        len(config["part2"]["diffusion_ridge_alphas"])
        * len(fold_definitions)
    )
    if len(alpha_fold_summary) != expected_alpha_fold_cells:
        raise RuntimeError("diffusion-alpha LORO fold Cartesian product is incomplete")
    alpha_fold_summary.to_csv(
        output / "diffusion_alpha_loro_fold_summary.csv",
        index=False, encoding="utf-8-sig",
    )
    alpha_summary = alpha_fold_summary.groupby(
        "diffusion_ridge_alpha", as_index=False
    ).agg(
        worst_fold_mean_swd_ratio=("fold_mean_swd_ratio", "max"),
        fold_count=("fold_id", "nunique"),
    )
    overall = alpha_frame.groupby(
        "diffusion_ridge_alpha", as_index=False
    ).agg(overall_mean_swd_ratio=("swd_ratio_to_persistence", "mean"))
    alpha_summary = alpha_summary.merge(
        overall, on="diffusion_ridge_alpha", validate="one_to_one"
    ).sort_values(
        [
            "worst_fold_mean_swd_ratio", "overall_mean_swd_ratio",
            "diffusion_ridge_alpha",
        ],
        kind="mergesort",
    )
    if not alpha_summary["fold_count"].eq(len(fold_definitions)).all():
        raise RuntimeError("diffusion-alpha summary is missing a LORO fold")
    alpha_summary.to_csv(
        output / "diffusion_alpha_loro_summary.csv",
        index=False, encoding="utf-8-sig",
    )
    best_alpha = float(alpha_summary.iloc[0]["diffusion_ridge_alpha"])
    selected = replace(selected, diffusion_ridge_alpha=best_alpha)

    rolling_fold_rows: list[dict[str, Any]] = []
    rolling_channel_rows: list[dict[str, Any]] = []
    selected_fold_models: dict[str, Any] = {}
    for fold_id, training_runs, validation_run in fold_definitions:
        model = part2_module.ResidualGraphRCSDE(
            selected, fold_part2_networks[fold_id]["adjacency"]
        ).fit([get_run(arrays, "ictal", run) for run in training_runs])
        selected_fold_models[fold_id] = model
        validation = get_run(arrays, "ictal", validation_run)
        positions = part2_module.forecast_positions(
            len(validation), context_samples=int(config["part2"]["context_samples"]),
            maximum_horizon=int(config["part2"]["rolling_display_samples"]),
            guard_samples=0, count=int(config["part2"]["validation_windows"]),
        )
        for block_samples in config["part2"]["rolling_block_candidates"]:
            details = _rolling_block_metrics(
                model, validation, positions,
                context_samples=int(config["part2"]["context_samples"]),
                block_samples=int(block_samples),
                total_samples=int(config["part2"]["rolling_display_samples"]),
                channels=channels,
            )
            detail_frame = pd.DataFrame(details)
            median_nrmse = float(detail_frame["channel_window_nrmse"].median())
            median_correlation = float(
                detail_frame["channel_window_correlation"].median()
            )
            gate_pass = bool(
                np.isfinite(median_nrmse)
                and np.isfinite(median_correlation)
                and median_correlation
                >= float(config["part2"]["rolling_selection_min_correlation"])
                and median_nrmse
                <= float(config["part2"]["rolling_selection_max_nrmse"])
            )
            rolling_fold_rows.append({
                "fold_id": fold_id,
                "validation_run": validation_run,
                "block_samples": int(block_samples),
                "median_nrmse": median_nrmse,
                "median_correlation": median_correlation,
                "rolling_gate_pass": gate_pass,
                "outer_or_test_used": False,
            })
            rolling_channel_rows.extend(
                {"fold_id": fold_id, "validation_run": validation_run, **row}
                for row in details
            )
    rolling_fold_frame = pd.DataFrame(rolling_fold_rows)
    rolling_fold_frame.to_csv(
        output / "rolling_block_loro_fold_screen.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.DataFrame(rolling_channel_rows).to_csv(
        output / "rolling_block_loro_channel_metrics.csv",
        index=False, encoding="utf-8-sig",
    )
    selected_block_samples, rolling_joint_rows = select_loro_rolling_block(
        rolling_fold_rows,
        fold_ids=[item[0] for item in fold_definitions],
        candidates=config["part2"]["rolling_block_candidates"],
        minimum_correlation=float(
            config["part2"]["rolling_selection_min_correlation"]
        ),
        maximum_nrmse=float(config["part2"]["rolling_selection_max_nrmse"]),
    )
    pd.DataFrame(rolling_joint_rows).to_csv(
        output / "rolling_block_loro_joint_screen.csv",
        index=False, encoding="utf-8-sig",
    )

    (output / "fold_models").mkdir()
    plant_rows: list[dict[str, Any]] = []
    for fold_id, training_runs, validation_run in fold_definitions:
        model = selected_fold_models[fold_id]
        assert_live_part2_model_identity(model, config)
        if array_sha256(model.adjacency_input) != array_sha256(
            fold_part2_networks[fold_id]["adjacency"]
        ):
            raise RuntimeError("fold model is not bound to its Part-II plant graph")
        model_path = output / "fold_models" / f"{fold_id}.joblib"
        joblib.dump(model, model_path, compress=3)
        plant_rows.append(
            {
                "fold_id": fold_id,
                "training_runs": training_runs,
                "validation_run": validation_run,
                "model_sha256": sha256_file(model_path),
                "part1_selection_network_sha256": sha256_file(
                    output / "fold_part1_selection_networks" / f"{fold_id}.npz"
                ),
                "part2_plant_network_sha256": sha256_file(
                    output / "fold_part2_plant_networks" / f"{fold_id}.npz"
                ),
                "part1_and_part2_networks_distinct": True,
                "model_plant_adjacency_input_sha256": array_sha256(
                    model.adjacency_input
                ),
                "model_normalized_plant_adjacency_sha256": array_sha256(
                    model.adjacency
                ),
                "latent_components": int(model.q),
            }
        )
    atomic_json(
        output / "selected_config.json",
        {
            "subject": "HUP080",
            "selected_candidate_index": selected_index,
            "selected_candidate_id": f"part2-candidate-{selected_index:03d}",
            "model_config": asdict(selected),
            "diffusion_ridge_alpha_selected_by_loro": best_alpha,
            "rolling_block_samples_selected_by_loro": selected_block_samples,
            "rolling_block_selection_rule": (
                "largest candidate passing correlation>=0.70 and nRMSE<=0.80 "
                "in every development LORO fold; empty set is NO_GO"
            ),
            "effective_diffusion_multiplier_for_part3": config["part2"]["effective_diffusion_multiplier"],
            "selection_context_indices": context_indices,
            "loro_candidate_aggregation": config["part2"]["loro_candidate_aggregation"],
            "network_role_contract": {
                "part1": "multiband_equal_4s_selection_only",
                "part2": "broadband_2s_graph_rc_plant_only",
                "part1_graph_reused_as_part2_plant": False,
            },
            "ctx5_used": False,
            "ctx6_used": False,
            "ctx7_used": False,
            "candidate_grid_sha256": sha256_json([asdict(item) for item in candidates]),
            "fold_models": plant_rows,
            "clinical_labels_used": False,
            "sealed_run_opened": False,
        },
    )
