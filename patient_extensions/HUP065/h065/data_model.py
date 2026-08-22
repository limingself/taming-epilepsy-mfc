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
)
from .staging import atomic_json


def _purge_mfc_pipeline() -> None:
    for name in tuple(sys.modules):
        if name == "mfc_pipeline" or name.startswith("mfc_pipeline."):
            del sys.modules[name]


def activate_canonical(config: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    """Idempotently bind one live set of hash-pinned D canonical modules."""

    root = Path(str(config["canonical_locks"]["part2_core"][0])).parents[1].resolve()
    loaded_paths = [
        Path(value.__file__).resolve()
        for name, value in sys.modules.items()
        if (name == "mfc_pipeline" or name.startswith("mfc_pipeline."))
        and getattr(value, "__file__", None)
    ]
    if any(path != root and root not in path.parents for path in loaded_paths):
        _purge_mfc_pipeline()
    sys.path[:] = [item for item in sys.path if os.path.normcase(item) != os.path.normcase(str(root))]
    sys.path.insert(0, str(root))
    data = importlib.import_module("mfc_pipeline.data")
    network = importlib.import_module("mfc_pipeline.network")
    part2 = importlib.import_module("mfc_pipeline.part2_state_dependent_rc_sde")
    part2_pipeline = importlib.import_module("mfc_pipeline.part2_data_pipeline")
    required = {
        "canonical_data_loader": data,
        "part1_network_core": network,
        "part2_core": part2,
        "part2_data_pipeline": part2_pipeline,
    }
    for lock_name, module in required.items():
        observed_path = Path(module.__file__).resolve()
        expected_path, expected_sha = config["canonical_locks"][lock_name]
        if observed_path != Path(str(expected_path)).resolve():
            raise ImportError(f"canonical module path changed: {lock_name}")
        if sha256_file(observed_path) != str(expected_sha).lower():
            raise PermissionError(f"canonical module hash changed: {lock_name}")
    return data, network, part2, part2_pipeline


def assert_live_part2_model_identity(model: Any, part2_module: Any) -> None:
    expected = part2_module.ResidualGraphRCSDE
    if model.__class__ is not expected:
        raise RuntimeError(
            "Part-II model class identity differs from the live hash-pinned module"
        )


def _run_key(prefix: str, run: str) -> str:
    return f"{prefix}_{str(run).replace('-', '_')}"


def assert_signal_run_allowed(phase: str, run: str) -> None:
    """Physical signal-access firewall for development versus sealed roles."""

    phase_id = str(phase)
    run_id = str(run)
    if phase_id == "OUTER":
        if run_id != "run-03":
            raise PermissionError("OUTER may open only the sealed run03 signal")
        return
    if run_id not in {"run-01", "run-02"}:
        raise PermissionError(f"{phase_id} may not open sealed or undeclared signal {run_id}")


def parse_hup065_split(config: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(config["source_data"]["split_proposal"]))
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("subject") == "HUP065"]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one HUP065 split row, observed {len(rows)}")
    row = rows[0]
    if row.get("schema_version") != "2" or row.get("split_status") != "ready":
        raise ValueError("HUP065 split row is not ready schema-v2")
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
        "leave-run-01-out", "leave-run-02-out"
    ]:
        raise PermissionError("HUP065 LORO fold identities changed")
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
        "run": "run-03",
        "start_s": 75.0,
        "stop_s": 115.0,
    }:
        raise PermissionError("sealed reference-test role changed")
    if ictal_test.get("run") != "run-03" or [
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
    """Read only run01/run02 ictal EDFs and their seizure-pre blocks."""

    import numpy as np

    data_module, _network_module, _part2, _part2_pipeline = activate_canonical(config)
    split = parse_hup065_split(config)
    data_cfg = config["source_data"]
    prep = config["preprocessing"]
    runs = tuple(data_cfg["development_runs"])
    dataset = data_module.BIDSZipDataset(data_cfg["zip_root"], subjects=["HUP065"])
    records = dataset.selected_records("HUP065", [("ictal", run) for run in runs])
    if tuple(record.run for record in records) != runs:
        raise PermissionError("selected development EDF order differs from the frozen role ledger")
    if any(record.task.casefold() != "ictal" or record.run == "run-03" for record in records):
        raise PermissionError("non-development or non-ictal record entered preparation")
    channels = tuple(dataset.common_good_channels("HUP065", records=records))
    if len(channels) != int(data_cfg["expected_channels"]):
        raise ValueError(f"expected 64 common-good channels, observed {len(channels)}")
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
        assert_signal_run_allowed("DEV_MATERIALIZE", record.run)
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
            if values.shape != (samples, 64) or not np.isfinite(values).all():
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
            "subject": "HUP065",
            "opened_signal_runs": list(runs),
            "sealed_signal_run": "run-03",
            "run03_signal_opened": False,
            "task_interictal_opened": False,
            "reference_label": "preictal_reference",
            "development_reference_windows": {run: [30.0, 70.0] for run in runs},
            "development_ictal_windows": {run: [120.0, 152.0] for run in runs},
            "context_positions_samples": list(positions),
            "context_roles": config["contexts"],
            "fold_reference_rule": "each fold fits its target only from the other development seizure",
            "split_proposal_sha256": config["source_data"]["split_proposal_sha256"],
        },
    )
    atomic_json(
        output / "provenance.json",
        {
            "source_archive": str(Path(data_cfg["raw_zip"]).resolve()),
            "source_archive_sha256": data_cfg["raw_zip_sha256"],
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


def build_part1_network(
    config: Mapping[str, Any], sequences: Sequence[Any], sfreq: float
) -> dict[str, Any]:
    """Build the multiband-equal Part-I graph used for actuator centrality."""

    import numpy as np

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


def build_part2_plant_adjacency(
    config: Mapping[str, Any], sequences: Sequence[Any], sfreq: float
) -> Any:
    """Build D Part-II's broadband 2-s PLV-median plant adjacency."""

    import numpy as np

    _data, _network, _part2, part2_pipeline = activate_canonical(config)
    adjacency = np.asarray(
        part2_pipeline.compute_adjacency(
            [np.asarray(sequence, dtype=np.float64) for sequence in sequences],
            float(sfreq),
        ),
        dtype=np.float64,
    )
    if adjacency.shape != (64, 64) or not np.isfinite(adjacency).all():
        raise ValueError("Part-II broadband plant adjacency shape/finite check failed")
    return adjacency


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
    """Frozen Part-III 256-sample context and 256-sample future."""

    positions = context_positions(len(sequence))
    boundary = positions[int(index)]
    return sequence[boundary - 256 : boundary].copy(), sequence[boundary : boundary + 256].copy(), boundary


def part2_validation_positions(
    sequence: Any, part2_module: Any, validation_windows: int = 6
) -> tuple[int, ...]:
    """Use the hash-locked D Part-II deterministic forecast-position rule."""

    positions = tuple(
        int(value)
        for value in part2_module.forecast_positions(
            len(sequence), context_samples=512, maximum_horizon=128,
            guard_samples=0, count=int(validation_windows),
        )
    )
    if len(positions) != int(validation_windows):
        raise RuntimeError("D Part-II forecast-position count changed")
    return positions


def part2_context_block(
    sequence: Any, index: int, positions: Sequence[int]
) -> tuple[Any, Any, int]:
    """Frozen Part-II 512-sample history and maximum 128-sample future."""

    boundary = int(positions[int(index)])
    context = sequence[boundary - 512 : boundary].copy()
    future = sequence[boundary : boundary + 128].copy()
    if context.shape[0] != 512 or future.shape[0] != 128:
        raise ValueError("Part-II context/future shape changed")
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
    if list(map(int, context_indices)) != list(range(6)):
        raise ValueError("Part-II direct validation must use six frozen windows")
    positions = part2_validation_positions(validation, part2_module, 6)
    for context_index in context_indices:
        context, future, boundary = part2_context_block(
            validation, int(context_index), positions
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


def rolling_block_fold_metrics(
    model: Any,
    validation: Any,
    part2_module: Any,
    block_samples: int,
    *,
    validation_windows: int = 6,
    total_samples: int = 128,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """D-canonical direct rolling forecast metrics on one left-out dev run."""

    import numpy as np

    block = int(block_samples)
    if block not in {4, 8, 12, 16, 24, 32}:
        raise ValueError("undeclared rolling block candidate")
    if int(total_samples) != 128 or int(validation_windows) != 6:
        raise ValueError("rolling validation horizon/window contract changed")
    positions = part2_validation_positions(validation, part2_module, validation_windows)
    rows: list[dict[str, Any]] = []
    for window, position in enumerate(positions):
        blocks = []
        for start in range(0, int(total_samples), block):
            boundary = int(position) + start
            context = validation[boundary - 512 : boundary]
            if context.shape[0] != 512:
                raise ValueError("rolling Part-II context is not 512 samples")
            steps = min(block, int(total_samples) - start)
            blocks.append(model.forecast_direct(context, steps))
        prediction = np.vstack(blocks)
        truth = validation[int(position) : int(position) + int(total_samples)]
        truth_scaled = model.transform.scaler.transform(truth)
        prediction_scaled = model.transform.scaler.transform(prediction)
        for channel_index in range(int(validation.shape[1])):
            observed = truth_scaled[:, channel_index]
            estimated = prediction_scaled[:, channel_index]
            nrmse = float(
                np.sqrt(np.mean((estimated - observed) ** 2))
                / max(float(np.std(observed)), 1.0e-8)
            )
            correlation = (
                float(np.corrcoef(observed, estimated)[0, 1])
                if min(float(np.std(observed)), float(np.std(estimated))) > 1.0e-8
                else float("nan")
            )
            rows.append(
                {
                    "block_samples": block,
                    "window": int(window),
                    "forecast_boundary_sample": int(position),
                    "channel_index": int(channel_index),
                    "channel_window_nrmse": nrmse,
                    "channel_window_correlation": correlation,
                }
            )
    nrmse_values = np.asarray([row["channel_window_nrmse"] for row in rows])
    correlation_values = np.asarray(
        [row["channel_window_correlation"] for row in rows]
    )
    finite = bool(np.isfinite(nrmse_values).all() and np.isfinite(correlation_values).all())
    summary = {
        "block_samples": block,
        "median_nrmse": float(np.median(nrmse_values)) if finite else float("inf"),
        "median_correlation": (
            float(np.median(correlation_values)) if finite else float("-inf")
        ),
        "finite": finite,
        "validation_windows": int(validation_windows),
        "total_samples": int(total_samples),
    }
    return summary, rows


def select_rolling_block_fail_closed(
    fold_rows: Sequence[Mapping[str, Any]], fold_ids: Sequence[str]
) -> int:
    """Freeze the largest block passing both gates in every LORO fold."""

    expected_folds = tuple(str(value) for value in fold_ids)
    candidates = (4, 8, 12, 16, 24, 32)
    eligible: list[int] = []
    for candidate in candidates:
        rows = [row for row in fold_rows if int(row["block_samples"]) == candidate]
        observed_folds = tuple(sorted(str(row["fold_id"]) for row in rows))
        if observed_folds != tuple(sorted(expected_folds)) or len(rows) != len(expected_folds):
            raise RuntimeError("rolling screen lacks one unique row per block and fold")
        if all(
            bool(row["finite"])
            and float(row["median_correlation"]) >= 0.70
            and float(row["median_nrmse"]) <= 0.80
            for row in rows
        ):
            eligible.append(candidate)
    if not eligible:
        raise RuntimeError("NO_GO_before_part3: no jointly eligible rolling block")
    return max(eligible)


def loro_part1(config: Mapping[str, Any], development_npz: Path, output: Path) -> None:
    """Compute only the two fresh LORO Part-I graphs from development runs."""

    import numpy as np

    arrays = load_development(development_npz)
    channels = arrays["channels"].astype(str).tolist()
    sfreq = float(arrays["sfreq"])
    runs = list(config["source_data"]["development_runs"])
    if runs != ["run-01", "run-02"] or len(channels) != 64:
        raise PermissionError("Part-I is bound to HUP065 run01/run02 and 64 channels")
    output.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, Any]] = []
    for validation_run in runs:
        training_runs = [run for run in runs if run != validation_run]
        fold_id = f"leave-{validation_run}-out"
        network = build_part1_network(
            config,
            [get_run(arrays, "ictal", run) for run in training_runs],
            sfreq,
        )
        path = output / f"{fold_id}.npz"
        save_network(path, network, channels)
        fold_rows.append(
            {
                "fold_id": fold_id,
                "training_runs": training_runs,
                "validation_run": validation_run,
                "part1_selection_network_sha256": sha256_file(path),
                "channel_count": len(channels),
                "clinical_labels_used": False,
                "sealed_run_opened": False,
            }
        )
    atomic_json(
        output / "part1_receipt.json",
        {
            "schema_version": "hup065-loro-part1-v1",
            "subject": "HUP065",
            "folds": fold_rows,
            "band_aggregation": "equal_weight_mean",
            "plv_window_seconds": float(config["part1"]["plv_window_s"]),
            "plv_overlap_fraction": float(config["part1"]["plv_overlap"]),
            "graph_role": "part1_multiband_selection_centrality_only",
            "graph_rule": "maximum_spanning_tree_plus_strongest_edges",
            "network_density": float(config["part1"]["network_density"]),
            "clinical_labels_used": False,
            "sealed_run_opened": False,
        },
    )


def loro_part1_part2(
    config: Mapping[str, Any],
    development_npz: Path,
    output: Path,
    *,
    frozen_part1_dir: Path | None = None,
) -> None:
    import joblib
    import numpy as np
    import pandas as pd
    from dataclasses import replace

    _data, _network, part2_module, _part2_pipeline = activate_canonical(config)
    arrays = load_development(development_npz)
    channels = arrays["channels"].astype(str).tolist()
    sfreq = float(arrays["sfreq"])
    runs = list(config["source_data"]["development_runs"])
    if int(config["part2"]["context_samples"]) != 512:
        raise PermissionError("Part-II context must remain 512 samples")
    if int(config["part2"]["validation_windows"]) != 6:
        raise PermissionError("Part-II validation must remain six windows")
    context_indices = list(range(int(config["part2"]["validation_windows"])))
    candidates = part2_grid(config, part2_module)
    fold_part1_networks: dict[str, dict[str, Any]] = {}
    fold_plant_adjacencies: dict[str, Any] = {}
    fold_definitions: list[tuple[str, list[str], str]] = []
    for validation_run in runs:
        training_runs = [run for run in runs if run != validation_run]
        fold_id = f"leave-{validation_run}-out"
        fold_definitions.append((fold_id, training_runs, validation_run))
        training_sequences = [get_run(arrays, "ictal", run) for run in training_runs]
        if frozen_part1_dir is None:
            fold_part1_networks[fold_id] = build_part1_network(
                config, training_sequences, sfreq
            )
        else:
            source = Path(frozen_part1_dir) / f"{fold_id}.npz"
            if not source.is_file():
                raise FileNotFoundError(f"frozen Part-I fold graph is absent: {source}")
            fold_part1_networks[fold_id] = load_network(source)
            if fold_part1_networks[fold_id]["channels"].astype(str).tolist() != channels:
                raise PermissionError(f"{fold_id}: Part-I channel order changed")
        fold_plant_adjacencies[fold_id] = build_part2_plant_adjacency(
            config, training_sequences, sfreq
        )

    output.mkdir(parents=True, exist_ok=True)
    (output / "fold_part1_selection_networks").mkdir()
    (output / "fold_part2_plant_adjacencies").mkdir()
    for fold_id, _training, _validation in fold_definitions:
        save_network(
            output / "fold_part1_selection_networks" / f"{fold_id}.npz",
            fold_part1_networks[fold_id], channels,
        )
        np.savez_compressed(
            output / "fold_part2_plant_adjacencies" / f"{fold_id}.npz",
            adjacency=np.asarray(fold_plant_adjacencies[fold_id], dtype=np.float64),
            channels=np.asarray(channels, dtype=str),
            graph_role=np.asarray("part2_broadband_2s_graph_rc_plant"),
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
                    candidate, fold_plant_adjacencies[fold_id]
                ).fit([get_run(arrays, "ictal", run) for run in training_runs])
                objective, details = _score_part2_candidate(
                    model,
                    get_run(arrays, "ictal", validation_run),
                    context_indices,
                    part2_module,
                    deterministic_seed(
                        "HUP060_consistent_sparse_rerun_common_v1", "HUP065", fold_id,
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
        mean_objective=("objective", "mean"),
        worst_fold_objective=("objective", "max"),
        successful_folds=("status", lambda values: int((values == "ok").sum())),
    )
    eligible = aggregate[
        aggregate["successful_folds"].eq(len(runs)) & np.isfinite(aggregate["mean_objective"])
    ].sort_values(
        ["worst_fold_objective", "mean_objective", "candidate_index"],
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
                alpha_config, fold_plant_adjacencies[fold_id]
            ).fit([get_run(arrays, "ictal", run) for run in training_runs])
            projections = part2_module.fixed_random_projections(64, n_projections=128, seed=20261002)
            validation_sequence = get_run(arrays, "ictal", validation_run)
            validation_positions = part2_validation_positions(
                validation_sequence, part2_module, 6
            )
            for context_index in context_indices:
                context, future, _ = part2_context_block(
                    validation_sequence, context_index, validation_positions
                )
                samples = model.sample_direct_scaled(
                    context, 128, 32,
                    deterministic_seed(
                        "HUP060_consistent_sparse_rerun_common_v1", "HUP065", fold_id,
                        "part2_diffusion", context_index, 20261002,
                    ),
                    diffusion_scale=1.0,
                )
                truth = model.transform.scaler.transform(future[:128])
                persistence = np.repeat(
                    model.transform.scaler.transform(context[-1:]), 128, axis=0
                )
                swd = part2_module.safe_swd(samples.reshape(-1, 64), truth, projections)
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
        validation_windows=("context_index", "nunique"),
    )
    if not alpha_fold_summary["validation_windows"].eq(6).all():
        raise RuntimeError("diffusion alpha LORO fold does not contain exactly 6 windows")
    alpha_fold_summary.to_csv(
        output / "diffusion_alpha_loro_fold_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    alpha_summary = alpha_fold_summary.groupby(
        "diffusion_ridge_alpha", as_index=False
    ).agg(
        mean_fold_swd_ratio=("fold_mean_swd_ratio", "mean"),
        worst_fold_mean_swd_ratio=("fold_mean_swd_ratio", "max"),
        successful_folds=("fold_id", "nunique"),
    ).sort_values(
        [
            "worst_fold_mean_swd_ratio",
            "mean_fold_swd_ratio",
            "diffusion_ridge_alpha",
        ],
        kind="mergesort",
    )
    if not alpha_summary["successful_folds"].eq(len(fold_definitions)).all():
        raise RuntimeError("diffusion alpha LORO aggregation lost a fold")
    alpha_summary.to_csv(
        output / "diffusion_alpha_loro_two_level_ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best_alpha = float(alpha_summary.iloc[0]["diffusion_ridge_alpha"])
    selected = replace(selected, diffusion_ridge_alpha=best_alpha)

    (output / "fold_models").mkdir()
    plant_rows: list[dict[str, Any]] = []
    rolling_fold_rows: list[dict[str, Any]] = []
    rolling_channel_rows: list[dict[str, Any]] = []
    for fold_id, training_runs, validation_run in fold_definitions:
        model = part2_module.ResidualGraphRCSDE(
            selected, fold_plant_adjacencies[fold_id]
        ).fit([get_run(arrays, "ictal", run) for run in training_runs])
        assert_live_part2_model_identity(model, part2_module)
        if array_sha256(np.asarray(model.adjacency_input)) != array_sha256(
            np.asarray(fold_plant_adjacencies[fold_id])
        ):
            raise PermissionError(f"{fold_id}: model adjacency_input changed")
        model_path = output / "fold_models" / f"{fold_id}.joblib"
        joblib.dump(model, model_path, compress=3)
        validation_sequence = get_run(arrays, "ictal", validation_run)
        for block_samples in config["part2"]["rolling_block_candidates_samples"]:
            rolling_summary, rolling_details = rolling_block_fold_metrics(
                model,
                validation_sequence,
                part2_module,
                int(block_samples),
                validation_windows=int(config["part2"]["validation_windows"]),
                total_samples=int(config["part2"]["rolling_display_samples"]),
            )
            rolling_fold_rows.append(
                {
                    "fold_id": fold_id,
                    "validation_run": validation_run,
                    **rolling_summary,
                    "correlation_gate_pass": bool(
                        rolling_summary["finite"]
                        and float(rolling_summary["median_correlation"])
                        >= float(config["part2"]["rolling_selection_min_correlation"])
                    ),
                    "nrmse_gate_pass": bool(
                        rolling_summary["finite"]
                        and float(rolling_summary["median_nrmse"])
                        <= float(config["part2"]["rolling_selection_max_nrmse"])
                    ),
                }
            )
            rolling_channel_rows.extend(
                {"fold_id": fold_id, "validation_run": validation_run, **row}
                for row in rolling_details
            )
        plant_rows.append(
            {
                "fold_id": fold_id,
                "training_runs": training_runs,
                "validation_run": validation_run,
                "model_sha256": sha256_file(model_path),
                "part1_selection_network_sha256": sha256_file(
                    output / "fold_part1_selection_networks" / f"{fold_id}.npz"
                ),
                "part2_plant_adjacency_sha256": sha256_file(
                    output / "fold_part2_plant_adjacencies" / f"{fold_id}.npz"
                ),
                "part2_plant_adjacency_input_array_sha256": array_sha256(
                    np.asarray(model.adjacency_input)
                ),
                "part2_model_normalized_adjacency_array_sha256": array_sha256(
                    np.asarray(model.adjacency)
                ),
                "latent_components": int(model.q),
            }
        )
    pd.DataFrame(rolling_fold_rows).to_csv(
        output / "rolling_block_loro_fold_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(rolling_channel_rows).to_csv(
        output / "rolling_block_loro_channel_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected_rolling_block = select_rolling_block_fail_closed(
        rolling_fold_rows, [item[0] for item in fold_definitions]
    )
    atomic_json(
        output / "selected_config.json",
        {
            "subject": "HUP065",
            "selected_candidate_index": selected_index,
            "model_config": asdict(selected),
            "diffusion_ridge_alpha_selected_by_loro": best_alpha,
            "rolling_block_samples_selected_by_loro": selected_rolling_block,
            "rolling_block_selection_rule": (
                "largest_candidate_passing_corr_ge_0.70_and_nrmse_le_0.80_in_every_fold"
            ),
            "rolling_no_eligible_block": "NO_GO_before_part3",
            "candidate_selection_order": [
                "worst_fold_objective", "mean_objective", "candidate_index"
            ],
            "diffusion_alpha_selection_order": [
                "worst_fold_mean_swd_ratio",
                "mean_fold_swd_ratio",
                "diffusion_ridge_alpha",
            ],
            "diffusion_alpha_loro_windows_per_fold": 6,
            "effective_diffusion_multiplier_for_part3": config["part2"]["effective_diffusion_multiplier"],
            "selection_context_indices": context_indices,
            "ctx5_used": False,
            "ctx6_used": False,
            "ctx7_used": False,
            "candidate_grid_sha256": sha256_json([asdict(item) for item in candidates]),
            "fold_models": plant_rows,
            "graph_roles": {
                "part1_selection_network": "multiband_equal_4s_centrality_and_direct_mask",
                "part2_plant_adjacency": "broadband_2s_windowed_plv_median_graph_rc_world",
                "adjacency_reused_between_roles": False,
            },
            "clinical_labels_used": False,
            "sealed_run_opened": False,
        },
    )
