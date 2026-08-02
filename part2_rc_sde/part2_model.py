#!/usr/bin/env python
"""Final Part-II state-dependent Graph-RC-SDE model and data export.

This is the single user-facing model entry for manuscript Figs. 2--5.

Commands
--------
``python part2_model.py inspect``
    Load and audit the frozen paper model.
``python part2_model.py source-data``
    Rebuild the exact Fig. 2--5 CSV tables from the frozen matched-rollout
    artifact produced by this model.
``python part2_model.py train [training options]``
    Run the validation-selected final training pipeline.  The expensive raw
    HUP060 data path must be available through the project YAML files.

The numerical implementation is kept in the shared :mod:`mfc_pipeline`
package because Part III imports the same fitted dynamics.  This file is the
only Part-II model command that users need to run.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, wasserstein_distance


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mfc_pipeline import part2_state_dependent_rc_sde as _core  # noqa: E402

# The frozen joblib was created under this historical import name.  Registering
# the alias changes no weights or equations; it only makes deserialization
# independent of the old multi-script folder layout.
sys.modules.setdefault("part2_state_dependent_base", _core)

ModelConfig = _core.ModelConfig
ResidualGraphRCSDE = _core.ResidualGraphRCSDE

MODEL_PATH = (
    ROOT
    / "artifacts"
    / "part2_hup060_state_dependent_optimized"
    / "hup060_state_dependent_rc.joblib"
)
ROLLOUT_PATH = (
    ROOT
    / "artifacts"
    / "part2_hup060_fixed_free_input_ablation"
    / "fixed_context_ablation_rollouts.npz"
)
NODE_TABLE_PATH = (
    ROOT
    / "output"
    / "part1"
    / "source_data"
    / "figure_01"
    / "hup060_plv_nodes_and_selection.csv"
)
SOURCE_02_04 = ROOT / "output" / "part2" / "source_data" / "figures_02_04"
SOURCE_05 = ROOT / "output" / "part2" / "source_data" / "figure_05"

HORIZON = 256
SAMPLING_RATE = 256.0
EXPECTED_CONTEXTS = 8
EXPECTED_PARTICLES = 32
FIXED_CONTEXT = 3
FIXED_PARTICLE = 0
NODE_SPECS = (
    ("RPFa3", "selected SOZ"),
    ("RA3", "selected non-SOZ"),
    ("RAFa4", "unselected"),
)
VARIANT_SPECS = (
    ("state_only", "State only"),
    ("state_delay", "State + delay"),
    ("state_delay_graph", "State + delay + graph"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).ravel()
    array = array[np.isfinite(array)]
    if len(array) < 3 or float(np.std(array)) < 1e-10:
        return np.zeros_like(grid)
    return np.asarray(gaussian_kde(array)(grid), dtype=np.float64)


def _density_grid(*arrays: np.ndarray, points: int = 400) -> np.ndarray:
    pooled = np.concatenate(
        [np.asarray(value, dtype=np.float64).ravel() for value in arrays]
    )
    low, high = np.quantile(pooled, [0.005, 0.995])
    padding = 0.08 * max(float(high - low), 0.1)
    return np.linspace(float(low - padding), float(high + padding), points)


def _trajectory_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = np.asarray(predicted) - np.asarray(observed)
    scale = max(float(np.std(observed)), 1e-8)
    correlation = (
        float(np.corrcoef(observed, predicted)[0, 1])
        if min(float(np.std(observed)), float(np.std(predicted))) > 1e-8
        else float("nan")
    )
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "nrmse": float(np.sqrt(np.mean(residual**2)) / scale),
        "correlation": correlation,
        "maximum_absolute_error": float(np.max(np.abs(residual))),
    }


def load_frozen_model() -> ResidualGraphRCSDE:
    """Load the exact state-dependent Graph-RC-SDE used in the manuscript."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    if not isinstance(model, ResidualGraphRCSDE):
        raise TypeError(f"unexpected model type: {type(model)!r}")
    if model.config.diffusion_mode != "state_dependent":
        raise RuntimeError("the frozen paper model must use state-dependent diffusion")
    return model


def _load_frozen_evaluation() -> dict[str, Any]:
    """Load the fixed, matched-noise run-02 evaluation artifact."""

    if not ROLLOUT_PATH.is_file():
        raise FileNotFoundError(ROLLOUT_PATH)
    with np.load(ROLLOUT_PATH, allow_pickle=False) as payload:
        data = {key: np.asarray(payload[key]) for key in payload.files}
    channels = [str(value) for value in data["channels"].tolist()]
    expected = (EXPECTED_CONTEXTS, EXPECTED_PARTICLES, HORIZON + 1, len(channels))
    for key in ("state_only", "state_delay", "state_delay_graph"):
        if data[key].shape != expected:
            raise RuntimeError(f"{key} has shape {data[key].shape}, expected {expected}")
    if data["observed"].shape != (EXPECTED_CONTEXTS, HORIZON, len(channels)):
        raise RuntimeError("observed evaluation tensor has an invalid shape")
    if float(np.max(np.ptp(data["state_delay_graph"][:, :, 0], axis=1))) > 1e-12:
        raise RuntimeError("particles do not share the fixed initial state")

    node_table = pd.read_csv(NODE_TABLE_PATH)
    for column in ("selected", "soz", "resection"):
        node_table[column] = _parse_bool(node_table[column])
    node_table = node_table.set_index("channel").loc[channels].reset_index()
    node_table["display_class"] = "unselected"
    node_table.loc[node_table["selected"], "display_class"] = "selected non-SOZ"
    node_table.loc[
        node_table["selected"] & (node_table["soz"] | node_table["resection"]),
        "display_class",
    ] = "selected SOZ/resection"
    data["channels"] = channels
    data["node_table"] = node_table
    return data


def _write_prediction_source_data(data: dict[str, Any]) -> None:
    SOURCE_02_04.mkdir(parents=True, exist_ok=True)
    channels = data["channels"]
    positions = np.asarray(data["positions"], dtype=np.int64)
    observed = np.asarray(data["observed"], dtype=np.float64)
    predicted = np.asarray(data["state_delay_graph"][:, :, 1:], dtype=np.float64)
    time = np.arange(HORIZON, dtype=np.float64) / SAMPLING_RATE

    metric_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    representative_density_rows: list[dict[str, Any]] = []
    for channel, node_class in NODE_SPECS:
        channel_index = channels.index(channel)
        observed_path = observed[FIXED_CONTEXT, :, channel_index]
        predicted_path = predicted[FIXED_CONTEXT, FIXED_PARTICLE, :, channel_index]
        predictive_law = predicted[FIXED_CONTEXT, :, :, channel_index].reshape(-1)
        metrics = _trajectory_metrics(observed_path, predicted_path)
        distribution_w1 = float(wasserstein_distance(observed_path, predictive_law))
        metric_rows.append(
            {
                "channel": channel,
                "node_class": node_class,
                "context_index": FIXED_CONTEXT,
                "particle_index": FIXED_PARTICLE,
                "forecast_boundary_sample": int(positions[FIXED_CONTEXT]),
                **metrics,
                "occupation_w1": distribution_w1,
                "distribution_particles": EXPECTED_PARTICLES,
            }
        )
        absolute_error = np.abs(predicted_path - observed_path)
        for sample in range(HORIZON):
            trajectory_rows.append(
                {
                    "channel": channel,
                    "node_class": node_class,
                    "context_index": FIXED_CONTEXT,
                    "particle_index": FIXED_PARTICLE,
                    "time_s": float(time[sample]),
                    "observed": float(observed_path[sample]),
                    "predicted": float(predicted_path[sample]),
                    "signed_error": float(predicted_path[sample] - observed_path[sample]),
                    "absolute_error": float(absolute_error[sample]),
                }
            )
        grid = _density_grid(observed_path, predictive_law)
        observed_density = _kde(observed_path, grid)
        predicted_density = _kde(predictive_law, grid)
        for value, obs_density, pred_density in zip(
            grid, observed_density, predicted_density, strict=True
        ):
            representative_density_rows.append(
                {
                    "channel": channel,
                    "node_class": node_class,
                    "context_index": FIXED_CONTEXT,
                    "standardized_amplitude": float(value),
                    "observed_density": float(obs_density),
                    "predicted_density": float(pred_density),
                }
            )

    pd.DataFrame(trajectory_rows).to_csv(
        SOURCE_02_04 / "representative_trajectory_and_error.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(representative_density_rows).to_csv(
        SOURCE_02_04 / "representative_distribution_curves.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(metric_rows).to_csv(
        SOURCE_02_04 / "representative_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    node_table = data["node_table"].set_index("channel")
    observed_context = observed[FIXED_CONTEXT]
    predicted_context = predicted[FIXED_CONTEXT]
    global_grid = _density_grid(observed_context, predicted_context, points=320)
    all_metrics: list[dict[str, Any]] = []
    all_curves: list[dict[str, Any]] = []
    for index, channel in enumerate(channels):
        observed_values = observed_context[:, index]
        predicted_values = predicted_context[:, :, index].reshape(-1)
        observed_density = _kde(observed_values, global_grid)
        predicted_density = _kde(predicted_values, global_grid)
        w1 = float(wasserstein_distance(observed_values, predicted_values))
        row = node_table.loc[channel]
        display_class = str(row["display_class"])
        all_metrics.append(
            {
                "channel": channel,
                "channel_index": index,
                "display_class": display_class,
                "selected": bool(row["selected"]),
                "soz": bool(row["soz"]),
                "resection": bool(row["resection"]),
                "occupation_w1": w1,
                "observed_mean": float(np.mean(observed_values)),
                "predicted_mean": float(np.mean(predicted_values)),
                "absolute_mean_error": float(abs(np.mean(predicted_values) - np.mean(observed_values))),
                "observed_sd": float(np.std(observed_values)),
                "predicted_sd": float(np.std(predicted_values)),
                "absolute_sd_error": float(abs(np.std(predicted_values) - np.std(observed_values))),
                "context_index": FIXED_CONTEXT,
                "particles": EXPECTED_PARTICLES,
            }
        )
        for value, obs_density, pred_density in zip(
            global_grid, observed_density, predicted_density, strict=True
        ):
            all_curves.append(
                {
                    "channel": channel,
                    "display_class": display_class,
                    "context_index": FIXED_CONTEXT,
                    "standardized_amplitude": float(value),
                    "observed_density": float(obs_density),
                    "predicted_density": float(pred_density),
                    "occupation_w1": w1,
                }
            )
    pd.DataFrame(all_metrics).to_csv(
        SOURCE_02_04 / "all36_distribution_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(all_curves).to_csv(
        SOURCE_02_04 / "all36_distribution_curves.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _write_ablation_source_data(data: dict[str, Any]) -> None:
    SOURCE_05.mkdir(parents=True, exist_ok=True)
    channels = data["channels"]
    observed = np.asarray(data["observed"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    density_rows: list[dict[str, Any]] = []
    for channel, node_class in NODE_SPECS:
        channel_index = channels.index(channel)
        observed_values = observed[FIXED_CONTEXT, :, channel_index]
        clouds = [observed_values]
        for key, _label in VARIANT_SPECS:
            clouds.append(data[key][FIXED_CONTEXT, :, 1:, channel_index].reshape(-1))
        pooled = np.concatenate(clouds)
        low, high = np.quantile(pooled, [0.005, 0.995])
        padding = 0.08 * max(float(high - low), 0.1)
        grid = np.linspace(low - padding, high + padding, 450)
        observed_density = _kde(observed_values, grid)
        for key, label in VARIANT_SPECS:
            predicted_values = data[key][FIXED_CONTEXT, :, 1:, channel_index].reshape(-1)
            predicted_density = _kde(predicted_values, grid)
            rows.append(
                {
                    "channel": channel,
                    "node_class": node_class,
                    "variant": key,
                    "variant_label": label,
                    "context_index": FIXED_CONTEXT,
                    "horizon_samples": HORIZON,
                    "particles": EXPECTED_PARTICLES,
                    "occupation_w1": float(wasserstein_distance(observed_values, predicted_values)),
                }
            )
            for x_value, density_value in zip(grid, predicted_density, strict=True):
                density_rows.append(
                    {
                        "channel": channel,
                        "node_class": node_class,
                        "variant": key,
                        "variant_label": label,
                        "standardized_amplitude": float(x_value),
                        "density": float(density_value),
                    }
                )
        for x_value, density_value in zip(grid, observed_density, strict=True):
            density_rows.append(
                {
                    "channel": channel,
                    "node_class": node_class,
                    "variant": "observed",
                    "variant_label": "Observed actual",
                    "standardized_amplitude": float(x_value),
                    "density": float(density_value),
                }
            )
    pd.DataFrame(rows).to_csv(
        SOURCE_05 / "fixed_context_ablation_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(density_rows).to_csv(
        SOURCE_05 / "fixed_context_ablation_density_curves.csv",
        index=False,
        encoding="utf-8-sig",
    )


def build_paper_source_data() -> None:
    """Rebuild every Part-II paper table from the frozen model evaluation."""

    load_frozen_model()  # fail early if model provenance is unavailable
    evaluation = _load_frozen_evaluation()
    _write_prediction_source_data(evaluation)
    _write_ablation_source_data(evaluation)


def inspect_model() -> dict[str, Any]:
    model = load_frozen_model()
    return {
        "model": str(MODEL_PATH.relative_to(ROOT)),
        "sha256": _sha256(MODEL_PATH),
        "class": type(model).__name__,
        "latent_dimension": int(model.q),
        "channels": int(np.asarray(model.adjacency).shape[0]),
        "diffusion_mode": str(model.config.diffusion_mode),
        "configuration": asdict(model.config),
        "paper_rollout": str(ROLLOUT_PATH.relative_to(ROOT)),
        "paper_rollout_sha256": _sha256(ROLLOUT_PATH),
    }


def _usage() -> str:
    return (
        "Usage: python part2_model.py {inspect|source-data|train} [options]\n"
        "  inspect      audit the exact frozen paper model\n"
        "  source-data  rebuild Fig. 2--5 CSV tables\n"
        "  train        rerun validation selection and final training\n"
    )


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help", "help"}:
        print(_usage())
        return 0
    command, forwarded = arguments[0], arguments[1:]
    if command == "inspect":
        print(json.dumps(inspect_model(), ensure_ascii=False, indent=2))
        return 0
    if command == "source-data":
        build_paper_source_data()
        print(json.dumps({"figures_02_04": str(SOURCE_02_04), "figure_05": str(SOURCE_05)}, ensure_ascii=False, indent=2))
        return 0
    if command == "train":
        from mfc_pipeline import part2_training

        old_argv = sys.argv
        try:
            sys.argv = [str(Path(__file__).resolve()), *forwarded]
            return int(part2_training.main())
        finally:
            sys.argv = old_argv
    raise SystemExit(f"unknown command {command!r}\n{_usage()}")


if __name__ == "__main__":
    raise SystemExit(main())
