#!/usr/bin/env python
"""Validation-selected state-dependent stochastic RC experiment for HUP060."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import itertools
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


from mfc_pipeline import part2_state_dependent_rc_sde as P2  # noqa: E402

BASE = P2.BASE


def model_grid(spec: dict) -> list:
    common = dict(
        reservoir_sparsity=float(spec["reservoir_sparsity"]),
        washout_samples=int(spec["washout_samples"]),
        random_seed=int(spec["random_seed"]),
        latent_variance_threshold=float(spec["latent_variance_threshold"]),
        latent_components_min=int(spec["latent_components_min"]),
        latent_components_max=int(spec["latent_components_max"]),
        diffusion_shrinkage=float(spec["diffusion_shrinkage"]),
        diffusion_relative_jitter=float(spec["diffusion_relative_jitter"]),
        diffusion_mode="state_dependent",
        diffusion_ridge_alpha=10.0,
    )
    return [
        P2.ModelConfig(
            reservoir_size=int(size),
            spectral_radius=float(radius),
            leak_rate=float(leak),
            input_scale=float(scale),
            delays_samples=tuple(int(v) for v in delays),
            ridge_alpha=float(ridge),
            **common,
        )
        for size, radius, leak, scale, delays, ridge in itertools.product(
            spec["reservoir_sizes"],
            spec["spectral_radii"],
            spec["leak_rates"],
            spec["input_scales"],
            spec["delay_sets_samples"],
            spec["ridge_alphas"],
        )
    ]


def load_plan(spec: dict):
    config = BASE.load_config(PROJECT / "config_v2.yaml")
    proposals, meta = BASE.load_split_proposal(BASE.DEFAULT_SPLIT_PROPOSAL, config)
    dataset, loader = BASE.build_loader(config)
    subject = str(spec["subject"])
    return BASE.load_ictal_plan(
        config, dataset, loader, subject, proposals[subject], meta["sha256"]
    )


def direct_validation_objective(metrics: pd.DataFrame) -> float:
    aggregate = metrics.groupby("horizon_samples").mean(numeric_only=True)
    return float(
        0.40 * aggregate.loc[16, "nrmse"]
        + 0.35 * aggregate.loc[32, "nrmse"]
        + 0.15 * aggregate.loc[64, "nrmse"]
        + 0.10 * aggregate.loc[128, "swd_ratio_to_persistence"]
    )


def distribution_metrics(
    model,
    sequence: np.ndarray,
    context_samples: int,
    horizon: int,
    windows: int,
    rollouts: int,
    seed: int,
    diffusion_scale: float = 1.0,
) -> tuple[pd.DataFrame, list[dict[str, np.ndarray]]]:
    positions = P2.forecast_positions(
        len(sequence),
        context_samples=context_samples,
        maximum_horizon=horizon,
        guard_samples=0,
        count=windows,
    )
    projections = P2.fixed_random_projections(
        sequence.shape[1], n_projections=128, seed=seed
    )
    assert model.transform is not None
    rows = []
    payload = []
    for window, position in enumerate(positions):
        context = sequence[position - context_samples : position]
        truth = sequence[position : position + horizon]
        truth_scaled = model.transform.scaler.transform(truth)
        samples = model.sample_direct_scaled(
            context,
            horizon,
            rollouts,
            seed + 100 + window,
            diffusion_scale=diffusion_scale,
        )
        sample_cloud = samples.reshape(-1, samples.shape[-1])
        persistence = np.repeat(
            model.transform.scaler.transform(context[-1:]), horizon, axis=0
        )
        swd = P2.sliced_wasserstein_distance(
            sample_cloud, truth_scaled, projections=projections
        )
        persistence_swd = P2.sliced_wasserstein_distance(
            persistence, truth_scaled, projections=projections
        )
        rows.append(
            {
                "window": window,
                "forecast_boundary_sample": int(position),
                "state_dependent_swd": float(swd),
                "persistence_swd": float(persistence_swd),
                "swd_ratio_to_persistence": float(swd / persistence_swd),
            }
        )
        payload.append(
            {
                "truth_pc1": truth_scaled @ model.transform.components_[0],
                "prediction_pc1": sample_cloud @ model.transform.components_[0],
                "truth_scaled": truth_scaled,
                "prediction_scaled": sample_cloud,
            }
        )
    return pd.DataFrame(rows), payload


def rolling_block_predictions(
    model,
    sequence: np.ndarray,
    positions: list[int],
    context_samples: int,
    block_samples: int,
    total_samples: int,
    channels: list[str],
    sfreq: float,
) -> pd.DataFrame:
    assert model.transform is not None
    rows = []
    for window, position in enumerate(positions):
        blocks = []
        for start in range(0, total_samples, block_samples):
            boundary = position + start
            context = sequence[boundary - context_samples : boundary]
            steps = min(block_samples, total_samples - start)
            blocks.append(model.forecast_direct(context, steps))
        prediction = np.vstack(blocks)
        truth = sequence[position : position + total_samples]
        truth_scaled = model.transform.scaler.transform(truth)
        prediction_scaled = model.transform.scaler.transform(prediction)
        for channel_index, channel in enumerate(channels):
            observed = truth_scaled[:, channel_index]
            estimated = prediction_scaled[:, channel_index]
            nrmse = float(
                np.sqrt(np.mean((estimated - observed) ** 2))
                / max(float(np.std(observed)), 1e-8)
            )
            correlation = (
                float(np.corrcoef(observed, estimated)[0, 1])
                if min(np.std(observed), np.std(estimated)) > 1e-8
                else float("nan")
            )
            for sample in range(total_samples):
                rows.append(
                    {
                        "window": window,
                        "sample": sample + 1,
                        "time_ms": 1000.0 * (sample + 1) / sfreq,
                        "channel_index": channel_index,
                        "channel": channel,
                        "observed_standardized": observed[sample],
                        "predicted_standardized": estimated[sample],
                        "channel_window_nrmse": nrmse,
                        "channel_window_correlation": correlation,
                        "forecast_protocol": f"rolling_{block_samples}_sample_blocks",
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "part2_hup060_state_dependent_optimized.yaml",
    )
    parser.add_argument("--reuse-grid", action="store_true")
    args = parser.parse_args()
    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = PROJECT / spec["output_dir"]
    figure_dir = PROJECT / spec["figure_dir"]
    source_dir = figure_dir / "source_data"
    output.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    plan = load_plan(spec)
    fold = plan["folds"][0]
    adjacency_validation = BASE.compute_adjacency(
        fold["train_sequences"], plan["sfreq"]
    )
    grid_path = output / "direct_validation_grid.csv"
    if args.reuse_grid:
        validation = pd.read_csv(grid_path)
    else:
        candidates = model_grid(spec)
        records = []
        print(f"Direct validation grid: {len(candidates)} candidates", flush=True)
        for index, config in enumerate(candidates):
            started = time.perf_counter()
            model = P2.ResidualGraphRCSDE(config, adjacency_validation).fit(
                fold["train_sequences"]
            )
            metrics = P2.evaluate_candidate(
                model,
                fold["validation_sequence"],
                [int(v) for v in spec["validation_horizons_samples"]],
                int(spec["context_samples"]),
                int(spec["validation_windows"]),
                20261001,
                forecast_mode="direct",
            )
            aggregate = metrics.groupby("horizon_samples").mean(numeric_only=True)
            records.append(
                {
                    "candidate_index": index,
                    **asdict(config),
                    "latent_components": model.q,
                    "validation_nrmse_16": float(aggregate.loc[16, "nrmse"]),
                    "validation_nrmse_32": float(aggregate.loc[32, "nrmse"]),
                    "validation_nrmse_64": float(aggregate.loc[64, "nrmse"]),
                    "validation_nrmse_128": float(aggregate.loc[128, "nrmse"]),
                    "validation_swd_ratio_128": float(
                        aggregate.loc[128, "swd_ratio_to_persistence"]
                    ),
                    "validation_objective": direct_validation_objective(metrics),
                    "fit_score_seconds": time.perf_counter() - started,
                }
            )
            if (index + 1) % 12 == 0 or index + 1 == len(candidates):
                print(f"validated {index + 1}/{len(candidates)}", flush=True)
        validation = pd.DataFrame(records)
        validation.to_csv(grid_path, index=False, encoding="utf-8-sig")
    selected = validation.sort_values(
        ["validation_objective", "candidate_index"], kind="mergesort"
    ).iloc[0]
    delays = selected["delays_samples"]
    if isinstance(delays, str):
        import ast

        delays = ast.literal_eval(delays)
    selected_config = P2.ModelConfig(
        reservoir_size=int(selected["reservoir_size"]),
        spectral_radius=float(selected["spectral_radius"]),
        leak_rate=float(selected["leak_rate"]),
        input_scale=float(selected["input_scale"]),
        delays_samples=tuple(int(v) for v in delays),
        ridge_alpha=float(selected["ridge_alpha"]),
        reservoir_sparsity=float(selected["reservoir_sparsity"]),
        washout_samples=int(selected["washout_samples"]),
        random_seed=int(selected["random_seed"]),
        latent_variance_threshold=float(selected["latent_variance_threshold"]),
        latent_components_min=int(selected["latent_components_min"]),
        latent_components_max=int(selected["latent_components_max"]),
        diffusion_shrinkage=float(selected["diffusion_shrinkage"]),
        diffusion_relative_jitter=float(selected["diffusion_relative_jitter"]),
        diffusion_mode="state_dependent",
        diffusion_ridge_alpha=10.0,
    )
    print("Selected direct model:", json.dumps(asdict(selected_config)), flush=True)

    alpha_records = []
    for alpha in spec["diffusion_ridge_alphas"]:
        config = replace(selected_config, diffusion_ridge_alpha=float(alpha))
        model = P2.ResidualGraphRCSDE(config, adjacency_validation).fit(
            fold["train_sequences"]
        )
        for scale in spec["diffusion_scales"]:
            metrics, _ = distribution_metrics(
                model,
                fold["validation_sequence"],
                int(spec["context_samples"]),
                int(spec["distribution_horizon_samples"]),
                int(spec["validation_windows"]),
                int(spec["validation_rollouts"]),
                20261002,
                diffusion_scale=float(scale),
            )
            alpha_records.append(
                {
                    "diffusion_ridge_alpha": float(alpha),
                    "diffusion_scale": float(scale),
                    "mean_swd_ratio_to_persistence": float(
                        metrics["swd_ratio_to_persistence"].mean()
                    ),
                    "median_swd": float(metrics["state_dependent_swd"].median()),
                    "improved_windows": int(
                        (metrics["swd_ratio_to_persistence"] < 1.0).sum()
                    ),
                }
            )
        print(f"diffusion alpha {alpha} scored", flush=True)
    alpha_screen = pd.DataFrame(alpha_records)
    alpha_screen.to_csv(
        output / "diffusion_validation_screen.csv", index=False, encoding="utf-8-sig"
    )
    best_state_row = alpha_screen.sort_values(
            ["mean_swd_ratio_to_persistence", "diffusion_ridge_alpha"],
            kind="mergesort",
        ).iloc[0]
    best_alpha = float(best_state_row["diffusion_ridge_alpha"])
    best_state_scale = float(best_state_row["diffusion_scale"])
    selected_config = replace(selected_config, diffusion_ridge_alpha=best_alpha)

    validation_horizon_model = P2.ResidualGraphRCSDE(
        selected_config, adjacency_validation
    ).fit(fold["train_sequences"])
    validation_positions = P2.forecast_positions(
        len(fold["validation_sequence"]),
        context_samples=int(spec["context_samples"]),
        maximum_horizon=int(spec["rolling_display_samples"]),
        guard_samples=0,
        count=int(spec["validation_windows"]),
    )
    rolling_screen_records = []
    for block_samples in spec["rolling_block_candidates"]:
        rolling_validation = rolling_block_predictions(
            validation_horizon_model,
            fold["validation_sequence"],
            validation_positions,
            int(spec["context_samples"]),
            int(block_samples),
            int(spec["rolling_display_samples"]),
            [str(v) for v in plan["channels"]],
            float(plan["sfreq"]),
        )
        independent_metrics = rolling_validation.drop_duplicates(
            ["window", "channel"]
        )
        rolling_screen_records.append(
            {
                "block_samples": int(block_samples),
                "block_ms": 1000.0 * float(block_samples) / float(plan["sfreq"]),
                "median_nrmse": float(
                    independent_metrics["channel_window_nrmse"].median()
                ),
                "median_correlation": float(
                    independent_metrics["channel_window_correlation"].median()
                ),
            }
        )
    rolling_screen = pd.DataFrame(rolling_screen_records)
    rolling_screen.to_csv(
        output / "rolling_horizon_validation_screen.csv",
        index=False,
        encoding="utf-8-sig",
    )
    eligible_horizons = rolling_screen[
        rolling_screen["median_correlation"].ge(
            float(spec["rolling_selection_min_correlation"])
        )
        & rolling_screen["median_nrmse"].le(
            float(spec["rolling_selection_max_nrmse"])
        )
    ]
    if eligible_horizons.empty:
        selected_block_samples = int(rolling_screen.iloc[0]["block_samples"])
    else:
        selected_block_samples = int(
            eligible_horizons.sort_values("block_samples").iloc[-1]["block_samples"]
        )

    constant_validation_model = P2.ResidualGraphRCSDE(
        replace(selected_config, diffusion_mode="constant"), adjacency_validation
    ).fit(fold["train_sequences"])
    constant_scale_records = []
    for scale in spec["diffusion_scales"]:
        metrics, _ = distribution_metrics(
            constant_validation_model,
            fold["validation_sequence"],
            int(spec["context_samples"]),
            int(spec["distribution_horizon_samples"]),
            int(spec["validation_windows"]),
            int(spec["validation_rollouts"]),
            20261002,
            diffusion_scale=float(scale),
        )
        constant_scale_records.append(
            {
                "diffusion_scale": float(scale),
                "mean_swd_ratio_to_persistence": float(
                    metrics["swd_ratio_to_persistence"].mean()
                ),
                "median_swd": float(metrics["state_dependent_swd"].median()),
            }
        )
    constant_scale_screen = pd.DataFrame(constant_scale_records)
    constant_scale_screen.to_csv(
        output / "constant_diffusion_validation_screen.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best_constant_scale = float(
        constant_scale_screen.sort_values(
            ["mean_swd_ratio_to_persistence", "diffusion_scale"],
            kind="mergesort",
        ).iloc[0]["diffusion_scale"]
    )

    adjacency_final = BASE.compute_adjacency(
        plan["final_train_sequences"], plan["sfreq"]
    )
    model = P2.ResidualGraphRCSDE(selected_config, adjacency_final).fit(
        plan["final_train_sequences"]
    )
    constant_model = P2.ResidualGraphRCSDE(
        replace(selected_config, diffusion_mode="constant"), adjacency_final
    ).fit(plan["final_train_sequences"])
    joblib.dump(model, output / "hup060_state_dependent_rc.joblib", compress=3)

    test_positions_for_screen = P2.forecast_positions(
        len(plan["test_sequence"]),
        context_samples=int(spec["context_samples"]),
        maximum_horizon=int(spec["rolling_display_samples"]),
        guard_samples=0,
        count=int(spec["test_windows"]),
    )
    rolling_test_records = []
    for block_samples in spec["rolling_block_candidates"]:
        rolling_test = rolling_block_predictions(
            model,
            plan["test_sequence"],
            test_positions_for_screen,
            int(spec["context_samples"]),
            int(block_samples),
            int(spec["rolling_display_samples"]),
            [str(v) for v in plan["channels"]],
            float(plan["sfreq"]),
        )
        independent_metrics = rolling_test.drop_duplicates(["window", "channel"])
        rolling_test_records.append(
            {
                "block_samples": int(block_samples),
                "block_ms": 1000.0 * float(block_samples) / float(plan["sfreq"]),
                "median_nrmse": float(
                    independent_metrics["channel_window_nrmse"].median()
                ),
                "median_correlation": float(
                    independent_metrics["channel_window_correlation"].median()
                ),
            }
        )
    rolling_test_screen = pd.DataFrame(rolling_test_records)
    rolling_test_screen.to_csv(
        output / "rolling_horizon_test_screen.csv",
        index=False,
        encoding="utf-8-sig",
    )

    horizons = [1, 16, 32, 64, 128]
    test_metrics = P2.evaluate_candidate(
        model,
        plan["test_sequence"],
        horizons,
        int(spec["context_samples"]),
        int(spec["test_windows"]),
        20261003,
        forecast_mode="direct",
    )
    test_metrics.to_csv(
        output / "test_open_loop_metrics.csv", index=False, encoding="utf-8-sig"
    )
    state_distribution, distribution_payload = distribution_metrics(
        model,
        plan["test_sequence"],
        int(spec["context_samples"]),
        int(spec["distribution_horizon_samples"]),
        int(spec["test_windows"]),
        int(spec["test_rollouts"]),
        20261004,
        diffusion_scale=best_state_scale,
    )
    constant_distribution, constant_payload = distribution_metrics(
        constant_model,
        plan["test_sequence"],
        int(spec["context_samples"]),
        int(spec["distribution_horizon_samples"]),
        int(spec["test_windows"]),
        int(spec["test_rollouts"]),
        20261004,
        diffusion_scale=best_constant_scale,
    )
    state_distribution["constant_diffusion_swd"] = constant_distribution[
        "state_dependent_swd"
    ]
    state_distribution.to_csv(
        output / "test_distribution_metrics.csv", index=False, encoding="utf-8-sig"
    )

    positions = P2.forecast_positions(
        len(plan["test_sequence"]),
        context_samples=int(spec["context_samples"]),
        maximum_horizon=int(spec["rolling_display_samples"]),
        guard_samples=0,
        count=int(spec["test_windows"]),
    )
    trajectories = rolling_block_predictions(
        model,
        plan["test_sequence"],
        positions,
        int(spec["context_samples"]),
        selected_block_samples,
        int(spec["rolling_display_samples"]),
        [str(v) for v in plan["channels"]],
        float(plan["sfreq"]),
    )
    trajectories.to_csv(
        source_dir / "hup060_rolling_multistep_trajectories.csv",
        index=False,
        encoding="utf-8-sig",
    )

    soz, nonsoz = P2.representative_channels([str(v) for v in plan["channels"]])
    display_window = int(spec["representative_window_index"])
    display = trajectories[trajectories["window"].eq(display_window)]
    channel_lookup = {str(channel): index for index, channel in enumerate(plan["channels"])}
    node_distributions: dict[str, dict[str, np.ndarray | float]] = {}
    node_distribution_rows = []
    for channel in (soz, nonsoz):
        channel_index = channel_lookup[channel]
        observed = np.concatenate(
            [item["truth_scaled"][:, channel_index] for item in distribution_payload]
        )
        predicted = np.concatenate(
            [
                item["prediction_scaled"][:, channel_index]
                for item in distribution_payload
            ]
        )
        constant_predicted = np.concatenate(
            [
                item["prediction_scaled"][:, channel_index]
                for item in constant_payload
            ]
        )
        node_distributions[channel] = {
            "observed": observed,
            "predicted": predicted,
            "constant_predicted": constant_predicted,
            "wasserstein_1d": float(wasserstein_distance(observed, predicted)),
            "constant_wasserstein_1d": float(
                wasserstein_distance(observed, constant_predicted)
            ),
        }
        node_distribution_rows.extend(
            {
                "channel": channel,
                "sample_type": "observed",
                "standardized_eeg": float(value),
            }
            for value in observed
        )
        node_distribution_rows.extend(
            {
                "channel": channel,
                "sample_type": "state_dependent_prediction",
                "standardized_eeg": float(value),
            }
            for value in predicted
        )
        node_distribution_rows.extend(
            {
                "channel": channel,
                "sample_type": "constant_diffusion_prediction",
                "standardized_eeg": float(value),
            }
            for value in constant_predicted
        )
    pd.DataFrame(node_distribution_rows).to_csv(
        source_dir / "hup060_node_state_distributions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    wins = int(
        (state_distribution["state_dependent_swd"]
         < state_distribution["constant_diffusion_swd"]).sum()
    )
    reduction = 1.0 - (
        state_distribution["state_dependent_swd"]
        / state_distribution["persistence_swd"]
    ).median()
    ablation_reduction = 1.0 - (
        state_distribution["state_dependent_swd"]
        / state_distribution["constant_diffusion_swd"]
    ).median()

    P2.style()
    fig, axes = P2.plt.subplots(2, 2, figsize=(7.20, 5.20))
    ax_dist_soz, ax_dist_nonsoz = axes[0]
    ax_soz, ax_nonsoz = axes[1]

    for ax, channel, title, title_color, label in (
        (ax_dist_soz, soz, f"SOZ distribution: {soz}", "#B64342", "a"),
        (ax_dist_nonsoz, nonsoz, f"Non-SOZ distribution: {nonsoz}", "#155A92", "b"),
    ):
        observed = np.asarray(node_distributions[channel]["observed"], dtype=float)
        predicted = np.asarray(node_distributions[channel]["predicted"], dtype=float)
        lo, hi = np.quantile(np.concatenate([observed, predicted]), [0.005, 0.995])
        density_grid = np.linspace(lo, hi, 500)
        predicted_density = P2.kde_curve(predicted, density_grid)
        ax.plot(
            density_grid,
            P2.kde_curve(observed, density_grid),
            color="#252525",
            lw=1.7,
            label="Observed",
        )
        ax.plot(
            density_grid,
            predicted_density,
            color="#2E6FB5",
            lw=1.7,
            label="State-dependent RC",
        )
        ax.fill_between(
            density_grid, 0, predicted_density, color="#2E6FB5", alpha=0.12
        )
        ax.text(
            0.97,
            0.94,
            f"W1={float(node_distributions[channel]['wasserstein_1d']):.3f}\n8 fixed windows",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=5.7,
        )
        ax.set_xlabel("Standardized EEG amplitude")
        ax.set_ylabel("Density")
        ax.set_title(title, color=title_color)
        P2.panel_label(ax, label)
    ax_dist_soz.legend(loc="upper left", fontsize=5.4)

    for ax, channel, title, color in (
        (ax_soz, soz, f"SOZ trajectory: {soz}", "#B64342"),
        (ax_nonsoz, nonsoz, f"Non-SOZ trajectory: {nonsoz}", "#155A92"),
    ):
        values = display[display["channel"].astype(str).eq(channel)]
        ax.plot(
            values["time_ms"], values["observed_standardized"],
            color="#252525", lw=1.2, label="Observed"
        )
        ax.plot(
            values["time_ms"], values["predicted_standardized"],
            color="#2E6FB5", lw=1.3, label="RC prediction"
        )
        metric = values.iloc[0]
        ax.text(
            0.98, 0.95,
            f"nRMSE={metric.channel_window_nrmse:.2f}\nr={metric.channel_window_correlation:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.7
        )
        ax.set_xlabel("Held-out time (ms)")
        ax.set_ylabel("Standardized EEG")
        selected_block_ms = 1000.0 * selected_block_samples / float(plan["sfreq"])
        ax.set_title(
            f"{title}\nRolling {selected_block_ms:.1f}-ms forecasts", color=color
        )
        ax.grid(axis="x", color="#DEDEDE", lw=0.45)
    ax_soz.legend(loc="lower left", fontsize=5.3)
    P2.panel_label(ax_soz, "c")
    P2.panel_label(ax_nonsoz, "d")
    fig.subplots_adjust(
        left=0.08, right=0.985, bottom=0.10, top=0.95, hspace=0.42, wspace=0.30
    )
    P2.save_figure(
        fig, figure_dir / "hup060_node_specific_distribution_and_trajectories"
    )

    node_ablation_rows = []
    for window, (state_item, constant_item) in enumerate(
        zip(distribution_payload, constant_payload, strict=True)
    ):
        for channel in (soz, nonsoz):
            channel_index = channel_lookup[channel]
            observed = state_item["truth_scaled"][:, channel_index]
            state_values = state_item["prediction_scaled"][:, channel_index]
            constant_values = constant_item["prediction_scaled"][:, channel_index]
            node_ablation_rows.append(
                {
                    "window": int(window),
                    "channel": channel,
                    "node_class": "SOZ" if channel == soz else "Non-SOZ",
                    "constant_diffusion_w1": float(
                        wasserstein_distance(observed, constant_values)
                    ),
                    "state_dependent_diffusion_w1": float(
                        wasserstein_distance(observed, state_values)
                    ),
                }
            )
    node_ablation = pd.DataFrame(node_ablation_rows)
    node_ablation["relative_w1_reduction"] = 1.0 - (
        node_ablation["state_dependent_diffusion_w1"]
        / node_ablation["constant_diffusion_w1"]
    )
    node_ablation.to_csv(
        source_dir / "hup060_node_diffusion_ablation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig_ablation, axes_ablation = P2.plt.subplots(2, 2, figsize=(7.20, 5.20))
    ax_density_soz, ax_density_nonsoz = axes_ablation[0]
    ax_pair_soz, ax_pair_nonsoz = axes_ablation[1]
    for ax_density, ax_pair, channel, title, title_color, labels in (
        (
            ax_density_soz,
            ax_pair_soz,
            soz,
            f"SOZ diffusion ablation: {soz}",
            "#B64342",
            ("a", "c"),
        ),
        (
            ax_density_nonsoz,
            ax_pair_nonsoz,
            nonsoz,
            f"Non-SOZ diffusion ablation: {nonsoz}",
            "#155A92",
            ("b", "d"),
        ),
    ):
        observed = np.asarray(node_distributions[channel]["observed"], dtype=float)
        constant_values = np.asarray(
            node_distributions[channel]["constant_predicted"], dtype=float
        )
        state_values = np.asarray(
            node_distributions[channel]["predicted"], dtype=float
        )
        lo, hi = np.quantile(
            np.concatenate([observed, constant_values, state_values]),
            [0.005, 0.995],
        )
        density_grid = np.linspace(lo, hi, 500)
        ax_density.plot(
            density_grid,
            P2.kde_curve(observed, density_grid),
            color="#252525",
            lw=1.7,
            label="Observed",
        )
        ax_density.plot(
            density_grid,
            P2.kde_curve(constant_values, density_grid),
            color="#C78A3B",
            lw=1.25,
            ls="--",
            label="Constant diffusion",
        )
        ax_density.plot(
            density_grid,
            P2.kde_curve(state_values, density_grid),
            color="#2E6FB5",
            lw=1.7,
            label="State-dependent diffusion",
        )
        ax_density.set_xlabel("Standardized EEG amplitude")
        ax_density.set_ylabel("Density")
        ax_density.set_title(title, color=title_color)
        P2.panel_label(ax_density, labels[0])

        values = node_ablation[node_ablation["channel"].eq(channel)]
        for row in values.itertuples(index=False):
            ax_pair.plot(
                [0, 1],
                [row.constant_diffusion_w1, row.state_dependent_diffusion_w1],
                color="#B0B0B0",
                lw=0.8,
                zorder=1,
            )
            ax_pair.scatter(
                0, row.constant_diffusion_w1, color="#C78A3B", s=20, zorder=2
            )
            ax_pair.scatter(
                1,
                row.state_dependent_diffusion_w1,
                color="#2E6FB5",
                s=20,
                zorder=2,
            )
        node_wins = int(
            (
                values["state_dependent_diffusion_w1"]
                < values["constant_diffusion_w1"]
            ).sum()
        )
        node_reduction = float(values["relative_w1_reduction"].median())
        ax_pair.text(
            0.50,
            0.97,
            f"Wins: {node_wins}/8\nmedian reduction: {100*node_reduction:.1f}%",
            transform=ax_pair.transAxes,
            ha="center",
            va="top",
            fontsize=5.7,
        )
        ax_pair.set_xticks([0, 1], ["Constant", "State-dependent"])
        ax_pair.set_ylabel("Node-wise Wasserstein distance")
        ax_pair.set_title("Eight fixed windows", color=title_color)
        P2.panel_label(ax_pair, labels[1])
    ax_density_soz.legend(loc="upper left", fontsize=5.2)
    fig_ablation.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.10,
        top=0.95,
        hspace=0.42,
        wspace=0.30,
    )
    P2.save_figure(
        fig_ablation, figure_dir / "hup060_node_specific_diffusion_ablation"
    )

    aggregate = test_metrics.groupby("horizon_samples", as_index=False).mean(
        numeric_only=True
    )

    representative_metrics = (
        trajectories[
            trajectories["channel"].isin([soz, nonsoz])
        ]
        .drop_duplicates(["window", "channel"])
        .groupby("channel")[["channel_window_nrmse", "channel_window_correlation"]]
        .median()
    )
    summary = {
        "subject": str(spec["subject"]),
        "hup060_role": "development_and_illustrative_case_after_model_revision",
        "test_used_in_grid_or_diffusion_alpha_selection": False,
        "selected_model": asdict(selected_config),
        "selected_state_dependent_diffusion_scale": best_state_scale,
        "selected_constant_diffusion_scale": best_constant_scale,
        "rolling_horizon_validation_screen": rolling_screen.to_dict(
            orient="records"
        ),
        "rolling_horizon_test_screen_descriptive": rolling_test_screen.to_dict(
            orient="records"
        ),
        "selected_rolling_block_samples": selected_block_samples,
        "selected_rolling_block_ms": 1000.0
        * selected_block_samples
        / float(plan["sfreq"]),
        "selected_validation_candidate": selected.to_dict(),
        "diffusion_validation_screen": alpha_screen.to_dict(orient="records"),
        "constant_diffusion_validation_screen": constant_scale_screen.to_dict(
            orient="records"
        ),
        "test_distribution": {
            "state_dependent_median_swd": float(
                state_distribution["state_dependent_swd"].median()
            ),
            "constant_median_swd": float(
                state_distribution["constant_diffusion_swd"].median()
            ),
            "persistence_median_swd": float(
                state_distribution["persistence_swd"].median()
            ),
            "state_dependent_wins_vs_constant": wins,
            "median_reduction_vs_constant": float(ablation_reduction),
            "median_reduction_vs_persistence": float(reduction),
        },
        "rolling_multistep_representatives": representative_metrics.to_dict(
            orient="index"
        ),
        "node_diffusion_ablation": (
            node_ablation.groupby("channel")
            .agg(
                wins=(
                    "relative_w1_reduction",
                    lambda values: int((values > 0).sum()),
                ),
                median_relative_w1_reduction=("relative_w1_reduction", "median"),
                median_constant_w1=("constant_diffusion_w1", "median"),
                median_state_dependent_w1=(
                    "state_dependent_diffusion_w1",
                    "median",
                ),
            )
            .to_dict(orient="index")
        ),
        "test_open_loop_summary": aggregate.to_dict(orient="records"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
