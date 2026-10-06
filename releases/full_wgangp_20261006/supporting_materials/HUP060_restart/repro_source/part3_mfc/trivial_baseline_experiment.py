#!/usr/bin/env python
"""Reviewer-requested trivial-controller and diffusion-shrinkage audit.

This experiment is deliberately separate from policy training.  It opens the
already-frozen HUP060 run-02 evaluation plant, initial Markov state and
innovation bank, then replaces only the control law (or, for the explicitly
labelled sensitivity analysis, the global diffusion multiplier).

The primary proportional controller is selected by input-energy matching
only; no run-02 reference-law score enters that selection.  A post-hoc
reference-oracle proportional row is exported as a transparent development
upper bound and must not be described as an independent test result.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import torch
from torch import Tensor


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from part3_mfc import part3_model as core
from mfc_pipeline.causal_ltv_particle_rollout import (
    FrozenIctalGraphRCBatchStepper,
    LTVPolicyAction,
    reconstruct_paired_normals,
)
from mfc_pipeline.causal_ltv_riccati import FrozenGraphRCMarkovAdapter
from mfc_pipeline.sequential_covariance_hjb import (
    controlled_particle_rollout,
    uncontrolled_particle_rollout,
)
from mfc_pipeline.square_wave_mfc import TorchGraphRCSDE


FULL_PAIR = (
    PROJECT
    / "output"
    / "part3"
    / "source_data"
    / "figures_06_08"
    / "paired_comparison.npz"
)
ARTIFACT_ROOT = PROJECT / "artifacts" / "part3_hup060_trivial_baselines_v1"
SOURCE_ROOT = (
    PROJECT
    / "output"
    / "part3"
    / "source_data"
    / "figure_09_trivial_baselines"
)

K_COARSE = np.asarray(
    [
        0.0,
        0.025,
        0.05,
        0.075,
        0.10,
        0.15,
        0.20,
        0.30,
        0.40,
        0.55,
        0.70,
        0.90,
        1.20,
        1.60,
        2.20,
    ],
    dtype=np.float64,
)
KAPPA_GRID = np.asarray(
    [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90, 1.0, 1.25, 1.50],
    dtype=np.float64,
)
RANDOM_SEED_BASE = 20261107


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def empirical_w1_axis(first: np.ndarray, second: np.ndarray, axis: int = 0) -> np.ndarray:
    """Exact equally weighted empirical W1 along one array axis.

    Sample counts may differ.  The implementation integrates the two sorted
    stepwise quantile functions over their common probability breakpoints and
    vectorizes over all remaining dimensions.
    """

    x = np.moveaxis(np.asarray(first, dtype=np.float64), axis, 0)
    y = np.moveaxis(np.asarray(second, dtype=np.float64), axis, 0)
    if x.shape[1:] != y.shape[1:]:
        raise ValueError("W1 arrays must agree outside the sample axis")
    n, m = int(x.shape[0]), int(y.shape[0])
    if n < 1 or m < 1:
        raise ValueError("W1 sample axes must be non-empty")
    breaks = np.unique(
        np.concatenate(
            [
                np.arange(n + 1, dtype=np.float64) / float(n),
                np.arange(m + 1, dtype=np.float64) / float(m),
            ]
        )
    )
    weights = np.diff(breaks)
    midpoint = 0.5 * (breaks[:-1] + breaks[1:])
    index_x = np.minimum((midpoint * n).astype(np.int64), n - 1)
    index_y = np.minimum((midpoint * m).astype(np.int64), m - 1)
    sorted_x = np.sort(x, axis=0)
    sorted_y = np.sort(y, axis=0)
    reshape = (weights.size,) + (1,) * (x.ndim - 1)
    return np.sum(
        np.abs(sorted_x[index_x] - sorted_y[index_y]) * weights.reshape(reshape),
        axis=0,
    )


def channel_time_w1(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return empirical_w1_axis(predicted, reference, axis=0).mean(axis=0)


def channel_occupation_w1(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    pred = np.asarray(predicted, dtype=np.float64).reshape(-1, predicted.shape[-1])
    ref = np.asarray(reference, dtype=np.float64).reshape(-1, reference.shape[-1])
    return empirical_w1_axis(pred, ref, axis=0)


def channel_time_to_observed(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(predicted - observed[None, :, :]), axis=(0, 1))


def raw_rms(control: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(control, dtype=np.float64) ** 2)))


def control_diagnostics(control: np.ndarray, control_map: np.ndarray) -> dict[str, float]:
    values = np.asarray(control, dtype=np.float64)
    effective = values @ np.asarray(control_map, dtype=np.float64)
    first = np.diff(values, axis=-2)
    second = np.diff(values, n=2, axis=-2)
    return {
        "control_rms": raw_rms(values),
        "effective_electrode_rms": raw_rms(effective),
        "control_peak": float(np.max(np.abs(values))),
        "maximum_first_difference": float(np.max(np.abs(first))) if first.size else 0.0,
        "maximum_second_difference": float(np.max(np.abs(second))) if second.size else 0.0,
        "control_energy": float(np.mean(np.sum(values**2, axis=-1))),
        "saturation_fraction": float(
            np.mean(np.abs(values) >= 0.95 * core.AMPLITUDE_LIMIT)
        ),
    }


@dataclass
class LinearDampingPolicy:
    selected_indices: np.ndarray
    gain: float
    horizon: int = core.HORIZON
    # The frozen Full Actor applies its bounded command directly
    # (actuator_alpha=1).  CONTROL_STEP_SCALE=0.5 belongs to the plant input
    # map and is already carried by FrozenGraphRCMarkovAdapter.
    feedback_alpha: float = 1.0
    amplitude_limit: float = core.AMPLITUDE_LIMIT

    def propose(
        self,
        *,
        step: int,
        markov_particles: Tensor,
        previous_control: Tensor,
        current_scaled: Tensor,
    ) -> LTVPolicyAction:
        del step, markov_particles
        index = torch.as_tensor(
            self.selected_indices, dtype=torch.long, device=current_scaled.device
        )
        command = torch.clamp(
            -float(self.gain) * current_scaled.index_select(1, index),
            -float(self.amplitude_limit),
            float(self.amplitude_limit),
        )
        applied = (
            (1.0 - float(self.feedback_alpha)) * previous_control
            + float(self.feedback_alpha) * command
        )
        common = command.mean(dim=0, keepdim=True)
        return LTVPolicyAction(
            command=command,
            applied_control=applied,
            common_command=common,
            deviation_command=command - common,
            variance_contraction_gate=applied.new_tensor(1.0),
        )


@dataclass
class PrescribedInputPolicy:
    command_bank: Tensor
    horizon: int = core.HORIZON
    feedback_alpha: float = 1.0

    def propose(
        self,
        *,
        step: int,
        markov_particles: Tensor,
        previous_control: Tensor,
        current_scaled: Tensor,
    ) -> LTVPolicyAction:
        del markov_particles, current_scaled
        command = self.command_bank[:, int(step)].to(
            dtype=previous_control.dtype, device=previous_control.device
        )
        if command.shape != previous_control.shape:
            raise ValueError("prescribed command bank and particle batch differ")
        applied = (
            (1.0 - float(self.feedback_alpha)) * previous_control
            + float(self.feedback_alpha) * command
        )
        common = command.mean(dim=0, keepdim=True)
        return LTVPolicyAction(
            command=command,
            applied_control=applied,
            common_command=common,
            deviation_command=command - common,
            variance_contraction_gate=applied.new_tensor(1.0),
        )


def applied_bank(commands: np.ndarray, alpha: float) -> np.ndarray:
    bank = np.asarray(commands, dtype=np.float64)
    previous = np.zeros((bank.shape[0], bank.shape[2]), dtype=np.float64)
    output = np.empty_like(bank)
    for step in range(bank.shape[1]):
        previous = (1.0 - float(alpha)) * previous + float(alpha) * bank[:, step]
        output[:, step] = previous
    return output


def energy_match_command_bank(raw: np.ndarray, target_rms: float) -> tuple[np.ndarray, float]:
    base = np.asarray(raw, dtype=np.float64)
    maximum = float(np.max(np.abs(base)))
    if maximum <= 0:
        raise ValueError("prescribed input bank is identically zero")

    def achieved(scale: float) -> float:
        command = np.clip(
            float(scale) * base, -core.AMPLITUDE_LIMIT, core.AMPLITUDE_LIMIT
        )
        return raw_rms(applied_bank(command, 1.0))

    low, high = 0.0, 1.0
    while achieved(high) < float(target_rms):
        high *= 2.0
        if high * maximum > 100.0:
            raise RuntimeError("could not bracket the target prescribed-input RMS")
    for _ in range(50):
        middle = 0.5 * (low + high)
        if achieved(middle) < float(target_rms):
            low = middle
        else:
            high = middle
    scale = 0.5 * (low + high)
    command = np.clip(
        scale * base, -core.AMPLITUDE_LIMIT, core.AMPLITUDE_LIMIT
    )
    return command, float(scale)


def metric_bundle(
    trajectory: np.ndarray,
    reference: np.ndarray,
    selected: np.ndarray,
    free_time: np.ndarray,
    free_occ: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    time = channel_time_w1(trajectory, reference)
    occupation = channel_occupation_w1(trajectory, reference)
    mask = np.zeros(trajectory.shape[-1], dtype=bool)
    mask[np.asarray(selected, dtype=np.int64)] = True
    summary = {
        "mean_time_w1": float(time.mean()),
        "mean_occupation_w1": float(occupation.mean()),
        "selected_time_w1": float(time[mask].mean()),
        "selected_occupation_w1": float(occupation[mask].mean()),
        "unselected_time_w1": float(time[~mask].mean()),
        "unselected_occupation_w1": float(occupation[~mask].mean()),
        "mean_time_w1_reduction_percent": float(
            100.0 * (1.0 - time.mean() / free_time.mean())
        ),
        "mean_occupation_w1_reduction_percent": float(
            100.0 * (1.0 - occupation.mean() / free_occ.mean())
        ),
    }
    return summary, time, occupation


def rollout_policy(
    stepper: FrozenIctalGraphRCBatchStepper,
    initial: Tensor,
    normals: Tensor,
    policy: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        result = controlled_particle_rollout(stepper, policy, initial, normals)
    return (
        result.scaled[:, 1:].detach().cpu().numpy(),
        result.controls.detach().cpu().numpy(),
        result.commands.detach().cpu().numpy(),
    )


def random_input_banks(
    *,
    kind: str,
    n_seeds: int,
    particles: int,
    horizon: int,
    actuators: int,
    target_rms: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if n_seeds % 2:
        raise ValueError("n_seeds must be even for paired random-input signs")
    rng = np.random.default_rng(RANDOM_SEED_BASE + (0 if kind == "constant" else 1))
    half = n_seeds // 2
    if kind == "constant":
        seed_bank = rng.choice([-1.0, 1.0], size=(half, 1, actuators))
        seed_bank = np.concatenate([seed_bank, -seed_bank], axis=0)
        seed_bank = np.repeat(seed_bank, horizon, axis=1)
    elif kind == "white":
        seed_bank = rng.choice([-1.0, 1.0], size=(half, horizon, actuators))
        seed_bank = np.concatenate([seed_bank, -seed_bank], axis=0)
    else:
        raise ValueError(f"unknown random-input kind: {kind}")
    expanded = np.repeat(seed_bank[:, None], particles, axis=1).reshape(
        n_seeds * particles, horizon, actuators
    )
    command, scale = energy_match_command_bank(expanded, target_rms)
    return command, seed_bank, scale


def summarize_random_variant(
    name: str,
    trajectories: np.ndarray,
    controls: np.ndarray,
    reference: np.ndarray,
    selected: np.ndarray,
    free_time: np.ndarray,
    free_occ: np.ndarray,
    control_map: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    rows: list[dict[str, Any]] = []
    time_values: list[np.ndarray] = []
    occ_values: list[np.ndarray] = []
    for seed in range(trajectories.shape[0]):
        metrics, time, occupation = metric_bundle(
            trajectories[seed], reference, selected, free_time, free_occ
        )
        row: dict[str, Any] = {"variant": name, "technical_seed": seed}
        row.update(metrics)
        row.update(control_diagnostics(controls[seed], control_map))
        rows.append(row)
        time_values.append(time)
        occ_values.append(occupation)
    return rows, np.stack(time_values), np.stack(occ_values)


def moment_audit(
    free: np.ndarray,
    reference: np.ndarray,
    observed: np.ndarray,
    full: np.ndarray,
    channels: np.ndarray,
) -> pd.DataFrame:
    free_pool = free.reshape(-1, free.shape[-1])
    reference_pool = reference.reshape(-1, reference.shape[-1])
    full_pool = full.reshape(-1, full.shape[-1])
    rows: list[dict[str, Any]] = []
    for channel, name in enumerate(channels):
        f = free_pool[:, channel]
        r = reference_pool[:, channel]
        o = observed[:, channel]
        c = full_pool[:, channel]
        mean_f, mean_r = float(f.mean()), float(r.mean())
        sd_f = float(f.std(ddof=1))
        sd_r = float(r.std(ddof=1))
        sd_o = float(o.std(ddof=1))
        sd_c = float(c.std(ddof=1))
        mean_aligned = f - mean_f + mean_r
        sd_aligned = mean_f + (f - mean_f) * (sd_r / max(sd_f, 1.0e-12))
        both_aligned = mean_r + (f - mean_f) * (sd_r / max(sd_f, 1.0e-12))
        d0 = float(wasserstein_distance(f, r))
        d_mu = float(wasserstein_distance(mean_aligned, r))
        d_sigma = float(wasserstein_distance(sd_aligned, r))
        d_both = float(wasserstein_distance(both_aligned, r))
        phi_mean = 0.5 * ((d0 - d_mu) + (d_sigma - d_both))
        phi_sd = 0.5 * ((d0 - d_sigma) + (d_mu - d_both))
        rows.append(
            {
                "channel": str(name),
                "free_to_reference_w1": d0,
                "mean_aligned_residual_w1": d_mu,
                "sd_aligned_residual_w1": d_sigma,
                "mean_sd_aligned_shape_residual_w1": d_both,
                "shapley_location_contribution": phi_mean,
                "shapley_scale_contribution": phi_sd,
                "shape_residual_contribution": d_both,
                "reference_sd": sd_r,
                "observed_ictal_sd": sd_o,
                "free_model_sd": sd_f,
                "full_controlled_sd": sd_c,
                "reference_to_observed_sd_ratio": sd_r / max(sd_o, 1.0e-12),
                "reference_to_free_sd_ratio": sd_r / max(sd_f, 1.0e-12),
                "absolute_full_reference_sd_difference": abs(sd_c - sd_r),
                "absolute_free_reference_sd_difference": abs(sd_f - sd_r),
            }
        )
    return pd.DataFrame(rows)


def aggregate_random(rows: pd.DataFrame, label: str) -> dict[str, Any]:
    subset = rows.loc[rows["variant"] == label]
    result: dict[str, Any] = {
        "variant": label,
        "selection_rule": "median across predeclared paired random seeds; no outcome-based seed selection",
        "n_technical_seeds": int(len(subset)),
    }
    columns = [
        "mean_time_w1",
        "mean_occupation_w1",
        "selected_time_w1",
        "selected_occupation_w1",
        "unselected_time_w1",
        "unselected_occupation_w1",
        "mean_time_w1_reduction_percent",
        "mean_occupation_w1_reduction_percent",
        "control_rms",
        "effective_electrode_rms",
        "control_peak",
        "maximum_first_difference",
        "maximum_second_difference",
        "control_energy",
        "saturation_fraction",
    ]
    for column in columns:
        values = subset[column].to_numpy(dtype=np.float64)
        result[column] = float(np.median(values))
        result[f"{column}_q025"] = float(np.quantile(values, 0.025))
        result[f"{column}_q975"] = float(np.quantile(values, 0.975))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-seeds", type=int, default=32)
    args = parser.parse_args(argv)
    if args.random_seeds < 2 or args.random_seeds % 2:
        parser.error("--random-seeds must be an even integer >= 2")

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)

    synthesis = np.load(core.SYNTHESIS_CONTRACT, allow_pickle=False)
    legal = np.load(core.LEGAL_EVALUATION, allow_pickle=False)
    full_pair = np.load(FULL_PAIR, allow_pickle=False)
    core.load_part2_model_definitions()
    model = joblib.load(core.MODEL_PATH)

    selected = np.asarray(synthesis["selected_indices"], dtype=np.int64)
    channels = np.asarray(synthesis["channels"]).astype(str)
    representatives = np.asarray(synthesis["representative_indices"], dtype=np.int64)
    past_context = np.asarray(synthesis["past_context_scaled"], dtype=np.float64)
    reference = np.asarray(
        synthesis["run01_reference_fit_pool_scaled"], dtype=np.float64
    ).reshape(30, core.HORIZON, 36)[15:]
    observed = np.asarray(legal["observed_evaluation_only_scaled"], dtype=np.float64)

    world = TorchGraphRCSDE(
        model,
        selected,
        core.FS,
        control_graph_diffusion_time=core.GRAPH_DIFFUSION_TIME,
        preserve_physical_control_residual=True,
        dtype=torch.float64,
        device="cpu",
    )
    adapter = FrozenGraphRCMarkovAdapter(
        world,
        control_step_scale=core.CONTROL_STEP_SCALE,
        dtype=torch.float64,
    )
    initial = adapter.initial_state_from_context(past_context)
    stepper = FrozenIctalGraphRCBatchStepper(
        world, adapter, diffusion_scale=core.DIFFUSION_SCALE
    )
    sealed = np.load(core.SEALED_CALIBRATED, allow_pickle=False)
    sealed_no_control = np.asarray(
        sealed["uncontrolled_scaled"][:, 1:], dtype=np.float64
    )
    normals, reconstruction = reconstruct_paired_normals(
        stepper, initial, sealed_no_control, tolerance=2.0e-6
    )

    free = np.asarray(full_pair["uncontrolled_scaled"], dtype=np.float64)
    full = np.asarray(full_pair["candidate_controlled_scaled"], dtype=np.float64)
    full_controls = np.asarray(full_pair["candidate_controls"], dtype=np.float64)
    target_rms = raw_rms(full_controls)
    if float(np.max(np.abs(free - sealed_no_control))) > 2.0e-6:
        raise RuntimeError("frozen no-control law changed before baseline evaluation")
    control_map = (
        core.CONTROL_STEP_SCALE
        * adapter.control_channel_map.detach().cpu().numpy()
    )

    free_time = channel_time_w1(free, reference)
    free_occ = channel_occupation_w1(free, reference)
    exact_time_check = core.channelwise_time_w1(free, reference)
    exact_occ_check = core.channelwise_occupation_w1(free, reference)
    if max(
        float(np.max(np.abs(free_time - exact_time_check))),
        float(np.max(np.abs(free_occ - exact_occ_check))),
    ) > 1.0e-10:
        raise RuntimeError("vectorized W1 implementation failed parity")

    full_metrics, full_time, full_occ = metric_bundle(
        full, reference, selected, free_time, free_occ
    )
    free_metrics, _, _ = metric_bundle(free, reference, selected, free_time, free_occ)

    def evaluate_damping(gain: float, *, keep: bool = False):
        if float(gain) == 0.0:
            trajectory = free
            controls = np.zeros(
                (core.PARTICLES, core.HORIZON, selected.size), dtype=np.float64
            )
            commands = controls.copy()
        else:
            policy = LinearDampingPolicy(selected, float(gain))
            trajectory, controls, commands = rollout_policy(
                stepper, initial, normals, policy
            )
        metrics, time, occupation = metric_bundle(
            trajectory, reference, selected, free_time, free_occ
        )
        row: dict[str, Any] = {"gain": float(gain)}
        row.update(metrics)
        row.update(control_diagnostics(controls, control_map))
        if keep:
            return row, trajectory, controls, commands, time, occupation
        return row

    damping_rows: list[dict[str, Any]] = []
    for gain in K_COARSE:
        row = evaluate_damping(float(gain))
        damping_rows.append(row)
        print(
            f"damping K={gain:.6g}: RMS={row['control_rms']:.6f}, "
            f"occ-W1={row['mean_occupation_w1']:.6f}",
            flush=True,
        )

    ordered = sorted(damping_rows, key=lambda item: float(item["gain"]))
    below = [row for row in ordered if row["control_rms"] <= target_rms]
    above = [row for row in ordered if row["control_rms"] >= target_rms]
    if not below or not above:
        raise RuntimeError("coarse damping grid did not bracket Full control RMS")
    low_gain = float(max(below, key=lambda item: item["control_rms"])["gain"])
    high_gain = float(min(above, key=lambda item: item["control_rms"])["gain"])
    for _ in range(8):
        middle = 0.5 * (low_gain + high_gain)
        row = evaluate_damping(middle)
        damping_rows.append(row)
        print(
            f"damping energy refinement K={middle:.8f}: "
            f"RMS={row['control_rms']:.6f}",
            flush=True,
        )
        if row["control_rms"] < target_rms:
            low_gain = middle
        else:
            high_gain = middle

    closest = min(
        damping_rows,
        key=lambda item: abs(float(item["control_rms"]) - target_rms),
    )
    matched_gain = float(closest["gain"])
    for multiplier in [0.98, 0.99, 1.01, 1.02]:
        local_gain = matched_gain * multiplier
        row = evaluate_damping(local_gain)
        damping_rows.append(row)
        print(
            f"damping local K={local_gain:.8f}: RMS={row['control_rms']:.6f}, "
            f"occ-W1={row['mean_occupation_w1']:.6f}",
            flush=True,
        )

    matched = min(
        damping_rows,
        key=lambda item: abs(float(item["control_rms"]) - target_rms),
    )
    eligible_oracle = [
        row
        for row in damping_rows
        if abs(float(row["control_rms"]) / target_rms - 1.0) <= 0.01
    ]
    if not eligible_oracle:
        eligible_oracle = [matched]
    oracle = min(eligible_oracle, key=lambda item: item["mean_occupation_w1"])
    (
        matched_row,
        damping_trajectory,
        damping_controls,
        damping_commands,
        damping_time,
        damping_occ,
    ) = evaluate_damping(float(matched["gain"]), keep=True)
    (
        oracle_row,
        oracle_trajectory,
        oracle_controls,
        oracle_commands,
        oracle_time,
        oracle_occ,
    ) = evaluate_damping(float(oracle["gain"]), keep=True)

    random_rows: list[dict[str, Any]] = []
    random_channel_rows: list[dict[str, Any]] = []
    representative_random: dict[str, np.ndarray] = {}
    for kind, label in [("constant", "constant_input"), ("white", "white_input")]:
        commands, seed_bank, scale = random_input_banks(
            kind=kind,
            n_seeds=args.random_seeds,
            particles=core.PARTICLES,
            horizon=core.HORIZON,
            actuators=selected.size,
            target_rms=target_rms,
        )
        repeated_normals = normals.repeat(args.random_seeds, 1, 1)
        policy = PrescribedInputPolicy(torch.as_tensor(commands, dtype=torch.float64))
        trajectories_flat, controls_flat, _ = rollout_policy(
            stepper, initial, repeated_normals, policy
        )
        trajectories = trajectories_flat.reshape(
            args.random_seeds, core.PARTICLES, core.HORIZON, 36
        )
        controls = controls_flat.reshape(
            args.random_seeds, core.PARTICLES, core.HORIZON, selected.size
        )
        rows, time_values, occ_values = summarize_random_variant(
            label,
            trajectories,
            controls,
            reference,
            selected,
            free_time,
            free_occ,
            control_map,
        )
        for row in rows:
            row["command_scale"] = scale
        random_rows.extend(rows)
        for seed in range(args.random_seeds):
            for channel, channel_name in enumerate(channels):
                random_channel_rows.append(
                    {
                        "variant": label,
                        "technical_seed": seed,
                        "channel": channel_name,
                        "time_w1": float(time_values[seed, channel]),
                        "occupation_w1": float(occ_values[seed, channel]),
                    }
                )
        representative_random[f"{label}_seed0_scaled"] = trajectories[0]
        representative_random[f"{label}_seed0_controls"] = controls[0]
        print(
            f"{label}: {args.random_seeds} paired seeds, "
            f"median occ-W1={np.median([row['mean_occupation_w1'] for row in rows]):.6f}",
            flush=True,
        )
        del trajectories_flat, controls_flat, trajectories, controls, repeated_normals

    random_df = pd.DataFrame(random_rows)
    random_channel_df = pd.DataFrame(random_channel_rows)

    diffusion_rows: list[dict[str, Any]] = []
    diffusion_rollouts: dict[float, np.ndarray] = {}
    for kappa in KAPPA_GRID:
        numerical_kappa = max(float(kappa), 1.0e-10)
        sensitivity_stepper = FrozenIctalGraphRCBatchStepper(
            world,
            adapter,
            diffusion_scale=core.DIFFUSION_SCALE * numerical_kappa,
        )
        with torch.no_grad():
            trajectory = uncontrolled_particle_rollout(
                sensitivity_stepper, initial, normals
            )[:, 1:].detach().cpu().numpy()
        metrics, time, occupation = metric_bundle(
            trajectory, reference, selected, free_time, free_occ
        )
        observed_time = channel_time_to_observed(trajectory, observed)
        observed_occ = channel_occupation_w1(trajectory, observed[None, :, :])
        row = {
            "kappa_sigma": float(kappa),
            "numerical_kappa_sigma": numerical_kappa,
            **metrics,
            "mean_time_w1_to_observed_ictal": float(observed_time.mean()),
            "mean_occupation_w1_to_observed_ictal": float(observed_occ.mean()),
            "control_rms": 0.0,
        }
        diffusion_rows.append(row)
        diffusion_rollouts[float(kappa)] = trajectory
        print(
            f"diffusion kappa={kappa:.2f}: ref-occ-W1={row['mean_occupation_w1']:.6f}, "
            f"observed-occ-W1={row['mean_occupation_w1_to_observed_ictal']:.6f}",
            flush=True,
        )

    kappa_one = min(diffusion_rows, key=lambda item: abs(item["kappa_sigma"] - 1.0))
    if abs(float(kappa_one["mean_occupation_w1"]) - float(free_occ.mean())) > 1.0e-8:
        raise RuntimeError("kappa=1 failed frozen no-control parity")
    kappa_oracle = min(diffusion_rows, key=lambda item: item["mean_occupation_w1"])

    moment_df = moment_audit(free, reference, observed, full, channels)
    moment_totals = {
        "free_to_reference_w1": float(moment_df["free_to_reference_w1"].mean()),
        "mean_aligned_residual_w1": float(
            moment_df["mean_aligned_residual_w1"].mean()
        ),
        "sd_aligned_residual_w1": float(moment_df["sd_aligned_residual_w1"].mean()),
        "mean_sd_aligned_shape_residual_w1": float(
            moment_df["mean_sd_aligned_shape_residual_w1"].mean()
        ),
        "shapley_location_contribution": float(
            moment_df["shapley_location_contribution"].mean()
        ),
        "shapley_scale_contribution": float(
            moment_df["shapley_scale_contribution"].mean()
        ),
        "shape_residual_contribution": float(
            moment_df["shape_residual_contribution"].mean()
        ),
        "median_reference_to_observed_sd_ratio": float(
            moment_df["reference_to_observed_sd_ratio"].median()
        ),
        "iqr_reference_to_observed_sd_ratio": [
            float(moment_df["reference_to_observed_sd_ratio"].quantile(0.25)),
            float(moment_df["reference_to_observed_sd_ratio"].quantile(0.75)),
        ],
        "median_reference_to_free_sd_ratio": float(
            moment_df["reference_to_free_sd_ratio"].median()
        ),
        "median_abs_full_reference_sd_difference": float(
            moment_df["absolute_full_reference_sd_difference"].median()
        ),
        "median_abs_free_reference_sd_difference": float(
            moment_df["absolute_free_reference_sd_difference"].median()
        ),
    }
    total_gap = moment_totals["free_to_reference_w1"]
    for key in [
        "shapley_location_contribution",
        "shapley_scale_contribution",
        "shape_residual_contribution",
    ]:
        moment_totals[f"{key}_percent"] = float(
            100.0 * moment_totals[key] / total_gap
        )

    summary_rows: list[dict[str, Any]] = []
    free_row: dict[str, Any] = {
        "variant": "free",
        "selection_rule": "frozen Graph-RC-SDE, u=0, kappa_sigma=1",
        **free_metrics,
        **control_diagnostics(
            np.zeros((core.PARTICLES, core.HORIZON, selected.size)), control_map
        ),
    }
    summary_rows.append(free_row)
    summary_rows.append(
        {
            "variant": "diffusion_reference_oracle",
            "selection_rule": (
                "post-hoc run-02 reference-W1 oracle; model-misspecification "
                "sensitivity, not a valid control arm"
            ),
            **kappa_oracle,
            "effective_electrode_rms": 0.0,
            "control_peak": 0.0,
            "maximum_first_difference": 0.0,
            "maximum_second_difference": 0.0,
            "control_energy": 0.0,
            "saturation_fraction": 0.0,
        }
    )
    summary_rows.append(aggregate_random(random_df, "constant_input"))
    summary_rows.append(aggregate_random(random_df, "white_input"))
    summary_rows.append(
        {
            "variant": "linear_damping_rms_matched",
            "selection_rule": (
                "scalar K selected only by absolute raw-RMS mismatch to Full; "
                "run-02 reference W1 not used"
            ),
            **matched_row,
        }
    )
    summary_rows.append(
        {
            "variant": "linear_damping_reference_oracle",
            "selection_rule": (
                "post-hoc minimum occupation W1 among K values within 1% of "
                "Full raw RMS; development upper bound only"
            ),
            **oracle_row,
        }
    )
    full_row: dict[str, Any] = {
        "variant": "full_actor_wgan",
        "selection_rule": "frozen primary Actor-WGAN checkpoint",
        **full_metrics,
        **control_diagnostics(full_controls, control_map),
    }
    summary_rows.append(full_row)
    summary_df = pd.DataFrame(summary_rows)

    full_occ_value = float(full_metrics["mean_occupation_w1"])
    damping_occ_value = float(matched_row["mean_occupation_w1"])
    incremental = 100.0 * (damping_occ_value - full_occ_value) / damping_occ_value
    if full_occ_value < damping_occ_value:
        positioning = (
            "structured policy retains an incremental occupation-W1 benefit "
            f"of {incremental:.2f}% relative to energy-matched scalar damping"
        )
    else:
        positioning = (
            "energy-matched scalar damping matches or exceeds the structured "
            "policy; Part III must be repositioned around incremental rather "
            "than absolute recovery"
        )

    experiment_summary = {
        "status": "reviewer_requested_trivial_baseline_audit",
        "patient": "HUP060",
        "run": "run-02 development evaluation",
        "horizon_samples": core.HORIZON,
        "particles": core.PARTICLES,
        "selected_actuators": int(selected.size),
        "frozen_plant_diffusion_scale": core.DIFFUSION_SCALE,
        "full_raw_control_rms_target": target_rms,
        "paired_noise_reconstruction": reconstruction,
        "primary_linear_damping": {
            "gain": float(matched_row["gain"]),
            "raw_control_rms": float(matched_row["control_rms"]),
            "raw_rms_relative_error": float(
                matched_row["control_rms"] / target_rms - 1.0
            ),
            "mean_time_w1": float(matched_row["mean_time_w1"]),
            "mean_occupation_w1": damping_occ_value,
        },
        "posthoc_linear_damping_oracle": {
            "gain": float(oracle_row["gain"]),
            "mean_time_w1": float(oracle_row["mean_time_w1"]),
            "mean_occupation_w1": float(oracle_row["mean_occupation_w1"]),
            "warning": "development upper bound; not an independent test result",
        },
        "diffusion_sensitivity": {
            "prediction_calibrated_primary_kappa": 1.0,
            "posthoc_reference_oracle_kappa": float(kappa_oracle["kappa_sigma"]),
            "posthoc_reference_oracle_occupation_w1": float(
                kappa_oracle["mean_occupation_w1"]
            ),
            "posthoc_reference_oracle_observed_ictal_occupation_w1": float(
                kappa_oracle["mean_occupation_w1_to_observed_ictal"]
            ),
            "warning": (
                "kappa != 1 changes the frozen predictive plant and is a "
                "misspecification sensitivity, not a control result"
            ),
        },
        "moment_audit": moment_totals,
        "positioning": positioning,
        "claim_guardrail": (
            "report relative reduction in model-to-reference W1, not physiological "
            "recovery; 36 contacts, particles and random seeds are technical units"
        ),
    }

    per_channel_rows: list[dict[str, Any]] = []
    variants_per_channel = {
        "free": (free_time, free_occ),
        "linear_damping_rms_matched": (damping_time, damping_occ),
        "linear_damping_reference_oracle": (oracle_time, oracle_occ),
        "full_actor_wgan": (full_time, full_occ),
    }
    for variant, (time_values, occ_values) in variants_per_channel.items():
        for index, channel in enumerate(channels):
            per_channel_rows.append(
                {
                    "variant": variant,
                    "channel": channel,
                    "selected_actuator": bool(index in set(selected.tolist())),
                    "time_w1": float(time_values[index]),
                    "occupation_w1": float(occ_values[index]),
                }
            )

    damping_df = pd.DataFrame(damping_rows).drop_duplicates(subset=["gain"])
    diffusion_df = pd.DataFrame(diffusion_rows)
    per_channel_df = pd.DataFrame(per_channel_rows)

    summary_df.to_csv(
        SOURCE_ROOT / "trivial_baseline_summary.csv", index=False, encoding="utf-8-sig"
    )
    damping_df.to_csv(
        SOURCE_ROOT / "linear_damping_gain_screen.csv", index=False, encoding="utf-8-sig"
    )
    diffusion_df.to_csv(
        SOURCE_ROOT / "diffusion_scale_sensitivity.csv", index=False, encoding="utf-8-sig"
    )
    random_df.to_csv(
        SOURCE_ROOT / "random_input_seed_metrics.csv", index=False, encoding="utf-8-sig"
    )
    random_channel_df.to_csv(
        SOURCE_ROOT / "random_input_per_channel.csv", index=False, encoding="utf-8-sig"
    )
    moment_df.to_csv(
        SOURCE_ROOT / "location_scale_moment_audit.csv", index=False, encoding="utf-8-sig"
    )
    per_channel_df.to_csv(
        SOURCE_ROOT / "primary_baselines_per_channel.csv", index=False, encoding="utf-8-sig"
    )
    (SOURCE_ROOT / "experiment_summary.json").write_text(
        json.dumps(json_ready(experiment_summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    np.savez_compressed(
        SOURCE_ROOT / "trivial_baseline_rollouts.npz",
        observed_scaled=observed,
        reference_validation_scaled=reference,
        free_scaled=free,
        full_scaled=full,
        full_controls=full_controls,
        linear_damping_scaled=damping_trajectory,
        linear_damping_controls=damping_controls,
        linear_damping_commands=damping_commands,
        oracle_damping_scaled=oracle_trajectory,
        oracle_damping_controls=oracle_controls,
        oracle_damping_commands=oracle_commands,
        diffusion_oracle_scaled=diffusion_rollouts[float(kappa_oracle["kappa_sigma"])],
        representative_indices=representatives,
        selected_indices=selected,
        channels=channels,
        **representative_random,
    )
    for filename in [
        "trivial_baseline_summary.csv",
        "linear_damping_gain_screen.csv",
        "diffusion_scale_sensitivity.csv",
        "random_input_seed_metrics.csv",
        "random_input_per_channel.csv",
        "location_scale_moment_audit.csv",
        "primary_baselines_per_channel.csv",
        "experiment_summary.json",
        "trivial_baseline_rollouts.npz",
    ]:
        source = SOURCE_ROOT / filename
        target = ARTIFACT_ROOT / filename
        target.write_bytes(source.read_bytes())

    print(json.dumps(json_ready(experiment_summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
