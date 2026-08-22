"""Leakage-safe closed-loop exposure calibration for a frozen Graph--RC SDE.

The one-step residual readout is trained under teacher forcing.  During a
one-second free rollout it is instead exposed to its own states, so a small
systematic increment bias can accumulate and spuriously return the model law
towards the interictal reference.  This module provides a deliberately small
calibration wrapper that changes neither the sampling rate nor the fitted
state-dependent diffusion:

    z[k+1] = z[k] + gamma * Delta_RC(chi[k])
              + eta * (z[k] - z[k-1]) + Sigma(chi[k]) xi[k].

``gamma`` is a shrinkage/expansion of the fitted one-sample increment and
``eta`` is a transparent one-sample persistence skip.  Both are selected only
from out-of-fold ictal futures and are then frozen for deployment.  The
wrapper subclasses the manuscript batch stepper, so the reservoir, graph
topology, delays, state-dependent diffusion, control map, and 256-Hz update
semantics remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
from torch import Tensor


HERE = Path(__file__).resolve().parent
PIPELINE_ROOT = HERE.parents[1] / "taming-epilepsy-mfc-baseline"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from mfc_pipeline.causal_ltv_particle_rollout import (
    FrozenIctalGraphRCBatchStepper,
)


@dataclass(frozen=True, order=True)
class MultiStepDriftCalibration:
    """Two-scalar closed-loop calibration selected on OOF ictal futures."""

    increment_scale: float = 1.0
    persistence_skip: float = 0.0

    def validate(self) -> None:
        if not 0.0 <= float(self.increment_scale) <= 1.5:
            raise ValueError("increment_scale must lie in [0, 1.5]")
        if not -0.5 <= float(self.persistence_skip) <= 0.75:
            raise ValueError("persistence_skip must lie in [-0.5, 0.75]")


class ExposureCalibratedIctalStepper(FrozenIctalGraphRCBatchStepper):
    """Frozen Graph--RC stepper with an OOF-selected two-scalar drift wrapper."""

    def __init__(self, *args, calibration: MultiStepDriftCalibration, **kwargs):
        calibration.validate()
        super().__init__(*args, **kwargs)
        self.multistep_calibration = calibration

    def conditional_mean_and_std(self, markov: Tensor) -> tuple[Tensor, Tensor]:
        base_mean, conditional_std = super().conditional_mean_and_std(markov)
        _, history, current, _ = self._feature(markov)
        if history.shape[1] < 2:
            raise RuntimeError("persistence skip requires at least two latent states")
        previous = history[:, -2]
        adjusted_mean = adjusted_conditional_mean(
            base_mean,
            current,
            previous,
            self.multistep_calibration,
        )
        return adjusted_mean, conditional_std

    @property
    def sampling_contract(self) -> dict[str, object]:
        """Machine-readable audit of what the wrapper does and does not change."""

        return {
            "increment_scale": float(
                self.multistep_calibration.increment_scale
            ),
            "persistence_skip": float(
                self.multistep_calibration.persistence_skip
            ),
            "sampling_rate_hz": 256.0,
            "update_interval_samples": 1,
            "state_dependent_diffusion_unchanged": True,
            "reservoir_unchanged": True,
            "graph_topology_unchanged": True,
            "delay_register_unchanged": True,
        }


def adjusted_conditional_mean(
    base_mean: Tensor,
    current: Tensor,
    previous: Tensor,
    calibration: MultiStepDriftCalibration,
) -> Tensor:
    """Apply the two-scalar calibration; factored out for direct unit testing."""

    calibration.validate()
    increment = base_mean - current
    return (
        current
        + float(calibration.increment_scale) * increment
        + float(calibration.persistence_skip) * (current - previous)
    )


def build_calibrated_stepper(
    world,
    adapter,
    *,
    diffusion_scale: float,
    increment_scale: float,
    persistence_skip: float,
) -> ExposureCalibratedIctalStepper:
    """Construct the calibrated stepper without mutating a fitted artifact."""

    return ExposureCalibratedIctalStepper(
        world,
        adapter,
        diffusion_scale=float(diffusion_scale),
        calibration=MultiStepDriftCalibration(
            increment_scale=float(increment_scale),
            persistence_skip=float(persistence_skip),
        ),
    )
