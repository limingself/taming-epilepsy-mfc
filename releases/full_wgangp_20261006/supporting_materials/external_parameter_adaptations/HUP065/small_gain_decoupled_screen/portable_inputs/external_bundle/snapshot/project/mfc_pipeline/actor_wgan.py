"""WGAN-GP coupling for the established empirical-FP control actor.

This module deliberately does not introduce an HJB value network.  The
``StructuredSplineCovarianceActor`` remains the policy network, the frozen
Graph--RC--SDE particle push-forward remains the empirical Fokker--Planck
solver, and a separate time-conditioned Kantorovich critic supplies an
additional distribution-matching gradient.  It is therefore an
Actor--WGAN candidate, not a neural HJB solver.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn

from .full_markov_hjb_fp_wgan import (
    TimeConditionedWassersteinCritic,
    WassersteinGPLoss,
    time_conditioned_wgan_gp_loss,
)


def balanced_antithetic_order(particles: int) -> Tensor:
    """Return an order that alternates the two halves of an antithetic bank.

    The shared WGAN-GP helper matches the smaller number of real/fake rows.
    HUP060 has 15 reference paths and 32 particles.  A raw ``fake[:15]``
    would otherwise use only the positive half of the antithetic bank.  The
    alternating order makes that deterministic truncation sign-balanced.
    """

    count = int(particles)
    if count < 2 or count % 2:
        raise ValueError("an even antithetic particle count is required")
    half = count // 2
    return torch.stack(
        (torch.arange(half), torch.arange(half, count)), dim=1
    ).reshape(-1)


def balanced_wgan_gp_loss(
    critic: TimeConditionedWassersteinCritic,
    fake_sequence: Tensor,
    reference_sequence: Tensor,
    time_indices: Iterable[int],
    *,
    gradient_penalty_weight: float = 10.0,
    critic_drift_weight: float = 1.0e-3,
    generator: torch.Generator | None = None,
) -> WassersteinGPLoss:
    """Evaluate WGAN-GP after sign-balancing an antithetic fake bank."""

    if fake_sequence.ndim != 3:
        raise ValueError("fake_sequence must have shape [particle,time,channel]")
    order = balanced_antithetic_order(int(fake_sequence.shape[0])).to(
        device=fake_sequence.device
    )
    return time_conditioned_wgan_gp_loss(
        critic,
        fake_sequence.index_select(0, order),
        reference_sequence,
        time_indices,
        gradient_penalty_weight=gradient_penalty_weight,
        critic_drift_weight=critic_drift_weight,
        generator=generator,
    )


def actor_wasserstein_loss(
    critic: TimeConditionedWassersteinCritic,
    controlled_sequence: Tensor,
    reference_sequence: Tensor,
    time_indices: Iterable[int],
) -> tuple[Tensor, Tensor]:
    """Return the Actor loss and the critic's signed dual estimate.

    With healthy samples treated as real, the critic maximizes

    ``E[D(reference)] - E[D(controlled)]``.

    Minimizing the same signed estimate with the critic frozen increases the
    controlled score and moves the generated empirical law toward the
    reference law.  The reference term is retained for an interpretable
    logged estimate even though it is constant with respect to the Actor.
    """

    if controlled_sequence.ndim != 3 or reference_sequence.ndim != 3:
        raise ValueError("sequences must have shape [sample,time,channel]")
    if controlled_sequence.shape[1:] != reference_sequence.shape[1:]:
        raise ValueError("controlled and reference dimensions differ")
    estimates: list[Tensor] = []
    for raw_step in time_indices:
        step = int(raw_step)
        if not 0 <= step < int(controlled_sequence.shape[1]):
            raise IndexError("WGAN time index lies outside the sequence")
        real_score = critic(reference_sequence[:, step], step).mean()
        fake_score = critic(controlled_sequence[:, step], step).mean()
        estimates.append(real_score - fake_score)
    if not estimates:
        raise ValueError("at least one WGAN time index is required")
    estimate = torch.stack(estimates).mean()
    return estimate, estimate


def actor_action_anchor_loss(
    actor: nn.Module,
    original_actor: nn.Module,
    rollout,
    time_indices: Iterable[int],
    *,
    amplitude_limit: float,
) -> Tensor:
    """Penalize deviation from the locked Actor on candidate-visited states."""

    if float(amplitude_limit) <= 0.0:
        raise ValueError("amplitude_limit must be positive")
    terms: list[Tensor] = []
    for raw_step in time_indices:
        step = int(raw_step)
        if not 0 <= step < int(rollout.controls.shape[1]):
            raise IndexError("anchor time index lies outside the rollout")
        previous = (
            torch.zeros_like(rollout.controls[:, 0])
            if step == 0
            else rollout.controls[:, step - 1]
        )
        with torch.no_grad():
            target = original_actor.propose(
                step=step,
                markov_particles=rollout.markov[:, step].detach(),
                previous_control=previous.detach(),
                current_scaled=rollout.scaled[:, step].detach(),
            ).applied_control
        proposed = actor.propose(
            step=step,
            markov_particles=rollout.markov[:, step],
            previous_control=previous,
            current_scaled=rollout.scaled[:, step],
        ).applied_control
        terms.append(
            ((proposed - target) / float(amplitude_limit)).square().mean()
        )
    if not terms:
        raise ValueError("at least one anchor time index is required")
    return torch.stack(terms).mean()


__all__ = [
    "actor_action_anchor_loss",
    "actor_wasserstein_loss",
    "balanced_antithetic_order",
    "balanced_wgan_gp_loss",
]
