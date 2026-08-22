from __future__ import annotations

import copy
from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

from .contracts import sha256_file
from .control import (
    PROTOCOL_ID,
    _contexts_and_futures,
    _controller_config,
    _standardized_reference,
    activate_controller,
    unprojected_rollout,
)
from .data_model import (
    array_sha256, assert_live_part2_model_identity,
    load_development, load_network,
)
from .staging import atomic_json


FORMAL_STAGE_IDS = (
    "S0_INITIAL_0_100",
    "S1_BALANCED_100_250",
    "S2S_SAFE_250_450",
    "S3_COVARIANCE_450_650",
    "S4_MARKOV_ALPHA100_650_770",
    "S5_SMOOTHA_770_870",
    "S6_PRECISION_ALL_PARAMS_870_1050",
)
MODERN_TEMPLATE_SEED = 20260921


def _load_source(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path(path).resolve())
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_stage_authorities(config: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    """Load code-only HUP060 semantics; never load an old model or checkpoint."""

    locks = config["adaptation_locks"]
    legacy = _load_source(
        "hup080_hup060_stage_semantics_v1", Path(locks["hup060_stage_runner"][0])
    )
    historical = _load_source(
        "hup080_historical_actor_architecture_v1",
        Path(locks["historical_actor_source"][0]),
    )
    contract_path = Path(locks["hup060_stage_contract"][0])
    stage_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    formal = [item for item in stage_contract["stages"] if bool(item.get("formal_chain"))]
    ids = tuple(str(item["stage_id"]) for item in formal)
    if ids != FORMAL_STAGE_IDS or any("S2R" in item for item in ids):
        raise RuntimeError("formal teacher chain changed or discarded S2R became executable")
    return legacy, historical, stage_contract


def validate_training_schedule(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    controller = config["controller"]
    stages = tuple(dict(item) for item in controller["teacher_stages"])
    ids = tuple(str(item["stage_id"]) for item in stages)
    if ids != FORMAL_STAGE_IDS or tuple(controller["teacher_stage_ids"]) != FORMAL_STAGE_IDS:
        raise RuntimeError("configured S0-S6 order differs from the shared protocol")
    if int(controller["teacher_epochs_total"]) != 1050:
        raise RuntimeError("teacher total must be 1050 epochs")
    if sum(int(item["epochs"]) for item in stages) != 1050:
        raise RuntimeError("teacher stage epochs do not sum to 1050")
    cursor = 0
    for item in stages:
        if int(item["epoch_offset"]) != cursor:
            raise RuntimeError(f"non-contiguous epoch offset at {item['stage_id']}")
        cursor += int(item["epochs"])
        if str(item["objective"]) not in controller["objective_weights"]:
            raise RuntimeError(f"undefined objective {item['objective']}")
    if bool(controller["discarded_s2r_executed"]):
        raise RuntimeError("discarded S2R must never execute")
    return stages


def _atomic_torch_save(torch: Any, path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _antithetic(ctrl: Any, seed: int, cfg: Any, q: int) -> Any:
    return ctrl.antithetic_noise(int(seed), int(cfg.particles), int(cfg.horizon), int(q))


def _channel_weights(reference: Any) -> tuple[Any, Any]:
    pooled = reference.reshape(-1, reference.shape[-1]).var(dim=0, unbiased=False)
    inverse = 1.0 / pooled.clamp_min(0.1**2)
    inverse = inverse / inverse.mean()
    initial = inverse.clamp(0.25, 4.0)
    initial = initial / initial.mean()
    balanced = (0.5 + 0.5 * inverse).clamp(0.5, 3.0)
    balanced = balanced / balanced.mean()
    return initial, balanced


def _fresh_historical_actor(historical: Any, stack: Sequence[Any], cfg: Any) -> Any:
    import numpy as np
    import torch

    torch.manual_seed(MODERN_TEMPLATE_SEED)
    np.random.seed(MODERN_TEMPLATE_SEED)
    random.seed(MODERN_TEMPLATE_SEED)
    modern_template = stack[-1]
    actor = historical.HistoricalStructuredSplineCovarianceActor(
        stack[2], stack[4], stack[5], stack[6], modern_template.base_gain.detach().clone(),
        horizon=int(cfg.horizon), basis_count=int(cfg.basis_count),
        hidden_size=int(cfg.decoded_hidden_size),
        amplitude_limit=float(cfg.amplitude_limit), actuator_alpha=0.25,
        residual_scale=float(cfg.residual_scale),
    )
    if hasattr(actor, "common_markov_residual"):
        raise RuntimeError("S0-S3 actor unexpectedly contains Markov parameters")
    return actor


def _migrate_to_modern(actor: Any, modern_template: Any) -> tuple[Any, dict[str, Any]]:
    import torch

    source = actor.state_dict()
    template_state = modern_template.state_dict()
    markov_initial_keys = sorted(
        key for key in template_state
        if key.startswith("common_markov_residual.")
        or key.startswith("deviation_markov_residual.")
    )
    digest = sha256()
    for key in markov_initial_keys:
        value = template_state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    new_markov_initialization_sha256 = digest.hexdigest()
    incompatible = modern_template.load_state_dict(source, strict=False)
    allowed_exact = {
        "local_deviation_gain_logits", "actuated_channel_indices", "local_inverse_effect",
        "markov_feature_center", "markov_feature_scale",
    }
    unexpected_missing = [
        key for key in incompatible.missing_keys
        if key not in allowed_exact
        and not key.startswith("common_markov_residual.")
        and not key.startswith("deviation_markov_residual.")
    ]
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"S3-to-S4 migration mismatch: {unexpected_missing}/{incompatible.unexpected_keys}"
        )
    if len(modern_template.state_dict()) != 44:
        raise RuntimeError("S3-to-S4 destination actor is not the frozen 44-key model")
    if not torch.equal(
        modern_template.local_deviation_gain_logits,
        torch.zeros_like(modern_template.local_deviation_gain_logits),
    ):
        raise RuntimeError("new local-gain logits are not neutral at migration")
    for module in (
        modern_template.common_markov_residual,
        modern_template.deviation_markov_residual,
    ):
        if not torch.equal(module[-1].weight, torch.zeros_like(module[-1].weight)):
            raise RuntimeError("new Markov output weights are not neutral")
        if not torch.equal(module[-1].bias, torch.zeros_like(module[-1].bias)):
            raise RuntimeError("new Markov output bias is not neutral")
    receipt = {
        "source_state_key_count": len(source),
        "destination_state_key_count": len(modern_template.state_dict()),
        "missing_keys": sorted(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "fresh_neutral_local_and_markov_parameters": True,
        "modern_template_initialization_seed": MODERN_TEMPLATE_SEED,
        "new_markov_initialization_keys": markov_initial_keys,
        "new_markov_initialization_sha256": new_markov_initialization_sha256,
    }
    return modern_template, receipt


def _configure_trainable(actor: Any, stage: Mapping[str, Any]) -> list[str]:
    for parameter in actor.parameters():
        parameter.requires_grad_(True)
    scope = str(stage["train_scope"])
    if scope == "all_pre_markov_parameters":
        if hasattr(actor, "common_markov_residual"):
            for module in (actor.common_markov_residual, actor.deviation_markov_residual):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
    elif scope == "deviation_only_freeze_common":
        actor.mean_gain_delta.requires_grad_(False)
        for parameter in actor.common_residual.parameters():
            parameter.requires_grad_(False)
        if hasattr(actor, "common_markov_residual"):
            for module in (actor.common_markov_residual, actor.deviation_markov_residual):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
    elif scope == "markov_only":
        if not hasattr(actor, "common_markov_residual"):
            raise RuntimeError("markov_only stage received the pre-Markov actor")
        for parameter in actor.parameters():
            parameter.requires_grad_(False)
        for module in (actor.common_markov_residual, actor.deviation_markov_residual):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    elif scope != "all_actor_parameters":
        raise RuntimeError(f"unknown train scope: {scope}")
    names = [name for name, parameter in actor.named_parameters() if parameter.requires_grad]
    if not names:
        raise RuntimeError(f"stage {stage['stage_id']} has no trainable parameters")
    if scope == "markov_only" and any("markov_residual." not in name for name in names):
        raise RuntimeError("S4/S5 exposed a non-Markov parameter")
    if scope == "all_actor_parameters" and len(names) != len(list(actor.named_parameters())):
        raise RuntimeError("S6 failed to unfreeze all actor parameters")
    return names


def _objective(
    legacy: Any, rollout: Any, reference: Any, channel_weights: Any,
    reference_scale: Any, projections: Any, actor: Any, baseline: Any,
    weights: Mapping[str, Any], *, regularize: bool,
) -> tuple[Any, dict[str, Any]]:
    return legacy.objective(
        rollout.scaled[:, 1:], rollout.controls, rollout.commands, reference,
        channel_weights, reference_scale, projections, actor, baseline, weights,
        parameter_regularization=regularize,
        smooth=float(weights.get("smoothness", 0.2)),
        curvature=float(weights.get("curvature", 0.1)),
        slew_barrier=float(weights.get("slew_barrier", 0.0)),
        curvature_barrier=float(weights.get("curvature_barrier", 0.0)),
        max_first=float(weights.get("maximum_first_difference", 0.35)),
        max_second=float(weights.get("maximum_second_difference", 0.5)),
    )


def _train_stage(
    ctrl: Any, legacy: Any, actor: Any, stepper: Any,
    fit_initial: Sequence[Any], selection_initial: Sequence[Any],
    fit_reference: Any, validation_reference: Any,
    initial_weights: Any, balanced_weights: Any, reference_scale: Any,
    projections: Any, cfg: Any, stage: Mapping[str, Any], weights: Mapping[str, Any],
    stage_root: Path, parent_hash: str | None, migration: Mapping[str, Any] | None,
) -> tuple[Any, dict[str, Any]]:
    import pandas as pd
    import torch

    actor.actuator_alpha = float(stage["actuator_alpha"])
    trainable = _configure_trainable(actor, stage)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in actor.parameters() if parameter.requires_grad],
        lr=float(stage["learning_rate"]), weight_decay=1.0e-5,
    )
    channel_weights = (
        initial_weights if str(stage["stage_id"]) == FORMAL_STAGE_IDS[0]
        else balanced_weights
    )
    validation_bank = []
    for replicate, initial in enumerate(selection_initial):
        seed = 20260922
        noise = _antithetic(ctrl, seed, cfg, stepper.q)
        with torch.no_grad():
            baseline = ctrl.uncontrolled_particle_rollout(stepper, initial, noise)[:, 1:]
        validation_bank.append((initial, noise, baseline, seed))
    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    history: list[dict[str, Any]] = []
    epochs = int(stage["epochs"])
    offset = int(stage["epoch_offset"])
    for local_epoch in range(epochs):
        absolute = offset + local_epoch
        fit_index = absolute % len(fit_initial)
        initial = fit_initial[fit_index]
        seed = 20260921 + 1009 * (absolute + 1)
        noise = _antithetic(ctrl, seed, cfg, stepper.q)
        with torch.no_grad():
            baseline = ctrl.uncontrolled_particle_rollout(stepper, initial, noise)[:, 1:]
        actor.train()
        optimizer.zero_grad(set_to_none=True)
        rollout = unprojected_rollout(ctrl, stepper, actor, initial, noise)
        loss, terms = _objective(
            legacy, rollout, fit_reference, channel_weights, reference_scale,
            projections, actor, baseline, weights, regularize=True,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss in {stage['stage_id']}")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in actor.parameters() if parameter.requires_grad], 1.0
        )
        optimizer.step()
        row: dict[str, Any] = {
            "stage_id": str(stage["stage_id"]), "absolute_epoch": absolute + 1,
            "fit_context_slot": fit_index, "noise_seed": seed,
            "train_loss": float(loss.detach()),
            "gradient_norm": float(torch.as_tensor(gradient)),
        }
        should_validate = (
            local_epoch == 0
            or (absolute + 1) % int(stage["validation_every"]) == 0
            or local_epoch + 1 == epochs
        )
        if should_validate:
            actor.eval()
            losses = []
            with torch.no_grad():
                for initial_v, noise_v, baseline_v, _seed_v in validation_bank:
                    rollout_v = unprojected_rollout(
                        ctrl, stepper, actor, initial_v, noise_v
                    )
                    value, _ = _objective(
                        legacy, rollout_v, validation_reference, channel_weights, reference_scale,
                        projections, actor, baseline_v, weights, regularize=False,
                    )
                    losses.append(float(value))
            validation_loss = float(sum(losses) / len(losses))
            row["ctx5_macro_validation_loss"] = validation_loss
            row["ctx5_run_losses_json"] = json.dumps(losses)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = absolute + 1
                best_state = copy.deepcopy(actor.state_dict())
        history.append(row)
        if (local_epoch + 1) % max(1, int(stage["validation_every"])) == 0:
            print(f"{stage['stage_id']} epoch {absolute + 1}", flush=True)
    if best_state is None:
        raise RuntimeError(f"{stage['stage_id']} produced no eligible ctx5 checkpoint")
    actor.load_state_dict(best_state, strict=True)
    actor.eval()
    stage_root.mkdir(parents=False, exist_ok=False)
    checkpoint = stage_root / "frozen_actor.pt"
    _atomic_torch_save(
        torch, checkpoint,
        {
            "actor_state_dict": best_state,
            "stage": dict(stage),
            "best_epoch": best_epoch,
            "best_ctx5_macro_validation_loss": best_loss,
            "parent_checkpoint_sha256": parent_hash,
            "migration": migration,
            "old_checkpoint_used": False,
        },
    )
    pd.DataFrame(history).to_csv(
        stage_root / "training_history.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "stage_id": str(stage["stage_id"]), "epochs_executed": epochs,
        "best_epoch": best_epoch, "best_ctx5_macro_validation_loss": best_loss,
        "train_scope": str(stage["train_scope"]), "objective": str(stage["objective"]),
        "trainable_parameter_names": trainable,
        "checkpoint_sha256": sha256_file(checkpoint),
        "parent_checkpoint_sha256": parent_hash, "migration": migration,
        "ctx5_selected_epoch_only": True, "ctx5_selected_arm": False,
        "teacher_training_noise_seed_rule": "20260921 + 1009*(absolute_epoch+1)",
        "teacher_validation_noise_seed": 20260922,
    }
    atomic_json(stage_root / "summary.json", summary)
    return actor, summary


def _validate_wgan_exact(
    ctrl: Any, legacy: Any, actor: Any, stepper: Any,
    selection_initial: Sequence[Any], validation_reference: Any,
    channel_weights: Any, reference_scale: Any, projections: Any,
    cfg: Any, covariance_weights: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np
    import torch

    laws: list[float] = []
    times: list[float] = []
    occupations: list[float] = []
    contexts: list[dict[str, Any]] = []
    display: dict[str, Any] = {}
    actor.eval()
    with torch.no_grad():
        for replicate, initial in enumerate(selection_initial):
            seed = 20260922
            noise = _antithetic(ctrl, seed, cfg, stepper.q)
            free = ctrl.uncontrolled_particle_rollout(stepper, initial, noise)[:, 1:]
            rollout = unprojected_rollout(ctrl, stepper, actor, initial, noise)
            law, _terms = _objective(
                legacy, rollout, validation_reference, channel_weights,
                reference_scale, projections, actor, free,
                covariance_weights, regularize=False,
            )
            controlled_np = rollout.scaled[:, 1:].cpu().numpy()
            free_np = free.cpu().numpy()
            reference_np = validation_reference.cpu().numpy()
            time_w1 = float(ctrl.channelwise_time_w1(controlled_np, reference_np).mean())
            occupation_w1 = float(
                ctrl.channelwise_occupation_w1(controlled_np, reference_np).mean()
            )
            laws.append(float(law))
            times.append(time_w1)
            occupations.append(occupation_w1)
            contexts.append({
                "ctx5_slot": replicate, "noise_seed": seed,
                "law": float(law), "mean_time_w1": time_w1,
                "mean_occupation_w1": occupation_w1,
            })
            if replicate == 0:
                display = {
                    "free": free_np, "controlled": controlled_np,
                    "reference": reference_np,
                    "controls": rollout.controls.cpu().numpy(),
                    "noise": noise.cpu().numpy(),
                }
    return {
        "aggregate": {
            "law": float(np.mean(laws)),
            "mean_time_w1": float(np.mean(times)),
            "mean_occupation_w1": float(np.mean(occupations)),
            "ctx5_contexts": len(contexts),
        },
        "contexts": contexts,
        "display": display,
    }


def _train_wgan_exact(
    ctrl: Any, legacy: Any, actor: Any, stepper: Any,
    fit_initial: Sequence[Any], selection_initial: Sequence[Any],
    fit_reference: Any, validation_reference: Any,
    channel_weights: Any, reference_scale: Any, cfg: Any,
    covariance_weights: Mapping[str, Any],
) -> tuple[Any, Any, list[dict[str, Any]], dict[str, Any], bool, dict[str, Any]]:
    """D-formal WGAN40 seeds/optimizers/clips with fail-closed noninferiority."""

    import numpy as np
    import torch

    torch.manual_seed(20261011)
    np.random.seed(20261011)
    random.seed(20261011)
    projections = ctrl.fixed_joint_projections(stepper.n_channels, 20261011)
    teacher = copy.deepcopy(actor)
    teacher.eval()
    teacher_state = copy.deepcopy(actor.state_dict())
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    critic = ctrl.TimeConditionedWassersteinCritic(
        fit_reference.reshape(-1, stepper.n_channels).mean(dim=0),
        reference_scale, horizon_samples=int(cfg.horizon), hidden_size=128,
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=1.0e-4, betas=(0.0, 0.9)
    )
    actor_optimizer = torch.optim.AdamW(
        actor.parameters(), lr=1.0e-5, weight_decay=1.0e-5
    )
    critic_time_indices = tuple(range(15, int(cfg.horizon), 16))
    anchor_time_indices = tuple(range(0, int(cfg.horizon), 16))
    banks = []
    with torch.no_grad():
        for bank in range(3):
            seed = 20261011 + 101 * (bank + 1)
            initial = fit_initial[bank % len(fit_initial)]
            rollout = unprojected_rollout(
                ctrl, stepper, teacher, initial,
                _antithetic(ctrl, seed, cfg, stepper.q)
            )
            banks.append(rollout.scaled[:, 1:].detach())
    history: list[dict[str, Any]] = []
    for step in range(24):
        critic_optimizer.zero_grad(set_to_none=True)
        gp_seed = 20261011 + 50000 + step
        generator = torch.Generator(device="cpu")
        generator.manual_seed(gp_seed)
        audit = ctrl.balanced_wgan_gp_loss(
            critic, banks[step % 3], fit_reference, critic_time_indices,
            gradient_penalty_weight=10.0, critic_drift_weight=0.001,
            generator=generator,
        )
        audit.loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
        critic_optimizer.step()
        history.append({
            "stage": "critic_pretrain", "step": step + 1,
            "gp_seed": gp_seed, "loss": float(audit.loss.detach()),
        })
    baseline_result = _validate_wgan_exact(
        ctrl, legacy, actor, stepper, selection_initial, validation_reference,
        channel_weights, reference_scale, projections, cfg, covariance_weights,
    )
    baseline = baseline_result["aggregate"]
    history.append({
        "stage": "actor_wgan", "epoch": 0, "eligible": False,
        "validation_law": baseline["law"],
        "validation_mean_time_w1": baseline["mean_time_w1"],
        "validation_mean_occupation_w1": baseline["mean_occupation_w1"],
    })
    def better(candidate: Any, incumbent: Any) -> bool:
        if incumbent is None:
            return True
        if not np.isclose(candidate[0], incumbent[0]):
            return bool(candidate[0] < incumbent[0])
        return bool(candidate[1] < incumbent[1])

    best_noninferior = None
    fallback_for_audit = None
    for epoch in range(1, 41):
        initial = fit_initial[(epoch - 1) % len(fit_initial)]
        noise_seed = 20261011 + 1009 * epoch
        noise = _antithetic(ctrl, noise_seed, cfg, stepper.q)
        actor.eval()
        with torch.no_grad():
            detached_rollout = unprojected_rollout(
                ctrl, stepper, actor, initial, noise
            )
            detached = detached_rollout.scaled[:, 1:].detach()
        for parameter in actor.parameters():
            parameter.requires_grad_(False)
        for parameter in critic.parameters():
            parameter.requires_grad_(True)
        critic.train()
        critic_losses = []
        gp_seeds = []
        for critic_step in range(3):
            critic_optimizer.zero_grad(set_to_none=True)
            gp_seed = 20261011 + 100000 * epoch + critic_step
            gp_seeds.append(gp_seed)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(gp_seed)
            audit = ctrl.balanced_wgan_gp_loss(
                critic, detached, fit_reference, critic_time_indices,
                gradient_penalty_weight=10.0, critic_drift_weight=0.001,
                generator=generator,
            )
            audit.loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
            critic_optimizer.step()
            critic_losses.append(float(audit.loss.detach()))
        for parameter in critic.parameters():
            parameter.requires_grad_(False)
        for parameter in actor.parameters():
            parameter.requires_grad_(True)
        actor.train()
        actor_optimizer.zero_grad(set_to_none=True)
        rollout = unprojected_rollout(ctrl, stepper, actor, initial, noise)
        with torch.no_grad():
            free = ctrl.uncontrolled_particle_rollout(stepper, initial, noise)[:, 1:]
        law, _terms = _objective(
            legacy, rollout, fit_reference, channel_weights, reference_scale,
            projections, actor, free, covariance_weights, regularize=True,
        )
        adversarial, _estimate = ctrl.actor_wasserstein_loss(
            critic, rollout.scaled[:, 1:], fit_reference, critic_time_indices
        )
        anchor = ctrl.actor_action_anchor_loss(
            actor, teacher, rollout, anchor_time_indices,
            amplitude_limit=float(cfg.amplitude_limit),
        )
        total = law + 0.5 * adversarial + 0.2 * anchor
        if not torch.isfinite(total):
            raise RuntimeError("non-finite formal WGAN actor loss")
        total.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        actor_optimizer.step()
        row: dict[str, Any] = {
            "stage": "actor_wgan", "epoch": epoch, "eligible": True,
            "noise_seed": noise_seed, "critic_gp_seeds_json": json.dumps(gp_seeds),
            "train_total": float(total.detach()), "train_law": float(law.detach()),
            "train_adversarial": float(adversarial.detach()),
            "train_anchor": float(anchor.detach()),
            "mean_critic_loss": float(np.mean(critic_losses)),
        }
        if epoch == 1 or epoch % 4 == 0 or epoch == 40:
            result = _validate_wgan_exact(
                ctrl, legacy, actor, stepper, selection_initial, validation_reference,
                channel_weights, reference_scale, projections, cfg, covariance_weights,
            )
            aggregate = result["aggregate"]
            score = (
                0.50 * aggregate["law"] / max(baseline["law"], 1.0e-12)
                + 0.25 * aggregate["mean_time_w1"] / max(baseline["mean_time_w1"], 1.0e-12)
                + 0.25 * aggregate["mean_occupation_w1"] / max(baseline["mean_occupation_w1"], 1.0e-12)
            )
            noninferior = aggregate["law"] <= 1.02 * baseline["law"]
            row.update({
                "validation_law": aggregate["law"],
                "validation_mean_time_w1": aggregate["mean_time_w1"],
                "validation_mean_occupation_w1": aggregate["mean_occupation_w1"],
                "selection_score": score, "law_noninferior": bool(noninferior),
            })
            candidate = (
                float(score), float(aggregate["law"]), int(epoch),
                copy.deepcopy(actor.state_dict()), copy.deepcopy(critic.state_dict()), result,
            )
            if better(candidate, fallback_for_audit):
                fallback_for_audit = candidate
            if noninferior and better(candidate, best_noninferior):
                best_noninferior = candidate
        history.append(row)
    accepted = best_noninferior is not None
    if accepted:
        selected = best_noninferior
        actor.load_state_dict(selected[3], strict=True)
        critic.load_state_dict(selected[4], strict=True)
        final_result = selected[5]
        selected_epoch = int(selected[2])
        selected_score = float(selected[0])
    else:
        actor.load_state_dict(teacher_state, strict=True)
        final_result = baseline_result
        selected_epoch = None
        selected_score = None
    actor.eval()
    summary = {
        "epochs_executed": 40, "accepted_noninferior_checkpoint": bool(accepted),
        "selected_epoch": selected_epoch, "selected_score": selected_score,
        "epoch_zero_eligible": False, "law_noninferiority_multiplier": 1.02,
        "fallback_update_observed": fallback_for_audit is not None,
        "fallback_update_promoted": False,
        "teacher_retained_if_no_noninferior_update": True,
        "seed_contract": {
            "critic_pretrain_noise": "20261011 + 101*(bank+1)",
            "critic_pretrain_gp": "20261011 + 50000 + step",
            "joint_epoch_noise": "20261011 + 1009*epoch",
            "joint_epoch_gp": "20261011 + 100000*epoch + critic_step",
        },
        "actor_gradient_clip": 1.0, "critic_gradient_clip": 5.0,
    }
    return actor, critic, history, final_result, accepted, summary


def _all_development_reference_pools(
    config: Mapping[str, Any], model: Any, arrays: Mapping[str, Any],
    runs: Sequence[str],
) -> tuple[Any, Any, dict[str, Any]]:
    """Build per-run fit/ctx5 blocks, then concatenate in the frozen run order."""

    import numpy as np

    split = config["reference_path_split"]
    fit_indices = list(map(int, split["neural_fit_indices_half_open"]))
    validation_indices = list(map(
        int, split["ctx5_checkpoint_validation_indices_half_open"]
    ))
    if fit_indices != [0, 15] or validation_indices != [15, 30]:
        raise PermissionError("all-development reference split changed")
    if list(runs) != list(split["apply_per_run_then_concatenate_in_order"]):
        raise PermissionError("all-development reference concatenation order changed")
    fit_blocks = []
    validation_blocks = []
    per_run = []
    for run in runs:
        fit_block = _standardized_reference(model, arrays, [run], fit_indices)
        validation_block = _standardized_reference(
            model, arrays, [run], validation_indices
        )
        if fit_block.shape[0] != 15 or validation_block.shape[0] != 15:
            raise RuntimeError("each development run must contribute 15+15 paths")
        if array_sha256(fit_block) == array_sha256(validation_block):
            raise RuntimeError(f"{run} fit and ctx5 reference blocks are not isolated")
        fit_blocks.append(fit_block)
        validation_blocks.append(validation_block)
        per_run.append({
            "run": str(run),
            "fit_indices_half_open": fit_indices,
            "fit_shape": list(fit_block.shape),
            "fit_sha256": array_sha256(fit_block),
            "ctx5_validation_indices_half_open": validation_indices,
            "ctx5_validation_shape": list(validation_block.shape),
            "ctx5_validation_sha256": array_sha256(validation_block),
            "overlap_by_index": False,
        })
    fit = np.concatenate(fit_blocks, axis=0)
    validation = np.concatenate(validation_blocks, axis=0)
    if fit.shape[0] != 45 or validation.shape[0] != 45:
        raise RuntimeError("HUP080 all-development pools must contain 45+45 paths")
    receipt = {
        "schema_version": "hup080-all-dev-reference-pools-v1",
        "run_order": list(runs),
        "construction": "split_each_run_then_concatenate_in_frozen_run_order",
        "per_run": per_run,
        "fit_combined_shape": list(fit.shape),
        "fit_combined_sha256": array_sha256(fit),
        "ctx5_validation_combined_shape": list(validation.shape),
        "ctx5_validation_combined_sha256": array_sha256(validation),
        "all_blocks_disjoint_by_preregistered_path_indices": True,
        "unused_tail_indices_half_open": list(map(
            int, split["unused_development_tail_indices_half_open"]
        )),
    }
    return fit, validation, receipt


def _formal_training_inputs(
    config: Mapping[str, Any], development_npz: Path,
    analytical_dir: Path, refit_dir: Path,
) -> dict[str, Any]:
    """Deterministically rebuild the same fit/ctx5 stack for every atomic stage."""

    import joblib
    import numpy as np
    import torch

    stages = validate_training_schedule(config)
    ctrl = activate_controller(config)
    legacy, historical, _stage_contract = load_stage_authorities(config)
    arrays = load_development(development_npz)
    runs = list(config["source_data"]["development_runs"])
    model_path = refit_dir / "selected_model.joblib"
    network_path = refit_dir / "part1_selection_network.npz"
    plant_network_path = refit_dir / "part2_plant_network.npz"
    model = joblib.load(model_path)
    assert_live_part2_model_identity(model, config)
    network = load_network(network_path)
    plant_network = load_network(plant_network_path)
    if array_sha256(model.adjacency_input) != array_sha256(
        plant_network["adjacency"]
    ):
        raise PermissionError("training model/Part-II plant graph mismatch")
    top1 = json.loads(
        (analytical_dir / "frozen_top1.json").read_text(encoding="utf-8")
    )["selection"]
    cfg = _controller_config(
        ctrl, config, fraction=float(top1["fraction"]), gain=float(top1["gain"]),
        tau=float(top1["tau"]), train=True,
    )
    fit_contexts, _fit_futures, fit_ledger = _contexts_and_futures(
        arrays, runs, config["contexts"]["analytical_and_controller_fit"]
    )
    ctx5_contexts, _ctx5_futures, ctx5_ledger = _contexts_and_futures(
        arrays, runs, config["contexts"]["checkpoint_epoch_selection_only"]
    )
    if any(int(row["context_index"]) != 5 for row in ctx5_ledger):
        raise RuntimeError("non-ctx5 data entered checkpoint selection")
    fit_reference_np, validation_reference_np, reference_pool_receipt = (
        _all_development_reference_pools(config, model, arrays, runs)
    )
    if array_sha256(fit_reference_np) == array_sha256(validation_reference_np):
        raise RuntimeError("neural fit and ctx5 validation reference pools are not isolated")
    modern_template_seed = int(
        config["controller"]["s3_to_s4_modern_template_initialization_seed"]
    )
    if modern_template_seed != MODERN_TEMPLATE_SEED:
        raise PermissionError("S3-to-S4 modern-template seed changed")
    torch.manual_seed(modern_template_seed)
    np.random.seed(modern_template_seed)
    random.seed(modern_template_seed)
    stack = ctrl.build_control_stack(model, network, fit_contexts, fit_reference_np, cfg)
    fit_reference = torch.as_tensor(fit_reference_np, dtype=torch.float64)
    validation_reference = torch.as_tensor(
        validation_reference_np, dtype=torch.float64
    )
    selection_initial = [
        stack[2].adapter.initial_state_from_context(item) for item in ctx5_contexts
    ]
    initial_weights, balanced_weights = _channel_weights(fit_reference)
    return {
        "stages": stages, "ctrl": ctrl, "legacy": legacy,
        "historical": historical, "arrays": arrays, "model_path": model_path,
        "network_path": network_path, "plant_network_path": plant_network_path,
        "network": network, "top1": top1,
        "cfg": cfg, "stack": stack, "fit_initial": stack[3],
        "selection_initial": selection_initial,
        "fit_reference_np": fit_reference_np,
        "validation_reference_np": validation_reference_np,
        "fit_reference": fit_reference,
        "validation_reference": validation_reference,
        "initial_weights": initial_weights, "balanced_weights": balanced_weights,
        "projections": ctrl.fixed_joint_projections(stack[2].n_channels, 20260921),
        "fit_ledger": fit_ledger, "ctx5_ledger": ctx5_ledger,
        "reference_pool_receipt": reference_pool_receipt,
        "modern_template_initialization_seed": modern_template_seed,
    }


def _validated_parent_state(
    torch: Any, parent_dir: Path, expected_stage_id: str,
    expected_reference_pool_receipt: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str]:
    checkpoint_path = parent_dir / expected_stage_id / "frozen_actor.pt"
    summary_path = parent_dir / expected_stage_id / "summary.json"
    if not checkpoint_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"atomic parent checkpoint missing for {expected_stage_id}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    observed_sha256 = sha256_file(checkpoint_path)
    if summary.get("stage_id") != expected_stage_id:
        raise PermissionError("parent stage identity changed")
    if summary.get("checkpoint_sha256") != observed_sha256:
        raise PermissionError("parent checkpoint differs from its stage receipt")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("stage", {}).get("stage_id") != expected_stage_id:
        raise PermissionError("parent checkpoint payload stage identity changed")
    phase_receipt_path = parent_dir / "teacher_stage_receipt.json"
    phase_receipt = json.loads(phase_receipt_path.read_text(encoding="utf-8"))
    if phase_receipt.get("reference_pool_receipt") != dict(
        expected_reference_pool_receipt
    ):
        raise PermissionError("parent checkpoint reference-pool receipt changed")
    return payload["actor_state_dict"], observed_sha256


def run_atomic_teacher_stage(
    config: Mapping[str, Any], development_npz: Path, analytical_dir: Path,
    refit_dir: Path, output: Path, *, stage_index: int,
    parent_stage_dir: Path | None,
) -> None:
    """Run one formal teacher stage from a strict hash-bound parent checkpoint."""

    import torch

    if not 0 <= int(stage_index) < len(FORMAL_STAGE_IDS):
        raise ValueError("teacher stage index is outside S0..S6")
    bundle = _formal_training_inputs(
        config, development_npz, analytical_dir, refit_dir
    )
    stage = bundle["stages"][int(stage_index)]
    if str(stage["stage_id"]) != FORMAL_STAGE_IDS[int(stage_index)]:
        raise RuntimeError("atomic teacher phase/stage mapping changed")
    parent_sha256 = None
    migration = None
    if int(stage_index) == 0:
        if parent_stage_dir is not None:
            raise PermissionError("S0 must start fresh and cannot receive a parent")
        actor = _fresh_historical_actor(
            bundle["historical"], bundle["stack"], bundle["cfg"]
        )
    else:
        if parent_stage_dir is None:
            raise PermissionError("S1..S6 require the preceding atomic checkpoint")
        parent_state, parent_sha256 = _validated_parent_state(
            torch, parent_stage_dir, FORMAL_STAGE_IDS[int(stage_index) - 1],
            bundle["reference_pool_receipt"],
        )
        if int(stage_index) <= 3:
            actor = _fresh_historical_actor(
                bundle["historical"], bundle["stack"], bundle["cfg"]
            )
            actor.load_state_dict(parent_state, strict=True)
        elif int(stage_index) == 4:
            historical_actor = _fresh_historical_actor(
                bundle["historical"], bundle["stack"], bundle["cfg"]
            )
            historical_actor.load_state_dict(parent_state, strict=True)
            actor, migration = _migrate_to_modern(
                historical_actor, bundle["stack"][-1]
            )
            if int(migration["destination_state_key_count"]) != 44:
                raise RuntimeError("S3-to-S4 migration did not produce 44 keys")
        else:
            actor = bundle["stack"][-1]
            if len(actor.state_dict()) != 44 or len(parent_state) != 44:
                raise RuntimeError("S5/S6 parent and destination must both have 44 keys")
            actor.load_state_dict(parent_state, strict=True)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    actor, summary = _train_stage(
        bundle["ctrl"], bundle["legacy"], actor, bundle["stack"][2],
        bundle["fit_initial"], bundle["selection_initial"],
        bundle["fit_reference"], bundle["validation_reference"],
        bundle["initial_weights"], bundle["balanced_weights"],
        bundle["stack"][6], bundle["projections"], bundle["cfg"], stage,
        config["controller"]["objective_weights"][stage["objective"]],
        output / str(stage["stage_id"]), parent_sha256, migration,
    )
    del actor
    atomic_json(
        output / "teacher_stage_receipt.json",
        {
            "schema_version": "hup080-atomic-teacher-stage-v1",
            "stage_index": int(stage_index), "stage_id": str(stage["stage_id"]),
            "epochs_executed": int(stage["epochs"]),
            "epoch_offset": int(stage["epoch_offset"]),
            "parent_stage_checkpoint_sha256": parent_sha256,
            "output_checkpoint_sha256": summary["checkpoint_sha256"],
            "strict_parent_load": int(stage_index) > 0,
            "fresh_initialization": int(stage_index) == 0,
            "s3_to_s4_migration": migration,
            "modern_template_initialization_seed": bundle[
                "modern_template_initialization_seed"
            ],
            "destination_state_key_count": 44 if int(stage_index) >= 4 else None,
            "controller_d_canonical_module_bindings": dict(
                bundle["ctrl"].HUP080_CANONICAL_MODULE_BINDINGS
            ),
            "control_stack_contract": dict(
                bundle["ctrl"].HUP080_CONTROL_STACK_CONTRACT
            ),
            "part1_selection_network_sha256": sha256_file(
                bundle["network_path"]
            ),
            "part2_plant_network_sha256": sha256_file(
                bundle["plant_network_path"]
            ),
            "fit_reference_sha256": array_sha256(bundle["fit_reference_np"]),
            "ctx5_validation_reference_sha256": array_sha256(
                bundle["validation_reference_np"]
            ),
            "reference_pool_receipt": bundle["reference_pool_receipt"],
            "reference_pool_overlap": False,
            "ctx5_selected_epoch_only": True, "ctx5_selected_arm": False,
            "discarded_s2r_executed": False, "sealed_run_opened": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


def run_atomic_wgan40_ctx5(
    config: Mapping[str, Any], development_npz: Path, analytical_dir: Path,
    refit_dir: Path, s6_stage_dir: Path, all_stage_dirs: Sequence[Path],
    output: Path,
) -> None:
    """Strict-load S6, run exact WGAN40, and freeze the ctx5-selected epoch."""

    import numpy as np
    import pandas as pd
    import torch

    if len(all_stage_dirs) != len(FORMAL_STAGE_IDS):
        raise ValueError("WGAN requires all seven atomically published teacher stages")
    stage_receipts = []
    for index, directory in enumerate(all_stage_dirs):
        receipt_path = directory / "teacher_stage_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("stage_id") != FORMAL_STAGE_IDS[index]:
            raise PermissionError("teacher phase order differs from S0..S6")
        if int(receipt.get("stage_index", -1)) != index:
            raise PermissionError("teacher stage index differs from its phase")
        stage_receipts.append({
            "stage_id": FORMAL_STAGE_IDS[index],
            "phase_receipt_sha256": sha256_file(receipt_path),
            "checkpoint_sha256": receipt["output_checkpoint_sha256"],
            "epochs_executed": int(receipt["epochs_executed"]),
        })
    if sum(item["epochs_executed"] for item in stage_receipts) != 1050:
        raise RuntimeError("atomically published teacher stages do not total 1050 epochs")
    bundle = _formal_training_inputs(
        config, development_npz, analytical_dir, refit_dir
    )
    parent_state, s6_sha256 = _validated_parent_state(
        torch, s6_stage_dir, FORMAL_STAGE_IDS[-1],
        bundle["reference_pool_receipt"],
    )
    actor = bundle["stack"][-1]
    if len(actor.state_dict()) != 44 or len(parent_state) != 44:
        raise RuntimeError("WGAN S6 parent and fresh destination must both have 44 keys")
    actor.load_state_dict(parent_state, strict=True)
    actor, critic, history, result, accepted, wgan_summary = _train_wgan_exact(
        bundle["ctrl"], bundle["legacy"], actor, bundle["stack"][2],
        bundle["fit_initial"], bundle["selection_initial"],
        bundle["fit_reference"], bundle["validation_reference"],
        bundle["balanced_weights"], bundle["stack"][6], bundle["cfg"],
        config["controller"]["objective_weights"]["covariance"],
    )
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(
        output / "wgan40_training_history.csv", index=False, encoding="utf-8-sig"
    )
    atomic_json(output / "wgan40_training_summary.json", wgan_summary)
    checkpoint_payload = {
        "schema_version": "hup080-fresh-s0-s6-1050-wgan40-v2",
        "subject": "HUP080", "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict() if accepted else None,
        "controller_config": asdict(bundle["cfg"]),
        "selected_stage": (
            "FORMAL_WGAN_FULL_0_40" if accepted
            else "S6_PRECISION_ALL_PARAMS_870_1050"
        ),
        "analytical_top1": bundle["top1"],
        "target_mask_sha256": array_sha256(bundle["network"]["target_mask"]),
        "model_sha256": sha256_file(bundle["model_path"]),
        "part1_selection_network_sha256": sha256_file(bundle["network_path"]),
        "part2_plant_network_sha256": sha256_file(bundle["plant_network_path"]),
        "teacher_epochs_executed": 1050, "wgan_epochs_executed": 40,
        "formal_stage_ids": list(FORMAL_STAGE_IDS),
        "atomic_teacher_stage_receipts": stage_receipts,
        "s6_parent_checkpoint_sha256": s6_sha256,
        "strict_s6_parent_load": True, "actor_state_key_count": 44,
        "controller_d_canonical_module_bindings": dict(
            bundle["ctrl"].HUP080_CANONICAL_MODULE_BINDINGS
        ),
        "control_stack_contract": dict(bundle["ctrl"].HUP080_CONTROL_STACK_CONTRACT),
        "modern_template_initialization_seed": bundle[
            "modern_template_initialization_seed"
        ],
        "discarded_s2r_executed": False,
        "wgan_accepted": bool(accepted), "epoch_zero_eligible": False,
        "ctx5_selected_epoch_only": True, "ctx5_selected_arm": False,
        "ctx6_used": False, "ctx7_used": False, "sealed_run_opened": False,
        "old_checkpoint_used": False, "old_fitted_model_used": False,
        "fresh_initialization_seed": int(bundle["cfg"].seed),
        "fit_reference_sha256": array_sha256(bundle["fit_reference_np"]),
        "ctx5_validation_reference_sha256": array_sha256(
            bundle["validation_reference_np"]
        ),
        "reference_pool_receipt": bundle["reference_pool_receipt"],
        "reference_pools_are_disjoint_by_preregistered_indices": True,
        "wgan_exact_summary": wgan_summary,
    }
    checkpoint = output / "selected_controller.pt"
    _atomic_torch_save(torch, checkpoint, checkpoint_payload)
    display = result["display"]
    np.savez_compressed(
        output / "ctx5_selected_rollout.npz",
        free_standardized=display["free"],
        controlled_standardized=display["controlled"],
        reference_standardized=display["reference"], controls=display["controls"],
        standard_normal=display["noise"], channels=bundle["arrays"]["channels"],
        direct_mask=np.asarray(bundle["network"]["target_mask"], dtype=bool),
    )
    atomic_json(
        output / "training_receipt.json",
        {
            "schema_version": "hup080-atomic-wgan40-ctx5-receipt-v1",
            "teacher_epochs_executed": 1050, "wgan_epochs_executed": 40,
            "atomic_teacher_stage_receipts": stage_receipts,
            "s6_parent_checkpoint_sha256": s6_sha256,
            "strict_s6_parent_load": True, "actor_state_key_count": 44,
            "controller_d_canonical_module_bindings": dict(
                bundle["ctrl"].HUP080_CANONICAL_MODULE_BINDINGS
            ),
            "control_stack_contract": dict(
                bundle["ctrl"].HUP080_CONTROL_STACK_CONTRACT
            ),
            "part1_selection_network_sha256": sha256_file(
                bundle["network_path"]
            ),
            "part2_plant_network_sha256": sha256_file(
                bundle["plant_network_path"]
            ),
            "modern_template_initialization_seed": bundle[
                "modern_template_initialization_seed"
            ],
            "discarded_s2r_executed": False, "wgan_accepted": bool(accepted),
            "ctx5_metrics": result["aggregate"],
            "ctx5_runs": bundle["ctx5_ledger"],
            "fit_context_count": len(bundle["fit_ledger"]),
            "reference_path_split": config["reference_path_split"],
            "fit_reference_sha256": array_sha256(bundle["fit_reference_np"]),
            "ctx5_validation_reference_sha256": array_sha256(
                bundle["validation_reference_np"]
            ),
            "reference_pool_receipt": bundle["reference_pool_receipt"],
            "reference_pool_overlap": False, "wgan_exact_summary": wgan_summary,
            "checkpoint_sha256": sha256_file(checkpoint),
            "arm_reselection_after_analytical": False,
            "sealed_run_opened": False,
        },
    )


def fresh_s0_s6_wgan40_ctx5(
    config: Mapping[str, Any], development_npz: Path, analytical_dir: Path,
    refit_dir: Path, output: Path,
) -> None:
    """Execute fresh formal S0-S6 (1050) then WGAN40; ctx5 selects epoch only."""

    import joblib
    import numpy as np
    import pandas as pd
    import torch

    stages = validate_training_schedule(config)
    ctrl = activate_controller(config)
    legacy, historical, _historical_contract = load_stage_authorities(config)
    arrays = load_development(development_npz)
    runs = list(config["source_data"]["development_runs"])
    model_path = refit_dir / "selected_model.joblib"
    network_path = refit_dir / "part1_selection_network.npz"
    model = joblib.load(model_path)
    assert_live_part2_model_identity(model, config)
    network = load_network(network_path)
    top1 = json.loads((analytical_dir / "frozen_top1.json").read_text(encoding="utf-8"))["selection"]
    cfg = _controller_config(
        ctrl, config, fraction=float(top1["fraction"]), gain=float(top1["gain"]),
        tau=float(top1["tau"]), train=True,
    )
    fit_contexts, _fit_futures, fit_ledger = _contexts_and_futures(
        arrays, runs, config["contexts"]["analytical_and_controller_fit"]
    )
    ctx5_contexts, _ctx5_futures, ctx5_ledger = _contexts_and_futures(
        arrays, runs, config["contexts"]["checkpoint_epoch_selection_only"]
    )
    if any(int(row["context_index"]) != 5 for row in ctx5_ledger):
        raise RuntimeError("non-ctx5 data entered checkpoint selection")
    fit_reference_np = _standardized_reference(
        model, arrays, runs,
        config["reference_path_split"]["neural_fit_indices_half_open"],
    )
    validation_reference_np = _standardized_reference(
        model, arrays, runs,
        config["reference_path_split"]["ctx5_checkpoint_validation_indices_half_open"],
    )
    fit_reference = torch.as_tensor(fit_reference_np, dtype=torch.float64)
    validation_reference = torch.as_tensor(validation_reference_np, dtype=torch.float64)
    if array_sha256(fit_reference_np) == array_sha256(validation_reference_np):
        raise RuntimeError("neural fit and ctx5 validation reference pools are not isolated")
    torch.manual_seed(MODERN_TEMPLATE_SEED)
    np.random.seed(MODERN_TEMPLATE_SEED)
    random.seed(MODERN_TEMPLATE_SEED)
    stack = ctrl.build_control_stack(model, network, fit_contexts, fit_reference_np, cfg)
    stepper = stack[2]
    fit_initial = stack[3]
    selection_initial = [
        stepper.adapter.initial_state_from_context(item) for item in ctx5_contexts
    ]
    initial_weights, balanced_weights = _channel_weights(fit_reference)
    projections = ctrl.fixed_joint_projections(stepper.n_channels, 20260921)
    actor = _fresh_historical_actor(historical, stack, cfg)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    parent_hash = None
    stage_summaries: list[dict[str, Any]] = []
    migration = None
    for index, stage in enumerate(stages):
        if index == 4:
            actor, migration = _migrate_to_modern(actor, stack[-1])
        stage_dir = output / str(stage["stage_id"])
        actor, summary = _train_stage(
            ctrl, legacy, actor, stepper, fit_initial, selection_initial,
            fit_reference, validation_reference,
            initial_weights, balanced_weights, stack[6], projections,
            cfg, stage, config["controller"]["objective_weights"][stage["objective"]],
            stage_dir, parent_hash, migration if index == 4 else None,
        )
        parent_hash = str(summary["checkpoint_sha256"])
        stage_summaries.append(summary)
    if sum(int(item["epochs_executed"]) for item in stage_summaries) != 1050:
        raise RuntimeError("formal teacher chain did not execute exactly 1050 epochs")
    actor, critic, wgan_history, wgan_result, wgan_accepted, wgan_summary = _train_wgan_exact(
        ctrl, legacy, actor, stepper, fit_initial, selection_initial,
        fit_reference, validation_reference, balanced_weights, stack[6], cfg,
        config["controller"]["objective_weights"]["covariance"],
    )
    wgan_dir = output / "FORMAL_WGAN_FULL_0_40"
    wgan_dir.mkdir(parents=False, exist_ok=False)
    pd.DataFrame(wgan_history).to_csv(
        wgan_dir / "training_history.csv", index=False, encoding="utf-8-sig"
    )
    atomic_json(wgan_dir / "training_summary.json", wgan_summary)
    checkpoint_payload = {
        "schema_version": "hup080-fresh-s0-s6-1050-wgan40-v1",
        "subject": "HUP080", "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict() if wgan_accepted else None,
        "controller_config": asdict(cfg),
        "selected_stage": (
            "FORMAL_WGAN_FULL_0_40" if wgan_accepted
            else "S6_PRECISION_ALL_PARAMS_870_1050"
        ),
        "analytical_top1": top1,
        "target_mask_sha256": array_sha256(network["target_mask"]),
        "model_sha256": sha256_file(model_path),
        "part1_selection_network_sha256": sha256_file(network_path),
        "teacher_epochs_executed": 1050, "wgan_epochs_executed": 40,
        "formal_stage_ids": list(FORMAL_STAGE_IDS),
        "discarded_s2r_executed": False,
        "wgan_accepted": bool(wgan_accepted), "epoch_zero_eligible": False,
        "ctx5_selected_epoch_only": True, "ctx5_selected_arm": False,
        "ctx6_used": False, "ctx7_used": False, "sealed_run_opened": False,
        "old_checkpoint_used": False, "old_fitted_model_used": False,
        "fresh_initialization_seed": int(cfg.seed),
        "fit_reference_sha256": array_sha256(fit_reference_np),
        "ctx5_validation_reference_sha256": array_sha256(validation_reference_np),
        "reference_pools_are_disjoint_by_preregistered_indices": True,
        "wgan_exact_summary": wgan_summary,
    }
    checkpoint = output / "selected_controller.pt"
    _atomic_torch_save(torch, checkpoint, checkpoint_payload)
    display = wgan_result["display"]
    np.savez_compressed(
        output / "ctx5_selected_rollout.npz",
        free_standardized=display["free"], controlled_standardized=display["controlled"],
        reference_standardized=display["reference"], controls=display["controls"],
        standard_normal=display["noise"], channels=arrays["channels"],
        direct_mask=np.asarray(network["target_mask"], dtype=bool),
    )
    atomic_json(
        output / "training_receipt.json",
        {
            "schema_version": "hup080-fresh-training-receipt-v2",
            "teacher_epochs_executed": 1050, "wgan_epochs_executed": 40,
            "stage_chain": stage_summaries,
            "discarded_s2r_executed": False,
            "wgan_accepted": bool(wgan_accepted),
            "ctx5_metrics": wgan_result["aggregate"],
            "ctx5_runs": ctx5_ledger, "fit_context_count": len(fit_ledger),
            "reference_path_split": config["reference_path_split"],
            "fit_reference_sha256": array_sha256(fit_reference_np),
            "ctx5_validation_reference_sha256": array_sha256(validation_reference_np),
            "reference_pool_overlap": False,
            "wgan_exact_summary": wgan_summary,
            "checkpoint_sha256": sha256_file(checkpoint),
            "elapsed_seconds": time.perf_counter() - started,
            "arm_reselection_after_analytical": False,
            "sealed_run_opened": False,
        },
    )


__all__ = [
    "FORMAL_STAGE_IDS", "fresh_s0_s6_wgan40_ctx5",
    "load_stage_authorities", "run_atomic_teacher_stage",
    "run_atomic_wgan40_ctx5", "validate_training_schedule",
]
