from __future__ import annotations

import copy
from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
import math
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
    activate_controller,
    standardized_reference_pools,
    unprojected_rollout,
)
from .data_model import (
    array_sha256,
    assert_live_part2_model_identity,
    load_development,
    load_network,
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
HISTORICAL_ACTOR_STATE_KEYS = 23
MODERN_ACTOR_STATE_KEYS = 44
MIGRATION_INITIALIZATION_SEED = 20260921
TEACHER_TRAINING_BASE_SEED = 20260921
TEACHER_VALIDATION_SEED = 20260922


def _require_actor_state_keys(state: Mapping[str, Any], expected: int, role: str) -> None:
    observed = len(state)
    if observed != int(expected):
        raise RuntimeError(
            f"{role} Actor state key count changed: expected={expected}, observed={observed}"
        )


def _state_subset_sha256(state: Mapping[str, Any], keys: Sequence[str]) -> str:
    digest = sha256()
    for key in sorted(str(value) for value in keys):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


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
    for key in ("hup060_stage_runner", "historical_actor_source", "hup060_stage_contract"):
        path, expected = Path(locks[key][0]).resolve(), str(locks[key][1]).lower()
        if sha256_file(path) != expected:
            raise PermissionError(f"code-only stage authority changed: {key}")
    legacy = _load_source(
        "hup065_hup060_stage_semantics_v1", Path(locks["hup060_stage_runner"][0])
    )
    historical = _load_source(
        "hup065_historical_actor_architecture_v1",
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
    if int(controller.get("s3_to_s4_modern_actor_initialization_seed", -1)) != (
        MIGRATION_INITIALIZATION_SEED
    ):
        raise RuntimeError("S3-to-S4 modern Actor seed must remain 20260921")
    if int(controller.get("teacher_rollout_base_seed", -1)) != TEACHER_TRAINING_BASE_SEED:
        raise RuntimeError("teacher rollout base seed changed")
    if controller.get("teacher_rollout_seed_rule") != (
        "20260921+1009*(absolute_epoch+1)"
    ):
        raise RuntimeError("teacher rollout seed rule changed")
    if int(controller.get("teacher_validation_noise_seed", -1)) != TEACHER_VALIDATION_SEED:
        raise RuntimeError("teacher validation noise seed changed")
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

    torch.manual_seed(MIGRATION_INITIALIZATION_SEED)
    np.random.seed(MIGRATION_INITIALIZATION_SEED)
    random.seed(MIGRATION_INITIALIZATION_SEED)
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
    _require_actor_state_keys(
        source, HISTORICAL_ACTOR_STATE_KEYS, "S3 historical migration source"
    )
    _require_actor_state_keys(
        modern_template.state_dict(), MODERN_ACTOR_STATE_KEYS,
        "S4 neutral modern migration target",
    )
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
        "modern_template_initialization_seed": MIGRATION_INITIALIZATION_SEED,
        "fresh_migration_parameter_sha256": _state_subset_sha256(
            modern_template.state_dict(), incompatible.missing_keys
        ),
    }
    _require_actor_state_keys(
        modern_template.state_dict(), MODERN_ACTOR_STATE_KEYS,
        "S4 migrated modern Actor",
    )
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
        seed = TEACHER_VALIDATION_SEED
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
        seed = TEACHER_TRAINING_BASE_SEED + 1009 * (absolute + 1)
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
        "teacher_training_noise_rule": "20260921+1009*(absolute_epoch+1)",
        "teacher_validation_noise_seed": TEACHER_VALIDATION_SEED,
    }
    atomic_json(stage_root / "summary.json", summary)
    return actor, summary


def _validate_wgan_exact(
    ctrl: Any,
    legacy: Any,
    actor: Any,
    stepper: Any,
    selection_initial: Sequence[Any],
    validation_reference: Any,
    channel_weights: Any,
    reference_scale: Any,
    projections: Any,
    cfg: Any,
    covariance_weights: Mapping[str, Any],
    *,
    validation_seed: int,
) -> dict[str, Any]:
    """Evaluate a WGAN checkpoint only on ctx5 and its isolated reference pool."""

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
            seed = int(validation_seed) + 1009 * replicate
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
            time_w1 = float(
                ctrl.channelwise_time_w1(controlled_np, reference_np).mean()
            )
            occupation_w1 = float(
                ctrl.channelwise_occupation_w1(controlled_np, reference_np).mean()
            )
            laws.append(float(law))
            times.append(time_w1)
            occupations.append(occupation_w1)
            contexts.append(
                {
                    "ctx5_slot": replicate,
                    "noise_seed": seed,
                    "law": float(law),
                    "mean_time_w1": time_w1,
                    "mean_occupation_w1": occupation_w1,
                }
            )
            if replicate == 0:
                display = {
                    "free": free_np,
                    "controlled": controlled_np,
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
    ctrl: Any,
    legacy: Any,
    actor: Any,
    stepper: Any,
    fit_initial: Sequence[Any],
    selection_initial: Sequence[Any],
    fit_reference: Any,
    validation_reference: Any,
    channel_weights: Any,
    reference_scale: Any,
    cfg: Any,
    covariance_weights: Mapping[str, Any],
    *,
    validation_seed: int,
) -> tuple[Any, Any, list[dict[str, Any]], dict[str, Any], bool, dict[str, Any]]:
    """Execute the exact D WGAN40 handoff with fail-closed checkpoint promotion."""

    import numpy as np
    import torch

    base_seed = int(cfg.seed)
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)
    random.seed(base_seed)
    projections = ctrl.fixed_joint_projections(stepper.n_channels, base_seed)
    teacher = copy.deepcopy(actor)
    teacher.eval()
    teacher_state = copy.deepcopy(actor.state_dict())
    _require_actor_state_keys(
        teacher_state, MODERN_ACTOR_STATE_KEYS, "WGAN40 frozen S6 teacher"
    )
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    critic = ctrl.TimeConditionedWassersteinCritic(
        fit_reference.reshape(-1, stepper.n_channels).mean(dim=0),
        reference_scale,
        horizon_samples=int(cfg.horizon),
        hidden_size=128,
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=float(cfg.critic_learning_rate),
        betas=(0.0, 0.9),
    )
    actor_optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=float(cfg.actor_learning_rate),
        weight_decay=1.0e-5,
    )
    critic_time_indices = tuple(range(15, int(cfg.horizon), 16))
    anchor_time_indices = tuple(range(0, int(cfg.horizon), 16))
    banks = []
    with torch.no_grad():
        for bank in range(int(cfg.critic_pretrain_banks)):
            noise_seed = base_seed + 101 * (bank + 1)
            initial = fit_initial[bank % len(fit_initial)]
            rollout = unprojected_rollout(
                ctrl,
                stepper,
                teacher,
                initial,
                _antithetic(ctrl, noise_seed, cfg, stepper.q),
            )
            banks.append(rollout.scaled[:, 1:].detach())
    history: list[dict[str, Any]] = []
    for step in range(int(cfg.critic_pretrain_steps)):
        critic_optimizer.zero_grad(set_to_none=True)
        gp_seed = base_seed + 50000 + step
        generator = torch.Generator(device="cpu")
        generator.manual_seed(gp_seed)
        audit = ctrl.balanced_wgan_gp_loss(
            critic,
            banks[step % len(banks)],
            fit_reference,
            critic_time_indices,
            gradient_penalty_weight=10.0,
            critic_drift_weight=0.001,
            generator=generator,
        )
        audit.loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
        critic_optimizer.step()
        history.append(
            {
                "stage": "critic_pretrain",
                "step": step + 1,
                "noise_bank": step % len(banks),
                "gp_seed": gp_seed,
                "loss": float(audit.loss.detach()),
            }
        )
    baseline_result = _validate_wgan_exact(
        ctrl,
        legacy,
        actor,
        stepper,
        selection_initial,
        validation_reference,
        channel_weights,
        reference_scale,
        projections,
        cfg,
        covariance_weights,
        validation_seed=validation_seed,
    )
    baseline = baseline_result["aggregate"]
    history.append(
        {
            "stage": "actor_wgan",
            "epoch": 0,
            "eligible": False,
            "validation_law": baseline["law"],
            "validation_mean_time_w1": baseline["mean_time_w1"],
            "validation_mean_occupation_w1": baseline["mean_occupation_w1"],
        }
    )

    def better(candidate: Any, incumbent: Any) -> bool:
        if incumbent is None:
            return True
        if not np.isclose(candidate[0], incumbent[0]):
            return bool(candidate[0] < incumbent[0])
        if not np.isclose(candidate[1], incumbent[1]):
            return bool(candidate[1] < incumbent[1])
        return bool(candidate[2] < incumbent[2])

    best_noninferior = None
    fallback_for_audit = None
    for epoch_index in range(40):
        epoch = epoch_index + 1
        initial = fit_initial[epoch_index % len(fit_initial)]
        noise_seed = base_seed + 1009 * (epoch_index + 1)
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
        for critic_step in range(int(cfg.critic_steps)):
            critic_optimizer.zero_grad(set_to_none=True)
            gp_seed = (
                base_seed
                + 100000 * (epoch_index + 1)
                + critic_step
            )
            gp_seeds.append(gp_seed)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(gp_seed)
            audit = ctrl.balanced_wgan_gp_loss(
                critic,
                detached,
                fit_reference,
                critic_time_indices,
                gradient_penalty_weight=10.0,
                critic_drift_weight=0.001,
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
            legacy,
            rollout,
            fit_reference,
            channel_weights,
            reference_scale,
            projections,
            actor,
            free,
            covariance_weights,
            regularize=True,
        )
        adversarial, _estimate = ctrl.actor_wasserstein_loss(
            critic, rollout.scaled[:, 1:], fit_reference, critic_time_indices
        )
        anchor = ctrl.actor_action_anchor_loss(
            actor,
            teacher,
            rollout,
            anchor_time_indices,
            amplitude_limit=float(cfg.amplitude_limit),
        )
        total = (
            law
            + float(cfg.adversarial_weight) * adversarial
            + float(cfg.teacher_anchor_weight) * anchor
        )
        if not torch.isfinite(total):
            raise RuntimeError("non-finite formal WGAN actor loss")
        total.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        actor_optimizer.step()
        row: dict[str, Any] = {
            "stage": "actor_wgan",
            "epoch": epoch,
            "eligible": True,
            "noise_seed": noise_seed,
            "critic_gp_seeds_json": json.dumps(gp_seeds),
            "train_total": float(total.detach()),
            "train_law": float(law.detach()),
            "train_adversarial": float(adversarial.detach()),
            "train_anchor": float(anchor.detach()),
            "mean_critic_loss": float(np.mean(critic_losses)),
        }
        if epoch == 1 or epoch % int(cfg.validation_every) == 0 or epoch == 40:
            result = _validate_wgan_exact(
                ctrl,
                legacy,
                actor,
                stepper,
                selection_initial,
                validation_reference,
                channel_weights,
                reference_scale,
                projections,
                cfg,
                covariance_weights,
                validation_seed=validation_seed,
            )
            aggregate = result["aggregate"]
            score = (
                0.50 * aggregate["law"] / max(baseline["law"], 1.0e-12)
                + 0.25
                * aggregate["mean_time_w1"]
                / max(baseline["mean_time_w1"], 1.0e-12)
                + 0.25
                * aggregate["mean_occupation_w1"]
                / max(baseline["mean_occupation_w1"], 1.0e-12)
            )
            noninferior = aggregate["law"] <= 1.02 * baseline["law"]
            row.update(
                {
                    "validation_law": aggregate["law"],
                    "validation_mean_time_w1": aggregate["mean_time_w1"],
                    "validation_mean_occupation_w1": aggregate[
                        "mean_occupation_w1"
                    ],
                    "selection_score": score,
                    "law_noninferior": bool(noninferior),
                }
            )
            candidate = (
                float(score),
                float(aggregate["law"]),
                int(epoch),
                copy.deepcopy(actor.state_dict()),
                copy.deepcopy(critic.state_dict()),
                result,
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
        "epochs_executed": 40,
        "accepted_noninferior_checkpoint": bool(accepted),
        "selected_epoch": selected_epoch,
        "selected_score": selected_score,
        "epoch_zero_eligible": False,
        "law_noninferiority_multiplier": 1.02,
        "fallback_update_observed": fallback_for_audit is not None,
        "fallback_update_promoted": False,
        "teacher_retained_if_no_noninferior_update": True,
        "seed_contract": {
            "critic_pretrain_noise": "seed+101*(bank+1)",
            "critic_pretrain_gp": "seed+50000+step",
            "joint_epoch_noise": "seed+1009*(epoch+1)",
            "joint_epoch_gp": "seed+100000*(epoch+1)+critic_step",
        },
        "critic_adam_betas": [0.0, 0.9],
        "actor_gradient_clip": 1.0,
        "critic_gradient_clip": 5.0,
        "rollout": "D_canonical_empirical_fp_rollout_unprojected",
        "hard_projection_or_rescaling_applied": False,
    }
    return actor, critic, history, final_result, accepted, summary


def _training_setup(
    config: Mapping[str, Any], development_npz: Path,
    analytical_dir: Path, refit_dir: Path,
) -> dict[str, Any]:
    """Rebuild the same fresh all-development stack for every resumable phase."""

    import joblib
    import numpy as np
    import torch

    stages = validate_training_schedule(config)
    ctrl = activate_controller(config)
    legacy, historical, _historical_contract = load_stage_authorities(config)
    arrays = load_development(development_npz)
    runs = list(config["source_data"]["development_runs"])
    if runs != ["run-01", "run-02"]:
        raise PermissionError("formal training may use only HUP065 run01/run02")
    model_path = refit_dir / "selected_model.joblib"
    network_path = refit_dir / "network.npz"
    plant_path = refit_dir / "part2_plant_adjacency.npz"
    model = joblib.load(model_path)
    assert_live_part2_model_identity(
        model, sys.modules["mfc_pipeline.part2_state_dependent_rc_sde"]
    )
    network = load_network(network_path)
    with np.load(plant_path, allow_pickle=False) as plant_archive:
        plant_adjacency_input = np.asarray(
            plant_archive["adjacency_input"], dtype=np.float64
        )
        plant_adjacency_normalized = np.asarray(
            plant_archive["adjacency_normalized"], dtype=np.float64
        )
    if not np.allclose(
        np.asarray(model.adjacency_input), plant_adjacency_input,
        rtol=0.0, atol=0.0,
    ):
        raise PermissionError("fitted Part-II model adjacency_input changed")
    if not np.allclose(
        np.asarray(model.adjacency), plant_adjacency_normalized,
        rtol=0.0, atol=0.0,
    ):
        raise PermissionError("fitted Part-II normalized plant adjacency changed")
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
    fit_reference_np, validation_reference_np, reference_split = (
        standardized_reference_pools(model, arrays, runs)
    )
    fit_reference = torch.as_tensor(fit_reference_np, dtype=torch.float64)
    validation_reference = torch.as_tensor(
        validation_reference_np, dtype=torch.float64
    )
    # The 44-key modern template is a migration target, not a WGAN restart.
    # Construct its new local/Markov layers under the reconstructed HUP060
    # seed before S0-S3 historical training is reloaded and migrated.
    torch.manual_seed(MIGRATION_INITIALIZATION_SEED)
    np.random.seed(MIGRATION_INITIALIZATION_SEED)
    random.seed(MIGRATION_INITIALIZATION_SEED)
    stack = ctrl.build_control_stack(model, network, fit_contexts, fit_reference_np, cfg)
    _require_actor_state_keys(
        stack[-1].state_dict(), MODERN_ACTOR_STATE_KEYS,
        "fresh seed20260921 modern Actor template",
    )
    if not np.allclose(
        stack[0].adjacency.detach().cpu().numpy(), plant_adjacency_normalized,
        rtol=0.0, atol=0.0,
    ):
        raise PermissionError("controller world did not inherit Part-II plant adjacency")
    selected_indices = np.flatnonzero(np.asarray(network["target_mask"], dtype=bool))
    if not np.array_equal(np.asarray(stack[0].actuator_indices), selected_indices):
        raise PermissionError("Actor selected indices do not come from Part-I direct mask")
    stepper = stack[2]
    selection_initial = [
        stepper.adapter.initial_state_from_context(item) for item in ctx5_contexts
    ]
    initial_weights, balanced_weights = _channel_weights(fit_reference)
    projections = ctrl.fixed_joint_projections(stepper.n_channels, 20260921)
    return {
        "stages": stages, "ctrl": ctrl, "legacy": legacy, "historical": historical,
        "arrays": arrays, "runs": runs, "model": model, "model_path": model_path,
        "network": network, "network_path": network_path,
        "plant_path": plant_path,
        "plant_adjacency_input": plant_adjacency_input,
        "plant_adjacency_normalized": plant_adjacency_normalized,
        "top1": top1,
        "cfg": cfg, "fit_ledger": fit_ledger, "ctx5_ledger": ctx5_ledger,
        "fit_reference_np": fit_reference_np,
        "validation_reference_np": validation_reference_np,
        "fit_reference": fit_reference,
        "validation_reference": validation_reference,
        "reference_split": reference_split,
        "stack": stack,
        "stepper": stepper, "fit_initial": stack[3],
        "selection_initial": selection_initial,
        "initial_weights": initial_weights, "balanced_weights": balanced_weights,
        "projections": projections,
        "graph_wiring": {
            "part1_selection_network_sha256": sha256_file(network_path),
            "part2_plant_adjacency_sha256": sha256_file(plant_path),
            "actor_selected_indices_from_part1_mask": True,
            "world_adjacency_from_part2_plant": True,
            "part1_adjacency_reused_as_world_plant": False,
        },
    }


def train_formal_stage(
    config: Mapping[str, Any], development_npz: Path, analytical_dir: Path,
    refit_dir: Path, phase_id: str, parent_stage_dir: Path | None,
    output: Path,
) -> dict[str, Any]:
    """Execute exactly one formal teacher stage and publish its fresh checkpoint."""

    import torch

    setup = _training_setup(config, development_npz, analytical_dir, refit_dir)
    stages = setup["stages"]
    ids = tuple(str(stage["stage_id"]) for stage in stages)
    if phase_id not in ids:
        raise ValueError(f"unknown formal teacher stage: {phase_id}")
    index = ids.index(phase_id)
    stage = stages[index]
    parent_hash = None
    migration = None
    if index <= 3:
        actor = _fresh_historical_actor(setup["historical"], setup["stack"], setup["cfg"])
        if index:
            if parent_stage_dir is None:
                raise FileNotFoundError(f"{phase_id} requires its parent checkpoint")
            parent_checkpoint = parent_stage_dir / "training" / "frozen_actor.pt"
            payload = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
            _require_actor_state_keys(
                payload["actor_state_dict"], HISTORICAL_ACTOR_STATE_KEYS,
                f"{phase_id} historical parent checkpoint",
            )
            actor.load_state_dict(payload["actor_state_dict"], strict=True)
            parent_hash = sha256_file(parent_checkpoint)
    else:
        if parent_stage_dir is None:
            raise FileNotFoundError(f"{phase_id} requires its parent checkpoint")
        parent_checkpoint = parent_stage_dir / "training" / "frozen_actor.pt"
        payload = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
        parent_hash = sha256_file(parent_checkpoint)
        if index == 4:
            _require_actor_state_keys(
                payload["actor_state_dict"], HISTORICAL_ACTOR_STATE_KEYS,
                "S4 historical parent checkpoint",
            )
            historical_actor = _fresh_historical_actor(
                setup["historical"], setup["stack"], setup["cfg"]
            )
            historical_actor.load_state_dict(payload["actor_state_dict"], strict=True)
            actor, migration = _migrate_to_modern(historical_actor, setup["stack"][-1])
        else:
            _require_actor_state_keys(
                payload["actor_state_dict"], MODERN_ACTOR_STATE_KEYS,
                f"{phase_id} modern parent checkpoint",
            )
            actor = setup["stack"][-1]
            actor.load_state_dict(payload["actor_state_dict"], strict=True)
    output.mkdir(parents=True, exist_ok=True)
    actor, summary = _train_stage(
        setup["ctrl"], setup["legacy"], actor, setup["stepper"],
        setup["fit_initial"], setup["selection_initial"],
        setup["fit_reference"], setup["validation_reference"],
        setup["initial_weights"], setup["balanced_weights"], setup["stack"][6],
        setup["projections"], setup["cfg"], stage,
        config["controller"]["objective_weights"][stage["objective"]],
        output / "training", parent_hash, migration,
    )
    _require_actor_state_keys(
        actor.state_dict(),
        HISTORICAL_ACTOR_STATE_KEYS if index <= 3 else MODERN_ACTOR_STATE_KEYS,
        f"{phase_id} published Actor",
    )
    receipt = {
        "phase": phase_id,
        "epochs": int(stage["epochs"]),
        "epoch_offset": int(stage["epoch_offset"]),
        "validation_every": int(stage["validation_every"]),
        "learning_rate": float(stage["learning_rate"]),
        "objective": str(stage["objective"]),
        "train_scope": str(stage["train_scope"]),
        "actuator_alpha": float(stage["actuator_alpha"]),
        "best_epoch": int(summary["best_epoch"]),
        "checkpoint_sha256": str(summary["checkpoint_sha256"]),
        "parent_checkpoint_sha256": parent_hash,
        "ctx5_selected_epoch_only": True,
        "arm_reselected": False,
        "reference_split": setup["reference_split"],
        "discarded_robust_branch_executed": False,
        "run03_opened": False,
        "hard_projection_or_rescaling_applied": False,
        "graph_wiring": setup["graph_wiring"],
    }
    atomic_json(output / "stage_execution.json", receipt)
    return receipt


def ctx6_veto_decision(summary: Mapping[str, Any]) -> bool:
    """Frozen terminal rule: pass/fail only, with no ranking or fallback."""

    required = (
        "gate_c_pass",
        "mean_time_w1_controlled",
        "mean_time_w1_free",
        "mean_occupation_w1_controlled",
        "mean_occupation_w1_free",
        "gate_b_pass_count",
    )
    if any(key not in summary for key in required):
        return False
    numeric = [
        summary["mean_time_w1_controlled"],
        summary["mean_time_w1_free"],
        summary["mean_occupation_w1_controlled"],
        summary["mean_occupation_w1_free"],
        summary["gate_b_pass_count"],
    ]
    try:
        finite = all(math.isfinite(float(value)) for value in numeric)
    except (TypeError, ValueError):
        return False
    return bool(
        finite
        and bool(summary["gate_c_pass"])
        and float(summary["mean_time_w1_controlled"])
        < float(summary["mean_time_w1_free"])
        and float(summary["mean_occupation_w1_controlled"])
        < float(summary["mean_occupation_w1_free"])
        and int(summary["gate_b_pass_count"]) >= 1
    )


def wgan40_and_ctx6_veto(
    config: Mapping[str, Any], development_npz: Path, analytical_dir: Path,
    refit_dir: Path, s6_stage_dir: Path, output: Path,
) -> dict[str, Any]:
    """Run canonical WGAN40, select only an epoch on ctx5, then apply ctx6 veto."""

    import numpy as np
    import pandas as pd
    import torch
    from .control import _contexts_and_futures, evaluate_detailed

    setup = _training_setup(config, development_npz, analytical_dir, refit_dir)
    s6_checkpoint = s6_stage_dir / "training" / "frozen_actor.pt"
    payload = torch.load(s6_checkpoint, map_location="cpu", weights_only=False)
    _require_actor_state_keys(
        payload["actor_state_dict"], MODERN_ACTOR_STATE_KEYS,
        "WGAN40 S6 parent checkpoint",
    )
    actor = setup["stack"][-1]
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    actor, critic, history, result, accepted, wgan_summary = _train_wgan_exact(
        setup["ctrl"], setup["legacy"], actor, setup["stepper"],
        setup["fit_initial"], setup["selection_initial"],
        setup["fit_reference"], setup["validation_reference"],
        setup["balanced_weights"], setup["stack"][6], setup["cfg"],
        config["controller"]["objective_weights"]["covariance"],
        validation_seed=int(config["controller"]["validation_seed"]),
    )
    _require_actor_state_keys(
        actor.state_dict(), MODERN_ACTOR_STATE_KEYS, "selected WGAN/S6 Actor"
    )
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(
        output / "wgan40_training_history.csv", index=False, encoding="utf-8-sig"
    )
    atomic_json(output / "wgan40_training_summary.json", wgan_summary)
    checkpoint_payload = {
        "schema_version": "hup065-fresh-teacher1050-wgan40-v1",
        "subject": "HUP065", "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict() if accepted else None,
        "controller_config": asdict(setup["cfg"]),
        "analytical_top1": setup["top1"],
        "teacher_parent_sha256": sha256_file(s6_checkpoint),
        "teacher_epochs_executed": 1050, "wgan_epochs_executed": 40,
        "formal_stage_ids": list(FORMAL_STAGE_IDS),
        "discarded_s2r_executed": False, "wgan_accepted": bool(accepted),
        "epoch_zero_eligible": False, "ctx5_selected_epoch_only": True,
        "wgan_selection_outcome": (
            "ACCEPTED_NONINFERIOR_WGAN" if accepted
            else "NO_ACCEPTED_WGAN_RETAIN_FROZEN_S6_TEACHER"
        ),
        "formal_wgan_handoff": dict(setup["ctrl"].FORMAL_WGAN_PATCH_RECEIPT),
        "wgan_exact_summary": wgan_summary,
        "formal_rollout": dict(setup["ctrl"].FORMAL_ROLLOUT_RECEIPT),
        "controller_adapter": dict(
            setup["ctrl"].FORMAL_CONTROLLER_ADAPTER_RECEIPT
        ),
        "ctx5_selected_arm": False, "ctx6_used": True, "ctx7_used": False,
        "run03_opened": False, "old_checkpoint_used": False,
        "old_fitted_model_used": False,
    }
    checkpoint = output / "selected_controller.pt"
    _atomic_torch_save(torch, checkpoint, checkpoint_payload)
    display = result["display"]
    np.savez_compressed(
        output / "ctx5_selected_rollout.npz",
        free_standardized=display["free"], controlled_standardized=display["controlled"],
        reference_standardized=display["reference"], controls=display["controls"],
        standard_normal=display["noise"], channels=setup["arrays"]["channels"],
        direct_mask=np.asarray(setup["network"]["target_mask"], dtype=bool),
    )
    ctx6_contexts, ctx6_futures, ctx6_ledger = _contexts_and_futures(
        setup["arrays"], setup["runs"], config["contexts"]["terminal_veto_only"]
    )
    if any(int(row["context_index"]) != 6 for row in ctx6_ledger):
        raise RuntimeError("non-ctx6 data entered terminal veto")
    observed = np.stack(
        [setup["model"].transform.scaler.transform(item) for item in ctx6_futures]
    )
    initials = [
        setup["stepper"].adapter.initial_state_from_context(item)
        for item in ctx6_contexts
    ]
    summary, rows, veto_display = evaluate_detailed(
        setup["ctrl"], actor, setup["stepper"], initials,
        setup["validation_reference_np"], setup["network"]["target_mask"], setup["cfg"],
        fold_id="all-development", stage="ctx6_terminal_veto",
        ledger=ctx6_ledger, observed_standardized=observed,
    )
    pd.DataFrame(rows).to_csv(
        output / "ctx6_channel_metrics.csv", index=False, encoding="utf-8-sig"
    )
    np.savez_compressed(
        output / "ctx6_rollout.npz",
        free_standardized=veto_display["free"],
        controlled_standardized=veto_display["controlled"],
        reference_standardized=veto_display["reference"],
        controls=veto_display["controls"], standard_normal=veto_display["noise"],
        channels=setup["arrays"]["channels"],
        direct_mask=np.asarray(setup["network"]["target_mask"], dtype=bool),
    )
    veto_pass = ctx6_veto_decision(summary)
    receipt = {
        "phase": "WGAN40_CTX5_SELECT_CTX6_VETO",
        "teacher_epochs_total": 1050, "wgan_epochs": 40,
        "formal_stage_ids": list(FORMAL_STAGE_IDS),
        "wgan_accepted": bool(accepted), "ctx6_terminal_veto_pass": veto_pass,
        "wgan_selection_outcome": (
            "ACCEPTED_NONINFERIOR_WGAN" if accepted
            else "NO_ACCEPTED_WGAN_RETAIN_FROZEN_S6_TEACHER"
        ),
        "formal_wgan_handoff": dict(setup["ctrl"].FORMAL_WGAN_PATCH_RECEIPT),
        "ctx6_veto_rule": {
            "all_values_finite_and_gate_c_all_contexts": True,
            "aggregate_time_w1_controlled_less_than_free": True,
            "aggregate_occupation_w1_controlled_less_than_free": True,
            "minimum_full_six_gate_b_channel_count": 1,
            "may_rank_or_rescue": False,
        },
        "context_5_role": "checkpoint_epoch_selection_only",
        "context_6_role": "terminal_veto_only",
        "arm_reselected": False, "discarded_robust_branch_executed": False,
        "run03_opened": False, "checkpoint_sha256": sha256_file(checkpoint),
        "ctx5_metrics": result["aggregate"], "ctx6_metrics": summary,
        "wgan_exact_summary": wgan_summary,
        "hard_projection_or_rescaling_applied": False,
        "controller_adapter": dict(
            setup["ctrl"].FORMAL_CONTROLLER_ADAPTER_RECEIPT
        ),
        "reference_split": setup["reference_split"],
        "graph_wiring": setup["graph_wiring"],
    }
    atomic_json(output / "final_development_freeze.json", receipt)
    if not veto_pass:
        raise RuntimeError("ctx6 terminal veto failed; OUTER must remain sealed")
    return receipt


def _legacy_monolithic_s0_s6_never_dispatched(
    config: Mapping[str, Any], development_npz: Path, analytical_dir: Path,
    refit_dir: Path, output: Path,
) -> None:
    """Permanently disabled predecessor of the seven atomic teacher phases."""

    raise RuntimeError("legacy monolithic S0-S6/WGAN route is permanently disabled")

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
    network_path = refit_dir / "network.npz"
    model = joblib.load(model_path)
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
    fit_reference_np, validation_reference_np, _reference_split = (
        standardized_reference_pools(model, arrays, runs)
    )
    fit_reference = torch.as_tensor(fit_reference_np, dtype=torch.float64)
    validation_reference = torch.as_tensor(validation_reference_np, dtype=torch.float64)
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    random.seed(int(cfg.seed))
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
            fit_reference, validation_reference, initial_weights, balanced_weights,
            stack[6], projections,
            cfg, stage, config["controller"]["objective_weights"][stage["objective"]],
            stage_dir, parent_hash, migration if index == 4 else None,
        )
        parent_hash = str(summary["checkpoint_sha256"])
        stage_summaries.append(summary)
    if sum(int(item["epochs_executed"]) for item in stage_summaries) != 1050:
        raise RuntimeError("formal teacher chain did not execute exactly 1050 epochs")
    actor, critic, wgan_history, wgan_result, wgan_accepted = ctrl.train_wgan(
        actor, stepper, fit_initial, selection_initial,
        fit_reference, validation_reference, None, stack[7], stack[6], projections, cfg,
        external_plant_fidelity_gate=True,
    )
    wgan_dir = output / "FORMAL_WGAN_FULL_0_40"
    wgan_dir.mkdir(parents=False, exist_ok=False)
    pd.DataFrame(wgan_history).to_csv(
        wgan_dir / "training_history.csv", index=False, encoding="utf-8-sig"
    )
    checkpoint_payload = {
        "schema_version": "hup065-fresh-s0-s6-1050-wgan40-v1",
        "subject": "HUP065", "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict() if wgan_accepted else None,
        "controller_config": asdict(cfg),
        "selected_stage": (
            "FORMAL_WGAN_FULL_0_40" if wgan_accepted
            else "S6_PRECISION_ALL_PARAMS_870_1050"
        ),
        "analytical_top1": top1,
        "target_mask_sha256": array_sha256(network["target_mask"]),
        "model_sha256": sha256_file(model_path), "network_sha256": sha256_file(network_path),
        "teacher_epochs_executed": 1050, "wgan_epochs_executed": 40,
        "formal_stage_ids": list(FORMAL_STAGE_IDS),
        "discarded_s2r_executed": False,
        "wgan_accepted": bool(wgan_accepted), "epoch_zero_eligible": False,
        "ctx5_selected_epoch_only": True, "ctx5_selected_arm": False,
        "ctx6_used": False, "ctx7_used": False, "sealed_run_opened": False,
        "old_checkpoint_used": False, "old_fitted_model_used": False,
        "fresh_initialization_seed": int(cfg.seed),
    }
    checkpoint = output / "selected_controller.pt"
    _atomic_torch_save(torch, checkpoint, checkpoint_payload)
    display = wgan_result.display
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
            "schema_version": "hup065-fresh-training-receipt-v2",
            "teacher_epochs_executed": 1050, "wgan_epochs_executed": 40,
            "stage_chain": stage_summaries,
            "discarded_s2r_executed": False,
            "wgan_accepted": bool(wgan_accepted),
            "ctx5_metrics": wgan_result.aggregate,
            "ctx5_runs": ctx5_ledger, "fit_context_count": len(fit_ledger),
            "checkpoint_sha256": sha256_file(checkpoint),
            "elapsed_seconds": time.perf_counter() - started,
            "arm_reselection_after_analytical": False,
            "sealed_run_opened": False,
        },
    )


__all__ = [
    "FORMAL_STAGE_IDS", "load_stage_authorities", "train_formal_stage",
    "validate_training_schedule", "wgan40_and_ctx6_veto", "ctx6_veto_decision",
]
