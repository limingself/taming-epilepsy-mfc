#!/usr/bin/env python
"""Patient-adaptive structured Actor--WGAN control on a frozen Graph-RC SDE.

The method is an empirical-particle HJB--FP-motivated structured
Actor--WGAN-GP; its critic is a Kantorovich critic for distributional
discrepancy.

This file generalizes the manuscript HUP060 Part-III implementation without
hard-coding the number of electrodes.  It has three deliberately separate
execution modes:

``analytical-screen``
    Evaluate one weighted-ridge feedback/graph-heat candidate.  This cheap
    stage is intended for the predeclared 3 x 3 development-only screen.
``development``
    Train the structured full-Markov Actor and, optionally, fine-tune it with
    a time-conditioned WGAN-GP critic.  Checkpoint selection uses development
    contexts and an independently partitioned interictal reference pool only.
``prefix-refit``
    Adapt a frozen development checkpoint from an observed seizure prefix.
    Validation/future inputs are forbidden and the final scheduled epoch is
    retained; there is no outcome-based checkpoint selection.

All particle comparisons use the same antithetic innovation bank for the
uncontrolled and controlled laws.  The state-dependent ictal diffusion is
part of the frozen Part-II plant and is never optimized against the Part-III
control objective.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import expm
import torch
from torch import Tensor, nn


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
PIPELINE_ROOT = WORKSPACE / "taming-epilepsy-mfc-baseline"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from mfc_pipeline.actor_wgan import (  # noqa: E402
    actor_action_anchor_loss,
    actor_wasserstein_loss,
    balanced_wgan_gp_loss,
)
from mfc_pipeline.causal_ltv_particle_rollout import (  # noqa: E402
    FrozenIctalGraphRCBatchStepper,
)
from mfc_pipeline.causal_ltv_riccati import (  # noqa: E402
    FrozenGraphRCMarkovAdapter,
)
from mfc_pipeline.full_markov_hjb_fp_wgan import (  # noqa: E402
    TimeConditionedWassersteinCritic,
    set_requires_grad,
)
from mfc_pipeline.part2_state_dependent_rc_sde import (  # noqa: E402,F401
    ResidualGraphRCSDE,
)
from mfc_pipeline.sequential_covariance_hjb import (  # noqa: E402
    StructuredSplineCovarianceActor,
)
from mfc_pipeline.square_wave_mfc import TorchGraphRCSDE  # noqa: E402
from multistep_drift_calibration import build_calibrated_stepper  # noqa: E402


SCHEMA_VERSION = "paper-exact-patient-controller-v2.2"
HORIZON = 256
SAMPLING_RATE_HZ = 256.0
PARTICLES = 32
CENTRALITY_QUANTILE = 0.65
CONTROL_STEP_SCALE = 0.5
AMPLITUDE_LIMIT = 1.8
REFERENCE_PER_ACTUATOR_RMS = 0.405
DEFAULT_SEED = 20261011
EVALUATION_SEEDS = (20261101, 20261102, 20261103)
DISPLAY_EVALUATION_SEED = EVALUATION_SEEDS[0]
METHOD_LABEL = "empirical-particle HJB--FP-motivated structured Actor--WGAN-GP"
FORBIDDEN_PREFIX_KEYS = ("future", "observed", "outcome", "endpoint", "test")


@dataclass(frozen=True)
class ControllerConfig:
    horizon: int = HORIZON
    sampling_rate_hz: float = SAMPLING_RATE_HZ
    particles: int = PARTICLES
    control_step_scale: float = CONTROL_STEP_SCALE
    diffusion_scale: float = 0.79451175
    increment_scale: float = 1.0
    persistence_skip: float = 0.0
    graph_diffusion_time: float = 0.5
    amplitude_limit: float = AMPLITUDE_LIMIT
    energy_rms: float = REFERENCE_PER_ACTUATOR_RMS
    total_episode_energy_budget: float | None = None
    target_quantile: float = CENTRALITY_QUANTILE
    base_gain_scale: float = 0.50
    base_gain_ridge: float = 0.20
    basis_count: int = 8
    decoded_hidden_size: int = 64
    markov_hidden_size: int = 96
    residual_scale: float = 0.15
    markov_residual_scale: float = 0.60
    actuator_alpha: float = 1.0
    local_gain_initial_fraction: float = 0.0
    local_gain_maximum_fraction: float = 0.05
    teacher_epochs: int = 180
    teacher_learning_rate: float = 3.0e-4
    wgan_epochs: int = 40
    actor_learning_rate: float = 1.0e-5
    critic_learning_rate: float = 1.0e-4
    validation_every: int = 4
    critic_pretrain_steps: int = 24
    critic_pretrain_banks: int = 3
    critic_steps: int = 3
    adversarial_weight: float = 0.50
    teacher_anchor_weight: float = 0.20
    gradient_penalty: float = 10.0
    critic_drift: float = 1.0e-3
    seed: int = DEFAULT_SEED


@dataclass
class BudgetedParticleRollout:
    markov: Tensor
    scaled: Tensor
    controls: Tensor
    commands: Tensor
    conditional_std: Tensor
    noise: Tensor
    delivered_energy: Tensor
    energy_budget: Tensor


@dataclass
class ValidationResult:
    aggregate: dict[str, float]
    contexts: list[dict[str, float]]
    display: dict[str, np.ndarray]


@dataclass(frozen=True)
class ValidationInvariantCache:
    """Actor-independent validation inputs bound to a fail-closed hash key."""

    noise: tuple[Tensor, ...]
    free: tuple[Tensor, ...]
    noise_sha256: tuple[str, ...]
    free_sha256: tuple[str, ...]
    cache_key_sha256: str
    plant_sha256: str
    config_sha256: str
    initial_state_sha256: tuple[str, ...]
    reference_sha256: str
    reference_shape: tuple[int, ...]
    model_shape_contract: tuple[int, ...]
    seed_offset: int
    particles: int
    horizon: int
    latent_dimension: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    return sha256_array(value.detach().cpu().numpy())


def _json_sha256(value: Any) -> str:
    payload = asdict(value) if is_dataclass(value) else value
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


_VALIDATION_STEPPER_TENSORS = (
    "variance_coef",
    "variance_intercept",
    "variance_calibration",
    "log_variance_bounds",
    "correlation_root",
)
_VALIDATION_ADAPTER_TENSORS = (
    "components",
    "pca_mean",
    "adjacency",
    "w_in",
    "w_res",
    "feature_mean",
    "feature_scale",
    "drift_coef",
    "drift_intercept",
    "control_map",
    "control_channel_map",
)


def _validation_plant_sha256(stepper: Any) -> str:
    digest = sha256()
    digest.update(
        f"{type(stepper).__module__}.{type(stepper).__qualname__}".encode("utf-8")
    )
    for name in _VALIDATION_STEPPER_TENSORS:
        digest.update(name.encode("ascii"))
        digest.update(_tensor_sha256(getattr(stepper, name)).encode("ascii"))
    adapter = stepper.adapter
    for name in _VALIDATION_ADAPTER_TENSORS:
        digest.update(f"adapter.{name}".encode("ascii"))
        digest.update(_tensor_sha256(getattr(adapter, name)).encode("ascii"))
    scalar_contract = {
        "diffusion_scale": float(stepper.diffusion_scale),
        "q": int(stepper.q),
        "n_channels": int(stepper.n_channels),
        "state_dim": int(stepper.state_dim),
        "actuator_dim": int(stepper.actuator_dim),
        "control_step_scale": float(adapter.control_step_scale),
        "reservoir_size": int(adapter.reservoir_size),
        "maximum_delay": int(adapter.maximum_delay),
        "delays": list(adapter.delays),
        "history_length": int(adapter.history_length),
        "leak_rate": float(adapter.leak_rate),
        "preserve_physical_control_residual": bool(
            adapter.preserve_physical_control_residual
        ),
        "multistep_calibration": (
            asdict(stepper.multistep_calibration)
            if is_dataclass(getattr(stepper, "multistep_calibration", None))
            else None
        ),
    }
    digest.update(_json_sha256(scalar_contract).encode("ascii"))
    return digest.hexdigest()


def _resolve_declared_artifact(
    declared: str | Path,
    *,
    selection_path: Path,
    fallback_directory: Path | None = None,
) -> Path:
    """Resolve a provenance path while tolerating relocation of one artifact tree."""

    raw = Path(str(declared))
    candidates = [raw] if raw.is_absolute() else [selection_path.parent / raw]
    candidates.append(selection_path.parent / raw.name)
    if fallback_directory is not None:
        candidates.append(fallback_directory / raw.name)
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"declared provenance artifact is unavailable: {declared}")


def load_frozen_oof_plant_selection(
    path: Path,
    *,
    deployment_model_path: Path,
) -> dict[str, Any]:
    """Validate the immutable conditional OOF Part-II plant decision.

    Stage A calibrates closed-loop drift exposure at kappa=1.  Stage B is
    conditionally available only after a Stage-A four-gate failure and jointly
    calibrates drift exposure and diffusion.  The resolver emits one canonical
    contract so downstream control code cannot silently mix the two stages.
    """

    selection_path = Path(path).resolve()
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "subject",
        "plant_pass",
        "selected_increment_scale",
        "selected_persistence_skip",
        "selected_diffusion_scale",
        "selected_metrics",
        "source_stage",
        "source_selection",
        "source_selection_sha256",
        "source_oof_bundle",
        "source_oof_bundle_sha256",
        "source_oof_manifest",
        "source_oof_manifest_sha256",
        "source_calibration_hashes",
        "protocol",
        "protocol_sha256",
        "stage_a",
        "stage_b",
        "controller_may_run",
        "sampling_rate_hz",
        "update_interval_samples",
        "deployment_refit_model_used",
        "controlled_rollout_used",
        "final_ictal_array_opened",
        "heldout_reference_opened",
        "support_contract",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise KeyError(f"frozen OOF selection lacks fields: {missing}")
    if payload["schema_version"] != "paper-exact-frozen-oof-plant-selection-v1.0":
        raise ValueError("unsupported frozen OOF plant-selection schema")
    for forbidden_flag in (
        "deployment_refit_model_used",
        "controlled_rollout_used",
        "final_ictal_array_opened",
        "heldout_reference_opened",
    ):
        if bool(payload[forbidden_flag]):
            raise PermissionError(f"OOF selection violates {forbidden_flag}=false")
    if not bool(payload["plant_pass"]) or not bool(payload["controller_may_run"]):
        raise PermissionError("frozen OOF Part-II plant fidelity gate failed")
    if payload["support_contract"] != "fold_matched_predictive_laws_v1":
        raise ValueError("OOF selection does not use fold-matched predictive laws")
    if int(payload["update_interval_samples"]) != 1 or not np.isclose(
        float(payload["sampling_rate_hz"]), SAMPLING_RATE_HZ
    ):
        raise ValueError("OOF plant selection changed the 256-Hz one-sample update")
    increment_scale = float(payload["selected_increment_scale"])
    persistence_skip = float(payload["selected_persistence_skip"])
    diffusion_scale = float(payload["selected_diffusion_scale"])
    if not (0.0 <= increment_scale <= 1.5):
        raise ValueError("invalid frozen OOF increment scale")
    if not (-0.5 <= persistence_skip <= 0.75):
        raise ValueError("invalid frozen OOF persistence skip")
    if not np.isfinite(diffusion_scale) or diffusion_scale <= 0.0:
        raise ValueError("invalid frozen OOF diffusion scale")
    if not isinstance(payload["selected_metrics"], Mapping):
        raise TypeError("selected_metrics must be a mapping")

    provenance_path = selection_path.parent / "canonical_provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError("canonical OOF selection lacks canonical_provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    selection_hash = sha256_file(selection_path)
    if provenance.get("canonical_selection_sha256") != selection_hash:
        raise ValueError("canonical OOF selection/provenance hash mismatch")
    resolver_path = HERE / "resolve_oof_plant_selection.py"
    if sha256_file(resolver_path) != str(provenance.get("resolver_sha256", "")):
        raise ValueError("canonical OOF resolver implementation hash mismatch")

    deployment_model_path = Path(deployment_model_path).resolve()
    artifact_directory = deployment_model_path.parent
    selected_config_path = artifact_directory / "selected_config.json"
    split_contract_path = artifact_directory / "split_contract.json"
    if not selected_config_path.is_file() or not split_contract_path.is_file():
        raise FileNotFoundError(
            "deployment plant must retain selected_config.json and split_contract.json"
        )
    selected_config = json.loads(selected_config_path.read_text(encoding="utf-8"))
    if str(selected_config.get("subject", "")).upper() != str(
        payload["subject"]
    ).upper():
        raise ValueError("OOF selection patient does not match deployment plant")

    oof_bundle = _resolve_declared_artifact(
        payload["source_oof_bundle"],
        selection_path=selection_path,
    )
    oof_manifest = _resolve_declared_artifact(
        payload["source_oof_manifest"],
        selection_path=selection_path,
    )
    source_selection = _resolve_declared_artifact(
        payload["source_selection"], selection_path=selection_path
    )
    protocol_path = _resolve_declared_artifact(
        payload["protocol"], selection_path=selection_path, fallback_directory=HERE
    )
    checks = (
        (oof_bundle, "source_oof_bundle_sha256"),
        (oof_manifest, "source_oof_manifest_sha256"),
        (source_selection, "source_selection_sha256"),
        (protocol_path, "protocol_sha256"),
    )
    for artifact, field in checks:
        if sha256_file(artifact) != str(payload[field]):
            raise ValueError(f"frozen OOF provenance hash mismatch: {field}")
    source_stage = str(payload["source_stage"])
    if source_stage not in {"A+diffusion", "B"}:
        raise ValueError("canonical OOF source stage must be A+diffusion or B")
    stage_a_record = dict(payload["stage_a"])
    stage_a_path = _resolve_declared_artifact(
        stage_a_record["selection"], selection_path=selection_path
    )
    if sha256_file(stage_a_path) != str(stage_a_record["selection_sha256"]):
        raise ValueError("canonical OOF Stage-A selection hash mismatch")
    if source_stage == "A+diffusion":
        diffusion_record = dict(payload.get("post_stage_a_diffusion") or {})
        diffusion_path = _resolve_declared_artifact(
            diffusion_record["selection"], selection_path=selection_path
        )
        if sha256_file(diffusion_path) != str(diffusion_record["selection_sha256"]):
            raise ValueError("canonical OOF diffusion selection hash mismatch")
        if diffusion_path != source_selection:
            raise ValueError("canonical OOF primary source is not its diffusion selection")
        if payload.get("stage_b") is not None:
            raise ValueError("Stage-B record is forbidden on the Stage-A-pass route")
    else:
        stage_b_record = dict(payload.get("stage_b") or {})
        stage_b_path = _resolve_declared_artifact(
            stage_b_record["selection"], selection_path=selection_path
        )
        if sha256_file(stage_b_path) != str(stage_b_record["selection_sha256"]):
            raise ValueError("canonical OOF Stage-B selection hash mismatch")
        if stage_b_path != source_selection:
            raise ValueError("canonical OOF primary source is not its Stage-B selection")
        if payload.get("post_stage_a_diffusion") is not None:
            raise ValueError("separate diffusion selection is forbidden on Stage B")
    prepared_oof_manifest = json.loads(oof_manifest.read_text(encoding="utf-8"))
    if "source_oof_manifest" not in prepared_oof_manifest:
        raise KeyError("prepared OOF manifest lacks its root-manifest provenance")
    root_oof_manifest = _resolve_declared_artifact(
        prepared_oof_manifest["source_oof_manifest"], selection_path=oof_manifest,
        fallback_directory=deployment_model_path.parent / "oof",
    )
    root_oof_hash = sha256_file(root_oof_manifest)
    if root_oof_hash != str(
        prepared_oof_manifest.get("source_oof_manifest_sha256", "")
    ):
        raise ValueError("prepared OOF manifest/root-manifest hash mismatch")
    prepared_root = _resolve_declared_artifact(
        prepared_oof_manifest["source_oof_manifest"],
        selection_path=oof_manifest,
        fallback_directory=deployment_model_path.parent / "oof",
    )
    if prepared_root != root_oof_manifest:
        raise ValueError("prepared and frozen selections name different OOF roots")

    root_manifest = json.loads(root_oof_manifest.read_text(encoding="utf-8"))
    if str(root_manifest.get("subject", "")).upper() != str(payload["subject"]).upper():
        raise ValueError("OOF root manifest subject mismatch")
    if sha256_file(deployment_model_path) != str(
        root_manifest.get("source_deployment_model_sha256", "")
    ):
        raise ValueError("OOF calibration was frozen for a different deployment plant")
    if sha256_file(selected_config_path) != str(
        root_manifest.get("source_selected_config_sha256", "")
    ):
        raise ValueError("OOF selection/model-configuration hash mismatch")
    if sha256_file(split_contract_path) != str(
        root_manifest.get("source_split_contract_sha256", "")
    ):
        raise ValueError("OOF selection/split-contract hash mismatch")

    source_hashes = dict(payload["source_calibration_hashes"])
    expected_sources = {
        "stage_a_runner_sha256": HERE / "calibrate_oof_multistep_drift.py",
        "stage_a_wrapper_sha256": HERE / "multistep_drift_calibration.py",
        "stage_a_metric_sha256": HERE / "matched_support_plant_metrics.py",
        "selected_stage_runner_sha256": (
            HERE / (
                "calibrate_oof_diffusion.py"
                if source_stage == "A+diffusion"
                else "calibrate_oof_joint_plant.py"
            )
        ),
        "selected_stage_wrapper_sha256": HERE / "multistep_drift_calibration.py",
        "selected_stage_metric_sha256": HERE / "matched_support_plant_metrics.py",
    }
    if source_stage == "A+diffusion":
        expected_sources["post_stage_a_diffusion_runner_sha256"] = (
            HERE / "calibrate_oof_diffusion.py"
        )
    for field, source_path in expected_sources.items():
        if sha256_file(source_path) != str(source_hashes.get(field, "")):
            raise ValueError(f"canonical OOF source hash mismatch: {field}")

    payload = dict(payload)
    payload["selected_increment_scale"] = increment_scale
    payload["selected_persistence_skip"] = persistence_skip
    payload["selected_diffusion_scale"] = diffusion_scale
    payload["source_root_oof_manifest"] = str(root_oof_manifest)
    payload["source_root_oof_manifest_sha256"] = root_oof_hash
    payload["selection_file"] = str(selection_path)
    payload["selection_file_sha256"] = selection_hash
    return payload


def config_with_frozen_oof_plant(
    config: ControllerConfig,
    selection: Mapping[str, Any],
) -> ControllerConfig:
    """Apply the three OOF-selected plant scalars before control ranking."""

    if not bool(selection.get("plant_pass", False)):
        raise PermissionError("cannot configure controller from a failed OOF plant")
    return replace(
        config,
        diffusion_scale=float(selection["selected_diffusion_scale"]),
        increment_scale=float(selection["selected_increment_scale"]),
        persistence_skip=float(selection["selected_persistence_skip"]),
    )


OOF_PROVENANCE_HASH_FIELDS = (
    "frozen_oof_plant_selection_sha256",
    "source_oof_bundle_sha256",
    "source_oof_manifest_sha256",
    "source_root_oof_manifest_sha256",
)


def inherit_frozen_oof_hashes(
    current: Mapping[str, str], development_checkpoint: Mapping[str, Any]
) -> dict[str, str]:
    """Carry the immutable Part-II decision into a prefix-refit checkpoint."""

    inherited = dict(current)
    source = dict(development_checkpoint.get("input_hashes", {}))
    missing = [field for field in OOF_PROVENANCE_HASH_FIELDS if field not in source]
    if missing:
        raise KeyError(
            "development checkpoint lacks frozen OOF provenance: "
            f"{missing}"
        )
    for field in OOF_PROVENANCE_HASH_FIELDS:
        value = str(source[field])
        if len(value) != 64:
            raise ValueError(f"invalid frozen OOF SHA-256 field: {field}")
        inherited[field] = value
    return inherited


def _load_legacy_hup060_module() -> None:
    """Register the historic joblib module name without changing its code."""

    name = "part2_state_dependent_base"
    if name in sys.modules:
        return
    path = (
        WORKSPACE
        / "distribution_control_paper_final"
        / "part2_rc_sde"
        / "code"
        / "state_dependent_rc_sde.py"
    )
    if not path.is_file():
        return
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import legacy model definition: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


def load_fitted_model(path: Path) -> Any:
    path = Path(path).resolve()
    try:
        model = joblib.load(path)
    except ModuleNotFoundError as error:
        if error.name != "part2_state_dependent_base":
            raise
        _load_legacy_hup060_module()
        model = joblib.load(path)
    required = (
        "transform",
        "config",
        "adjacency",
        "diffusion_variance_model",
        "diffusion_correlation_root",
    )
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise TypeError(f"model is not a fitted ResidualGraphRCSDE: {missing}")
    if str(model.config.diffusion_mode) != "state_dependent":
        raise ValueError("Part III requires the frozen state-dependent ictal diffusion")
    return model


def load_network(path: Path, model: Any) -> dict[str, np.ndarray]:
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as archive:
        network = {key: archive[key].copy() for key in archive.files}
    required = {"adjacency", "target_mask", "centrality_score", "channels"}
    missing = sorted(required.difference(network))
    if missing:
        raise KeyError(f"network artifact lacks keys: {missing}")
    adjacency = np.asarray(network["adjacency"], dtype=np.float64)
    channels = np.asarray(network["channels"]).astype(str)
    score = np.asarray(network["centrality_score"], dtype=np.float64)
    mask = np.asarray(network["target_mask"], dtype=bool)
    n_channels = int(model.transform.n_channels_)
    if adjacency.shape != (n_channels, n_channels):
        raise ValueError("network/model channel dimensions differ")
    if channels.shape != (n_channels,) or score.shape != (n_channels,):
        raise ValueError("network channel metadata has an invalid shape")
    if mask.shape != (n_channels,) or not mask.any():
        raise ValueError("Part-I target mask is empty or malformed")
    threshold = float(np.quantile(score, CENTRALITY_QUANTILE, method="linear"))
    expected = score > threshold
    if not np.array_equal(mask, expected):
        raise ValueError("target_mask is not the strict Part-I S_i > Q_0.65 rule")
    adjacency_input = getattr(model, "adjacency_input", None)
    if adjacency_input is not None and not np.allclose(
        np.asarray(adjacency_input), adjacency, rtol=1.0e-8, atol=1.0e-10
    ):
        raise ValueError("frozen RC model and Part-I adjacency differ")
    network["target_mask"] = mask
    network["channels"] = channels
    return network


def _npz_keys(path: Path) -> tuple[str, ...]:
    with np.load(path, allow_pickle=False) as archive:
        return tuple(str(key) for key in archive.files)


def load_numeric_array(
    path: Path,
    *,
    key: str | None,
    preferred_keys: Sequence[str],
    prefix_strict: bool = False,
) -> np.ndarray:
    path = Path(path).resolve()
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if path.suffix.lower() != ".npz":
        raise ValueError(f"only .npy and .npz arrays are accepted: {path}")
    keys = _npz_keys(path)
    if prefix_strict:
        forbidden = [
            item
            for item in keys
            if any(token in item.lower() for token in FORBIDDEN_PREFIX_KEYS)
        ]
        if forbidden:
            raise PermissionError(
                f"prefix-only context archive contains forbidden keys: {forbidden}"
            )
    selected = key
    if selected is None:
        selected = next((item for item in preferred_keys if item in keys), None)
    if selected is None and len(keys) == 1:
        selected = keys[0]
    if selected is None or selected not in keys:
        raise KeyError(f"cannot choose an array from {path.name}; keys={keys}")
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive[selected], dtype=np.float64)


def validate_coordinate_metadata(
    path: Path,
    *,
    metadata_key: str,
    declared: str,
) -> None:
    """Cross-check optional NPZ coordinate metadata against the CLI contract."""

    path = Path(path).resolve()
    if path.suffix.lower() != ".npz":
        return
    with np.load(path, allow_pickle=False) as archive:
        if metadata_key in archive.files:
            value = np.asarray(archive[metadata_key])
            if value.size != 1:
                raise ValueError(f"{metadata_key} metadata must be scalar")
            observed = str(value.reshape(-1)[0])
        elif "metadata_json" in archive.files:
            raw = np.asarray(archive["metadata_json"])
            if raw.size != 1:
                raise ValueError("metadata_json must be scalar")
            metadata = json.loads(str(raw.reshape(-1)[0]))
            if metadata_key == "context_coordinate":
                phrase = str(metadata.get("context_coordinate", "")).lower()
                observed = (
                    "raw_model_input"
                    if "raw model input" in phrase
                    else "legacy_hup060_scaled_model_input"
                    if "scaled" in phrase
                    else ""
                )
            elif metadata_key == "law_coordinate":
                phrases = " ".join(
                    str(metadata.get(key, ""))
                    for key in (
                        "reference_coordinate",
                        "reference_target_coordinate",
                        "observed_validation_coordinate",
                    )
                ).lower()
                observed = (
                    "decoded_standardized"
                    if "standardized decoded-output" in phrases
                    else ""
                )
            else:
                observed = ""
            if not observed:
                raise ValueError(
                    f"{path.name}: metadata_json does not certify {metadata_key}"
                )
        else:
            # A .npy or a minimal NPZ can still be used, but its coordinate is
            # then an explicit CLI declaration recorded and hashed in run_config.
            return
    if observed != str(declared):
        raise ValueError(
            f"{path.name}: declared {metadata_key}={declared!r}, "
            f"archive records {observed!r}"
        )


def normalize_contexts(values: np.ndarray, n_channels: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3 or array.shape[2] != int(n_channels):
        raise ValueError("contexts must have shape [context,time,channel]")
    if array.shape[1] < 33:
        raise ValueError("each context must cover at least the 32-sample delay")
    if not np.isfinite(array).all():
        raise ValueError("contexts contain non-finite values")
    return array


def normalize_reference(
    values: np.ndarray, n_channels: int, horizon: int = HORIZON
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2:
        if array.shape[1] != int(n_channels) or len(array) % int(horizon):
            raise ValueError(
                "2-D reference must contain an integer number of 1-s paths"
            )
        array = array.reshape(-1, int(horizon), int(n_channels))
    if array.ndim != 3 or array.shape[1:] != (int(horizon), int(n_channels)):
        raise ValueError("reference must have shape [path,256,channel]")
    if len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("at least two finite reference paths are required")
    return array


def normalize_observed_validation(
    values: np.ndarray,
    n_contexts: int,
    n_channels: int,
    horizon: int = HORIZON,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2:
        array = array[None]
    if array.shape != (int(n_contexts), int(horizon), int(n_channels)):
        raise ValueError(
            "observed validation must have one [256,channel] future per context"
        )
    if not np.isfinite(array).all():
        raise ValueError("observed validation contains non-finite values")
    return array


def antithetic_noise(
    seed: int,
    particles: int,
    horizon: int,
    q: int,
    *,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    if int(particles) < 2 or int(particles) % 2:
        raise ValueError("an even particle count is required")
    rng = np.random.default_rng(int(seed))
    half = rng.standard_normal((int(particles) // 2, int(horizon), int(q)))
    return torch.as_tensor(np.concatenate([half, -half]), dtype=dtype)


def reference_statistics(reference: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if reference.ndim != 3:
        raise ValueError("reference must have shape [path,time,channel]")
    channels = int(reference.shape[2])
    pooled_variance = reference.reshape(-1, channels).var(dim=0, unbiased=False)
    scale = torch.sqrt(pooled_variance.clamp_min(0.10**2))
    inverse = 1.0 / pooled_variance.clamp_min(0.10**2)
    inverse = inverse / inverse.mean()
    weights = (0.50 + 0.50 * inverse).clamp(0.50, 3.0)
    weights = weights / weights.mean()
    mean = reference.mean(dim=0)
    variance = (
        0.25 * reference.var(dim=0, unbiased=False)
        + 0.75 * pooled_variance[None]
    ).clamp_min(1.0e-5)
    return mean, variance, scale, weights


def build_markov_normalization(
    adapter: FrozenGraphRCMarkovAdapter,
    initial_states: Sequence[Tensor],
    fit_reference: Tensor,
) -> tuple[Tensor, Tensor]:
    initial = torch.stack([value.detach() for value in initial_states])
    center = initial.mean(dim=0)
    scale = torch.ones_like(center)
    reservoir_scale = initial[:, adapter.slices.reservoir].std(
        dim=0, unbiased=False
    ).clamp_min(0.25)
    scale[adapter.slices.reservoir] = reservoir_scale
    latent_reference = (
        fit_reference - adapter.pca_mean[None, None]
    ) @ adapter.components.T
    latent_scale = latent_reference.reshape(-1, adapter.q).std(
        dim=0, unbiased=False
    ).clamp_min(0.10)
    scale[adapter.slices.history] = latent_scale.repeat(adapter.history_length)
    topology_reference = (
        fit_reference @ adapter.adjacency.T
    ) @ adapter.components.T
    topology_scale = topology_reference.reshape(-1, adapter.q).std(
        dim=0, unbiased=False
    ).clamp_min(0.10)
    scale[adapter.slices.topology] = topology_scale
    return center, scale


def analytical_weighted_ridge_gain(
    adapter: FrozenGraphRCMarkovAdapter,
    channel_weights: Tensor,
    reference_scale: Tensor,
    *,
    feedback_channel_map: Tensor | None = None,
    ridge: float,
    gain_scale: float,
) -> Tensor:
    if float(ridge) <= 0.0 or float(gain_scale) < 0.0:
        raise ValueError("ridge must be positive and gain scale non-negative")
    if feedback_channel_map is None:
        feedback_channel_map = adapter.control_channel_map
    if feedback_channel_map.shape != adapter.control_channel_map.shape:
        raise ValueError("feedback channel map has invalid shape")
    output_map = adapter.control_step_scale * feedback_channel_map
    physical_weight = channel_weights / reference_scale.square()
    weighted_map = output_map * physical_weight[None]
    gram = weighted_map @ output_map.T
    gain = torch.linalg.solve(
        gram + float(ridge) * torch.eye(
            adapter.actuator_dim, dtype=adapter.dtype, device=adapter.device
        ),
        weighted_map,
    )
    return float(gain_scale) * gain


def graph_aware_feedback_channel_map(
    adapter: FrozenGraphRCMarkovAdapter,
    graph_diffusion_time: float,
) -> Tensor:
    """Return a graph-aware *feedback* map without changing physical actuation.

    The heat kernel is used only to construct a feedback prior that anticipates
    later network propagation.  The plant input matrix itself remains the
    exact local electrode selector, so no unselected channel receives a
    same-sample physical input.
    """

    tau = float(graph_diffusion_time)
    if tau < 0.0:
        raise ValueError("graph diffusion time must be non-negative")
    adjacency = adapter.adjacency.detach().cpu().numpy().astype(np.float64)
    degree = adjacency.sum(axis=1)
    transition = adjacency / np.maximum(degree[:, None], 1.0e-12)
    random_walk_laplacian = np.eye(adapter.n_channels) - transition
    heat_kernel = expm(-tau * random_walk_laplacian)
    local_selector = adapter.control_channel_map.detach().cpu().numpy()
    return torch.as_tensor(
        local_selector @ heat_kernel,
        dtype=adapter.dtype,
        device=adapter.device,
    )


def build_control_stack(
    model: Any,
    network: Mapping[str, np.ndarray],
    contexts: np.ndarray,
    fit_reference: np.ndarray,
    config: ControllerConfig,
) -> tuple[
    TorchGraphRCSDE,
    FrozenGraphRCMarkovAdapter,
    FrozenIctalGraphRCBatchStepper,
    list[Tensor],
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    StructuredSplineCovarianceActor,
]:
    selected = np.flatnonzero(np.asarray(network["target_mask"], dtype=bool))
    world = TorchGraphRCSDE(
        model,
        selected,
        float(config.sampling_rate_hz),
        # Physical stimulation is strictly local.  Graph diffusion is a
        # dynamical consequence and a feedback feature, never dense same-step
        # feedthrough in the plant input matrix.
        control_graph_diffusion_time=0.0,
        preserve_physical_control_residual=True,
        dtype=torch.float64,
        device="cpu",
    )
    local_selector = np.zeros(
        (len(selected), int(world.n_channels)), dtype=np.float64
    )
    local_selector[np.arange(len(selected)), selected] = 1.0
    world.control_channel_map = torch.as_tensor(
        local_selector, dtype=world.dtype, device=world.device
    )
    world.control_map = (
        world.control_channel_map @ world.components.T
    ).contiguous()
    adapter = FrozenGraphRCMarkovAdapter(
        world,
        control_step_scale=float(config.control_step_scale),
        dtype=torch.float64,
        device="cpu",
    )
    stepper = build_calibrated_stepper(
        world,
        adapter,
        diffusion_scale=float(config.diffusion_scale),
        increment_scale=float(config.increment_scale),
        persistence_skip=float(config.persistence_skip),
    )
    initial_states = [adapter.initial_state_from_context(item) for item in contexts]
    reference = torch.as_tensor(fit_reference, dtype=torch.float64)
    reference_mean, reference_variance, reference_scale, channel_weights = (
        reference_statistics(reference)
    )
    markov_center, markov_scale = build_markov_normalization(
        adapter, initial_states, reference
    )
    base_gain = analytical_weighted_ridge_gain(
        adapter,
        channel_weights,
        reference_scale,
        feedback_channel_map=graph_aware_feedback_channel_map(
            adapter, float(config.graph_diffusion_time)
        ),
        ridge=float(config.base_gain_ridge),
        gain_scale=float(config.base_gain_scale),
    )
    actor = StructuredSplineCovarianceActor(
        stepper,
        reference_mean,
        reference_variance,
        reference_scale,
        base_gain,
        selected,
        horizon=int(config.horizon),
        basis_count=int(config.basis_count),
        hidden_size=int(config.decoded_hidden_size),
        amplitude_limit=float(config.amplitude_limit),
        actuator_alpha=float(config.actuator_alpha),
        residual_scale=float(config.residual_scale),
        local_gain_initial_fraction=float(config.local_gain_initial_fraction),
        local_gain_maximum_fraction=float(config.local_gain_maximum_fraction),
        markov_feature_center=markov_center,
        markov_feature_scale=markov_scale,
        markov_residual_scale=float(config.markov_residual_scale),
        markov_hidden_size=int(config.markov_hidden_size),
        maximum_slew=None,
    )
    return (
        world,
        adapter,
        stepper,
        initial_states,
        reference_mean,
        reference_variance,
        reference_scale,
        channel_weights,
        actor,
    )


def episode_energy_budget(config: ControllerConfig, actuators: int) -> float:
    if config.total_episode_energy_budget is not None:
        if float(config.total_episode_energy_budget) <= 0.0:
            raise ValueError("fixed total episode-energy budget must be positive")
        return float(config.total_episode_energy_budget)
    duration = float(config.horizon) / float(config.sampling_rate_hz)
    return duration * int(actuators) * float(config.energy_rms) ** 2


class _FiniteGradientSqrt(torch.autograd.Function):
    """Exact square-root forward with a finite right derivative at zero."""

    @staticmethod
    def forward(ctx: Any, value: Tensor, epsilon: float) -> Tensor:
        ctx.save_for_backward(value)
        ctx.epsilon = float(epsilon)
        return torch.sqrt(value)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        (value,) = ctx.saved_tensors
        denominator = 2.0 * torch.sqrt(value.clamp_min(ctx.epsilon))
        return grad_output / denominator, None


def causal_energy_projection_scale(
    raw_step_energy: Tensor,
    remaining_energy: Tensor,
    *,
    epsilon: float = 1.0e-12,
) -> Tensor:
    """Return a differentiable causal scale for the episode-energy cap.

    The exact cap can be exhausted before the final sample.  Differentiating
    ``sqrt(remaining_energy)`` at zero has an infinite derivative and can turn
    otherwise finite Actor gradients into NaNs.  The custom autograd operation
    preserves the exact square-root forward value (including exact zero) and
    floors only the backward denominator.  Thus the locked analytical forward
    calculation and hard episode-energy cap remain unchanged.
    """

    if float(epsilon) <= 0.0:
        raise ValueError("energy-projection epsilon must be positive")
    safe_raw = raw_step_energy.clamp_min(float(epsilon))
    ratio = remaining_energy.clamp_min(0.0) / safe_raw
    return torch.minimum(
        raw_step_energy.new_ones(()),
        _FiniteGradientSqrt.apply(ratio, float(epsilon)),
    )


def network_with_target_quantile(
    network: Mapping[str, np.ndarray], quantile: float
) -> dict[str, np.ndarray]:
    q = float(quantile)
    if q not in (0.50, 0.65, 0.80):
        raise ValueError("target quantile must be one of 0.50, 0.65, or 0.80")
    answer = {key: np.asarray(value).copy() for key, value in network.items()}
    score = np.asarray(answer["centrality_score"], dtype=np.float64)
    threshold = float(np.quantile(score, q, method="linear"))
    mask = score > threshold
    if not mask.any():
        raise ValueError("candidate target quantile produced an empty actuator mask")
    answer["target_mask"] = mask
    answer["candidate_target_quantile"] = np.asarray(q, dtype=np.float64)
    answer["candidate_centrality_threshold"] = np.asarray(
        threshold, dtype=np.float64
    )
    return answer


def budgeted_particle_rollout(
    stepper: FrozenIctalGraphRCBatchStepper,
    actor: StructuredSplineCovarianceActor,
    initial_markov: Tensor,
    standard_normal: Tensor,
    *,
    energy_budget: float,
) -> BudgetedParticleRollout:
    """Empirical FP rollout with a causal differentiable episode-energy cap."""

    noise = torch.as_tensor(
        standard_normal, dtype=stepper.dtype, device=stepper.device
    )
    if noise.ndim != 3 or noise.shape[2] != stepper.q:
        raise ValueError("standard_normal must have shape [particle,time,q]")
    particles, horizon, _ = noise.shape
    if int(horizon) != int(actor.horizon):
        raise ValueError("noise and actor horizons differ")
    if float(energy_budget) <= 0.0:
        raise ValueError("energy_budget must be positive")
    state = stepper.repeat_initial(initial_markov, int(particles))
    previous = state.new_zeros(int(particles), stepper.actuator_dim)
    current = stepper.current_output(state)
    remaining = state.new_tensor(float(energy_budget))
    states = [state]
    outputs = [current]
    controls: list[Tensor] = []
    commands: list[Tensor] = []
    conditional_std: list[Tensor] = []
    delivered = state.new_zeros(())
    for step in range(int(horizon)):
        action = actor.propose(
            step=step,
            markov_particles=state,
            previous_control=previous,
            current_scaled=current,
        )
        raw = action.applied_control
        # Energy is dt*sum(u^2); the paper-exact controller runs at 256 Hz.
        raw_step_energy = raw.square().sum(dim=1).mean() / SAMPLING_RATE_HZ
        scale = causal_energy_projection_scale(
            raw_step_energy,
            remaining,
        )
        applied = raw * scale
        command = action.command * scale
        step_energy = applied.square().sum(dim=1).mean() / SAMPLING_RATE_HZ
        remaining = (remaining - step_energy).clamp_min(0.0)
        delivered = delivered + step_energy
        transition = stepper.step(state, applied, noise[:, step])
        state = transition.next_markov
        current = transition.next_scaled
        previous = applied
        states.append(state)
        outputs.append(current)
        controls.append(applied)
        commands.append(command)
        conditional_std.append(transition.conditional_std)
    return BudgetedParticleRollout(
        markov=torch.stack(states, dim=1),
        scaled=torch.stack(outputs, dim=1),
        controls=torch.stack(controls, dim=1),
        commands=torch.stack(commands, dim=1),
        conditional_std=torch.stack(conditional_std, dim=1),
        noise=noise,
        delivered_energy=delivered,
        energy_budget=delivered.new_tensor(float(energy_budget)),
    )


def uncontrolled_particle_rollout(
    stepper: FrozenIctalGraphRCBatchStepper,
    initial_markov: Tensor,
    standard_normal: Tensor,
) -> Tensor:
    noise = torch.as_tensor(
        standard_normal, dtype=stepper.dtype, device=stepper.device
    )
    particles, horizon, q = noise.shape
    if q != stepper.q:
        raise ValueError("noise latent dimension differs from the plant")
    state = stepper.repeat_initial(initial_markov, int(particles))
    zero = state.new_zeros(int(particles), stepper.actuator_dim)
    outputs = [stepper.current_output(state)]
    for step in range(int(horizon)):
        transition = stepper.step(state, zero, noise[:, step])
        state = transition.next_markov
        outputs.append(transition.next_scaled)
    return torch.stack(outputs, dim=1)


def _per_channel_components(
    sequence: Tensor,
    reference: Tensor,
    reference_scale: Tensor,
) -> dict[str, Tensor]:
    particles, horizon, channels = sequence.shape
    del particles
    reference_mean = reference.mean(dim=0)
    predicted_mean = sequence.mean(dim=0)
    pooled_reference_variance = reference.reshape(-1, channels).var(
        dim=0, unbiased=False
    ).clamp_min(1.0e-5)
    reference_variance = (
        0.25 * reference.var(dim=0, unbiased=False)
        + 0.75 * pooled_reference_variance[None]
    ).clamp_min(1.0e-5)
    predicted_variance = sequence.var(dim=0, unbiased=False).clamp_min(1.0e-5)
    mean = (((predicted_mean - reference_mean) / reference_scale[None]) ** 2).mean(0)
    log_sd = (
        0.25
        * (torch.log(predicted_variance) - torch.log(reference_variance)).square()
    ).mean(0)
    time_indices = torch.arange(0, horizon, 4, device=sequence.device)
    probabilities = torch.linspace(
        0.10, 0.90, 9, dtype=sequence.dtype, device=sequence.device
    )
    predicted_quantiles = torch.quantile(
        sequence[:, time_indices], probabilities, dim=0
    )
    reference_quantiles = torch.quantile(
        reference[:, time_indices], probabilities, dim=0
    )
    time_quantile = (
        (predicted_quantiles - reference_quantiles)
        / reference_scale[None, None]
    ).square().mean(dim=(0, 1))
    occupancy_probabilities = torch.linspace(
        0.05, 0.95, 19, dtype=sequence.dtype, device=sequence.device
    )
    predicted_occupancy = torch.quantile(
        sequence.reshape(-1, channels), occupancy_probabilities, dim=0
    )
    reference_occupancy = torch.quantile(
        reference.reshape(-1, channels), occupancy_probabilities, dim=0
    )
    occupation = (
        (predicted_occupancy - reference_occupancy) / reference_scale[None]
    ).square().mean(dim=0)
    return {
        "mean": mean,
        "log_sd": log_sd,
        "time_quantile": time_quantile,
        "occupation": occupation,
    }


def distribution_control_objective(
    rollout: BudgetedParticleRollout,
    reference: Tensor,
    free_sequence: Tensor,
    channel_weights: Tensor,
    reference_scale: Tensor,
    joint_projections: Tensor,
    actor: StructuredSplineCovarianceActor,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Channel-resolved 1-s law loss used by the structured mean-field Actor."""

    sequence = rollout.scaled[:, 1:]
    if sequence.shape[1:] != reference.shape[1:]:
        raise ValueError("controlled and reference path dimensions differ")
    controlled = _per_channel_components(sequence, reference, reference_scale)
    with torch.no_grad():
        baseline = _per_channel_components(free_sequence, reference, reference_scale)
    weighted = {
        name: (value * channel_weights).mean()
        for name, value in controlled.items()
    }
    channel_identity = torch.stack(
        [controlled[name] for name in ("mean", "log_sd", "time_quantile", "occupation")]
    ).sum(dim=0)
    baseline_identity = torch.stack(
        [baseline[name] for name in ("mean", "log_sd", "time_quantile", "occupation")]
    ).sum(dim=0)
    harm = torch.relu(channel_identity - baseline_identity)
    worst_count = max(1, int(np.ceil(0.10 * sequence.shape[2])))
    no_harm = (harm * channel_weights).mean() + torch.topk(
        harm * channel_weights, worst_count
    ).values.mean()
    identity_tail = torch.topk(
        channel_identity * channel_weights, worst_count
    ).values.mean()

    reference_mean = reference.mean(dim=0)
    joint_indices = range(0, int(sequence.shape[1]), 16)
    joint_probabilities = torch.linspace(
        0.10, 0.90, 9, dtype=sequence.dtype, device=sequence.device
    )
    joint_terms: list[Tensor] = []
    scale_weight = torch.sqrt(channel_weights)[None]
    for step in joint_indices:
        predicted_joint = (
            (sequence[:, step] - reference_mean[step][None])
            / reference_scale[None]
        ) * scale_weight
        reference_joint = (
            (reference[:, step] - reference_mean[step][None])
            / reference_scale[None]
        ) * scale_weight
        predicted_projection = predicted_joint @ joint_projections
        reference_projection = reference_joint @ joint_projections
        joint_terms.append(
            (
                torch.quantile(predicted_projection, joint_probabilities, dim=0)
                - torch.quantile(reference_projection, joint_probabilities, dim=0)
            ).square().mean()
        )
    joint = torch.stack(joint_terms).mean()

    energy_ratio = rollout.delivered_energy / rollout.energy_budget.clamp_min(1.0e-12)
    first = (rollout.controls[:, 1:] - rollout.controls[:, :-1]) / actor.amplitude_limit
    second = (
        rollout.controls[:, 2:]
        - 2.0 * rollout.controls[:, 1:-1]
        + rollout.controls[:, :-2]
    ) / actor.amplitude_limit
    smoothness = first.square().mean()
    curvature = second.square().mean()
    saturation = torch.relu(
        rollout.commands.abs() / actor.amplitude_limit - 0.95
    ).square().mean()
    energy_excess = torch.relu(energy_ratio - 1.0).square()
    total = (
        5.0 * weighted["mean"]
        + 5.0 * weighted["log_sd"]
        + 6.0 * weighted["time_quantile"]
        + 4.0 * weighted["occupation"]
        + 2.0 * identity_tail
        + 20.0 * no_harm
        + 0.50 * joint
        + 0.02 * energy_ratio.square()
        + 100.0 * energy_excess
        + 0.20 * smoothness
        + 0.10 * curvature
        + 1.00 * saturation
    )
    return total, {
        "loss": total,
        "time_mean": weighted["mean"],
        "time_log_sd": weighted["log_sd"],
        "time_quantile": weighted["time_quantile"],
        "full_occupation": weighted["occupation"],
        "channel_identity_tail": identity_tail,
        "no_harm": no_harm,
        "joint_projection": joint,
        "energy_ratio": energy_ratio,
        "energy_excess": energy_excess,
        "smoothness": smoothness,
        "curvature": curvature,
        "saturation": saturation,
    }


@lru_cache(maxsize=32)
def _uniform_w1_plan(first_count: int, second_count: int) -> tuple[np.ndarray, ...]:
    """Exact inverse-CDF integration plan for two uniform empirical laws."""

    first, second = int(first_count), int(second_count)
    if first < 1 or second < 1:
        raise ValueError("empirical distributions must be nonempty")
    lattice = int(np.lcm(first, second))
    first_ticks = np.arange(0, lattice + 1, lattice // first, dtype=np.int64)
    second_ticks = np.arange(0, lattice + 1, lattice // second, dtype=np.int64)
    boundaries = np.union1d(first_ticks, second_ticks)
    left, right = boundaries[:-1], boundaries[1:]
    midpoint_twice = left + right
    first_index = np.minimum(
        (midpoint_twice * first) // (2 * lattice), first - 1
    ).astype(np.int64)
    second_index = np.minimum(
        (midpoint_twice * second) // (2 * lattice), second - 1
    ).astype(np.int64)
    weights = (right - left).astype(np.float64) / float(lattice)
    for array in (first_index, second_index, weights):
        array.setflags(write=False)
    return first_index, second_index, weights


def batched_uniform_wasserstein(
    first: np.ndarray, second: np.ndarray
) -> np.ndarray:
    """Uniform empirical W1 along axis 0, vectorized over trailing axes."""

    lhs = np.asarray(first, dtype=np.float64)
    rhs = np.asarray(second, dtype=np.float64)
    if lhs.ndim < 1 or rhs.ndim != lhs.ndim or lhs.shape[1:] != rhs.shape[1:]:
        raise ValueError("sample arrays must share every non-sample dimension")
    if not np.isfinite(lhs).all() or not np.isfinite(rhs).all():
        raise ValueError("Wasserstein inputs must be finite")
    lhs_sorted = np.sort(lhs, axis=0)
    rhs_sorted = np.sort(rhs, axis=0)
    lhs_index, rhs_index, weights = _uniform_w1_plan(len(lhs), len(rhs))
    difference = np.abs(lhs_sorted[lhs_index] - rhs_sorted[rhs_index])
    weight_shape = (len(weights),) + (1,) * (difference.ndim - 1)
    return np.sum(difference * weights.reshape(weight_shape), axis=0)


def channelwise_time_w1(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if predicted.ndim != 3 or reference.ndim != 3:
        raise ValueError("time-resolved laws must have shape [sample,time,channel]")
    return batched_uniform_wasserstein(predicted, reference).mean(axis=0)


def channelwise_occupation_w1(
    predicted: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    if predicted.ndim != 3 or reference.ndim != 3:
        raise ValueError("occupation laws must have shape [sample,time,channel]")
    channels = predicted.shape[2]
    return batched_uniform_wasserstein(
        predicted.reshape(-1, channels), reference.reshape(-1, channels)
    )


def control_metrics(
    controlled: np.ndarray,
    free: np.ndarray,
    reference: np.ndarray,
    controls: np.ndarray,
    *,
    energy_budget: float,
    amplitude_limit: float,
) -> dict[str, float]:
    controlled_time = channelwise_time_w1(controlled, reference)
    free_time = channelwise_time_w1(free, reference)
    controlled_occupation = channelwise_occupation_w1(controlled, reference)
    free_occupation = channelwise_occupation_w1(free, reference)
    controlled_mean = controlled.mean(axis=(0, 1))
    free_mean = free.mean(axis=(0, 1))
    reference_mean = reference.mean(axis=(0, 1))
    controlled_sd = controlled.std(axis=(0, 1))
    free_sd = free.std(axis=(0, 1))
    reference_sd = reference.std(axis=(0, 1))
    delivered_energy = float(
        np.mean(np.sum(np.square(controls), axis=(1, 2))) / SAMPLING_RATE_HZ
    )
    both = (controlled_time < free_time) & (
        controlled_occupation < free_occupation
    )
    return {
        "mean_time_w1_free_reference": float(free_time.mean()),
        "mean_time_w1_controlled_reference": float(controlled_time.mean()),
        "mean_occupation_w1_free_reference": float(free_occupation.mean()),
        "mean_occupation_w1_controlled_reference": float(
            controlled_occupation.mean()
        ),
        "q90_occupation_w1_free_reference": float(np.quantile(free_occupation, 0.90)),
        "q90_occupation_w1_controlled_reference": float(
            np.quantile(controlled_occupation, 0.90)
        ),
        "fraction_channels_improved_both": float(both.mean()),
        "mean_absolute_mean_error": float(
            np.mean(np.abs(controlled_mean - reference_mean))
        ),
        "mean_absolute_mean_error_free": float(
            np.mean(np.abs(free_mean - reference_mean))
        ),
        "mean_absolute_sd_error": float(np.mean(np.abs(controlled_sd - reference_sd))),
        "mean_absolute_sd_error_free": float(
            np.mean(np.abs(free_sd - reference_sd))
        ),
        "delivered_energy": delivered_energy,
        "energy_budget": float(energy_budget),
        "energy_ratio": delivered_energy / float(energy_budget),
        "control_rms_per_actuator": float(np.sqrt(np.mean(np.square(controls)))),
        "control_peak": float(np.max(np.abs(controls))),
        "saturation_fraction": float(
            np.mean(np.abs(controls) >= 0.99 * float(amplitude_limit))
        ),
    }


def plant_fidelity_metrics(
    free: np.ndarray,
    observed: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    """Compare the uncontrolled plant with an early-development ictal future."""

    if observed.ndim != 2 or observed.shape != free.shape[1:]:
        raise ValueError("observed future must have shape [time,channel]")
    observed_law = observed[None]
    free_observed_time = channelwise_time_w1(free, observed_law)
    free_reference_time = channelwise_time_w1(free, reference)
    observed_reference_time = channelwise_time_w1(observed_law, reference)
    free_observed_occupation = channelwise_occupation_w1(free, observed_law)
    free_reference_occupation = channelwise_occupation_w1(free, reference)
    observed_reference_occupation = channelwise_occupation_w1(
        observed_law, reference
    )
    free_sd = free.std(axis=(0, 1))
    observed_sd = observed.std(axis=0)
    return {
        "mean_time_w1_free_observed": float(free_observed_time.mean()),
        "mean_time_w1_free_reference_plant": float(free_reference_time.mean()),
        "mean_time_w1_observed_reference": float(
            observed_reference_time.mean()
        ),
        "mean_occupation_w1_free_observed": float(
            free_observed_occupation.mean()
        ),
        "mean_occupation_w1_free_reference_plant": float(
            free_reference_occupation.mean()
        ),
        "mean_occupation_w1_observed_reference": float(
            observed_reference_occupation.mean()
        ),
        "mean_absolute_sd_error_free_observed": float(
            np.mean(np.abs(free_sd - observed_sd))
        ),
    }


def fixed_joint_projections(channels: int, seed: int) -> Tensor:
    rng = np.random.default_rng(int(seed))
    values = rng.normal(size=(int(channels), min(16, int(channels))))
    values /= np.maximum(np.linalg.norm(values, axis=0, keepdims=True), 1.0e-12)
    return torch.as_tensor(values, dtype=torch.float64)


def validate_actor(
    actor: StructuredSplineCovarianceActor,
    stepper: FrozenIctalGraphRCBatchStepper,
    initial_states: Sequence[Tensor],
    reference: Tensor,
    channel_weights: Tensor,
    reference_scale: Tensor,
    joint_projections: Tensor,
    config: ControllerConfig,
    *,
    observed: np.ndarray | None = None,
    external_plant_fidelity_gate: bool | None = None,
    seed_offset: int = 0,
) -> ValidationResult:
    actor.eval()
    reference_np = reference.detach().cpu().numpy()
    budget = episode_energy_budget(config, stepper.actuator_dim)
    rows: list[dict[str, float]] = []
    display: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for context_index, initial in enumerate(initial_states):
            noise = antithetic_noise(
                config.seed + 700000 + int(seed_offset) + context_index,
                config.particles,
                config.horizon,
                stepper.q,
            )
            free = uncontrolled_particle_rollout(stepper, initial, noise)[:, 1:]
            controlled_rollout = budgeted_particle_rollout(
                stepper, actor, initial, noise, energy_budget=budget
            )
            controlled = controlled_rollout.scaled[:, 1:]
            objective, terms = distribution_control_objective(
                controlled_rollout,
                reference,
                free,
                channel_weights,
                reference_scale,
                joint_projections,
                actor,
            )
            free_np = free.cpu().numpy()
            controlled_np = controlled.cpu().numpy()
            controls_np = controlled_rollout.controls.cpu().numpy()
            metric = control_metrics(
                controlled_np,
                free_np,
                reference_np,
                controls_np,
                energy_budget=budget,
                amplitude_limit=config.amplitude_limit,
            )
            metric.update(
                {
                    "context_index": float(context_index),
                    "objective": float(objective),
                    **{
                        f"loss_{name}": float(value)
                        for name, value in terms.items()
                    },
                }
            )
            if observed is not None:
                metric.update(
                    plant_fidelity_metrics(
                        free_np,
                        np.asarray(observed[context_index], dtype=np.float64),
                        reference_np,
                    )
                )
            rows.append(metric)
            if context_index == 0:
                display = {
                    "free": free_np,
                    "controlled": controlled_np,
                    "reference": reference_np,
                    "controls": controls_np,
                    "noise": noise.cpu().numpy(),
                }
    numeric_keys = [
        key for key in rows[0] if key != "context_index"
    ]
    aggregate = {
        key: float(np.mean([row[key] for row in rows])) for key in numeric_keys
    }
    aggregate["contexts"] = float(len(rows))
    aggregate["controller_feasible"] = float(
        aggregate["energy_ratio"] <= 1.0 + 1.0e-9
        and aggregate["control_peak"] <= config.amplitude_limit + 1.0e-9
        and aggregate["saturation_fraction"] < 0.01
        and aggregate["mean_time_w1_controlled_reference"]
        < aggregate["mean_time_w1_free_reference"]
        and aggregate["mean_occupation_w1_controlled_reference"]
        < aggregate["mean_occupation_w1_free_reference"]
        and aggregate["fraction_channels_improved_both"] >= 0.75
        and aggregate["q90_occupation_w1_controlled_reference"]
        <= 1.10 * aggregate["q90_occupation_w1_free_reference"]
    )
    if observed is None:
        aggregate["plant_fidelity_gate"] = (
            float("nan")
            if external_plant_fidelity_gate is None
            else float(bool(external_plant_fidelity_gate))
        )
        aggregate["plant_fidelity_gate_from_frozen_oof"] = float(
            external_plant_fidelity_gate is not None
        )
    else:
        aggregate["plant_fidelity_gate"] = float(
            aggregate["mean_time_w1_free_observed"]
            < aggregate["mean_time_w1_free_reference_plant"]
            and aggregate["mean_time_w1_free_observed"]
            < aggregate["mean_time_w1_observed_reference"]
            and aggregate["mean_occupation_w1_free_observed"]
            < aggregate["mean_occupation_w1_free_reference_plant"]
            and aggregate["mean_occupation_w1_free_observed"]
            < aggregate["mean_occupation_w1_observed_reference"]
        )
        aggregate["plant_fidelity_gate_from_frozen_oof"] = 0.0
    return ValidationResult(aggregate=aggregate, contexts=rows, display=display)


def validation_cache_key(
    stepper: FrozenIctalGraphRCBatchStepper,
    initial_states: Sequence[Tensor],
    reference: Tensor,
    config: ControllerConfig,
    *,
    seed_offset: int,
) -> dict[str, Any]:
    """Bind a validation cache to every Actor-independent frozen input."""

    initial_hashes = tuple(_tensor_sha256(item) for item in initial_states)
    plant_hash = _validation_plant_sha256(stepper)
    config_hash = _json_sha256(config)
    reference_hash = _tensor_sha256(reference)
    model_shapes = (
        int(stepper.q),
        int(stepper.n_channels),
        int(stepper.state_dim),
        int(stepper.actuator_dim),
        int(stepper.adapter.reservoir_size),
        int(stepper.variance_coef.shape[1]),
    )
    payload = {
        "schema_version": "validation-free-cache-v1.0",
        "seed_offset": int(seed_offset),
        "plant_sha256": plant_hash,
        "config_sha256": config_hash,
        "initial_state_sha256": list(initial_hashes),
        "reference_sha256": reference_hash,
        "reference_shape": list(reference.shape),
        "model_shape_contract": list(model_shapes),
    }
    return {**payload, "cache_key_sha256": _json_sha256(payload)}


def build_validation_invariant_cache(
    stepper: FrozenIctalGraphRCBatchStepper,
    initial_states: Sequence[Tensor],
    reference: Tensor,
    config: ControllerConfig,
    *,
    seed_offset: int,
) -> ValidationInvariantCache:
    """Compute fixed noise banks and uncontrolled validation paths once."""

    key = validation_cache_key(
        stepper, initial_states, reference, config, seed_offset=seed_offset
    )
    noises: list[Tensor] = []
    free_sequences: list[Tensor] = []
    with torch.no_grad():
        for context_index, initial in enumerate(initial_states):
            noise = antithetic_noise(
                config.seed + 700000 + int(seed_offset) + context_index,
                config.particles,
                config.horizon,
                stepper.q,
            )
            free = uncontrolled_particle_rollout(stepper, initial, noise)[:, 1:]
            noises.append(noise.detach().clone())
            free_sequences.append(free.detach().clone())
    return ValidationInvariantCache(
        noise=tuple(noises),
        free=tuple(free_sequences),
        noise_sha256=tuple(_tensor_sha256(value) for value in noises),
        free_sha256=tuple(_tensor_sha256(value) for value in free_sequences),
        cache_key_sha256=str(key["cache_key_sha256"]),
        plant_sha256=str(key["plant_sha256"]),
        config_sha256=str(key["config_sha256"]),
        initial_state_sha256=tuple(key["initial_state_sha256"]),
        reference_sha256=str(key["reference_sha256"]),
        reference_shape=tuple(int(value) for value in key["reference_shape"]),
        model_shape_contract=tuple(
            int(value) for value in key["model_shape_contract"]
        ),
        seed_offset=int(seed_offset),
        particles=int(config.particles),
        horizon=int(config.horizon),
        latent_dimension=int(stepper.q),
    )


def validate_actor_cached(
    actor: StructuredSplineCovarianceActor,
    stepper: FrozenIctalGraphRCBatchStepper,
    initial_states: Sequence[Tensor],
    reference: Tensor,
    channel_weights: Tensor,
    reference_scale: Tensor,
    joint_projections: Tensor,
    config: ControllerConfig,
    cache: ValidationInvariantCache,
    *,
    observed: np.ndarray | None = None,
    external_plant_fidelity_gate: bool | None = None,
    seed_offset: int = 0,
) -> ValidationResult:
    """Validate with cached invariants while recomputing every controlled path."""

    expected_key = validation_cache_key(
        stepper, initial_states, reference, config, seed_offset=seed_offset
    )
    if (
        str(expected_key["cache_key_sha256"]) != cache.cache_key_sha256
        or len(cache.noise) != len(initial_states)
        or len(cache.free) != len(initial_states)
    ):
        raise ValueError("validation cache does not match this frozen evaluation")
    if tuple(_tensor_sha256(value) for value in cache.noise) != cache.noise_sha256:
        raise PermissionError("cached innovation bank was mutated")
    if tuple(_tensor_sha256(value) for value in cache.free) != cache.free_sha256:
        raise PermissionError("cached free rollout was mutated")

    actor.eval()
    reference_np = reference.detach().cpu().numpy()
    budget = episode_energy_budget(config, stepper.actuator_dim)
    rows: list[dict[str, float]] = []
    display: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for context_index, initial in enumerate(initial_states):
            noise = cache.noise[context_index]
            free = cache.free[context_index]
            controlled_rollout = budgeted_particle_rollout(
                stepper, actor, initial, noise, energy_budget=budget
            )
            controlled = controlled_rollout.scaled[:, 1:]
            objective, terms = distribution_control_objective(
                controlled_rollout,
                reference,
                free,
                channel_weights,
                reference_scale,
                joint_projections,
                actor,
            )
            free_np = free.cpu().numpy()
            controlled_np = controlled.cpu().numpy()
            controls_np = controlled_rollout.controls.cpu().numpy()
            metric = control_metrics(
                controlled_np,
                free_np,
                reference_np,
                controls_np,
                energy_budget=budget,
                amplitude_limit=config.amplitude_limit,
            )
            metric.update(
                {
                    "context_index": float(context_index),
                    "objective": float(objective),
                    **{
                        f"loss_{name}": float(value)
                        for name, value in terms.items()
                    },
                }
            )
            if observed is not None:
                metric.update(
                    plant_fidelity_metrics(
                        free_np,
                        np.asarray(observed[context_index], dtype=np.float64),
                        reference_np,
                    )
                )
            rows.append(metric)
            if context_index == 0:
                display = {
                    "free": free_np,
                    "controlled": controlled_np,
                    "reference": reference_np,
                    "controls": controls_np,
                    "noise": noise.cpu().numpy(),
                }

    numeric_keys = [key for key in rows[0] if key != "context_index"]
    aggregate = {
        key: float(np.mean([row[key] for row in rows])) for key in numeric_keys
    }
    aggregate["contexts"] = float(len(rows))
    aggregate["controller_feasible"] = float(
        aggregate["energy_ratio"] <= 1.0 + 1.0e-9
        and aggregate["control_peak"] <= config.amplitude_limit + 1.0e-9
        and aggregate["saturation_fraction"] < 0.01
        and aggregate["mean_time_w1_controlled_reference"]
        < aggregate["mean_time_w1_free_reference"]
        and aggregate["mean_occupation_w1_controlled_reference"]
        < aggregate["mean_occupation_w1_free_reference"]
        and aggregate["fraction_channels_improved_both"] >= 0.75
        and aggregate["q90_occupation_w1_controlled_reference"]
        <= 1.10 * aggregate["q90_occupation_w1_free_reference"]
    )
    if observed is None:
        aggregate["plant_fidelity_gate"] = (
            float("nan")
            if external_plant_fidelity_gate is None
            else float(bool(external_plant_fidelity_gate))
        )
        aggregate["plant_fidelity_gate_from_frozen_oof"] = float(
            external_plant_fidelity_gate is not None
        )
    else:
        aggregate["plant_fidelity_gate"] = float(
            aggregate["mean_time_w1_free_observed"]
            < aggregate["mean_time_w1_free_reference_plant"]
            and aggregate["mean_time_w1_free_observed"]
            < aggregate["mean_time_w1_observed_reference"]
            and aggregate["mean_occupation_w1_free_observed"]
            < aggregate["mean_occupation_w1_free_reference_plant"]
            and aggregate["mean_occupation_w1_free_observed"]
            < aggregate["mean_occupation_w1_observed_reference"]
        )
        aggregate["plant_fidelity_gate_from_frozen_oof"] = 0.0
    return ValidationResult(aggregate=aggregate, contexts=rows, display=display)


def selection_score(
    current: Mapping[str, float], baseline: Mapping[str, float]
) -> float:
    epsilon = 1.0e-12
    return float(
        0.50 * current["objective"] / max(baseline["objective"], epsilon)
        + 0.25
        * current["mean_time_w1_controlled_reference"]
        / max(baseline["mean_time_w1_controlled_reference"], epsilon)
        + 0.25
        * current["mean_occupation_w1_controlled_reference"]
        / max(baseline["mean_occupation_w1_controlled_reference"], epsilon)
    )


def checkpoint_eligible(
    current: Mapping[str, float],
    baseline: Mapping[str, float],
    score: float,
) -> bool:
    """Apply all predeclared development gates before accepting an update."""

    objective = float(current.get("objective", float("nan")))
    baseline_objective = float(baseline.get("objective", float("nan")))
    selection = float(score)
    return bool(
        float(current.get("controller_feasible", 0.0)) == 1.0
        and float(current.get("plant_fidelity_gate", 0.0)) == 1.0
        and np.isfinite(objective)
        and np.isfinite(baseline_objective)
        and objective <= 1.02 * baseline_objective
        and np.isfinite(selection)
        and selection < 1.0
    )


def analytical_screen_score(metrics: Mapping[str, float]) -> float:
    """Matched-budget candidate score normalized to its paired free law."""

    epsilon = 1.0e-12
    return float(
        0.40
        * metrics["mean_time_w1_controlled_reference"]
        / max(metrics["mean_time_w1_free_reference"], epsilon)
        + 0.40
        * metrics["mean_occupation_w1_controlled_reference"]
        / max(metrics["mean_occupation_w1_free_reference"], epsilon)
        + 0.075
        * metrics["mean_absolute_mean_error"]
        / max(metrics["mean_absolute_mean_error_free"], epsilon)
        + 0.075
        * metrics["mean_absolute_sd_error"]
        / max(metrics["mean_absolute_sd_error_free"], epsilon)
        + 0.05 * metrics["energy_ratio"]
    )


def parse_float_grid(value: str, *, name: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in str(value).split(","))
    except ValueError as error:
        raise ValueError(f"{name} must be a comma-separated numeric grid") from error
    if not parsed or any(not np.isfinite(item) for item in parsed):
        raise ValueError(f"{name} is empty or non-finite")
    return parsed


def evaluate_uncontrolled_plant(
    stepper: FrozenIctalGraphRCBatchStepper,
    validation_contexts: np.ndarray,
    observed_validation: np.ndarray,
    validation_reference: np.ndarray,
    config: ControllerConfig,
) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for index, context in enumerate(validation_contexts):
            initial = stepper.adapter.initial_state_from_context(context)
            noise = antithetic_noise(
                config.seed + 700000 + index,
                config.particles,
                config.horizon,
                stepper.q,
            )
            free = uncontrolled_particle_rollout(stepper, initial, noise)[:, 1:]
            rows.append(
                plant_fidelity_metrics(
                    free.cpu().numpy(), observed_validation[index], validation_reference
                )
            )
    aggregate = {
        key: float(np.mean([row[key] for row in rows])) for key in rows[0]
    }
    aggregate["plant_fidelity_gate"] = float(
        aggregate["mean_time_w1_free_observed"]
        < aggregate["mean_time_w1_free_reference_plant"]
        and aggregate["mean_time_w1_free_observed"]
        < aggregate["mean_time_w1_observed_reference"]
        and aggregate["mean_occupation_w1_free_observed"]
        < aggregate["mean_occupation_w1_free_reference_plant"]
        and aggregate["mean_occupation_w1_free_observed"]
        < aggregate["mean_occupation_w1_observed_reference"]
    )
    return aggregate


def _take_with_one_percent_tie_break(
    candidates: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    remaining = list(candidates)
    selected: list[dict[str, Any]] = []
    while remaining and len(selected) < int(count):
        best_score = min(float(row["analytical_screen_score"]) for row in remaining)
        near = [
            row
            for row in remaining
            if float(row["analytical_screen_score"]) <= 1.01 * best_score
        ]
        choice = min(
            near,
            key=lambda row: (
                float(row["delivered_energy"]),
                -float(row["target_quantile"]),
                float(row["base_gain_scale"]),
                float(row["graph_diffusion_time"]),
                str(row["candidate_id"]),
            ),
        )
        selected.append(choice)
        remaining.remove(choice)
    return selected


def run_analytical_grid(
    model: Any,
    network: Mapping[str, np.ndarray],
    fit_contexts: np.ndarray,
    validation_contexts: np.ndarray,
    fit_reference: np.ndarray,
    validation_reference: np.ndarray,
    base_config: ControllerConfig,
    *,
    frozen_oof_selection: Mapping[str, Any],
    target_quantile_grid: Sequence[float],
    gain_grid: Sequence[float],
    tau_grid: Sequence[float],
) -> tuple[
    ControllerConfig,
    ValidationResult,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
]:
    """Use frozen OOF plant calibration, then screen matched-energy controls."""

    primary_network = network_with_target_quantile(network, 0.65)
    primary_k = int(np.asarray(primary_network["target_mask"]).sum())
    primary_budget = (
        float(base_config.horizon)
        / float(base_config.sampling_rate_hz)
        * primary_k
        * float(base_config.energy_rms) ** 2
    )
    selected_diffusion = float(frozen_oof_selection["selected_diffusion_scale"])
    selected_increment = float(frozen_oof_selection["selected_increment_scale"])
    selected_persistence = float(frozen_oof_selection["selected_persistence_skip"])
    plant_failure = not bool(frozen_oof_selection["plant_pass"])
    if plant_failure:
        raise PermissionError("frozen OOF plant gate failed")
    selected_metrics = {
        f"oof_{key}": value
        for key, value in dict(frozen_oof_selection["selected_metrics"]).items()
        if isinstance(value, (bool, int, float, np.integer, np.floating))
    }
    plant_rows: list[dict[str, Any]] = [
        {
            "diffusion_scale": selected_diffusion,
            "increment_scale": selected_increment,
            "persistence_skip": selected_persistence,
            "calibration_source": "canonical_conditional_oof_part2_selection",
            "source_stage": str(frozen_oof_selection["source_stage"]),
            "support_contract": str(frozen_oof_selection["support_contract"]),
            "trajectory_diagnostics_used_in_score_or_gate": False,
            "selection_file_sha256": str(
                frozen_oof_selection["selection_file_sha256"]
            ),
            "plant_fidelity_gate": 1.0,
            "selected_by_plant_only": True,
            "plant_fidelity_failure": False,
            **selected_metrics,
        }
    ]

    rows: list[dict[str, Any]] = []
    results: dict[str, tuple[ControllerConfig, ValidationResult]] = {}
    for target_quantile in target_quantile_grid:
        candidate_network = network_with_target_quantile(network, target_quantile)
        for gain in gain_grid:
            for tau in tau_grid:
                candidate_id = (
                    f"q-{target_quantile:.2f}_gain-{gain:.3f}_tau-{tau:.3f}"
                )
                config = replace(
                    base_config,
                    diffusion_scale=selected_diffusion,
                    increment_scale=selected_increment,
                    persistence_skip=selected_persistence,
                    target_quantile=float(target_quantile),
                    total_episode_energy_budget=primary_budget,
                    base_gain_scale=float(gain),
                    graph_diffusion_time=float(tau),
                    teacher_epochs=0,
                    wgan_epochs=0,
                )
                stack = build_control_stack(
                    model, candidate_network, fit_contexts, fit_reference, config
                )
                stepper = stack[2]
                validation_initial = [
                    stepper.adapter.initial_state_from_context(item)
                    for item in validation_contexts
                ]
                result = validate_actor(
                    stack[-1],
                    stepper,
                    validation_initial,
                    torch.as_tensor(validation_reference, dtype=torch.float64),
                    stack[7],
                    stack[6],
                    fixed_joint_projections(stepper.n_channels, config.seed + 811),
                    config,
                    observed=None,
                    external_plant_fidelity_gate=True,
                )
                row: dict[str, Any] = {
                    "candidate_id": candidate_id,
                    "target_quantile": float(target_quantile),
                    "actuator_count": int(stepper.actuator_dim),
                    "base_gain_scale": float(gain),
                    "graph_diffusion_time": float(tau),
                    "frozen_part2_diffusion_scale": selected_diffusion,
                    "frozen_part2_increment_scale": selected_increment,
                    "frozen_part2_persistence_skip": selected_persistence,
                    "fixed_q065_total_episode_energy_budget": primary_budget,
                    "analytical_screen_score": analytical_screen_score(
                        result.aggregate
                    ),
                    **result.aggregate,
                }
                rows.append(row)
                results[candidate_id] = (config, result)
    eligible = [
        row for row in rows if bool(row["controller_feasible"]) and not plant_failure
    ]
    selected_rows = _take_with_one_percent_tie_break(eligible, 2)
    top_two = [str(row["candidate_id"]) for row in selected_rows]
    diagnostic_pool = eligible or rows
    diagnostic = _take_with_one_percent_tie_break(diagnostic_pool, 1)[0]
    selected_config, selected_result = results[str(diagnostic["candidate_id"])]
    ranked = sorted(
        rows,
        key=lambda row: (
            0 if row["candidate_id"] in top_two else 1,
            float(row["analytical_screen_score"]),
            float(row["delivered_energy"]),
            -float(row["target_quantile"]),
            float(row["base_gain_scale"]),
            float(row["graph_diffusion_time"]),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["display_rank"] = int(rank)
        row["selected_for_neural_stage"] = bool(row["candidate_id"] in top_two)
        row["plant_fidelity_failure"] = bool(plant_failure)
    return selected_config, selected_result, plant_rows, ranked, top_two


def scalar_terms(terms: Mapping[str, Tensor], prefix: str = "") -> dict[str, float]:
    return {
        f"{prefix}{key}": float(value.detach().cpu())
        for key, value in terms.items()
    }


def train_teacher(
    actor: StructuredSplineCovarianceActor,
    stepper: FrozenIctalGraphRCBatchStepper,
    fit_initial: Sequence[Tensor],
    validation_initial: Sequence[Tensor],
    fit_reference: Tensor,
    validation_reference: Tensor,
    validation_observed: np.ndarray | None,
    channel_weights: Tensor,
    reference_scale: Tensor,
    projections: Tensor,
    config: ControllerConfig,
    *,
    external_plant_fidelity_gate: bool | None = None,
) -> tuple[
    StructuredSplineCovarianceActor,
    list[dict[str, float]],
    ValidationResult,
    bool,
]:
    validation_cache = build_validation_invariant_cache(
        stepper,
        validation_initial,
        validation_reference,
        config,
        seed_offset=10000,
    )
    baseline = validate_actor_cached(
        actor,
        stepper,
        validation_initial,
        validation_reference,
        channel_weights,
        reference_scale,
        projections,
        config,
        validation_cache,
        observed=validation_observed,
        external_plant_fidelity_gate=external_plant_fidelity_gate,
        seed_offset=10000,
    )
    history: list[dict[str, float]] = [
        {"stage": "teacher", "epoch": 0.0, "selection_score": 1.0, **baseline.aggregate}
    ]
    best_state = copy.deepcopy(actor.state_dict())
    best_score = 1.0
    best_result = baseline
    accepted = False
    optimizer = torch.optim.AdamW(
        actor.parameters(), lr=config.teacher_learning_rate, weight_decay=1.0e-5
    )
    budget = episode_energy_budget(config, stepper.actuator_dim)
    for epoch in range(int(config.teacher_epochs)):
        actor.train()
        context_index = epoch % len(fit_initial)
        noise = antithetic_noise(
            config.seed + 1009 * (epoch + 1),
            config.particles,
            config.horizon,
            stepper.q,
        )
        with torch.no_grad():
            free = uncontrolled_particle_rollout(
                stepper, fit_initial[context_index], noise
            )[:, 1:]
        optimizer.zero_grad(set_to_none=True)
        rollout = budgeted_particle_rollout(
            stepper,
            actor,
            fit_initial[context_index],
            noise,
            energy_budget=budget,
        )
        loss, terms = distribution_control_objective(
            rollout,
            fit_reference,
            free,
            channel_weights,
            reference_scale,
            projections,
            actor,
        )
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(actor.parameters(), 2.0)
        optimizer.step()
        if (epoch + 1) % int(config.validation_every) == 0 or (
            epoch + 1 == int(config.teacher_epochs)
        ):
            result = validate_actor_cached(
                actor,
                stepper,
                validation_initial,
                validation_reference,
                channel_weights,
                reference_scale,
                projections,
                config,
                validation_cache,
                observed=validation_observed,
                external_plant_fidelity_gate=external_plant_fidelity_gate,
                seed_offset=10000,
            )
            score = selection_score(result.aggregate, baseline.aggregate)
            eligible = checkpoint_eligible(
                result.aggregate, baseline.aggregate, score
            )
            history.append(
                {
                    "stage": "teacher",
                    "epoch": float(epoch + 1),
                    "selection_score": score,
                    "eligible": float(eligible),
                    "train_gradient_norm": float(torch.as_tensor(gradient)),
                    **scalar_terms(terms, "train_"),
                    **result.aggregate,
                }
            )
            if eligible and score < best_score:
                best_score = score
                best_state = copy.deepcopy(actor.state_dict())
                best_result = result
                accepted = True
    actor.load_state_dict(best_state, strict=True)
    actor.eval()
    del validation_cache
    return actor, history, best_result, accepted


def train_wgan(
    actor: StructuredSplineCovarianceActor,
    stepper: FrozenIctalGraphRCBatchStepper,
    fit_initial: Sequence[Tensor],
    validation_initial: Sequence[Tensor],
    fit_reference: Tensor,
    validation_reference: Tensor,
    validation_observed: np.ndarray | None,
    channel_weights: Tensor,
    reference_scale: Tensor,
    projections: Tensor,
    config: ControllerConfig,
    *,
    external_plant_fidelity_gate: bool | None = None,
) -> tuple[
    StructuredSplineCovarianceActor,
    TimeConditionedWassersteinCritic,
    list[dict[str, float]],
    ValidationResult,
    bool,
]:
    teacher = copy.deepcopy(actor)
    teacher.eval()
    set_requires_grad(teacher, False)
    critic = TimeConditionedWassersteinCritic(
        fit_reference.reshape(-1, stepper.n_channels).mean(dim=0),
        reference_scale,
        horizon_samples=config.horizon,
        hidden_size=128,
    )
    actor_optimizer = torch.optim.AdamW(
        actor.parameters(), lr=config.actor_learning_rate, weight_decay=1.0e-5
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=config.critic_learning_rate, betas=(0.0, 0.9)
    )
    budget = episode_energy_budget(config, stepper.actuator_dim)
    wgan_indices = tuple(range(15, config.horizon, 16))
    anchor_indices = tuple(range(0, config.horizon, 16))

    pretrain_banks: list[Tensor] = []
    with torch.no_grad():
        for bank in range(int(config.critic_pretrain_banks)):
            noise = antithetic_noise(
                config.seed + 30000 + 101 * bank,
                config.particles,
                config.horizon,
                stepper.q,
            )
            initial = fit_initial[bank % len(fit_initial)]
            pretrain_banks.append(
                budgeted_particle_rollout(
                    stepper, teacher, initial, noise, energy_budget=budget
                ).scaled[:, 1:].detach()
            )
    history: list[dict[str, float]] = []
    for update in range(int(config.critic_pretrain_steps)):
        critic_optimizer.zero_grad(set_to_none=True)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed + 40000 + update)
        audit = balanced_wgan_gp_loss(
            critic,
            pretrain_banks[update % len(pretrain_banks)],
            fit_reference,
            wgan_indices,
            gradient_penalty_weight=config.gradient_penalty,
            critic_drift_weight=config.critic_drift,
            generator=generator,
        )
        audit.loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
        critic_optimizer.step()
        history.append(
            {
                "stage": "critic_pretrain",
                "epoch": float(update + 1),
                "critic_loss": float(audit.loss.detach()),
                "critic_estimate": float(audit.estimate.detach()),
                "critic_gradient_penalty": float(audit.gradient_penalty.detach()),
                "critic_gradient_norm": float(torch.as_tensor(gradient)),
            }
        )

    validation_cache = build_validation_invariant_cache(
        stepper,
        validation_initial,
        validation_reference,
        config,
        seed_offset=20000,
    )
    baseline = validate_actor_cached(
        actor,
        stepper,
        validation_initial,
        validation_reference,
        channel_weights,
        reference_scale,
        projections,
        config,
        validation_cache,
        observed=validation_observed,
        external_plant_fidelity_gate=external_plant_fidelity_gate,
        seed_offset=20000,
    )
    best_score = 1.0
    best_state: dict[str, Tensor] | None = None
    best_critic: dict[str, Tensor] | None = None
    best_result = baseline
    for epoch in range(int(config.wgan_epochs)):
        context_index = epoch % len(fit_initial)
        initial = fit_initial[context_index]
        noise = antithetic_noise(
            config.seed + 50000 + 1009 * (epoch + 1),
            config.particles,
            config.horizon,
            stepper.q,
        )
        actor.eval()
        with torch.no_grad():
            detached = budgeted_particle_rollout(
                stepper, actor, initial, noise, energy_budget=budget
            ).scaled[:, 1:].detach()
        set_requires_grad(actor, False)
        set_requires_grad(critic, True)
        critic.train()
        critic_audit = None
        for critic_step in range(int(config.critic_steps)):
            critic_optimizer.zero_grad(set_to_none=True)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                config.seed + 60000 + 10000 * (epoch + 1) + critic_step
            )
            critic_audit = balanced_wgan_gp_loss(
                critic,
                detached,
                fit_reference,
                wgan_indices,
                gradient_penalty_weight=config.gradient_penalty,
                critic_drift_weight=config.critic_drift,
                generator=generator,
            )
            critic_audit.loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
            critic_optimizer.step()

        set_requires_grad(critic, False)
        set_requires_grad(actor, True)
        actor.train()
        actor_optimizer.zero_grad(set_to_none=True)
        rollout = budgeted_particle_rollout(
            stepper, actor, initial, noise, energy_budget=budget
        )
        with torch.no_grad():
            free = uncontrolled_particle_rollout(stepper, initial, noise)[:, 1:]
        law, terms = distribution_control_objective(
            rollout,
            fit_reference,
            free,
            channel_weights,
            reference_scale,
            projections,
            actor,
        )
        adversarial, estimate = actor_wasserstein_loss(
            critic, rollout.scaled[:, 1:], fit_reference, wgan_indices
        )
        anchor = actor_action_anchor_loss(
            actor,
            teacher,
            rollout,
            anchor_indices,
            amplitude_limit=config.amplitude_limit,
        )
        total = (
            law
            + config.adversarial_weight * adversarial
            + config.teacher_anchor_weight * anchor
        )
        total.backward()
        actor_gradient = torch.nn.utils.clip_grad_norm_(actor.parameters(), 2.0)
        actor_optimizer.step()
        set_requires_grad(critic, True)

        if (epoch + 1) % int(config.validation_every) == 0 or (
            epoch + 1 == int(config.wgan_epochs)
        ):
            result = validate_actor_cached(
                actor,
                stepper,
                validation_initial,
                validation_reference,
                channel_weights,
                reference_scale,
                projections,
                config,
                validation_cache,
                observed=validation_observed,
                external_plant_fidelity_gate=external_plant_fidelity_gate,
                seed_offset=20000,
            )
            score = selection_score(result.aggregate, baseline.aggregate)
            eligible = checkpoint_eligible(
                result.aggregate, baseline.aggregate, score
            )
            history.append(
                {
                    "stage": "actor_wgan",
                    "epoch": float(epoch + 1),
                    "selection_score": score,
                    "eligible": float(eligible),
                    "train_total": float(total.detach()),
                    "train_adversarial": float(adversarial.detach()),
                    "train_critic_estimate": float(estimate.detach()),
                    "train_anchor": float(anchor.detach()),
                    "train_actor_gradient_norm": float(
                        torch.as_tensor(actor_gradient)
                    ),
                    **scalar_terms(terms, "train_"),
                    **result.aggregate,
                }
            )
            if eligible and score < best_score:
                best_score = score
                best_state = copy.deepcopy(actor.state_dict())
                best_critic = copy.deepcopy(critic.state_dict())
                best_result = result
    accepted = best_state is not None
    if accepted:
        actor.load_state_dict(best_state, strict=True)
        assert best_critic is not None
        critic.load_state_dict(best_critic, strict=True)
    else:
        actor.load_state_dict(teacher.state_dict(), strict=True)
    actor.eval()
    critic.eval()
    del validation_cache
    return actor, critic, history, best_result, accepted


def prefix_refit(
    actor: StructuredSplineCovarianceActor,
    stepper: FrozenIctalGraphRCBatchStepper,
    initial: Tensor,
    reference: Tensor,
    channel_weights: Tensor,
    reference_scale: Tensor,
    projections: Tensor,
    config: ControllerConfig,
    *,
    epochs: int,
) -> list[dict[str, float]]:
    """Fixed-schedule prefix adaptation; no future or checkpoint selection."""

    if int(epochs) < 1:
        raise ValueError("prefix-refit epochs must be positive")
    optimizer = torch.optim.AdamW(
        actor.parameters(), lr=config.actor_learning_rate, weight_decay=1.0e-5
    )
    budget = episode_energy_budget(config, stepper.actuator_dim)
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        noise = antithetic_noise(
            config.seed + 90000 + 1009 * (epoch + 1),
            config.particles,
            config.horizon,
            stepper.q,
        )
        actor.train()
        optimizer.zero_grad(set_to_none=True)
        rollout = budgeted_particle_rollout(
            stepper, actor, initial, noise, energy_budget=budget
        )
        with torch.no_grad():
            free = uncontrolled_particle_rollout(stepper, initial, noise)[:, 1:]
        loss, terms = distribution_control_objective(
            rollout,
            reference,
            free,
            channel_weights,
            reference_scale,
            projections,
            actor,
        )
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(actor.parameters(), 2.0)
        optimizer.step()
        history.append(
            {
                "stage": "prefix_refit_fixed_schedule",
                "epoch": float(epoch + 1),
                "gradient_norm": float(torch.as_tensor(gradient)),
                **scalar_terms(terms),
            }
        )
    actor.eval()
    return history


def save_checkpoint(
    path: Path,
    actor: StructuredSplineCovarianceActor,
    critic: TimeConditionedWassersteinCritic | None,
    *,
    config: ControllerConfig,
    selected_stage: str,
    input_hashes: Mapping[str, str],
    config_sha256: str,
) -> None:
    target_indices = (
        actor.actuated_channel_indices.detach().cpu().numpy().astype(np.int64)
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "selected_stage": str(selected_stage),
        "neural_stage_authorized": bool(
            str(selected_stage)
            in {
                "analytical_grid_selected_for_neural_stage",
                "structured_full_markov_teacher",
                "structured_full_markov_actor_wgan_gp",
                "prefix_refit_fixed_final_epoch",
            }
        ),
        "prefix_refit_authorized": bool(
            str(selected_stage)
            in {
                "structured_full_markov_teacher",
                "structured_full_markov_actor_wgan_gp",
                "prefix_refit_fixed_final_epoch",
            }
        ),
        "controller_config": asdict(config),
        "config_sha256": str(config_sha256),
        "input_hashes": dict(input_hashes),
        "controller_source_sha256": sha256_file(Path(__file__).resolve()),
        "source_model_sha256": str(input_hashes.get("model_sha256", "")),
        "source_network_sha256": str(input_hashes.get("network_sha256", "")),
        "context_bundle_sha256": str(
            input_hashes.get("context_fit_file_sha256", "")
        ),
        "reference_bundle_sha256": str(
            input_hashes.get("reference_fit_file_sha256", "")
        ),
        "target_indices": target_indices.tolist(),
        "target_indices_sha256": sha256_array(target_indices),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "display_evaluation_seed": int(DISPLAY_EVALUATION_SEED),
        "method_label": METHOD_LABEL,
        "critic_role": "Kantorovich critic for empirical distribution discrepancy",
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": None if critic is None else critic.state_dict(),
        "channels": int(actor.stepper.n_channels),
        "actuators": int(actor.stepper.actuator_dim),
        "latent_dimension": int(actor.stepper.q),
        "markov_dimension": int(actor.stepper.state_dim),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_trainable_actor_parameters(
    actor: StructuredSplineCovarianceActor,
    checkpoint_state: Mapping[str, Tensor],
) -> None:
    """Load learned policy weights while recomputing prefix/reference buffers.

    Hyperparameters and trainable weights stay frozen from development.  The
    target moments and Markov normalization are sufficient statistics of the
    pooled historical interictal target and the currently observed prefix, so
    they are rebuilt without using any future ictal outcome.
    """

    current = actor.state_dict()
    parameter_names = {name for name, _ in actor.named_parameters()}
    missing = sorted(parameter_names.difference(checkpoint_state))
    if missing:
        raise ValueError(f"checkpoint lacks actor parameters: {missing}")
    for name in parameter_names:
        value = checkpoint_state[name]
        if current[name].shape != value.shape:
            raise ValueError(f"checkpoint parameter shape differs for {name}")
        current[name] = value
    actor.load_state_dict(current, strict=True)


def save_rollout_npz(path: Path, display: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **display)
    os.replace(temporary, path)


def audited_rollout_payload(
    display: Mapping[str, np.ndarray],
    target_indices: np.ndarray,
    config: ControllerConfig,
    *,
    mode: str,
) -> dict[str, np.ndarray]:
    noise = np.ascontiguousarray(np.asarray(display["noise"], dtype=np.float64))
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "mode": str(mode),
        "method_label": METHOD_LABEL,
        "critic_role": "Kantorovich critic for empirical distribution discrepancy",
        "law_coordinate": "decoded_standardized",
        "standard_normal_pairing": "byte-identical common random numbers",
        "diffusion_note": (
            "saved banks are standard-normal drivers; the frozen "
            "state-dependent diffusion maps them to latent innovations"
        ),
        "physical_input_map": "strict local selector",
        "graph_diffusion_role": "feedback feature and later Graph-RC propagation",
        "state_dependent_diffusion_scale": float(config.diffusion_scale),
        "multistep_increment_scale": float(config.increment_scale),
        "multistep_persistence_skip": float(config.persistence_skip),
        "target_quantile": float(config.target_quantile),
        "fixed_total_episode_energy_budget": float(
            config.total_episode_energy_budget
        ),
    }
    payload = {
        **{key: np.asarray(value) for key, value in display.items()},
        "free_standardized": np.asarray(display["free"]),
        "controlled_standardized": np.asarray(display["controlled"]),
        "free_standard_normal": noise.copy(),
        "controlled_standard_normal": noise.copy(),
        "target_indices": np.asarray(target_indices, dtype=np.int64),
        "amplitude_limit": np.asarray(config.amplitude_limit, dtype=np.float64),
        "dt": np.asarray(1.0 / config.sampling_rate_hz, dtype=np.float64),
        "metadata_json": np.asarray(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        ),
    }
    # The development-only files retain aliases for existing diagnostics.
    # Prefix/final artifacts use only the unambiguous standard-normal names.
    if str(mode) != "prefix-refit":
        payload["free_innovations"] = noise.copy()
        payload["controlled_innovations"] = noise.copy()
    return payload


def build_manifest(output: Path) -> None:
    rows = []
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "artifact_manifest.csv":
            rows.append(
                {
                    "filename": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    pd.DataFrame(rows).to_csv(
        output / "artifact_manifest.csv", index=False, encoding="utf-8-sig"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact HUP060-style patient-adaptive Part-III controller"
    )
    parser.add_argument(
        "--mode",
        choices=(
            "analytical-grid",
            "analytical-screen",
            "development",
            "prefix-refit",
        ),
        default="development",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--context-fit", type=Path, required=True)
    parser.add_argument("--context-fit-key")
    parser.add_argument(
        "--context-coordinate",
        choices=("raw_model_input", "legacy_hup060_scaled_model_input"),
        required=True,
        help="coordinate passed directly to fitted_model._context_state",
    )
    parser.add_argument("--context-validation", type=Path)
    parser.add_argument("--context-validation-key")
    parser.add_argument("--reference-fit", type=Path, required=True)
    parser.add_argument("--reference-fit-key")
    parser.add_argument("--reference-validation", type=Path)
    parser.add_argument("--reference-validation-key")
    parser.add_argument(
        "--observed-validation",
        type=Path,
        help="early-development ictal futures paired one-to-one with validation contexts",
    )
    parser.add_argument("--observed-validation-key")
    parser.add_argument(
        "--law-coordinate",
        choices=("decoded_standardized",),
        default="decoded_standardized",
        help="coordinate of reference and observed laws",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--frozen-config", type=Path)
    parser.add_argument(
        "--frozen-oof-plant-selection",
        type=Path,
        help=(
            "canonical immutable OOF Part-II drift/diffusion selection; mandatory for "
            "analytical-grid/development and never generated by this runner"
        ),
    )
    parser.add_argument("--prefix-epochs", type=int, default=20)
    parser.add_argument("--diffusion-scale", type=float, default=0.79451175)
    parser.add_argument(
        "--diffusion-scale-grid",
        default="0.60,0.79451175,1.00",
        help=(
            "legacy audit echo of the external Part-II diffusion grid; "
            "the controller never ranks or overrides the canonical plant selection"
        ),
    )
    parser.add_argument("--graph-diffusion-time", type=float, default=0.50)
    parser.add_argument("--base-gain-scale", type=float, default=0.50)
    parser.add_argument(
        "--target-quantile",
        type=float,
        choices=(0.50, 0.65, 0.80),
        default=0.65,
        help="SOZ-blind centrality threshold selected on development data",
    )
    parser.add_argument(
        "--target-quantile-grid",
        default="0.50,0.65,0.80",
        help="fixed SOZ-blind actuator-count grid",
    )
    parser.add_argument(
        "--base-gain-grid",
        default="0.10,0.50,1.00",
        help="fixed development-only analytical grid",
    )
    parser.add_argument(
        "--graph-diffusion-grid",
        default="0.25,0.50,0.75",
        help="fixed development-only graph heat-kernel grid",
    )
    parser.add_argument("--base-gain-ridge", type=float, default=0.20)
    parser.add_argument("--markov-residual-scale", type=float, default=0.60)
    parser.add_argument("--teacher-epochs", type=int, default=180)
    parser.add_argument("--wgan-epochs", type=int, default=40)
    parser.add_argument("--validation-every", type=int, default=4)
    parser.add_argument("--teacher-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--actor-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--critic-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--critic-pretrain-steps", type=int, default=24)
    parser.add_argument("--critic-pretrain-banks", type=int, default=3)
    parser.add_argument("--critic-steps", type=int, default=3)
    parser.add_argument("--adversarial-weight", type=float, default=0.50)
    parser.add_argument("--teacher-anchor-weight", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--nonmanuscript-smoke",
        action="store_true",
        help="allow reduced epochs/particles only for execution tests",
    )
    parser.add_argument("--particles", type=int, default=PARTICLES)
    return parser.parse_args(argv)


def validate_cli(args: argparse.Namespace) -> None:
    if args.mode in {"development", "analytical-screen", "analytical-grid"}:
        if args.context_validation is None or args.reference_validation is None:
            raise ValueError("development modes require independent validation inputs")
        if args.mode == "analytical-screen" and args.observed_validation is None:
            raise ValueError(
                "legacy analytical-screen requires observed ictal futures"
            )
        if args.teacher_checkpoint is not None or args.frozen_config is not None:
            raise ValueError("development modes cannot consume a final-prefix checkpoint")
    if args.mode in {"analytical-grid", "development"}:
        if args.frozen_oof_plant_selection is None:
            raise ValueError(
                "analytical-grid/development require a canonical frozen OOF plant selection"
            )
        if args.observed_validation is not None:
            raise PermissionError(
                "deployment observed futures are forbidden after frozen OOF selection"
            )
    if args.mode == "prefix-refit":
        if (
            args.context_validation is not None
            or args.reference_validation is not None
            or args.observed_validation is not None
        ):
            raise PermissionError("prefix-refit forbids validation/future inputs")
        if args.teacher_checkpoint is None or args.frozen_config is None:
            raise ValueError("prefix-refit requires a frozen development checkpoint/config")
        if args.frozen_oof_plant_selection is not None:
            raise ValueError("prefix-refit inherits the OOF decision from its checkpoint")
    if not args.nonmanuscript_smoke and int(args.particles) != PARTICLES:
        raise ValueError("manuscript execution requires exactly 32 particles")
    if int(args.particles) < 2 or int(args.particles) % 2:
        raise ValueError("particles must be a positive even integer")
    if min(
        args.diffusion_scale,
        args.base_gain_ridge,
        args.teacher_learning_rate,
        args.actor_learning_rate,
        args.critic_learning_rate,
    ) <= 0.0:
        raise ValueError("scales, ridge, and learning rates must be positive")
    if args.graph_diffusion_time < 0.0 or args.base_gain_scale < 0.0:
        raise ValueError("graph diffusion/gain scale cannot be negative")
    gain_grid = parse_float_grid(args.base_gain_grid, name="base_gain_grid")
    tau_grid = parse_float_grid(
        args.graph_diffusion_grid, name="graph_diffusion_grid"
    )
    diffusion_grid = parse_float_grid(
        args.diffusion_scale_grid, name="diffusion_scale_grid"
    )
    target_grid = parse_float_grid(
        args.target_quantile_grid, name="target_quantile_grid"
    )
    if any(item < 0.0 for item in gain_grid + tau_grid) or any(
        item <= 0.0 for item in diffusion_grid
    ):
        raise ValueError("analytical grids cannot contain negative values")
    if args.mode == "analytical-grid":
        if gain_grid != (0.10, 0.50, 1.00):
            raise ValueError("paper-exact analytical gain grid is frozen")
        if tau_grid != (0.25, 0.50, 0.75):
            raise ValueError("paper-exact graph diffusion grid is frozen")
        if target_grid != (0.50, 0.65, 0.80):
            raise ValueError("paper-exact target-quantile grid is frozen")
    if not 0.0 <= args.markov_residual_scale <= 1.0:
        raise ValueError("markov residual scale must lie in [0,1]")
    if args.mode == "development" and args.teacher_epochs < 1:
        raise ValueError("development training requires teacher_epochs >= 1")
    if args.wgan_epochs < 0 or args.validation_every < 1:
        raise ValueError("WGAN epochs and validation interval are invalid")
    if args.mode == "prefix-refit" and args.prefix_epochs < 1:
        raise ValueError("prefix-refit requires a positive fixed epoch count")


def args_to_config(args: argparse.Namespace) -> ControllerConfig:
    return ControllerConfig(
        particles=int(args.particles),
        diffusion_scale=float(args.diffusion_scale),
        graph_diffusion_time=float(args.graph_diffusion_time),
        base_gain_scale=float(args.base_gain_scale),
        target_quantile=float(args.target_quantile),
        base_gain_ridge=float(args.base_gain_ridge),
        markov_residual_scale=float(args.markov_residual_scale),
        teacher_epochs=(
            0
            if args.mode in {"analytical-screen", "analytical-grid"}
            else int(args.teacher_epochs)
        ),
        teacher_learning_rate=float(args.teacher_learning_rate),
        wgan_epochs=(
            0
            if args.mode in {"analytical-screen", "analytical-grid"}
            else int(args.wgan_epochs)
        ),
        actor_learning_rate=float(args.actor_learning_rate),
        critic_learning_rate=float(args.critic_learning_rate),
        validation_every=int(args.validation_every),
        critic_pretrain_steps=int(args.critic_pretrain_steps),
        critic_pretrain_banks=int(args.critic_pretrain_banks),
        critic_steps=int(args.critic_steps),
        adversarial_weight=float(args.adversarial_weight),
        teacher_anchor_weight=float(args.teacher_anchor_weight),
        seed=int(args.seed),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_cli(args)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    started = time.perf_counter()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    model_path = args.model.resolve()
    network_path = args.network.resolve()
    if (
        args.context_coordinate == "legacy_hup060_scaled_model_input"
        and "hup060" not in str(model_path).lower()
    ):
        raise ValueError(
            "legacy scaled model-input coordinates are restricted to HUP060 artifacts"
        )
    validate_coordinate_metadata(
        args.context_fit,
        metadata_key="context_coordinate",
        declared=args.context_coordinate,
    )
    validate_coordinate_metadata(
        args.reference_fit,
        metadata_key="law_coordinate",
        declared=args.law_coordinate,
    )
    if args.context_validation is not None:
        validate_coordinate_metadata(
            args.context_validation,
            metadata_key="context_coordinate",
            declared=args.context_coordinate,
        )
    if args.reference_validation is not None:
        validate_coordinate_metadata(
            args.reference_validation,
            metadata_key="law_coordinate",
            declared=args.law_coordinate,
        )
    if args.observed_validation is not None:
        validate_coordinate_metadata(
            args.observed_validation,
            metadata_key="law_coordinate",
            declared=args.law_coordinate,
        )
    model = load_fitted_model(model_path)
    network = load_network(network_path, model)
    n_channels = int(model.transform.n_channels_)
    prefix_strict = args.mode == "prefix-refit"
    context_fit = normalize_contexts(
        load_numeric_array(
            args.context_fit,
            key=args.context_fit_key,
            preferred_keys=(
                "context_fit_model_input_raw",
                "ictal_context_model_input_raw",
                "contexts",
                "past_context_scaled",
                "context",
            ),
            prefix_strict=prefix_strict,
        ),
        n_channels,
    )
    reference_fit = normalize_reference(
        load_numeric_array(
            args.reference_fit,
            key=args.reference_fit_key,
            preferred_keys=(
                "reference_fit_standardized_paths",
                "reference_target_standardized_paths",
                "reference_paths",
                "run01_reference_fit_pool_scaled",
                "reference",
            ),
        ),
        n_channels,
    )
    if args.mode == "prefix-refit":
        context_validation = None
        reference_validation = None
        observed_validation = None
    else:
        assert args.context_validation is not None
        assert args.reference_validation is not None
        context_validation = normalize_contexts(
            load_numeric_array(
                args.context_validation,
                key=args.context_validation_key,
                preferred_keys=(
                    "context_validation_model_input_raw",
                    "contexts",
                    "past_context_scaled",
                    "context",
                ),
            ),
            n_channels,
        )
        reference_validation = normalize_reference(
            load_numeric_array(
                args.reference_validation,
                key=args.reference_validation_key,
                preferred_keys=(
                    "reference_validation_standardized_paths",
                    "reference_paths",
                    "run01_reference_fit_pool_scaled",
                    "reference",
                ),
            ),
            n_channels,
        )
        observed_validation = (
            None
            if args.observed_validation is None
            else normalize_observed_validation(
                load_numeric_array(
                    args.observed_validation,
                    key=args.observed_validation_key,
                    preferred_keys=(
                        "observed_validation",
                        "observed_validation_standardized",
                        "observed_futures",
                        "observed_future",
                    ),
                ),
                len(context_validation),
                n_channels,
            )
        )
    if args.nonmanuscript_smoke:
        context_fit = context_fit[:1]
        reference_fit = reference_fit[: min(4, len(reference_fit))]
        if context_validation is not None:
            context_validation = context_validation[:1]
        if reference_validation is not None:
            reference_validation = reference_validation[
                : min(4, len(reference_validation))
            ]
        if observed_validation is not None:
            observed_validation = observed_validation[:1]

    input_hashes = {
        "model_sha256": sha256_file(model_path),
        "network_sha256": sha256_file(network_path),
        "context_fit_file_sha256": sha256_file(args.context_fit.resolve()),
        "context_fit_array_sha256": sha256_array(context_fit),
        "reference_fit_file_sha256": sha256_file(args.reference_fit.resolve()),
        "reference_fit_array_sha256": sha256_array(reference_fit),
    }
    if context_validation is not None and reference_validation is not None:
        input_hashes.update(
            {
                "context_validation_file_sha256": sha256_file(
                    args.context_validation.resolve()
                ),
                "context_validation_array_sha256": sha256_array(context_validation),
                "reference_validation_file_sha256": sha256_file(
                    args.reference_validation.resolve()
                ),
                "reference_validation_array_sha256": sha256_array(reference_validation),
            }
        )
    if observed_validation is not None:
        assert args.observed_validation is not None
        input_hashes.update(
            {
                "observed_validation_file_sha256": sha256_file(
                    args.observed_validation.resolve()
                ),
                "observed_validation_array_sha256": sha256_array(
                    observed_validation
                ),
            }
        )

    config = args_to_config(args)
    frozen_oof_selection: dict[str, Any] | None = None
    if args.mode in {"analytical-grid", "development"}:
        assert args.frozen_oof_plant_selection is not None
        frozen_oof_selection = load_frozen_oof_plant_selection(
            args.frozen_oof_plant_selection,
            deployment_model_path=model_path,
        )
        config = config_with_frozen_oof_plant(config, frozen_oof_selection)
        input_hashes["frozen_oof_plant_selection_sha256"] = str(
            frozen_oof_selection["selection_file_sha256"]
        )
        input_hashes["source_oof_bundle_sha256"] = str(
            frozen_oof_selection["source_oof_bundle_sha256"]
        )
        input_hashes["source_oof_manifest_sha256"] = str(
            frozen_oof_selection["source_oof_manifest_sha256"]
        )
        input_hashes["source_root_oof_manifest_sha256"] = str(
            frozen_oof_selection["source_root_oof_manifest_sha256"]
        )
    loaded_checkpoint: dict[str, Any] | None = None
    if args.mode == "prefix-refit":
        assert args.teacher_checkpoint is not None and args.frozen_config is not None
        frozen_config = json.loads(
            args.frozen_config.resolve().read_text(encoding="utf-8")
        )
        loaded_checkpoint = torch.load(
            args.teacher_checkpoint.resolve(), map_location="cpu", weights_only=False
        )
        if not bool(loaded_checkpoint.get("prefix_refit_authorized", False)):
            raise PermissionError(
                "prefix-refit requires a trained, plant-passing development checkpoint"
            )
        expected_config_hash = sha256_file(args.frozen_config.resolve())
        if loaded_checkpoint.get("config_sha256") != expected_config_hash:
            raise ValueError("checkpoint and frozen configuration hashes differ")
        stored = loaded_checkpoint["controller_config"]
        config = ControllerConfig(**stored)
        if loaded_checkpoint["input_hashes"]["model_sha256"] != input_hashes[
            "model_sha256"
        ]:
            raise ValueError("prefix checkpoint was trained with a different plant")
        if loaded_checkpoint["input_hashes"]["network_sha256"] != input_hashes[
            "network_sha256"
        ]:
            raise ValueError("prefix checkpoint was trained with a different network")
        input_hashes = inherit_frozen_oof_hashes(input_hashes, loaded_checkpoint)
        input_hashes["teacher_checkpoint_sha256"] = sha256_file(
            args.teacher_checkpoint.resolve()
        )
        input_hashes["frozen_config_sha256"] = expected_config_hash

    config_payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "execution_class": (
            "nonmanuscript_smoke" if args.nonmanuscript_smoke else "manuscript_candidate"
        ),
        "controller_config": asdict(config),
        "method_contract": {
            "method_label": METHOD_LABEL,
            "critic_role": "Kantorovich critic for empirical distribution discrepancy",
            "pde_claim": (
                "HJB--FP motivated empirical-particle optimization without "
                "direct PDE-residual enforcement"
            ),
            "physical_input_map": "strict local selector on selected electrodes",
            "graph_diffusion_role": (
                "feedback prior and later propagation through frozen Graph-RC dynamics"
            ),
            "evaluation_seeds": list(EVALUATION_SEEDS),
            "display_evaluation_seed": int(DISPLAY_EVALUATION_SEED),
        },
        "prefix_refit_epochs": int(args.prefix_epochs),
        "input_paths": {
            "model": model_path,
            "network": network_path,
            "context_fit": args.context_fit.resolve(),
            "context_validation": (
                None
                if args.context_validation is None
                else args.context_validation.resolve()
            ),
            "reference_fit": args.reference_fit.resolve(),
            "reference_validation": (
                None
                if args.reference_validation is None
                else args.reference_validation.resolve()
            ),
            "observed_validation": (
                None
                if args.observed_validation is None
                else args.observed_validation.resolve()
            ),
            "frozen_oof_plant_selection": (
                None
                if args.frozen_oof_plant_selection is None
                else args.frozen_oof_plant_selection.resolve()
            ),
        },
        "input_hashes": input_hashes,
        "frozen_oof_plant_selection": (
            None
            if frozen_oof_selection is None
            else {
                "subject": frozen_oof_selection["subject"],
                "selected_increment_scale": frozen_oof_selection[
                    "selected_increment_scale"
                ],
                "selected_persistence_skip": frozen_oof_selection[
                    "selected_persistence_skip"
                ],
                "selected_diffusion_scale": frozen_oof_selection[
                    "selected_diffusion_scale"
                ],
                "plant_pass": frozen_oof_selection["plant_pass"],
                "source_stage": frozen_oof_selection["source_stage"],
                "support_contract": frozen_oof_selection["support_contract"],
                "trajectory_diagnostics_used_in_score_or_gate": False,
                "selection_file_sha256": frozen_oof_selection[
                    "selection_file_sha256"
                ],
                "deployment_refit_model_used": False,
            }
        ),
        "causal_contract": {
            "plant_dynamics": (
                "frozen state-dependent ictal Part-II diffusion with the "
                "OOF-selected multistep exposure calibration"
            ),
            "context_coordinate": args.context_coordinate,
            "context_transform_rule": (
                "passed unchanged to fitted_model._context_state; that method applies "
                "the RC training transform exactly once"
            ),
            "law_coordinate": args.law_coordinate,
            "paired_common_random_numbers": True,
            "validation_cache": (
                "within each neural stage, cache only the fixed innovation banks "
                "and uncontrolled rollouts under a hash-bound fail-closed key; "
                "recompute every Actor-dependent controlled rollout"
            ),
            "empirical_w1_evaluation": (
                "vectorized exact inverse-CDF integration for uniform empirical laws"
            ),
            "target_mask": (
                "strict Part-I S_i > Q selected only by the development grid; "
                "Q_0.65 anchors the common total-energy budget"
            ),
            "prefix_future_used": False,
            "prefix_checkpoint_selected_by_outcome": False,
            "prefix_trainable_weights": "loaded from frozen development checkpoint",
            "prefix_sufficient_statistics": (
                "recomputed from frozen historical interictal target and observed "
                "past context only"
            ),
        },
    }
    analytical_grid_rows: list[dict[str, Any]] = []
    plant_calibration_rows: list[dict[str, Any]] = []
    analytical_grid_top_two: list[str] = []
    analytical_grid_result: ValidationResult | None = None
    if args.mode == "analytical-grid":
        assert context_validation is not None
        assert reference_validation is not None
        assert frozen_oof_selection is not None
        (
            config,
            analytical_grid_result,
            plant_calibration_rows,
            analytical_grid_rows,
            analytical_grid_top_two,
        ) = run_analytical_grid(
                model,
                network,
                context_fit,
                context_validation,
                reference_fit,
                reference_validation,
                config,
                frozen_oof_selection=frozen_oof_selection,
                target_quantile_grid=parse_float_grid(
                    args.target_quantile_grid, name="target_quantile_grid"
                ),
                gain_grid=parse_float_grid(
                    args.base_gain_grid, name="base_gain_grid"
                ),
                tau_grid=parse_float_grid(
                    args.graph_diffusion_grid, name="graph_diffusion_grid"
                ),
            )
        pd.DataFrame(plant_calibration_rows).to_csv(
            output / "plant_calibration_frozen_oof_selection.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(analytical_grid_rows).to_csv(
            output / "analytical_screen_27_candidates.csv",
            index=False,
            encoding="utf-8-sig",
        )
        config_payload["controller_config"] = asdict(config)
        config_payload["analytical_grid"] = {
            "target_quantiles": parse_float_grid(
                args.target_quantile_grid, name="target_quantile_grid"
            ),
            "base_gain_scales": parse_float_grid(
                args.base_gain_grid, name="base_gain_grid"
            ),
            "graph_diffusion_times": parse_float_grid(
                args.graph_diffusion_grid, name="graph_diffusion_grid"
            ),
            "serialized_frozen_selection_row_count": len(
                plant_calibration_rows
            ),
            "plant_calibration_source": "canonical_conditional_oof_part2_selection",
            "control_candidate_count": len(analytical_grid_rows),
            "selected_diffusion_scale_by_plant_only": config.diffusion_scale,
            "selected_increment_scale_by_plant_only": config.increment_scale,
            "selected_persistence_skip_by_plant_only": config.persistence_skip,
            "top_two_for_neural_stage": analytical_grid_top_two,
            "controller_training_authorized": bool(analytical_grid_top_two),
        }
    primary_k = int(np.asarray(network["target_mask"], dtype=bool).sum())
    primary_budget = (
        float(config.horizon)
        / float(config.sampling_rate_hz)
        * primary_k
        * float(config.energy_rms) ** 2
    )
    if config.total_episode_energy_budget is None:
        config = replace(
            config,
            total_episode_energy_budget=primary_budget,
            target_quantile=float(config.target_quantile),
        )
    effective_network = network_with_target_quantile(
        network, config.target_quantile
    )
    config_payload["controller_config"] = asdict(config)
    config_payload["energy_contract"] = {
        "q065_anchor_actuator_count": primary_k,
        "fixed_total_episode_energy_budget": primary_budget,
        "candidate_actuator_count": int(
            np.asarray(effective_network["target_mask"], dtype=bool).sum()
        ),
        "same_budget_for_all_target_quantiles": True,
        "peak_cap": config.amplitude_limit,
    }
    config_path = output / "run_config.json"
    write_json(config_path, config_payload)
    config_hash = sha256_file(config_path)

    (
        _world,
        _adapter,
        stepper,
        fit_initial,
        _reference_mean,
        _reference_variance,
        reference_scale,
        channel_weights,
        actor,
    ) = build_control_stack(
        model, effective_network, context_fit, reference_fit, config
    )
    projections = fixed_joint_projections(n_channels, config.seed + 811)
    fit_reference_tensor = torch.as_tensor(reference_fit, dtype=torch.float64)
    history: list[dict[str, float]] = []
    critic: TimeConditionedWassersteinCritic | None = None
    selected_stage = "analytical_weighted_ridge"
    teacher_accepted = False
    wgan_accepted = False
    evaluation_seed_payload: dict[str, np.ndarray] = {}

    if loaded_checkpoint is not None:
        load_trainable_actor_parameters(
            actor, loaded_checkpoint["actor_state_dict"]
        )
        history = prefix_refit(
            actor,
            stepper,
            fit_initial[0],
            fit_reference_tensor,
            channel_weights,
            reference_scale,
            projections,
            config,
            epochs=int(args.prefix_epochs),
        )
        selected_stage = "prefix_refit_fixed_final_epoch"
        budget = episode_energy_budget(config, stepper.actuator_dim)
        free_by_seed: list[np.ndarray] = []
        controlled_by_seed: list[np.ndarray] = []
        controls_by_seed: list[np.ndarray] = []
        free_noise_by_seed: list[np.ndarray] = []
        controlled_noise_by_seed: list[np.ndarray] = []
        display = {}
        for evaluation_seed in EVALUATION_SEEDS:
            noise = antithetic_noise(
                int(evaluation_seed),
                config.particles,
                config.horizon,
                stepper.q,
            )
            with torch.no_grad():
                free = uncontrolled_particle_rollout(
                    stepper, fit_initial[0], noise
                )[:, 1:]
                controlled = budgeted_particle_rollout(
                    stepper, actor, fit_initial[0], noise, energy_budget=budget
                )
            free_np = free.cpu().numpy()
            controlled_np = controlled.scaled[:, 1:].cpu().numpy()
            controls_np = controlled.controls.cpu().numpy()
            noise_np = noise.cpu().numpy()
            free_by_seed.append(free_np)
            controlled_by_seed.append(controlled_np)
            controls_by_seed.append(controls_np)
            free_noise_by_seed.append(noise_np.copy())
            controlled_noise_by_seed.append(noise_np.copy())
            if int(evaluation_seed) == int(DISPLAY_EVALUATION_SEED):
                display = {
                    "free": free_np,
                    "controlled": controlled_np,
                    "reference": reference_fit,
                    "controls": controls_np,
                    "noise": noise_np,
                }
        if not display:
            raise AssertionError("display evaluation seed was not evaluated")
        evaluation_seed_payload = {
            "evaluation_seeds": np.asarray(EVALUATION_SEEDS, dtype=np.int64),
            "display_evaluation_seed": np.asarray(
                DISPLAY_EVALUATION_SEED, dtype=np.int64
            ),
            "free_standardized_by_seed": np.stack(free_by_seed),
            "controlled_standardized_by_seed": np.stack(controlled_by_seed),
            "controls_by_seed": np.stack(controls_by_seed),
            "free_standard_normal_by_seed": np.stack(free_noise_by_seed),
            "controlled_standard_normal_by_seed": np.stack(
                controlled_noise_by_seed
            ),
        }
        if not np.array_equal(
            evaluation_seed_payload["free_standard_normal_by_seed"],
            evaluation_seed_payload["controlled_standard_normal_by_seed"],
        ):
            raise AssertionError("multi-seed paired standard-normal banks differ")
        validation_result = None
    else:
        assert context_validation is not None and reference_validation is not None
        validation_initial = [
            stepper.adapter.initial_state_from_context(item)
            for item in context_validation
        ]
        validation_reference_tensor = torch.as_tensor(
            reference_validation, dtype=torch.float64
        )
        if args.mode == "analytical-grid":
            assert analytical_grid_result is not None
            validation_result = analytical_grid_result
            selected_stage = (
                "analytical_grid_selected_for_neural_stage"
                if analytical_grid_top_two
                else "plant_or_controller_gate_failure_diagnostic"
            )
            history.append(
                {
                    "stage": "analytical_grid_selected_diagnostic",
                    "epoch": 0.0,
                    "neural_stage_authorized": float(
                        bool(analytical_grid_top_two)
                    ),
                    **validation_result.aggregate,
                }
            )
        elif args.mode == "analytical-screen":
            assert observed_validation is not None
            validation_result = validate_actor(
                actor,
                stepper,
                validation_initial,
                validation_reference_tensor,
                channel_weights,
                reference_scale,
                projections,
                config,
                observed=observed_validation,
            )
            history.append(
                {
                    "stage": "analytical_screen",
                    "epoch": 0.0,
                    **validation_result.aggregate,
                }
            )
            if not bool(validation_result.aggregate["plant_fidelity_gate"]):
                selected_stage = "plant_fidelity_failure_diagnostic"
        else:
            assert frozen_oof_selection is not None
            plant_probe = validate_actor(
                actor,
                stepper,
                validation_initial,
                validation_reference_tensor,
                channel_weights,
                reference_scale,
                projections,
                config,
                observed=None,
                external_plant_fidelity_gate=bool(
                    frozen_oof_selection["plant_pass"]
                ),
                seed_offset=10000,
            )
            if not bool(plant_probe.aggregate["plant_fidelity_gate"]):
                validation_result = plant_probe
                selected_stage = "plant_fidelity_failure_no_training"
                history.append(
                    {
                        "stage": selected_stage,
                        "epoch": 0.0,
                        **validation_result.aggregate,
                    }
                )
            else:
                (
                    actor,
                    teacher_history,
                    teacher_result,
                    teacher_accepted,
                ) = train_teacher(
                    actor,
                    stepper,
                    fit_initial,
                    validation_initial,
                    fit_reference_tensor,
                    validation_reference_tensor,
                    observed_validation,
                    channel_weights,
                    reference_scale,
                    projections,
                    config,
                    external_plant_fidelity_gate=True,
                )
                history.extend(teacher_history)
                validation_result = teacher_result
                if teacher_accepted:
                    selected_stage = "structured_full_markov_teacher"
                    save_checkpoint(
                        output / "teacher_checkpoint.pt",
                        actor,
                        None,
                        config=config,
                        selected_stage=selected_stage,
                        input_hashes=input_hashes,
                        config_sha256=config_hash,
                    )
                else:
                    selected_stage = "teacher_non_responder_previous_analytical_retained"
                if teacher_accepted and config.wgan_epochs > 0:
                    (
                        actor,
                        critic,
                        wgan_history,
                        wgan_result,
                        wgan_accepted,
                    ) = train_wgan(
                        actor,
                        stepper,
                        fit_initial,
                        validation_initial,
                        fit_reference_tensor,
                        validation_reference_tensor,
                        observed_validation,
                        channel_weights,
                        reference_scale,
                        projections,
                        config,
                        external_plant_fidelity_gate=True,
                    )
                    history.extend(wgan_history)
                    if wgan_accepted:
                        selected_stage = "structured_full_markov_actor_wgan_gp"
                        validation_result = wgan_result
        display = validation_result.display

    pd.DataFrame(history).to_csv(
        output / "training_history.csv", index=False, encoding="utf-8-sig"
    )
    if validation_result is not None:
        pd.DataFrame(validation_result.contexts).to_csv(
            output / "validation_context_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        metrics: dict[str, Any] = validation_result.aggregate
    else:
        metrics = {
            "validation_performed": False,
            "checkpoint_selection_performed": False,
            "prefix_refit_final_epoch": int(args.prefix_epochs),
        }
    metrics.update(
        {
            "selected_stage": selected_stage,
            "teacher_updated_checkpoint_accepted": bool(teacher_accepted),
            "wgan_updated_checkpoint_accepted": bool(wgan_accepted),
            "controller_non_responder": bool(
                selected_stage
                in {
                    "plant_fidelity_failure_no_training",
                    "teacher_non_responder_previous_analytical_retained",
                }
            ),
            "channels": n_channels,
            "actuators_selected": int(stepper.actuator_dim),
            "actuators_q065_anchor": int(primary_k),
            "selected_target_quantile": float(config.target_quantile),
            "latent_dimension": int(stepper.q),
            "markov_dimension": int(stepper.state_dim),
            "elapsed_seconds": float(time.perf_counter() - started),
            "analytical_grid_candidate_count": len(analytical_grid_rows),
            "analytical_grid_top_two": analytical_grid_top_two,
            "neural_stage_authorized": bool(
                (
                    loaded_checkpoint.get("neural_stage_authorized", False)
                    if loaded_checkpoint is not None
                    else (
                        args.mode != "analytical-grid"
                        or analytical_grid_top_two
                    )
                )
            ),
        }
    )
    write_json(output / "metrics.json", metrics)
    rollout_payload = audited_rollout_payload(
        display,
        np.flatnonzero(np.asarray(effective_network["target_mask"], dtype=bool)),
        config,
        mode=args.mode,
    )
    rollout_payload.update(evaluation_seed_payload)
    if not np.array_equal(
        rollout_payload["free_standard_normal"],
        rollout_payload["controlled_standard_normal"],
    ):
        raise AssertionError("paired free/controlled standard-normal banks differ")
    save_rollout_npz(output / "paired_rollout.npz", rollout_payload)
    save_checkpoint(
        output / "selected_controller.pt",
        actor,
        critic if wgan_accepted else None,
        config=config,
        selected_stage=selected_stage,
        input_hashes=input_hashes,
        config_sha256=config_hash,
    )
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "mode": args.mode,
        "execution_class": config_payload["execution_class"],
        "final_future_array_opened": False,
        "outcome_based_checkpoint_selection": False,
        "interictal_reference_used_as_diffusion": False,
        "plant_diffusion_tuned_by_control_objective": False,
        "plant_diffusion_selected_before_control_ranking": True,
        "plant_multistep_drift_tuned_by_control_objective": False,
        "plant_multistep_drift_selected_before_control_ranking": True,
        "paired_free_control_noise": True,
        "method_label": METHOD_LABEL,
        "critic_role": "Kantorovich critic for empirical distribution discrepancy",
        "physical_input_map": "strict local selector",
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "display_evaluation_seed": int(DISPLAY_EVALUATION_SEED),
        "source_target_mask_q065_sha256": sha256_array(network["target_mask"]),
        "selected_target_mask_sha256": sha256_array(
            effective_network["target_mask"]
        ),
        "selected_target_quantile": config.target_quantile,
        "input_hashes": input_hashes,
        "output_hashes": {
            "run_config_sha256": config_hash,
            "metrics_sha256": sha256_file(output / "metrics.json"),
            "checkpoint_sha256": sha256_file(output / "selected_controller.pt"),
            "rollout_sha256": sha256_file(output / "paired_rollout.npz"),
        },
    }
    write_json(output / "provenance.json", provenance)
    build_manifest(output)
    print(json.dumps(json_ready(metrics), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
