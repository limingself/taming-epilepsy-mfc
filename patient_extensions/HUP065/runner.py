#!/usr/bin/env python
"""Fail-closed, resumable orchestration for the fresh HUP065 sparse rerun.

This file is the execution boundary, not a notebook.  Importing it never opens
patient data, imports the D-drive scientific modules, trains, evaluates, or
plots.  Static validation is pure Python.  Scientific phase implementations
are loaded lazily only after the run has an authenticated preflight receipt and the
caller supplies the explicit science authorization flag.

The outer phase has an additional human-created GO receipt and a one-shot
ledger.  No development phase is allowed to resolve or open run-03.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import hmac
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import sys
import tempfile
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
import zipfile


# This must precede every lazy import from the immutable HUP060 authority.
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.json"
SCIENCE_AUTHORIZATION_ENV = "HUP065_EXPECTED_SCIENCE_AUTHORIZATION"
OUTER_AUTHORIZATION_ENV = "HUP065_EXPECTED_OUTER_AUTHORIZATION"
OUTER_AUTHORIZATION_CONFIG_MARKER = f"ENV:{OUTER_AUTHORIZATION_ENV}"
PREFLIGHT_PHASE = "P0_PREFLIGHT"
FREEZE_PHASE = "WGAN40_CTX5_SELECT_CTX6_VETO"
TEACHER_STAGE_SPECS = (
    ("S0_INITIAL_0_100", 100, 0, 5, 3.0e-4, "initial", 0.25, "all_pre_markov_parameters"),
    ("S1_BALANCED_100_250", 150, 100, 5, 2.0e-4, "initial_balanced_channel_weights", 0.25, "all_pre_markov_parameters"),
    ("S2S_SAFE_250_450", 200, 250, 5, 1.0e-4, "balanced_safe", 0.25, "all_pre_markov_parameters"),
    ("S3_COVARIANCE_450_650", 200, 450, 5, 1.0e-4, "covariance", 0.25, "deviation_only_freeze_common"),
    ("S4_MARKOV_ALPHA100_650_770", 120, 650, 10, 2.0e-4, "covariance", 1.0, "markov_only"),
    ("S5_SMOOTHA_770_870", 100, 770, 10, 1.0e-4, "covariance_smoothA", 1.0, "markov_only"),
    ("S6_PRECISION_ALL_PARAMS_870_1050", 180, 870, 10, 5.0e-5, "covariance", 1.0, "all_actor_parameters"),
)
TEACHER_PHASES = tuple(item[0] for item in TEACHER_STAGE_SPECS)
PHASE_ORDER = (
    PREFLIGHT_PHASE,
    "DEV_MATERIALIZE",
    "P1_LORO_GRAPHS",
    "P2_LORO_RC_SDE",
    "ANALYTICAL_TOP1",
    "ALLDEV_REFIT",
    *TEACHER_PHASES,
    FREEZE_PHASE,
    "OUTER",
)
DEVELOPMENT_SCIENCE_PHASES = frozenset(PHASE_ORDER[1:-1])
PHASE_OUTPUT_ROOT_DIRECTORY = "p"

# Exhaustive relative artifact inventory for the active runner/backend dispatch.
# Logical phase IDs and every scientific artifact basename remain unchanged;
# only their physical parent container is shortened for Windows path headroom.
_RUNNER_PHASE_ARTIFACTS = (
    "authority_after.json",
    "backend_receipt.json",
    "COMPLETE.json",
)
FORMAL_PHASE_ARTIFACTS: Mapping[str, tuple[str, ...]] = {
    "P0_PREFLIGHT": (
        "authority_before.json",
        "canonical_lock_receipts.json",
        "code_reference_lock_receipts.json",
        "source_adaptation_audit_receipt.json",
        "shared_protocol_receipt.json",
        "split_receipt.json",
        "raw_archive_stat_and_central_directory_only.json",
        "COMPLETE.json",
    ),
    "DEV_MATERIALIZE": (
        "development_arrays.npz",
        "role_ledger.json",
        "provenance.json",
        *_RUNNER_PHASE_ARTIFACTS,
    ),
    "P1_LORO_GRAPHS": (
        "leave-run-01-out.npz",
        "leave-run-02-out.npz",
        "part1_receipt.json",
        *_RUNNER_PHASE_ARTIFACTS,
    ),
    "P2_LORO_RC_SDE": (
        "fold_part1_selection_networks/leave-run-01-out.npz",
        "fold_part1_selection_networks/leave-run-02-out.npz",
        "fold_part2_plant_adjacencies/leave-run-01-out.npz",
        "fold_part2_plant_adjacencies/leave-run-02-out.npz",
        "part2_grid_fold_metrics.csv",
        "part2_grid_context_metrics.csv",
        "diffusion_alpha_loro.csv",
        "diffusion_alpha_loro_fold_summary.csv",
        "diffusion_alpha_loro_two_level_ranking.csv",
        "fold_models/leave-run-01-out.joblib",
        "fold_models/leave-run-02-out.joblib",
        "rolling_block_loro_fold_summary.csv",
        "rolling_block_loro_channel_metrics.csv",
        "selected_config.json",
        *_RUNNER_PHASE_ARTIFACTS,
    ),
    "ANALYTICAL_TOP1": (
        "analytical_fold_summary.csv",
        "analytical_channel_context_metrics.csv",
        "analytical_ranking.csv",
        "frozen_top1.json",
        *_RUNNER_PHASE_ARTIFACTS,
    ),
    "ALLDEV_REFIT": (
        "selected_model.joblib",
        "network.npz",
        "part2_plant_adjacency.npz",
        "frozen_mask.csv",
        "refit_receipt.json",
        *_RUNNER_PHASE_ARTIFACTS,
    ),
    **{
        phase: (
            "training/frozen_actor.pt",
            "training/training_history.csv",
            "training/summary.json",
            "stage_execution.json",
            *_RUNNER_PHASE_ARTIFACTS,
        )
        for phase in TEACHER_PHASES
    },
    "WGAN40_CTX5_SELECT_CTX6_VETO": (
        "wgan40_training_history.csv",
        "wgan40_training_summary.json",
        "selected_controller.pt",
        "ctx5_selected_rollout.npz",
        "ctx6_channel_metrics.csv",
        "ctx6_rollout.npz",
        "final_development_freeze.json",
        *_RUNNER_PHASE_ARTIFACTS,
    ),
    "OUTER": (
        "ACCESS_STARTED.json",
        "run03_sealed_segments.npz",
        "run03_payload_access_ledger.json",
        "run03_all64_channel_context_bank_metrics.csv",
        "run03_frozen_display_context07_bank0_rollout.npz",
        "outer_gate_vectors.json",
        "outer_safety_vectors.json",
        "hup065_run03_context07_ofrc_candidate.npz",
        "COMMON_RENDERER_BINDING_PENDING_AUDIT.json",
        "run03_retrospective_report.json",
        *_RUNNER_PHASE_ARTIFACTS,
    ),
}


class ContractError(RuntimeError):
    """The frozen contract was violated; execution must stop."""


class ExistingOutputError(ContractError):
    """A named output already exists and may not be overwritten."""


class ScienceAuthorizationError(ContractError):
    """A scientific or outer-data action lacks its explicit authorization."""


@dataclass(frozen=True)
class SparseCandidate:
    candidate_id: str
    actuator_fraction: float
    centrality_quantile: float
    graph_diffusion_time: float
    analytical_base_gain: float


@dataclass(frozen=True)
class CandidateSummary:
    candidate_id: str
    integrity_pass: bool
    energy_pass: bool
    worst_fold_full_gate_b_count: int
    worst_fold_mean_time_plus_occupation_w1: float
    actuator_count: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_uuid_token() -> str:
    """Return 128 cryptographically random bits in 22 Windows-safe chars."""

    return base64.urlsafe_b64encode(secrets.token_bytes(16)).decode("ascii").rstrip("=")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, payload: Any) -> None:
    """Write one file atomically; refuse to replace an existing artifact."""

    if path.exists():
        raise ExistingOutputError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Do not repeat a potentially long destination filename in the temporary
    # basename. The URL-safe token retains all 128 UUID bits.
    temporary = path.with_name(f".j-{_short_uuid_token()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                ).encode("utf-8")
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload = load_json(path.resolve())
    if not isinstance(payload, dict):
        raise ContractError("config root must be an object")
    return payload


def configure_runtime(config: Mapping[str, Any], *, create: bool) -> Path:
    """Redirect every known cache away from immutable source/data trees."""

    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    runtime = Path(str(config["runtime_root"])).resolve()
    if create:
        runtime.mkdir(parents=True, exist_ok=True)
    cache_map = {
        "MNE_HOME": runtime / "mne_home",
        "_MNE_FAKE_HOME_DIR": runtime / "mne_home",
        "MNE_DATA": runtime / "mne_data",
        "MPLCONFIGDIR": runtime / "matplotlib",
        "XDG_CACHE_HOME": runtime / "xdg_cache",
        "JOBLIB_TEMP_FOLDER": runtime / "joblib",
        "TMPDIR": runtime / "tmp",
        "TEMP": runtime / "tmp",
        "TMP": runtime / "tmp",
    }
    for key, directory in cache_map.items():
        os.environ[key] = str(directory)
        if create:
            directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")
    if not sys.dont_write_bytecode or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ContractError("bytecode suppression is not active")
    return runtime


def _close(first: float, second: float, atol: float = 1.0e-12) -> bool:
    return math.isclose(float(first), float(second), rel_tol=0.0, abs_tol=atol)


def validate_static_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate frozen scalars and roles without touching external files."""

    if config.get("schema_version") != "hup065-hup060-sparse-runner-v1.0":
        raise ContractError("unsupported config schema")
    if config.get("status") != (
        "IMPLEMENTATION_READY_AFTER_PROTOCOL_CONFORMANCE_REPAIR_NO_ACTIVE_SCIENCE_RUN"
    ):
        raise ContractError(
            "config repaired-implementation/no-active-science-run status is absent"
        )
    if config.get("subject") != "HUP065" or int(config.get("expected_channels", -1)) != 64:
        raise ContractError("this runner is bound only to HUP065/64 channels")
    if config["shared_protocol"].get("executed_sha256") != (
        "a59141b4e239eb6676c915108aaa82ba330cf9574a40158c4702479f6d2f1399"
    ):
        raise ContractError("executed shared-protocol provenance binding changed")
    required_d_plant_hashes = {
        "causal_particle_rollout": "47b3328091a85f851fc8e42275c24847564f2795c7292f53f51c37746832305c",
        "causal_riccati": "748567f5fb616a641c3916a69dbb9385cf2a3cd0e64d890b49301bb3244c3ae2",
    }
    executed_locks = config.get("public_portability", {}).get(
        "executed_canonical_lock_sha256", {}
    )
    for name, expected_sha in required_d_plant_hashes.items():
        if str(executed_locks.get(name, "")).lower() != expected_sha:
            raise ContractError(f"executed D-plant provenance binding changed: {name}")

    root = Path(str(config["code_root"])).resolve()
    if root != ROOT:
        raise ContractError(f"code root mismatch: {root} != {ROOT}")
    if root.name not in {"HUP065_hup060_sparse_rerun_v1", "HUP065"}:
        raise ContractError("code root does not match an audited private/public layout")
    run_root = Path(str(config["default_run_root"])).resolve()
    try:
        run_root.relative_to(root)
    except ValueError as error:
        raise ContractError("science run root must remain under the isolated code root") from error

    split = config["split"]
    if split["outer_ictal_run"] != "run-03" or split["outer_reference_run"] != "run-03":
        raise ContractError("outer run must be run-03 for both ictal and reference roles")
    if split["final_refit_runs"] != ["run-01", "run-02"]:
        raise ContractError("final refit runs changed")
    if split["reference_train_runs"] != ["run-01", "run-02"]:
        raise ContractError("reference training runs changed")
    if list(map(float, split["reference_train_absolute_seconds_half_open"])) != [30.0, 70.0]:
        raise ContractError("reference training window must be absolute [30,70)")
    if list(map(float, split["outer_reference_absolute_seconds_half_open"])) != [75.0, 115.0]:
        raise ContractError("outer reference window must be absolute [75,115)")
    if list(map(float, split["outer_ictal_absolute_seconds_half_open"])) != [120.0, 152.0]:
        raise ContractError("outer ictal window must be absolute [120,152)")
    if split["actor_fit_context_indices"] != [0, 1, 2, 3, 4]:
        raise ContractError("actor fit contexts changed")
    if int(split["checkpoint_selection_context_index"]) != 5:
        raise ContractError("ctx5 must select checkpoint epoch only")
    if int(split["terminal_veto_context_index"]) != 6:
        raise ContractError("ctx6 must be terminal veto only")
    if int(split["unused_development_context_index"]) != 7:
        raise ContractError("development ctx7 must remain unused")
    expected_boundaries = list(split["development_context_boundaries_samples"])
    if expected_boundaries != [512, 1573, 2633, 3694, 4754, 5815, 6875, 7936]:
        raise ContractError("outer/development context boundaries changed")
    if len(split["outer_preview_windows"]) != 8:
        raise ContractError("all eight outer contexts must be frozen before opening")
    for index, window in enumerate(split["outer_preview_windows"]):
        if window.get("window_id") != f"outer-context-{index:02d}":
            raise ContractError("outer context identity/order changed")
        bounds = [
            int(window["context_start_sample"]), int(window["context_stop_sample"]),
            int(window["future_start_sample"]), int(window["future_stop_sample"]),
        ]
        if not (0 <= bounds[0] < bounds[1] == bounds[2] < bounds[3] <= 8192):
            raise ContractError(f"invalid frozen outer window: {window}")
        if bounds[1] - bounds[0] != 256 or bounds[3] - bounds[2] != 256:
            raise ContractError("outer context/future must each contain 256 samples")
        if bounds[1] != expected_boundaries[index]:
            raise ContractError("outer context boundary differs from the preregistration")

    reference_rule = split["loro_reference_rule"]
    if reference_rule != {
        "fold_fit_reference": "training_run_preictal_30_70_only",
        "leftout_validation_reference": "leftout_run_preictal_30_70_only",
        "leftout_reference_may_enter_fit": False,
    }:
        raise ContractError("LORO fit/leftout reference roles changed")
    all_dev_reference = split["all_development_reference_path_split"]
    if all_dev_reference != {
        "deterministic_run_order": ["run-01", "run-02"],
        "path_duration_samples": 256,
        "per_run_fit_path_indices_half_open": [0, 15],
        "per_run_ctx5_validation_path_indices_half_open": [15, 30],
        "per_run_unused_tail_path_indices_half_open": [30, 40],
        "pool_construction": "fit_concat_each_run_paths0_15_ctx5_concat_each_run_paths15_30",
        "outcome_dependent_split": False,
        "same_tensor_for_fit_and_validation": False,
    }:
        raise ContractError("all-development fit/ctx5 reference split changed")

    fractions = [float(value) for value in config["analytical_screen"]["actuator_fraction_grid"]]
    taus = [float(value) for value in config["analytical_screen"]["graph_diffusion_time_grid"]]
    gains = [float(value) for value in config["analytical_screen"]["analytical_base_gain_grid"]]
    if fractions != [0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]:
        raise ContractError("sparse fraction grid changed")
    if taus != [0.0, 0.05, 0.10, 0.20, 0.50]:
        raise ContractError("graph diffusion grid changed")
    if gains != [0.35, 0.45, 0.668]:
        raise ContractError("analytical gain grid changed")
    expected_grid = len(fractions) * len(taus) * len(gains)
    if expected_grid != 120 or int(config["analytical_screen"]["candidate_count"]) != 120:
        raise ContractError("analytical grid must contain 120 arms")
    if int(config["analytical_screen"]["freeze_unique_top_k"]) != 1:
        raise ContractError("analytical LORO must freeze one unique top1")
    if bool(config["analytical_screen"]["neural_training_during_screen"]):
        raise ContractError("analytical screen may not train a neural controller")
    if int(config["analytical_screen"].get("evaluation_crn_banks", -1)) != 3:
        raise ContractError("evaluation must use exactly three preregistered CRN banks")

    part1 = config["part1"]
    if not _close(part1.get("plv_window_seconds"), 4.0):
        raise ContractError("Part-I multiband selection PLV window must remain 4 s")
    if not _close(part1.get("plv_overlap"), 0.50):
        raise ContractError("Part-I multiband selection PLV overlap changed")

    part2 = config["part2"]
    calculated_part2 = math.prod(
        len(part2[key])
        for key in (
            "reservoir_sizes", "spectral_radii", "leak_rates",
            "input_scales", "delay_sets_samples", "ridge_alphas",
        )
    )
    if calculated_part2 != 144 or int(part2["candidate_count"]) != 144:
        raise ContractError("canonical Part-II grid must contain 144 candidates")
    if int(part2.get("context_samples", -1)) != 512:
        raise ContractError("canonical Part-II context must contain 512 samples")
    if int(part2.get("validation_windows", -1)) != 6:
        raise ContractError("canonical Part-II direct validation must use six windows")
    if list(map(int, part2.get("validation_horizons_samples", []))) != [16, 32, 64, 128]:
        raise ContractError("canonical Part-II validation horizons changed")
    if str(part2.get("fold_aggregation")) != "worst_fold":
        raise ContractError("Part-II LORO selection must be worst-fold first")
    expected_plant = {
        "plant_network_source": "broadband_windowed_plv_median_not_part1_multiband_graph",
        "plant_plv_window_seconds": 2.0,
        "plant_plv_window_overlap_fraction": 0.50,
        "plant_graph_density": 0.10,
        "part1_graph_may_be_reused_as_plant_graph": False,
    }
    for key, expected in expected_plant.items():
        observed = part2.get(key)
        if isinstance(expected, float):
            if not _close(observed, expected):
                raise ContractError(f"Part-II plant graph field changed: {key}")
        elif observed != expected:
            raise ContractError(f"Part-II plant graph field changed: {key}")
    if list(map(int, part2.get("rolling_block_candidates_samples", []))) != [4, 8, 12, 16, 24, 32]:
        raise ContractError("Part-II rolling block candidates changed")
    if int(part2.get("rolling_display_samples", -1)) != 128:
        raise ContractError("Part-II rolling display horizon changed")
    if not _close(part2.get("rolling_selection_min_correlation"), 0.70):
        raise ContractError("Part-II rolling correlation gate changed")
    if not _close(part2.get("rolling_selection_max_nrmse"), 0.80):
        raise ContractError("Part-II rolling NRMSE gate changed")
    if part2.get("rolling_no_eligible_block") != "NO_GO_before_part3":
        raise ContractError("Part-II rolling screen no longer fails closed")
    if not _close(part2.get("effective_diffusion_multiplier"), 0.79451175):
        raise ContractError("D-canonical Part-II/III diffusion multiplier changed")

    training = config["part3"]["training"]
    if int(training["teacher_epochs_total"]) != 1050 or int(training["wgan_epochs"]) != 40:
        raise ContractError("canonical cumulative teacher1050/WGAN40 schedule changed")
    if int(training.get("s3_to_s4_modern_actor_initialization_seed", -1)) != 20260921:
        raise ContractError("S3-to-S4 modern Actor initialization seed changed")
    if int(training.get("teacher_rollout_base_seed", -1)) != 20260921:
        raise ContractError("teacher rollout base seed changed")
    if training.get("teacher_rollout_seed_rule") != "20260921+1009*(absolute_epoch+1)":
        raise ContractError("teacher rollout seed rule changed")
    if int(training.get("teacher_validation_noise_seed", -1)) != 20260922:
        raise ContractError("teacher validation noise seed changed")
    if bool(training.get("discarded_robust_branch_executed", True)):
        raise ContractError("discarded S2R robust branch must never execute")
    if tuple(training["teacher_formal_stage_ids"]) != TEACHER_PHASES:
        raise ContractError("formal teacher stage identities changed")
    observed_teacher = training["teacher_stages"]
    if len(observed_teacher) != len(TEACHER_STAGE_SPECS):
        raise ContractError("formal teacher schedule must contain seven stages")
    for observed, expected in zip(observed_teacher, TEACHER_STAGE_SPECS):
        (
            stage_id,
            epochs,
            offset,
            validation_every,
            learning_rate,
            objective,
            actuator_alpha,
            train_scope,
        ) = expected
        exact = {
            "stage_id": stage_id,
            "epochs": epochs,
            "epoch_offset": offset,
            "validation_every": validation_every,
            "objective": objective,
            "train_scope": train_scope,
        }
        for key, expected_value in exact.items():
            if observed.get(key) != expected_value:
                raise ContractError(f"teacher {stage_id} field {key} changed")
        if not _close(observed["learning_rate"], learning_rate):
            raise ContractError(f"teacher {stage_id} learning rate changed")
        if not _close(observed["actuator_alpha"], actuator_alpha):
            raise ContractError(f"teacher {stage_id} actuator alpha changed")
    if sum(int(stage["epochs"]) for stage in observed_teacher) != 1050:
        raise ContractError("teacher stages do not sum to 1050 epochs")
    for left, right in zip(observed_teacher, observed_teacher[1:]):
        if int(left["epoch_offset"]) + int(left["epochs"]) != int(right["epoch_offset"]):
            raise ContractError("teacher stages are not cumulative and contiguous")
    if not bool(config["part3"]["fresh_initialization_required"]):
        raise ContractError("fresh Actor initialization must be required")
    plant_adapter = config["part3"]["plant_adapter"]
    if plant_adapter != {
        "causal_particle_rollout": "D_canonical_hash_locked",
        "causal_riccati": "D_canonical_hash_locked",
        "graph_diffusion_time_role": "TorchGraphRCSDE_plant_input",
        "local_control_map_override": False,
        "feedback_only_heat_map_substitution": False,
        "stepper": "D_canonical_FrozenIctalGraphRCBatchStepper",
        "local_multistep_wrapper_used": False,
    }:
        raise ContractError("D-canonical plant adapter semantics changed")
    if config["part3"]["checkpoint_selection"]["may_select_arm"] is not False:
        raise ContractError("ctx5 may not select or replace the analytical top1 arm")
    if config["part3"]["terminal"]["may_rank_rescue_or_replace"] is not False:
        raise ContractError("ctx6 may not rank, rescue, or replace the arm")
    handoff = training["formal_wgan_handoff"]
    expected_handoff = {
        "critic_pretrain_noise_rule": "seed+101*(bank+1)",
        "critic_pretrain_gp_seed_rule": "seed+50000+step",
        "joint_epoch_noise_rule": "seed+1009*(epoch+1)",
        "joint_epoch_gp_seed_rule": "seed+100000*(epoch+1)+critic_step",
        "actor_gradient_clip": 1.0,
        "critic_gradient_clip": 5.0,
        "critic_adam_betas": [0.0, 0.9],
        "epoch_zero_eligible": False,
        "law_noninferiority_required": True,
        "empty_noninferior_pool": (
            "retain_frozen_s6_teacher_after_full_wgan40_and_record_NO_ACCEPTED_WGAN"
        ),
        "arm_reselection_allowed": False,
    }
    if handoff != expected_handoff:
        raise ContractError("formal WGAN handoff changed")
    terminal = config["part3"]["terminal"]
    if not (
        int(terminal["context_index"]) == 6
        and terminal["role"] == "veto_only"
        and terminal["finite_and_gate_c_all_contexts_required"] is True
        and terminal["aggregate_time_w1_improvement_required"] is True
        and terminal["aggregate_occupation_w1_improvement_required"] is True
        and int(terminal["minimum_full_six_gate_b_channel_count"]) == 1
    ):
        raise ContractError("ctx6 terminal veto rule changed")

    energy = config["energy"]
    expected_energy = (13.0 / 36.0) * 64.0 * (0.405**2)
    if not _close(energy["total_energy_cap"], expected_energy):
        raise ContractError("fixed total energy cap changed")
    if not _close(energy["total_l2_rms_cap"], math.sqrt(expected_energy)):
        raise ContractError("fixed total L2 RMS cap changed")
    if not _close(energy["per_actuator_rms_max"], 0.405):
        raise ContractError("per-actuator RMS cap changed")
    if not _close(energy["peak_control_max"], 1.8):
        raise ContractError("peak cap changed")
    if not _close(energy["saturation_fraction_max_exclusive"], 0.01):
        raise ContractError("saturation cap changed")
    if energy.get("hard_projection_or_rescaling_allowed") is not False:
        raise ContractError("energy projection/rescaling must remain forbidden")
    if config["part3"]["actor"].get(
        "runtime_control_projection_or_rescaling"
    ) is not False:
        raise ContractError("Actor runtime control projection must remain disabled")
    gates = config["gates"]["per_channel"]
    expected_gates = {
        "minimum_time_w1_relative_reduction": 0.10,
        "minimum_occupation_w1_relative_reduction": 0.10,
        "time_w1_controlled_reference_max": 0.35,
        "occupation_w1_controlled_reference_max": 0.25,
        "controlled_reference_mean_abs_error_max": 0.10,
        "controlled_reference_symmetric_sd_ratio_max": 2.0,
    }
    if gates != expected_gates:
        raise ContractError("six-component Gate-B thresholds changed")

    phases = [str(item["id"]) for item in config["phases"]]
    if tuple(phases) != PHASE_ORDER:
        raise ContractError("phase order changed")
    expected_dependencies = {
        phase_id: ([] if index == 0 else [PHASE_ORDER[index - 1]])
        for index, phase_id in enumerate(PHASE_ORDER)
    }
    for phase in config["phases"]:
        if list(phase.get("depends_on", [])) != expected_dependencies[str(phase["id"])]:
            raise ContractError(f"phase dependency changed: {phase['id']}")
    if config["outer_go"]["required_authorization_text"] != OUTER_AUTHORIZATION_CONFIG_MARKER:
        raise ContractError("outer authorization environment binding changed")
    if config["outer_go"].get("development_freeze_phase_id") != FREEZE_PHASE:
        raise ContractError("outer GO is not bound to the final development freeze")
    expected_outer_ids = [f"outer-context-{index:02d}" for index in range(8)]
    if config["outer_go"].get("aggregate_window_ids") != expected_outer_ids:
        raise ContractError("outer aggregation must include all eight preregistered contexts")
    if config["outer_go"].get("display_window_id") != "outer-context-07":
        raise ContractError("outer representative display must remain context07")
    if int(config["outer_go"].get("display_crn_bank", -1)) != 0:
        raise ContractError("outer representative display must remain CRN bank0")
    if config["outer_go"].get("reference_and_ictal_preprocessing_state") != (
        "independent_loader_instances"
    ):
        raise ContractError("outer ictal/reference preprocessing state is not independent")
    if config["outer_go"].get("renderer_execution_in_outer") is not False:
        raise ContractError("OUTER may not execute the common renderer")
    safety = config["runtime_safety"]
    required_true = (
        "python_dont_write_bytecode_required", "redirect_mne_cache",
        "redirect_matplotlib_cache", "phase_atomic_publish",
        "authority_before_after_exact_hash_invariant",
    )
    if not all(bool(safety[key]) for key in required_true):
        raise ContractError("one or more runtime safety controls are disabled")
    if any(
        bool(safety[key])
        for key in (
            "static_self_test_may_read_d_drive",
            "static_self_test_may_read_patient_data",
            "static_self_test_may_run_science",
            "overleaf_write_allowed",
        )
    ):
        raise ContractError("static/Overleaf guardrail changed")
    if safety.get("runtime_control_projection_or_rescaling_allowed") is not False:
        raise ContractError("runtime projection guardrail changed")
    if safety.get("phase_output_root_directory") != PHASE_OUTPUT_ROOT_DIRECTORY:
        raise ContractError("short Windows physical phase-root directory changed")
    if safety.get("phase_staging_directory_pattern") != ".sNN-{random128_base64url22}":
        raise ContractError("short Windows phase-staging scheme changed")
    if safety.get("atomic_json_temp_pattern") != ".j-{random128_base64url22}":
        raise ContractError("short Windows atomic-JSON scheme changed")
    if safety.get("random_token_source") != "secrets.token_bytes(16)":
        raise ContractError("atomic publication token is not full-entropy random128")
    if int(safety.get("formal_path_characters_max_exclusive", -1)) != 200:
        raise ContractError("formal Windows path budget changed")

    return {
        "subject": "HUP065",
        "channels": 64,
        "part2_candidates": calculated_part2,
        "analytical_candidates": expected_grid,
        "energy_cap": expected_energy,
        "teacher_epochs_total": 1050,
        "teacher_stage_ids": list(TEACHER_PHASES),
        "wgan_epochs": 40,
        "phase_order": list(PHASE_ORDER),
    }


def build_sparse_grid(config: Mapping[str, Any]) -> tuple[SparseCandidate, ...]:
    screen = config["analytical_screen"]
    candidates: list[SparseCandidate] = []
    for fraction, tau, gain in itertools.product(
        screen["actuator_fraction_grid"],
        screen["graph_diffusion_time_grid"],
        screen["analytical_base_gain_grid"],
    ):
        fraction = float(fraction)
        quantile = round(1.0 - fraction, 12)
        tau = float(tau)
        gain = float(gain)
        candidate_id = f"f{fraction:.2f}_q{quantile:.2f}_tau{tau:.2f}_gain{gain:.3f}"
        candidates.append(
            SparseCandidate(candidate_id, fraction, quantile, tau, gain)
        )
    identifiers = [item.candidate_id for item in candidates]
    if len(candidates) != 120 or len(set(identifiers)) != 120:
        raise ContractError("sparse candidate grid is not exactly 120 unique arms")
    return tuple(candidates)


def linear_quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered or not all(math.isfinite(value) for value in ordered):
        raise ContractError("centrality values must be non-empty and finite")
    if not 0.0 <= quantile <= 1.0:
        raise ContractError("quantile must lie in [0,1]")
    position = (len(ordered) - 1) * float(quantile)
    lower, upper = math.floor(position), math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def strict_quantile_mask(values: Sequence[float], actuator_fraction: float) -> tuple[bool, ...]:
    if not 0.0 < actuator_fraction < 1.0:
        raise ContractError("actuator fraction must lie strictly inside (0,1)")
    threshold = linear_quantile(values, round(1.0 - actuator_fraction, 12))
    mask = tuple(float(value) > threshold for value in values)
    if not any(mask):
        raise ContractError("strict quantile produced an empty mask")
    maximum_direct = math.floor(0.80 * len(mask) + 1.0e-12)
    if sum(mask) > maximum_direct or len(mask) - sum(mask) < len(mask) - maximum_direct:
        raise ContractError("strict quantile mask violates the >=20% indirect bound")
    return mask


def candidate_rank_key(summary: CandidateSummary) -> tuple[Any, ...]:
    if summary.worst_fold_full_gate_b_count < 0 or summary.actuator_count < 1:
        raise ContractError("candidate counts must be positive")
    if not math.isfinite(summary.worst_fold_mean_time_plus_occupation_w1):
        raise ContractError("candidate W1 summary must be finite")
    feasible = summary.integrity_pass and summary.energy_pass
    return (
        0 if feasible else 1,
        -int(summary.worst_fold_full_gate_b_count),
        float(summary.worst_fold_mean_time_plus_occupation_w1),
        int(summary.actuator_count),
        str(summary.candidate_id),
    )


def control_feasibility(
    controls: Sequence[Sequence[Sequence[float]]],
    *,
    amplitude_limit: float = 1.8,
    per_actuator_rms_max: float = 0.405,
    total_energy_cap: float = 3.7908,
    saturation_fraction_max_exclusive: float = 0.01,
) -> dict[str, Any]:
    """Evaluate the frozen energy gate for a (particle,time,actuator) cube."""

    if not controls or not controls[0] or not controls[0][0]:
        raise ContractError("controls must be a non-empty rank-three cube")
    particles = len(controls)
    time_steps = len(controls[0])
    actuators = len(controls[0][0])
    sums = [0.0] * actuators
    peak = 0.0
    saturated = 0
    total_values = particles * time_steps * actuators
    for particle in controls:
        if len(particle) != time_steps:
            raise ContractError("ragged control particle axis")
        for row in particle:
            if len(row) != actuators:
                raise ContractError("ragged control actuator axis")
            for index, raw in enumerate(row):
                value = float(raw)
                if not math.isfinite(value):
                    raise ContractError("non-finite control value")
                magnitude = abs(value)
                sums[index] += value * value
                peak = max(peak, magnitude)
                saturated += int(magnitude >= amplitude_limit)
    denominator = particles * time_steps
    actuator_rms = [math.sqrt(value / denominator) for value in sums]
    total_energy = sum(value / denominator for value in sums)
    saturation_fraction = saturated / total_values
    passes = (
        peak <= amplitude_limit
        and max(actuator_rms) <= per_actuator_rms_max
        and total_energy <= total_energy_cap
        and saturation_fraction < saturation_fraction_max_exclusive
    )
    return {
        "pass": passes,
        "particles": particles,
        "time_steps": time_steps,
        "actuators": actuators,
        "actuator_rms": actuator_rms,
        "maximum_actuator_rms": max(actuator_rms),
        "total_energy": total_energy,
        "total_l2_rms": math.sqrt(total_energy),
        "peak": peak,
        "saturation_fraction": saturation_fraction,
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def assert_output_path(config: Mapping[str, Any], path: Path) -> None:
    code_root = Path(str(config["code_root"])).resolve()
    resolved = path.resolve()
    if not _is_within(resolved, code_root):
        raise ContractError(f"output escapes isolated root: {resolved}")
    for authority in config["canonical"]["authority_roots"]:
        if _is_within(resolved, Path(str(authority))):
            raise ContractError("output overlaps immutable HUP060 authority")


def assert_science_input_path(
    config: Mapping[str, Any], path: Path, *, phase: str
) -> None:
    """Reject old artifacts and any early resolution of the sealed run."""

    resolved = path.resolve()
    raw_root = Path(str(config["raw_data"]["zip_root"])).resolve()
    run_root = Path(str(config["default_run_root"])).resolve()
    allowed = _is_within(resolved, raw_root) or _is_within(resolved, run_root)
    if not allowed:
        raise ContractError(f"science input is outside raw-data/run roots: {resolved}")
    lowered = str(resolved).casefold()
    forbidden_markers = (
        "old_model", "old_checkpoint", "ofrc", "paired_rollout",
        "final_plot_ready", "interictal", "frozen_actor",
    )
    if any(marker in lowered for marker in forbidden_markers):
        raise ContractError(f"forbidden legacy/role input: {resolved}")
    # Serialized scientific artifacts are legal only when they were freshly
    # published inside this run root.  This blocks every historical joblib/PT
    # checkpoint while allowing resumable P2/Actor products from this run.
    if not _is_within(resolved, run_root) and resolved.suffix.casefold() in {
        ".joblib", ".pt", ".pth", ".ckpt", ".pickle", ".pkl",
    }:
        raise ContractError(f"serialized artifact is not fresh run-local input: {resolved}")
    if phase != "OUTER" and "run-03" in lowered:
        raise ContractError("run-03 may not be resolved before OUTER")


def verify_file_lock(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"locked file is absent: {path}")
    observed = sha256_file(path)
    if observed != str(expected_sha256).lower():
        raise ContractError(
            f"locked file changed: {path}; expected={expected_sha256}, observed={observed}"
        )
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": observed}


def verify_shared_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = config["shared_protocol"]
    protocol_receipt = verify_file_lock(
        Path(str(binding["path"])), str(binding["sha256"])
    )
    module_receipt = verify_file_lock(
        Path(str(binding["common_module_path"])), str(binding["common_module_sha256"])
    )
    protocol = load_json(Path(str(binding["path"])))
    authority = protocol["authority_freeze"]
    if Path(str(authority["allowed_output_parent"])).resolve() != ROOT.parent:
        raise ContractError("shared protocol output parent changed")
    if ROOT.name not in authority["allowed_new_output_directory_names"]:
        raise ContractError("isolated HUP065 root is absent from shared allowlist")
    subject = protocol["subjects"]["HUP065"]
    if int(subject["expected_channels"]) != 64:
        raise ContractError("shared protocol HUP065 channel count changed")
    if subject["development_runs"] != ["run-01", "run-02"]:
        raise ContractError("shared protocol HUP065 development runs changed")
    if subject["sealed_test_run"] != "run-03":
        raise ContractError("shared protocol HUP065 sealed run changed")
    if subject["reference_train_window_seconds_half_open"] != [30.0, 70.0]:
        raise ContractError("shared protocol reference training role changed")
    if subject["reference_test_window_seconds_half_open"] != [75.0, 115.0]:
        raise ContractError("shared protocol reference outer role changed")
    implementation = protocol["implementation_stages"]
    if implementation.get("canonical_runtime_module_binding") != (
        "single hash-and-path-validated live module identity; no purge/reimport while model objects exist"
    ):
        raise ContractError("shared canonical live-module identity contract changed")
    if int(implementation["analytical_screen"]["freeze_unique_top_k"]) != 1:
        raise ContractError("shared protocol no longer freezes analytical top1")
    if implementation["final_development_refit"]["context_5_may_select_or_replace_arm"]:
        raise ContractError("shared protocol permits ctx5 arm selection")
    if implementation["final_development_refit"]["context_6_may_rank_or_rescue_candidate"]:
        raise ContractError("shared protocol permits ctx6 rescue")
    shared_part1 = protocol["common_model"]["part1"]
    if (
        not _close(shared_part1.get("plv_window_seconds"), 4.0)
        or not _close(shared_part1.get("plv_window_overlap_fraction"), 0.50)
        or shared_part1.get("network_role")
        != "actuator_ranking_and_sparse_mask_selection_only"
    ):
        raise ContractError("shared Part-I multiband selection graph contract changed")
    shared_part2 = protocol["common_model"]["part2"]
    if (
        shared_part2.get("plant_network_source")
        != "broadband_windowed_plv_median_not_part1_multiband_graph"
        or not _close(shared_part2.get("plant_plv_window_seconds"), 2.0)
        or shared_part2.get("part1_graph_may_be_reused_as_plant_graph") is not False
        or int(shared_part2.get("context_samples", -1)) != 512
        or int(shared_part2.get("direct_validation_windows", -1)) != 6
        or int(shared_part2.get("direct_validation_maximum_horizon_samples", -1)) != 128
        or shared_part2.get("direct_validation_positions")
        != "D_canonical_forecast_positions"
        or shared_part2.get("loro_candidate_aggregation")
        != "minimize_worst_fold_then_mean_then_candidate_id"
        or shared_part2.get("diffusion_alpha_loro_aggregation")
        != "mean_swd_ratio_over_six_windows_per_fold_then_minimize_worst_fold_then_overall_mean_then_alpha"
        or shared_part2.get("adjacency_binding")
        != {
            "raw_part2_plant_adjacency": "model.adjacency_input",
            "normalized_model_adjacency": "model.adjacency",
            "world_adjacency": "model.adjacency",
        }
    ):
        raise ContractError("shared Part-II graph/context/LORO contract changed")
    rolling = shared_part2.get("rolling_gate", {})
    if (
        rolling.get("block_candidates_samples") != [4, 8, 12, 16, 24, 32]
        or not _close(rolling.get("median_correlation_min"), 0.70)
        or not _close(rolling.get("median_nrmse_max"), 0.80)
        or rolling.get("selection_rule") != "largest_eligible_block"
        or rolling.get("no_eligible_block") != "NO_GO_before_part3"
    ):
        raise ContractError("shared Part-II rolling fail-closed gate changed")
    shared_schedule = protocol["common_model"]["part3"]["fixed_training_schedule"]
    if int(shared_schedule["teacher_epochs_total"]) != 1050:
        raise ContractError("shared protocol teacher total changed")
    if int(shared_schedule.get("s3_to_s4_modern_actor_initialization_seed", -1)) != 20260921:
        raise ContractError("shared S3-to-S4 modern Actor initialization seed changed")
    shared_crn = protocol["common_random_numbers"]
    if (
        int(shared_crn.get("teacher_training_base_seed", -1)) != 20260921
        or shared_crn.get("teacher_training_noise_rule")
        != "20260921+1009*(absolute_epoch+1)"
        or int(shared_crn.get("teacher_validation_seed", -1)) != 20260922
        or int(shared_crn.get("wgan_training_base_seed", -1)) != 20261011
    ):
        raise ContractError("shared teacher/WGAN random-number contract changed")
    if tuple(shared_schedule["teacher_formal_stage_ids"]) != TEACHER_PHASES:
        raise ContractError("shared protocol teacher stage identities changed")
    if bool(shared_schedule.get("discarded_robust_branch_executed", True)):
        raise ContractError("shared protocol permits the discarded S2R branch")
    if int(shared_schedule["wgan_epochs"]) != 40:
        raise ContractError("shared protocol WGAN schedule changed")
    local_stages = config["part3"]["training"]["teacher_stages"]
    if canonical_json_bytes(shared_schedule["teacher_stages"]) != canonical_json_bytes(local_stages):
        raise ContractError("local teacher schedule differs from shared protocol")
    shared_actor = protocol["common_model"]["part3"]["actor"]
    if shared_actor.get("runtime_control_projection_or_rescaling") is not False:
        raise ContractError("shared protocol permits runtime control projection")
    if protocol["energy"].get("hard_projection_or_rescaling_allowed") is not False:
        raise ContractError("shared protocol permits energy projection/rescaling")
    if protocol["selection_rule"].get(
        "analytical_feasibility_prefilter_excludes_gate_a_and_full_gate_count"
    ) is not True:
        raise ContractError("shared analytical feasibility now prefilters Gate-A/Gate-B count")
    forbidden = protocol["forbidden_training_inputs"]
    if any(
        forbidden.get(name) is not True
        for name in (
            "old_checkpoints",
            "old_fitted_models",
            "old_ofrc_artifacts",
            "old_rollouts_or_old_controller_outputs",
            "warm_start_from_any_prior_patient_result",
        )
    ):
        raise ContractError("shared prior-artifact training-input prohibition changed")
    outer = implementation["outer_evaluation"]
    if (
        outer.get("aggregate_context_indices") != list(range(8))
        or int(outer.get("common_random_number_banks", -1)) != 3
        or int(outer.get("required_unique_context_bank_evaluations_per_channel", -1)) != 24
        or outer.get("aggregation_requires_complete_cartesian_product") is not True
        or int(outer.get("fixed_display_context_index", -1)) != 7
        or int(outer.get("fixed_display_crn_bank", -1)) != 0
        or outer.get("reference_and_ictal_windows_use_independent_canonical_preprocessing_state")
        is not True
    ):
        raise ContractError("shared outer 8x3/display-context07-bank0 contract changed")
    shared_locks = protocol["canonical_hup060_locks"]
    local_to_shared_lock = {
        "canonical_data_loader": "part3_data",
        "causal_particle_rollout": "part3_causal_particle_rollout",
        "causal_riccati": "part3_riccati",
        "square_wave_mfc": "part3_square_wave",
    }
    for name, pair in config["canonical"]["locks"].items():
        shared_name = local_to_shared_lock.get(name, name)
        if shared_name not in shared_locks:
            raise ContractError(f"local canonical lock is absent from shared protocol: {name}")
        expected = shared_locks[shared_name]
        if Path(str(pair[0])).resolve() != Path(str(expected["path"])).resolve():
            raise ContractError(f"canonical lock path differs from shared protocol: {name}")
        if str(pair[1]).lower() != str(expected["sha256"]).lower():
            raise ContractError(f"canonical lock SHA differs from shared protocol: {name}")
    return {"protocol": protocol_receipt, "common_module": module_receipt}


def verify_canonical_locks(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    receipts = []
    for name, pair in sorted(config["canonical"]["locks"].items()):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ContractError(f"invalid canonical lock entry: {name}")
        receipt = verify_file_lock(Path(str(pair[0])), str(pair[1]))
        receipt["lock_name"] = name
        receipts.append(receipt)
    return receipts


def verify_code_reference_locks(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Hash code-only adapters/renderers; these are never scientific inputs."""

    receipts = []
    for name, pair in sorted(config["code_reference_locks"].items()):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ContractError(f"invalid code reference lock entry: {name}")
        receipt = verify_file_lock(Path(str(pair[0])), str(pair[1]))
        receipt["lock_name"] = name
        receipt["role"] = "code_reference_only_never_model_or_result_input"
        receipts.append(receipt)
    return receipts


def verify_implementation_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the executable local science sources to the reviewed adaptation audit."""

    binding = config["implementation_audit"]
    path = Path(str(binding["path"])).resolve()
    if path.parent != ROOT or path.name != "SOURCE_ADAPTATION_AUDIT.json":
        raise ContractError("implementation audit escaped the isolated HUP065 root")
    receipt = verify_file_lock(path, str(binding["sha256"]))
    audit = load_json(path)
    if audit.get("schema_version") != "hup065-source-adaptation-audit-v1":
        raise ContractError("unsupported source-adaptation audit schema")
    if audit.get("shared_protocol_sha256") != config["shared_protocol"]["sha256"]:
        raise ContractError("source-adaptation audit protocol binding changed")
    source_receipts = []
    for relative, expected in sorted(audit["scientific_implementation_sources"].items()):
        source = (ROOT / str(relative)).resolve()
        try:
            source.relative_to(ROOT)
        except ValueError as error:
            raise ContractError(f"implementation source escaped isolated root: {source}") from error
        source_receipts.append(
            {"relative_path": str(relative), **verify_file_lock(source, str(expected))}
        )
    semantics = audit["numerical_semantics_changed_from_d_canonical"]
    if any(bool(value) for value in semantics.values()):
        raise ContractError("source-adaptation audit declares a D numerical-semantic change")
    rejected = audit["explicitly_rejected_baseline_adapted_sources"]
    if rejected.get("causal_ltv_particle_rollout.py") != (
        "bc073733efe3aa7b1a85e1b4674a8400e950b5d2695c7091695da9588604779e"
    ) or rejected.get("causal_ltv_riccati.py") != (
        "47557ec71dec40723ec7d2cf0c41f4f2f742bbea854fbafb79ef875b696aa846"
    ):
        raise ContractError("rejected baseline plant-source hashes changed")
    receipt["source_receipts"] = source_receipts
    receipt["numerical_semantics_changed"] = False
    return receipt


def zip_central_directory_inventory(archive: Path) -> dict[str, Any]:
    """Inventory ZIP metadata without opening/decompressing any member payload."""

    records = []
    with zipfile.ZipFile(archive, mode="r") as bundle:
        for item in bundle.infolist():
            records.append(
                {
                    "filename": item.filename,
                    "crc32": int(item.CRC),
                    "compressed_bytes": int(item.compress_size),
                    "uncompressed_bytes": int(item.file_size),
                    "compress_type": int(item.compress_type),
                    "flag_bits": int(item.flag_bits),
                    "header_offset": int(item.header_offset),
                    "date_time": list(item.date_time),
                }
            )
    return {
        "archive": str(archive.resolve()),
        "archive_bytes": archive.stat().st_size,
        "member_count": len(records),
        "members": records,
        "central_directory_metadata_sha256": sha256_bytes(
            canonical_json_bytes(records)
        ),
        "member_payload_opened": False,
        "whole_archive_sha256_computed": False,
    }


def validate_hup065_split_row(config: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(config["canonical"]["locks"]["formal_patient_split_proposal"][0])
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("subject") == "HUP065"]
    if len(rows) != 1:
        raise ContractError(f"expected one HUP065 split row, found {len(rows)}")
    row = rows[0]
    exact = {
        "split_status": "ready",
        "split_scheme": "held_out_seizure",
        "ictal_train_runs": "run-01;run-02",
        "ictal_test_run": "run-03",
        "inner_validation_scheme": "leave_one_eligible_training_seizure_out",
        "reference_isolation_level": "seizure-level",
        "reference_label": "preictal_reference",
        "train_test_run_overlap": "False",
        "final_refit_test_overlap": "False",
        "reference_train_test_overlap": "False",
    }
    for field, expected in exact.items():
        if str(row.get(field, "")) != expected:
            raise ContractError(f"split field {field} changed: {row.get(field)!r}")
    reference_train = json.loads(row["reference_train_windows_json"])
    expected_train = [
        {"run": "run-01", "start_s": 30.0, "stop_s": 70.0},
        {"run": "run-02", "start_s": 30.0, "stop_s": 70.0},
    ]
    simplified = [
        {"run": str(item["run"]), "start_s": float(item["start_s"]), "stop_s": float(item["stop_s"])}
        for item in reference_train
    ]
    if simplified != expected_train:
        raise ContractError("HUP065 reference train windows changed")
    reference_test = json.loads(row["reference_test_window_json"])
    if {
        "run": str(reference_test["run"]),
        "start_s": float(reference_test["start_s"]),
        "stop_s": float(reference_test["stop_s"]),
    } != {"run": "run-03", "start_s": 75.0, "stop_s": 115.0}:
        raise ContractError("HUP065 reference outer window changed")
    test = json.loads(row["ictal_test_window_json"])
    if {
        "run": str(test["run"]),
        "start_s": float(test["start_s"]),
        "stop_s": float(test["stop_s"]),
    } != {"run": "run-03", "start_s": 120.0, "stop_s": 152.0}:
        raise ContractError("HUP065 ictal outer window changed")
    return {
        "subject": "HUP065",
        "development_runs": ["run-01", "run-02"],
        "outer_run": "run-03",
        "reference_train": simplified,
        "reference_outer": {"run": "run-03", "start_s": 75.0, "stop_s": 115.0},
    }


def recursive_inventory(roots: Sequence[str]) -> dict[str, Any]:
    """Hash immutable authority roots.  This is never called by static tests."""

    records: list[dict[str, Any]] = []
    for raw_root in roots:
        root = Path(str(raw_root)).resolve()
        if not root.is_dir():
            raise ContractError(f"authority root is absent: {root}")
        for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: str(item).casefold()):
            relative = path.relative_to(root).as_posix()
            records.append(
                {
                    "authority_root": str(root),
                    "relative_path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    digest = sha256_bytes(canonical_json_bytes(records))
    return {"schema_version": 1, "created_utc": utc_now(), "records": records, "inventory_sha256": digest}


def assert_inventory_equal(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    if before.get("inventory_sha256") != after.get("inventory_sha256"):
        raise ContractError("HUP060 authority before/after inventory changed")
    if before.get("records") != after.get("records"):
        raise ContractError("HUP060 authority path/size/hash inventory changed")


def phase_spec(config: Mapping[str, Any], phase_id: str) -> Mapping[str, Any]:
    matches = [item for item in config["phases"] if item["id"] == phase_id]
    if len(matches) != 1:
        raise ContractError(f"unknown or duplicate phase: {phase_id}")
    return matches[0]


def phase_dir(run_root: Path, phase_id: str) -> Path:
    if phase_id not in PHASE_ORDER:
        raise ContractError(f"unknown physical phase directory request: {phase_id}")
    return run_root / PHASE_OUTPUT_ROOT_DIRECTORY / phase_id


def enumerate_formal_output_paths(
    config: Mapping[str, Any], run_root: Path
) -> list[dict[str, Any]]:
    """Enumerate every active formal target, staging target, and explicit temp.

    This is a pure path calculation: it performs no filesystem reads or writes.
    The sample token is exactly the fixed 22-character width produced by
    ``secrets.token_bytes(16)`` plus URL-safe base64 without padding.
    """

    run_root = Path(run_root)
    token = "A" * 22
    if set(FORMAL_PHASE_ARTIFACTS) != set(PHASE_ORDER):
        raise ContractError("formal phase artifact inventory is incomplete")
    records: list[dict[str, Any]] = []

    def add(kind: str, operation: str, path: Path, phase: str | None = None) -> None:
        rendered = str(path)
        records.append(
            {
                "kind": kind,
                "operation": operation,
                "phase": phase,
                "path": rendered,
                "characters": len(rendered),
            }
        )

    init_stage = run_root.with_name(f".i-{token}")
    add("target", "science_run_directory", run_root)
    add("staging", "init_staging_directory", init_stage)
    for relative in ("config.snapshot.json", "RUN.json"):
        add("target", f"init_publish:{relative}", run_root / relative)
        add("staging", f"init_write:{relative}", init_stage / relative)
        add("atomic_temp", f"init_atomic_json:{relative}", init_stage / f".j-{token}")
    add(
        "target", "physical_phase_root_directory",
        run_root / PHASE_OUTPUT_ROOT_DIRECTORY,
    )
    add(
        "staging", "init_physical_phase_root_directory",
        init_stage / PHASE_OUTPUT_ROOT_DIRECTORY,
    )

    for relative in (
        str(config["outer_go"]["receipt_filename"]),
        str(config["outer_go"]["single_open_ledger_filename"]),
    ):
        add("target", f"run_root_receipt:{relative}", run_root / relative)
    add(
        "atomic_temp", "outer_opened_atomic_json",
        run_root / f".j-{token}", "OUTER",
    )

    for phase_index, phase in enumerate(PHASE_ORDER):
        final_directory = phase_dir(run_root, phase)
        staging_directory = (
            run_root / PHASE_OUTPUT_ROOT_DIRECTORY / f".s{phase_index:02d}-{token}"
        )
        add("target", "phase_directory", final_directory, phase)
        add("staging", "phase_staging_directory", staging_directory, phase)
        for relative in FORMAL_PHASE_ARTIFACTS[phase]:
            relative_path = Path(relative)
            final_path = final_directory / relative_path
            staging_path = staging_directory / relative_path
            add("target", f"phase_artifact:{relative}", final_path, phase)
            add("staging", f"phase_artifact:{relative}", staging_path, phase)
            if relative_path.suffix.casefold() == ".json":
                add(
                    "atomic_temp", f"atomic_json_for:{relative}",
                    staging_path.parent / f".j-{token}", phase,
                )
            if relative_path.suffix.casefold() == ".pt":
                add(
                    "atomic_temp", f"atomic_torch_for:{relative}",
                    staging_path.with_suffix(staging_path.suffix + ".tmp"), phase,
                )
    return records


def validate_formal_path_budget(
    config: Mapping[str, Any], run_root: Path
) -> dict[str, Any]:
    """Fail closed unless every enumerated formal path is strictly below 200."""

    limit = int(config["runtime_safety"]["formal_path_characters_max_exclusive"])
    records = enumerate_formal_output_paths(config, run_root)
    failures = [record for record in records if int(record["characters"]) >= limit]
    if failures:
        worst = max(failures, key=lambda record: int(record["characters"]))
        raise ContractError(
            "formal output path budget exceeded: "
            f"limit=<{limit}, characters={worst['characters']}, path={worst['path']}"
        )
    worst_by_kind = {
        kind: max(
            (record for record in records if record["kind"] == kind),
            key=lambda record: (int(record["characters"]), str(record["path"])),
        )
        for kind in ("target", "staging", "atomic_temp")
    }
    return {
        "status": "FORMAL_PATH_BUDGET_PASS",
        "maximum_characters_exclusive": limit,
        "record_count": len(records),
        "worst_by_kind": worst_by_kind,
        "overall_worst": max(
            records,
            key=lambda record: (int(record["characters"]), str(record["path"])),
        ),
        "records": records,
    }


def completion_receipt(run_root: Path, phase_id: str) -> Path:
    return phase_dir(run_root, phase_id) / "COMPLETE.json"


def validate_phase_dependencies(
    config: Mapping[str, Any], run_root: Path, phase_id: str
) -> None:
    for dependency in phase_spec(config, phase_id).get("depends_on", []):
        receipt = completion_receipt(run_root, str(dependency))
        if not receipt.is_file():
            raise ContractError(f"{phase_id} requires completed {dependency}: {receipt}")


@contextmanager
def staged_phase_directory(
    config: Mapping[str, Any], run_root: Path, phase_id: str
) -> Iterator[Path]:
    """Create a private sibling staging directory and atomically publish it."""

    assert_output_path(config, run_root)
    destination = phase_dir(run_root, phase_id)
    if destination.exists():
        raise ExistingOutputError(f"phase output already exists: {destination}")
    phase_parent = destination.parent
    phase_parent.mkdir(parents=True, exist_ok=True)
    try:
        phase_index = PHASE_ORDER.index(phase_id)
    except ValueError as error:
        raise ContractError(f"unknown phase staging request: {phase_id}") from error
    staging = phase_parent / f".s{phase_index:02d}-{_short_uuid_token()}"
    staging.mkdir(parents=False, exist_ok=False)
    published = False
    try:
        yield staging
        os.replace(staging, destination)
        published = True
    finally:
        # Preserve non-empty failed staging for forensic audit.  Empty staging
        # contains no scientific result and may be removed safely.
        if not published and staging.exists() and not any(staging.iterdir()):
            staging.rmdir()


def init_run(config: Mapping[str, Any], run_root: Path) -> Path:
    """Create a new run root exactly once.  An existing root is always NO_GO."""

    assert_output_path(config, run_root)
    if run_root.exists():
        raise ExistingOutputError(f"science run root already exists: {run_root}")
    run_root.parent.mkdir(parents=True, exist_ok=True)
    staging = run_root.with_name(f".i-{_short_uuid_token()}")
    if staging.exists():
        raise ExistingOutputError(f"unexpected init staging exists: {staging}")
    staging.mkdir()
    try:
        snapshot = json.loads(json.dumps(config))
        write_json(staging / "config.snapshot.json", snapshot)
        receipt = {
            "schema_version": 1,
            "status": "INITIALIZED_NO_SCIENCE_EXECUTED",
            "created_utc": utc_now(),
            "config_sha256": sha256_bytes(canonical_json_bytes(snapshot)),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "bytecode_suppressed": bool(sys.dont_write_bytecode),
            "runtime_root": str(Path(str(config["runtime_root"])).resolve()),
        }
        write_json(staging / "RUN.json", receipt)
        (staging / PHASE_OUTPUT_ROOT_DIRECTORY).mkdir()
        os.replace(staging, run_root)
    finally:
        if staging.exists() and not any(staging.iterdir()):
            staging.rmdir()
    return run_root


def load_run_receipt(config: Mapping[str, Any], run_root: Path) -> Mapping[str, Any]:
    receipt_path = run_root / "RUN.json"
    snapshot_path = run_root / "config.snapshot.json"
    if not receipt_path.is_file() or not snapshot_path.is_file():
        raise ContractError("run root lacks authenticated RUN/config snapshot")
    receipt = load_json(receipt_path)
    snapshot = load_json(snapshot_path)
    expected = sha256_bytes(canonical_json_bytes(snapshot))
    if receipt.get("config_sha256") != expected:
        raise ContractError("run config snapshot hash mismatch")
    if canonical_json_bytes(snapshot) != canonical_json_bytes(config):
        raise ContractError("current config differs from the initialized run snapshot")
    if receipt.get("runner_sha256") != sha256_file(Path(__file__).resolve()):
        raise ContractError("runner bytes differ from the initialized run")
    return receipt


def execute_preflight(config: Mapping[str, Any], run_root: Path) -> Path:
    """Metadata/hash-only preflight.  No EDF member is opened."""

    load_run_receipt(config, run_root)
    validate_phase_dependencies(config, run_root, PREFLIGHT_PHASE)
    shared = verify_shared_protocol(config)
    locks = verify_canonical_locks(config)
    code_locks = verify_code_reference_locks(config)
    implementation_audit = verify_implementation_audit(config)
    split = validate_hup065_split_row(config)
    archive = Path(str(config["raw_data"]["subject_archive"])).resolve()
    if not archive.is_file():
        raise ContractError(f"raw subject ZIP is absent: {archive}")
    # Central-directory metadata is permitted; no member is opened and neither
    # preflight nor DEV computes a whole-archive hash that would consume run03.
    archive_metadata = zip_central_directory_inventory(archive)
    before = recursive_inventory(config["canonical"]["authority_roots"])
    with staged_phase_directory(config, run_root, PREFLIGHT_PHASE) as staging:
        write_json(staging / "authority_before.json", before)
        write_json(staging / "canonical_lock_receipts.json", locks)
        write_json(staging / "code_reference_lock_receipts.json", code_locks)
        write_json(staging / "source_adaptation_audit_receipt.json", implementation_audit)
        write_json(staging / "shared_protocol_receipt.json", shared)
        write_json(staging / "split_receipt.json", split)
        write_json(
            staging / "raw_archive_stat_and_central_directory_only.json",
            archive_metadata,
        )
        complete = {
            "schema_version": 1,
            "phase": PREFLIGHT_PHASE,
            "status": "COMPLETE",
            "created_utc": utc_now(),
            "science_executed": False,
            "patient_signal_opened": False,
            "run03_opened": False,
            "zip_member_payload_opened": False,
            "whole_zip_sha256_computed": False,
            "authority_inventory_sha256": before["inventory_sha256"],
        }
        write_json(staging / "COMPLETE.json", complete)
    return phase_dir(run_root, PREFLIGHT_PHASE)


def verify_authority_invariant(config: Mapping[str, Any], run_root: Path) -> dict[str, Any]:
    before_path = phase_dir(run_root, PREFLIGHT_PHASE) / "authority_before.json"
    if not before_path.is_file():
        raise ContractError("preflight authority inventory is absent")
    before = load_json(before_path)
    after = recursive_inventory(config["canonical"]["authority_roots"])
    assert_inventory_equal(before, after)
    return after


def lazy_load_science_backend(config: Mapping[str, Any]):
    """Load the adjacent backend only after all immutable locks passed.

    The backend is intentionally separate so importing this orchestration file
    remains a static operation.  It must expose ``execute_phase(context)``.
    """

    path = ROOT / "science_backend.py"
    if not path.is_file():
        raise ContractError(
            "science_backend.py is absent; this static delivery does not authorize "
            "or execute patient science"
        )
    spec = importlib.util.spec_from_file_location("hup065_fresh_science_backend", path)
    if spec is None or spec.loader is None:
        raise ContractError("unable to load science backend")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "execute_phase", None)):
        raise ContractError("science backend lacks execute_phase(context)")
    return module


def teacher_stage_config(
    config: Mapping[str, Any], phase_id: str
) -> Mapping[str, Any] | None:
    matches = [
        stage
        for stage in config["part3"]["training"]["teacher_stages"]
        if stage["stage_id"] == phase_id
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ContractError(f"duplicate teacher stage: {phase_id}")
    return matches[0]


def validate_backend_receipt(phase_id: str, result: Mapping[str, Any]) -> None:
    """Enforce phase-specific claims before any staging directory is published."""

    if result.get("phase") != phase_id:
        raise ContractError("science backend receipt phase mismatch")
    if bool(result.get("run03_opened", False)):
        raise ContractError(f"{phase_id} backend reported forbidden run03 access")
    if bool(result.get("discarded_robust_branch_executed", False)):
        raise ContractError("discarded S2R robust branch was executed")
    if result.get("hard_projection_or_rescaling_applied") is not False:
        raise ContractError(f"{phase_id} did not attest unprojected empirical FP semantics")
    if phase_id == "DEV_MATERIALIZE":
        if result.get("opened_signal_runs") != ["run-01", "run-02"]:
            raise ContractError("development materialization signal roles changed")
        if result.get("whole_zip_sha256_computed") is not False:
            raise ContractError("development computed a forbidden whole-ZIP hash")
    if phase_id == "P1_LORO_GRAPHS" and int(result.get("folds", -1)) != 2:
        raise ContractError("Part-I did not execute both LORO folds")
    if phase_id == "P2_LORO_RC_SDE":
        if int(result.get("folds", -1)) != 2 or int(result.get("candidate_count", -1)) != 144:
            raise ContractError("Part-II did not execute the frozen 2x144 LORO grid")
    if phase_id == "ANALYTICAL_TOP1":
        if int(result.get("candidate_count", -1)) != 120:
            raise ContractError("analytical backend did not evaluate all 120 arms")
        if int(result.get("freeze_unique_top_k", -1)) != 1:
            raise ContractError("analytical backend did not freeze unique top1")
        if bool(result.get("neural_training_executed", True)):
            raise ContractError("neural training occurred during analytical prescreen")
    if phase_id in TEACHER_PHASES:
        expected = TEACHER_STAGE_SPECS[TEACHER_PHASES.index(phase_id)]
        if int(result.get("epochs", -1)) != int(expected[1]):
            raise ContractError(f"{phase_id} epoch count mismatch")
        if int(result.get("epoch_offset", -1)) != int(expected[2]):
            raise ContractError(f"{phase_id} cumulative offset mismatch")
        if result.get("objective") != expected[5] or result.get("train_scope") != expected[7]:
            raise ContractError(f"{phase_id} objective/train-scope mismatch")
    if phase_id == FREEZE_PHASE:
        if int(result.get("teacher_epochs_total", -1)) != 1050:
            raise ContractError("final development receipt lacks teacher1050 completion")
        if int(result.get("wgan_epochs", -1)) != 40:
            raise ContractError("final development receipt lacks WGAN40 completion")
        if bool(result.get("arm_reselected", True)):
            raise ContractError("ctx5/ctx6 backend reselected the analytical top1 arm")
        if result.get("context_5_role") != "checkpoint_epoch_selection_only":
            raise ContractError("ctx5 role changed")
        if result.get("context_6_role") != "terminal_veto_only":
            raise ContractError("ctx6 role changed")
        if result.get("ctx6_terminal_veto_pass") is not True:
            raise ContractError("ctx6 terminal veto did not pass; OUTER must remain sealed")
        if result.get("wgan_selection_outcome") not in {
            "ACCEPTED_NONINFERIOR_WGAN",
            "NO_ACCEPTED_WGAN_RETAIN_FROZEN_S6_TEACHER",
        }:
            raise ContractError("WGAN fail-closed selection outcome changed")


def execute_science_phase(
    config: Mapping[str, Any],
    run_root: Path,
    phase_id: str,
    *,
    authorization: str,
) -> Path:
    if phase_id not in DEVELOPMENT_SCIENCE_PHASES:
        raise ContractError(f"not a development science phase: {phase_id}")
    expected = os.environ.get(SCIENCE_AUTHORIZATION_ENV, "")
    if not expected or not hmac.compare_digest(authorization, expected):
        raise ScienceAuthorizationError("development science authorization is absent")
    load_run_receipt(config, run_root)
    validate_phase_dependencies(config, run_root, phase_id)
    verify_shared_protocol(config)
    verify_canonical_locks(config)
    verify_implementation_audit(config)
    before = load_json(phase_dir(run_root, PREFLIGHT_PHASE) / "authority_before.json")
    module = lazy_load_science_backend(config)
    with staged_phase_directory(config, run_root, phase_id) as staging:
        context = {
            "phase": phase_id,
            "config": dict(config),
            "run_root": run_root,
            "staging_dir": staging,
            "python_executable": sys.executable,
            "python_subprocess_prefix": [sys.executable, "-B"],
            "sealed_run": "run-03",
            "teacher_stage": teacher_stage_config(config, phase_id),
        }
        result = module.execute_phase(context)
        if not isinstance(result, Mapping):
            raise ContractError("science backend must return a mapping receipt")
        validate_backend_receipt(phase_id, result)
        after = recursive_inventory(config["canonical"]["authority_roots"])
        assert_inventory_equal(before, after)
        write_json(staging / "authority_after.json", after)
        write_json(staging / "backend_receipt.json", dict(result))
        write_json(
            staging / "COMPLETE.json",
            {
                "schema_version": 1,
                "phase": phase_id,
                "status": "COMPLETE",
                "created_utc": utc_now(),
                "run03_opened": False,
                "authority_inventory_sha256": after["inventory_sha256"],
            },
        )
    return phase_dir(run_root, phase_id)


def validate_outer_go(config: Mapping[str, Any], run_root: Path) -> Mapping[str, Any]:
    go_path = run_root / str(config["outer_go"]["receipt_filename"])
    if not go_path.is_file():
        raise ScienceAuthorizationError(f"outer GO receipt is absent: {go_path}")
    receipt = load_json(go_path)
    expected = os.environ.get(OUTER_AUTHORIZATION_ENV, "")
    supplied = str(receipt.get("authorization", ""))
    if not expected or not hmac.compare_digest(supplied, expected):
        raise ScienceAuthorizationError("outer GO authorization text is invalid")
    if receipt.get("subject") != "HUP065" or receipt.get("outer_run") != "run-03":
        raise ScienceAuthorizationError("outer GO subject/run binding is invalid")
    freeze_phase = str(config["outer_go"]["development_freeze_phase_id"])
    freeze_path = completion_receipt(run_root, freeze_phase)
    if not freeze_path.is_file():
        raise ScienceAuthorizationError("final development freeze is not complete")
    if receipt.get("development_freeze_complete_sha256") != sha256_file(freeze_path):
        raise ScienceAuthorizationError(
            "outer GO is not bound to the frozen development receipt"
        )
    if receipt.get("config_sha256") != sha256_file(run_root / "config.snapshot.json"):
        raise ScienceAuthorizationError("outer GO is not bound to the run config snapshot")
    return receipt


def execute_outer(
    config: Mapping[str, Any], run_root: Path, *, authorization: str
) -> Path:
    expected = os.environ.get(OUTER_AUTHORIZATION_ENV, "")
    if not expected or not hmac.compare_digest(authorization, expected):
        raise ScienceAuthorizationError("outer command-line authorization is absent")
    load_run_receipt(config, run_root)
    validate_phase_dependencies(config, run_root, "OUTER")
    validate_outer_go(config, run_root)
    opened_path = run_root / str(config["outer_go"]["single_open_ledger_filename"])
    if opened_path.exists():
        raise ExistingOutputError("outer run was already opened; a second attempt is NO_GO")
    verify_shared_protocol(config)
    verify_canonical_locks(config)
    verify_implementation_audit(config)
    before = load_json(phase_dir(run_root, PREFLIGHT_PHASE) / "authority_before.json")
    module = lazy_load_science_backend(config)
    # Publish the one-shot ledger before calling the backend.  A crash is still
    # counted as the single outer opening and cannot be retried silently.
    write_json(
        opened_path,
        {
            "schema_version": 1,
            "subject": "HUP065",
            "outer_run": "run-03",
            "status": "OPENING_CONSUMED_NO_RETRY",
            "created_utc": utc_now(),
            "development_freeze_phase": FREEZE_PHASE,
            "development_freeze_complete_sha256": sha256_file(
                completion_receipt(run_root, FREEZE_PHASE)
            ),
        },
    )
    with staged_phase_directory(config, run_root, "OUTER") as staging:
        context = {
            "phase": "OUTER",
            "config": dict(config),
            "run_root": run_root,
            "staging_dir": staging,
            "python_executable": sys.executable,
            "python_subprocess_prefix": [sys.executable, "-B"],
            "sealed_run": "run-03",
            "outer_windows": config["split"]["outer_preview_windows"],
        }
        result = module.execute_phase(context)
        if not isinstance(result, Mapping) or not bool(result.get("run03_opened", False)):
            raise ContractError("outer backend did not attest its one-time run03 opening")
        if bool(result.get("candidate_reselected", True)):
            raise ContractError("outer backend reselected a candidate")
        if result.get("aggregate_window_ids") != [
            f"outer-context-{index:02d}" for index in range(8)
        ]:
            raise ContractError("outer backend did not aggregate all eight contexts")
        if result.get("display_window_id") != "outer-context-07":
            raise ContractError("outer backend changed the fixed display context")
        if int(result.get("display_crn_bank", -1)) != 0:
            raise ContractError("outer backend changed the fixed display CRN bank")
        if result.get("common_renderer_executed") is not False:
            raise ContractError("OUTER executed a renderer before independent audit/GO")
        if result.get("renderer_binding_status") != "NO_GO_PENDING_INDEPENDENT_AUDIT":
            raise ContractError("outer renderer binding did not remain pending independent audit")
        after = recursive_inventory(config["canonical"]["authority_roots"])
        assert_inventory_equal(before, after)
        write_json(staging / "authority_after.json", after)
        write_json(staging / "backend_receipt.json", dict(result))
        write_json(
            staging / "COMPLETE.json",
            {
                "schema_version": 1,
                "phase": "OUTER",
                "status": "COMPLETE_ONE_TIME_RETROSPECTIVE_REPORT",
                "created_utc": utc_now(),
                "run03_opened": True,
                "candidate_reselected": False,
                "fallback_or_rescue": False,
                "authority_inventory_sha256": after["inventory_sha256"],
            },
        )
    return phase_dir(run_root, "OUTER")


def status(config: Mapping[str, Any], run_root: Path) -> dict[str, Any]:
    if not run_root.exists():
        return {"run_root": str(run_root), "initialized": False, "phases": {}}
    load_run_receipt(config, run_root)
    phases = {}
    for phase_id in PHASE_ORDER:
        directory = phase_dir(run_root, phase_id)
        receipt = completion_receipt(run_root, phase_id)
        phases[phase_id] = {
            "directory_exists": directory.exists(),
            "complete": receipt.is_file(),
            "receipt_sha256": sha256_file(receipt) if receipt.is_file() else None,
        }
    return {
        "run_root": str(run_root),
        "initialized": True,
        "outer_open_consumed": (
            run_root / str(config["outer_go"]["single_open_ledger_filename"])
        ).exists(),
        "phases": phases,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-root", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("static-check", help="local static validation; no D/patient read")
    sub.add_parser(
        "path-check",
        help="enumerate formal target/staging/temp paths; no filesystem write",
    )
    sub.add_parser("plan", help="print the frozen phase plan")
    sub.add_parser("init", help="create a new science run root; existing is NO_GO")
    sub.add_parser("preflight", help="hash/metadata preflight only; no patient signal open")
    run = sub.add_parser("run", help="execute one authorized development backend phase")
    run.add_argument("phase", choices=PHASE_ORDER[1:-1])
    run.add_argument("--authorization", required=True)
    outer = sub.add_parser("outer", help="one-time run03 phase after external GO")
    outer.add_argument("--authorization", required=True)
    sub.add_parser("status", help="read local phase receipts")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    configure_runtime(
        config, create=args.command not in {"static-check", "path-check", "plan"}
    )
    summary = validate_static_config(config)
    run_root = (
        args.run_root.resolve()
        if args.run_root is not None
        else Path(str(config["default_run_root"])).resolve()
    )
    path_budget = validate_formal_path_budget(config, run_root)
    if args.command == "static-check":
        print(
            json.dumps(
                {
                    "status": "STATIC_CHECK_PASS",
                    **summary,
                    "formal_path_budget": {
                        key: value
                        for key, value in path_budget.items()
                        if key != "records"
                    },
                },
                indent=2,
            )
        )
        return 0
    if args.command == "path-check":
        print(json.dumps(path_budget, ensure_ascii=False, indent=2))
        return 0
    if args.command == "plan":
        print(json.dumps({"phases": config["phases"], **summary}, indent=2))
        return 0
    if args.command == "init":
        path = init_run(config, run_root)
        print(json.dumps({"status": "INITIALIZED", "run_root": str(path)}, indent=2))
        return 0
    if args.command == "preflight":
        path = execute_preflight(config, run_root)
        print(json.dumps({"status": "PREFLIGHT_COMPLETE", "path": str(path)}, indent=2))
        return 0
    if args.command == "run":
        path = execute_science_phase(
            config,
            run_root,
            str(args.phase),
            authorization=str(args.authorization),
        )
        print(json.dumps({"status": f"{args.phase}_COMPLETE", "path": str(path)}, indent=2))
        return 0
    if args.command == "outer":
        path = execute_outer(
            config, run_root, authorization=str(args.authorization)
        )
        print(json.dumps({"status": "OUTER_COMPLETE", "path": str(path)}, indent=2))
        return 0
    if args.command == "status":
        print(json.dumps(status(config, run_root), ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
