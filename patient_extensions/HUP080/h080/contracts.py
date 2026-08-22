from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping
import zipfile


CONFIG_NAME = "config.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_SCIENCE_INPUT_TOKENS = (
    "task-interictal",
    "task_interictal",
    "interictal",
    "old_checkpoint",
    "old_model",
    "ofrc",
    "development_m54",
    "development_sparse75",
    "parameter_only_v1",
)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(encoded)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict[str, Any]:
    resolved = project_root() / CONFIG_NAME if path is None else Path(path).resolve()
    config = load_json(resolved)
    validate_config(config, resolved)
    return config


def _same_path(left: str | Path, right: str | Path) -> bool:
    return os.path.normcase(os.path.abspath(str(left))) == os.path.normcase(
        os.path.abspath(str(right))
    )


def assert_no_forbidden_science_path(path: str | Path) -> None:
    text = str(path).replace("\\", "/").casefold()
    hits = [token for token in FORBIDDEN_SCIENCE_INPUT_TOKENS if token in text]
    if hits:
        raise PermissionError(f"forbidden prior/interictal science input {path!s}: {hits}")


def validate_config(config: Mapping[str, Any], source: Path | None = None) -> None:
    if config.get("schema_version") != "hup080-hup060-sparse-rerun-v2.0":
        raise ValueError("unsupported HUP080 rerun configuration schema")
    if config.get("subject") != "HUP080":
        raise ValueError("this runner is locked to HUP080")
    root = Path(str(config["project_root"]))
    if source is not None and not _same_path(root, source.resolve().parent):
        raise ValueError("config project_root differs from its actual directory")
    science = Path(str(config["science_root"]))
    if science.parent != root or science.name != "science_run":
        raise ValueError("science_root must be the dedicated project_root/science_run")
    amendment = config.get("exploratory_v2_delta", {})
    if amendment.get("amendment_id") != (
        "HUP080_POST_V1_PHASE02_NO_GO_ROLLING_GRID_ONLY_V2"
    ):
        raise ValueError("exploratory v2 amendment identity changed")
    if amendment.get("classification") != (
        "post_v1_phase02_no_go_protocol_amendment"
    ):
        raise ValueError("exploratory v2 amendment classification changed")
    if amendment.get("base_config_sha256") != (
        "0539881d94304cd7116839bf7a9ffa1fcb8a0382cce5bb90dc1eb641aa0fac1a"
    ):
        raise ValueError("exploratory v2 base config binding changed")
    if amendment.get("base_implementation_freeze_sha256") != (
        "359745077c159ec9ea5a6aebca045e8f55f3d8af0b9024777f68df0004982ffd"
    ):
        raise ValueError("exploratory v2 base freeze binding changed")
    if amendment.get("base_candidates") != [4, 8, 12, 16, 24, 32]:
        raise ValueError("exploratory v2 base rolling candidates changed")
    if amendment.get("extended_candidates") != [1, 2, 4, 8, 12, 16, 24, 32]:
        raise ValueError("exploratory v2 extended rolling candidates changed")
    if amendment.get("amended_json_pointer") != "/part2/rolling_block_candidates":
        raise ValueError("exploratory v2 amended field changed")
    if amendment.get("only_added_candidates") != [1, 2]:
        raise ValueError("exploratory v2 may add only blocks 1 and 2")
    if amendment.get("thresholds_or_selection_rule_changed") is not False:
        raise ValueError("exploratory v2 changed a forbidden threshold or rule")
    if amendment.get("parent_no_go_remains_valid") is not True:
        raise ValueError("v1 NO_GO must remain valid")
    if amendment.get("may_be_described_as_original_preregistered_replication") is not False:
        raise ValueError("exploratory v2 must not be called the original preregistered replication")
    if amendment.get("run04_seen_before_amendment") is not False:
        raise ValueError("run-04 was seen before exploratory v2 amendment")
    if amendment.get("thresholds_unchanged") != {
        "minimum_correlation": 0.7, "maximum_nrmse": 0.8,
    }:
        raise ValueError("exploratory v2 rolling thresholds changed")
    if amendment.get("joint_fold_rule_unchanged") != (
        "every_development_loro_fold_must_pass"
    ):
        raise ValueError("exploratory v2 joint-fold rule changed")
    if amendment.get("v1_failure_evidence_role") != (
        "AUDIT_ONLY_DO_NOT_REUSE_FOR_SCIENCE_SELECTION_OR_TRAINING"
    ):
        raise ValueError("v1 failure evidence role changed")
    if amendment.get("v2_may_read_v1_failure_as_science_input") is not False:
        raise ValueError("v1 failure evidence must never become a science input")
    if amendment.get("sealed_run04_role_unchanged") is not True:
        raise ValueError("sealed run-04 role changed in exploratory v2")
    if amendment.get("authorized_write_roots") != [
        str(science), str(Path(str(config["runtime_root"])))
    ]:
        raise ValueError("exploratory v2 write-root ledger changed")

    shared = config["shared_protocol"]
    for field in (
        "sha256", "common_protocol_py_sha256", "static_self_test_py_sha256",
        "readme_sha256",
    ):
        if not HEX64.fullmatch(str(shared[field])):
            raise ValueError(f"shared_protocol.{field} is not a pinned SHA-256")
    data = config["source_data"]
    if data["expected_channels"] != 96:
        raise ValueError("HUP080 expected channel count must remain 96")
    if data["development_runs"] != ["run-01", "run-02", "run-03"]:
        raise ValueError("development run roles changed")
    if data["sealed_run"] != "run-04":
        raise ValueError("sealed run role changed")
    if data["ictal_window_absolute_seconds_half_open"] != [120.0, 152.0]:
        raise ValueError("ictal window changed")
    if data["development_reference_window_absolute_seconds_half_open"] != [30.0, 70.0]:
        raise ValueError("development seizure-pre window changed")
    if data["sealed_reference_window_absolute_seconds_half_open"] != [75.0, 115.0]:
        raise ValueError("sealed seizure-pre window changed")
    if data["reference_label"] != "preictal_reference":
        raise ValueError("reference semantics changed")
    if str(data["forbidden_task"]).casefold() != "interictal":
        raise ValueError("task-interictal must remain forbidden")
    if "raw_zip_sha256" in data:
        raise ValueError("whole-ZIP SHA must not be an executable runtime gate")
    if int(data["raw_zip_bytes"]) != 92298727:
        raise ValueError("raw ZIP byte-size receipt changed")
    if not HEX64.fullmatch(str(data["raw_zip_central_directory_metadata_sha256"])):
        raise ValueError("raw ZIP central-directory digest is invalid")
    if len(data["development_member_contract"]) != 14:
        raise ValueError("development member allowlist must contain exactly 14 entries")
    if any("run-04" in str(item["filename"]) for item in data["development_member_contract"]):
        raise ValueError("run-04 member entered the development allowlist")
    if any("interictal" in str(item["filename"]).casefold() for item in data["development_member_contract"]):
        raise ValueError("interictal member entered the development allowlist")

    fractions = [float(v) for v in config["controller"]["fraction_grid"]]
    if fractions != [0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]:
        raise ValueError("sparse fraction grid changed")
    if [float(v) for v in config["controller"]["tau_grid"]] != [0.0, 0.05, 0.10, 0.20, 0.50]:
        raise ValueError("tau grid changed")
    if [float(v) for v in config["controller"]["gain_grid"]] != [0.35, 0.45, 0.668]:
        raise ValueError("gain grid changed")
    if int(config["controller"]["analytical_candidate_count"]) != 120:
        raise ValueError("analytical grid must contain 8*5*3=120 arms")
    if int(config["controller"]["quantile_numeric_precision_decimals"]) != 12:
        raise ValueError("quantile numerical precision guard changed")
    if int(config["controller"].get(
        "s3_to_s4_modern_template_initialization_seed", -1
    )) != 20260921:
        raise ValueError("S3-to-S4 modern-template initialization seed changed")
    if config["controller"].get("teacher_training_noise_seed_rule") != (
        "20260921 + 1009*(absolute_epoch+1)"
    ):
        raise ValueError("teacher absolute-epoch noise seed rule changed")
    if int(config["controller"].get("teacher_validation_noise_seed", -1)) != 20260922:
        raise ValueError("teacher validation noise seed changed")
    if int(config["controller"].get("wgan_validation_noise_seed", -1)) != 20260922:
        raise ValueError("WGAN validation noise seed changed")
    expected_energy = (13.0 / 36.0) * 96.0 * (0.405**2)
    if abs(float(config["controller"]["total_energy_cap"]) - expected_energy) > 1e-12:
        raise ValueError("fixed total energy cap changed")
    if float(config["controller"]["per_actuator_rms_max"]) != 0.405:
        raise ValueError("per-actuator RMS cap changed")
    if float(config["controller"]["amplitude_limit"]) != 1.8:
        raise ValueError("amplitude limit changed")
    if bool(config["controller"]["runtime_control_projection_or_rescaling"]):
        raise ValueError("runtime control projection/rescaling is forbidden")
    if int(config["controller"]["freeze_top_k"]) != 1:
        raise ValueError("exactly one analytical arm must be frozen")
    if int(config["controller"]["validation_crn_banks"]) != 3:
        raise ValueError("validation must use exactly three frozen CRN banks")
    stages = list(config["controller"]["teacher_stages"])
    expected_stage_ids = [
        "S0_INITIAL_0_100", "S1_BALANCED_100_250", "S2S_SAFE_250_450",
        "S3_COVARIANCE_450_650", "S4_MARKOV_ALPHA100_650_770",
        "S5_SMOOTHA_770_870", "S6_PRECISION_ALL_PARAMS_870_1050",
    ]
    if [str(item["stage_id"]) for item in stages] != expected_stage_ids:
        raise ValueError("formal S0-S6 chain changed")
    if list(config["controller"]["teacher_stage_ids"]) != expected_stage_ids:
        raise ValueError("formal teacher stage-id receipt changed")
    if int(config["controller"]["teacher_epochs_total"]) != 1050:
        raise ValueError("formal teacher must total 1050 epochs")
    if sum(int(item["epochs"]) for item in stages) != 1050:
        raise ValueError("formal teacher stages do not sum to 1050")
    if bool(config["controller"]["discarded_s2r_executed"]):
        raise ValueError("discarded S2R branch may not execute")
    if int(config["controller"]["wgan_epochs"]) != 40:
        raise ValueError("formal WGAN must run 40 epochs")
    expected_wgan_seeds = {
        "critic_pretrain_noise": "20261011 + 101*(bank+1)",
        "critic_pretrain_gp": "20261011 + 50000 + step",
        "joint_epoch_noise": "20261011 + 1009*epoch",
        "joint_epoch_gp": "20261011 + 100000*epoch + critic_step",
        "actor_gradient_clip": 1.0,
        "critic_gradient_clip": 5.0,
        "epoch_zero_eligible": False,
        "noninferiority_multiplier": 1.02,
        "fallback_update_may_be_promoted": False,
    }
    if dict(config["controller"]["wgan_exact_seed_contract"]) != expected_wgan_seeds:
        raise ValueError("formal WGAN seed/optimizer safety contract changed")
    expected_ctx6 = {
        "finite_all_contexts_and_crn_banks": True,
        "gate_c_all_contexts_and_crn_banks": True,
        "aggregate_time_w1_controlled_lt_free": True,
        "aggregate_occupation_w1_controlled_lt_free": True,
        "minimum_full_six_gate_b_channel_count": 1,
        "may_rank_rescue_or_replace": False,
    }
    if dict(config["ctx6_veto_rule"]) != expected_ctx6:
        raise ValueError("ctx6 terminal-veto rule changed")
    expected_gate_b = {
        "definition": "six_frozen_components",
        "minimum_time_w1_relative_reduction": 0.1,
        "minimum_occupation_w1_relative_reduction": 0.1,
        "time_w1_controlled_reference_max": 0.35,
        "occupation_w1_controlled_reference_max": 0.25,
        "controlled_reference_mean_abs_error_max": 0.1,
        "controlled_reference_symmetric_sd_ratio_max": 2.0,
        "finite_required": True,
        "aggregation": (
            "AND each component over every declared context and all three CRN banks per channel"
        ),
        "both_improved_reported_separately": True,
        "aggregate_both_w1_improvement_or_gate_b_may_be_analytical_feasibility_prefilter": False,
        "all_channels_reported": True,
    }
    if dict(config["gate_b"]) != expected_gate_b:
        raise ValueError("frozen six-component Gate-B contract changed")
    expected_analytical_eligibility = {
        "gate_a_name": "static_protocol_integrity",
        "gate_a_components": [
            "canonical_source_locks_recorded",
            "formal_split_roles_preserved",
            "no_test_or_reference_test_access_during_selection",
            "common_random_numbers_within_each_fold",
            "uncontrolled_branch_parity",
        ],
        "gate_c_name": "trajectory_safety",
        "gate_c_components": [
            "finite_outputs", "total_energy_cap", "per_actuator_rms_cap",
            "peak_cap", "saturation_fraction_cap",
        ],
        "requires_gate_a_and_gate_c": True,
        "plant_four_inequality_status": "diagnostic_only",
        "plant_may_filter_rank_or_veto_analytical_arms": False,
        "aggregate_both_w1_improvement_status": "diagnostic_only",
        "full_gate_b_count_status": "ranking_metric_not_feasibility_prefilter",
        "trajectory_safety_values_reported": True,
    }
    if dict(config.get("analytical_eligibility_contract", {})) != (
        expected_analytical_eligibility
    ):
        raise ValueError("analytical Gate-A/Gate-C eligibility contract changed")

    contexts = config["contexts"]
    expected_roles = {
        "analytical_and_controller_fit": [0, 1, 2, 3, 4],
        "checkpoint_epoch_selection_only": [5],
        "terminal_veto_only": [6],
        "posthoc_only_never_success": [7],
    }
    for key, expected in expected_roles.items():
        if list(contexts[key]) != expected:
            raise ValueError(f"context role changed: {key}")
    flattened = [item for key in expected_roles for item in contexts[key]]
    if sorted(flattened) != list(range(8)) or len(set(flattened)) != 8:
        raise ValueError("context roles must partition indices 0..7")
    if int(contexts["fixed_display_context_index"]) != 7:
        raise ValueError("outer display must be prebound to context index 7")
    outer = config["outer_evaluation"]
    expected_outer_ids = [f"outer-context-{index:02d}" for index in range(8)]
    if outer["display_window_id"] != "outer-context-07":
        raise ValueError("outer representative display window changed")
    if int(outer["display_context_fixed_before_open"]) != 7:
        raise ValueError("outer representative display index changed")
    if int(outer["display_crn_bank"]) != 0:
        raise ValueError("outer representative display CRN bank changed")
    if list(outer["aggregate_window_ids"]) != expected_outer_ids:
        raise ValueError("outer aggregate-window ledger changed")
    if list(outer["aggregate_context_indices"]) != list(range(8)):
        raise ValueError("outer aggregate context indices changed")
    if not bool(outer["open_sealed_ictal_and_reference_segments_together_once"]):
        raise ValueError("sealed ictal/reference one-time access contract changed")
    if not bool(outer["independent_canonical_preprocessing_state_per_sealed_window"]):
        raise ValueError("sealed windows must retain independent canonical preprocessing state")
    if not bool(outer["evaluate_all_eight_contexts_from_single_sealed_ictal_segment"]):
        raise ValueError("outer eight-context evaluation contract changed")
    reference_split = config["reference_path_split"]
    if reference_split["neural_fit_indices_half_open"] != [0, 15]:
        raise ValueError("neural reference-fit path split changed")
    if reference_split["ctx5_checkpoint_validation_indices_half_open"] != [15, 30]:
        raise ValueError("ctx5 reference-validation path split changed")
    if reference_split["unused_development_tail_indices_half_open"] != [30, 40]:
        raise ValueError("development reference tail split changed")
    if reference_split["apply_per_run_then_concatenate_in_order"] != data["development_runs"]:
        raise ValueError("reference split run order changed")
    if not bool(reference_split.get(
        "loro_fit_uses_each_training_run_full_preictal_window", False
    )):
        raise ValueError("LORO fit must use every training run's full 40-s reference")
    if not bool(reference_split.get(
        "loro_validation_uses_left_out_run_full_preictal_window", False
    )):
        raise ValueError("LORO validation must use the left-out run's full 40-s reference")
    if "loro_fit_uses_training_run_fit_indices_only" in reference_split:
        raise ValueError("all-development 15-path split leaked into LORO fitting")

    grid = config["part2"]["grid"]
    if float(config["part1"]["plv_window_s"]) != 4.0:
        raise ValueError("Part-I multiband selection PLV window must remain 4.0 s")
    if float(config["part1"]["plv_overlap"]) != 0.5:
        raise ValueError("Part-I PLV overlap must match D canonical 50 percent")
    if config["part2"].get("loro_candidate_aggregation") != (
        "minimize_worst_fold_then_mean_then_candidate_id"
    ):
        raise ValueError("Part-II LORO aggregation order changed")
    if config["part2"].get("plant_network_source") != (
        "broadband_windowed_plv_median_not_part1_multiband_graph"
    ):
        raise ValueError("Part-II plant adjacency source changed")
    if float(config["part2"].get("plant_plv_window_s", -1.0)) != 2.0:
        raise ValueError("Part-II broadband plant PLV window must remain 2.0 s")
    if float(config["part2"].get("plant_plv_overlap", -1.0)) != 0.5:
        raise ValueError("Part-II broadband plant PLV overlap must remain 50 percent")
    if bool(config["part2"].get("part1_graph_may_be_reused_as_plant_graph", True)):
        raise ValueError("Part-I graph reuse by the Part-II plant is forbidden")
    if int(config["part2"].get("context_samples", -1)) != 512:
        raise ValueError("Part-II context must match D canonical 512 samples")
    if int(config["part2"].get("maximum_validation_horizon_samples", -1)) != 128:
        raise ValueError("Part-II maximum validation horizon must remain 128 samples")
    if int(config["part2"].get("validation_windows", -1)) != 6:
        raise ValueError("Part-II LORO must use six D-canonical validation windows")
    if list(config["part2"].get("selection_context_indices", [])) != list(range(6)):
        raise ValueError("Part-II LORO validation window indices must be 0..5")
    if list(config["part2"].get("rolling_block_candidates", [])) != [
        1, 2, 4, 8, 12, 16, 24, 32,
    ]:
        raise ValueError("Part-II exploratory v2 rolling block candidate grid changed")
    if float(config["part2"].get("rolling_selection_min_correlation", -1.0)) != 0.70:
        raise ValueError("Part-II rolling correlation gate changed")
    if float(config["part2"].get("rolling_selection_max_nrmse", -1.0)) != 0.80:
        raise ValueError("Part-II rolling nRMSE gate changed")
    if int(config["part2"].get("rolling_display_samples", -1)) != 128:
        raise ValueError("Part-II rolling display horizon changed")
    if int(config["contexts"].get("context_samples", -1)) != 256:
        raise ValueError("Part-III context must remain 256 samples")
    count = 1
    for key in (
        "reservoir_sizes", "spectral_radii", "leak_rates", "input_scales",
        "delay_sets_samples", "ridge_alphas",
    ):
        count *= len(grid[key])
    if count != 144 or int(config["part2"]["candidate_count"]) != count:
        raise ValueError("Part-II grid must contain exactly 144 configurations")
    expected_formal_locks = {
        "runner", "selftest", "readme", "source_binding_audit", "package_init",
        "contracts", "staging", "data_model", "control", "training", "outer",
        "pipeline", "renderer_input",
    }
    if set(config.get("formal_implementation_locks", {})) != expected_formal_locks:
        raise ValueError("formal implementation lock ledger is incomplete or changed")
    for name, (path, _digest) in config["formal_implementation_locks"].items():
        try:
            Path(path).resolve().relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(
                f"formal implementation lock escapes project_root: {name}"
            ) from error
    for group in (
        "canonical_locks", "adaptation_locks", "common_renderer_locks",
        "formal_implementation_locks",
    ):
        for name, record in config[group].items():
            if not isinstance(record, list) or len(record) != 2:
                raise ValueError(f"invalid lock record: {group}.{name}")
            if not HEX64.fullmatch(str(record[1])):
                raise ValueError(f"invalid SHA-256: {group}.{name}")

    # These are the only patient/science inputs.  Hash-pinned implementation
    # sources may retain historical directory names, but no artifact from
    # those directories is accepted by a data/model/checkpoint loader.
    safe_science_inputs = (data["raw_zip"], data["split_proposal"])
    for item in safe_science_inputs:
        assert_no_forbidden_science_path(item)


def configure_runtime(config: Mapping[str, Any]) -> None:
    """Redirect all mutable runtime state before third-party imports."""

    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    runtime = Path(str(config["runtime_root"]))
    directories = {
        "MPLCONFIGDIR": runtime / "matplotlib",
        "MNE_HOME": runtime / "mne_home",
        "_MNE_FAKE_HOME_DIR": runtime / "mne_home",
        "MNE_DATA": runtime / "mne_data",
        "JOBLIB_TEMP_FOLDER": runtime / "joblib",
        "TMPDIR": runtime / "tmp",
        "TEMP": runtime / "tmp",
        "TMP": runtime / "tmp",
    }
    for variable, directory in directories.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)
    os.environ["MNE_LOGGING_LEVEL"] = "ERROR"


def assert_bytecode_guard() -> None:
    if not sys.dont_write_bytecode or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("bytecode guard is not active; invoke with python -B")


def verify_lock(path: Path, expected_sha256: str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    observed = sha256_file(resolved)
    if observed != str(expected_sha256).casefold():
        raise PermissionError(
            f"pinned source changed: {resolved}; expected={expected_sha256}, observed={observed}"
        )
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": observed}


def verify_external_locks(config: Mapping[str, Any], *, include_raw_zip: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    shared = config["shared_protocol"]
    rows.append(verify_lock(Path(shared["path"]), str(shared["sha256"])))
    rows.append(
        verify_lock(Path(shared["common_protocol_py"]), str(shared["common_protocol_py_sha256"]))
    )
    rows.append(
        verify_lock(Path(shared["static_self_test_py"]), str(shared["static_self_test_py_sha256"]))
    )
    rows.append(verify_lock(Path(shared["readme"]), str(shared["readme_sha256"])))
    for group in (
        "canonical_locks", "adaptation_locks", "common_renderer_locks",
        "formal_implementation_locks",
    ):
        for name, (path, digest) in config[group].items():
            row = verify_lock(Path(path), str(digest))
            row["lock_name"] = name
            row["lock_group"] = group
            rows.append(row)
    data = config["source_data"]
    rows.append(verify_lock(Path(data["split_proposal"]), str(data["split_proposal_sha256"])))
    if include_raw_zip:
        raise PermissionError(
            "whole-ZIP hashing is forbidden; use zip_central_directory_receipt"
        )
    return rows


def zip_central_directory_receipt(path: Path) -> dict[str, Any]:
    """Inventory ZIP metadata without opening or hashing any member payload."""

    archive_path = Path(path).resolve()
    stat = archive_path.stat()
    with zipfile.ZipFile(archive_path, "r") as archive:
        rows = [
            {
                "filename": info.filename,
                "crc32": int(info.CRC),
                "uncompressed_bytes": int(info.file_size),
                "compressed_bytes": int(info.compress_size),
                "header_offset": int(info.header_offset),
                "compression": int(info.compress_type),
                "is_directory": bool(info.is_dir()),
            }
            for info in archive.infolist()
        ]
    return {
        "path": str(archive_path),
        "archive_bytes": int(stat.st_size),
        "archive_mtime_ns": int(stat.st_mtime_ns),
        "member_count": len(rows),
        "central_directory_rows": rows,
        "central_directory_metadata_sha256": sha256_json(rows),
        "member_payload_opened": False,
        "whole_archive_sha256_computed": False,
    }


def validate_zip_central_directory_contract(
    config: Mapping[str, Any], receipt: Mapping[str, Any]
) -> None:
    data = config["source_data"]
    if int(receipt["archive_bytes"]) != int(data["raw_zip_bytes"]):
        raise PermissionError("raw ZIP byte size changed")
    if str(receipt["central_directory_metadata_sha256"]) != str(
        data["raw_zip_central_directory_metadata_sha256"]
    ):
        raise PermissionError("raw ZIP central-directory metadata changed")
    observed = {
        str(row["filename"]): (
            int(row["crc32"]), int(row["uncompressed_bytes"]),
            int(row["compressed_bytes"]),
        )
        for row in receipt["central_directory_rows"]
    }
    for expected in data["development_member_contract"]:
        key = str(expected["filename"])
        triple = (
            int(expected["crc32"]), int(expected["uncompressed_bytes"]),
            int(expected["compressed_bytes"]),
        )
        if observed.get(key) != triple:
            raise PermissionError(f"development ZIP member contract changed: {key}")


def context_positions(length: int, context_samples: int = 256, future_samples: int = 256) -> tuple[int, ...]:
    if length < 32 * 256:
        raise ValueError("formal context ledger requires a 32-s 256-Hz array")
    start = 2 * int(context_samples)
    stop = int(length) - int(future_samples)
    values = [round(start + index * (stop - start) / 7.0) for index in range(8)]
    if len(set(values)) != 8:
        raise AssertionError("context positions are not unique")
    return tuple(int(value) for value in values)


def fraction_to_quantile(fraction: float) -> float:
    value = float(fraction)
    if not 0.0 < value < 1.0:
        raise ValueError("actuator fraction must lie strictly inside (0,1)")
    # Decimal grid values such as 0.8 must not become
    # 0.19999999999999996 and accidentally admit one extra actuator at an
    # exactly representable sample quantile.
    return round(1.0 - value, 12)


def deterministic_seed(
    protocol_id: str,
    subject: str,
    fold: str,
    stage: str,
    replicate: int,
    base_seed: int,
) -> int:
    """Candidate-independent common-random-number seed."""

    payload = f"{protocol_id}|{subject}|{fold}|{stage}|{replicate}|{int(base_seed)}"
    return int.from_bytes(sha256(payload.encode("utf-8")).digest()[:8], "big") % (2**31 - 1)


def assert_path_within(path: Path, root: Path) -> Path:
    resolved = Path(path).resolve()
    base = Path(root).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as error:
        raise PermissionError(f"path escapes managed root: {resolved}") from error
    return resolved


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    dependency: str | None


def phase_specs(config: Mapping[str, Any]) -> tuple[PhaseSpec, ...]:
    names = [str(value) for value in config["phase_order"]]
    return tuple(
        PhaseSpec(name=name, dependency=None if index == 0 else names[index - 1])
        for index, name in enumerate(names)
    )


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(Path(root).rglob("*"), key=lambda item: str(item).casefold()):
        if path.is_file() and not path.is_symlink():
            yield path
