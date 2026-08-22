from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping


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
    if config.get("schema_version") != "hup065-hup060-sparse-rerun-v1.0":
        raise ValueError("unsupported HUP065 rerun configuration schema")
    if config.get("subject") != "HUP065":
        raise ValueError("this runner is locked to HUP065")
    root = Path(str(config["project_root"]))
    if source is not None and not _same_path(root, source.resolve().parent):
        raise ValueError("config project_root differs from its actual directory")
    science = Path(str(config["science_root"]))
    if science.parent != root or science.name != "science_run":
        raise ValueError("science_root must be the dedicated project_root/science_run")

    shared = config["shared_protocol"]
    for field in ("sha256", "common_protocol_py_sha256", "static_self_test_py_sha256"):
        if not HEX64.fullmatch(str(shared[field])):
            raise ValueError(f"shared_protocol.{field} is not a pinned SHA-256")
    data = config["source_data"]
    if data["expected_channels"] != 64:
        raise ValueError("HUP065 expected channel count must remain 64")
    if data["development_runs"] != ["run-01", "run-02"]:
        raise ValueError("development run roles changed")
    if data["sealed_run"] != "run-03":
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

    fractions = [float(v) for v in config["controller"]["fraction_grid"]]
    if fractions != [0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]:
        raise ValueError("sparse fraction grid changed")
    if [float(v) for v in config["controller"]["tau_grid"]] != [0.0, 0.05, 0.10, 0.20, 0.50]:
        raise ValueError("tau grid changed")
    if [float(v) for v in config["controller"]["gain_grid"]] != [0.35, 0.45, 0.668]:
        raise ValueError("gain grid changed")
    if int(config["controller"]["analytical_candidate_count"]) != 120:
        raise ValueError("analytical grid must contain 8*5*3=120 arms")
    expected_energy = (13.0 / 36.0) * 64.0 * (0.405**2)
    if abs(float(config["controller"]["total_energy_cap"]) - expected_energy) > 1e-12:
        raise ValueError("fixed total energy cap changed")
    if float(config["controller"]["per_actuator_rms_max"]) != 0.405:
        raise ValueError("per-actuator RMS cap changed")
    if float(config["controller"]["amplitude_limit"]) != 1.8:
        raise ValueError("amplitude limit changed")
    if int(config["controller"]["freeze_top_k"]) != 1:
        raise ValueError("exactly one analytical arm must be frozen")
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

    grid = config["part2"]["grid"]
    count = 1
    for key in (
        "reservoir_sizes", "spectral_radii", "leak_rates", "input_scales",
        "delay_sets_samples", "ridge_alphas",
    ):
        count *= len(grid[key])
    if count != 144 or int(config["part2"]["candidate_count"]) != count:
        raise ValueError("Part-II grid must contain exactly 144 configurations")
    for group in ("canonical_locks", "adaptation_locks"):
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
    for group in ("canonical_locks", "adaptation_locks"):
        for name, (path, digest) in config[group].items():
            row = verify_lock(Path(path), str(digest))
            row["lock_name"] = name
            row["lock_group"] = group
            rows.append(row)
    data = config["source_data"]
    rows.append(verify_lock(Path(data["split_proposal"]), str(data["split_proposal_sha256"])))
    if include_raw_zip:
        row = verify_lock(Path(data["raw_zip"]), str(data["raw_zip_sha256"]))
        row["lock_name"] = "raw_zip"
        rows.append(row)
    return rows


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
    # Decimal grid semantics are canonical.  This avoids f=0.80 becoming a
    # binary q=0.199999... and changing a strict-threshold tie decision.
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
