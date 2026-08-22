"""Static common-contract helpers for the HUP060/HUP065/HUP080 rerun.

This module contains no patient-data loader and performs no training.  It is a
small contract layer intended to be imported later by patient-specific
adapters.  The only file it reads is the adjacent ``protocol.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "protocol.json"


class ProtocolError(RuntimeError):
    """Raised when a patient adapter violates the frozen common contract."""


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ProtocolError("unsupported protocol schema")
    if payload.get("status") != "STATIC_COMMON_PROTOCOL_ONLY_NO_SCIENCE_RUN":
        raise ProtocolError("static-only guard is absent")
    return payload


@dataclass(frozen=True)
class PatientAdapter:
    """Shape and split metadata; arrays and model objects remain external."""

    subject: str
    n_channels: int
    actuator_indices: tuple[int, ...]
    development_runs: tuple[str, ...]
    validation_folds: tuple[str, ...]
    final_refit_runs: tuple[str, ...]
    sealed_test_run: str
    reference_train_runs: tuple[str, ...]
    reference_train_window_seconds_half_open: tuple[float, float]
    reference_test_run: str
    reference_test_window_seconds_half_open: tuple[float, float]
    reference_role: str

    @property
    def n_actuators(self) -> int:
        return len(self.actuator_indices)

    @property
    def actuator_fraction(self) -> float:
        return self.n_actuators / self.n_channels


@dataclass(frozen=True)
class CandidateMetrics:
    """LORO analytical summary used before any neural candidate training."""

    candidate_id: str
    integrity_pass: bool
    energy_feasible: bool
    worst_fold_full_gate_b_count: int
    worst_fold_mean_w1: float
    actuator_count: int


def _as_tuple(values: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(values)


def adapter_from_protocol(
    subject: str,
    actuator_indices: Sequence[int],
    *,
    protocol: Mapping[str, Any] | None = None,
) -> PatientAdapter:
    contract = load_protocol() if protocol is None else protocol
    try:
        spec = contract["subjects"][subject]
    except KeyError as error:
        raise ProtocolError(f"unknown subject: {subject}") from error
    adapter = PatientAdapter(
        subject=subject,
        n_channels=int(spec["expected_channels"]),
        actuator_indices=tuple(int(index) for index in actuator_indices),
        development_runs=_as_tuple(spec["development_runs"]),
        validation_folds=_as_tuple(spec["validation_folds"]),
        final_refit_runs=_as_tuple(spec["final_refit_runs"]),
        sealed_test_run=str(spec["sealed_test_run"]),
        reference_train_runs=_as_tuple(spec["reference_train_runs"]),
        reference_train_window_seconds_half_open=tuple(
            float(value) for value in spec["reference_train_window_seconds_half_open"]
        ),
        reference_test_run=str(spec["reference_test_run"]),
        reference_test_window_seconds_half_open=tuple(
            float(value) for value in spec["reference_test_window_seconds_half_open"]
        ),
        reference_role=str(spec["reference_role"]),
    )
    validate_adapter(adapter, protocol=contract)
    return adapter


def validate_adapter(
    adapter: PatientAdapter,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> None:
    contract = load_protocol() if protocol is None else protocol
    spec = contract["subjects"].get(adapter.subject)
    if spec is None:
        raise ProtocolError(f"unknown subject: {adapter.subject}")
    if adapter.n_channels != int(spec["expected_channels"]):
        raise ProtocolError("adapter channel count differs from the subject contract")
    if adapter.n_channels < 1:
        raise ProtocolError("channel count must be positive")
    if len(set(adapter.actuator_indices)) != len(adapter.actuator_indices):
        raise ProtocolError("actuator indices must be unique")
    if not adapter.actuator_indices:
        raise ProtocolError("at least one sparse actuator is required")
    if min(adapter.actuator_indices) < 0 or max(adapter.actuator_indices) >= adapter.n_channels:
        raise ProtocolError("an actuator index is outside the patient channel range")
    maximum = float(contract["sparse_actuation"]["maximum_fraction"])
    # The strict q=1-f linear-quantile rule yields at most floor(f*n) direct
    # outputs when channel scores are distinct.  The cap must never round up
    # above the declared 80% sparse bound.
    maximum_count = math.floor(maximum * adapter.n_channels)
    if adapter.n_actuators > maximum_count:
        raise ProtocolError("adapter violates the frozen sparse-actuation cap")
    fields = {
        "development_runs": adapter.development_runs,
        "validation_folds": adapter.validation_folds,
        "final_refit_runs": adapter.final_refit_runs,
        "sealed_test_run": adapter.sealed_test_run,
        "reference_train_runs": adapter.reference_train_runs,
        "reference_train_window_seconds_half_open": adapter.reference_train_window_seconds_half_open,
        "reference_test_run": adapter.reference_test_run,
        "reference_test_window_seconds_half_open": adapter.reference_test_window_seconds_half_open,
        "reference_role": adapter.reference_role,
    }
    for name, actual in fields.items():
        expected = spec[name]
        if isinstance(actual, tuple):
            expected = tuple(expected)
        if actual != expected:
            raise ProtocolError(f"{name} differs from the formal subject split")
    if adapter.sealed_test_run in adapter.development_runs:
        raise ProtocolError("sealed test run overlaps the development pool")
    if adapter.reference_test_run in adapter.reference_train_runs:
        raise ProtocolError("reference test run overlaps the reference fit pool")
    if adapter.reference_role != "preictal_reference":
        raise ProtocolError("only the formal preictal reference role is permitted")


def validate_runtime_shapes(
    adapter: PatientAdapter,
    *,
    state_shape: Sequence[int],
    control_shape: Sequence[int],
    horizon_samples: int,
) -> None:
    """Validate shape metadata without importing NumPy or reading arrays."""

    if len(state_shape) != 3 or len(control_shape) != 3:
        raise ProtocolError("state and control tensors must be rank three")
    if state_shape[-1] != adapter.n_channels:
        raise ProtocolError("state channel dimension differs from the adapter")
    if control_shape[-1] != adapter.n_actuators:
        raise ProtocolError("control dimension differs from the sparse mask")
    if state_shape[0] != control_shape[0]:
        raise ProtocolError("state and control particle counts differ")
    if state_shape[1] != horizon_samples or control_shape[1] != horizon_samples:
        raise ProtocolError("state/control horizon differs from the frozen contract")


def crn_seed(
    *,
    protocol_id: str,
    subject: str,
    fold: str,
    stage: str,
    replicate: int,
    base_seed: int,
) -> int:
    """Stable CRN seed; candidate identity is deliberately not an input."""

    if replicate < 0:
        raise ProtocolError("replicate must be non-negative")
    key = "|".join(
        [protocol_id, subject, fold, stage, str(int(replicate)), str(int(base_seed))]
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**31 - 1)


def epoch_noise_seed(base_seed: int, epoch_one_based: int, stride: int = 1009) -> int:
    if epoch_one_based < 1:
        raise ProtocolError("epoch index must be one-based and positive")
    return int(base_seed) + int(stride) * int(epoch_one_based)


def validation_checkpoint_score(
    law: float,
    initial_law: float,
    time_w1: float,
    initial_time_w1: float,
    occupation_w1: float,
    initial_occupation_w1: float,
) -> float:
    values = [law, initial_law, time_w1, initial_time_w1, occupation_w1, initial_occupation_w1]
    if not all(math.isfinite(value) for value in values):
        raise ProtocolError("checkpoint score inputs must be finite")
    if min(initial_law, initial_time_w1, initial_occupation_w1) <= 0.0:
        raise ProtocolError("epoch-zero denominators must be positive")
    return (
        0.50 * law / initial_law
        + 0.25 * time_w1 / initial_time_w1
        + 0.25 * occupation_w1 / initial_occupation_w1
    )


def law_noninferior(law: float, initial_law: float, multiplier: float = 1.02) -> bool:
    return math.isfinite(law) and math.isfinite(initial_law) and law <= multiplier * initial_law


def channel_recovery_mask(
    controlled_time_w1: Sequence[float],
    uncontrolled_time_w1: Sequence[float],
    controlled_occupation_w1: Sequence[float],
    uncontrolled_occupation_w1: Sequence[float],
) -> tuple[bool, ...]:
    lengths = {
        len(controlled_time_w1), len(uncontrolled_time_w1),
        len(controlled_occupation_w1), len(uncontrolled_occupation_w1),
    }
    if len(lengths) != 1:
        raise ProtocolError("per-channel W1 vectors must have identical lengths")
    recovered: list[bool] = []
    for controlled_t, uncontrolled_t, controlled_o, uncontrolled_o in zip(
        controlled_time_w1,
        uncontrolled_time_w1,
        controlled_occupation_w1,
        uncontrolled_occupation_w1,
        strict=True,
    ):
        values = (controlled_t, uncontrolled_t, controlled_o, uncontrolled_o)
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ProtocolError("W1 values must be finite and non-negative")
        recovered.append(controlled_t < uncontrolled_t and controlled_o < uncontrolled_o)
    return tuple(recovered)


def normalized_control_energy(
    controls: Iterable[float], amplitude_limit: float = 1.80
) -> tuple[float, float, float]:
    values = tuple(float(value) for value in controls)
    if not values or not all(math.isfinite(value) for value in values):
        raise ProtocolError("controls must be a non-empty finite sequence")
    if amplitude_limit <= 0.0:
        raise ProtocolError("amplitude limit must be positive")
    peak = max(abs(value) for value in values)
    energy = sum((value / amplitude_limit) ** 2 for value in values) / len(values)
    return energy, math.sqrt(energy), peak


def total_energy_cap(n_channels: int, per_actuator_rms_anchor: float = 0.405) -> float:
    """Canonical total cap, independent of the selected actuator count."""

    if n_channels < 1 or per_actuator_rms_anchor <= 0.0:
        raise ProtocolError("energy-cap arguments must be positive")
    return (13.0 / 36.0) * int(n_channels) * float(per_actuator_rms_anchor) ** 2


def energy_feasibility(
    *,
    n_channels: int,
    per_actuator_rms: Sequence[float],
    total_energy: float,
    peak: float,
    saturation_fraction: float,
) -> bool:
    """Frozen Gate-C-style energy and amplitude feasibility test."""

    values = [*per_actuator_rms, total_energy, peak, saturation_fraction]
    if not per_actuator_rms or not all(math.isfinite(float(value)) for value in values):
        raise ProtocolError("energy diagnostics must be finite and non-empty")
    return (
        max(float(value) for value in per_actuator_rms) <= 0.405
        and float(total_energy) <= total_energy_cap(n_channels)
        and float(peak) <= 1.80
        and 0.0 <= float(saturation_fraction) < 0.01
    )


def candidate_rank_key(candidate: CandidateMetrics) -> tuple[Any, ...]:
    """Ascending key for the unique LORO analytical top-1 prescreen."""

    numeric = (candidate.worst_fold_mean_w1,)
    if not all(math.isfinite(value) for value in numeric):
        raise ProtocolError("candidate metrics must be finite")
    if candidate.worst_fold_full_gate_b_count < 0:
        raise ProtocolError("Gate-B count must be non-negative")
    if candidate.actuator_count < 1:
        raise ProtocolError("candidate must use at least one actuator")
    passes = candidate.integrity_pass and candidate.energy_feasible
    return (
        0 if passes else 1,
        -candidate.worst_fold_full_gate_b_count,
        candidate.worst_fold_mean_w1,
        candidate.actuator_count,
        candidate.candidate_id,
    )


def centrality_quantile_for_fraction(
    fraction: float, *, protocol: Mapping[str, Any] | None = None
) -> float:
    contract = load_protocol() if protocol is None else protocol
    grid = tuple(float(value) for value in contract["sparse_actuation"]["candidate_fraction_grid"])
    if not any(math.isclose(float(fraction), value, rel_tol=0.0, abs_tol=1.0e-12) for value in grid):
        raise ProtocolError("actuator fraction is outside the frozen grid")
    # Decimal-grid semantics are intentional: e.g. f=.80 means q=.20, not
    # the binary float immediately below .20, which can select one extra node.
    return round(1.0 - float(fraction), 12)


def validate_part3_tuning_values(
    *,
    actuator_fraction: float,
    graph_diffusion_time: float,
    analytical_base_gain: float,
    protocol: Mapping[str, Any] | None = None,
) -> None:
    contract = load_protocol() if protocol is None else protocol
    centrality_quantile_for_fraction(actuator_fraction, protocol=contract)
    allowed = contract["patient_specific_tuning_whitelist"]["part3_allowed_values"]
    for name, value in {
        "graph_diffusion_time": graph_diffusion_time,
        "analytical_base_gain": analytical_base_gain,
    }.items():
        if not any(
            math.isclose(float(value), float(candidate), rel_tol=0.0, abs_tol=1.0e-12)
            for candidate in allowed[name]
        ):
            raise ProtocolError(f"{name} is outside the frozen values")


def assert_authority_inventory_unchanged(
    before: Mapping[str, tuple[int, str]],
    after: Mapping[str, tuple[int, str]],
) -> None:
    """Compare externally produced recursive inventories without reading roots."""

    if dict(before) != dict(after):
        raise ProtocolError("NO_GO: a read-only HUP060 authority inventory changed")
    for path, (size, digest) in before.items():
        if not path or int(size) < 0 or len(str(digest)) != 64:
            raise ProtocolError("authority inventory row is malformed")


def validate_output_target(
    subject: str,
    target_name: str,
    *,
    target_exists: bool,
    protocol: Mapping[str, Any] | None = None,
) -> None:
    contract = load_protocol() if protocol is None else protocol
    expected = {
        "HUP065": "HUP065_hup060_sparse_rerun_v1",
        "HUP080": "HUP080_hup060_sparse_rerun_v1",
    }.get(subject)
    if expected is None or target_name != expected:
        raise ProtocolError("NO_GO: output target is outside the two allowed directories")
    if target_exists:
        raise ProtocolError("NO_GO: output target already exists")


def validate_tuning_keys(
    part: str,
    keys: Iterable[str],
    *,
    protocol: Mapping[str, Any] | None = None,
) -> None:
    contract = load_protocol() if protocol is None else protocol
    whitelist = contract["patient_specific_tuning_whitelist"]
    allowed = set(whitelist.get(part, ())) | set(whitelist["dimension_only"])
    forbidden = set(whitelist["forbidden"])
    requested = set(keys)
    denied = sorted((requested - allowed) | (requested & forbidden))
    if denied:
        raise ProtocolError(f"patient-specific tuning keys are not permitted: {denied}")


__all__ = [
    "CandidateMetrics",
    "PatientAdapter",
    "ProtocolError",
    "adapter_from_protocol",
    "candidate_rank_key",
    "centrality_quantile_for_fraction",
    "channel_recovery_mask",
    "crn_seed",
    "energy_feasibility",
    "epoch_noise_seed",
    "law_noninferior",
    "load_protocol",
    "normalized_control_energy",
    "total_energy_cap",
    "assert_authority_inventory_unchanged",
    "validate_adapter",
    "validate_runtime_shapes",
    "validate_tuning_keys",
    "validate_part3_tuning_values",
    "validate_output_target",
    "validation_checkpoint_score",
]
