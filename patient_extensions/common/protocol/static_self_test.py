"""Static self-test: no D-drive reads, patient-data reads, or science runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from common_protocol import (
    CandidateMetrics,
    ProtocolError,
    adapter_from_protocol,
    assert_authority_inventory_unchanged,
    candidate_rank_key,
    centrality_quantile_for_fraction,
    channel_recovery_mask,
    crn_seed,
    energy_feasibility,
    epoch_noise_seed,
    load_protocol,
    normalized_control_energy,
    total_energy_cap,
    validate_output_target,
    validate_part3_tuning_values,
    validate_runtime_shapes,
    validate_tuning_keys,
    validation_checkpoint_score,
)


ROOT = Path(__file__).resolve().parent
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expect_protocol_error(function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except ProtocolError:
        return
    raise AssertionError(f"{function.__name__} did not reject an invalid contract")


def main() -> int:
    protocol = load_protocol()
    assert protocol["guardrails"]["d_drive_read_by_static_self_test"] is False
    assert protocol["guardrails"]["terminal_data_read_by_static_self_test"] is False
    assert protocol["guardrails"]["training_authorized"] is False
    assert protocol["guardrails"]["overleaf_write_authorized"] is False

    locks = protocol["canonical_hup060_locks"]
    assert len(locks) >= 10
    repository_root = Path(__file__).resolve().parents[3]
    for item in locks.values():
        Path(item["path"]).resolve().relative_to(repository_root)
        assert SHA256_RE.fullmatch(str(item["sha256"]))

    schedule = protocol["common_model"]["part3"]["fixed_training_schedule"]
    assert protocol["common_model"]["part1"]["plv_window_seconds"] == 4.0
    assert protocol["common_model"]["part1"]["plv_window_overlap_fraction"] == 0.50
    assert protocol["common_model"]["part1"]["network_role"] == "actuator_ranking_and_sparse_mask_selection_only"
    assert protocol["common_model"]["part2"]["plant_plv_window_seconds"] == 2.0
    assert protocol["common_model"]["part2"]["part1_graph_may_be_reused_as_plant_graph"] is False
    assert protocol["common_model"]["part2"]["context_samples"] == 512
    assert protocol["common_model"]["part2"]["direct_validation_windows"] == 6
    assert protocol["common_model"]["part2"]["direct_validation_maximum_horizon_samples"] == 128
    assert protocol["common_model"]["part2"]["diffusion_alpha_loro_aggregation"].startswith("mean_swd_ratio_over_six_windows_per_fold")
    assert protocol["common_model"]["part2"]["adjacency_binding"]["world_adjacency"] == "model.adjacency"
    assert protocol["common_model"]["part2"]["rolling_gate"]["block_candidates_samples"] == [4, 8, 12, 16, 24, 32]
    assert protocol["common_model"]["part2"]["rolling_gate"]["selection_rule"] == "largest_eligible_block"
    assert protocol["implementation_stages"]["distinct_part1_selection_and_part2_plant_graphs_required"] is True
    assert protocol["implementation_stages"]["joblib_model_class_identity_required_before_dump_and_after_load"] is True
    assert protocol["common_model"]["part2"]["loro_candidate_aggregation"] == "minimize_worst_fold_then_mean_then_candidate_id"
    stages = schedule["teacher_stages"]
    assert schedule["teacher_epochs_total"] == 1050
    assert schedule["s3_to_s4_modern_actor_initialization_seed"] == 20260921
    assert sum(int(stage["epochs"]) for stage in stages) == 1050
    assert [stage["stage_id"] for stage in stages] == schedule["teacher_formal_stage_ids"]
    assert schedule["discarded_robust_branch_executed"] is False
    assert schedule["wgan_epochs"] == 40
    assert protocol["common_model"]["part3"]["actor"]["runtime_control_projection_or_rescaling"] is False
    actor_contract = protocol["common_model"]["part3"]["actor"]
    assert actor_contract["plant_control_graph_diffusion_time_source"] == "patient_specific_candidate_graph_diffusion_time"
    assert actor_contract["plant_control_channel_map_override_allowed"] is False
    assert actor_contract["extra_feedback_only_graph_diffusion_substitution_allowed"] is False
    assert actor_contract["stepper_class"] == "FrozenIctalGraphRCBatchStepper"
    assert actor_contract["stepper_construction"] == "direct_D_canonical_constructor_without_increment_or_persistence_transform"
    assert abs(float(actor_contract["canonical_diffusion_scale"]) - 0.79451175) < 1.0e-12
    executed_locks = protocol["public_portability"][
        "executed_canonical_hup060_lock_sha256"
    ]
    assert executed_locks["part3_causal_particle_rollout"] == "47b3328091a85f851fc8e42275c24847564f2795c7292f53f51c37746832305c"
    assert executed_locks["part3_riccati"] == "748567f5fb616a641c3916a69dbb9385cf2a3cd0e64d890b49301bb3244c3ae2"
    for name in ("part3_causal_particle_rollout", "part3_riccati"):
        assert sha256(Path(locks[name]["path"])) == locks[name]["sha256"]
    assert protocol["energy"]["hard_projection_or_rescaling_allowed"] is False
    assert "never alter" in protocol["energy"]["energy_violation_action"]
    outer = protocol["implementation_stages"]["outer_evaluation"]
    assert outer["aggregate_context_indices"] == list(range(8))
    assert outer["common_random_number_banks"] == 3
    assert outer["required_unique_context_bank_evaluations_per_channel"] == 24
    assert outer["aggregation_requires_complete_cartesian_product"] is True
    assert (outer["fixed_display_context_index"], outer["fixed_display_crn_bank"]) == (7, 0)
    assert outer["reference_and_ictal_windows_use_independent_canonical_preprocessing_state"] is True
    assert protocol["selection_rule"]["analytical_feasibility_prefilter_excludes_gate_a_and_full_gate_count"] is True

    grid = protocol["sparse_actuation"]["candidate_fraction_grid"]
    assert grid == [0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]
    assert max(grid) == protocol["sparse_actuation"]["maximum_fraction"]
    assert centrality_quantile_for_fraction(0.35, protocol=protocol) == 0.65
    assert centrality_quantile_for_fraction(0.80, protocol=protocol) == 0.20

    adapters = {
        "HUP060": adapter_from_protocol("HUP060", tuple(range(13)), protocol=protocol),
        "HUP065": adapter_from_protocol("HUP065", tuple(range(51)), protocol=protocol),
        "HUP080": adapter_from_protocol("HUP080", tuple(range(76)), protocol=protocol),
    }
    assert adapters["HUP065"].n_channels == 64
    assert adapters["HUP080"].n_channels == 96
    expect_protocol_error(
        adapter_from_protocol, "HUP065", tuple(range(52)), protocol=protocol
    )
    expect_protocol_error(
        adapter_from_protocol, "HUP080", tuple(range(77)), protocol=protocol
    )
    assert all(adapter.reference_role == "preictal_reference" for adapter in adapters.values())
    assert adapters["HUP065"].reference_train_window_seconds_half_open == (30.0, 70.0)
    assert adapters["HUP080"].reference_test_window_seconds_half_open == (75.0, 115.0)
    for adapter in adapters.values():
        validate_runtime_shapes(
            adapter,
            state_shape=(32, 256, adapter.n_channels),
            control_shape=(32, 256, adapter.n_actuators),
            horizon_samples=256,
        )
    expect_protocol_error(
        validate_runtime_shapes,
        adapters["HUP080"],
        state_shape=(32, 256, 64),
        control_shape=(32, 256, adapters["HUP080"].n_actuators),
        horizon_samples=256,
    )

    seed_a = crn_seed(
        protocol_id=protocol["protocol_id"], subject="HUP065",
        fold="leave-run-01-out", stage="validation", replicate=0,
        base_seed=protocol["common_random_numbers"]["analytical_validation_base_seed"],
    )
    seed_b = crn_seed(
        protocol_id=protocol["protocol_id"], subject="HUP065",
        fold="leave-run-01-out", stage="validation", replicate=0,
        base_seed=protocol["common_random_numbers"]["analytical_validation_base_seed"],
    )
    assert seed_a == seed_b
    assert epoch_noise_seed(20260921, 1) == 20261930
    assert epoch_noise_seed(20261011, 1) == 20262020

    score = validation_checkpoint_score(1.0, 1.0, 0.8, 1.0, 0.6, 1.0)
    assert abs(score - 0.85) < 1.0e-12
    assert channel_recovery_mask([0.5, 1.1], [1.0, 1.0], [0.7, 0.8], [1.0, 1.0]) == (True, False)
    energy, rms, peak = normalized_control_energy([0.0, 0.9, -0.9])
    assert abs(energy - (0.0 + 0.25 + 0.25) / 3.0) < 1.0e-12
    assert abs(rms * rms - energy) < 1.0e-12 and peak == 0.9
    expected_cap_96 = (13.0 / 36.0) * 96 * 0.405**2
    assert abs(total_energy_cap(96) - expected_cap_96) < 1.0e-12
    assert energy_feasibility(
        n_channels=96,
        per_actuator_rms=[0.405, 0.20],
        total_energy=expected_cap_96,
        peak=1.80,
        saturation_fraction=0.009,
    )
    assert not energy_feasibility(
        n_channels=96,
        per_actuator_rms=[0.406],
        total_energy=0.1,
        peak=1.0,
        saturation_fraction=0.0,
    )

    more_gate_b = CandidateMetrics("b", True, True, 52, 0.95, 30)
    lower_w1 = CandidateMetrics("a", True, True, 50, 0.80, 20)
    failed_energy = CandidateMetrics("c", True, False, 60, 0.50, 18)
    assert min([lower_w1, more_gate_b, failed_energy], key=candidate_rank_key) == more_gate_b

    validate_tuning_keys("part2", ["reservoir_size", "effective_diffusion_multiplier"], protocol=protocol)
    validate_tuning_keys(
        "part3",
        ["actuator_fraction", "graph_diffusion_time", "analytical_base_gain", "n_channels"],
        protocol=protocol,
    )
    validate_part3_tuning_values(
        actuator_fraction=0.35,
        graph_diffusion_time=0.10,
        analytical_base_gain=0.668,
        protocol=protocol,
    )
    expect_protocol_error(validate_tuning_keys, "part3", ["actor_learning_rate"], protocol=protocol)
    expect_protocol_error(validate_tuning_keys, "part3", ["critic_architecture"], protocol=protocol)
    expect_protocol_error(validate_tuning_keys, "part1", ["test_selected_mask"], protocol=protocol)

    inventory = {"PAPER_FINAL_VERSION.json": (2119, "a" * 64)}
    assert_authority_inventory_unchanged(inventory, dict(inventory))
    expect_protocol_error(
        assert_authority_inventory_unchanged,
        inventory,
        {"PAPER_FINAL_VERSION.json": (2119, "b" * 64)},
    )
    validate_output_target(
        "HUP065", "HUP065_hup060_sparse_rerun_v1", target_exists=False, protocol=protocol
    )
    validate_output_target(
        "HUP080", "HUP080_hup060_sparse_rerun_v1", target_exists=False, protocol=protocol
    )
    expect_protocol_error(
        validate_output_target,
        "HUP080",
        "HUP080_hup060_sparse_rerun_v1",
        target_exists=True,
        protocol=protocol,
    )

    receipt = {
        "status": "STATIC_SELF_TEST_PASS",
        "science_run": False,
        "d_drive_read": False,
        "patient_data_read": False,
        "protocol_sha256": sha256(ROOT / "protocol.json"),
        "common_module_sha256": sha256(ROOT / "common_protocol.py"),
        "supported_channel_adapters": {key: value.n_channels for key, value in adapters.items()},
        "checks": [
            "canonical SHA syntax", "formal split metadata", "sparse cap",
            "distinct Part-I selection and Part-II plant graphs",
            "36/64/96-channel adapter shapes", "CRN determinism",
            "checkpoint score identity", "six-metric Gate-B contract",
            "fixed S0-S6 1050 plus WGAN40 schedule", "fixed total energy cap",
            "no runtime control projection or rescaling",
            "D-canonical plant graph diffusion and direct frozen RC stepper",
            "outer 8-context by 3-bank Cartesian completeness and fixed display",
            "analytical top-1 selection", "tuning whitelist",
            "HUP060 before/after authority inventory", "exclusive output targets"
        ]
    }
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
