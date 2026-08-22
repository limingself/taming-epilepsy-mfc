#!/usr/bin/env python
from __future__ import annotations

import ast
from hashlib import sha256
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import zipfile

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from h080.contracts import (  # noqa: E402
    assert_bytecode_guard, configure_runtime, context_positions, load_config,
    phase_specs, sha256_file, validate_config, zip_central_directory_receipt,
)
from h080.control import (  # noqa: E402
    _standardized_reference, build_d_canonical_control_stack,
    frozen_gate_b_components, frozen_prerank_eligibility,
    network_for_fraction, unprojected_rollout,
)
from h080.data_model import (  # noqa: E402
    _rolling_block_metrics,
    activate_canonical, array_sha256, assert_live_part2_model_identity,
    part2_context_block, part3_context_block,
    select_loro_rolling_block,
)
from h080.outer import validate_outer_metric_cartesian, verify_run04_go  # noqa: E402
from h080.renderer_input import freeze_common_renderer_input  # noqa: E402
from h080.staging import PhaseStore, atomic_json, validate_completed_phase  # noqa: E402
from h080.training import (  # noqa: E402
    FORMAL_STAGE_IDS, _all_development_reference_pools,
    validate_training_schedule,
)


ROOT = Path(__file__).resolve().parent


def _sha_json(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _assert_exploratory_v2_delta(config: dict[str, object]) -> dict[str, object]:
    """Validate bundled hash-only parent-v1 provenance without opening its stage."""
    amendment = config["exploratory_v2_delta"]
    receipt_path = Path(amendment["v1_failure_public_receipt"])
    assert receipt_path == ROOT / "results" / "v1_rolling_no_go.json"
    assert receipt_path.is_file()
    assert sha256_file(receipt_path) == amendment["v1_failure_public_receipt_sha256"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["parent_v1_executed_config_sha256"] == amendment["base_config_sha256"]
    assert receipt["parent_v1_implementation_freeze_sha256"] == (
        amendment["base_implementation_freeze_sha256"]
    )
    assert receipt["parent_v1_candidates_samples"] == [4, 8, 12, 16, 24, 32]
    assert receipt["minimum_correlation"] == 0.7
    assert receipt["maximum_nrmse"] == 0.8
    assert receipt["joint_fold_rule"] == "every_development_loro_fold_must_pass"
    assert receipt["outcome"] == "NO_GO_before_part3"
    assert receipt["success_marker_present"] is False
    assert receipt["formal_part3_phase_present"] is False
    assert receipt["outer_run04_seen"] is False
    assert receipt["private_failed_stage_required_for_public_verification"] is False
    assert receipt["failure_artifacts_read_by_v2_science"] is False
    inventory = receipt["failure_artifact_inventory"]
    assert len(inventory) == 13
    for name, row in inventory.items():
        assert name and int(row["bytes"]) > 0
        digest = str(row["sha256"])
        assert len(digest) == 64
        int(digest, 16)

    assert config["part2"]["rolling_block_candidates"] == [
        1, 2, 4, 8, 12, 16, 24, 32,
    ]
    assert amendment["base_candidates"] == receipt["parent_v1_candidates_samples"]
    assert amendment["extended_candidates"] == config["part2"]["rolling_block_candidates"]
    assert amendment["thresholds_unchanged"] == {
        "minimum_correlation": 0.7, "maximum_nrmse": 0.8,
    }
    assert amendment["joint_fold_rule_unchanged"] == (
        "every_development_loro_fold_must_pass"
    )
    assert amendment["classification"] == "post_v1_phase02_no_go_protocol_amendment"
    assert amendment["amended_json_pointer"] == "/part2/rolling_block_candidates"
    assert amendment["only_added_candidates"] == [1, 2]
    assert amendment["thresholds_or_selection_rule_changed"] is False
    assert amendment["parent_no_go_remains_valid"] is True
    assert amendment["may_be_described_as_original_preregistered_replication"] is False
    assert amendment["run04_seen_before_amendment"] is False
    public_delta = receipt["exploratory_v2"]
    assert public_delta["extended_candidates_samples"] == amendment["extended_candidates"]
    assert public_delta["thresholds_changed"] is False
    assert public_delta["joint_fold_rule_changed"] is False
    assert public_delta["parent_no_go_remains_valid"] is True
    assert public_delta["may_be_described_as_original_preregistered_replication"] is False
    assert amendment["v1_failure_evidence_role"] == (
        "AUDIT_ONLY_DO_NOT_REUSE_FOR_SCIENCE_SELECTION_OR_TRAINING"
    )
    assert amendment["v2_may_read_v1_failure_as_science_input"] is False
    assert not Path(config["science_root"]).exists()

    execution_sources = [
        ROOT / "runner.py", ROOT / "h080" / "data_model.py",
        ROOT / "h080" / "control.py", ROOT / "h080" / "training.py",
        ROOT / "h080" / "outer.py", ROOT / "h080" / "pipeline.py",
        ROOT / "h080" / "renderer_input.py",
    ]
    joined = "\n".join(path.read_text(encoding="utf-8") for path in execution_sources)
    assert str(receipt_path) not in joined
    assert "failure_artifact_inventory" not in joined
    assert "v1_failure_public_receipt" not in joined
    assert "hup065" not in joined.casefold()
    assert '"overleaf_write": True' not in joined
    assert amendment["authorized_write_roots"] == [
        config["science_root"], config["runtime_root"],
    ]
    return {
        "post_v1_no_go_amendment_disclosed": True,
        "sole_scientific_delta": "rolling candidates add 1 and 2 samples",
        "unique_delta_receipt_validated": True,
        "v1_failure_artifacts_hash_bound": len(inventory),
        "v1_failure_artifacts_used_as_science_input": False,
        "private_v1_stage_opened": False,
        "v2_science_root_exists": False,
        "run04_role_unchanged_and_sealed": amendment["sealed_run04_role_unchanged"],
        "hup065_or_overleaf_write_route_present": False,
    }


def _assert_source_routes() -> dict[str, object]:
    from h080 import control, data_model, outer, pipeline, renderer_input, training

    sources = {
        "control": inspect.getsource(control),
        "training": inspect.getsource(training),
        "outer": inspect.getsource(outer),
        "pipeline": inspect.getsource(pipeline),
        "renderer_input": inspect.getsource(renderer_input),
        "data_model": inspect.getsource(data_model),
    }
    joined = "\n".join(sources.values())
    forbidden = ("budgeted_particle_rollout", "causal_energy_projection_scale")
    for token in forbidden:
        assert token not in joined, f"projection path remains reachable: {token}"
    assert "ctrl.train_wgan(" not in sources["training"]
    for forbidden_hash in (
        "bc073733efe3aa7b1a85e1b4674a8400e950b5d2695c7091695da9588604779e",
        "47557ec71dec40723ec7d2cf0c41f4f2f742bbea854fbafb79ef875b696aa846",
        "ad3cbe84f61980f63ffae1a0522767e6540be1f25bb05bc33b39ce8978b0b149",
    ):
        assert forbidden_hash not in (ROOT / "config.json").read_text(encoding="utf-8")
    for token in (
        '"mfc_pipeline.causal_ltv_particle_rollout": config["canonical_locks"]["canonical_particle_rollout"]',
        '"mfc_pipeline.causal_ltv_riccati": config["canonical_locks"]["canonical_riccati"]',
        '"mfc_pipeline.square_wave_mfc": config["canonical_locks"]["canonical_square_wave"]',
        "controller rebound {module_name} outside D canonical",
    ):
        assert token in sources["control"], token
    for token in (
        "world.control_channel_map =",
        "graph_aware_feedback_channel_map(",
        "build_calibrated_stepper(",
    ):
        assert token not in sources["control"], f"altered plant construction remains: {token}"
    for token in (
        "control_graph_diffusion_time=float(config.graph_diffusion_time)",
        "stepper = ctrl.FrozenIctalGraphRCBatchStepper(",
        "non-identity multistep calibration is forbidden",
    ):
        assert token in sources["control"], token
    for token in (
        "20261011 + 101 * (bank + 1)",
        "20261011 + 50000 + step",
        "20261011 + 1009 * epoch",
        "20261011 + 100000 * epoch + critic_step",
        "clip_grad_norm_(actor.parameters(), 1.0)",
        "clip_grad_norm_(critic.parameters(), 5.0)",
        "fallback_update_promoted\": False",
    ):
        assert token in sources["training"], f"formal WGAN contract missing: {token}"
    for token in (
        "seed = 20260921 + 1009 * (absolute + 1)",
        "seed = 20260922",
        '"teacher_training_noise_seed_rule": "20260921 + 1009*(absolute_epoch+1)"',
        '"teacher_validation_noise_seed": 20260922',
    ):
        assert token in sources["training"], f"formal teacher CRN contract missing: {token}"
    phase_calls = {
        "01_prepare_development": "prepare_development(",
        "02_loro_part1_part2": "loro_part1_part2(",
        "03_loro_analytical_top1": "loro_analytical_top1(",
        "04_final_refit": "final_refit(",
        "12_wgan40_ctx5_select": "run_atomic_wgan40_ctx5(",
        "13_ctx6_terminal_veto": "ctx6_terminal_veto(",
        "14_freeze_outer": "freeze_outer_contract(",
        "15_run04_once": "open_and_evaluate_run04_once(",
        "16_common_renderer_input_freeze": "freeze_common_renderer_input(",
        "17_inventory_after": "compare_inventories(",
    }
    for phase, call in phase_calls.items():
        assert f'phase == "{phase}"' in sources["pipeline"]
        assert call in sources["pipeline"], f"{phase} lacks a real backend call"
    teacher_phases = [
        "05_teacher_s0", "06_teacher_s1", "07_teacher_s2s",
        "08_teacher_s3", "09_teacher_s4", "10_teacher_s5", "11_teacher_s6",
    ]
    for phase in teacher_phases:
        assert f'"{phase}"' in sources["pipeline"]
    assert "run_atomic_teacher_stage(" in sources["pipeline"]
    for token in (
        "_validated_parent_state(",
        "strict=True",
        'len(modern_template.state_dict()) != 44',
        'len(actor.state_dict()) != 44',
        'sum(item["epochs_executed"] for item in stage_receipts) != 1050',
    ):
        assert token in sources["training"], token
    assert "render_channel_preview" not in joined
    assert "matplotlib" not in joined
    assert sources["outer"].index('output / "ACCESS_STARTED.json"') < sources["outer"].index(
        "loader.load("
    )
    outer_open_source = inspect.getsource(outer.open_and_evaluate_run04_once)
    assert outer_open_source.count("loader.load(") == 2
    assert "start_s=120.0, duration_s=32.0" in outer_open_source
    assert "start_s=75.0, duration_s=40.0" in outer_open_source
    assert '"burn_in_start_s": 100.0' in outer_open_source
    assert '"burn_in_start_s": 55.0' in outer_open_source
    assert "sha256_file(archive_path)" not in sources["outer"]
    assert "include_raw_zip=False" in sources["pipeline"]
    assert "display_local_index=display_local_index" in sources["outer"]
    assert "aggregate_context_indices" in sources["outer"]
    assert "context_bank_evaluations" in sources["outer"]
    assert '["worst_fold_objective", "mean_objective", "candidate_id"]' in sources["data_model"]
    assert "context, future, boundary = part2_context_block(" in sources["data_model"]
    assert "part2_module.forecast_positions" in sources["data_model"]
    assert "def context_block(" not in sources["data_model"]
    for token in (
        "def build_part1_selection_network(",
        '"network_role": np.asarray("part1_selection_only")',
        "def build_part2_plant_network(",
        '"network_role": np.asarray("part2_plant_only")',
        "canonical_pipeline.compute_adjacency(",
        "canonical_part2_pipeline_module=part2_pipeline",
        'fold_part2_networks[fold_id]["adjacency"]',
        "canonical_network_module=canonical_network",
        "assert_live_part2_model_identity(model, config)",
        'fold_part1_selection_networks" / f"{fold_id}.npz',
        'fold_part2_plant_networks" / f"{fold_id}.npz',
        'raise RuntimeError("no rolling block passed every development LORO fold")',
        "selected_block_samples, rolling_joint_rows = select_loro_rolling_block(",
        'output / "diffusion_alpha_loro_fold_summary.csv"',
        '"worst_fold_mean_swd_ratio", "overall_mean_swd_ratio"',
        'alpha_fold_summary["validation_window_rows"].eq(6).all()',
    ):
        assert token in sources["data_model"], token
    assert 'refit_dir / "part1_selection_network.npz"' in sources["training"]
    assert 'output / "part2_plant_network.npz"' in sources["control"]
    formal_inputs_source = inspect.getsource(training._formal_training_inputs)
    seed_index = formal_inputs_source.index("torch.manual_seed(modern_template_seed)")
    stack_index = formal_inputs_source.index("stack = ctrl.build_control_stack(")
    assert seed_index < stack_index
    assert 'config["controller"]["s3_to_s4_modern_template_initialization_seed"]' in formal_inputs_source
    assert "new_markov_initialization_sha256" in sources["training"]
    loro_source = inspect.getsource(control.loro_analytical_top1)
    assert "fit_reference = _standardized_reference(model, arrays, training_runs)" in loro_source
    assert "neural_fit_indices_half_open" not in loro_source
    assert "fit_reference_path_count" in loro_source
    eligibility_source = inspect.getsource(control.frozen_prerank_eligibility)
    assert "plant_pass" not in inspect.signature(
        control.frozen_prerank_eligibility
    ).parameters
    assert "gate_a_integrity_pass and gate_c_pass" in eligibility_source
    for token in (
        '"plant_diagnostic_used_for_eligibility": False',
        '"gate_a_static_integrity_pass": gate_a_integrity_pass',
        '"analytical_eligibility_pass": frozen_prerank_eligibility(',
        'output / "analytical_trajectory_safety_metrics.csv"',
        '"gate_c_trajectory_safety_pass": gate_c',
    ):
        assert token in sources["control"], token
    assert 'group["plant_four_inequality_diagnostic_pass"]' in loro_source
    assert 'ranking["all_folds_analytical_eligible"]' in loro_source
    assert 'ranking["all_fold_plant_four_inequality_diagnostic_pass"]' not in loro_source
    assert "world_adjacency, np.asarray(model.adjacency" in sources["control"]
    for token in (
        'schema_version=np.asarray("ofrc-final-candidate-v1")',
        '"status": "PENDING_INDEPENDENT_RESULT_AUDIT_DO_NOT_RENDER"',
        '"display_window_id": display_window_id',
        '"aggregate_window_ids": aggregate_window_ids',
    ):
        assert token in sources["renderer_input"], token
    return {
        "no_projection_tokens": list(forbidden),
        "real_phase_calls": phase_calls,
        "embedded_renderer_absent": True,
        "access_marker_precedes_edf_load": True,
        "outer_all_windows_x_banks": True,
        "renderer_binding_pending_independent_audit": True,
        "atomic_teacher_phases": teacher_phases,
        "strict_parent_and_44_key_migration": True,
        "d_canonical_particle_riccati_square_wave_binding": True,
    }


def _assert_instrumented_zip_firewall(runtime: Path) -> dict[str, object]:
    probe_root = runtime / "selftest_zip_probe"
    probe_root.mkdir(parents=True, exist_ok=True)
    path = probe_root / "synthetic.zip"
    if path.exists():
        path.unlink()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("dev.txt", b"development-only")
    original_open = zipfile.ZipFile.open
    member_open_calls = 0

    def blocked_member_open(*args, **kwargs):
        nonlocal member_open_calls
        member_open_calls += 1
        raise AssertionError("central-directory audit attempted to open a member payload")

    zipfile.ZipFile.open = blocked_member_open
    try:
        receipt = zip_central_directory_receipt(path)
    finally:
        zipfile.ZipFile.open = original_open
    assert member_open_calls == 0
    assert receipt["member_payload_opened"] is False
    assert receipt["whole_archive_sha256_computed"] is False
    path.unlink()
    probe_root.rmdir()
    return {
        "instrumented_member_open_calls": member_open_calls,
        "whole_archive_hash_computed": False,
    }


def _assert_reference_isolation(config: dict) -> dict[str, object]:
    import numpy as np

    arrays: dict[str, object] = {}
    runs = config["source_data"]["development_runs"]
    for run_index, run in enumerate(runs):
        paths = np.empty((40, 256, 2), dtype=np.float64)
        for path_index in range(40):
            paths[path_index] = 1000.0 * run_index + path_index
        arrays[f"preictal_reference_{run.replace('-', '_')}"] = paths.reshape(-1, 2)

    class IdentityScaler:
        @staticmethod
        def transform(value):
            return np.asarray(value)

    model = SimpleNamespace(transform=SimpleNamespace(scaler=IdentityScaler()))
    fit, validation, receipt = _all_development_reference_pools(
        config, model, arrays, runs
    )
    assert fit.shape == validation.shape == (45, 256, 2)
    assert set(np.unique(fit)).isdisjoint(set(np.unique(validation)))
    assert receipt["run_order"] == ["run-01", "run-02", "run-03"]
    assert len(receipt["per_run"]) == 3
    assert all(row["fit_shape"] == [15, 256, 2] for row in receipt["per_run"])
    assert all(
        row["ctx5_validation_shape"] == [15, 256, 2]
        for row in receipt["per_run"]
    )
    assert receipt["fit_combined_sha256"] == array_sha256(fit)
    loro_fit = _standardized_reference(model, arrays, runs[:2])
    loro_validation = _standardized_reference(model, arrays, [runs[2]])
    assert loro_fit.shape == (80, 256, 2)
    assert loro_validation.shape == (40, 256, 2)
    return {
        "fit_shape": list(fit.shape), "validation_shape": list(validation.shape),
        "fit_sha256": sha256(fit.tobytes()).hexdigest(),
        "validation_sha256": sha256(validation.tobytes()).hexdigest(),
        "per_run_receipt_count": len(receipt["per_run"]),
        "loro_full_fit_shape": list(loro_fit.shape),
        "loro_full_left_out_validation_shape": list(loro_validation.shape),
        "split_each_run_then_concatenate": True, "overlap": False,
    }


def _assert_context_roles() -> dict[str, object]:
    import numpy as np

    sequence = np.arange(8192 * 2, dtype=np.float64).reshape(8192, 2)
    calls = []

    def fake_forecast_positions(
        length, *, context_samples, maximum_horizon, guard_samples, count,
    ):
        calls.append({
            "length": int(length), "context_samples": int(context_samples),
            "maximum_horizon": int(maximum_horizon),
            "guard_samples": int(guard_samples), "count": int(count),
        })
        return np.unique(np.rint(np.linspace(
            context_samples, length - maximum_horizon - guard_samples, count
        )).astype(np.int64))

    part2_context, part2_future, part2_boundary = part2_context_block(
        sequence, 0, fake_forecast_positions
    )
    part3_context, part3_future, part3_boundary = part3_context_block(sequence, 0)
    _part2_last_context, _part2_last_future, part2_last_boundary = (
        part2_context_block(sequence, 5, fake_forecast_positions)
    )
    _part3_last_context, _part3_last_future, part3_last_boundary = (
        part3_context_block(sequence, 7)
    )
    assert part2_boundary == part3_boundary == 512
    assert part2_last_boundary == 8192 - 128
    assert part3_last_boundary == 8192 - 256
    assert part2_context.shape == (512, 2)
    assert part2_future.shape == (128, 2)
    assert part3_context.shape == (256, 2)
    assert part3_future.shape == (256, 2)
    assert all(call == {
        "length": 8192, "context_samples": 512,
        "maximum_horizon": 128, "guard_samples": 0, "count": 6,
    } for call in calls)
    return {
        "part2_context_samples": 512,
        "part2_maximum_future_samples": 128,
        "part2_validation_windows": 6,
        "part3_context_samples": 256,
        "part3_horizon_samples": 256,
        "part2_latest_boundary": part2_last_boundary,
        "part3_latest_boundary": part3_last_boundary,
        "separate_helpers": True,
        "d_forecast_positions_arguments_verified": True,
    }


def _assert_distinct_graph_roles(config: dict) -> dict[str, object]:
    import numpy as np
    from h080 import data_model

    calls = []

    class FakeCanonicalNetwork:
        @staticmethod
        def windowed_plv_median(
            values, sfreq, *, window_s, overlap, time_axis,
        ):
            del sfreq, overlap, time_axis
            calls.append(float(window_s))
            channels = int(np.asarray(values).shape[1])
            result = np.full((channels, channels), float(window_s))
            np.fill_diagonal(result, 1.0)
            return result

        @staticmethod
        def mst_proportional_graph(similarity, *, density):
            assert float(density) == 0.10
            return np.asarray(similarity, dtype=np.float64)

        @staticmethod
        def centrality_composite(adjacency, *, weights):
            del adjacency, weights
            score = np.arange(4, dtype=np.float64)
            return {
                "composite": score, "degree": score,
                "betweenness": score, "eigenvector": score,
            }

    class FakePart2Pipeline:
        @staticmethod
        def compute_adjacency(sequences, sfreq):
            del sfreq
            calls.append(2.0)
            channels = int(np.asarray(sequences[0]).shape[1])
            result = np.full((channels, channels), 2.0, dtype=np.float64)
            np.fill_diagonal(result, 1.0)
            return result

    original_activate = data_model.activate_canonical
    original_bandpass = data_model.bandpass_for_plv
    data_model.activate_canonical = lambda _config: (
        None, FakeCanonicalNetwork(), None, FakePart2Pipeline()
    )
    data_model.bandpass_for_plv = lambda values, sfreq, band: np.asarray(values)
    try:
        sequence = np.zeros((1024, 4), dtype=np.float64)
        selection = data_model.build_part1_selection_network(
            config, [sequence], 256.0
        )
        plant = data_model.build_part2_plant_network(
            config, [sequence], 256.0
        )
    finally:
        data_model.activate_canonical = original_activate
        data_model.bandpass_for_plv = original_bandpass
    assert str(selection["network_role"]) == "part1_selection_only"
    assert str(plant["network_role"]) == "part2_plant_only"
    assert calls.count(4.0) == 5
    assert calls.count(2.0) == 1
    assert array_sha256(selection["adjacency"]) != array_sha256(plant["adjacency"])
    return {
        "part1_multiband_window_seconds": 4.0,
        "part1_band_count": 5,
        "part2_broadband_window_seconds": 2.0,
        "part1_graph_reused_by_part2": False,
        "distinct_adjacency_hashes": True,
    }


def _assert_live_model_identity_guard() -> dict[str, object]:
    module_name = "mfc_pipeline.part2_state_dependent_rc_sde"
    original = sys.modules.get(module_name)

    class CurrentModel:
        pass

    class StaleModel:
        pass

    sys.modules[module_name] = SimpleNamespace(ResidualGraphRCSDE=CurrentModel)
    try:
        assert_live_part2_model_identity(CurrentModel())
        try:
            assert_live_part2_model_identity(StaleModel())
        except PermissionError:
            stale_rejected = True
        else:
            stale_rejected = False
    finally:
        if original is None:
            del sys.modules[module_name]
        else:
            sys.modules[module_name] = original
    assert stale_rejected
    return {
        "current_class_identity_accepted": True,
        "stale_class_identity_rejected": True,
        "module_reimport_while_live_model_forbidden": True,
    }


def _assert_canonical_idempotent_reuse(runtime: Path) -> dict[str, object]:
    module_names = (
        "mfc_pipeline.data", "mfc_pipeline.network",
        "mfc_pipeline.part2_state_dependent_rc_sde",
        "mfc_pipeline.part2_data_pipeline",
    )
    originals = {name: sys.modules.get(name) for name in module_names}
    with TemporaryDirectory(dir=runtime / "tmp") as temporary:
        root = Path(temporary)
        paths = {}
        for index, name in enumerate(module_names):
            path = root / f"module_{index}.py"
            path.write_text(f"# synthetic canonical identity {name}\n", encoding="utf-8")
            paths[name] = path

        class CurrentModel:
            pass

        modules = {
            name: SimpleNamespace(__file__=str(paths[name]))
            for name in module_names
        }
        modules["mfc_pipeline.part2_state_dependent_rc_sde"].ResidualGraphRCSDE = (
            CurrentModel
        )
        synth_config = {
            "canonical_locks": {
                "data_core": [str(paths[module_names[0]]), sha256_file(paths[module_names[0]])],
                "network_core": [str(paths[module_names[1]]), sha256_file(paths[module_names[1]])],
                "part2_core": [str(paths[module_names[2]]), sha256_file(paths[module_names[2]])],
                "part2_data_pipeline": [
                    str(paths[module_names[3]]), sha256_file(paths[module_names[3]])
                ],
            }
        }
        try:
            sys.modules.update(modules)
            first = activate_canonical(synth_config)
            model = CurrentModel()
            assert_live_part2_model_identity(model, synth_config)
            second = activate_canonical(synth_config)
            assert all(left is right for left, right in zip(first, second))
            assert model.__class__ is second[2].ResidualGraphRCSDE
        finally:
            for name, original in originals.items():
                if original is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original
    return {
        "valid_live_modules_reused_by_identity": True,
        "live_model_class_identity_survives_reactivation": True,
        "purge_on_valid_live_module_set": False,
    }


def _assert_rolling_screen() -> dict[str, object]:
    import numpy as np

    folds = ["fold-1", "fold-2", "fold-3"]
    candidates = [1, 2, 4, 8, 12, 16, 24, 32]
    rows = []
    for fold in folds:
        for block in candidates:
            jointly_eligible = block <= 4
            fold_pass = jointly_eligible or (block == 8 and fold != "fold-3")
            rows.append({
                "fold_id": fold, "block_samples": block,
                "median_correlation": 0.75 if fold_pass else 0.69,
                "median_nrmse": 0.70,
                "rolling_gate_pass": fold_pass,
            })
    selected, summary = select_loro_rolling_block(
        rows, fold_ids=folds, candidates=candidates,
        minimum_correlation=0.70, maximum_nrmse=0.80,
    )
    assert selected == 4
    assert len(summary) == 8
    assert len(rows) == 24
    try:
        select_loro_rolling_block(
            rows[:-1], fold_ids=folds, candidates=candidates,
            minimum_correlation=0.70, maximum_nrmse=0.80,
        )
    except PermissionError:
        missing_rejected = True
    else:
        missing_rejected = False
    assert missing_rejected
    try:
        select_loro_rolling_block(
            [*rows, dict(rows[0])], fold_ids=folds, candidates=candidates,
            minimum_correlation=0.70, maximum_nrmse=0.80,
        )
    except PermissionError:
        duplicate_rejected = True
    else:
        duplicate_rejected = False
    assert duplicate_rejected
    nan_rows = [dict(row) for row in rows]
    nan_rows[0]["median_correlation"] = float("nan")
    nan_selected, nan_summary = select_loro_rolling_block(
        nan_rows, fold_ids=folds, candidates=candidates,
        minimum_correlation=0.70, maximum_nrmse=0.80,
    )
    assert nan_selected == 4
    assert not next(
        row for row in nan_summary if row["block_samples"] == 1
    )["all_folds_pass"]
    no_pass = [
        {**row, "median_correlation": 0.0, "rolling_gate_pass": False}
        for row in rows
    ]
    try:
        select_loro_rolling_block(
            no_pass, fold_ids=folds, candidates=candidates,
            minimum_correlation=0.70, maximum_nrmse=0.80,
        )
    except RuntimeError:
        empty_pool_no_go = True
    else:
        empty_pool_no_go = False
    assert empty_pool_no_go

    class _IdentityScaler:
        @staticmethod
        def transform(value):
            return np.asarray(value, dtype=np.float64)

    class _SyntheticModel:
        def __init__(self):
            self.transform = SimpleNamespace(scaler=_IdentityScaler())
            self.calls: list[int] = []

        def forecast_direct(self, context, steps):
            self.calls.append(int(steps))
            return np.repeat(np.asarray(context[-1:, :]), int(steps), axis=0)

    sequence = np.sin(
        np.arange(1700 * 96, dtype=np.float64).reshape(1700, 96) / 97.0
    )
    positions = [512, 700, 888, 1076, 1264, 1452]
    executor_rows = {}
    executor_call_counts = {}
    for block in (1, 2):
        model = _SyntheticModel()
        details = _rolling_block_metrics(
            model, sequence, positions, context_samples=512,
            block_samples=block, total_samples=128,
            channels=[f"S{index + 1:02d}" for index in range(96)],
        )
        assert len(details) == 6 * 96
        assert {row["block_samples"] for row in details} == {block}
        assert all(np.isfinite(row["channel_window_nrmse"]) for row in details)
        assert len(model.calls) == 6 * ((128 + block - 1) // block)
        assert all(1 <= steps <= block for steps in model.calls)
        executor_rows[str(block)] = len(details)
        executor_call_counts[str(block)] = len(model.calls)
    return {
        "largest_jointly_eligible": selected,
        "candidate_grid": candidates,
        "new_one_and_two_sample_candidates_exercised": True,
        "fold_candidate_cartesian_rows": len(rows),
        "complete_fold_candidate_cartesian_required": True,
        "missing_cell_rejected": True,
        "duplicate_cell_rejected": True,
        "nonfinite_metric_fails_closed": True,
        "empty_eligible_pool_no_go": True,
        "block_1_and_2_executor_rows_each": executor_rows,
        "block_1_and_2_executor_call_counts": executor_call_counts,
        "outer_or_test_used": False,
    }


def _assert_gate_b() -> dict[str, object]:
    import numpy as np

    gate = frozen_gate_b_components(
        np.asarray([1.0, 0.40]), np.asarray([0.80, 0.30]),
        np.asarray([1.0, 0.30]), np.asarray([0.80, 0.20]),
        np.asarray([0.05, 0.05]), np.asarray([1.2, 1.2]),
    )
    improved = np.asarray([True, True])
    full = np.asarray(gate["full_gate_b_pass"], dtype=bool)
    assert improved.tolist() == [True, True]
    assert full.tolist() == [False, True]
    assert not bool(gate["time_absolute_pass"][0])
    assert not bool(gate["occupation_absolute_pass"][0])
    plant_four_inequality_diagnostic_pass = False
    assert not plant_four_inequality_diagnostic_pass
    assert frozen_prerank_eligibility(
        gate_a_integrity_pass=True, gate_c_pass=True
    )
    assert not frozen_prerank_eligibility(
        gate_a_integrity_pass=False, gate_c_pass=True
    )
    assert not frozen_prerank_eligibility(
        gate_a_integrity_pass=True, gate_c_pass=False
    )
    return {
        "both_improved": improved.tolist(), "full_six_gate_b": full.tolist(),
        "distinguishes_improvement_from_full_gate": True,
        "eligibility_requires_gate_a_static_integrity_and_gate_c_safety": True,
        "plant_four_inequality_false_still_eligible": True,
        "plant_is_diagnostic_only": True,
        "gate_b_is_ranked_not_feasibility_prefilter": True,
    }


def _assert_unprojected_runtime() -> dict[str, object]:
    sentinel = object()

    class FakeControl:
        calls = 0

        def empirical_fp_rollout(self, stepper, actor, initial, noise):
            del stepper, actor, initial, noise
            self.calls += 1
            return sentinel

    fake = FakeControl()
    result = unprojected_rollout(fake, object(), object(), object(), object())
    assert result is sentinel and fake.calls == 1
    return {"empirical_fp_calls": fake.calls, "projection_calls": 0}


def _assert_dynamic96_d_canonical_stack() -> dict[str, object]:
    import numpy as np
    import torch

    sentinel_control_map = object()
    observed: dict[str, object] = {}

    class FakeWorld:
        def __init__(
            self, model, selected, sampling_rate_hz, *,
            control_graph_diffusion_time, preserve_physical_control_residual,
            dtype, device,
        ):
            del sampling_rate_hz, dtype, device
            self.n_channels = int(model.n_channels)
            self.selected = np.asarray(selected, dtype=np.int64)
            self.control_graph_diffusion_time = float(control_graph_diffusion_time)
            self.control_channel_map = sentinel_control_map
            self.adjacency = torch.as_tensor(
                model.adjacency, dtype=torch.float64
            )
            observed["preserve_physical_control_residual"] = bool(
                preserve_physical_control_residual
            )

    class FakeAdapter:
        def __init__(self, world, *, control_step_scale, dtype, device):
            del control_step_scale, dtype, device
            self.world = world
            self.n_channels = int(world.n_channels)
            self.actuator_dim = int(world.selected.size)
            self.state_dim = 7

        def initial_state_from_context(self, item):
            del item
            return torch.zeros(self.state_dim, dtype=torch.float64)

    class FakeStepper:
        def __init__(self, world, adapter, *, diffusion_scale):
            self.world = world
            self.adapter = adapter
            self.diffusion_scale = float(diffusion_scale)

    FakeStepper.__module__ = "mfc_pipeline.causal_ltv_particle_rollout"

    class FakeActor:
        def __init__(
            self, stepper, reference_mean, reference_variance, reference_scale,
            base_gain, selected, **kwargs,
        ):
            del stepper, reference_mean, reference_variance, reference_scale, kwargs
            self.base_gain = base_gain
            self.selected = np.asarray(selected, dtype=np.int64)

    class FakeControl:
        TorchGraphRCSDE = FakeWorld
        FrozenGraphRCMarkovAdapter = FakeAdapter
        FrozenIctalGraphRCBatchStepper = FakeStepper
        StructuredSplineCovarianceActor = FakeActor

        @staticmethod
        def reference_statistics(reference):
            assert tuple(reference.shape) == (15, 256, 96)
            return (
                torch.zeros(96, dtype=torch.float64),
                torch.ones(96, dtype=torch.float64),
                torch.ones(96, dtype=torch.float64),
                torch.ones(96, dtype=torch.float64),
            )

        @staticmethod
        def build_markov_normalization(adapter, initial_states, reference):
            assert len(initial_states) == 2
            assert tuple(reference.shape) == (15, 256, 96)
            return (
                torch.zeros(adapter.state_dim, dtype=torch.float64),
                torch.ones(adapter.state_dim, dtype=torch.float64),
            )

        @staticmethod
        def analytical_weighted_ridge_gain(
            adapter, channel_weights, reference_scale, **kwargs,
        ):
            assert "feedback_channel_map" not in kwargs
            observed["ridge_kwargs"] = sorted(kwargs)
            return torch.zeros(
                (adapter.actuator_dim, adapter.n_channels), dtype=torch.float64
            )

    mask = np.zeros(96, dtype=bool)
    mask[:76] = True
    network = {"target_mask": mask}
    controller_config = SimpleNamespace(
        sampling_rate_hz=256.0,
        graph_diffusion_time=0.2,
        control_step_scale=0.5,
        diffusion_scale=0.79451175,
        base_gain_ridge=0.001,
        base_gain_scale=0.668,
        horizon=256,
        basis_count=8,
        decoded_hidden_size=32,
        amplitude_limit=1.8,
        actuator_alpha=0.25,
        residual_scale=0.0,
        local_gain_initial_fraction=0.0,
        local_gain_maximum_fraction=0.0,
        markov_residual_scale=0.1,
        markov_hidden_size=32,
    )
    stack = build_d_canonical_control_stack(
        FakeControl(), SimpleNamespace(n_channels=96, adjacency=np.eye(96)), network,
        np.zeros((2, 256, 96), dtype=np.float64),
        np.zeros((15, 256, 96), dtype=np.float64), controller_config,
    )
    world, adapter, stepper, *_, actor = stack
    assert int(world.n_channels) == 96
    assert int(adapter.actuator_dim) == 76
    assert tuple(actor.base_gain.shape) == (76, 96)
    assert world.control_channel_map is sentinel_control_map
    assert float(world.control_graph_diffusion_time) == 0.2
    assert type(stepper).__module__ == "mfc_pipeline.causal_ltv_particle_rollout"
    assert observed["preserve_physical_control_residual"] is True
    return {
        "patient_channels": 96,
        "direct_actuators": 76,
        "base_gain_shape": [76, 96],
        "candidate_tau_in_plant": True,
        "world_inherits_model_normalized_adjacency": True,
        "canonical_control_map_preserved": True,
        "canonical_stepper_used": True,
        "extra_feedback_map_argument": False,
        "runtime_projection_or_rescaling": False,
    }


def _assert_mask_grid(config: dict) -> dict[str, object]:
    import numpy as np

    network = {
        "centrality_score": np.arange(96, dtype=np.float64),
        "adjacency": np.eye(96), "target_mask": np.zeros(96, dtype=bool),
    }
    counts = {}
    for fraction in config["controller"]["fraction_grid"]:
        selected = network_for_fraction(network, float(fraction))["target_mask"]
        count = int(np.asarray(selected, dtype=bool).sum())
        assert 1 <= count <= 76
        counts[f"{float(fraction):.2f}"] = count
    assert max(counts.values()) <= int(0.80 * 96)
    return counts


def _assert_atomic_store(config: dict, runtime: Path) -> dict[str, object]:
    with TemporaryDirectory(dir=runtime / "tmp") as temporary:
        root = Path(temporary) / "science"
        store = PhaseStore(root, "a" * 64)
        store.initialize_root()
        with store.publish("synthetic_phase", None) as stage:
            assert stage is not None
            atomic_json(stage / "payload.json", {"real": True})
        receipt = validate_completed_phase(store.path("synthetic_phase"))
        with store.publish("synthetic_phase", None) as resumed:
            assert resumed is None
        return {
            "atomic_publish": True, "resume_validates_manifest": True,
            "artifact_count": int(receipt["artifact_count"]),
        }


def _assert_run04_go(config: dict, runtime: Path) -> dict[str, object]:
    with TemporaryDirectory(dir=runtime / "tmp") as temporary:
        root = Path(temporary)
        freeze_dir = root / "freeze"
        freeze_dir.mkdir()
        atomic_json(freeze_dir / "outer_freeze.json", {"frozen": True})
        go = {
            "decision": "GO",
            "outer_freeze_sha256": sha256_file(freeze_dir / "outer_freeze.json"),
            "config_sha256": sha256_file(ROOT / "config.json"),
            "acknowledge_retrospective_one_time_sealed_segment_access": True,
            "acknowledge_no_reselection_rescue_or_success_claim": True,
        }
        go_path = root / "go.json"
        atomic_json(go_path, go)
        assert verify_run04_go(config, freeze_dir, go_path) == go
        bad = dict(go)
        bad["outer_freeze_sha256"] = "0" * 64
        atomic_json(go_path, bad)
        try:
            verify_run04_go(config, freeze_dir, go_path)
        except PermissionError:
            rejected = True
        else:
            rejected = False
        assert rejected
        return {"valid_go_accepted": True, "hash_mismatch_rejected": True}


def _assert_outer_cartesian() -> dict[str, object]:
    import pandas as pd

    window_ids = [f"outer-context-{index:02d}" for index in range(8)]
    rows = [
        {
            "channel_index": channel,
            "window_id": window_id,
            "context_index": context,
            "crn_bank": bank,
        }
        for channel in range(4)
        for window_id, context in zip(window_ids, range(8))
        for bank in range(3)
    ]
    complete = pd.DataFrame(rows)
    receipt = validate_outer_metric_cartesian(
        complete, window_ids, list(range(8)), 3, channel_count=4
    )
    assert receipt["unique_window_bank_pairs_per_channel"] == 24
    try:
        validate_outer_metric_cartesian(
            complete.iloc[:-1].copy(), window_ids, list(range(8)), 3,
            channel_count=4,
        )
    except RuntimeError:
        missing_rejected = True
    else:
        missing_rejected = False
    duplicate = complete.copy()
    duplicate.iloc[-1] = duplicate.iloc[-2]
    try:
        validate_outer_metric_cartesian(
            duplicate, window_ids, list(range(8)), 3, channel_count=4
        )
    except RuntimeError:
        duplicate_rejected = True
    else:
        duplicate_rejected = False
    assert missing_rejected and duplicate_rejected
    return {
        **receipt, "missing_cell_rejected": True, "duplicate_cell_rejected": True,
        "display_window_id": "outer-context-07", "display_crn_bank": 0,
    }


def _load_renderer_module(config: dict) -> object:
    source = Path(config["common_renderer_locks"]["renderer"][0])
    spec = importlib.util.spec_from_file_location("hup080_common_renderer_contract", source)
    if spec is None or spec.loader is None:
        raise ImportError(source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_renderer_candidate_contract(config: dict, runtime: Path) -> dict[str, object]:
    import numpy as np
    import pandas as pd

    with TemporaryDirectory(dir=runtime / "tmp") as temporary:
        root = Path(temporary)
        outer = root / "outer"
        frozen = root / "frozen"
        outer.mkdir()
        window_ids = [f"outer-context-{index:02d}" for index in range(8)]
        channels = np.asarray([f"CH{index:03d}" for index in range(96)])
        direct = np.zeros(96, dtype=bool)
        direct[:76] = True
        observed = np.arange(32 * 96, dtype=np.float64).reshape(32, 96) / 1000.0
        free = np.repeat(observed[None, :, :], 2, axis=0)
        controlled = free * 0.5
        reference = np.repeat((observed * 0.25)[None, :, :], 2, axis=0)
        np.savez_compressed(
            outer / "run04_frozen_rollout.npz",
            free_standardized=free, controlled_standardized=controlled,
            reference_standardized=reference, observed_standardized=observed,
            channels=channels, direct_mask=direct,
            display_window_id=np.asarray("outer-context-07"),
            display_crn_bank=np.asarray(0), aggregate_window_ids=np.asarray(window_ids),
        )
        gate = pd.DataFrame({
            "channel_index": np.arange(96), "channel": channels,
            "direct_actuated": direct,
            "time_relative_reduction_pass": np.ones(96, dtype=bool),
            "occupation_relative_reduction_pass": np.ones(96, dtype=bool),
            "time_absolute_pass": np.ones(96, dtype=bool),
            "occupation_absolute_pass": np.ones(96, dtype=bool),
            "mean_absolute_error_pass": np.ones(96, dtype=bool),
            "symmetric_sd_ratio_pass": np.ones(96, dtype=bool),
            "full_gate_b_pass": np.ones(96, dtype=bool),
            "safety_gate_pass": np.ones(96, dtype=bool),
            "full_gate_pass": np.ones(96, dtype=bool),
        })
        gate.to_csv(
            outer / "run04_all96_aggregate_gate_vectors.csv", index=False,
            encoding="utf-8-sig",
        )
        atomic_json(
            outer / "run04_outer_safety.json",
            {
                "display_window_id": "outer-context-07", "display_crn_bank": 0,
                "aggregate_window_ids": window_ids, "crn_banks": 3,
                "context_bank_evaluations": 24,
            },
        )
        atomic_json(
            outer / "run04_retrospective_report.json",
            {
                "outer_result_used_for_success_or_rescue": False,
                "outer_gate_and_safety_fail_closed_over_all_windows_x_banks": True,
                "display_window_id": "outer-context-07", "display_crn_bank": 0,
                "aggregate_window_ids": window_ids,
                "aggregate_context_indices": list(range(8)),
                "metrics": {
                    "declared_contexts": 8, "crn_banks": 3,
                    "context_bank_evaluations": 24,
                },
                "outer_freeze_sha256": "a" * 64,
                "candidate_id": "synthetic-frozen-top1",
                "checkpoint_sha256": "b" * 64,
            },
        )
        freeze_common_renderer_input(config, outer, frozen)
        candidate_path = frozen / "common_renderer_input.npz"
        pending_path = frozen / "PENDING_FINAL_CANDIDATE_BINDING.json"
        renderer = _load_renderer_module(config)
        renderer_config = renderer._load_config(
            Path(config["common_renderer_locks"]["config"][0])
        )
        candidate = renderer._load_candidate(candidate_path, renderer_config)
        assert candidate["subject_id"] == "HUP080"
        assert int(candidate["direct_mask"].sum()) == 76
        pending = json.loads(pending_path.read_text(encoding="utf-8"))
        assert pending["status"] != "GO"
        assert pending["display_window_id"] == "outer-context-07"
        assert pending["display_crn_bank"] == 0
        assert pending["aggregate_window_ids"] == window_ids
        try:
            renderer._load_binding(
                pending_path, input_path=candidate_path, candidate=candidate
            )
        except renderer.ContractError:
            pending_rejected = True
        else:
            pending_rejected = False
        assert pending_rejected
        assert "matplotlib" not in sys.modules
        return {
            "renderer_v1_candidate_loaded": True,
            "pending_binding_rejected": True,
            "direct_count": 76, "indirect_count": 20,
            "display_window_id": "outer-context-07", "display_crn_bank": 0,
            "aggregate_window_count": 8,
        }


def main() -> int:
    config = load_config()
    configure_runtime(config)
    assert_bytecode_guard()
    validate_config(config, ROOT / "config.json")
    assert not Path(config["science_root"]).exists(), "science_root exists during static build test"
    assert "mne" not in sys.modules
    runtime = Path(config["runtime_root"])
    for variable in (
        "MPLCONFIGDIR", "MNE_HOME", "_MNE_FAKE_HOME_DIR", "MNE_DATA",
        "JOBLIB_TEMP_FOLDER", "TMPDIR", "TEMP", "TMP",
    ):
        target = Path(os.environ[variable]).resolve()
        target.relative_to(runtime.resolve())
        assert target.is_dir()
        assert target != Path.home() / ".mne"

    protocol_path = Path(config["shared_protocol"]["path"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert sha256_file(protocol_path) == config["shared_protocol"]["sha256"]
    assert config["shared_protocol"]["executed_sha256"] == (
        "a59141b4e239eb6676c915108aaa82ba330cf9574a40158c4702479f6d2f1399"
    )
    assert protocol["common_model"]["part3"]["fixed_training_schedule"]["teacher_epochs_total"] == 1050
    assert protocol["common_model"]["part1"]["plv_window_seconds"] == 4.0
    assert protocol["common_model"]["part2"]["plant_plv_window_seconds"] == 2.0
    assert protocol["common_model"]["part2"]["part1_graph_may_be_reused_as_plant_graph"] is False
    assert protocol["common_model"]["part2"]["context_samples"] == 512
    assert protocol["common_model"]["part2"]["direct_validation_windows"] == 6
    assert protocol["common_model"]["part2"]["diffusion_alpha_loro_aggregation"] == (
        "mean_swd_ratio_over_six_windows_per_fold_then_minimize_worst_fold_then_overall_mean_then_alpha"
    )
    assert protocol["common_model"]["part2"]["adjacency_binding"] == {
        "raw_part2_plant_adjacency": "model.adjacency_input",
        "normalized_model_adjacency": "model.adjacency",
        "world_adjacency": "model.adjacency",
    }
    parent_rolling = protocol["common_model"]["part2"]["rolling_gate"]
    assert parent_rolling["block_candidates_samples"] == [4, 8, 12, 16, 24, 32]
    assert parent_rolling["median_correlation_min"] == 0.70
    assert parent_rolling["median_nrmse_max"] == 0.80
    assert parent_rolling["selection_rule"] == "largest_eligible_block"
    assert parent_rolling["no_eligible_block"] == "NO_GO_before_part3"
    assert protocol["common_random_numbers"]["teacher_training_noise_rule"] == (
        "20260921+1009*(absolute_epoch+1)"
    )
    assert protocol["common_random_numbers"]["teacher_validation_seed"] == 20260922
    assert protocol["energy"]["hard_projection_or_rescaling_allowed"] is False
    assert protocol["energy"]["energy_is_feasibility_gate"] is True
    outer_protocol = protocol["implementation_stages"]["outer_evaluation"]
    assert protocol["implementation_stages"]["canonical_runtime_module_binding"] == (
        "single hash-and-path-validated live module identity; no purge/reimport while model objects exist"
    )
    assert protocol["implementation_stages"][
        "joblib_model_class_identity_required_before_dump_and_after_load"
    ] is True
    assert outer_protocol["aggregate_context_indices"] == list(range(8))
    assert outer_protocol["common_random_number_banks"] == 3
    assert outer_protocol["required_unique_context_bank_evaluations_per_channel"] == 24
    assert outer_protocol["fixed_display_context_index"] == 7
    assert outer_protocol["fixed_display_crn_bank"] == 0
    assert outer_protocol[
        "reference_and_ictal_windows_use_independent_canonical_preprocessing_state"
    ] is True
    assert protocol["selection_rule"][
        "analytical_feasibility_prefilter_excludes_gate_a_and_full_gate_count"
    ] is True
    assert protocol["authority_freeze"][
        "hup060_patient_arrays_models_checkpoints_and_outputs_may_be_training_input"
    ] is False
    for name, path_field, sha_field in (
        ("common_protocol", "common_protocol_py", "common_protocol_py_sha256"),
        ("shared_selftest", "static_self_test_py", "static_self_test_py_sha256"),
        ("shared_readme", "readme", "readme_sha256"),
    ):
        assert sha256_file(Path(config["shared_protocol"][path_field])) == config["shared_protocol"][sha_field], name
    for name, (path, digest) in config["adaptation_locks"].items():
        assert not str(path).startswith("D:"), name
        assert sha256_file(Path(path)) == digest, name
    for name, (path, digest) in config["common_renderer_locks"].items():
        assert not str(path).startswith("D:"), name
        assert sha256_file(Path(path)) == digest, name
    expected_formal_locks = {
        "runner", "selftest", "readme", "source_binding_audit", "package_init",
        "contracts", "staging", "data_model", "control", "training", "outer",
        "pipeline", "renderer_input",
    }
    assert set(config["formal_implementation_locks"]) == expected_formal_locks
    for name, (path, digest) in config["formal_implementation_locks"].items():
        resolved = Path(path).resolve()
        resolved.relative_to(ROOT.resolve())
        assert resolved.is_file(), name
        assert sha256_file(resolved) == digest, name

    py_files = sorted([*ROOT.glob("*.py"), *(ROOT / "h080").glob("*.py")])
    for path in py_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    stages = validate_training_schedule(config)
    assert tuple(item["stage_id"] for item in stages) == FORMAL_STAGE_IDS
    assert sum(int(item["epochs"]) for item in stages) == 1050
    assert context_positions(8192) == tuple(round(512 + i * (7936 - 512) / 7) for i in range(8))
    assert [item.name for item in phase_specs(config)] == config["phase_order"]
    assert config["ctx6_veto_rule"]["minimum_full_six_gate_b_channel_count"] == 1
    assert config["ctx6_veto_rule"]["may_rank_rescue_or_replace"] is False
    assert config["gate_b"]["definition"] == "six_frozen_components"
    assert config["gate_b"][
        "aggregate_both_w1_improvement_or_gate_b_may_be_analytical_feasibility_prefilter"
    ] is False
    eligibility = config["analytical_eligibility_contract"]
    assert eligibility["gate_a_name"] == "static_protocol_integrity"
    assert eligibility["gate_c_name"] == "trajectory_safety"
    assert eligibility["requires_gate_a_and_gate_c"] is True
    assert eligibility["plant_four_inequality_status"] == "diagnostic_only"
    assert eligibility["plant_may_filter_rank_or_veto_analytical_arms"] is False
    assert eligibility["trajectory_safety_values_reported"] is True
    assert config["outer_evaluation"]["display_window_id"] == "outer-context-07"
    assert config["outer_evaluation"]["display_crn_bank"] == 0
    assert config["outer_evaluation"][
        "independent_canonical_preprocessing_state_per_sealed_window"
    ] is True
    assert config["outer_evaluation"]["aggregate_window_ids"] == [
        f"outer-context-{index:02d}" for index in range(8)
    ]
    assert config["outer_evaluation"]["aggregate_context_indices"] == list(range(8))
    assert config["part1"]["plv_window_s"] == 4.0
    assert config["part1"]["plv_overlap"] == 0.5
    assert config["part2"]["loro_candidate_aggregation"] == (
        "minimize_worst_fold_then_mean_then_candidate_id"
    )
    assert config["part2"]["context_samples"] == 512
    assert config["part2"]["maximum_validation_horizon_samples"] == 128
    assert config["part2"]["validation_windows"] == 6
    assert config["part2"]["selection_context_indices"] == list(range(6))
    assert config["part2"]["plant_plv_window_s"] == 2.0
    assert config["part2"]["plant_plv_overlap"] == 0.5
    assert config["part2"]["part1_graph_may_be_reused_as_plant_graph"] is False
    assert config["part2"]["rolling_block_candidates"] == [
        1, 2, 4, 8, 12, 16, 24, 32,
    ]
    assert config["part2"]["rolling_block_candidates"] == sorted(set(
        parent_rolling["block_candidates_samples"] + [1, 2]
    ))
    assert all(
        isinstance(value, int) and value > 0 and value <= 128
        for value in config["part2"]["rolling_block_candidates"]
    )
    assert config["part2"]["rolling_selection_min_correlation"] == 0.7
    assert config["part2"]["rolling_selection_max_nrmse"] == 0.8
    assert config["controller"][
        "s3_to_s4_modern_template_initialization_seed"
    ] == 20260921
    assert config["controller"]["teacher_training_noise_seed_rule"] == (
        "20260921 + 1009*(absolute_epoch+1)"
    )
    assert config["controller"]["teacher_validation_noise_seed"] == 20260922
    assert config["controller"]["wgan_validation_noise_seed"] == 20260922
    assert config["reference_path_split"][
        "loro_fit_uses_each_training_run_full_preictal_window"
    ] is True
    assert config["contexts"]["context_samples"] == 256
    checks = {
        "exploratory_v2_delta": _assert_exploratory_v2_delta(config),
        "source_routes": _assert_source_routes(),
        "zip_firewall": _assert_instrumented_zip_firewall(runtime),
        "reference_isolation": _assert_reference_isolation(config),
        "context_roles": _assert_context_roles(),
        "distinct_graph_roles": _assert_distinct_graph_roles(config),
        "live_part2_model_identity": _assert_live_model_identity_guard(),
        "canonical_idempotent_reuse": _assert_canonical_idempotent_reuse(runtime),
        "part2_rolling_screen": _assert_rolling_screen(),
        "gate_b": _assert_gate_b(),
        "unprojected_runtime": _assert_unprojected_runtime(),
        "dynamic96_d_canonical_stack": _assert_dynamic96_d_canonical_stack(),
        "mask_grid_counts": _assert_mask_grid(config),
        "atomic_store": _assert_atomic_store(config, runtime),
        "run04_go": _assert_run04_go(config, runtime),
        "outer_cartesian": _assert_outer_cartesian(),
        "renderer_candidate_contract": _assert_renderer_candidate_contract(
            config, runtime
        ),
    }
    code_hashes = {
        path.relative_to(ROOT).as_posix(): sha256_file(path) for path in py_files
    }
    receipt = {
        "status": "STATIC_SYNTHETIC_SELFTEST_PASS",
        "science_run": False, "d_drive_read": False, "patient_data_read": False,
        "run04_member_payload_read": False, "optimizer_constructed": False,
        "matplotlib_imported": False, "mne_imported": False,
        "protocol_sha256": sha256_file(protocol_path),
        "config_sha256": sha256_file(ROOT / "config.json"),
        "formal_teacher_epochs": 1050, "formal_wgan_epochs": 40,
        "formal_stage_ids": list(FORMAL_STAGE_IDS),
        "checks": checks, "code_sha256": code_hashes,
    }
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
