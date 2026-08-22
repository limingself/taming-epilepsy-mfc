#!/usr/bin/env python
"""Static/synthetic proof for the isolated HUP065 scientific runner.

No D-drive file, patient byte, science backend, Torch, MNE, Matplotlib, plot,
or formal science_run is touched by this test.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any


sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import runner


class SelfTestFailure(AssertionError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SelfTestFailure(message)


def expect_raises(exception_type: type[BaseException], function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except exception_type:
        return
    raise SelfTestFailure(f"expected {exception_type.__name__}: {function.__name__}")


def synthetic_cube(value: float, actuators: int) -> list[list[list[float]]]:
    return [[[float(value) for _ in range(actuators)] for _ in range(4)] for _ in range(2)]


def synthetic_outer_rows() -> list[dict[str, Any]]:
    bool_fields = (
        "gate_time_w1_relative_reduction_pass",
        "gate_occupation_w1_relative_reduction_pass",
        "gate_time_w1_absolute_pass",
        "gate_occupation_w1_absolute_pass",
        "gate_mean_error_pass",
        "gate_sd_ratio_pass",
        "gate_b_pass",
        "gate_c_trajectory_pass",
    )
    rows = []
    for context in range(8):
        for bank in range(3):
            for channel in range(64):
                row: dict[str, Any] = {
                    "window_id": f"outer-context-{context:02d}",
                    "crn_bank": bank,
                    "channel_index": channel,
                }
                row.update({field: True for field in bool_fields})
                rows.append(row)
    return rows


def run_tests() -> dict[str, object]:
    config = runner.load_config()
    original_path_open = Path.open
    d_drive_open_attempts: list[str] = []

    def guarded_path_open(path: Path, *args, **kwargs):
        if Path(path).drive.casefold() == "d:":
            d_drive_open_attempts.append(str(path))
            raise SelfTestFailure(f"static test attempted D-drive open: {path}")
        return original_path_open(path, *args, **kwargs)

    Path.open = guarded_path_open
    initial_pycache = sorted(str(path) for path in runner.ROOT.rglob("__pycache__"))
    try:
        import numpy as np
        from h065 import control, data_model, outer, training

        summary = runner.validate_static_config(config)
        check(summary["part2_candidates"] == 144, "Part-II grid")
        check(summary["analytical_candidates"] == 120, "analytical grid")
        check(summary["teacher_epochs_total"] == 1050, "teacher1050")
        check(summary["wgan_epochs"] == 40, "WGAN40")
        shared = runner.verify_shared_protocol(config)
        check(shared["protocol"]["sha256"] == config["shared_protocol"]["sha256"], "protocol SHA")
        check(shared["common_module"]["sha256"] == config["shared_protocol"]["common_module_sha256"], "common helper SHA")
        adaptation = runner.verify_implementation_audit(config)
        check(adaptation["numerical_semantics_changed"] is False, "D numerical semantics unchanged")
        check(len(adaptation["source_receipts"]) == 4, "four pinned scientific implementation sources")

        grid = runner.build_sparse_grid(config)
        check(len(grid) == len({item.candidate_id for item in grid}) == 120, "120 unique arms")
        for item in grid:
            check(item.centrality_quantile == round(1.0 - item.actuator_fraction, 12), "decimal q=1-f")
        check(next(item for item in grid if item.actuator_fraction == 0.80).centrality_quantile == 0.20, "f=.8 q=.2")
        scores = [float(index) for index in range(64)]
        counts = {0.20: 13, 0.30: 19, 0.35: 23, 0.40: 26, 0.50: 32, 0.60: 38, 0.70: 45, 0.80: 51}
        selection_network = {"centrality_score": np.asarray(scores), "adjacency": np.eye(64)}
        for fraction, expected in counts.items():
            check(sum(runner.strict_quantile_mask(scores, fraction)) == expected, f"runner strict mask {fraction}")
            formal = control.network_for_fraction(selection_network, fraction)
            check(int(np.asarray(formal["target_mask"]).sum()) == expected, f"formal strict mask {fraction}")
        check(sum(runner.strict_quantile_mask(scores, 0.80)) == 51, "52/64 forbidden")
        check(sum(runner.strict_quantile_mask([0.0] * 32 + [1.0] * 32, 0.50)) == 32, "strict tied threshold")

        ranked = sorted(
            [
                runner.CandidateSummary("bad", True, False, 64, 0.01, 1),
                runner.CandidateSummary("four", True, True, 4, 0.20, 13),
                runner.CandidateSummary("five_more", True, True, 5, 0.20, 19),
                runner.CandidateSummary("five_best", True, True, 5, 0.10, 51),
            ],
            key=runner.candidate_rank_key,
        )
        check(ranked[0].candidate_id == "five_best" and ranked[-1].candidate_id == "bad", "analytical rank")

        # The four quantitative plant-fidelity inequalities are diagnostics,
        # not preregistered analytical feasibility inputs.  A false diagnostic
        # must not mask an otherwise integrity- and Gate-C-eligible arm.
        plant_fidelity_diagnostic_pass = False
        protocol_eligible = control.analytical_candidate_eligible(
            protocol_gate_a_integrity_pass=True,
            gate_c_energy_finite_pass=True,
        )
        check(
            not plant_fidelity_diagnostic_pass and protocol_eligible,
            "plant diagnostic false remains eligible under integrity plus Gate-C",
        )
        diagnostic_rank = sorted(
            [
                runner.CandidateSummary(
                    "plant-diagnostic-false", True, True, 3, 0.10, 13
                ),
                runner.CandidateSummary(
                    "energy-failed", True, False, 64, 0.01, 1
                ),
            ],
            key=runner.candidate_rank_key,
        )
        check(
            diagnostic_rank[0].candidate_id == "plant-diagnostic-false",
            "diagnostic-false eligible arm participates in declared rank",
        )

        screen = config["analytical_screen"]
        check(
            screen["eligibility_prefilter"]
            == "protocol_gate_a_static_integrity_and_gate_c_energy_finite_only",
            "analytical eligibility contract",
        )
        check(
            screen["plant_fidelity_four_inequality_role"]
            == "diagnostic_report_only_never_eligibility_rank_no_go_rescue_or_replacement",
            "plant fidelity is diagnostic only",
        )
        check(
            screen["trajectory_gate_c_operands_repeated_in_each_channel_context_row"]
            is True,
            "trajectory Gate-C operands are row-reconstructable",
        )

        low = runner.control_feasibility(synthetic_cube(0.10, 51))
        cap = (13.0 / 36.0) * 64.0 * (0.405**2)
        check(low["pass"] and math.isclose(low["total_energy"], 0.51), "raw energy calculation")
        check(not runner.control_feasibility(synthetic_cube(0.50, 1))["pass"], "RMS gate")
        check(not runner.control_feasibility(synthetic_cube(1.80, 1))["pass"], "saturation gate")
        check(math.isclose(config["energy"]["total_energy_cap"], cap), "fixed Emax")
        check(config["energy"]["hard_projection_or_rescaling_allowed"] is False, "no projection")

        sd_fail = control.frozen_gate_b_components(
            np.asarray([1.0]), np.asarray([0.20]), np.asarray([0.50]),
            np.asarray([0.10]), np.asarray([0.05]), np.asarray([3.0]),
        )
        check(sd_fail["relative_time_w1_reduction"][0] > 0, "time improves")
        check(sd_fail["relative_occupation_w1_reduction"][0] > 0, "occupation improves")
        check(not bool(sd_fail["full_gate_b_pass"][0]), "SD failure vetoes full Gate-B")
        gate_pass = control.frozen_gate_b_components(
            np.asarray([1.0]), np.asarray([0.20]), np.asarray([0.50]),
            np.asarray([0.10]), np.asarray([0.05]), np.asarray([1.5]),
        )
        check(bool(gate_pass["full_gate_b_pass"][0]), "all six Gate-B components")

        ctx6 = {
            "gate_c_pass": True,
            "mean_time_w1_controlled": 0.2,
            "mean_time_w1_free": 0.4,
            "mean_occupation_w1_controlled": 0.1,
            "mean_occupation_w1_free": 0.3,
            "gate_b_pass_count": 1,
        }
        check(training.ctx6_veto_decision(ctx6), "ctx6 full pass")
        changed = dict(ctx6, gate_b_pass_count=0)
        check(not training.ctx6_veto_decision(changed), "ctx6 full-six count")
        changed = dict(ctx6, mean_time_w1_controlled=0.4)
        check(not training.ctx6_veto_decision(changed), "ctx6 strict time improvement")

        class IdentityScaler:
            @staticmethod
            def transform(values):
                return np.asarray(values)

        model = SimpleNamespace(transform=SimpleNamespace(scaler=IdentityScaler()))

        def references(offset: float) -> Any:
            ids = np.arange(40, dtype=np.float64) + offset
            return np.broadcast_to(ids[:, None, None], (40, 256, 64)).reshape(10240, 64).copy()

        arrays = {
            "preictal_reference_run_01": references(0.0),
            "preictal_reference_run_02": references(100.0),
        }
        fit, validation, reference_receipt = control.standardized_reference_pools(
            model, arrays, ["run-01", "run-02"]
        )
        check(fit.shape == validation.shape == (30, 256, 64), "reference pool shapes")
        check(np.array_equal(fit[:, 0, 0], np.r_[0:15, 100:115]), "per-run fit concat")
        check(np.array_equal(validation[:, 0, 0], np.r_[15:30, 115:130]), "per-run ctx5 concat")
        check(not np.shares_memory(fit, validation), "fit/ctx5 distinct tensors")
        check(reference_receipt["fit_reference_sha256"] != reference_receipt["validation_reference_sha256"], "reference hashes distinct")

        class FakePart2:
            calls: list[dict[str, int]] = []

            @classmethod
            def forecast_positions(cls, length, **kwargs):
                cls.calls.append({"length": int(length), **{key: int(value) for key, value in kwargs.items()}})
                return np.rint(np.linspace(512, int(length) - 128, 6)).astype(np.int64)

        sequence = np.zeros((8192, 64), dtype=np.float64)
        positions = data_model.part2_validation_positions(sequence, FakePart2, 6)
        p2_context, p2_future, _ = data_model.part2_context_block(sequence, 0, positions)
        p3_context, p3_future, _ = data_model.part3_context_block(sequence, 0)
        check(p2_context.shape == (512, 64) and p2_future.shape == (128, 64), "Part-II 512/128")
        check(p3_context.shape == (256, 64) and p3_future.shape == (256, 64), "Part-III 256/256")
        check(FakePart2.calls[-1] == {"length": 8192, "context_samples": 512, "maximum_horizon": 128, "guard_samples": 0, "count": 6}, "D Part-II forecast_positions contract")

        rolling_rows = []
        for block in (4, 8, 12, 16, 24, 32):
            for fold in ("leave-run-01-out", "leave-run-02-out"):
                rolling_rows.append(
                    {
                        "block_samples": block,
                        "fold_id": fold,
                        "finite": True,
                        "median_correlation": 0.71 if block <= 24 else 0.69,
                        "median_nrmse": 0.79,
                    }
                )
        check(data_model.select_rolling_block_fail_closed(rolling_rows, ["leave-run-01-out", "leave-run-02-out"]) == 24, "largest jointly eligible rolling block")
        for row in rolling_rows:
            row["median_correlation"] = 0.0
        expect_raises(RuntimeError, data_model.select_rolling_block_fail_closed, rolling_rows, ["leave-run-01-out", "leave-run-02-out"])

        class LiveModel:
            pass

        data_model.assert_live_part2_model_identity(LiveModel(), SimpleNamespace(ResidualGraphRCSDE=LiveModel))
        expect_raises(RuntimeError, data_model.assert_live_part2_model_identity, object(), SimpleNamespace(ResidualGraphRCSDE=LiveModel))

        data_model.assert_signal_run_allowed("DEV_MATERIALIZE", "run-01")
        data_model.assert_signal_run_allowed("DEV_MATERIALIZE", "run-02")
        expect_raises(PermissionError, data_model.assert_signal_run_allowed, "DEV_MATERIALIZE", "run-03")
        data_model.assert_signal_run_allowed("OUTER", "run-03")

        complete = synthetic_outer_rows()
        check(outer.validate_outer_metric_cartesian(complete)["cells"] == 1536, "outer 8x3x64 Cartesian")
        expect_raises(RuntimeError, outer.validate_outer_metric_cartesian, complete[:-1])
        duplicate = [dict(row) for row in complete]
        duplicate[-1] = dict(duplicate[0])
        expect_raises(RuntimeError, outer.validate_outer_metric_cartesian, duplicate)
        non_bool = [dict(row) for row in complete]
        non_bool[0]["gate_b_pass"] = 1
        expect_raises(RuntimeError, outer.validate_outer_metric_cartesian, non_bool)

        for mutation in (
            ("diffusion", lambda value: value["part2"].__setitem__("effective_diffusion_multiplier", 1.0)),
            ("teacher180", lambda value: value["part3"]["training"].__setitem__("teacher_epochs_total", 180)),
            ("part1_window", lambda value: value["part1"].__setitem__("plv_window_seconds", 2.0)),
            ("part2_context", lambda value: value["part2"].__setitem__("context_samples", 256)),
        ):
            mutated = copy.deepcopy(config)
            mutation[1](mutated)
            expect_raises(runner.ContractError, runner.validate_static_config, mutated)

        expect_raises(runner.ScienceAuthorizationError, runner.execute_science_phase, config, Path(config["default_run_root"]), "DEV_MATERIALIZE", authorization="NOT_AUTHORIZED")
        expect_raises(runner.ScienceAuthorizationError, runner.execute_outer, config, Path(config["default_run_root"]), authorization="NOT_AUTHORIZED")

        paths = sorted(runner.ROOT.glob("*.py")) + sorted((runner.ROOT / "h065").glob("*.py"))
        sources = {}
        for path in paths:
            key = path.name if path.parent == runner.ROOT else f"h065/{path.name}"
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=key)
            sources[key] = source
        backend = sources["science_backend.py"]
        tree = ast.parse(backend)
        check(not any(isinstance(node, ast.Pass) for node in ast.walk(tree)), "no backend placeholders")
        for call in (
            "data_model.prepare_development(", "data_model.loro_part1(",
            "data_model.loro_part1_part2(", "control.loro_analytical_top1(",
            "control.final_refit(", "training.train_formal_stage(",
            "training.wgan40_and_ctx6_veto(", "outer.open_and_evaluate_run03_once(",
        ):
            check(call in backend, f"real backend call absent: {call}")
        check(backend.index("_IMPORT_RUNTIME_RECEIPT = _install_runtime_firewall") < backend.index("from h065 import control"), "runtime firewall import order")
        check("import mne" not in backend and "import matplotlib" not in backend, "deferred MNE/Matplotlib")

        formal = "\n".join((sources["h065/control.py"], sources["h065/training.py"], backend))
        for banned in ("causal_energy_projection_scale", "budgeted_particle_rollout"):
            check(banned not in formal, f"banned projection helper: {banned}")
        check("unprojected_rollout(" in formal, "D empirical FP rollout formal path")
        training_text = sources["h065/training.py"]
        check("legacy monolithic S0-S6/WGAN route is permanently disabled" in training_text, "legacy training route disabled")
        for exact in (
            "base_seed + 101 * (bank + 1)", "base_seed + 50000 + step",
            "base_seed + 1009 * (epoch_index + 1)", "+ 100000 * (epoch_index + 1)",
            "clip_grad_norm_(critic.parameters(), 5.0)",
            "clip_grad_norm_(actor.parameters(), 1.0)", "betas=(0.0, 0.9)",
            "actor.load_state_dict(teacher_state, strict=True)",
            "fallback_update_promoted\": False", "epoch_zero_eligible\": False",
            "HISTORICAL_ACTOR_STATE_KEYS = 23", "MODERN_ACTOR_STATE_KEYS = 44",
            "MIGRATION_INITIALIZATION_SEED = 20260921",
            "TEACHER_TRAINING_BASE_SEED = 20260921",
            "TEACHER_VALIDATION_SEED = 20260922",
            "TEACHER_TRAINING_BASE_SEED + 1009 * (absolute + 1)",
            "fresh_migration_parameter_sha256",
        ):
            check(exact in training_text, f"WGAN/Actor contract absent: {exact}")
        training._require_actor_state_keys({str(index): index for index in range(44)}, 44, "synthetic")
        expect_raises(RuntimeError, training._require_actor_state_keys, {str(index): index for index in range(43)}, 44, "synthetic")

        control_text = sources["h065/control.py"]
        check("legacy teacher180 route is permanently disabled" in control_text, "teacher180 route disabled")
        check("standalone legacy ctx6 route is disabled" in control_text and "def ctx6_terminal_veto(" not in control_text, "legacy ctx6 route disabled")
        check(
            '"controller_feasible": analytical_candidate_eligible(' in control_text
            and '"plant_fidelity_used_for_eligibility_or_rank": False'
            in control_text
            and '"plant_diagnostic_used_as_pre_rank_eligibility": False'
            in control_text,
            "plant diagnostic excluded from formal analytical eligibility",
        )
        check(
            '"controller_feasible": bool(gate_c and plant_pass)' not in control_text
            and "LORO plant/control/energy gate" not in control_text,
            "undeclared plant prefilter removed",
        )
        check(
            '"aggregate_control_improvement_diagnostic_pass"' in control_text
            and '"aggregate_control_improvement_or_gate_b_used_as_pre_rank_eligibility": False'
            in control_text
            and '"gate_a_or_gate_b_used_as_pre_rank_eligibility"' not in control_text
            and '"gate_a_pass"' not in control_text,
            "aggregate improvement is not mislabeled protocol Gate-A",
        )
        for field in (
            "trajectory_total_energy",
            "trajectory_maximum_per_actuator_rms",
            "trajectory_control_peak",
            "trajectory_saturation_fraction",
            "trajectory_finite",
        ):
            check(f'"{field}"' in control_text, f"trajectory safety operand {field}")
        for exact in (
            "control_graph_diffusion_time=float(config.graph_diffusion_time)",
            "source_text.replace(local_override, \"\")",
            "source_text.replace(feedback_override, \"\")",
            "FrozenIctalGraphRCBatchStepper(",
            "module.empirical_fp_rollout = empirical_fp_rollout",
            "unselected_channels_may_receive_graph_mediated_effective_input\": True",
        ):
            check(exact in control_text, f"D controller restoration absent: {exact}")
        data_text = sources["h065/data_model.py"]
        check("build_part1_network(" in data_text and "build_part2_plant_adjacency(" in data_text, "two distinct graph builders")
        check("part2_pipeline.compute_adjacency(" in data_text, "D broadband plant graph")
        check("[\"worst_fold_objective\", \"mean_objective\", \"candidate_index\"]" in data_text, "Part-II worst-fold-first")
        check("select_rolling_block_fail_closed(" in data_text, "rolling fail-closed screen")
        check("assert_live_part2_model_identity(model, part2_module)" in data_text, "joblib class identity")

        executed_locks = config["public_portability"]["executed_canonical_lock_sha256"]
        check(executed_locks["causal_particle_rollout"] == "47b3328091a85f851fc8e42275c24847564f2795c7292f53f51c37746832305c", "executed D rollout provenance SHA")
        check(executed_locks["causal_riccati"] == "748567f5fb616a641c3916a69dbb9385cf2a3cd0e64d890b49301bb3244c3ae2", "executed D Riccati provenance SHA")
        check(executed_locks["canonical_data_loader"] == "f61cf1a68b31ffd775084bd7f0274334c3ef395b08ec2fda417e39523fff7601", "executed D data provenance SHA")
        check(executed_locks["square_wave_mfc"] == "11969e75d4357396d86cae41ffe11c529d0167310213c29346871ba25e70b64a", "executed D square-wave provenance SHA")
        check("bc073" not in json.dumps(config) and "47557" not in json.dumps(config), "adapted plant hashes rejected")

        outer_text = sources["h065/outer.py"]
        check("validate_outer_metric_cartesian(rows" in outer_text, "outer Cartesian audit called")
        check("display_local_index=7" in outer_text and "NO_GO_PENDING_INDEPENDENT_AUDIT" in outer_text, "display07/bank0 pending renderer")
        check(outer_text.count("observed[7]") == 2 and "observed[1]" not in outer_text, "display OFRC observed path is context07")
        check("ictal_loader = data_module.EEGSegmentLoader(" in outer_text and "reference_loader = data_module.EEGSegmentLoader(" in outer_text, "outer independent preprocessing loaders")
        check("matplotlib" not in outer_text and "render_channel_plate" not in outer_text, "outer does not render")
        all_text = "\n".join(
            source for name, source in sources.items() if name != "static_self_test.py"
        ) + json.dumps(config)
        check(not re.search(r"double_window|open_both_windows", all_text, flags=re.I), "obsolete outer wording absent")
        runner_text = sources["runner.py"]
        inventory = runner_text.split("def zip_central_directory_inventory", 1)[1].split("\ndef ", 1)[0]
        check("bundle.infolist()" in inventory, "P0 central directory only")
        check("archive.open(" not in inventory and "archive.read(" not in inventory, "P0 no member payload")
        check("sha256_file(archive" not in runner_text, "P0 no whole ZIP hash")
        # Keep the synthetic test read-only.  Atomic-publication semantics are
        # proved against the parsed implementation so this test cannot create
        # even a temporary output beside the formal runner.
        for exact in (
            "def staged_phase_directory(",
            "def _short_uuid_token(",
            "base64.urlsafe_b64encode(secrets.token_bytes(16))",
            "f\".s{phase_index:02d}-{_short_uuid_token()}\"",
            "if destination.exists()",
            "os.replace(staging, destination)",
        ):
            check(exact in runner_text, f"atomic phase publication absent: {exact}")
        for exact in (
            "def init_run(",
            "if run_root.exists()",
            "f\".i-{_short_uuid_token()}\"",
            "os.replace(staging, run_root)",
        ):
            check(exact in runner_text, f"atomic run initialization absent: {exact}")
        tokens = {runner._short_uuid_token() for _ in range(128)}
        check(len(tokens) == 128, "random128 tokens remain unique in synthetic bank")
        check(
            all(
                len(token) == 22
                and re.fullmatch(r"[A-Za-z0-9_-]{22}", token)
                for token in tokens
            ),
            "full-entropy random128 uses fixed URL-safe base64 width",
        )
        sample = next(iter(tokens))
        short_staging = (
            Path(config["default_run_root"])
            / config["runtime_safety"]["phase_output_root_directory"]
            / f".s00-{sample}"
        )
        short_atomic = short_staging / f".j-{sample}"
        old_failure_template = (
            Path("synthetic-long-root-" + "x" * 180)
            / "phases"
            / (".P0_PREFLIGHT.staging-" + "0" * 32)
            / (
                ".raw_archive_stat_and_central_directory_only.json.tmp-"
                + "0" * 32
            )
        )
        check(
            len(str(old_failure_template)) >= 260,
            "synthetic reproduces old Windows MAX_PATH failure",
        )
        check(len(str(short_atomic)) < 200, "short atomic path keeps strict headroom")
        path_budget = runner.validate_formal_path_budget(
            config, Path(config["default_run_root"])
        )
        check(
            path_budget["status"] == "FORMAL_PATH_BUDGET_PASS"
            and int(path_budget["maximum_characters_exclusive"]) == 200,
            "formal path enumerator passes strict <200 budget",
        )
        check(
            set(path_budget["worst_by_kind"]) == {"target", "staging", "atomic_temp"}
            and all(
                int(record["characters"]) < 200
                for record in path_budget["records"]
            ),
            "every enumerated target/staging/atomic path is <200",
        )
        check(
            set(runner.FORMAL_PHASE_ARTIFACTS) == set(runner.PHASE_ORDER)
            and runner.PHASE_OUTPUT_ROOT_DIRECTORY == "p",
            "all active phases use the short physical phase root",
        )
        check(
            "f\".j-{_short_uuid_token()}\"" in runner_text
            and 'temporary.open("xb")' in runner_text,
            "runner atomic JSON uses short unique exclusive-create temp",
        )
        staging_module_text = sources["h065/staging.py"]
        check(
            'temporary.open("x", encoding="utf-8")' in staging_module_text
            and "os.replace(temporary, path)" in staging_module_text,
            "backend atomic JSON uses short unique atomic replace",
        )
        check(
            "base64.urlsafe_b64encode(secrets.token_bytes(16))"
            in staging_module_text,
            "backend atomic JSON uses cryptographic random128 token",
        )
        check(
            'PHASE_OUTPUT_ROOT_DIRECTORY = "p"' in backend
            and '/ PHASE_OUTPUT_ROOT_DIRECTORY / phase_id' in backend
            and 'run_root / "phases"' not in backend,
            "backend downstream phase references mechanically use short root",
        )
        failure_binding = config["public_failure_provenance"]
        failure_path = Path(failure_binding["path"])
        check(
            runner.sha256_file(failure_path) == failure_binding["sha256"],
            "public failure-provenance receipt is hash-bound",
        )
        failure_receipt = json.loads(failure_path.read_text(encoding="utf-8"))
        check(
            failure_receipt["status"]
            == "HASH_ONLY_AUDIT_PROVENANCE_NOT_A_SCIENCE_INPUT"
            and failure_receipt["private_failure_stages_bundled"] is False
            and failure_receipt["private_failure_stages_required_for_public_self_test"]
            is False,
            "private failure stages remain absent and unnecessary",
        )
        preflight_receipt = failure_receipt["preflight_path_length_failure"]
        archived_expected = {
            "config.snapshot.json": "ffd44d0836f0f29ca0ee1568c52108bd9cd30a71a9d8cf3b2ed7cde68e2d8724",
            "phases/.P0_PREFLIGHT.staging-237456dc804a4c3f859ece778385da29/authority_before.json": "05906b46c87077838aa4e92549571c684eec4e0f010a4b5f93191a6c1af6aeb7",
            "phases/.P0_PREFLIGHT.staging-237456dc804a4c3f859ece778385da29/canonical_lock_receipts.json": "b33aa10293075bf373bf1b1b5a76fd140a6313bc6ca80274ef845b9e86287681",
            "phases/.P0_PREFLIGHT.staging-237456dc804a4c3f859ece778385da29/code_reference_lock_receipts.json": "f53046ee6b4c79abfa3aa620874e232df05463bb8f019dae4fa893ee7ff97f2e",
            "phases/.P0_PREFLIGHT.staging-237456dc804a4c3f859ece778385da29/shared_protocol_receipt.json": "8be96090084bb88d20968142582746c125b496cb8d8d915b60a93d379d4c02be",
            "phases/.P0_PREFLIGHT.staging-237456dc804a4c3f859ece778385da29/source_adaptation_audit_receipt.json": "8ea50fa1bfa8be0f4192af1fa5bec525cf0845bd1e2fc5e052ecdaf642d93e3b",
            "phases/.P0_PREFLIGHT.staging-237456dc804a4c3f859ece778385da29/split_receipt.json": "b04723e43caf9cb44f4a93bb058a5a738d31bc66ea7e6de50a141f481fd01f58",
            "RUN.json": "b76be2cd7978ec4affde21c31f70fe815ed4232142d553d682816e6d374b05fc",
        }
        archived_receipt = {
            item["name"]: item["sha256"] for item in preflight_receipt["artifacts"]
        }
        check(
            preflight_receipt["artifact_count"] == 8
            and preflight_receipt["audit_sha256"]
            == "eade8a399dbf6132cfd0f7b8fe888da8c9480afbd348a2366a070e2b706547a8"
            and archived_receipt == archived_expected,
            "failed preflight archive remains hash-only provenance",
        )
        analytical_receipt = failure_receipt["analytical_prefilter_failure"]
        check(
            analytical_receipt["audit_sha256"]
            == "39a86042e2fc78eab409a7e3280274e880c949384e8dbde8c3e873f4f96ce341",
            "analytical plant-prefilter failure audit remains hash-bound",
        )
        check(
            failure_receipt["selection_use"]
            == "DO_NOT_REUSE_FOR_SELECTION",
            "failed analytical outputs are forbidden selection inputs",
        )
        check(
            analytical_receipt["artifact_count"] == 42
            and analytical_receipt["total_bytes"] == 106976589
            and analytical_receipt["archive_inventory_sha256"]
            == "a512cd649bb7148a1cec7a7b6fffd2eb3c5a3509dfd403ead707c1516eca8941",
            "failed analytical archive inventory remains hash-only provenance",
        )
        analytical_key_hashes = {
            "RUN.json": "e93054a4468a1003a56c1d2732515fc57914ad8d4a179776a41e58bff09b550b",
            "p/.s04-LdfVP9KiITqjEO5Z8JSmHA/analytical_channel_context_metrics.csv": "1eff72ee2c004e37d9dd20b29918f90548e527db0a5da9fd1e28e621ec28a105",
            "p/.s04-LdfVP9KiITqjEO5Z8JSmHA/analytical_fold_summary.csv": "f2de16ca11daa8fa639c03162d2ff4064b4f77f9b55c16bf58363412a16608cf",
            "p/.s04-LdfVP9KiITqjEO5Z8JSmHA/analytical_ranking.csv": "473e3a458bb47dcd9ec699bbfa90b8d51f79746fce25be5cf3a4218aa548f234",
        }
        observed_key_hashes = {
            item["name"]: item["sha256"] for item in analytical_receipt["key_artifacts"]
        }
        check(
            observed_key_hashes == analytical_key_hashes,
            "failed analytical key evidence remains hash-only provenance",
        )
        early_run03 = Path(config["raw_data"]["zip_root"]) / "sub-HUP065_run-03.edf"
        expect_raises(runner.ContractError, runner.assert_science_input_path, config, early_run03, phase="DEV_MATERIALIZE")
        runner.assert_science_input_path(config, early_run03, phase="OUTER")

        final_pycache = sorted(str(path) for path in runner.ROOT.rglob("__pycache__"))
        check(final_pycache == initial_pycache, "no bytecode cache")
        check(not d_drive_open_attempts, "no D-drive opens")
        check("mne" not in sys.modules and "matplotlib" not in sys.modules, "no plotting/EDF imports")
        check("torch" not in sys.modules, "no Torch import")
        check("hup065_fresh_science_backend" not in sys.modules, "science backend not imported")
        check(not Path(config["default_run_root"]).exists(), "science_run not created")
        return {
            "status": "STATIC_SELF_TEST_PASS",
            "subject": "HUP065",
            "channels": 64,
            "shared_protocol_sha256": config["shared_protocol"]["sha256"],
            "d_drive_files_opened": 0,
            "patient_signal_opened": False,
            "science_executed": False,
            "run03_opened": False,
            "plot_created": False,
            "part2_candidates": 144,
            "part2_direct_validation_windows": 6,
            "analytical_candidates": 120,
            "teacher_epochs_total": 1050,
            "wgan_epochs": 40,
            "outer_context_bank_channel_cells": 1536,
            "reference_fit_paths": 30,
            "reference_ctx5_paths": 30,
            "energy_cap": cap,
            "phase_count": len(runner.PHASE_ORDER),
            "formal_path_records": path_budget["record_count"],
            "formal_path_overall_worst_characters": path_budget["overall_worst"]["characters"],
        }
    finally:
        Path.open = original_path_open


def main() -> int:
    print(json.dumps(run_tests(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
