from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    phase_specs, sha256_file, validate_zip_central_directory_contract,
    verify_external_locks, zip_central_directory_receipt,
)
from .staging import (
    PhaseStore, atomic_json, compare_inventories, recursive_inventory,
    validate_completed_phase,
)


def run_phase(
    config: Mapping[str, Any], store: PhaseStore, phase: str,
    *, run04_go_file: Path | None = None,
) -> None:
    specs = {item.name: item for item in phase_specs(config)}
    if phase not in specs:
        raise KeyError(f"unknown phase {phase!r}")
    spec = specs[phase]
    with store.publish(phase, spec.dependency) as stage:
        if stage is None:
            return
        paths = {name: store.path(name) for name in specs}
        if phase == "00_inventory_before":
            locks = verify_external_locks(config, include_raw_zip=False)
            inventory = recursive_inventory([Path(item) for item in config["authority_roots"]])
            atomic_json(stage / "external_locks.json", locks)
            zip_receipt = zip_central_directory_receipt(
                Path(config["source_data"]["raw_zip"])
            )
            validate_zip_central_directory_contract(config, zip_receipt)
            atomic_json(stage / "raw_zip_central_directory_only.json", zip_receipt)
            atomic_json(stage / "hup060_authority_inventory.json", inventory)
        elif phase == "01_prepare_development":
            from .data_model import prepare_development

            prepare_development(config, stage)
        elif phase == "02_loro_part1_part2":
            from .data_model import loro_part1_part2

            loro_part1_part2(
                config,
                paths["01_prepare_development"] / "development_arrays.npz",
                stage,
            )
        elif phase == "03_loro_analytical_top1":
            from .control import loro_analytical_top1

            loro_analytical_top1(
                config,
                paths["01_prepare_development"] / "development_arrays.npz",
                paths["02_loro_part1_part2"], stage,
            )
        elif phase == "04_final_refit":
            from .control import final_refit

            final_refit(
                config,
                paths["01_prepare_development"] / "development_arrays.npz",
                paths["02_loro_part1_part2"],
                paths["03_loro_analytical_top1"], stage,
            )
        elif phase in {
            "05_teacher_s0", "06_teacher_s1", "07_teacher_s2s",
            "08_teacher_s3", "09_teacher_s4", "10_teacher_s5",
            "11_teacher_s6",
        }:
            from .training import run_atomic_teacher_stage

            teacher_phases = [
                "05_teacher_s0", "06_teacher_s1", "07_teacher_s2s",
                "08_teacher_s3", "09_teacher_s4", "10_teacher_s5",
                "11_teacher_s6",
            ]
            stage_index = teacher_phases.index(phase)
            parent = None if stage_index == 0 else paths[teacher_phases[stage_index - 1]]
            run_atomic_teacher_stage(
                config,
                paths["01_prepare_development"] / "development_arrays.npz",
                paths["03_loro_analytical_top1"], paths["04_final_refit"], stage,
                stage_index=stage_index, parent_stage_dir=parent,
            )
        elif phase == "12_wgan40_ctx5_select":
            from .training import run_atomic_wgan40_ctx5

            teacher_phases = [
                "05_teacher_s0", "06_teacher_s1", "07_teacher_s2s",
                "08_teacher_s3", "09_teacher_s4", "10_teacher_s5",
                "11_teacher_s6",
            ]
            teacher_dirs = [store.require(name) for name in teacher_phases]
            run_atomic_wgan40_ctx5(
                config,
                paths["01_prepare_development"] / "development_arrays.npz",
                paths["03_loro_analytical_top1"], paths["04_final_refit"],
                teacher_dirs[-1], teacher_dirs, stage,
            )
        elif phase == "13_ctx6_terminal_veto":
            from .control import ctx6_terminal_veto

            ctx6_terminal_veto(
                config,
                paths["01_prepare_development"] / "development_arrays.npz",
                paths["04_final_refit"], paths["12_wgan40_ctx5_select"], stage,
            )
        elif phase == "14_freeze_outer":
            from .outer import freeze_outer_contract

            freeze_outer_contract(
                config, paths["03_loro_analytical_top1"], paths["04_final_refit"],
                paths["12_wgan40_ctx5_select"], paths["13_ctx6_terminal_veto"], stage,
            )
        elif phase == "15_run04_once":
            if run04_go_file is None:
                raise PermissionError("run04 phase requires --run04-go-file")
            from .outer import open_and_evaluate_run04_once

            open_and_evaluate_run04_once(
                config,
                paths["01_prepare_development"] / "development_arrays.npz",
                paths["04_final_refit"], paths["12_wgan40_ctx5_select"],
                paths["14_freeze_outer"], run04_go_file, stage,
            )
        elif phase == "16_common_renderer_input_freeze":
            from .renderer_input import freeze_common_renderer_input

            freeze_common_renderer_input(config, paths["15_run04_once"], stage)
        elif phase == "17_inventory_after":
            before_path = paths["00_inventory_before"] / "hup060_authority_inventory.json"
            before = json.loads(before_path.read_text(encoding="utf-8"))
            after = recursive_inventory([Path(item) for item in config["authority_roots"]])
            compare_inventories(before, after)
            atomic_json(stage / "hup060_authority_inventory.json", after)
            atomic_json(
                stage / "inventory_comparison.json",
                {
                    "schema_version": "hup060-authority-after-comparison-v1",
                    "unchanged": True, "before_inventory_sha256": sha256_file(before_path),
                    "row_count": len(after),
                },
            )
        else:
            raise AssertionError(phase)


def status(config: Mapping[str, Any], store: PhaseStore) -> list[dict[str, Any]]:
    rows = []
    for spec in phase_specs(config):
        path = store.path(spec.name)
        if not path.exists():
            state = "pending"
        else:
            try:
                validate_completed_phase(path)
                state = "complete"
            except Exception:
                state = "invalid"
        rows.append({"phase": spec.name, "dependency": spec.dependency, "status": state})
    return rows


__all__ = ["run_phase", "status"]
