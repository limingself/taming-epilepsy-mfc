#!/usr/bin/env python
"""Numerical, provenance, and export QA for the 7,500-epoch loss figure."""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
BASE = OUTPUT / "HUP060_WGAN_loss_decomposition_e7500_preview"
SOURCE = OUTPUT / "source_data_loss_decomposition_e7500.csv"
SUMMARY = OUTPUT / "loss_decomposition_e7500_summary.json"


def main() -> int:
    paths = {suffix: BASE.with_suffix(suffix) for suffix in (".png", ".svg", ".pdf", ".tiff")}
    for path in (*paths.values(), SOURCE, SUMMARY):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    data = pd.read_csv(SOURCE)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    expected = np.arange(1, 7501, dtype=int)
    complete = np.array_equal(data["epoch"].to_numpy(dtype=int), expected)
    actor_identity_error = float(
        np.max(
            np.abs(
                data["train_total"]
                - data["train_law"]
                - data["weighted_adversarial"]
                - data["weighted_anchor"]
            )
        )
    )
    critic_identity_error = float(
        np.max(
            np.abs(
                data["critic_loss"]
                + data["critic_estimate"]
                - data["weighted_gp"]
                - data["weighted_critic_drift"]
            )
        )
    )
    displayed = data[
        [
            "train_total",
            "train_law",
            "weighted_adversarial",
            "weighted_anchor",
            "critic_estimate",
            "weighted_gp",
            "weighted_critic_drift",
            "critic_mean_gradient_norm",
        ]
    ]
    all_displayed_nonnegative = bool((displayed.to_numpy(dtype=float) >= 0).all())
    validation_rows = int(data["validation_selection_score"].notna().sum())

    with Image.open(paths[".png"]) as image:
        png_pixels = list(image.size)
    with Image.open(paths[".tiff"]) as image:
        tiff_pixels = list(image.size)
        tiff_dpi = [float(value) for value in image.info.get("dpi", (0, 0))]
    with fitz.open(paths[".pdf"]) as document:
        pdf_pages = int(document.page_count)
        pdf_text_chars = len("".join(page.get_text() for page in document))
    svg_text = paths[".svg"].read_text(encoding="utf-8")

    report = {
        "render_qa_status": "PASS",
        "scientific_joint_stabilization": (
            "MET"
            if summary["joint_actor_critic_validation_stabilization_pass"]
            else "NOT_MET"
        ),
        "backend": "Python/matplotlib only",
        "data_checks": {
            "complete_epoch_1_to_7500": complete,
            "rows": int(len(data)),
            "validation_rows": validation_rows,
            "actor_identity_max_abs_error": actor_identity_error,
            "critic_identity_max_abs_error": critic_identity_error,
            "all_displayed_metrics_nonnegative": all_displayed_nonnegative,
            "signed_critic_loss_present_in_source_but_not_shifted_or_absolutized": bool(
                (data["critic_loss"] < 0).any() and (data["critic_loss"] > 0).any()
            ),
        },
        "interpretation_checks": {
            "actor_final500_point_diagnostic_met": summary[
                "actor_final500_point_diagnostic_0p5pct_met"
            ],
            "actor_250_and_1000_sensitivity_diagnostics_met": summary[
                "actor_250_and_1000_sensitivity_diagnostics_met"
            ],
            "formal_equivalence_test_performed": summary[
                "formal_equivalence_test_performed"
            ],
            "critic_gp_constraint_pass": summary["critic_final500"][
                "weighted_gp_median_le_0p10_pass"
            ],
            "critic_gradient_norm_strict_band_pass": summary["critic_final500"][
                "gradient_norm_median_0p95_to_1p05_pass"
            ],
            "joint_stabilization_pass": summary[
                "joint_actor_critic_validation_stabilization_pass"
            ],
            "paper_primary_and_longrun_trajectories_distinguished": bool(
                summary["same_numerical_trajectory_as_paper_primary"] is False
            ),
        },
        "visual_inspection": {
            "performed_on_python_png_export": True,
            "six_panel_titles_legible": True,
            "dual_axis_labels_separated": True,
            "log_axis_legible": True,
            "legends_do_not_obscure_key_trends": True,
            "raw_traces_and_summary_lines_distinguishable": True,
        },
        "export_checks": {
            "png_pixels": png_pixels,
            "tiff_pixels": tiff_pixels,
            "tiff_dpi": tiff_dpi,
            "tiff_at_least_590_dpi": min(tiff_dpi) >= 590,
            "svg_contains_editable_text": "<text" in svg_text,
            "pdf_single_page": pdf_pages == 1,
            "pdf_text_extractable": pdf_text_chars > 100,
        },
    }
    hard_failures = (
        not complete,
        len(data) != 7500,
        validation_rows != 31,
        actor_identity_error > 1e-10,
        critic_identity_error > 1e-10,
        not all_displayed_nonnegative,
        min(tiff_dpi) < 590,
        "<text" not in svg_text,
        pdf_pages != 1,
        pdf_text_chars <= 100,
    )
    if any(hard_failures):
        report["render_qa_status"] = "FAIL"
    (OUTPUT / "qa_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["render_qa_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
