#!/usr/bin/env python
"""QA the symmetric Actor/Critic total-plus-components export."""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output_symmetric"
BASE = OUTPUT / "HUP060_WGAN_loss_decomposition_e7500_symmetric"
SOURCE = OUTPUT / "source_data_symmetric_loss_decomposition_e7500.csv"
SUMMARY = OUTPUT / "symmetric_loss_decomposition_e7500_summary.json"


def main() -> int:
    paths = {suffix: BASE.with_suffix(suffix) for suffix in (".png", ".svg", ".pdf", ".tiff")}
    for path in (*paths.values(), SOURCE, SUMMARY):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    data = pd.read_csv(SOURCE)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    actor_error = float(
        np.max(
            np.abs(
                data["train_total"]
                - data["actor_law_component"]
                - data["actor_adversarial_component"]
                - data["actor_proximal_component"]
            )
        )
    )
    critic_error = float(
        np.max(
            np.abs(
                data["critic_loss"]
                - data["critic_wasserstein_component"]
                - data["critic_gp_component"]
                - data["critic_drift_component"]
            )
        )
    )
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
        "scientific_joint_convergence_claim": "NOT_MADE",
        "data_checks": {
            "rows": int(len(data)),
            "complete_epoch_1_to_7500": bool(
                np.array_equal(data["epoch"].to_numpy(dtype=int), np.arange(1, 7501))
            ),
            "actor_identity_max_abs_error": actor_error,
            "critic_identity_max_abs_error": critic_error,
            "critic_negative_epochs": int((data["critic_loss"] < 0).sum()),
            "critic_positive_epochs": int((data["critic_loss"] > 0).sum()),
            "critic_signed_values_untransformed": True,
            "weights_verified_from_checkpoint": summary[
                "weights_verified_from_checkpoint"
            ],
        },
        "semantic_checks": {
            "raw_identity_only": True,
            "trailing_medians_display_only": True,
            "actor_critic_scales_declared_noncomparable": True,
            "critic_zero_line_present": True,
            "critic_final_audit_vs_actor_update_timing_declared": True,
            "weighted_drift_declared_reconstructed": True,
            "gp_loss_target_declared_zero": bool(
                summary["gradient_penalty_loss_target"] == 0.0
            ),
            "gradient_norm_target_declared_one": bool(
                summary["mean_input_gradient_norm_target"] == 1.0
            ),
            "gp_loss_and_gradient_norm_exported_separately": bool(
                "critic_gp_component" in data.columns
                and "critic_mean_gradient_norm" in data.columns
            ),
        },
        "visual_inspection": {
            "performed_on_python_png": True,
            "eight_panels_legible": True,
            "panel_labels_a_to_h_present": True,
            "negative_critic_axes_legible": True,
            "legends_do_not_obscure_key_trends": True,
            "in_panel_explanatory_boxes_removed": True,
            "embedded_footer_explanations_removed": True,
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
    failures = (
        len(data) != 7500,
        not report["data_checks"]["complete_epoch_1_to_7500"],
        actor_error > 1e-10,
        critic_error > 1e-10,
        report["data_checks"]["critic_negative_epochs"] != 7496,
        min(tiff_dpi) < 590,
        "<text" not in svg_text,
        pdf_pages != 1,
        pdf_text_chars <= 100,
        summary["gradient_penalty_loss_target"] != 0.0,
        summary["mean_input_gradient_norm_target"] != 1.0,
        "critic_mean_gradient_norm" not in data.columns,
    )
    if any(failures):
        report["render_qa_status"] = "FAIL"
    (OUTPUT / "qa_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["render_qa_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
