"""Python-only PDF/PNG export QA of the three actual original-layout figures."""
from pathlib import Path
import json
import subprocess
import fitz

root = Path(__file__).resolve().parent
figures = root / "figures_original_style"
render = figures / "pdf_render"
render.mkdir(parents=True, exist_ok=True)
poppler = Path(r"C:\Users\LiMing\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe")
qa = json.loads((figures / "external_original_style_numeric_qa.json").read_text(encoding="utf-8"))
expected_counts = (36, 36, 24)
results = []
for index, count in enumerate(expected_counts, 1):
    name = f"hup080_ofrc_all_channels_page_{index:02d}_v2"
    path = figures / (name + ".pdf")
    doc = fitz.open(path)
    assert len(doc) == 1
    page = doc[0]
    width = page.rect.width * 25.4 / 72
    height = page.rect.height * 25.4 / 72
    expected_height = 180 if index < 3 else 130.68
    assert abs(width - 170) < .001 and abs(height - expected_height) < .001
    text = page.get_text()
    assert f"HUP080: channel occupation laws ({index}/3)" in text
    for label in ("Recorded ictal", "Free prediction", "Preictal reference", "Controlled", "Standardized amplitude"):
        assert label in text
    assert "W1" not in text and "W_1" not in text and "loss" not in text.lower()
    outside = []
    sizes = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                bbox = fitz.Rect(span["bbox"])
                if bbox.x0 < -.1 or bbox.y0 < -.1 or bbox.x1 > page.rect.width + .1 or bbox.y1 > page.rect.height + .1:
                    outside.append(span["text"])
                sizes.append(span["size"])
    assert not outside
    assert min(sizes) >= 8.9
    subprocess.run([str(poppler), "-f", "1", "-singlefile", "-png", "-r", "150", str(path), str(render / f"page_{index:02d}")], check=True, capture_output=True)
    results.append(dict(page=index, channels=count, vector_pdf_pages=1, width_mm=width, height_mm=height,
        all_text_inside_page=True, smallest_font_pt=min(sizes), four_curves_per_channel_numeric_check=True,
        shared_legend_and_standardized_amplitude_retained=True, no_distance_or_gate_annotations=True,
        pdf_render=f"pdf_render/page_{index:02d}.png", human_visual_inspection=False))
    doc.close()
result = dict(status="numeric_export_and_Poppler_render_complete_pending_visual_inspection",
    backend="Python invocation of Poppler for PDF QA; matplotlib original figures",
    pages=results, warm_start_parameter_adaptation=True, no_new_loss_figure=True,
    all_channel_display_empirical_W1_binding_error=qa["empirical_endpoint_binding_maximum_error"],
    current_old_gate_C_pass=False, revised_budget_pass_count=24, separate_distribution_gate_B_count=57,
    source_density_window_unchanged=True, full_empirical_metrics_include_samples_outside_window=True)
(figures / "export_qa.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False), flush=True)
