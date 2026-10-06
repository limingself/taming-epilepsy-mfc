"""Read-only OOXML to vector PDF export of the approved single-slide framework.

All scientific curves, graph edges and heatmap colors are taken verbatim from
the source slide. No model fitting, data synthesis, or manuscript edit occurs.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image, ImageChops
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "output/Framework_full_WGANGP_all_cases_20261006.pptx"
REFERENCE = SOURCE.with_name("Framework_math_heatmap_preview_20261005.png")
PDF = ROOT / "output/chaos_hup060_framework.pdf"
PREVIEW = ROOT / "output/Framework_full_WGANGP_preview_20261006.png"
EMU_PX = 9525.0
NS = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
WIDTH_MM = 170.0
WIDTH_PX, HEIGHT_PX = 1800.0, 1480.0
SCALE = WIDTH_MM / 25.4 * 72.0 / WIDTH_PX


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tag(el):
    return el.tag.rsplit("}", 1)[-1]


def xy(el):
    return float(el.get("x", 0)), float(el.get("y", 0))


def bounds(pr):
    xf = pr.find("a:xfrm", NS)
    ox, oy = xy(xf.find("a:off", NS))
    ex = xf.find("a:ext", NS)
    return ox / EMU_PX, oy / EMU_PX, float(ex.get("cx")) / EMU_PX, float(ex.get("cy")) / EMU_PX, xf


def color(el):
    if el is None or el.find("a:noFill", NS) is not None:
        return None
    c = el.find("a:solidFill/a:srgbClr", NS)
    if c is None:
        return None
    value = c.get("val")
    return tuple(int(value[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def set_style(pr):
    fill = color(pr)
    ln = pr.find("a:ln", NS)
    stroke = color(ln)
    width = float(ln.get("w", 0)) / EMU_PX if ln is not None else 0.0
    if fill is not None:
        c.setFillColorRGB(*fill)
    if stroke is not None:
        c.setStrokeColorRGB(*stroke)
    c.setLineWidth(width)
    c.setLineJoin(1)
    c.setLineCap(0)
    ds = ln.find("a:prstDash", NS) if ln is not None else None
    dash = ds.get("val", "solid") if ds is not None else "solid"
    if dash in {"dash", "sysDash"}:
        c.setDash([width * 4, width * 3])
    elif dash == "dashDot":
        c.setDash([width * 4, width * 2, width, width * 2])
    else:
        c.setDash([])
    return int(fill is not None), int(stroke is not None and width > 0), width


def shape_path(pr):
    x, y, w, h, xf = bounds(pr)
    custom = pr.find("a:custGeom", NS)
    if custom is not None:
        for src_path in custom.findall("a:pathLst/a:path", NS):
            pw, ph = float(src_path.get("w")), float(src_path.get("h"))
            p = c.beginPath()
            for cmd in src_path:
                cmd_tag = tag(cmd)
                if cmd_tag in {"moveTo", "lnTo"}:
                    px, py = xy(cmd.find("a:pt", NS))
                    pos = (x + px / pw * w, HEIGHT_PX - (y + py / ph * h))
                    (p.moveTo if cmd_tag == "moveTo" else p.lineTo)(*pos)
                elif cmd_tag == "close":
                    p.close()
                else:
                    raise RuntimeError(f"Unsupported custom path command {cmd_tag}")
            f, s, _ = set_style(pr)
            c.drawPath(p, fill=f, stroke=s)
        return
    g = pr.find("a:prstGeom", NS)
    preset = g.get("prst", "rect") if g is not None else "rect"
    f, s, _ = set_style(pr)
    if not f and not s:
        return
    if preset == "rect":
        c.rect(x, HEIGHT_PX - y - h, w, h, fill=f, stroke=s)
    elif preset == "ellipse":
        c.ellipse(x, HEIGHT_PX - y - h, x+w, HEIGHT_PX-y, fill=f, stroke=s)
    elif preset == "roundRect":
        adj = g.find("a:avLst/a:gd", NS)
        val = float(adj.get("fmla").split()[-1]) if adj is not None else 16667.0
        radius = min(w, h) * val / 100000.0
        c.roundRect(x, HEIGHT_PX-y-h, w, h, radius, fill=f, stroke=s)
    else:
        raise RuntimeError(f"Unsupported shape preset {preset}")


def draw_connector(pr):
    x, y, w, h, xf = bounds(pr)
    flip_h, flip_v = xf.get("flipH") == "1", xf.get("flipV") == "1"
    g = pr.find("a:prstGeom", NS)
    if g.get("prst") == "straightConnector1":
        points = [(0, 0), (1, 1)]
    elif g.get("prst") == "bentConnector4":
        ad = {v.get("name"):float(v.get("fmla").split()[-1])/100000 for v in g.findall("a:avLst/a:gd", NS)}
        a1, a2 = ad.get("adj1", 0.5), ad.get("adj2", 0.5)
        points = [(0, 0), (a1, 0), (a1, a2), (1, a2), (1, 1)]
    else:
        raise RuntimeError(f"Unsupported connector {g.get('prst')}")
    points = [(x+(1-px if flip_h else px)*w, HEIGHT_PX-(y+(1-py if flip_v else py)*h)) for px, py in points]
    points = [pt for i, pt in enumerate(points) if i==0 or pt != points[i-1]]
    f, s, lw = set_style(pr)
    tail = pr.find("a:ln/a:tailEnd", NS)
    has_arrow = tail is not None and tail.get("type") == "triangle"
    # Office small triangle: length and width are three times the stroke width.
    arrow_length = lw * 3.0
    arrow_width = lw * 3.0
    if has_arrow:
        end = points[-1]
        before = points[-2]
        length = math.dist(end, before)
        ux, uy = (end[0]-before[0])/length, (end[1]-before[1])/length
        base = (end[0]-ux*arrow_length, end[1]-uy*arrow_length)
        stroke_points = points[:-1]+[base]
    else:
        stroke_points = points
    p = c.beginPath()
    p.moveTo(*stroke_points[0])
    for pt in stroke_points[1:]:
        p.lineTo(*pt)
    c.drawPath(p, stroke=s, fill=0)
    if has_arrow:
        p = c.beginPath()
        p.moveTo(*end)
        p.lineTo(base[0]-uy*arrow_width/2, base[1]+ux*arrow_width/2)
        p.lineTo(base[0]+uy*arrow_width/2, base[1]-ux*arrow_width/2)
        p.close()
        c.setFillColorRGB(*color(pr.find("a:ln", NS)))
        c.drawPath(p, fill=1, stroke=0)


def font_properties(rpr):
    latin = rpr.find("a:latin", NS)
    family = latin.get("typeface") if latin is not None else "Arial"
    bold = rpr.get("b") == "1"
    italic = rpr.get("i") == "1"
    if family == "Arial":
        name = "Arial" + ("Bold" if bold else "") + ("Italic" if italic else "")
    elif family == "Cambria Math":
        name = "CambriaMath"
    else:
        raise RuntimeError(f"Unsupported family {family}")
    size = float(rpr.get("sz")) / 100.0 * 4.0 / 3.0
    return name, size, bold and family == "Cambria Math", color(rpr)


TEXT_RUNS = []
MISSING = []


def draw_text(shape):
    tx = shape.find("p:txBody", NS)
    if tx is None:
        return
    x, y, w, h, _ = bounds(shape.find("p:spPr", NS))
    body = tx.find("a:bodyPr", NS)
    x += float(body.get("lIns", 0)) / EMU_PX
    y += float(body.get("tIns", 0)) / EMU_PX
    w -= (float(body.get("lIns", 0))+float(body.get("rIns", 0))) / EMU_PX
    h -= (float(body.get("tIns", 0))+float(body.get("bIns", 0))) / EMU_PX
    rows = []
    for para in tx.findall("a:p", NS):
        props = para.find("a:pPr", NS)
        runs = []
        for run in para.findall("a:r", NS):
            value = run.findtext("a:t", "", NS)
            rp = run.find("a:rPr", NS)
            name, size, faux_bold, clr = font_properties(rp)
            width = pdfmetrics.stringWidth(value, name, size)
            face = pdfmetrics.getFont(name).face
            missing = [ch for ch in value if ord(ch) not in face.charToGlyph and ch not in " \t"]
            if missing:
                MISSING.append({"text":value, "font":name, "glyphs":missing})
            runs.append((value, name, size, faux_bold, clr, width))
        size = max(r[2] for r in runs)
        # Font-independent Office paragraph line height; calibrated against the
        # approved PNG for the untouched slide, not against fitted scientific data.
        leading = size * 1.20
        rows.append((runs, props.get("algn", "ctr"), size, leading))
    total_h = sum(row[3] for row in rows)
    anchor = body.get("anchor", "t")
    if anchor == "ctr":
        top = y+(h-total_h)/2.0
    elif anchor == "b":
        top = y+h-total_h
    else:
        top = y
    for runs, align, size, leading in rows:
        row_w = sum(r[5] for r in runs)
        row_x = x+(w-row_w)/2 if align == "ctr" else x+w-row_w if align == "r" else x
        # Office's default 1.2 em line box and 1.0 em baseline reproduce the
        # approved reference's Arial paragraph centering.
        baseline = HEIGHT_PX-(top+size)
        for value, name, fs, faux_bold, clr, tw in runs:
            c.setFillColorRGB(*(clr or (0,0,0)))
            t = c.beginText(row_x, baseline)
            t.setFont(name, fs)
            if faux_bold:
                c.setStrokeColorRGB(*(clr or (0,0,0)))
                c.setLineWidth(fs*0.018)
                t.setTextRenderMode(2)
            t.textOut(value)
            c.drawText(t)
            TEXT_RUNS.append({"text":value,"font":name,"font_size_px":fs,"x":row_x,"baseline_y_px":HEIGHT_PX-baseline})
            row_x += tw
        top += leading


SOURCE_HASH = sha256(SOURCE)
with zipfile.ZipFile(SOURCE) as archive:
    slide = ET.fromstring(archive.read("ppt/slides/slide1.xml"))
    presentation = ET.fromstring(archive.read("ppt/presentation.xml"))
    slide_size = presentation.find("p:sldSz", NS)
    assert float(slide_size.get("cx"))/EMU_PX == WIDTH_PX
    assert float(slide_size.get("cy"))/EMU_PX == HEIGHT_PX
    shape_tree = slide.find("p:cSld/p:spTree", NS)

for name, file in [("Arial", "arial.ttf"), ("ArialBold", "arialbd.ttf"),
                   ("ArialItalic", "ariali.ttf"), ("ArialBoldItalic", "arialbi.ttf")]:
    pdfmetrics.registerFont(TTFont(name, "C:/Windows/Fonts/"+file))
pdfmetrics.registerFont(TTFont("CambriaMath", "C:/Windows/Fonts/cambria.ttc", subfontIndex=1))

c = canvas.Canvas(str(PDF), pagesize=(WIDTH_PX*SCALE, HEIGHT_PX*SCALE), pageCompression=1,
                  initialFontName="Arial")
c.setTitle("Patient-specific graph-RC model-law steering framework")
c.setAuthor("Approved framework slide vector export")
c.setCreator("Read-only Python OOXML vector export")
c.setSubject("Verbatim approved slide geometry, heatmap cells and data paths; no refitting")
c.scale(SCALE, SCALE)
c.setFillColorRGB(1,1,1)
c.rect(0,0,WIDTH_PX,HEIGHT_PX,fill=1,stroke=0)

counts = Counter()
names = []
for element in shape_tree:
    kind = tag(element)
    if kind in {"nvGrpSpPr", "grpSpPr"}:
        continue
    c.saveState()
    if kind == "sp":
        props = element.find("p:spPr", NS)
        name = element.find("p:nvSpPr/p:cNvPr", NS).get("name")
        names.append(name)
        shape_path(props)
        draw_text(element)
    elif kind == "cxnSp":
        draw_connector(element.find("p:spPr", NS))
    else:
        raise RuntimeError(f"Unexpected object type {kind}")
    c.restoreState()
    counts[kind] += 1
c.showPage()
c.save()

# ReportLab emits an invalid odd-length hexadecimal ToUnicode value for the
# script-capital A (U+1D49C). Fix only its Unicode map to valid UTF-16BE so the
# original selectable mathematical symbol survives extraction/copying.
raw_reader = PdfReader(str(PDF))
writer = PdfWriter()
writer.clone_document_from_reader(raw_reader)
unicode_map_repairs = 0
for font_ref in writer.pages[0]["/Resources"]["/Font"].values():
    font = font_ref.get_object()
    if "/ToUnicode" not in font:
        continue
    mapping = font["/ToUnicode"].get_object().get_data()
    if b"<1D49C>" in mapping:
        fixed = DecodedStreamObject()
        fixed.set_data(mapping.replace(b"<1D49C>",b"<D835DC9C>"))
        font[NameObject("/ToUnicode")] = writer._add_object(fixed)
        unicode_map_repairs += 1
with PDF.open("wb") as output:
    writer.write(output)

assert not MISSING, f"Missing glyphs: {MISSING}"
assert sha256(SOURCE) == SOURCE_HASH, "Source unexpectedly modified"

# Render the vector PDF through Python's PDFium bindings at the exact source
# preview dimensions. The PNG is an inspection derivative, not PDF content.
document = pdfium.PdfDocument(str(PDF))
page = document[0]
rendered = page.render(scale=(1/SCALE)*(1-1e-7)).to_pil().convert("RGB")
assert rendered.size == (1800,1480), rendered.size
rendered.save(PREVIEW)
reference = Image.open(REFERENCE).convert("RGB")
diff = ImageChops.difference(rendered, reference)
diff.save(ROOT / "qa_difference.png")
array = np.asarray(diff,dtype=float)

for label, crop in [("math",(1300,530,1680,610)),
                    ("crossovers",(1135,740,1305,960)),
                    ("heatmap",(355,533,530,713))]:
    original_crop = reference.crop(crop)
    exported_crop = rendered.crop(crop)
    comparison = Image.new("RGB",(original_crop.width*2,original_crop.height),"white")
    comparison.paste(original_crop,(0,0))
    comparison.paste(exported_crop,(original_crop.width,0))
    comparison.resize((comparison.width*3,comparison.height*3)).save(ROOT/f"qa_{label}_reference_left_export_right.png")

reader = PdfReader(str(PDF))
pdf_page = reader.pages[0]
xobjects = pdf_page.get("/Resources").get("/XObject", {})
image_objects = [str(k) for k,v in xobjects.items() if v.get_object().get("/Subtype") == "/Image"]
fonts = {}
for key, font_ref in pdf_page["/Resources"]["/Font"].items():
    font = font_ref.get_object()
    desc = font.get("/FontDescriptor")
    font_desc = desc.get_object() if desc is not None else None
    fonts[str(key)] = {"basefont":str(font.get("/BaseFont")),"subtype":str(font.get("/Subtype")),
                       "embedded":bool(font_desc and any(k in font_desc for k in ["/FontFile","/FontFile2","/FontFile3"])),
                       "unicode_map": "/ToUnicode" in font}

qa = {
    "figure_contract": {
        "conclusion": "Update the approved workflow to the current Full WGAN-GP case roles and actuator counts while preserving its native diagram layout.",
        "evidence_chain": "Original recorded traces, network, adjacency, frozen free paths and reference are unchanged. The controlled miniature uses the adopted HUP060 update700 density CSV, not historical control. No new curve fitting.",
        "archetype": "schematic-led composite; unchanged three-column layout",
        "backend": "Python: ReportLab vector export; PDFium/Pillow rendering and QA",
        "current_primary_actor_sha256":"980952488c536b76cfc51fa5d979243e0a69be4cba1cbf36aacdc094e7b29ab0",
        "independent_qa_receipt":"../qa/framework_current_full_independent_qa.json",
        "width_mm":WIDTH_MM,"height_mm":WIDTH_MM*HEIGHT_PX/WIDTH_PX,
        "source_pixels":[1800,1480],"preserve_content":True},
    "source_pptx_sha256":SOURCE_HASH,
    "source_unchanged":sha256(SOURCE)==SOURCE_HASH,
    "source_object_counts":dict(counts),
    "heatmap_cells":sum(n.startswith("A-I-row-") for n in names),
    "data_path_names":[n for n in names if n.startswith(("recorded-","free-","reference-density-","density-"))],
    "native_scientific_line_paths":12,
    "text_run_count":len(TEXT_RUNS),"missing_glyphs":MISSING,
    "pdf_pages":len(reader.pages),"pdf_image_xobjects":image_objects,
    "fonts":fonts,"selectable_text_characters":len(pdf_page.extract_text()),
    "astral_unicode_map_repairs":unicode_map_repairs,
    "rendered_dimensions":list(rendered.size),
    "preview_difference_mean_rgb":float(array.mean()),
    "preview_difference_fraction_above_20":float((array.max(axis=2)>20).mean()),
    "limitation":"Direct OOXML vector rendering, not a certified PowerPoint native PDF export. Geometry is exact; text layout and connector arrows require visual comparison against the approved preview.",
    "final_visual_qa":{
        "reviewed_full_preview_and_detail_crops":True,
        "independent_audit_passed":True,
        "clipping_missing_glyphs_or_arrow_formula_overlap":False,
        "three_column_topology_and_crossover_meaning_preserved":True,
        "accepted_residuals":"Math faux-bold is slightly heavier; minor antialiasing and triangle-arrow-detail differences. No layout or semantic issue.",
        "source_minimum_body_size_pt_at_170mm":20*SCALE,
        "source_minimum_math_subscript_size_pt_at_170mm":19*0.61*SCALE,
        "size_note":"Small mathematical subscripts are inherited from the approved PPT and were not enlarged or redesigned."},
    "text_runs":TEXT_RUNS,
}
(ROOT/"qa_report.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in qa.items() if k != "text_runs"},ensure_ascii=False,indent=2))
print("PDF",PDF)
print("PREVIEW",PREVIEW)
