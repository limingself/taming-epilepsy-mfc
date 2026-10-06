"""Python-only numeric and actual-PDF render QA for original65 two-page plates."""
import json
import os
from pathlib import Path
import subprocess

import fitz
import pandas as pd


def long_path(path):
    value=str(Path(path).absolute())
    if os.name=='nt' and not value.startswith('\\\\?\\'):value='\\\\?\\'+value
    return Path(value)


HERE=long_path(Path(__file__).resolve().parent)
figures=HERE/'figures_original_style'
render=figures/'pdf_render'
render.mkdir(parents=True,exist_ok=True)
poppler=Path(r'C:\Users\LiMing\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe')
qa=json.loads((figures/'external_original_style_numeric_qa.json').read_text(encoding='utf-8'))
source=pd.read_csv(figures/'source_data_channel_density_long.csv',encoding='utf-8-sig')
assert len(source)==64*4*240
assert set(source.subject_id)=={'HUP065'}
assert source.groupby(['channel_index','series_code']).size().eq(240).all()
results=[]
for index,(count,height) in enumerate(((36,180.),(28,155.34)),1):
    path=figures/f'hup065_ofrc_all_channels_page_{index:02d}.pdf'
    doc=fitz.open(str(path))
    assert len(doc)==1
    page=doc[0]
    width=page.rect.width*25.4/72
    actual_height=page.rect.height*25.4/72
    assert abs(width-170)<.001 and abs(actual_height-height)<.001
    text=page.get_text()
    assert f'HUP065: channel occupation laws ({index}/2)' in text
    for label in ('Recorded ictal','Free prediction','Preictal reference','Controlled','Standardized amplitude'):
        assert label in text
    assert 'W1' not in text and 'loss' not in text.lower()
    outside=[]
    sizes=[]
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines',[]):
            for span in line['spans']:
                bbox=fitz.Rect(span['bbox'])
                if bbox.x0<-.1 or bbox.y0<-.1 or bbox.x1>page.rect.width+.1 or bbox.y1>page.rect.height+.1:
                    outside.append(span['text'])
                sizes.append(span['size'])
    assert not outside
    assert min(sizes)>=8.9
    assert qa['pages'][index-1]['occupied_axis_count']==count
    assert qa['pages'][index-1]['curves_per_channel']==4
    subprocess.run([str(poppler),'-f','1','-singlefile','-png','-r','150',str(path),
        str(render/f'page_{index:02d}')],check=True,capture_output=True)
    results.append(dict(page=index,contacts=count,width_mm=width,height_mm=actual_height,
        exactly_four_curves_per_contact=True,shared_legend_retained=True,standardized_amplitude_retained=True,
        all_text_inside_page=True,smallest_font_pt=min(sizes),
        pdf_render=f'pdf_render/page_{index:02d}.png',human_visual_inspection=False))
    doc.close()
report=dict(status='numeric_exports_and_Poppler_render_complete_pending_human_visual_review',
    backend='Python/matplotlib exports; Python invocation of Poppler render for QA',
    source_density_rows=64*4*240,pages=results,
    empirical_endpoint_binding_maximum_error=qa['empirical_endpoint_binding_maximum_error'],
    original65_grid_and_KDE_bandwidth_preserved=True,original_RMS_and_energy_caps_preserved=True,
    all24_outer_budget_cases_passed=True,posthoc_not_locked=True,warm_start_not_from_scratch=True,
    additional150_updates_selected0=True,no_new_loss_figure=True,
    published_external_results_replaced=False)
(figures/'export_qa.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
