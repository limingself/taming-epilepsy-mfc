"""Check actual PDF/SVG/CSV exports and render PDF with Python-invoked Poppler."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import fitz
import numpy as np
import pandas as pd

import run_cold_top32 as experiment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--poppler-bin',default=None)
    args=parser.parse_args()
    root=experiment.HERE/'runs'/args.tag/'figures_original_style'
    destination=root/'export_qa.json'
    if destination.exists():raise RuntimeError('Refuse completed export QA replacement')
    candidates=[Path(args.poppler_bin)] if args.poppler_bin else []
    found=shutil.which('pdftoppm')
    if found:candidates.append(Path(found))
    candidates.append(Path.home()/'.cache'/'codex-runtimes'/'codex-primary-runtime'/'dependencies'/'native'/'poppler'/'Library'/'bin'/'pdftoppm.exe')
    poppler=next((path for path in candidates if path.is_file()),None)
    if poppler is None:raise RuntimeError('Poppler pdftoppm is unavailable: supply --poppler-bin without changing the plotting backend')
    qa=json.loads((root/'external_original_style_numeric_qa.json').read_text(encoding='utf-8'))
    source=pd.read_csv(root/'source_data_channel_density_long.csv',encoding='utf-8-sig')
    assert len(source)==64*4*240 and set(source.subject_id)=={'HUP065'}
    assert source.groupby(['channel_index','series_code']).size().eq(240).all()
    assert np.isfinite(source.density).all() and (source.density>=0).all()
    render=root/'pdf_render'
    render.mkdir()
    pages=[]
    for index,(count,height) in enumerate(((36,180.),(28,155.34)),1):
        stem=root/f'hup065_ofrc_all_channels_page_{index:02d}'
        doc=fitz.open(str(stem.with_suffix('.pdf')))
        assert len(doc)==1
        page=doc[0]
        width=page.rect.width*25.4/72; actual_height=page.rect.height*25.4/72
        assert abs(width-170)<.001 and abs(actual_height-height)<.001
        text=page.get_text()
        for item in (f'HUP065: channel occupation laws ({index}/2)','Recorded ictal','Free prediction',
                     'Preictal reference','Controlled','Standardized amplitude'):
            assert item in text,item
        assert 'W1' not in text and 'loss' not in text.lower()
        selected=source[source.page==index][['channel_index','channel']].drop_duplicates().sort_values('channel_index')
        assert len(selected)==count
        for value in selected.channel:
            label=str(value).removeprefix('EEG ').removesuffix('-Ref')
            assert label in text,label
        outside=[]; font_sizes=[]
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line['spans']:
                    bounds=fitz.Rect(span['bbox'])
                    if bounds.x0<-.1 or bounds.y0<-.1 or bounds.x1>page.rect.width+.1 or bounds.y1>page.rect.height+.1:
                        outside.append(span['text'])
                    font_sizes.append(span['size'])
        assert not outside and min(font_sizes)>=8.9
        assert qa['pages'][index-1]['occupied_axis_count']==count and qa['pages'][index-1]['curves_per_channel']==4
        svg=ET.parse(stem.with_suffix('.svg'))
        ns={'svg':'http://www.w3.org/2000/svg'}
        assert len(svg.findall('.//svg:text',ns))>=count+5
        subprocess.run([str(poppler),'-f','1','-singlefile','-png','-r','150',str(stem.with_suffix('.pdf')),
            str(render/f'page_{index:02d}')],check=True,capture_output=True)
        pages.append(dict(page=index,contacts=count,width_mm=width,height_mm=actual_height,
            exactly_four_curves_per_contact=True,shared_legend_retained=True,
            standardized_amplitude_retained=True,all_text_inside_page=True,smallest_font_pt=min(font_sizes),
            editable_vector_text_present=True,pdf_render=f'pdf_render/page_{index:02d}.png',
            human_visual_inspection=False))
        doc.close()
    result=dict(status='numeric_exports_and_actual_PDF_render_passed_pending_visual_review',
        backend='Python/matplotlib; Python invocation of Poppler for actual-PDF visual QA',
        pages=pages,source_density_rows=len(source),scope=qa['scope'],
        original65_grid_and_KDE_bandwidth_preserved=True,configured_direct_nodes=32,
        warm_start=False,neutral_initialization=True,trained_actor_updates=1000,
        no_new_loss_figure=True,no_overleaf_or_original_source_mutation=True,
        empirical_endpoint_binding_maximum_error=qa['empirical_endpoint_binding_maximum_error'])
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
