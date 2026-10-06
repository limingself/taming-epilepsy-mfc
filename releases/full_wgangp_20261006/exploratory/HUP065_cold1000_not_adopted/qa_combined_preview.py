"""Bind combined PDF page content/fonts/geometry and record rendering differences."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import fitz
from PIL import Image

import run_cold_top32 as experiment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True)
    args=parser.parse_args()
    root=experiment.HERE/'runs'/args.tag/'figures_original_style'
    output=root/'combined_render_qa.json'
    if output.exists():raise RuntimeError('Refuse combined QA replacement')
    combined=root/'HUP065_top32_cold1000_development_preview_2pages.pdf'
    result=json.loads((root/'combined_preview_qa.json').read_text(encoding='utf-8'))
    if experiment.sha(combined)!=result['output_sha256']:raise RuntimeError('Combined preview changed since assembly')
    poppler=Path.home()/'.cache'/'codex-runtimes'/'codex-primary-runtime'/'dependencies'/'native'/'poppler'/'Library'/'bin'/'pdftoppm.exe'
    if not poppler.is_file():raise RuntimeError('Poppler missing')
    directory=root/'combined_pdf_render'
    directory.mkdir(exist_ok=True)
    merged_doc=fitz.open(str(combined))
    comparisons=[]
    for index in (1,2):
        prefix=directory/f'page_{index:02d}'
        subprocess.run([str(poppler),'-f',str(index),'-l',str(index),'-singlefile','-png','-r','150',
            str(combined),str(prefix)],check=True,capture_output=True)
        original=np.asarray(Image.open(root/'pdf_render'/f'page_{index:02d}.png'))
        merged=np.asarray(Image.open(prefix.with_suffix('.png')))
        component_doc=fitz.open(str(root/f'hup065_ofrc_all_channels_page_{index:02d}.pdf'))
        original_page=component_doc[0]; merged_page=merged_doc[index-1]
        content_sha=hashlib.sha256(original_page.read_contents()).hexdigest()
        if hashlib.sha256(merged_page.read_contents()).hexdigest()!=content_sha or original_page.mediabox!=merged_page.mediabox:
            raise RuntimeError('Combined PDF changed source page content/MediaBox')
        def fonts(document,page):
            return sorted((font[3],font[1],font[2],font[5],hashlib.sha256(document.extract_font(font[0])[3]).hexdigest()) for font in page.get_fonts())
        if fonts(component_doc,original_page)!=fonts(merged_doc,merged_page):raise RuntimeError('Embedded source fonts changed')
        if original.shape!=merged.shape:raise RuntimeError('Combined PDF raster shape changed')
        count=int(np.any(original!=merged,axis=2).sum())
        fraction=float(count/(original.shape[0]*original.shape[1]))
        maximum=int(np.abs(original.astype(int)-merged.astype(int)).max())
        comparisons.append(dict(page=index,source_contentstream_sha256=content_sha,
            contentstream_MediaBox_and_embedded_fonts_identical=True,
            changed_raster_pixels=count,changed_pixel_fraction=fraction,maximum_RGB_difference=maximum,
            tiny_antialias_differences_recorded_not_scientific_data_changes=True,
            rendered_png_sha256=experiment.sha(prefix.with_suffix('.png'))))
        component_doc.close()
    merged_doc.close()
    output.write_text(json.dumps(dict(status='passed_original_page_content_geometry_and_font_binding',pages=comparisons,
        combined_pdf_sha256=experiment.sha(combined),scope=result['scope']),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='passed_original_page_content_geometry_and_font_binding',pages=comparisons)),flush=True)


if __name__=='__main__':main()
