"""Combine unchanged two-page figure components into one development-preview PDF."""
import argparse
import json

import fitz

import run_cold_top32 as experiment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True)
    args=parser.parse_args()
    root=experiment.HERE/'runs'/args.tag/'figures_original_style'
    output=root/'HUP065_top32_cold1000_development_preview_2pages.pdf'
    if output.exists():raise RuntimeError('Refuse existing combined-preview overwrite')
    qa=json.loads((root/'external_original_style_numeric_qa.json').read_text(encoding='utf-8'))
    if qa['scope']['role']!='development_only_failed_improvement_preview':
        raise RuntimeError('This combined deliverable is explicitly development-only')
    writer=fitz.open()
    inputs=[]
    for index in (1,2):
        path=root/f'hup065_ofrc_all_channels_page_{index:02d}.pdf'
        digest=experiment.sha(path)
        if digest!=qa['pages'][index-1]['artifact_sha256']['pdf']:
            raise RuntimeError('Source component PDF bytes changed since numeric QA')
        reader=fitz.open(str(path))
        if len(reader)!=1:raise RuntimeError('One page per unchanged source component required')
        writer.insert_pdf(reader)
        reader.close()
        inputs.append(dict(path=path.name,sha256=digest))
    writer.set_metadata({'title':'HUP065 top32 cold1000 development preview',
        'subject':'Run-01/context5, selected update900; development only, not an outer-validation improvement or manuscript replacement'})
    writer.save(str(output))
    writer.close()
    doc=fitz.open(str(output))
    if len(doc)!=2:raise RuntimeError('Combined preview must contain exactly two pages')
    for index,page in enumerate(doc,1):
        if f'HUP065: channel occupation laws ({index}/2)' not in page.get_text():
            raise RuntimeError('Source figure titles changed during PDF assembly')
        expected=180. if index==1 else 155.34
        if abs(page.rect.width*25.4/72-170.)>.001 or abs(page.rect.height*25.4/72-expected)>.001:
            raise RuntimeError('Original figure geometry changed during assembly')
    doc.close()
    result=dict(status='passed_unchanged_two_page_PDF_assembly',input_components=inputs,
        output=output.name,output_sha256=experiment.sha(output),pages=2,
        original_vector_geometry_text_curves_retained=True,scope=qa['scope'],
        not_external_or_manuscript_replacement=True)
    (root/'combined_preview_qa.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
