"""Recompute an authorized dev-only cold65 preview from its saved samples.

Requires completed1000/final guard decision before loading the new development
preview NPZ. Never reads any sealed outer data or executes an experiment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import fitz
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, wasserstein_distance

QA = Path(__file__).resolve().parent
HERE = QA.parent
RUN = HERE / 'runs' / 'cold_top32_seed20261011_u1000_occ6_worst8'
FIGURES = RUN / 'figures_original_style'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def exact_time_w1(x, y):
    return float(np.mean([wasserstein_distance(x[:, t], y[:, t])
                          for t in range(256)]))


def main():
    training = read(RUN / 'training_summary.json')
    selection = read(RUN / 'selection_receipt.json')
    terminal = read(RUN / 'terminal_once/final_status.json')
    receipt = read(FIGURES / 'external_original_style_numeric_qa.json')
    scope = receipt['scope']
    protocol = read(RUN / 'protocol.json')
    checks = []

    def check(name, condition, detail=None):
        checks.append(dict(check=name, passed=bool(condition), detail=detail))
        if not condition:
            raise RuntimeError(f'Independent preview check failed: {name}: {detail}')

    check('completed cold1000 actual actor updates', training['trained_updates'] == 1000
          and training['actor_teacher_or_critic_warmstart'] is False)
    check('failed qualification scope cannot imply an outer experiment',
          terminal['status'] in ('development_dual_endpoint_comparison_failed', 'ctx6_terminal_veto_failed')
          and terminal['outer_arrays_opened'] is False)
    if terminal['status'] == 'development_dual_endpoint_comparison_failed':
        check('failed dual endpoint leaves ctx6 unopened as well',
              terminal['ctx6_opened'] is False
              and not (RUN / 'terminal_once/ctx6_terminal_veto').exists())
    check('no new outer output directory exists', not (RUN / 'terminal_once/outer_posthoc_amendment').exists())
    check('preview explicitly development run01 context5 fixed seed, no cherry-pick',
          scope['role'] == 'development_only_failed_improvement_preview'
          and scope['run'] == 'run-01' and scope['context'] == 5 and scope['bank'] == 0
          and scope['noise_seed'] == 20260922
          and scope['outer_arrays_opened_by_plot'] is False
          and scope['not_external_improvement_evidence'] is True)
    check('plot metadata correctly distinguishes new dev replay from no retraining/rescaling',
          receipt['no_new_training_by_plot'] is True
          and receipt['no_sample_or_density_rescaling'] is True
          and receipt['new_development_preview_replay'] is True)
    check('preview frozen checkpoint source identity',
          receipt['source_sha256']['checkpoint'] == selection['frozen_checkpoint_sha256']
          == training['frozen_checkpoint_sha256'] == sha(RUN / 'frozen_actor_wgan.pt'))
    display = RUN / 'development_preview_context05_run01/display_context_rollout.npz'
    check('saved dev preview digest matches scope', sha(display) == scope['display_source_sha256'])
    # Only the newly authorized development preview NPZ is opened, after all
    # classification/once-only guard checks above. No opaque outer path is read.
    with np.load(display, allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    check('all original4 sample populations finite',
          arrays['observed_standardized'].shape == (256, 64)
          and arrays['free_standardized'].shape == (32, 256, 64)
          and arrays['controlled_standardized'].shape == (32, 256, 64)
          and arrays['reference_standardized'].shape == (30, 256, 64)
          and all(np.isfinite(arrays[key]).all() for key in
                  ('observed_standardized', 'free_standardized', 'controlled_standardized', 'reference_standardized')))
    check('weighted top32 source mask retained in displayed samples',
          arrays['selected_indices'].tolist() == protocol['selected_indices']
          == np.flatnonzero(arrays['direct_mask']).tolist())
    density = pd.read_csv(FIGURES / 'source_data_channel_density_long.csv', encoding='utf-8-sig')
    original = pd.read_csv(HERE / 'frozen_inputs/original65_density_grid.csv', encoding='utf-8-sig')
    metrics = pd.read_csv(FIGURES / 'source_data_display_metrics.csv', encoding='utf-8-sig')
    frozen = pd.read_csv(RUN / 'validation' / f"u{selection['selected_update']:04d}" / 'channel_metrics.csv')
    frozen = frozen[frozen.run_index == 0].sort_values('channel_index')
    check('every64 contact has exactly4 curves and240 coordinates',
          len(density) == 64 * 4 * 240
          and set(density.series_code) == {'O', 'F', 'R', 'C'}
          and density.groupby(['channel_index', 'series_code']).size().eq(240).all())
    series = {'O': 'observed_standardized', 'F': 'free_standardized',
              'R': 'reference_standardized', 'C': 'controlled_standardized'}
    maximum_density_error = 0.
    maximum_grid_error = 0.
    calculated = []
    for channel in range(64):
        reference = arrays['reference_standardized'][..., channel]
        values = dict(channel_index=channel,
                      time_w1_free=exact_time_w1(arrays['free_standardized'][..., channel], reference),
                      time_w1_controlled=exact_time_w1(arrays['controlled_standardized'][..., channel], reference),
                      occupation_w1_free=wasserstein_distance(arrays['free_standardized'][..., channel].ravel(), reference.ravel()),
                      occupation_w1_controlled=wasserstein_distance(arrays['controlled_standardized'][..., channel].ravel(), reference.ravel()))
        calculated.append(values)
        for code, key in series.items():
            rows = density[(density.channel_index == channel) & (density.series_code == code)]
            old = original[(original.channel_index == channel) & (original.series_code == code)]
            grid = rows.standardized_amplitude.to_numpy()
            maximum_grid_error = max(maximum_grid_error, float(np.abs(grid - old.standardized_amplitude.to_numpy()).max()))
            samples = arrays[key][..., channel].ravel()
            sd = float(samples.std(ddof=1))
            if sd < 1e-10:
                z = (grid - samples.mean()) / .16
                actual = np.exp(-.5 * z * z) / (.16 * np.sqrt(2 * np.pi))
            else:
                actual = gaussian_kde(samples, bw_method=.16 / sd)(grid)
            maximum_density_error = max(maximum_density_error, float(np.abs(actual - rows.density.to_numpy()).max()))
    check('all64x4 original x coordinate grids exactly retained', maximum_grid_error == 0, maximum_grid_error)
    check('all64x4 KDE curves recomputed from unchanged saved samples and absolute h=.16',
          maximum_density_error < 1e-12, maximum_density_error)
    calculated = pd.DataFrame(calculated).sort_values('channel_index')
    binding_error = max(float(np.abs(calculated[key].to_numpy() - frame[key].to_numpy()).max())
                        for key in ('time_w1_free', 'time_w1_controlled', 'occupation_w1_free', 'occupation_w1_controlled')
                        for frame in (metrics.sort_values('channel_index'), frozen))
    check('all-tail empirical endpoints match both displayCSV and selected dev run0CSV', binding_error < 1e-12, binding_error)
    controls = arrays['controls']
    actual_inputs = dict(maximum_per_actuator_rms=float(np.sqrt(np.mean(controls**2, axis=(0, 1))).max()),
                         total_energy=float(np.mean(np.sum(controls**2, axis=-1))),
                         control_peak=float(np.abs(controls).max()))
    check('display budget values independently match saved controls',
          max(abs(value - receipt[key]) for key, value in actual_inputs.items()) < 1e-12)
    check('display remains within original unraised input caps',
          actual_inputs['maximum_per_actuator_rms'] <= .405 + 1e-9
          and actual_inputs['total_energy'] <= 3.7908 + 1e-9
          and actual_inputs['control_peak'] <= 1.8 + 1e-9)
    svg_ns = {'s': 'http://www.w3.org/2000/svg'}
    artifact_records = []
    for page, count in ((1, 36), (2, 28)):
        base = FIGURES / f'hup065_ofrc_all_channels_page_{page:02d}'
        artifact_hashes = {ext: sha(base.with_suffix('.' + ext)) for ext in ('pdf', 'png', 'svg')}
        declared = receipt['pages'][page - 1]['artifact_sha256']
        check(f'page{page} exact PDF/PNG/SVG artifact binding', artifact_hashes == declared)
        pdf = fitz.open(base.with_suffix('.pdf'))
        text = ''.join(p.get_text() for p in pdf)
        check(f'page{page} original width and single-page vector export', len(pdf) == 1
              and abs(pdf[0].rect.width * 25.4 / 72 - 170) < .001)
        for name in ('Recorded ictal', 'Free prediction', 'Preictal reference', 'Controlled', 'Standardized amplitude'):
            check(f'page{page} retained label {name}', name in text)
        check(f'page{page} no actor comparator, loss or distance/gate annotation',
              not any(word in text.lower() for word in ('pretrained', 'actor', 'loss', 'w1', 'gate')))
        svg = ET.parse(base.with_suffix('.svg'))
        axes = [node for node in svg.iter() if re.fullmatch(r'axes_\d+', node.attrib.get('id', ''))]
        check(f'page{page} exported occupied axis count', len(axes) == count)
        curve_counts = {}
        for color in ('#272727', '#9a9a9a', '#3c8d62', '#2166ac'):
            curve_counts[color] = sum(1 for axis in axes for node in axis.iter()
                                       if node.tag.endswith('path') and f'stroke: {color}' in node.attrib.get('style', '').lower())
        check(f'page{page} native SVG has each of4 curve strokes once per channel',
              all(value == count for value in curve_counts.values()), curve_counts)
        artifact_records.append(dict(page=page, channels=count, sha256=artifact_hashes,
                                     curve_counts=curve_counts, actual_pdf_width_mm=pdf[0].rect.width * 25.4 / 72))
        pdf.close()
    check('no new loss plot produced in current run',
          not any('loss' in path.stem.lower() and path.suffix.lower() in ('.pdf', '.png', '.svg')
                  for path in RUN.rglob('*') if path.is_file()))
    record = dict(status='passed_independent_dev_only_preview_numeric_and_vector_audit',
                  checks_count=len(checks), checks=checks, scope=scope, selected_update=selection['selected_update'],
                  maximum_density_recalculation_error=maximum_density_error,
                  maximum_empirical_endpoint_binding_error=binding_error,
                  original_grid_error=maximum_grid_error, actual_inputs=actual_inputs,
                  artifacts=artifact_records, source_checkpoint_sha256=selection['frozen_checkpoint_sha256'],
                  no_training_or_veto_or_outer_executed=True, no_checkpoint_or_model_deserialization=True,
                  only_saved_authorized_development_preview_arrays_loaded=True,
                  sealed_or_new_outer_arrays_never_opened=True, no_scientific_sources_modified=True,
                  human_visual_review=False)
    (QA / 'independent_final_preview_numeric_vector_audit.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=record['status'], checks=len(checks), selected_update=selection['selected_update'],
                          maximum_density_error=maximum_density_error, empirical_binding_error=binding_error), ensure_ascii=False))


if __name__ == '__main__':
    main()
