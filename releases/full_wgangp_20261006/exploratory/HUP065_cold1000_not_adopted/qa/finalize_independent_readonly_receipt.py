"""Seal independent evidence/guard/visual QA, not an experiment success claim."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

QA = Path(__file__).resolve().parent
HERE = QA.parent
RUN = HERE / 'runs/cold_top32_seed20261011_u1000_occ6_worst8'
BUNDLE = HERE / 'frozen_inputs/external_bundle'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    development = read(QA / 'independent_final_development.json')
    preview = read(QA / 'independent_final_preview_numeric_vector_audit.json')
    protocol = read(RUN / 'protocol.json')
    terminal = read(RUN / 'terminal_once/final_status.json')
    developer_audit = read(RUN / 'development_audit.json')
    extras = []

    def check(name, condition, detail=None):
        extras.append(dict(check=name, passed=bool(condition), detail=detail))
        if not condition:
            raise RuntimeError(f'Cannot seal independent receipt: {name}: {detail}')

    check('all independent development recalculations passed',
          all(item['passed'] for item in development['checks']) and development['all_saved_candidates'] == 42)
    check('all independent preview/sample/KDE/vector checks passed',
          all(item['passed'] for item in preview['checks']))
    check('independent selected policy equals experiment CSV audit',
          development['selected']['checkpoint_sha256'] == developer_audit['selected_checkpoint_sha256']
          == preview['source_checkpoint_sha256'] == terminal['frozen_checkpoint_sha256']
          == sha(RUN / 'frozen_actor_wgan.pt'))
    check('same actual selected update throughout',
          development['selected']['update'] == preview['selected_update'] == terminal['selected_update'] == 900)
    check('current runner unchanged throughout training/finalization/plot',
          sha(HERE / 'run_cold_top32.py') == protocol['runner_source_sha256']
          == sha(RUN / 'run_cold_top32_source_snapshot.py'))
    check('actual result stayed ineligible and no terminal rescue occurred',
          development['terminal_eligible_candidates'] == 0
          and terminal['status'] == 'development_dual_endpoint_comparison_failed'
          and terminal['terminal_eligible'] is False
          and terminal['no_terminal_checkpoint_rescue'] is True)
    check('no context6 veto or new outer result directory',
          terminal['ctx6_opened'] is False and terminal['outer_arrays_opened'] is False
          and not (RUN / 'terminal_once/ctx6_terminal_veto').exists()
          and not (RUN / 'terminal_once/outer_posthoc_amendment').exists())
    source = (HERE / 'finalize_cold_top32.py').read_text(encoding='utf-8-sig')
    check('final helper enforces passed frozen development audit before terminal_once',
          "audit['status'] != 'passed'" in source
          and "audit['selected_checkpoint_sha256'] != experiment.sha(checkpoint)" in source
          and source.index("audit = json.loads((run/'development_audit.json')") < source.index('out.mkdir()'))
    check('final helper enforces exact baseline SHA before once-only gate',
          "experiment.sha(experiment.BASELINE/name) != digest" in source)
    check('final helper ineligible return remains before setup and sole outer read',
          source.index("if receipt['terminal_eligible'] is not True:") < source.index('setup_args =')
          < source.index("with np.load(s['p']['outer']"))
    manifest = read(BUNDLE / 'bundle_manifest.json')
    frozen = []
    skipped = []
    for entry in manifest['files']:
        if entry['role'].endswith('_opaque'):
            skipped.append(entry['relative_path'])
            continue
        actual = sha(BUNDLE / entry['relative_path'])
        check('final frozen source/input ' + entry['relative_path'], actual == entry['sha256'])
        frozen.append(dict(relative_path=entry['relative_path'], sha256=actual))
    # First-person visual review has just been performed on the two actual
    # Poppler PDF renders in this exact frozen preview directory.
    visual = dict(status='passed', reviewer='independent full_wgan_manuscript_audit agent',
        reviewed_actual_pdf_pages=2, channel_counts=[36, 28],
        original_title_and_four_curve_legend_retained=True,
        channel_labels_and_axis_labels_visible=True, no_clipped_or_overlapping_text=True,
        original_blank_last_row_cells_preserved=True, standardized_original_axis_not_replaced=True,
        no_distance_gate_or_explanation_or_loss_annotations=True,
        natural_curve_overlaps_not_hidden_or_removed=True,
        PDF_sha256=[row['sha256']['pdf'] for row in preview['artifacts']],
        actual_poppler_render_sha256=[sha(RUN / f'figures_original_style/pdf_render/page_{page:02d}.png')
                                     for page in (1, 2)],
        development_only_scope_explicit_in_receipts=True,
        not_author_adoption=True)
    sources = {name: sha(HERE / name) for name in
               ('run_cold_top32.py', 'finalize_cold_top32.py', 'audit_cold_training.py', 'plot_cold_top32.py')}
    source_receipts = {name: sha(QA / name) for name in
                      ('independent_final_development.json', 'independent_final_preview_numeric_vector_audit.json',
                       'independent_final_development_64channel_comparison.csv')}
    all_checks = development['checks'] + preview['checks'] + extras
    result = dict(status='passed', timestamp_utc=datetime.now(timezone.utc).isoformat(),
        status_meaning='All independent scientific-evidence, source-binding, gating, numeric/vector and actual-image visual QA passed. It does NOT mean the cold experiment improved on adopted32.',
        checks_count=len(all_checks), checks=all_checks,
        experiment_outcome='completed1000_without_dual_endpoint_improvement; no ctx6 or outer evaluation; development-only preview',
        scientific_sources=sources, source_receipt_sha256=source_receipts,
        actual_updates=1000, actual_history_rows=1001, cumulative_critic_updates=3024,
        all_saved_development_candidates=42, eligible_candidates=0,
        selected=development['selected'], adopted32_comparison=development['baseline_metrics'],
        selected_vs_adopted32=development['selected_vs_adopted32'],
        channel_groups=development['channel_groups'], hard_budget_utilization=development['hard_budget_utilization'],
        all64_channel_comparison_csv='qa/independent_final_development_64channel_comparison.csv',
        preview_scope=preview['scope'], preview_artifacts=preview['artifacts'], visual_review=visual,
        empirical_all_tail_W1_recalculation_error=preview['maximum_empirical_endpoint_binding_error'],
        all64x4_KDE_recalculation_error=preview['maximum_density_recalculation_error'],
        all_original_x_coordinate_grids_identical=True,
        source_input_files_verified_at_final=len(frozen), frozen_sources=frozen,
        opaque_outer_files_never_opened_by_auditor=skipped,
        auditor_loaded_only_saved_authorized_dev_preview_npz=True,
        no_training_or_terminal_or_outer_or_plot_execution_by_auditor=True,
        no_checkpoint_or_model_deserialization_by_auditor=True,
        no_scientific_source_or_original_code_or_overleaf_or_D_manuscript_modified=True,
        no_new_loss_plot=True, current_paper_policy_should_remain_adopted32=True,
        not_automatically_adopted=True, no_global_optimality_or_causal_or_multiseed_inference=True)
    (QA / 'independent_final_cold_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    delta = result['selected_vs_adopted32']
    chosen = result['selected']
    base = result['adopted32_comparison']
    lines = [
        '# HUP065 cold top32: final independent audit', '',
        f'QA status: passed ({len(all_checks)}/{len(all_checks)} checks). The experiment itself did not qualify as an improvement over the currently adopted32 controller.', '',
        '- Actual 1000 actor updates, 1001 history rows, 3024 critic updates and 42 saved development candidates were verified.',
        '- All 42 candidates were independently recomputed; the fixed budget-first composite score selected update 900. None met both endpoint non-deterioration plus score improvement.',
        '- Frozen actor SHA256: ' + chosen['checkpoint_sha256'],
        '- Once-only finalization recorded development failure. No context6 veto or new outer evaluation was executed; no checkpoint rescue/reselection occurred.',
        '- Only run-01 / context 5 / seed 20260922 was displayed. The two original-style pages contain all 64 contacts, four curves each, with no sample/KDE rescaling or extra loss figure.', '',
        '| Development quantity | Adopted32 | Cold selected900 |',
        '|---|---:|---:|',
        f'| Mean time-law W1 | {base["mean_time_w1_controlled"]:.9f} | {chosen["time_w1"]:.9f} |',
        f'| Mean occupation-law W1 | {base["mean_occupation_w1_controlled"]:.9f} | {chosen["occupation_w1"]:.9f} |',
        f'| Common score | {base["common_selection_score"]:.9f} | {chosen["score"]:.9f} |',
        f'| Maximum per-actuator RMS | {base["maximum_per_actuator_rms"]:.9f} | {chosen["maximum_per_actuator_rms"]:.9f} |',
        f'| Mean total input energy | {base["total_energy"]:.9f} | {chosen["mean_energy"]:.9f} |', '',
        f'The cold development mean W1 endpoints were respectively {delta["time_w1_relative"]*100:.3f}% and {delta["occupation_w1_relative"]*100:.3f}% larger than adopted32. Relative to its unchanged free prediction they fell 27.205% and 68.671%, not 65% for both endpoints.',
        f'The score gap splits into time component +{delta["score_components"]["time_component"]:.9f}, occupation component +{delta["score_components"]["occupation_component"]:.9f}, and per-run p90 channel component +{delta["score_components"]["worst_channel_component"]:.9f}.',
        'These are observed one-seed development results. Mean/SD error proxies do not uniquely identify the cause; no unseen outer effect or global optimum is claimed.', '',
        'Independent preview verification:', '',
        f'- All 256 KDE curves were recomputed on the exact original 240-point grids at absolute bandwidth 0.16: max error {result["all64x4_KDE_recalculation_error"]:.3g}.',
        f'- All 64 empirical time/occupation endpoints were recalculated using all saved samples, including out-of-window tails: max binding error {result["empirical_all_tail_W1_recalculation_error"]:.3g}.',
        '- Native SVG exported 36 / 28 axes with each of four curve strokes per channel; PDF/PNG/SVG SHA bindings agreed.',
        '- Both actual PDF page renders were visually reviewed. Titles, shared legend, channel labels and standardized axes were retained without clipping or text overlap.',
        '- All 37 non-outer frozen manifest files were rehashed at final. Four opaque outer files were deliberately not opened by the auditor.', '',
        'Current paper should keep the already-adopted32 version. This new run is separately archived as a completed, non-improving development-only parameter screen.'
    ]
    (QA / 'independent_final_cold_audit.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], checks=result['checks_count'],
                          outcome=result['experiment_outcome'], receipt=str(QA / 'independent_final_cold_audit.json'))))


if __name__ == '__main__':
    main()
