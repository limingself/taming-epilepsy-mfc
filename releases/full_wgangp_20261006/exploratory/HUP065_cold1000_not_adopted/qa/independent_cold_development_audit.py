"""Read-only recomputation of all saved HUP065 cold development evidence.

No experiment imports, model/checkpoint deserialization, NPZ access, or calls to
training, finalization, terminal, outer or plotting code. Intermediate output
is explicitly provisional. Final output requires full1000 completion receipts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

QA = Path(__file__).resolve().parent
HERE = QA.parent
RUN = HERE / 'runs' / 'cold_top32_seed20261011_u1000_occ6_worst8'
BASE = HERE / 'frozen_inputs' / 'adopted32_development_baseline'


def jread(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def scoring(channel):
    parts = []
    for run in (0, 1):
        c = channel[channel.run_index == run]
        per = (.5 * c.time_w1_controlled / np.maximum(c.time_w1_free, .01)
               + .5 * c.occupation_w1_controlled / np.maximum(c.occupation_w1_free, .01))
        parts.append(dict(
            run_index=run,
            time_component=.4 * c.time_w1_controlled.mean() / c.time_w1_free.mean(),
            occupation_component=.4 * c.occupation_w1_controlled.mean() / c.occupation_w1_free.mean(),
            worst_channel_component=.2 * np.quantile(per, .9),
            p90_ratio=np.quantile(per, .9),
            mean_abs_log_sd_ratio=np.log(c.symmetric_sd_ratio).mean()))
    result = {key: float(np.mean([part[key] for part in parts]))
              for key in parts[0] if key != 'run_index'}
    result['score'] = sum(result[key] for key in
                          ('time_component', 'occupation_component', 'worst_channel_component'))
    return result


def channels_aggregate(c):
    numeric = ['time_w1_free', 'time_w1_controlled', 'occupation_w1_free',
               'occupation_w1_controlled', 'symmetric_sd_ratio', 'mean_abs_error']
    g = c.groupby('channel_index', sort=True)[numeric].mean()
    if 'channel' in c.columns:
        g['channel'] = c.groupby('channel_index', sort=True).channel.first()
    if 'direct_actuated' in c.columns:
        g['direct_actuated'] = c.groupby('channel_index', sort=True).direct_actuated.first()
    g['mean_abs_log_sd_ratio'] = np.log(c.symmetric_sd_ratio).groupby(c.channel_index).mean()
    g['time_relative_reduction'] = 1 - g.time_w1_controlled / g.time_w1_free
    g['occupation_relative_reduction'] = 1 - g.occupation_w1_controlled / g.occupation_w1_free
    per = .5 * c.time_w1_controlled / np.maximum(c.time_w1_free, .01) + .5 * c.occupation_w1_controlled / np.maximum(c.occupation_w1_free, .01)
    g['mean_channel_score_ratio'] = per.groupby(c.channel_index).mean()
    return g


def numeric_stats(series):
    values = np.asarray(series, dtype=float)
    return dict(minimum=float(values.min()), mean=float(values.mean()),
                median=float(np.median(values)), p90=float(np.quantile(values, .9)),
                maximum=float(values.max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--intermediate', action='store_true')
    parser.add_argument('--through-update', type=int)
    args = parser.parse_args()
    if args.through_update is not None and not args.intermediate:
        raise RuntimeError('A finite evidence cutoff is allowed only for intermediate analysis')
    progress = jread(RUN / 'progress.json')
    protocol = jread(RUN / 'protocol.json')
    preflight = jread(RUN / 'preflight.json')
    checks = []

    def check(name, condition, detail=None):
        checks.append(dict(check=name, passed=bool(condition), detail=detail))
        if not condition:
            raise RuntimeError(f'Independent check failed: {name}: {detail}')

    summary = None if args.intermediate else jread(RUN / 'training_summary.json')
    selection = None if args.intermediate else jread(RUN / 'selection_receipt.json')
    history = pd.read_csv(RUN / 'training_history.csv')
    if not args.intermediate:
        check('1000 complete actual actor updates plus initial row',
              summary['trained_updates'] == 1000
              and summary['status'] == 'completed_neutral_cold1000_parameter_screen'
              and history['update'].tolist() == list(range(1001)))
        check('actual training remained cold and neither terminal nor outer opened',
              summary['actor_and_critic_checkpoint_load_calls'] == []
              and summary['actor_teacher_or_critic_warmstart'] is False
              and summary['outer_opened'] is False and summary['ctx6_opened'] is False
              and summary['all_inputs_unchanged'] is True
              and summary['critic_updates'] == 3024
              and summary['no_new_loss_figure'] is True)
        limit = 1000
    else:
        limit = int(progress['update']) if args.through_update is None else args.through_update
        if limit > int(progress['update']) or limit < 1:
            raise RuntimeError('Intermediate evidence cut-off must already exist in live progress')
        check('intermediate actual history has every reported update',
              history['update'].iloc[:limit + 1].tolist() == list(range(limit + 1)))
    check('current runner and immutable snapshot equal frozen source',
          sha(HERE / 'run_cold_top32.py') == protocol['runner_source_sha256']
          == sha(RUN / 'run_cold_top32_source_snapshot.py'))
    for name, digest in protocol['baseline_source_hashes'].items():
        check(f'adopted baseline file {name} unchanged', sha(BASE / name) == digest)
    check('neutral preflight exact predictor equality and zero input',
          preflight['neutral_free_max_error'] == 0 and preflight['neutral_control_peak'] == 0
          and preflight['forbidden_torch_load_calls'] == [])
    baseline_metrics = protocol['adopted32_development_endpoints']
    base_channels = pd.read_csv(BASE / 'selected_validation_channel_metrics.csv')
    base_score = scoring(base_channels)
    check('adopted comparison same scoring formula from128 original channel rows',
          len(base_channels) == 128
          and abs(base_score['score'] - baseline_metrics['common_selection_score']) < 1e-12)
    candidates = []
    saved_channels = {}
    expected = [0, 1] + list(range(25, limit + 1, 25))
    for update in expected:
        folder = RUN / 'validation' / f'u{update:04d}'
        metrics = jread(folder / 'summary.json')
        channels = pd.read_csv(folder / 'channel_metrics.csv')
        trajectories = pd.read_csv(folder / 'trajectory_metrics.csv')
        check(f'u{update:04d} complete two-run context5 channel ledger',
              len(channels) == 128 and len(trajectories) == 2
              and set(trajectories.context_index) == {5})
        for index in (0, 1):
            c = channels[channels.run_index == index].sort_values('channel_index')
            check(f'u{update:04d} run{index} top32 and all64 channel order',
                  c.channel_index.tolist() == list(range(64))
                  and np.flatnonzero(c.direct_actuated.to_numpy()).tolist() == protocol['selected_indices'])
        sc = scoring(channels)
        time = float(channels.time_w1_controlled.mean())
        occ = float(channels.occupation_w1_controlled.mean())
        check(f'u{update:04d} all reported score and endpoint values recomputed',
              max(abs(sc['score'] - metrics['common_selection_score']),
                  abs(time - metrics['mean_time_w1_controlled']),
                  abs(occ - metrics['mean_occupation_w1_controlled']),
                  abs(sc['mean_abs_log_sd_ratio'] - metrics['mean_abs_log_sd_ratio'])) < 1e-12)
        budget = bool(trajectories.finite.all()
                      and trajectories.maximum_per_actuator_rms.max() <= .405 + 1e-9
                      and trajectories.total_energy.max() <= 3.7908 + 1e-9
                      and trajectories.control_peak.max() <= 1.8 + 1e-9
                      and trajectories.saturation_fraction.max() < .01)
        endpoint_eligible = bool(time <= baseline_metrics['mean_time_w1_controlled'] + 1e-12
                                 and occ <= baseline_metrics['mean_occupation_w1_controlled'] + 1e-12)
        eligible = bool(budget and endpoint_eligible
                        and sc['score'] < baseline_metrics['common_selection_score'] - 1e-8)
        check(f'u{update:04d} budget and dual-endpoint+score eligibility recomputed',
              budget == metrics['budget_pass']
              and endpoint_eligible == metrics['both_mean_endpoints_not_worse_than_adopted32']
              and eligible == metrics['terminal_candidate_eligible'])
        checkpoint = RUN / 'checkpoints' / f'actor_critic_u{update:04d}.pt'
        check(f'u{update:04d} even ineligible actor/critic snapshot retained', checkpoint.is_file())
        candidates.append(dict(update=update, budget_pass=budget, eligible=eligible,
                               score=sc['score'], sd_tie=sc['mean_abs_log_sd_ratio'],
                               time_w1=time, occupation_w1=occ, components=sc,
                               maximum_per_actuator_rms=float(trajectories.maximum_per_actuator_rms.max()),
                               maximum_total_energy=float(trajectories.total_energy.max()),
                               mean_energy=float(trajectories.total_energy.mean()),
                               maximum_peak=float(trajectories.control_peak.max()),
                               maximum_saturation=float(trajectories.saturation_fraction.max()),
                               checkpoint_sha256=sha(checkpoint)))
        saved_channels[update] = channels
    budget_candidates = [row for row in candidates if row['budget_pass']]
    eligible_candidates = [row for row in candidates if row['eligible']]
    budget_best = min(budget_candidates, key=lambda row: (row['score'], row['sd_tie']))
    selected = min(eligible_candidates, key=lambda row: (row['score'], row['sd_tie'])) if eligible_candidates else budget_best
    if not args.intermediate:
        check('budget-first best source recomputed independently',
              budget_best['update'] == selection['best_budget_update'])
        check('final selected actor and eligibility match all42 CSV candidates',
              selected['update'] == selection['selected_update']
              and bool(eligible_candidates) == selection['terminal_eligible']
              and selected['checkpoint_sha256'] == selection['frozen_checkpoint_sha256']
              == summary['frozen_checkpoint_sha256'] == sha(RUN / 'frozen_actor_wgan.pt'))
        check('completed selection used no ctx6 or outer rescue/reselection',
              selection['ctx6_used_for_reselection'] is False
              and selection['outer_used_for_selection'] is False
              and selection['all_validation_candidates_retained'] is True
              and selection['neutral_initialization'] is True)
        audit_file = RUN / 'development_audit.json'
        if audit_file.exists():
            completed_audit = jread(audit_file)
            check('training-agent completed CSV audit independently agrees on frozen selection',
                  completed_audit['status'] == 'passed'
                  and completed_audit['selected_checkpoint_sha256'] == selected['checkpoint_sha256']
                  and completed_audit['dual_endpoint_eligible_candidates'] == len(eligible_candidates))
    chosen = saved_channels[selected['update']]
    cold = channels_aggregate(chosen)
    original = channels_aggregate(base_channels)
    compare = cold.copy()
    for key in ('time_w1_controlled', 'occupation_w1_controlled', 'mean_channel_score_ratio',
                'mean_abs_error', 'mean_abs_log_sd_ratio', 'symmetric_sd_ratio'):
        compare['adopted32_' + key] = original[key]
        compare['cold_minus_adopted32_' + key] = cold[key] - original[key]
    compare['both_endpoints_better_than_adopted32'] = (
        (compare.cold_minus_adopted32_time_w1_controlled < 0)
        & (compare.cold_minus_adopted32_occupation_w1_controlled < 0))
    compare['both_endpoints_worse_than_adopted32'] = (
        (compare.cold_minus_adopted32_time_w1_controlled > 0)
        & (compare.cold_minus_adopted32_occupation_w1_controlled > 0))
    groups = {}
    for name, mask in [('all', np.ones(64, dtype=bool)),
                       ('direct32', compare.direct_actuated.to_numpy()),
                       ('indirect32', ~compare.direct_actuated.to_numpy())]:
        c = compare[mask]
        groups[name] = dict(channels=len(c),
            mean_time_w1_free=float(c.time_w1_free.mean()),
            mean_time_w1_controlled=float(c.time_w1_controlled.mean()),
            mean_occupation_w1_free=float(c.occupation_w1_free.mean()),
            mean_occupation_w1_controlled=float(c.occupation_w1_controlled.mean()),
            time_reduction_of_group_mean=float(1 - c.time_w1_controlled.mean() / c.time_w1_free.mean()),
            occupation_reduction_of_group_mean=float(1 - c.occupation_w1_controlled.mean() / c.occupation_w1_free.mean()),
            mean_time_delta_from_adopted32=float(c.cold_minus_adopted32_time_w1_controlled.mean()),
            mean_occupation_delta_from_adopted32=float(c.cold_minus_adopted32_occupation_w1_controlled.mean()),
            mean_absolute_mean_error=float(c.mean_abs_error.mean()),
            mean_absolute_mean_error_adopted32=float(c.adopted32_mean_abs_error.mean()),
            mean_abs_log_sd_ratio=float(c.mean_abs_log_sd_ratio.mean()),
            mean_abs_log_sd_ratio_adopted32=float(c.adopted32_mean_abs_log_sd_ratio.mean()),
            mean_error_worse_than_adopted32=int((c.cold_minus_adopted32_mean_abs_error > 0).sum()),
            sd_error_worse_than_adopted32=int((c.cold_minus_adopted32_mean_abs_log_sd_ratio > 0).sum()),
            both_endpoints_better_than_adopted32=int(c.both_endpoints_better_than_adopted32.sum()),
            both_endpoints_worse_than_adopted32=int(c.both_endpoints_worse_than_adopted32.sum()),
            time_endpoint_worse_than_free=int((c.time_w1_controlled > c.time_w1_free).sum()),
            occupation_endpoint_worse_than_free=int((c.occupation_w1_controlled > c.occupation_w1_free).sum()),
            worst_channel_ratio_distribution=numeric_stats(c.mean_channel_score_ratio),
            channel_time_reduction_distribution=numeric_stats(c.time_relative_reduction),
            channel_occupation_reduction_distribution=numeric_stats(c.occupation_relative_reduction))
    trained = history[(history['update'] > 0) & (history['update'] <= limit)]
    check('actual rotating10 fitting contexts and critic accounting',
          np.array_equal(trained.fit_context_slot.to_numpy(), np.arange(limit) % 10)
          and np.array_equal(trained.cumulative_critic_updates.to_numpy(), 24 + 3 * np.arange(1, limit + 1)))
    check('WGAN term present every update, canonical existing coefficient overrides only',
          np.abs(trained.train_total - (trained.train_law + .5 * trained.train_adversarial)).max() < 1e-9
          and np.abs(trained.train_law - (trained.train_law_loss + 4 * trained.train_law_occupancy
                                          + 3 * trained.train_law_worst_quantile)).max() < 1e-9)
    check('all numerical training values finite',
          np.isfinite(trained.filter(regex='^(train_|critic_|actor_lr)').to_numpy()).all())
    final_status = None
    terminal_file = RUN / 'terminal_once' / 'final_status.json'
    if not args.intermediate and terminal_file.exists():
        final_status = jread(terminal_file)
        if not eligible_candidates:
            check('failed development eligibility terminates before ctx6 and outer',
                  final_status['status'] == 'development_dual_endpoint_comparison_failed'
                  and final_status['outer_arrays_opened'] is False
                  and final_status['ctx6_opened'] is False
                  and not (RUN / 'terminal_once' / 'ctx6_terminal_veto').exists()
                  and not (RUN / 'terminal_once' / 'outer_posthoc_amendment').exists())
    after_selected = [row for row in candidates if row['update'] > selected['update']]
    delta = dict(
        time_w1_absolute=selected['time_w1'] - baseline_metrics['mean_time_w1_controlled'],
        time_w1_relative=(selected['time_w1'] / baseline_metrics['mean_time_w1_controlled'] - 1),
        occupation_w1_absolute=selected['occupation_w1'] - baseline_metrics['mean_occupation_w1_controlled'],
        occupation_w1_relative=(selected['occupation_w1'] / baseline_metrics['mean_occupation_w1_controlled'] - 1),
        score_absolute=selected['score'] - baseline_metrics['common_selection_score'],
        score_relative=selected['score'] / baseline_metrics['common_selection_score'] - 1,
        mean_energy_relative=selected['mean_energy'] / baseline_metrics['total_energy'] - 1,
        score_components={key: selected['components'][key] - base_score[key]
                          for key in ('time_component', 'occupation_component', 'worst_channel_component')})
    record = dict(
        status='intermediate_only_not_a_final_verdict' if args.intermediate else 'final_development_selection_independently_recomputed',
        timestamp_utc=datetime.now(timezone.utc).isoformat(), actual_actor_updates_read=limit,
        all_saved_candidates=len(candidates), checks_count=len(checks), checks=checks,
        budget_feasible_candidates=len(budget_candidates), terminal_eligible_candidates=len(eligible_candidates),
        budget_best=budget_best, selected=selected, baseline_metrics=baseline_metrics,
        baseline_score_components=base_score, selected_vs_adopted32=delta, channel_groups=groups,
        all_candidates=candidates,
        top10_worst_ratio_channels=compare.sort_values('mean_channel_score_ratio', ascending=False).head(10).reset_index().to_dict('records'),
        top10_occupation_deterioration_vs_adopted32=compare.sort_values('cold_minus_adopted32_occupation_w1_controlled', ascending=False).head(10).reset_index().to_dict('records'),
        hard_budget_utilization=dict(rms=selected['maximum_per_actuator_rms']/.405,
                                     total_energy=selected['maximum_total_energy']/3.7908,
                                     peak=selected['maximum_peak']/1.8),
        actual_late_training_numerical_summary={key: numeric_stats(trained.iloc[-100:][key])
                                                for key in ('train_law', 'train_adversarial', 'critic_gp', 'critic_input_gradient_norm')},
        after_selected_checkpoint_count=len(after_selected),
        after_selected_all_score_at_least_selected=all(row['score'] >= selected['score'] for row in after_selected),
        terminal_status_if_already_completed=final_status,
        auditor_did_not_import_or_execute_experiment=True, no_checkpoint_or_model_deserialization=True,
        no_npz_arrays_or_opaque_outer_report_opened=True, no_new_training_or_veto_or_outer_or_plot_run=True,
        no_loss_figure_created=True, original_scientific_source_not_modified=True,
        causal_limits=[
            'Input-energy change and mean W1 changes are reported separately. Positive energy-relative change means more energy; negative means less. This observed one-seed comparison does not prove that cold initialization or an energy trade-off caused the endpoint gap.',
            'Sub-limit actuator RMS/energy/peak means the hard evaluation caps are not active at this candidate. Soft energy and shape penalties still remain in its unchanged objective.',
            'Validation scores and two W1 endpoints need not be minimized at the same update; reported per-channel p90 and no-harm terms are distinct soft training/selection quantities.',
            'Different initialization/optimization paths and only one cold seed prevent a causal or global optimality claim. Development-only figures cannot show new outer improvement.'
        ])
    stem = f'independent_intermediate_development_u{limit:04d}' if args.intermediate else 'independent_final_development'
    (QA / (stem + '.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    compare.reset_index().to_csv(QA / (stem + '_64channel_comparison.csv'), index=False)
    print(json.dumps(dict(status=record['status'], update=limit, checks=len(checks), candidates=len(candidates),
                          selected_update=selected['update'], eligible=len(eligible_candidates),
                          selected_vs_adopted32=delta, direct_vs_indirect=groups), ensure_ascii=False))


if __name__ == '__main__':
    main()
