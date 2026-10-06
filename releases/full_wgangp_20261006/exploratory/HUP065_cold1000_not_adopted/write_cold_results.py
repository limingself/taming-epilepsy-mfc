"""Create an evidence-bound result note after all once-only decisions and QA."""
from __future__ import annotations
import argparse
import json

import numpy as np
import pandas as pd

import run_cold_top32 as experiment
from finalize_cold_top32 import verify_frozen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True)
    args=parser.parse_args()
    run=experiment.HERE/'runs'/args.tag
    if (run/'FINAL_RESULTS.md').exists():raise RuntimeError('Refuse result-note replacement')
    training,receipt,protocol,checkpoint=verify_frozen(run)
    audit=json.loads((run/'development_audit.json').read_text(encoding='utf-8'))
    status=json.loads((run/'terminal_once'/'final_status.json').read_text(encoding='utf-8'))
    qa=json.loads((run/'figures_original_style'/'external_original_style_numeric_qa.json').read_text(encoding='utf-8'))
    if audit['status']!='passed' or audit['selected_checkpoint_sha256']!=experiment.sha(checkpoint):
        raise RuntimeError('Bound successful development audit required')
    selected=receipt['selected_update']
    folder=run/'validation'/f'u{selected:04d}'
    metrics=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    old=receipt['adopted32_development_baseline']
    channels=pd.read_csv(folder/'channel_metrics.csv')
    controls_summary=dict(maximum_per_actuator_rms=metrics['maximum_per_actuator_rms'],
        maximum_total_energy=metrics['maximum_total_energy'],control_peak=metrics['control_peak'],
        budget_pass=metrics['budget_pass'])
    grouped=channels.groupby('channel_index',sort=True)
    crosscase=pd.DataFrame(dict(channel=grouped.channel.first(),direct_actuated=grouped.direct_actuated.first(),
        mean_time_w1_free=grouped.time_w1_free.mean(),mean_time_w1_controlled=grouped.time_w1_controlled.mean(),
        mean_occupation_w1_free=grouped.occupation_w1_free.mean(),mean_occupation_w1_controlled=grouped.occupation_w1_controlled.mean(),
        both_improved_each_dev_run=grouped.both_improved.all()))
    crosscase['both_mean_endpoints_improved']=((crosscase.mean_time_w1_controlled<crosscase.mean_time_w1_free)&
        (crosscase.mean_occupation_w1_controlled<crosscase.mean_occupation_w1_free))
    crosscase.to_csv(run/'selected_development_cross_run_channels.csv')
    result=dict(subject='HUP065',experiment='single-seed neutral32 WGAN-GP1000 existing-parameter screen',
        initialization='actual neutral; no previous teacher, actor or critic checkpoint loaded during training',
        trained_actor_updates=1000,critic_updates=3024,selected_actor_update=selected,
        elapsed_training_seconds=training['elapsed_seconds'],configured_direct_nodes=32,total_channels=64,
        actuation_fraction=.5,weighted_mask_indices=protocol['selected_indices'],
        adopted32_development_baseline=old,cold_selected_development=metrics,
        relative_change_vs_adopted32_development_percent={key:100*(metrics[key]/old[key]-1) for key in
            ('mean_time_w1_controlled','mean_occupation_w1_controlled','common_selection_score')},
        development_budget=controls_summary,dual_endpoint_terminal_eligibility=receipt['terminal_eligible'],
        independent_development_audit_checks=audit['checks_count'],terminal_status=status,
        selected_checkpoint_sha256=experiment.sha(checkpoint),
        development_channel_counts=dict(mean_both_endpoints_improved=int(crosscase.both_mean_endpoints_improved.sum()),
            both_improved_each_run=int(crosscase.both_improved_each_dev_run.sum()),
            mean_both_nondirect=int(crosscase.loc[~crosscase.direct_actuated,'both_mean_endpoints_improved'].sum())),
        preview_scope=qa['scope'],unchanged_original65_layout=True,no_new_loss_figure=True,
        historical_code_results_and_overleaf_untouched=True,not_automatically_adopted=True,
        no_global_optimum_or_every_channel_success_claim=True)
    (run/'FINAL_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if not receipt['terminal_eligible']:
        outcome='本次冷启动未满足预声明的“双dev平均端点均不比已采用32版差、共同score更低”条件。因此只保存开发预览，未开启context6或外层run03，不替换当前结果。'
    elif status['status']=='ctx6_terminal_veto_failed':
        outcome='本次冷启动开发端点满足门槛，但context6终端veto失败。外层run03未开启，也未根据veto重新选checkpoint。返回开发预览，不替换当前结果。'
    else:
        outcome='本次冷启动开发比较与context6终端veto通过，随后仅一次评价已揭示run03。外层结果是post-hoc诊断，不是新的locked held-out验证；是否采用仍需作者确认。'
    lines=['# HUP065冷启动32节点实验：本次真实结果','',outcome,'',
        f"完整完成1000次actor更新、3024次critic更新，用时{training['elapsed_seconds']/60:.2f}分钟；选中actor update{selected}。1000是更新数，不是完整数据epoch。",'',
        '| 同一context5评价端点 | 当前采用32 warm-start版 | 本次cold最佳冻结点 |',
        '|---|---:|---:|',
        f"| 平均time-resolved W1 | {old['mean_time_w1_controlled']:.9f} | {metrics['mean_time_w1_controlled']:.9f} |",
        f"| 平均occupation W1 | {old['mean_occupation_w1_controlled']:.9f} | {metrics['mean_occupation_w1_controlled']:.9f} |",
        f"| 共同selection score（越小越好） | {old['common_selection_score']:.9f} | {metrics['common_selection_score']:.9f} |",'',
        f"本次选择点最大单执行器RMS={metrics['maximum_per_actuator_rms']:.6f}（上限0.405），最大总能量={metrics['maximum_total_energy']:.6f}（上限3.7908），峰值={metrics['control_peak']:.6f}（上限1.8）。原预算pass={metrics['budget_pass']}。",'',
        '模型、噪声、scaler、患者参考和Part-I加权前32节点均冻结。控制器从neutral初始化，既有occupation/worst-quantile系数分别为6/8；不增加误差项、不修改actor结构、不加载旧actor/critic/teacher。所有42个开发检查点及失败候选均保留。',
        '',f"CSV独立重算通过{audit['checks_count']}项检查。通道图使用原65四曲线、36+28分页、原240点横轴窗口及0.16绝对KDE带宽；没有为美观改样本尺度或密度归一化。预览范围：{qa['scope']['role']}，context{qa['scope']['context']} / bank{qa['scope']['bank']}。",'',
        '本目录为独立预览归档，不覆盖已采用论文、Overleaf、原VS Code工程或既往模型。数字和全部参数见FINAL_RESULTS.json与protocol.json。']
    (run/'FINAL_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('selected_actor_update','dual_endpoint_terminal_eligibility',
        'relative_change_vs_adopted32_development_percent','preview_scope')}),flush=True)


if __name__=='__main__':main()
