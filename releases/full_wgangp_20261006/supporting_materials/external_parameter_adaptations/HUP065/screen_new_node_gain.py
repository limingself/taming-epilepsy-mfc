"""Finite development-only screen of existing new-node spline gain coefficients.

No gradient training, new architecture/loss or outer/veto reads. Sets diagonal
entries of the existing mean/deviation gain-delta parameters on newly added
Part-I-ranked nodes. Common validation score and unchanged budgets are inherited.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

import optimize_hup065 as exp


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes',type=int,choices=(32,40),default=32)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    runner=exp.load_runner()
    out=exp.HERE/f'new_node_gain_grid_top{args.nodes}'
    if out.exists():raise RuntimeError('Refusing finite grid overwrite')
    out.mkdir(parents=True)
    config=SimpleNamespace(seed=20261011,nodes=args.nodes,tau=.2,init_checkpoint=None)
    s=exp.setup(runner,config)
    initial=copy.deepcopy(s['actor'].state_dict())
    old=set(s['migration']['source_actuators'])
    added=[(j,int(c)) for j,c in enumerate(s['selected']) if int(c) not in old]
    candidates=(0.,.35,.8,1.2)
    runner.dump(out/'registered_grid.json',dict(nodes=args.nodes,actuation_fraction=args.nodes/64,
        coefficients=list(candidates),existing_terms='mean_gain_delta and deviation_gain_delta diagonal, constant across all8 spline bases',
        newly_added_channel_indices=[c for j,c in added],outer_used_for_selection=False,ctx6_used_for_selection=False,
        input_energy_cap=3.7908,per_actuator_rms_cap=.405,amplitude_limit=1.8,
        model_sha256=runner.MODEL_HASHES['HUP065'],source_checkpoint_sha256=s['migration']['source_checkpoint_sha256'],
        score_inherited_unchanged=True,source_code_sha256=runner.sha(__file__)))
    cache=runner.validation_cache(s)
    rows=[]
    for coefficient in candidates:
        s['actor'].load_state_dict(initial,strict=True)
        with torch.no_grad():
            for actuator,channel in added:
                s['actor'].mean_gain_delta[:,actuator,channel]=coefficient
                s['actor'].deviation_gain_delta[:,actuator,channel]=coefficient
        metrics,trajectory,channels=exp.validate(runner,s,cache)
        rows.append(dict(coefficient=coefficient,**metrics))
        label=str(coefficient).replace('.','p')
        trajectory.to_csv(out/f'gain_{label}_dev_trajectory.csv',index=False)
        channels.to_csv(out/f'gain_{label}_dev_channel.csv',index=False)
        torch.save(dict(actor_state_dict=copy.deepcopy(s['actor'].state_dict()),
            critic_state_dict=copy.deepcopy(s['critic'].state_dict()),best_update=0,
            training_contract=dict(subject='HUP065',nodes=args.nodes,graph_spread=.2,seed=20261011,
                model_sha256=runner.MODEL_HASHES['HUP065'],input_paths=s['hashes'],
                explicit_warm_start=True,new_node_initial_diagonal_gain=coefficient,
                parameter_only_initialization_screen=True,not_adopted_in_paper=True),
            selected_validation=metrics),out/f'gain_{label}_initial_actor.pt')
        print(json.dumps(rows[-1]),flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'grid_dev_results.csv',index=False)
    eligible=frame[frame.budget_pass]
    winner=eligible.sort_values('common_selection_score').iloc[0].to_dict() if len(eligible) else None
    runner.dump(out/'selection_summary.json',dict(status='completed_finite_gain_initialization_grid',
        nodes=args.nodes,candidates=rows,selected=winner,outer_opened=False,ctx6_opened=False,
        not_adopted_in_paper=True))


if __name__=='__main__':main()
