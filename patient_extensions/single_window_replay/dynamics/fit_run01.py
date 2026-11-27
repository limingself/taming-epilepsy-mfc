"""Re-fit the HUP060 final Graph-RC-SDE specification to the selected patient’s run-01 only."""
from pathlib import Path
from hashlib import sha256
from dataclasses import asdict
import sys
import json
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from threadpoolctl import threadpool_limits
import joblib

from replay_config import settings
case = settings()
OUT = case.output / 'dynamics'
sys.path.insert(0, str(case.project))
from mfc_pipeline import part2_state_dependent_rc_sde as p2
from mfc_pipeline import part2_data_pipeline as pipe
from mfc_pipeline import network
from mfc_pipeline.rc_v2 import select_latent_dimension

source = case.run01_ictal
def digest(path):
    with path.open('rb') as fh:
        return sha256(fh.read()).hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'fit_manifest.json').exists():
        raise FileExistsError('frozen fit exists; use a fresh directory')
    with np.load(source, allow_pickle=False) as a:
        x = np.array(a['data'].T, dtype=np.float64, order='C')
        channels = a['channel_names'].astype(str)
        meta = json.loads(str(a['metadata_json']))
    assert x.shape == (8192, case.channels) and len(set(channels)) == case.channels and np.isfinite(x).all()
    assert meta['run']=='run-01' and meta['sfreq']==256 and meta['provenance']['preprocessing_fully_causal']
    cfg = p2.ModelConfig(reservoir_size=96, spectral_radius=.40, leak_rate=.10, input_scale=.05,
                         delays_samples=(1,8,32), ridge_alpha=1., reservoir_sparsity=.9, washout_samples=32,
                         random_seed=11, latent_variance_threshold=.95, latent_components_min=8, latent_components_max=24,
                         diffusion_shrinkage=.05,diffusion_relative_jitter=1e-8,
                         diffusion_mode='state_dependent',diffusion_ridge_alpha=.1)
    hz=256
    train=x[:22*hz]; valid=x[24*hz:]
    early=p2.ResidualGraphRCSDE(cfg,pipe.compute_adjacency([train],hz)).fit([train])
    inner=p2.evaluate_candidate(early,valid,[16,32,64,128],512,6,20260727)
    inner.to_csv(OUT/'run01_inner_validation.csv',index=False)
    print('inner validation finished',flush=True)
    graph=pipe.compute_adjacency([x],hz)
    model=p2.ResidualGraphRCSDE(cfg,graph).fit([x])
    assert model.training_pairs==8127 and np.count_nonzero(graph)>0
    joblib.dump(model,OUT/'plant.joblib',compress=3)
    bands=[(1,4),(4,8),(8,13),(13,30),(30,50)]
    sims=[]
    for low,high in bands:
        sos=butter(4,[low/(hz/2),high/(hz/2)],btype='band',output='sos')
        sig=sosfiltfilt(sos,x,axis=0)
        sims.append(network.windowed_plv_median(sig,hz,window_s=4.,overlap=.5,time_axis=0))
    plv=np.mean(sims,axis=0)
    a=network.mst_proportional_graph(plv,density=.1)
    slope=network.centrality_composite(a,weights={'degree':.30,'betweenness':.60,'eigenvector':.10})
    score=np.array(slope['composite'])
    rank=np.argsort(-score,kind='stable')
    top=np.sort(rank[:case.actuators])
    np.savez_compressed(OUT/'graph.npz',channels=channels,part1_adjacency=a,part2_adjacency=graph,
                        selected_indices=top,ranked_indices=rank,composite=score,
                        degree=slope['degree'],betweenness=slope['betweenness'],eigenvector=slope['eigenvector'])
    pd.DataFrame({'channel_index':np.arange(case.channels),'channel':channels,'centrality':score,
                  'rank':np.argsort(rank)+1,'selected':np.isin(np.arange(case.channels),top)}).to_csv(OUT/'actuator_ranking.csv',index=False)
    pca=select_latent_dimension([x],variance_threshold=.95,minimum_components=8,maximum_components=24)[1]
    manifest=dict(source=source.name,source_sha256=digest(source),
                  training_runs=['run-01'],training_period_original_s=[120,152],signal_unit=meta['units'],
                  sampling_rate=hz,raw_sampling_rate=meta['source_sfreq'],preprocessing=meta['provenance'],
                  source_code={f.name:digest(f) for f in [Path(p2.__file__),Path(pipe.__file__),Path(network.__file__)]},
                  hyperparameter_rule='exact transfer of frozen HUP060 final hyperparameters, not a new patient-specific hyperparameter search',
                  inner_audit='train run01 [0,22)s; gap [22,24)s; validate [24,32)s; 6 deterministic HUP060 contexts; full 32s refit',
                  actual_frozen_config=asdict(cfg),full_pca=pca,training_pairs=model.training_pairs,
                  increment_training_nrmse=model.training_one_step_nrmse,increment_training_r=model.training_one_step_correlation,
                  channels=channels.tolist(),selected_indices=top.tolist(),selection=f'fixed budget {case.actuators}, HUP060 weighted composite ranking .30/.60/.10, stable ties by channel order',
                  lowest_selected_score=float(score[rank[case.actuators-1]]),highest_unselected_score=float(score[rank[case.actuators]]),
                  n_selected_if_literal_060_quantile=int(sum(score>np.quantile(score,.65))),
                  part1=dict(bands=bands,window_s=4.,overlap=.5,density=.1),
                  part2=dict(window_s=pipe.PLV_WINDOW_S,overlap=pipe.PLV_OVERLAP,density=pipe.NETWORK_DENSITY),
                  constant_C=.79451175,kappa_sigma=1.,plant_sha256=digest(OUT/'plant.joblib'),graph_sha256=digest(OUT/'graph.npz'))
    (OUT/'fit_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8')
    print('frozen',model.q,model.training_pairs,top,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
