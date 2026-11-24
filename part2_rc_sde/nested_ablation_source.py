"""Export Fig. 3 Source Data from private paired predictions, or audit the public tables."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import gaussian_kde, wasserstein_distance
from threadpoolctl import threadpool_limits

ARMS = ("state","delay","graph","full")
CHANNELS = ("RPFa3","RA3","RAFa4")
METRICS = "fixed_context_ablation_metrics.csv"
DENSITY = "fixed_context_ablation_density_curves.csv"
SUMMARY = "unified_ablation_summary.json"
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/part2/source_data/figure_05"
ROLLOUT = ROOT / "artifacts/part2_hup060_fixed_free_input_ablation/nested_runs.npz"
def digest(p):
    # CSV writer uses CRLF. Canonicalize line endings only so Git LF checkouts
    # can verify identical numeric Source Data without any rounding.
    data=p.read_bytes().replace(b"\r\n",b"\n").replace(b"\n",b"\r\n")
    return hashlib.sha256(data).hexdigest()

def validate(out):
    audit = json.loads((out / SUMMARY).read_text(encoding="utf-8"))
    for name in (METRICS,DENSITY):
        if digest(out/name) != audit["source_sha256"][name]:
            raise ValueError(f"Source Data checksum mismatch: {name}")
    with (out / METRICS).open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=36 or len({r["channel"] for r in rows})!=36:
        raise ValueError("Expected 36 unique contacts")
    vals=np.array([[float(row[k]) for k in ARMS] for row in rows])
    np.testing.assert_allclose(vals.mean(0),[.177761414905,.144331757228,.158009792705,.109965067826],atol=1e-9,rtol=0)
    np.testing.assert_array_equal((np.diff(vals,axis=1)<0).sum(0),[31,3,27])
    return audit

def build(rollouts,out,parity_reference=None):
    with np.load(rollouts,allow_pickle=False) as z:
        contacts=z["channels"].astype(str).tolist()
        obs=z["observed"]; pred={k:z[k] for k in ARMS}
        if z["positions"].tolist()!=[3547] or obs.shape!=(256,36) or len(set(contacts))!=36 or not np.isfinite(obs).all():
            raise ValueError("Incorrect context or observation shape")
        if any(a.shape!=(32,256,36) or not np.isfinite(a).all() for a in pred.values()):
            raise ValueError("Incorrect prediction ensemble")
    out.mkdir(parents=True,exist_ok=True)
    # Use a second independently archived Part-III zero-input evaluation as parity reference.
    # With only the public Source Data, use validate() instead of rebuilding.
    if parity_reference is None: raise ValueError("Provide the original paired-comparison NPZ for the full-arm check.")
    with np.load(parity_reference,allow_pickle=False) as z:
        np.testing.assert_array_equal(z["channels"].astype(str),contacts)
        np.testing.assert_array_equal(obs,z["observed_scaled"])
        full_error=float(np.max(np.abs(pred["full"]-z["uncontrolled_scaled"])))
        if full_error>1e-9: raise ValueError(f"Full/archived mismatch: {full_error:g}")
    scores={k:[wasserstein_distance(obs[:,j],v[:,:,j].ravel()) for j in range(len(contacts))] for k,v in pred.items()}
    with (out / METRICS).open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); w.writerow(("channel",*ARMS)); w.writerows(zip(contacts,*scores.values()))
    with (out / DENSITY).open("w",encoding="utf-8",newline="") as f, threadpool_limits(limits=1):
        w=csv.writer(f); w.writerow(("channel","amplitude","observed",*ARMS))
        for channel in CHANNELS:
            j=contacts.index(channel)
            samples=[obs[:,j],*(pred[k][:,:,j].ravel() for k in ARMS)]
            lo,hi=np.quantile(np.concatenate(samples),[.005,.995]); pad=.08*max(hi-lo,.1)
            grid=np.linspace(lo-pad,hi+pad,450)
            lines=[gaussian_kde(v)(grid) for v in samples]
            w.writerows((channel,*r) for r in zip(grid,*lines))
    audit={"subject":"HUP060","split":"run-02 development","context_index":3,
           "forecast_boundary_sample":3547,"analysis_sampling_rate_hz":256,"horizon_samples":256,
           "particles":32,"future_observation_refresh":False,"effective_diffusion_multiplier":.79451175,
           "training":"run-01, 32 s; identical 8127 one-step pairs; original PCA dimension 13",
           "arms":dict(zip(ARMS,["current state","current state + delays (1,8,32)",
                      "state + delays + graph","state + delays + graph + 96-unit reservoir"])),
           "trainable_components":"Drift and conditional diffusion are refitted for each of the three zero-reservoir arms; full uses the original frozen model.",
           "metric":"Empirical occupation W1 between recorded and predicted standardized amplitudes. KDE for display only.",
           "mean_occupation_w1":{k:float(np.mean(v)) for k,v in scores.items()},
           "n_improved_vs_previous":{b:int(np.sum(np.array(scores[b])<scores[a])) for a,b in zip(ARMS,ARMS[1:])},
           "full_saved_rollout_parity_max_abs":full_error,
           "source_sha256":{n:digest(out/n) for n in (METRICS,DENSITY)}}
    (out / SUMMARY).write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    return validate(out)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollouts",type=Path)
    p.add_argument("--paired-reference",type=Path)
    p.add_argument("--out",type=Path,default=SOURCE)
    a=p.parse_args()
    print(json.dumps(build(a.rollouts,a.out,a.paired_reference) if a.rollouts else validate(a.out),indent=2))
