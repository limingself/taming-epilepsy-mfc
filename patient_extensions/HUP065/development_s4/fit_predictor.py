"""Reproduce the fold-1 Part II surrogate used in HUP065 Figure S4 (run02 fit only)."""
import argparse
import hashlib
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--arrays", type=Path, required=True, help="Restricted development_arrays.npz, containing run02")
    parser.add_argument("--graph", type=Path, required=True, help="Trusted archived fold-1 graph model")
    parser.add_argument("--out", type=Path, required=True, help="New private output joblib; existing files are not overwritten")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out already exists")
    sys.path.insert(0, str(args.repo.resolve()))
    import joblib
    import numpy as np
    from threadpoolctl import threadpool_limits
    from mfc_pipeline import part2_state_dependent_rc_sde as p2
    sys.modules.setdefault("part2_state_dependent_base", p2)

    with args.graph.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != "a55366d14a55e89ef4993cd46f5381e4468997e9d7c8535fe51709cb37542ed4":
            raise ValueError("fold-1 graph source does not match the archived run")
    with np.load(args.arrays, allow_pickle=False) as arrays:
        run02 = arrays["ictal_run_02"]
        assert run02.shape == (8192, 64) and float(arrays["sfreq"]) == 256
    with threadpool_limits(limits=1):
        graph = joblib.load(args.graph)
        model = p2.ResidualGraphRCSDE(graph.config, np.zeros_like(graph.adjacency_input)).fit([run02])
        assert model.training_pairs == graph.training_pairs and model.q == graph.q == 35
        assert np.count_nonzero(model.adjacency) == 0
        for x, y in [(model.transform.components_, graph.transform.components_),
                     (model.transform.scaler.scale_, graph.transform.scaler.scale_),
                     (model.w_res, graph.w_res), (model.w_in, graph.w_in)]:
            np.testing.assert_allclose(x, y, rtol=1e-11, atol=1e-13)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out, compress=3)


if __name__ == "__main__":
    main()
