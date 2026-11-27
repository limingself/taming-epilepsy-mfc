"""Opt-in integration check of real local --through noise results (no samples bundled)."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay_config import PATIENTS, WINDOWS


EXPECTED = {
    "HUP065": {
        "signal_sha": ["be69a10be0ded2bf8f1fa715b78f94e61200b8e959320a646d32dd720c228b7c",
                       "0dfc5fccff0aab09afcb036bddfdf85ed78f69fb727fef86fedfa86d78452b7f",
                       "1e1b448ae6dd86b7766b6c5dba1483a2133131df5d323e1416608fea375e24f8"],
        "rank_sha": "1645e8624ac6dc7b0438ced68e8df4a73baef873719e0bbd59fab8d0cbdc5877",
        "q": 19, "inner_q": 17, "variance": .9532311123928238,
        "kappa_scores": [1.122803424489297, 1.0521867979295252, .9717141342969607, .8845334265624505],
    },
    "HUP080": {
        "signal_sha": ["7865da0139dcde4939d1fdf4e4016d7b8168fde4a54ded7b39ea3720cbd6f9b7",
                       "f26b308431f0a7abd47e26006370b30bd972c7e162cbe6bbd10dd818506ea194",
                       "071201fae91f9e7a34fdfb31b7646d7a8ab7d9e2fe3185c38106790bf4ccaebe"],
        "rank_sha": "b0f7d65375327b615c79d45a1078bab8559a418582837f6aa170b343d5b17dc1",
        "q": 24, "inner_q": 24, "variance": .7054491926123618,
        "kappa_scores": [.9515594394727195, .8902065382469936, .822368231067084, .7522738263229645],
    },
}


def check(subject, output, mapping_path):
    e = EXPECTED[subject]
    m = json.loads(mapping_path.read_text(encoding="utf-8"))
    if m["subject"] != subject:
        raise ValueError("input map and subject disagree")
    for role, digest in zip(WINDOWS, e["signal_sha"], strict=True):
        with np.load(mapping_path.parent / m[role], allow_pickle=False) as z:
            actual = sha256(z["data"].astype("<f4", copy=False).tobytes()).hexdigest()
        if digest != actual:
            raise ValueError(f"{role}: float32 sample digest does not match the archived crop")
    dyn = output / "dynamics"
    fit = json.loads((dyn / "fit_manifest.json").read_text(encoding="utf-8"))
    noise = json.loads((dyn / "noise_selection.json").read_text(encoding="utf-8"))
    np.testing.assert_equal(fit["training_runs"], ["run-01"])
    np.testing.assert_equal(fit["training_pairs"], 8127)
    np.testing.assert_equal(fit["full_pca"]["latent_components"], e["q"])
    np.testing.assert_allclose(fit["full_pca"]["explained_variance_at_q"], e["variance"], atol=1e-8)
    np.testing.assert_equal(noise["selected_kappa"], 1.25)
    np.testing.assert_equal(noise["inner_q"], e["inner_q"])
    np.testing.assert_allclose([noise["compared"][str(k)] for k in (.5, .75, 1., 1.25)],
                               e["kappa_scores"], atol=1e-6, rtol=1e-6)
    with np.load(dyn / "graph.npz", allow_pickle=False) as graph:
        rank = graph["ranked_indices"]
        np.testing.assert_array_equal(graph["selected_indices"], np.sort(rank[:PATIENTS[subject][1]]))
        if sha256(rank.astype("<i8").tobytes()).hexdigest() != e["rank_sha"]:
            raise ValueError("actuator ranking differs from the archived fit")
    print(f"{subject}: exact sample digests and rankings; PCA, fit size and all four scale scores passed.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", required=True, choices=PATIENTS)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--input-map", required=True, type=Path)
    args = p.parse_args()
    check(args.subject, args.output, args.input_map)
