"""Run with: python -m unittest discover -s patient_extensions/single_window_replay/tests"""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.stats import wasserstein_distance

spec = importlib.util.spec_from_file_location("replay_renderer", Path(__file__).resolve().parents[1]/"render.py")
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


class Distances(unittest.TestCase):
    def test_unequal_particle_counts_match_independent_scipy(self):
        rng = np.random.default_rng(23)
        a, b = rng.normal(size=(7, 11, 4)), rng.normal(size=(3, 11, 4))
        expected = [np.mean([wasserstein_distance(a[:,t,c], b[:,t,c]) for t in range(11)])
                    for c in range(4)]
        np.testing.assert_allclose(renderer.time_w1(a, b), expected, rtol=1e-13)

    def test_occupation_is_time_order_invariant_but_time_distance_is_not(self):
        x = np.array([[[0., 5.], [4., 8.]]])
        y = np.repeat(x[:, ::-1, :], 3, axis=0)
        np.testing.assert_allclose(renderer.occupation_w1(x, y), (0., 0.))
        np.testing.assert_allclose(renderer.time_w1(x, y), (4., 3.))

    def test_fidelity_and_reference_distance_have_different_targets(self):
        observed = np.arange(8.).reshape(4, 2)
        free = np.stack((observed+1,observed+1)); reference = (observed+4)[None]
        data = dict(observed=observed,past=observed,free=free,controlled=reference,
                    reference=reference,channels=["A","B"],selected_indices=[0],controls=np.zeros((2,4,1)))
        _, total = renderer.scores(data)
        self.assertEqual(total["forecast_W1"],1)
        ref = total["reference_distances"]
        for k in ("time_W1","occupation_W1"):
            self.assertAlmostEqual(ref[f"observed_{k}_to_ref"],4)
            self.assertAlmostEqual(ref[f"free_{k}_to_ref"],3)
            self.assertEqual(ref[f"controlled_{k}_to_ref"],0)
            self.assertEqual(total[k]["direct_improved"],1)
            self.assertEqual(total[k]["indirect_improved"],1)


if __name__ == "__main__":
    unittest.main()
