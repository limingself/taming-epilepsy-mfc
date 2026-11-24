"""Small public tests with an artificial scalar plant; no participant data."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

if __package__:
    from .tracking_mpc import CAP, TrackingMPC
    from .tracking_mpc_metrics import contact_metrics
else:
    from tracking_mpc import CAP, TrackingMPC
    from tracking_mpc_metrics import contact_metrics


class ScalarPlant:
    q = actuator_dim = 1

    def repeat_initial(self, initial, n):
        return initial.reshape(1, 1).repeat(n, 1)

    def step(self, state, control, noise):
        next_state = .8*state + control + noise
        return SimpleNamespace(next_markov=next_state, next_scaled=next_state)


class TrackingTest(unittest.TestCase):
    def test_limits_nonanticipation_and_objective(self):
        T = 7
        zero = torch.zeros(1, dtype=torch.float64)
        target = torch.full((T, 1), .5, dtype=torch.float64)
        policy = TrackingMPC(ScalarPlant(), zero, target, torch.ones_like(zero))
        noise = torch.zeros((2, T, 1), dtype=torch.float64)
        # Particles receive different perturbations only when the fourth command
        # has already been chosen. A look-ahead tracker would generally react earlier.
        noise[1, 3] = .7
        x, u, checks, _ = policy.run(zero, noise)
        np.testing.assert_allclose(u[0, :4], u[1, :4], rtol=0, atol=1e-12)
        self.assertNotAlmostEqual(u[0, 4, 0], u[1, 4, 0], places=5)
        self.assertLessEqual(np.max(np.abs(u)), CAP+1e-12)
        self.assertTrue(np.all(checks[:, :, 1] <= checks[:, :, 0]+1e-11))
        self.assertLess(np.square(x[0]-.5).mean(), .5**2)

    def test_distances_do_not_confuse_time_order_with_pooled_spread(self):
        # Swapping two time bins leaves the pooled distribution exactly intact.
        ref = np.array([[[0.], [2.]], [[2.], [4.]]])
        x = ref[:, ::-1, :]
        time, occ, sd, center = contact_metrics(x, ref)
        np.testing.assert_allclose([time[0], occ[0], sd[0], center[0]], [2., 0., 0., 0.])


if __name__ == "__main__":
    unittest.main()
