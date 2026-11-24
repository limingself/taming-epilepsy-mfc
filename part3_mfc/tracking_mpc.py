"""Four-step nominal tracking MPC: HUP060 primary comparison, manuscript Fig. 7.

Evaluation only. Uses the existing Part III plant and frozen Full policy.
The optimizer does not get future stochastic innovations or validation paths.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .tracking_mpc_metrics import score_bank
else:
    from tracking_mpc_metrics import score_bank

H, CAP, ENERGY, SLEW = 4, 1.8, 0.01, 0.02
OUTER, INNER = 3, 40
CHECKPOINT_SHA256 = "980952488c536b76cfc51fa5d979243e0a69be4cba1cbf36aacdc094e7b29ab0"


class TrackingMPC:
    """Bounded per-particle tracker on the nonlinear frozen stochastic stepper."""

    def __init__(self, plant, initial, target, weights):
        self.plant, self.target, self.weights = plant, target, weights
        self.Q, self.L = {}, {}
        # Initial-state finite differences provide optimizer curvature only;
        # every objective evaluation and gradient uses the nonlinear plant.
        for h in range(1, min(H, len(target)) + 1):
            r = plant.actuator_dim
            n = h * r
            eye = torch.eye(n, dtype=torch.float64)
            du = eye.reshape(n, h, r) * 1e-4
            state = plant.repeat_initial(initial, 2*n)
            y = self.predict(state, torch.cat([du, -du], dim=0))
            j = (y[:n]-y[n:]) / (2e-4)
            jw = (j * torch.sqrt(weights[None, None, :])).reshape(n, -1)
            d = torch.eye(h, dtype=torch.float64) - torch.diag(
                torch.ones(h-1, dtype=torch.float64), diagonal=-1)
            dtd = torch.kron(d.T @ d, torch.eye(r, dtype=torch.float64))
            q = 2*(jw @ jw.T / (target.shape[1]*h)
                   + (ENERGY*eye + SLEW*dtd) / (h*r*CAP**2))
            self.Q[h] = q
            self.L[h] = torch.linalg.eigvalsh(q)[-1].item()

    def predict(self, state, plan):
        zero = state.new_zeros(len(state), self.plant.q)
        ys = []
        for command in plan.unbind(1):
            p = self.plant.step(state, command, zero)
            state = p.next_markov
            ys.append(p.next_scaled)
        return torch.stack(ys, dim=1)

    def cost(self, state, plan, prev, t):
        y = self.predict(state, plan)
        du = torch.cat([plan[:, :1]-prev[:, None, :], plan[:, 1:]-plan[:, :-1]], dim=1)
        return (((y-self.target[None, t:t+plan.shape[1], :]).square()*self.weights).mean((1, 2))
                + ENERGY*(plan/CAP).square().mean((1, 2))
                + SLEW*(du/CAP).square().mean((1, 2)))

    def value_gradient(self, state, plan, prev, t):
        with torch.enable_grad():
            p = plan.detach().requires_grad_(True)
            c = self.cost(state, p, prev, t)
            grad = torch.autograd.grad(c.sum(), p)[0].detach()
        return c.detach(), grad

    def solve(self, state, t, prev, plan):
        h = plan.shape[1]
        lip, q = self.L[h], self.Q[h]
        backs = 0
        for i in range(OUTER):
            c, g = self.value_gradient(state, plan, prev, t)
            if i == 0:
                initial_cost = c
            anchor = plan.flatten(1)
            v = z = anchor
            tau = 1.
            for _ in range(INNER):
                nz = torch.clamp(v - (g.flatten(1)+(v-anchor) @ q)/lip, -CAP, CAP)
                nt = .5*(1.+(1.+4*tau*tau)**.5)
                v = nz+(tau-1)/nt*(nz-z)
                z, tau = nz, nt
            # Keep arithmetic order of the archived solver.
            step = z.reshape_as(plan) - plan
            trial = plan + step
            with torch.no_grad():
                cost = self.cost(state, trial, prev, t)
                for _ in range(9):
                    bad = cost > c+1e-14
                    if not torch.any(bad):
                        break
                    backs += int(bad.sum())
                    trial = torch.where(bad[:, None, None], (trial+plan)*.5, trial)
                    cost = self.cost(state, trial, prev, t)
                plan = torch.where((cost<=c+1e-14)[:, None, None], trial, plan)
        final_cost, grad = self.value_gradient(state, plan, prev, t)
        resid = (lip*(plan-torch.clamp(plan-grad/lip, -CAP, CAP))).abs().amax((1, 2))
        return plan, initial_cost.numpy(), final_cost.numpy(), resid.numpy(), backs

    def run(self, initial, noise):
        if noise.shape[1] != len(self.target):
            raise ValueError("Reference-mean trajectory and replay horizon must agree.")
        state = self.plant.repeat_initial(initial, len(noise))
        prev = state.new_zeros(len(state), self.plant.actuator_dim)
        plan = prev[:, None, :].repeat(1, H, 1)
        xs, us, checks, nback = [], [], [], 0
        for t in range(len(self.target)):
            h = min(H, len(self.target)-t)
            plan, c0, c1, residual, n = self.solve(state, t, prev, plan[:, :h].contiguous())
            if (not torch.isfinite(plan).all() or plan.abs().max() > CAP+1e-12
                    or not np.all(c1 <= c0+1e-11)):
                raise RuntimeError("Nonfinite, unbounded, or cost-increasing MPC update")
            checks.append(np.stack([c0, c1, residual], axis=1))
            nback += n
            prev = plan[:, 0]
            plan = torch.cat([plan[:, 1:], plan[:, -1:]], dim=1)
            with torch.no_grad():
                tr = self.plant.step(state, prev, noise[:, t])
                state = tr.next_markov
                xs.append(tr.next_scaled)
                us.append(prev)
        return torch.stack(xs, 1).numpy(), torch.stack(us, 1).numpy(), np.asarray(checks), nback


def replay(root):
    """Replay the primary development window with the original Full checkpoint."""
    runner_path = root / "part3_mfc" / "train_full_wgangp.py"
    if not runner_path.is_file():
        raise FileNotFoundError("The requested project root lacks part3_mfc/train_full_wgangp.py")
    spec = importlib.util.spec_from_file_location("fig7_full_training", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    checkpoint = runner.HERE / "runs/fresh_seed20261011_e1000/frozen_actor_wgan.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError("Replay requires the existing private Full bundle; see TRACKING_MPC.md.")
    if runner.sha(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Not the frozen HUP060 Fig. 7 Full checkpoint.")
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    m = runner.source_module()
    contract, fit, val, selected, _, initial, plant, _, _, actor, _, model_hash = runner.setup(m)
    contract.close()
    if ck["best_epoch"] != 700 or model_hash != ck["model_sha256"]:
        raise ValueError("Wrong checkpoint iteration or stochastic plant.")
    actor.load_state_dict(ck["actor_state_dict"], strict=True)
    actor.eval()
    m.set_requires_grad(actor, False)
    # The archived MPC computes target moments on a contiguous copy of the fit
    # split. Keep the same reduction order for exact CPU replay.
    target, _, _, weights = m.reference_statistics(fit.contiguous())
    policy = TrackingMPC(plant, initial, target, weights)
    # Open the paired evaluation bundle only after both policies are fixed.
    with np.load(runner.PAPER_PAIRED, allow_pickle=False) as b:
        if not np.array_equal(b["reference_validation_scaled"], val.numpy()) or not np.array_equal(
                b["selected_indices"], selected):
            raise ValueError("Evaluation mask or reference differs from the frozen contract.")
        archived_free = b["uncontrolled_scaled"].copy()
    eps, _ = m.reconstruct_paired_normals(plant, initial, archived_free, tolerance=2e-6)
    with torch.inference_mode():
        full = m.empirical_fp_rollout(plant, actor, initial, eps)
        free = m.uncontrolled_particle_rollout(plant, initial, eps)[:, 1:].numpy()
    if np.max(np.abs(free-archived_free)) > 2e-6:
        raise ValueError("Paired no-input rollout disagrees with the archived primary bank.")
    x, u, checks, backs = policy.run(initial, eps)
    return dict(free=free, full=full.scaled[:, 1:].numpy(), full_u=full.controls.numpy(),
                mpc=x, mpc_u=u, solver_checks=checks, ref=val.numpy()), backs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1],
                   help="Existing Part III project, including the trusted frozen-model assets.")
    p.add_argument("--check-against", type=Path, help="Optional authorized primary trajectory NPZ.")
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    bank, backs = replay(args.project_root.resolve())
    out = score_bank(bank)
    out["outer_backtracking_count"] = backs
    if args.check_against is not None:
        with np.load(args.check_against, allow_pickle=False) as old:
            errors = {}
            for key, value in bank.items():
                historical = old[key]
                if historical.shape != value.shape:
                    raise ValueError(f"Unexpected {key} shape.")
                errors[key] = float(np.max(np.abs(value-historical)))
            if not all(v < 1e-10 for v in errors.values()):
                raise ValueError(f"Frozen replay discrepancy: {errors}")
            out["max_absolute_replay_errors"] = errors
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
