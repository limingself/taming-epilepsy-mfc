"""Part III adapter for a fresh, retrospective, single-window Full controller.

The loss, optimizing loop and selection schedule are imported from the pinned
HUP060 implementation shipped in this repository. The bounded edits in
trainer() below handle a fixed-width projection, provenance, and a pilot clock;
none changes the full-run update law. No reference policy is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from zipfile import ZipFile

import joblib
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get("SWR_PROJECT", BASE.parents[1])).resolve()
sys.path.insert(0, str(PROJECT))
from mfc_pipeline import part2_state_dependent_rc_sde as p2

# This public Part II class was saved under this historical pickle module name.
sys.modules.setdefault("part2_state_dependent_base", p2)

HORIZON = 256
BOUNDARY = 3547
BASE_DIFFUSION = .79451175
EVAL_SEED = 20260813
CORE_PATH = PROJECT / "part3_mfc/model_core.py"
TRAIN_PATH = PROJECT / "part3_mfc/train_full_wgangp.py"
PIN = {CORE_PATH: "57da573dbe300ad6bd585fbd69e46102d9f315dba8862e63c7d3e8320b6943e1",
       TRAIN_PATH: "a6be4aa162ffda89a0893b64dd76f43fb63efc493fe28ba7e8abc216320bee84"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_pinned_module(path):
    # Git on Windows may materialize CRLF. Pin logical source, not EOL style.
    normalized = path.read_bytes().replace(b"\r\n", b"\n")
    require(hashlib.sha256(normalized).hexdigest() == PIN[path],
            "Canonical source changed; review the case adapter before running: " + str(path))
    name = "single_window_canonical_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def replace_one(code, old, new):
    require(code.count(old) == 1, "Unexpected canonical source around " + old)
    return code.replace(old, new)


def options(pilot=False):
    return argparse.Namespace(seed=20261011, epochs=12 if pilot else 1000,
                              actor_lr=3e-4, actor_min_lr=1e-5, critic_lr=1e-4,
                              n_critic=3, adv_weight=.5, validation_every=50)


def load_graph(c):
    with np.load(c.output / "dynamics" / "graph.npz", allow_pickle=False) as z:
        idx, names, raw_graph = z["selected_indices"], z["channels"].astype(str), z["part2_adjacency"]
    require(names.shape == (c.channels,) and len(set(names)) == c.channels, "Invalid contact axis")
    require(np.issubdtype(idx.dtype, np.integer) and idx.shape == (c.actuators,)
            and len(set(idx.tolist())) == c.actuators
            and np.all((0 <= idx) & (idx < c.channels)), "Invalid actuator axis")
    return idx, names, raw_graph


def segment_slice(path, channels, start, stop):
    """Materialize only the requested samples, including with compressed NPZ input.

    z["data"][:, start:stop] first constructs the *whole* signal array. Instead
    seek in its embedded NPY stream so the synthesis stage only receives the
    permitted prefix values. Both NPY storage orders use the same contact axis.
    """
    with np.load(path, allow_pickle=False) as z:
        require(np.array_equal(z["channel_names"].astype(str), channels), "Segment contact order changed")
    with ZipFile(path) as archive, archive.open("data.npy") as f:
        version = np.lib.format.read_magic(f)
        require(version in [(1, 0), (2, 0)], "Unsupported signal NPY format")
        header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                  else np.lib.format.read_array_header_2_0)
        shape, fortran, dtype = header(f)
        require(len(shape) == 2 and shape[0] == len(channels) and 0 <= start < stop <= shape[1]
                and dtype.kind in "fiu", "Invalid input shape, span or signal dtype")
        base, count, width = f.tell(), stop-start, dtype.itemsize
        if fortran:
            f.seek(base + start*shape[0]*width)
            x = np.frombuffer(f.read(count*shape[0]*width), dtype=dtype).reshape(count, shape[0])
        else:
            block = np.empty((shape[0], count), dtype=dtype)
            for i in range(shape[0]):
                f.seek(base + (i*shape[1]+start)*width)
                block[i] = np.frombuffer(f.read(count*width), dtype=dtype)
            x = block.T
        x = x.astype(np.float64)
    require(x.shape == (stop-start, len(channels)) and np.isfinite(x).all(), "Invalid segment span")
    return x


def reference(c, model, names):
    # Cache begins at EDF t=30 s. Fifteen one-second windows fit, fifteen validate.
    x = segment_slice(c.run01_reference, names, 0, 30 * HORIZON)
    return model.transform.scaler.transform(x).reshape(30, HORIZON, c.channels)


def eval_bank(q):
    primary = torch.randn(1, 16, HORIZON, q, dtype=torch.float32, device="cpu",
                          generator=torch.Generator(device="cpu").manual_seed(EVAL_SEED))[0]
    return torch.cat([primary, -primary]).double().numpy()


def trainer(c, model_path, kappa, alpha, contract_path, *, pilot=False):
    m = load_pinned_module(CORE_PATH)
    src = load_pinned_module(TRAIN_PATH)
    r = replace_one(inspect.getsource(m.reference_statistics), "reshape(-1, 36)",
                    "reshape(-1, reference.shape[-1])")
    exec(compile(r, str(CORE_PATH), "exec"), m.__dict__)
    m.MODEL_PATH, m.SYNTHESIS_CONTRACT = model_path, contract_path
    m.DIFFUSION_SCALE, m.CONTROL_STEP_SCALE = BASE_DIFFUSION * kappa, alpha
    src.ADAPTER_SOURCE = Path(__file__).resolve()
    src.CONTROL_SUBJECT = c.subject
    src.BINDINGS = c.output / "control" / "binding.json"
    s = inspect.getsource(src.train)
    s = replace_one(s, "rng.normal(size=(36, 16))", "rng.normal(size=(fit.shape[-1], 16))")
    s = replace_one(s, "(m.MODEL_PATH, m.ORIGINAL_ACTOR, m.ORIGINAL_ROLLOUT, SOURCE)",
                    "(m.MODEL_PATH, m.SYNTHESIS_CONTRACT, SOURCE, ADAPTER_SOURCE)")
    if pilot:
        s = replace_one(s, "T_max=args.epochs", "T_max=1000")
        flag = '"run02_access_during_training": False, "run01_pilot_alpha": m.CONTROL_STEP_SCALE'
    else:
        flag = ('"run02_past_context_used": True, "run02_forecast_outcome_used": False, '
                '"subject": CONTROL_SUBJECT, "control_step_scale": m.CONTROL_STEP_SCALE, '
                '"frozen_bindings_sha256": sha(BINDINGS)')
    s = replace_one(s, '"run02_access_during_training": False', flag)
    exec(compile(s, str(TRAIN_PATH), "exec"), src.__dict__)

    def build(h, ref, sel):
        require(h.shape == (HORIZON, c.channels) and ref.shape == (30, HORIZON, c.channels),
                "Invalid history/reference axes")
        require(len(sel) == len(np.unique(sel)) == c.actuators
                and ((sel >= 0) & (sel < c.channels)).all(), "Invalid mask")
        model = joblib.load(model_path)
        fit, val = [torch.as_tensor(a, dtype=torch.float64) for a in (ref[:15], ref[15:])]
        world = m.TorchGraphRCSDE(model, sel, m.FS, control_graph_diffusion_time=m.GRAPH_DIFFUSION_TIME,
                                  preserve_physical_control_residual=True, dtype=torch.float64, device="cpu")
        adapter = m.FrozenGraphRCMarkovAdapter(world, control_step_scale=alpha, dtype=torch.float64)
        initial = adapter.initial_state_from_context(h)  # preprocessed physical units; adapter scales internally
        stepper = m.FrozenIctalGraphRCBatchStepper(world, adapter, diffusion_scale=m.DIFFUSION_SCALE)
        mean, var, scale, weights = m.reference_statistics(fit)
        center, state_scale = m.build_markov_normalization(adapter, initial, fit)
        actor = m.StructuredSplineCovarianceActor(
            stepper, mean, var, scale, torch.zeros(c.actuators, c.channels, dtype=torch.float64), sel,
            horizon=256, basis_count=8, hidden_size=64, amplitude_limit=1.8, actuator_alpha=1.0,
            residual_scale=.15, local_gain_initial_fraction=0., local_gain_maximum_fraction=.05,
            markov_feature_center=center, markov_feature_scale=state_scale, markov_residual_scale=.6,
            markov_hidden_size=96, maximum_slew=None)
        critic = m.TimeConditionedWassersteinCritic(fit.reshape(-1, c.channels).mean(dim=0), scale,
                                                   horizon_samples=256, hidden_size=128)
        return None, fit, val, sel, adapter, initial, stepper, scale, weights, actor, critic, sha(model_path)

    def setup(_m):
        with np.load(contract_path, allow_pickle=False) as z:
            keys = {"history_raw", "reference_scaled", "selected_indices", "channels", "model_sha256"}
            require(set(z.files) == keys and str(z["model_sha256"].item()) == sha(model_path),
                    "Invalid synthesis-only contract or changed plant")
            return build(z["history_raw"], z["reference_scaled"], z["selected_indices"])

    src.setup, src.build_for_history = setup, build
    return m, src
