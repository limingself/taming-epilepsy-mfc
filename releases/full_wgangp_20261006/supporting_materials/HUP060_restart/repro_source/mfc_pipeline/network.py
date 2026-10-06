"""Functional-network construction and actuator-target utilities.

The functions in this module deliberately keep the biological unit out of the
graph calculation: a graph is constructed within one leakage-safe data split,
and patient-level aggregation happens later in :mod:`mfc_pipeline.statistics`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import ceil
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import hilbert
from scipy.stats import hypergeom


DEFAULT_CENTRALITY_WEIGHTS: dict[str, float] = {
    "degree": 0.3,
    "betweenness": 0.6,
    "eigenvector": 0.1,
}


def _time_by_channel(signals: ArrayLike, time_axis: int = 0) -> NDArray[np.float64]:
    """Return a finite, two-dimensional ``(time, channel)`` array."""

    values = np.asarray(signals, dtype=float)
    if values.ndim != 2:
        raise ValueError("signals must be a two-dimensional array")
    values = np.moveaxis(values, time_axis, 0)
    if values.shape[0] < 2:
        raise ValueError("at least two time samples are required")
    if values.shape[1] < 1:
        raise ValueError("at least one channel is required")
    if not np.all(np.isfinite(values)):
        raise ValueError("signals contain NaN or infinite values")
    return values


def plv_matrix(
    signals: ArrayLike,
    *,
    time_axis: int = 0,
    input_is_phase: bool = False,
) -> NDArray[np.float64]:
    """Compute the all-to-all phase-locking-value (PLV) matrix.

    Parameters
    ----------
    signals:
        A time-by-channel array (or an array with ``time_axis`` specified).
        Filtering should be performed before this function is called.
    input_is_phase:
        If true, values are interpreted as instantaneous phase in radians;
        otherwise phase is obtained using the Hilbert transform.

    Returns
    -------
    numpy.ndarray
        Symmetric channel-by-channel PLV matrix with a unit diagonal.
    """

    values = _time_by_channel(signals, time_axis=time_axis)
    phase = values if input_is_phase else np.angle(hilbert(values, axis=0))
    unit_phase = np.exp(1j * phase)
    plv = np.abs(unit_phase.conj().T @ unit_phase) / values.shape[0]
    plv = np.clip((plv + plv.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(plv, 1.0)
    return plv.astype(float, copy=False)


def windowed_plv(
    signals: ArrayLike,
    sfreq: float,
    *,
    window_s: float = 2.0,
    overlap: float = 0.5,
    time_axis: int = 0,
    input_is_phase: bool = False,
) -> NDArray[np.float64]:
    """Compute one PLV matrix per complete, overlapping time window."""

    values = _time_by_channel(signals, time_axis=time_axis)
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq must be positive")
    if not np.isfinite(window_s) or window_s <= 0:
        raise ValueError("window_s must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")

    window_samples = int(round(window_s * sfreq))
    if window_samples < 2:
        raise ValueError("window_s corresponds to fewer than two samples")
    if values.shape[0] < window_samples:
        raise ValueError(
            f"recording has {values.shape[0]} samples but a window needs "
            f"{window_samples}"
        )
    step = max(1, int(round(window_samples * (1.0 - overlap))))
    # Obtain instantaneous phase once on the complete segment. Applying a
    # separate Hilbert transform to every window would create artificial phase
    # discontinuities at each window boundary.
    phase = values if input_is_phase else np.angle(hilbert(values, axis=0))
    starts = range(0, values.shape[0] - window_samples + 1, step)
    matrices = [
        plv_matrix(
            phase[start : start + window_samples],
            input_is_phase=True,
        )
        for start in starts
    ]
    return np.stack(matrices, axis=0)


def windowed_plv_median(
    signals: ArrayLike,
    sfreq: float,
    *,
    window_s: float = 2.0,
    overlap: float = 0.5,
    time_axis: int = 0,
    input_is_phase: bool = False,
    return_windows: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the element-wise median PLV across complete windows.

    Median aggregation reduces the leverage of brief high-synchrony artifacts.
    Set ``return_windows=True`` when window-level stability or sensitivity is
    needed; the return value is then ``(median_plv, window_plvs)``.
    """

    matrices = windowed_plv(
        signals,
        sfreq,
        window_s=window_s,
        overlap=overlap,
        time_axis=time_axis,
        input_is_phase=input_is_phase,
    )
    median = np.median(matrices, axis=0)
    median = (median + median.T) / 2.0
    np.fill_diagonal(median, 1.0)
    return (median, matrices) if return_windows else median


def _similarity_matrix(similarity: ArrayLike) -> NDArray[np.float64]:
    weights = np.asarray(similarity, dtype=float)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("similarity must be a square matrix")
    if weights.shape[0] < 2:
        raise ValueError("at least two nodes are required")
    if not np.all(np.isfinite(weights)):
        raise ValueError("similarity contains NaN or infinite values")
    if np.any(weights < 0):
        raise ValueError("similarity weights must be non-negative")
    weights = (weights + weights.T) / 2.0
    np.fill_diagonal(weights, 0.0)
    return weights


class _DisjointSet:
    def __init__(self, n_nodes: int) -> None:
        self.parent = list(range(n_nodes))
        self.rank = [0] * n_nodes

    def find(self, node: int) -> int:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left: int, right: int) -> bool:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return False
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1
        return True


def mst_proportional_graph(
    similarity: ArrayLike,
    density: float = 0.10,
) -> NDArray[np.float64]:
    """Build a connected weighted graph at the requested proportional density.

    A deterministic maximum spanning tree is selected first. The strongest
    remaining edges are then added until the graph has
    ``max(n - 1, ceil(density * n * (n - 1) / 2))`` undirected edges. Thus a
    requested density below the minimum connected density is transparently
    raised to that minimum.
    """

    weights = _similarity_matrix(similarity)
    if not np.isfinite(density) or not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")

    n_nodes = weights.shape[0]
    possible_edges = n_nodes * (n_nodes - 1) // 2
    target_edges = min(
        possible_edges,
        max(n_nodes - 1, int(ceil(density * possible_edges))),
    )
    edges = [
        (float(weights[left, right]), left, right)
        for left in range(n_nodes)
        for right in range(left + 1, n_nodes)
    ]
    # Stable tie breaking makes sensitivity analyses reproducible.
    edges.sort(key=lambda item: (-item[0], item[1], item[2]))

    selected: set[tuple[int, int]] = set()
    components = _DisjointSet(n_nodes)
    for _, left, right in edges:
        if components.union(left, right):
            selected.add((left, right))
            if len(selected) == n_nodes - 1:
                break
    for _, left, right in edges:
        if len(selected) >= target_edges:
            break
        selected.add((left, right))

    adjacency = np.zeros_like(weights)
    tiny = np.finfo(float).eps
    for left, right in selected:
        # Retain a graph edge even in the degenerate case of zero similarity.
        edge_weight = max(float(weights[left, right]), tiny)
        adjacency[left, right] = adjacency[right, left] = edge_weight
    return adjacency


# Explicit name used by some pipeline stages.
build_mst_proportional_graph = mst_proportional_graph


def _minmax(values: NDArray[np.float64]) -> NDArray[np.float64]:
    low, high = float(np.min(values)), float(np.max(values))
    if np.isclose(low, high):
        return np.zeros_like(values)
    return (values - low) / (high - low)


def centrality_composite(
    adjacency: ArrayLike,
    *,
    weights: Mapping[str, float] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute normalized centralities and their pre-specified composite.

    Weighted degree uses edge strength. Weighted betweenness treats inverse
    strength as path length. Eigenvector centrality is obtained from the leading
    eigenvector of the symmetric weighted adjacency. Each component is min-max
    normalized across electrodes before applying the requested weights.
    """

    matrix = _similarity_matrix(adjacency)
    n_nodes = matrix.shape[0]
    graph = nx.from_numpy_array(matrix)
    if not nx.is_connected(graph):
        raise ValueError("adjacency must describe a connected graph")

    degree_raw = matrix.sum(axis=1)
    for left, right, attributes in graph.edges(data=True):
        attributes["distance"] = 1.0 / max(float(matrix[left, right]), 1e-12)
    betweenness_raw_dict = nx.betweenness_centrality(
        graph, normalized=True, weight="distance"
    )
    betweenness_raw = np.array(
        [betweenness_raw_dict[index] for index in range(n_nodes)], dtype=float
    )

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvector_raw = np.abs(eigenvectors[:, int(np.argmax(eigenvalues))])

    normalized = {
        "degree": _minmax(degree_raw),
        "betweenness": _minmax(betweenness_raw),
        "eigenvector": _minmax(eigenvector_raw),
    }
    requested = dict(DEFAULT_CENTRALITY_WEIGHTS if weights is None else weights)
    if set(requested) != set(normalized):
        raise ValueError(
            "weights must contain exactly degree, betweenness, and eigenvector"
        )
    if any(not np.isfinite(value) or value < 0 for value in requested.values()):
        raise ValueError("centrality weights must be finite and non-negative")
    total_weight = float(sum(requested.values()))
    if total_weight <= 0:
        raise ValueError("at least one centrality weight must be positive")
    requested = {key: value / total_weight for key, value in requested.items()}
    composite = sum(requested[key] * normalized[key] for key in normalized)

    return {
        **normalized,
        "composite": np.asarray(composite, dtype=float),
        "degree_raw": degree_raw,
        "betweenness_raw": betweenness_raw,
        "eigenvector_raw": eigenvector_raw,
    }


compute_centrality_scores = centrality_composite


def top_fraction_mask(
    scores: ArrayLike,
    fraction: float = 0.10,
) -> NDArray[np.bool_]:
    """Select exactly ``ceil(fraction * n_nodes)`` highest-scoring nodes."""

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores contain NaN or infinite values")
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    count = min(values.size, max(1, int(ceil(fraction * values.size))))
    # lexsort uses node index as a deterministic secondary key.
    order = np.lexsort((np.arange(values.size), -values))
    mask = np.zeros(values.size, dtype=bool)
    mask[order[:count]] = True
    return mask


select_actuator_mask = top_fraction_mask


def synthesize_control_mask(
    adjacency: ArrayLike,
    *,
    target_fraction: float = 0.10,
    centrality_weights: Mapping[str, float] | None = None,
) -> tuple[NDArray[np.bool_], dict[str, NDArray[np.float64]]]:
    """Compute the framework's centrality-based binary actuator mask."""

    scores = centrality_composite(adjacency, weights=centrality_weights)
    mask = top_fraction_mask(scores["composite"], fraction=target_fraction)
    return mask, scores


def control_matrix(mask: ArrayLike) -> NDArray[np.float64]:
    """Convert a one-dimensional actuator mask into diagonal matrix ``B``."""

    values = np.asarray(mask)
    if values.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    return np.diag(values.astype(bool).astype(float))


def _node_set(values: ArrayLike | Iterable[Any]) -> tuple[set[Any], int | None]:
    # ``np.asarray(set(...))`` produces a scalar object array, so materialize
    # generic iterables while preserving ndarray/list mask semantics.
    if isinstance(values, np.ndarray):
        array = values
    else:
        array = np.asarray(list(values))
    if array.ndim != 1:
        raise ValueError("node sets/masks must be one-dimensional")
    if array.dtype == bool:
        return set(np.flatnonzero(array).tolist()), int(array.size)
    return set(array.tolist()), None


def set_overlap_metrics(
    selected: ArrayLike | Iterable[Any],
    reference: ArrayLike | Iterable[Any],
    *,
    n_nodes: int | None = None,
) -> dict[str, float | int]:
    """Quantify overlap between selected targets and a clinical annotation.

    In addition to counts, the result reports target precision, annotation
    coverage (recall), Dice, Jaccard, chance expectation, fold enrichment, and a
    one-sided hypergeometric enrichment p value.
    """

    selected_set, selected_mask_n = _node_set(selected)
    reference_set, reference_mask_n = _node_set(reference)
    if (
        selected_mask_n is not None
        and reference_mask_n is not None
        and selected_mask_n != reference_mask_n
    ):
        raise ValueError("selected and reference masks must have matching lengths")
    inferred = [value for value in (selected_mask_n, reference_mask_n) if value]
    if n_nodes is None and inferred:
        n_nodes = max(inferred)
    if n_nodes is None:
        numeric = selected_set | reference_set
        if numeric and all(isinstance(item, (int, np.integer)) for item in numeric):
            n_nodes = int(max(numeric)) + 1
        else:
            n_nodes = len(numeric)
    if n_nodes < len(selected_set | reference_set):
        raise ValueError("n_nodes is smaller than the observed node universe")
    numeric = selected_set | reference_set
    if numeric and all(isinstance(item, (int, np.integer)) for item in numeric):
        if min(numeric) < 0 or max(numeric) >= n_nodes:
            raise ValueError("node indices must lie in [0, n_nodes)")

    overlap = len(selected_set & reference_set)
    n_selected, n_reference = len(selected_set), len(reference_set)
    precision = overlap / n_selected if n_selected else float("nan")
    recall = overlap / n_reference if n_reference else float("nan")
    dice_denominator = n_selected + n_reference
    union = len(selected_set | reference_set)
    dice = 2 * overlap / dice_denominator if dice_denominator else float("nan")
    jaccard = overlap / union if union else float("nan")
    expected = n_selected * n_reference / n_nodes if n_nodes else float("nan")
    enrichment = overlap / expected if expected > 0 else float("nan")
    p_value = (
        float(hypergeom.sf(overlap - 1, n_nodes, n_reference, n_selected))
        if n_nodes and n_selected and n_reference
        else float("nan")
    )
    return {
        "n_nodes": int(n_nodes),
        "selected_n": n_selected,
        "reference_n": n_reference,
        "overlap_n": overlap,
        "precision": float(precision),
        "recall": float(recall),
        "dice": float(dice),
        "jaccard": float(jaccard),
        "expected_overlap": float(expected),
        "enrichment": float(enrichment),
        "hypergeom_p": p_value,
    }


def clinical_overlap_metrics(
    selected: ArrayLike | Iterable[Any],
    *,
    soz: ArrayLike | Iterable[Any] | None = None,
    resection: ArrayLike | Iterable[Any] | None = None,
    n_nodes: int | None = None,
) -> dict[str, float | int]:
    """Return prefixed SOZ and/or resection overlap metrics."""

    if soz is None and resection is None:
        raise ValueError("at least one of soz or resection must be provided")
    output: dict[str, float | int] = {}
    for name, annotation in (("soz", soz), ("resection", resection)):
        if annotation is None:
            continue
        values = set_overlap_metrics(selected, annotation, n_nodes=n_nodes)
        output.update({f"{name}_{key}": value for key, value in values.items()})
    return output


soz_resection_overlap = clinical_overlap_metrics
