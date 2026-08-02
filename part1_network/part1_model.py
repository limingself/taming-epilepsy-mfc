#!/usr/bin/env python
"""Build the final Part I connected PLV graph and write Fig. 1 Source Data.

This is the only Part I model/data-processing entry point.  It performs no
plotting: the publication figure is rendered independently by
``figure_01_plv_network_selection.py`` from the tables written here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from mfc_pipeline.network import centrality_composite, mst_proportional_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "part1_network" / "config.yaml"
    )
    return parser.parse_args()


def circular_positions(n_nodes: int) -> dict[int, np.ndarray]:
    angles = np.linspace(
        np.pi / 2.0, np.pi / 2.0 - 2.0 * np.pi, n_nodes, endpoint=False
    )
    return {i: np.array([np.cos(angle), np.sin(angle)]) for i, angle in enumerate(angles)}


def matrices_from_tables(
    nodes: pd.DataFrame, edges: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the weighted adjacency and unnormalized Laplacian."""
    n_nodes = len(nodes)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for row in edges.itertuples():
        left, right, weight = int(row.source), int(row.target), float(row.plv_weight)
        adjacency[left, right] = weight
        adjacency[right, left] = weight
    degree = adjacency.sum(axis=1)
    return adjacency, np.diag(degree) - adjacency


def build_source_data(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute the connected graph, centralities, and frozen node selection."""
    network_path = (PROJECT / str(config["source_network"])).resolve()
    with np.load(network_path, allow_pickle=False) as archive:
        channels = archive["channels"].astype(str)
        plv_key = f"plv_{config['network']['plv_source']}"
        if plv_key not in archive.files:
            raise KeyError(f"missing {plv_key!r} in {network_path}")
        similarity = np.asarray(archive[plv_key], dtype=np.float64)

    adjacency = mst_proportional_graph(similarity, float(config["network"]["density"]))
    graph = nx.from_numpy_array(adjacency)
    if not nx.is_connected(graph):
        raise RuntimeError("MST-backed PLV graph must be connected")

    weights = {
        str(key): float(value)
        for key, value in config["weighted_centrality"]["weights"].items()
    }
    centrality = centrality_composite(adjacency, weights=weights)
    score = np.asarray(centrality["composite"], dtype=np.float64)
    quantile = float(config["selection"]["threshold_quantile"])
    threshold = float(np.quantile(score, quantile, method="linear"))
    selected = score > threshold

    clinical_path = (PROJECT / str(config["clinical_node_table"])).resolve()
    clinical_table = pd.read_csv(clinical_path).set_index("channel").loc[channels]
    if tuple(channels) != tuple(clinical_table.index.astype(str)):
        raise RuntimeError("network and frozen clinical-label channel orders differ")

    def parse_bool(values: pd.Series) -> np.ndarray:
        return (
            values.astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes"})
            .to_numpy()
        )

    soz = parse_bool(clinical_table["soz"])
    resection = parse_bool(clinical_table["resection"])
    clinical = soz | resection
    category = np.full(len(channels), "not_selected", dtype=object)
    category[selected & ~clinical] = "selected_nonclinical"
    category[selected & clinical] = "selected_clinical_match"
    percentile = (pd.Series(score).rank(method="average").to_numpy() - 1.0) / max(
        len(score) - 1, 1
    )
    position = circular_positions(len(channels))

    nodes = pd.DataFrame(
        {
            "node": np.arange(len(channels), dtype=int),
            "channel": channels,
            "x": [position[i][0] for i in range(len(channels))],
            "y": [position[i][1] for i in range(len(channels))],
            "degree_normalized": centrality["degree"],
            "betweenness_normalized": centrality["betweenness"],
            "eigenvector_normalized": centrality["eigenvector"],
            "weighted_centrality_score": score,
            "score_percentile": percentile,
            "selection_threshold": threshold,
            "selected": selected,
            "soz": soz,
            "resection": resection,
            "clinical_union": clinical,
            "category": category,
            "status_description": clinical_table["status_description"].astype(str).tolist(),
        }
    )

    edge_rows: list[dict[str, Any]] = []
    for left in range(len(channels)):
        for right in range(left + 1, len(channels)):
            if adjacency[left, right] > 0:
                edge_rows.append(
                    {
                        "source": left,
                        "target": right,
                        "source_channel": channels[left],
                        "target_channel": channels[right],
                        "plv_weight": float(adjacency[left, right]),
                    }
                )
    edges = pd.DataFrame(edge_rows)
    overlap = int(np.sum(selected & clinical))
    degree = adjacency.sum(axis=1)
    laplacian = np.diag(degree) - adjacency
    eigenvalues = np.linalg.eigvalsh(laplacian)
    summary = {
        "subject": str(config["subject"]),
        "n_nodes": int(len(nodes)),
        "n_edges": int(len(edges)),
        "requested_density": float(config["network"]["density"]),
        "realized_density": float(nx.density(graph)),
        "connected_components": int(nx.number_connected_components(graph)),
        "plv_source": str(config["network"]["plv_source"]),
        "connectivity_rule": str(config["network"]["connectivity_rule"]),
        "centrality_weights": weights,
        "threshold_quantile": quantile,
        "threshold_raw_score": threshold,
        "selection_rule": "weighted_centrality_score > threshold_raw_score",
        "selected_count": int(selected.sum()),
        "selected_clinical_match_count": overlap,
        "selected_nonclinical_count": int(np.sum(selected & ~clinical)),
        "soz_count": int(soz.sum()),
        "resection_count": int(resection.sum()),
        "clinical_union_count": int(clinical.sum()),
        "selected_soz_overlap_count": int(np.sum(selected & soz)),
        "selected_resection_overlap_count": int(np.sum(selected & resection)),
        "selected_clinical_precision": overlap / max(int(selected.sum()), 1),
        "clinical_union_recall": overlap / max(int(clinical.sum()), 1),
        "clinical_labels_used_for_selection": False,
        "adjacency_symmetric_max_abs_error": float(
            np.max(np.abs(adjacency - adjacency.T))
        ),
        "laplacian_max_abs_row_sum": float(np.max(np.abs(laplacian.sum(axis=1)))),
        "laplacian_zero_eigenvalue_count": int(
            np.sum(np.abs(eigenvalues) < 1e-10)
        ),
        "laplacian_algebraic_connectivity": float(eigenvalues[1]),
    }
    return nodes, edges, summary


def write_source_data(
    config: dict[str, Any],
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    """Write all tables needed by the independent Fig. 1 renderer."""
    output_dir = (PROJECT / str(config["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(
        output_dir / "hup060_plv_nodes_and_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    edges.to_csv(
        output_dir / "hup060_connected_plv_edges.csv",
        index=False,
        encoding="utf-8-sig",
    )
    adjacency, laplacian = matrices_from_tables(nodes, edges)
    channels = nodes.sort_values("node")["channel"].astype(str).tolist()
    pd.DataFrame(adjacency, index=channels, columns=channels).rename_axis(
        "channel"
    ).to_csv(output_dir / "hup060_adjacency_matrix.csv", encoding="utf-8-sig")
    pd.DataFrame(laplacian, index=channels, columns=channels).rename_axis(
        "channel"
    ).to_csv(output_dir / "hup060_laplacian_matrix.csv", encoding="utf-8-sig")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    red = nodes.loc[
        nodes["category"].eq("selected_clinical_match"), "channel"
    ].tolist()
    blue = nodes.loc[
        nodes["category"].eq("selected_nonclinical"), "channel"
    ].tolist()
    report = f"""# HUP060 Part 1: connected PLV network and actuator selection

- Graph: training-only equal-weight multiband PLV; maximum-spanning-tree backbone plus the strongest edges to density 0.10.
- Size: {summary['n_nodes']} nodes, {summary['n_edges']} edges, {summary['connected_components']} connected component.
- Score: `0.30 degree + 0.60 betweenness + 0.10 eigenvector`; each component is min-max normalized within HUP060.
- Threshold: training-score {100.0 * float(summary['threshold_quantile']):.0f}th percentile, raw cutoff `{summary['threshold_raw_score']:.6f}`; selection uses a strict greater-than rule. The actuator-density hyperparameter was selected during development and is fixed before frozen-test evaluation.
- Clinical labels: SOZ/resection labels are excluded from graph construction, scoring, and threshold selection; they are used only for retrospective coloring.
- Red nodes (selected and SOZ/resection matched, n={len(red)}): {', '.join(red)}.
- Blue nodes (selected without SOZ/resection match, n={len(blue)}): {', '.join(blue)}.
- All unselected nodes are grey.

Descriptive overlap: {summary['selected_clinical_match_count']}/{summary['selected_count']} selected nodes match the SOZ/resection union, covering {summary['selected_clinical_match_count']}/{summary['clinical_union_count']} clinically labelled nodes. This is a one-patient retrospective description, not an inferential localization result.
"""
    (output_dir / "method_and_results_zh.md").write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    nodes, edges, summary = build_source_data(config)
    write_source_data(config, nodes, edges, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
