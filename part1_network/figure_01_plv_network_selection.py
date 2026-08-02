#!/usr/bin/env python
"""Render paper Fig. 1 exclusively from the frozen Part I Source Data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parents[1]

COLORS = {
    "clinical_target": "#C8453D",
    "nonclinical_target": "#2E6FB5",
    "other": "#D9DEE3",
    "other_edge": "#9099A2",
    "ink": "#252525",
    "edge": "#8299AB",
    "grid": "#E1E4E8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "part1_network" / "config.yaml"
    )
    return parser.parse_args()


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.4,
            "axes.titlesize": 7.5,
            "axes.labelsize": 6.8,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.1,
            "legend.fontsize": 5.7,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.10) -> None:
    ax.text(
        x,
        1.045,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def category_color(category: str) -> str:
    return {
        "selected_clinical_match": COLORS["clinical_target"],
        "selected_nonclinical": COLORS["nonclinical_target"],
        "not_selected": COLORS["other"],
    }[category]


def load_source_data(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load, but never recompute, every quantitative element of Fig. 1."""
    source_dir = (PROJECT / str(config["output_dir"])).resolve()
    required = {
        "nodes": source_dir / "hup060_plv_nodes_and_selection.csv",
        "edges": source_dir / "hup060_connected_plv_edges.csv",
        "adjacency": source_dir / "hup060_adjacency_matrix.csv",
        "laplacian": source_dir / "hup060_laplacian_matrix.csv",
        "summary": source_dir / "summary.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Part I Source Data are incomplete. Run part1_model.py first:\n"
            + "\n".join(missing)
        )

    nodes = pd.read_csv(required["nodes"])
    edges = pd.read_csv(required["edges"])
    adjacency_table = pd.read_csv(required["adjacency"], index_col=0)
    laplacian_table = pd.read_csv(required["laplacian"], index_col=0)
    summary = json.loads(required["summary"].read_text(encoding="utf-8"))

    nodes = nodes.sort_values("node").reset_index(drop=True)
    channels = nodes["channel"].astype(str).tolist()
    if adjacency_table.index.astype(str).tolist() != channels:
        raise RuntimeError("adjacency row order differs from the node Source Data")
    if adjacency_table.columns.astype(str).tolist() != channels:
        raise RuntimeError("adjacency column order differs from the node Source Data")
    if laplacian_table.index.astype(str).tolist() != channels:
        raise RuntimeError("Laplacian row order differs from the node Source Data")
    if laplacian_table.columns.astype(str).tolist() != channels:
        raise RuntimeError("Laplacian column order differs from the node Source Data")
    if int(summary["n_nodes"]) != len(nodes) or int(summary["n_edges"]) != len(edges):
        raise RuntimeError("summary counts differ from the tabulated Source Data")

    return (
        nodes,
        edges,
        adjacency_table.to_numpy(dtype=np.float64),
        laplacian_table.to_numpy(dtype=np.float64),
        summary,
    )


def draw_network(
    ax: plt.Axes,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    graph = nx.Graph()
    graph.add_nodes_from(nodes["node"].astype(int))
    for row in edges.itertuples():
        graph.add_edge(int(row.source), int(row.target), weight=float(row.plv_weight))
    position = {int(row.node): np.array([row.x, row.y]) for row in nodes.itertuples()}
    edge_values = edges["plv_weight"].to_numpy(float)
    edge_norm = (edge_values - edge_values.min()) / max(np.ptp(edge_values), 1e-12)
    nx.draw_networkx_edges(
        graph,
        position,
        ax=ax,
        edgelist=list(zip(edges["source"], edges["target"])),
        width=(0.30 + 1.35 * edge_norm).tolist(),
        alpha=(0.16 + 0.42 * edge_norm).tolist(),
        edge_color=COLORS["edge"],
    )
    sizes = 38 + 92 * nodes["weighted_centrality_score"].to_numpy(float)
    for category in ("not_selected", "selected_nonclinical", "selected_clinical_match"):
        mask = nodes["category"].eq(category).to_numpy()
        nodelist = nodes.loc[mask, "node"].astype(int).tolist()
        nx.draw_networkx_nodes(
            graph,
            position,
            ax=ax,
            nodelist=nodelist,
            node_size=sizes[mask].tolist(),
            node_color=category_color(category),
            edgecolors=COLORS["other_edge"] if category == "not_selected" else "white",
            linewidths=0.65 if category == "not_selected" else 1.05,
        )
    red_mask = nodes["category"].eq("selected_clinical_match").to_numpy()
    nx.draw_networkx_nodes(
        graph,
        position,
        ax=ax,
        nodelist=nodes.loc[red_mask, "node"].astype(int).tolist(),
        node_size=(sizes[red_mask] + 22).tolist(),
        node_color="none",
        edgecolors=COLORS["ink"],
        linewidths=0.75,
    )
    for row in nodes.itertuples():
        tx, ty = 1.17 * position[int(row.node)]
        ha = "left" if tx > 0.09 else ("right" if tx < -0.09 else "center")
        if row.category == "selected_clinical_match":
            color, weight = COLORS["clinical_target"], "bold"
        elif row.category == "selected_nonclinical":
            color, weight = COLORS["nonclinical_target"], "bold"
        else:
            color, weight = "#59616A", "normal"
        ax.text(
            tx,
            ty,
            row.channel,
            fontsize=4.65,
            fontweight=weight,
            color=color,
            ha=ha,
            va="center",
        )
    ax.set_title("Connected multiband PLV network", loc="left", fontweight="bold")
    ax.text(
        0.01,
        0.985,
        f"{summary['n_nodes']} nodes | {summary['n_edges']} edges | density {summary['realized_density']:.3f} | connected",
        transform=ax.transAxes,
        fontsize=5.6,
        color="#4C535A",
        ha="left",
        va="top",
    )
    ax.set(xlim=(-1.38, 1.38), ylim=(-1.30, 1.30), aspect="equal")
    ax.axis("off")
    panel_label(ax, "a")


def draw_scores(
    ax: plt.Axes, nodes: pd.DataFrame, summary: dict[str, Any]
) -> None:
    ranked = nodes.sort_values(["weighted_centrality_score", "channel"]).reset_index(
        drop=True
    )
    y = np.arange(len(ranked), dtype=float)
    values = ranked["weighted_centrality_score"].to_numpy(float)
    colors = [category_color(category) for category in ranked["category"]]
    threshold = float(summary["threshold_raw_score"])
    ax.axvspan(
        threshold, 1.02, color=COLORS["nonclinical_target"], alpha=0.035, linewidth=0
    )
    ax.hlines(y, 0, values, color=colors, linewidth=1.05, alpha=0.86)
    ax.scatter(
        values,
        y,
        c=colors,
        s=np.where(ranked["selected"].to_numpy(bool), 25, 13),
        edgecolors=np.where(
            ranked["selected"].to_numpy(bool), "white", COLORS["other_edge"]
        ),
        linewidths=0.55,
        zorder=3,
    )
    ax.vlines(
        threshold,
        -0.8,
        len(ranked) - 0.35,
        color=COLORS["ink"],
        linewidth=0.9,
        linestyle="--",
    )
    ax.set_yticks(y, ranked["channel"].tolist())
    for label, category in zip(ax.get_yticklabels(), ranked["category"]):
        label.set_color(
            category_color(category) if category != "not_selected" else "#59616A"
        )
        label.set_fontweight("bold" if category != "not_selected" else "normal")
    ax.set(
        xlim=(0, 1.03),
        ylim=(-0.8, len(ranked) + 4.0),
        xlabel="Weighted centrality score",
    )
    ax.set_title("Threshold-based actuator selection", loc="left", fontweight="bold")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.5)
    ax.text(
        0.01,
        len(ranked) + 3.25,
        r"$S_i=0.30C_{deg}+0.60C_{bet}+0.10C_{eig}$"
        "\n"
        rf"select if $S_i>Q_{{{summary['threshold_quantile']:.2f}}}(S)={threshold:.3f}$; labels applied after selection"
        "\n"
        f"{summary['selected_count']}/36 selected: {summary['selected_clinical_match_count']} red + {summary['selected_nonclinical_count']} blue",
        fontsize=5.6,
        color="#3F454B",
        ha="left",
        va="top",
        linespacing=1.3,
    )
    panel_label(ax, "b", x=-0.13)


def draw_matrix_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    nodes: pd.DataFrame,
    *,
    kind: str,
    label: str,
) -> None:
    channels = nodes.sort_values("node")["channel"].astype(str).tolist()
    n_nodes = len(channels)
    tick_positions = np.unique(
        np.concatenate([np.arange(0, n_nodes, 4, dtype=int), [n_nodes - 1]])
    )
    if kind == "adjacency":
        vmax = max(float(np.max(matrix)), np.finfo(float).eps)
        image = ax.imshow(
            matrix,
            cmap="Blues",
            vmin=0.0,
            vmax=vmax,
            origin="upper",
            interpolation="nearest",
            rasterized=False,
        )
        title = r"Sparsified weighted adjacency $A$"
        colorbar_label = "Retained PLV weight"
    elif kind == "laplacian":
        limit = max(float(np.max(np.abs(matrix))), np.finfo(float).eps)
        image = ax.imshow(
            matrix,
            cmap="PuOr_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            origin="upper",
            interpolation="nearest",
            rasterized=False,
        )
        title = r"Graph Laplacian $L=D-A$"
        colorbar_label = "Laplacian entry"
    else:
        raise ValueError(f"unknown matrix kind: {kind}")

    ax.set_xticks(tick_positions, [channels[index] for index in tick_positions])
    ax.set_yticks(tick_positions, [channels[index] for index in tick_positions])
    ax.tick_params(axis="x", labelrotation=90, labelsize=4.3, length=1.8, pad=1.0)
    ax.tick_params(axis="y", labelsize=4.3, length=1.8, pad=1.0)
    ax.set_xlabel("Channel order", labelpad=2.0)
    ax.set_ylabel("Channel order", labelpad=2.0)
    ax.set_title(title, loc="left", fontweight="bold", pad=5.0)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.45)
        spine.set_color("#8A929A")
    colorbar = ax.figure.colorbar(
        image,
        ax=ax,
        orientation="horizontal",
        fraction=0.055,
        pad=0.16,
        aspect=28,
    )
    colorbar.ax.tick_params(labelsize=4.5, length=2.0, width=0.45)
    colorbar.set_label(colorbar_label, fontsize=5.3, labelpad=1.5)
    colorbar.outline.set_linewidth(0.45)
    panel_label(ax, label, x=-0.18)


def make_figure(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    adjacency: np.ndarray,
    laplacian: np.ndarray,
    summary: dict[str, Any],
) -> plt.Figure:
    fig = plt.figure(figsize=(7.20, 6.85))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.20, 0.80],
        left=0.045,
        right=0.985,
        bottom=0.105,
        top=0.965,
        hspace=0.33,
    )
    top = outer[0].subgridspec(1, 2, width_ratios=[1.18, 0.82], wspace=0.22)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.36)
    draw_network(fig.add_subplot(top[0, 0]), nodes, edges, summary)
    draw_scores(fig.add_subplot(top[0, 1]), nodes, summary)
    draw_matrix_heatmap(
        fig.add_subplot(bottom[0, 0]),
        adjacency,
        nodes,
        kind="adjacency",
        label="c",
    )
    draw_matrix_heatmap(
        fig.add_subplot(bottom[0, 1]),
        laplacian,
        nodes,
        kind="laplacian",
        label="d",
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["clinical_target"],
            markeredgecolor=COLORS["ink"],
            markersize=6,
            label="Selected and SOZ/resection matched",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["nonclinical_target"],
            markeredgecolor="white",
            markersize=6,
            label="Selected without SOZ/resection match",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["other"],
            markeredgecolor=COLORS["other_edge"],
            markersize=6,
            label="Not selected",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["edge"],
            linewidth=1.3,
            alpha=0.65,
            label="Retained PLV edge (width/opacity proportional to PLV)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.010),
        ncol=2,
        columnspacing=1.9,
        handletextpad=0.7,
    )
    return fig


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_legend(figure_dir: Path, summary: dict[str, Any]) -> None:
    quantile = float(summary["threshold_quantile"])
    legend = f"""# HUP060 connected PLV network and thresholded actuator selection

**a,** A {summary['n_nodes']}-node HUP060 functional network was constructed from training-only equal-weight multiband PLV. A maximum-spanning-tree backbone was retained before adding the strongest remaining edges to a density of {summary['realized_density']:.3f}, yielding one connected component. Edge width and opacity encode retained PLV strength. Node size encodes the weighted centrality score. Red nodes are selected actuators that coincide with either the clinically annotated seizure-onset zone (SOZ) or resection region; blue nodes are selected actuators without either clinical label; all unselected nodes are grey. **b,** Degree, inverse-distance betweenness, and eigenvector centrality were normalized within HUP060 and combined with fixed weights 0.30, 0.60, and 0.10. Nodes were selected when their score exceeded the training-score {100.0 * quantile:.0f}th percentile (tau = {summary['threshold_raw_score']:.3f}). The actuator-density hyperparameter was selected on development data and fixed before frozen-test evaluation. Clinical annotations were not used for graph construction, score calculation, or threshold selection and were applied only for retrospective coloring. The threshold selected {summary['selected_count']}/{summary['n_nodes']} nodes, of which {summary['selected_clinical_match_count']} matched the SOZ/resection union. **c,** Weighted adjacency matrix $A$ of the same sparsified PLV graph, displayed in the identical channel order used in panel a. **d,** Unnormalized graph Laplacian $L=D-A$, where $D$ is the diagonal weighted-degree matrix. Its single zero eigenvalue and positive algebraic connectivity ({summary['laplacian_algebraic_connectivity']:.3f}) independently confirm that the retained graph is connected.
"""
    (figure_dir / "hup060_part1_plv_network_selection_legend.md").write_text(
        legend, encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    style()
    nodes, edges, adjacency, laplacian, summary = load_source_data(config)
    figure = make_figure(nodes, edges, adjacency, laplacian, summary)
    figure_dir = (PROJECT / str(config["figure_dir"])).resolve()
    save_figure(figure, figure_dir / "hup060_part1_plv_network_selection")
    write_legend(figure_dir, summary)
    print(
        json.dumps(
            {
                "figure": "figure_01",
                "reads_frozen_source_data": True,
                "n_nodes": int(summary["n_nodes"]),
                "n_edges": int(summary["n_edges"]),
                "selected_count": int(summary["selected_count"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
