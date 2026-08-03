"""Optional plotting helpers for PMU placements."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from .placement import observed_nodes


def placement_figure(
    graph: nx.Graph,
    pmu_nodes: Iterable[int],
    *,
    title: str,
    failed_edge: tuple[int, int] | None = None,
    highlight_unobserved: bool = False,
):
    """Create a deterministic PMU placement figure without showing it."""
    pmus = set(pmu_nodes)
    observed = observed_nodes(graph, pmus)
    simple_graph = nx.Graph(graph)
    position = nx.kamada_kawai_layout(simple_graph)

    colors = []
    for node in simple_graph.nodes():
        if node in pmus:
            colors.append("orange")
        elif highlight_unobserved and node not in observed:
            colors.append("lightgray")
        else:
            colors.append("skyblue")

    figure, axis = plt.subplots(figsize=(10, 8))
    nx.draw(
        simple_graph,
        position,
        with_labels=True,
        node_color=colors,
        node_size=900,
        edgecolors="black",
        font_weight="bold",
        ax=axis,
    )
    if failed_edge and all(node in position for node in failed_edge):
        u, v = failed_edge
        axis.plot(
            [position[u][0], position[v][0]],
            [position[u][1], position[v][1]],
            color="red",
            linewidth=5,
        )
    axis.set_title(title, fontweight="bold")
    figure.tight_layout()
    return figure


def save_figure(
    figure,
    output_stem: str | Path,
    *,
    inkscape_executable: str | Path | None = None,
) -> dict[str, Path]:
    """Save PNG and SVG, with optional EMF conversion through Inkscape."""
    stem = Path(output_stem).resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    svg_path = stem.with_suffix(".svg")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(svg_path, format="svg", bbox_inches="tight")
    outputs = {"png": png_path, "svg": svg_path}

    if inkscape_executable is not None:
        inkscape = Path(inkscape_executable).resolve()
        if not inkscape.is_file():
            raise FileNotFoundError(f"Inkscape executable not found: {inkscape}")
        emf_path = stem.with_suffix(".emf")
        subprocess.run(
            [str(inkscape), str(svg_path), "--export-type=emf", f"--export-filename={emf_path}"],
            check=True,
        )
        outputs["emf"] = emf_path
    return outputs
