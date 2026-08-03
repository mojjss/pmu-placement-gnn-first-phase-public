"""Graph-only PMU placement and coverage-aware decoding algorithms."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import networkx as nx
import numpy as np


def observed_nodes(graph: nx.Graph, pmu_nodes: Iterable[int]) -> set[int]:
    """Return buses observed by PMUs using the one-hop observability model."""
    observed: set[int] = set()
    for node in pmu_nodes:
        if node not in graph:
            raise ValueError(f"PMU bus {node} is not present in the graph")
        observed.add(node)
        observed.update(graph.neighbors(node))
    return observed


def coverage_percent(graph: nx.Graph, pmu_nodes: Iterable[int]) -> float:
    """Return one-hop bus coverage as a percentage."""
    if graph.number_of_nodes() == 0:
        return 100.0
    return 100.0 * len(observed_nodes(graph, pmu_nodes)) / graph.number_of_nodes()


def greedy_pmu_placement(graph: nx.Graph) -> list[int]:
    """Select a deterministic greedy one-hop dominating set.

    The algorithm matches the notebooks' heuristic. Ties are resolved by the
    smallest bus identifier, making repeated runs and generated labels stable.
    """
    candidates = sorted(int(node) for node in graph.nodes())
    all_nodes = set(candidates)
    observed: set[int] = set()
    selected: list[int] = []
    selected_set: set[int] = set()

    while observed != all_nodes:
        best_node: int | None = None
        best_gain = -1

        for node in candidates:
            if node in selected_set:
                continue
            gain = len(observed_nodes(graph, [node]) - observed)
            if gain > best_gain:
                best_node = node
                best_gain = gain

        if best_node is None or best_gain <= 0:
            missing = sorted(all_nodes - observed)
            raise RuntimeError(f"Greedy placement stalled with unobserved buses: {missing}")

        selected.append(best_node)
        selected_set.add(best_node)
        observed.update(observed_nodes(graph, [best_node]))

    return selected


def neighbors_from_edge_index(edge_index: np.ndarray, num_nodes: int) -> list[set[int]]:
    """Build undirected neighbor sets from a PyG-style edge index."""
    edges = np.asarray(edge_index)
    if edges.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {edges.shape}")

    neighbors = [set() for _ in range(num_nodes)]
    for source, target in edges.T:
        u = int(source)
        v = int(target)
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise ValueError(f"Edge ({u}, {v}) is outside 0..{num_nodes - 1}")
        if u != v:
            neighbors[u].add(v)
            neighbors[v].add(u)
    return neighbors


def coverage_aware_selection(
    probabilities: Sequence[float] | np.ndarray,
    edge_index: np.ndarray,
    k: int,
    candidate_limit: int | None = None,
) -> list[int]:
    """Choose ``k`` nodes by coverage gain, using GNN scores as tie-breakers.

    Python integer bit masks keep the decoder fast while preserving the logic
    used in the later notebook variants.
    """
    scores = np.asarray(probabilities, dtype=float).reshape(-1)
    num_nodes = len(scores)
    if k < 0 or k > num_nodes:
        raise ValueError(f"k must be between 0 and {num_nodes}, got {k}")
    if k == 0:
        return []
    if not np.isfinite(scores).all():
        raise ValueError("probabilities must contain only finite values")

    neighbors = neighbors_from_edge_index(edge_index, num_nodes)
    masks: list[int] = []
    for node, adjacent in enumerate(neighbors):
        mask = 1 << node
        for neighbor in adjacent:
            mask |= 1 << neighbor
        masks.append(mask)

    ranked = sorted(range(num_nodes), key=lambda node: (-scores[node], node))
    if candidate_limit is not None:
        if candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive")
        ranked = ranked[: min(num_nodes, max(k, candidate_limit))]

    selected: list[int] = []
    selected_set: set[int] = set()
    observed_mask = 0

    for _ in range(k):
        best_node: int | None = None
        best_gain = -1
        for node in ranked:
            if node in selected_set:
                continue
            gain = (masks[node] & ~observed_mask).bit_count()
            if gain > best_gain:
                best_node = node
                best_gain = gain

        if best_node is None:
            break
        selected.append(best_node)
        selected_set.add(best_node)
        observed_mask |= masks[best_node]

    return selected

