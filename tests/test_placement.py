from __future__ import annotations

import networkx as nx
import numpy as np

from pmu_placement_gnn.placement import (
    coverage_aware_selection,
    coverage_percent,
    greedy_pmu_placement,
)


def test_greedy_placement_is_deterministic():
    graph = nx.path_graph(4)

    assert greedy_pmu_placement(graph) == [1, 2]
    assert coverage_percent(graph, [1, 2]) == 100.0


def test_coverage_aware_selection_prefers_gain_then_probability():
    edge_index = np.asarray([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=np.int64)
    probabilities = np.asarray([0.99, 0.20, 0.30, 0.10])

    assert coverage_aware_selection(probabilities, edge_index, k=1) == [2]


def test_coverage_aware_selection_rejects_invalid_k():
    edge_index = np.empty((2, 0), dtype=np.int64)

    try:
        coverage_aware_selection([0.5], edge_index, k=2)
    except ValueError as exc:
        assert "k must be" in str(exc)
    else:
        raise AssertionError("invalid k was accepted")
