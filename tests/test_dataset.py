from __future__ import annotations

import csv
import json

import numpy as np

from pmu_placement_gnn.dataset import build_dataset, graph_to_arrays
from pmu_placement_gnn.power_network import BranchFailure, build_graph


def test_graph_arrays_are_bidirectional_and_keep_parallel_edges(fake_net):
    graph = build_graph(fake_net)
    arrays = graph_to_arrays(graph, fake_net, [0, 2])

    assert arrays["x"].shape == (4, 5)
    assert arrays["edge_index"].shape == (2, 8)
    assert arrays["edge_attr"].shape == (8, 6)
    directed = [tuple(edge) for edge in arrays["edge_index"].T.tolist()]
    assert directed.count((0, 1)) == 2
    assert directed.count((1, 0)) == 2
    assert arrays["y"].tolist() == [1, 0, 1, 0]


def test_dataset_has_consistent_schema_and_relative_paths(fake_net, tmp_path):
    failures = [(BranchFailure("line", 0), BranchFailure("trafo", 0))]
    result = build_dataset(
        fake_net,
        failures,
        tmp_path / "dataset",
        system_name="TEST4",
        seed=11,
    )

    with result.index_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert not rows[1]["file"].startswith(("/", "\\"))
    assert json.loads(rows[1]["failures_json"]) == [
        {"kind": "line", "idx": 0},
        {"kind": "trafo", "idx": 0},
    ]

    graph_y_shapes = []
    for row in rows:
        with np.load(result.root / row["file"], allow_pickle=False) as sample:
            graph_y_shapes.append(sample["graph_y"].shape)
            assert sample["edge_index"].shape[0] == 2
    assert graph_y_shapes == [(4,), (4,)]

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["num_samples"] == 2

