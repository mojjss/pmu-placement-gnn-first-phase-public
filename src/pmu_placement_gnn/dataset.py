"""Build portable NPZ graph datasets from pandapower networks."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from .placement import coverage_percent, greedy_pmu_placement
from .power_network import BranchFailure, apply_failures, build_graph, failures_json

NODE_FEATURES = ("voltage_kv", "degree", "has_load", "has_generator", "has_external_grid")
EDGE_FEATURES = (
    "length_km",
    "resistance_ohm_per_km",
    "rated_power_mva",
    "impedance_percent",
    "is_line",
    "is_transformer",
)
GRAPH_TARGETS = ("fixed_coverage_pct", "reoptimized_coverage_pct", "delta_pmus", "components")


@dataclass(frozen=True)
class DatasetBuildResult:
    """Paths and counts produced by :func:`build_dataset`."""

    root: Path
    index_path: Path
    metadata_path: Path
    sample_count: int


def _bus_membership(net: Any, table_name: str) -> set[int]:
    if not hasattr(net, table_name):
        return set()
    table = getattr(net, table_name)
    if len(table) == 0 or "bus" not in table:
        return set()
    if "in_service" in table:
        table = table[table["in_service"].fillna(True).astype(bool)]
    return set(int(bus) for bus in table["bus"].tolist())


def _graph_edges(graph: nx.Graph):
    if graph.is_multigraph():
        yield from graph.edges(keys=True, data=True)
    else:
        for edge_number, (u, v, data) in enumerate(graph.edges(data=True)):
            yield u, v, edge_number, data


def graph_to_arrays(
    graph: nx.Graph,
    net: Any,
    pmu_nodes: list[int] | tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Convert a graph and labels to PyG-compatible NumPy arrays.

    Every physical undirected branch is emitted in both directions. Parallel
    branches remain separate edges, so message passing does not depend on the
    arbitrary order in which pandapower rows were read.
    """
    buses = sorted(int(bus) for bus in graph.nodes())
    bus_to_index = {bus: index for index, bus in enumerate(buses)}
    load_buses = _bus_membership(net, "load")
    generator_buses = _bus_membership(net, "gen")
    generator_buses.update(_bus_membership(net, "sgen"))
    external_buses = _bus_membership(net, "ext_grid")

    node_features = np.zeros((len(buses), len(NODE_FEATURES)), dtype=np.float32)
    for bus in buses:
        index = bus_to_index[bus]
        node_features[index] = (
            float(graph.nodes[bus].get("voltage_level", 0.0)),
            float(graph.degree[bus]),
            float(bus in load_buses),
            float(bus in generator_buses),
            float(bus in external_buses),
        )

    directed_edges: list[tuple[int, int]] = []
    directed_features: list[tuple[float, ...]] = []
    for u, v, _key, data in _graph_edges(graph):
        feature = (
            float(data.get("length", 0.0)),
            float(data.get("resistance", 0.0)),
            float(data.get("sn_mva", 0.0)),
            float(data.get("impedance", 0.0)),
            float(data.get("kind") == "line"),
            float(data.get("kind") in {"trafo", "trafo3w"}),
        )
        ui = bus_to_index[int(u)]
        vi = bus_to_index[int(v)]
        directed_edges.extend(((ui, vi), (vi, ui)))
        directed_features.extend((feature, feature))

    if directed_edges:
        edge_index = np.asarray(directed_edges, dtype=np.int64).T
        edge_attr = np.asarray(directed_features, dtype=np.float32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, len(EDGE_FEATURES)), dtype=np.float32)

    labels = np.zeros(len(buses), dtype=np.int64)
    for bus in pmu_nodes:
        if int(bus) not in bus_to_index:
            raise ValueError(f"PMU bus {bus} is not present in the graph")
        labels[bus_to_index[int(bus)]] = 1

    return {
        "x": node_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "y": labels,
        "bus_ids": np.asarray(buses, dtype=np.int64),
    }


def _save_sample(
    path: Path,
    arrays: dict[str, np.ndarray],
    *,
    graph_y: np.ndarray,
    scenario_type: str,
    serialized_failures: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **arrays,
        graph_y=np.asarray(graph_y, dtype=np.float32),
        scenario_type=np.asarray([scenario_type]),
        failures_json=np.asarray([serialized_failures]),
    )


def _scenario_row(
    *,
    sample_id: str,
    relative_path: str,
    scenario_type: str,
    serialized_failures: str,
    fixed_coverage: float,
    reoptimized_coverage: float,
    base_pmu_count: int,
    optimized_pmu_count: int,
    components: int,
    greedy_time_s: float,
) -> dict[str, object]:
    failures = json.loads(serialized_failures)
    return {
        "sample_id": sample_id,
        "file": relative_path,
        "scenario_type": scenario_type,
        "failures_json": serialized_failures,
        "num_failures": len(failures),
        "coverage_fixed_pct": fixed_coverage,
        "coverage_reoptimized_pct": reoptimized_coverage,
        "base_num_pmus": base_pmu_count,
        "optimized_num_pmus": optimized_pmu_count,
        "delta_pmus": optimized_pmu_count - base_pmu_count,
        "components": components,
        "greedy_time_s": greedy_time_s,
    }


def build_dataset(
    net: Any,
    scenarios: list[tuple[BranchFailure, ...]],
    output_dir: str | Path,
    *,
    system_name: str,
    base_pmus: list[int] | None = None,
    seed: int = 42,
) -> DatasetBuildResult:
    """Build intact and faulted samples with replayable scenario metadata."""
    root = Path(output_dir).resolve()
    samples_dir = root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    intact_graph = build_graph(net)
    if base_pmus is None:
        base_pmus = greedy_pmu_placement(intact_graph)
    base_pmus = sorted(int(bus) for bus in base_pmus)
    base_coverage = coverage_percent(intact_graph, base_pmus)

    rows: list[dict[str, object]] = []
    intact_arrays = graph_to_arrays(intact_graph, net, base_pmus)
    intact_path = samples_dir / "intact.npz"
    intact_graph_y = np.asarray(
        [base_coverage, base_coverage, 0.0, nx.number_connected_components(intact_graph)],
        dtype=np.float32,
    )
    _save_sample(
        intact_path,
        intact_arrays,
        graph_y=intact_graph_y,
        scenario_type="intact",
        serialized_failures="[]",
    )
    rows.append(
        _scenario_row(
            sample_id="intact",
            relative_path=intact_path.relative_to(root).as_posix(),
            scenario_type="intact",
            serialized_failures="[]",
            fixed_coverage=base_coverage,
            reoptimized_coverage=base_coverage,
            base_pmu_count=len(base_pmus),
            optimized_pmu_count=len(base_pmus),
            components=nx.number_connected_components(intact_graph),
            greedy_time_s=0.0,
        )
    )

    for scenario_number, failures in enumerate(scenarios, start=1):
        serialized = failures_json(failures)
        faulted_net = apply_failures(net, failures)
        faulted_graph = build_graph(faulted_net)
        fixed_coverage = coverage_percent(faulted_graph, base_pmus)
        start = time.perf_counter()
        optimized_pmus = greedy_pmu_placement(faulted_graph)
        greedy_time_s = time.perf_counter() - start
        reoptimized_coverage = coverage_percent(faulted_graph, optimized_pmus)
        components = nx.number_connected_components(faulted_graph)

        sample_id = f"fault_{scenario_number:05d}"
        sample_path = samples_dir / f"{sample_id}.npz"
        arrays = graph_to_arrays(faulted_graph, faulted_net, optimized_pmus)
        graph_y = np.asarray(
            [
                fixed_coverage,
                reoptimized_coverage,
                len(optimized_pmus) - len(base_pmus),
                components,
            ],
            dtype=np.float32,
        )
        _save_sample(
            sample_path,
            arrays,
            graph_y=graph_y,
            scenario_type="faulted",
            serialized_failures=serialized,
        )
        rows.append(
            _scenario_row(
                sample_id=sample_id,
                relative_path=sample_path.relative_to(root).as_posix(),
                scenario_type="faulted",
                serialized_failures=serialized,
                fixed_coverage=fixed_coverage,
                reoptimized_coverage=reoptimized_coverage,
                base_pmu_count=len(base_pmus),
                optimized_pmu_count=len(optimized_pmus),
                components=components,
                greedy_time_s=greedy_time_s,
            )
        )

    index_path = root / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "schema_version": 1,
        "system": system_name,
        "seed": seed,
        "num_samples": len(rows),
        "num_faulted_samples": len(scenarios),
        "base_pmus": base_pmus,
        "node_features": list(NODE_FEATURES),
        "edge_features": list(EDGE_FEATURES),
        "graph_targets": list(GRAPH_TARGETS),
        "edge_convention": "Each physical undirected branch is stored in both directions.",
        "scenario_replay": "Apply every component in failures_json to the original network.",
    }
    metadata_path = root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return DatasetBuildResult(root, index_path, metadata_path, len(rows))

