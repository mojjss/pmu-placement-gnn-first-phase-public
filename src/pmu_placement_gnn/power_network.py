"""Pandapower-to-graph conversion and reproducible branch-fault scenarios."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Literal

import networkx as nx
import numpy as np

FailureKind = Literal["line", "trafo", "trafo3w"]


@dataclass(frozen=True, order=True)
class BranchFailure:
    """A physical network component that is set out of service."""

    kind: FailureKind
    idx: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _in_service(row: Any) -> bool:
    value = row.get("in_service", True)
    return bool(value)


def _active_indices(table: Any) -> list[int]:
    if table is None or len(table) == 0:
        return []
    return [int(idx) for idx, row in table.iterrows() if _in_service(row)]


def build_graph(
    net: Any,
    *,
    include_transformers: bool = True,
    include_three_winding: bool = True,
) -> nx.MultiGraph:
    """Build a bus multigraph while preserving parallel physical branches."""
    graph = nx.MultiGraph()

    for bus, row in net.bus.iterrows():
        if not _in_service(row):
            continue
        graph.add_node(int(bus), voltage_level=float(row.get("vn_kv", 0.0)))

    for idx, row in net.line.iterrows():
        if not _in_service(row):
            continue
        u = int(row["from_bus"])
        v = int(row["to_bus"])
        if u not in graph or v not in graph:
            continue
        graph.add_edge(
            u,
            v,
            key=f"line:{int(idx)}",
            kind="line",
            idx=int(idx),
            length=float(row.get("length_km", 0.0)),
            resistance=float(row.get("r_ohm_per_km", 0.0)),
            sn_mva=0.0,
            impedance=0.0,
        )

    if include_transformers and hasattr(net, "trafo"):
        for idx, row in net.trafo.iterrows():
            if not _in_service(row):
                continue
            u = int(row["hv_bus"])
            v = int(row["lv_bus"])
            if u not in graph or v not in graph:
                continue
            graph.add_edge(
                u,
                v,
                key=f"trafo:{int(idx)}",
                kind="trafo",
                idx=int(idx),
                length=0.0,
                resistance=0.0,
                sn_mva=float(row.get("sn_mva", 0.0)),
                impedance=float(row.get("vk_percent", 0.0)),
            )

    if include_three_winding and hasattr(net, "trafo3w"):
        for idx, row in net.trafo3w.iterrows():
            if not _in_service(row):
                continue
            buses = [int(row[name]) for name in ("hv_bus", "mv_bus", "lv_bus")]
            if any(bus not in graph for bus in buses):
                continue
            sn_mva = float(
                max(row.get("sn_hv_mva", 0.0), row.get("sn_mv_mva", 0.0), row.get("sn_lv_mva", 0.0))
            )
            impedance = float(
                max(
                    row.get("vk_hv_percent", 0.0),
                    row.get("vk_mv_percent", 0.0),
                    row.get("vk_lv_percent", 0.0),
                )
            )
            for pair_number, (u, v) in enumerate(combinations(buses, 2)):
                graph.add_edge(
                    u,
                    v,
                    key=f"trafo3w:{int(idx)}:{pair_number}",
                    kind="trafo3w",
                    idx=int(idx),
                    length=0.0,
                    resistance=0.0,
                    sn_mva=sn_mva,
                    impedance=impedance,
                )

    return graph


def available_failures(net: Any) -> list[BranchFailure]:
    """List active physical branches in deterministic order."""
    failures = [BranchFailure("line", idx) for idx in _active_indices(net.line)]
    if hasattr(net, "trafo"):
        failures.extend(BranchFailure("trafo", idx) for idx in _active_indices(net.trafo))
    if hasattr(net, "trafo3w"):
        failures.extend(BranchFailure("trafo3w", idx) for idx in _active_indices(net.trafo3w))
    return sorted(failures)


def apply_failures(net: Any, failures: tuple[BranchFailure, ...] | list[BranchFailure]) -> Any:
    """Return a deep-copied network with the exact failure set applied."""
    faulted = copy.deepcopy(net)
    for failure in failures:
        if not hasattr(faulted, failure.kind):
            raise ValueError(f"Network has no component table {failure.kind!r}")
        table = getattr(faulted, failure.kind)
        if failure.idx not in table.index:
            raise ValueError(f"Unknown {failure.kind} index {failure.idx}")
        table.at[failure.idx, "in_service"] = False
    return faulted


def failures_json(failures: tuple[BranchFailure, ...] | list[BranchFailure]) -> str:
    """Serialize a failure set in a stable, replayable form."""
    return json.dumps([failure.to_dict() for failure in sorted(failures)], separators=(",", ":"))


def generate_failure_scenarios(
    net: Any,
    *,
    mode: Literal["n-1", "random"] = "n-1",
    num_random_scenarios: int = 50,
    max_random_failures: int = 3,
    seed: int = 42,
) -> list[tuple[BranchFailure, ...]]:
    """Generate deterministic N-1 or unique random multi-fault scenarios."""
    components = available_failures(net)
    if mode == "n-1":
        return [(failure,) for failure in components]
    if mode != "random":
        raise ValueError("mode must be 'n-1' or 'random'")
    if num_random_scenarios <= 0:
        raise ValueError("num_random_scenarios must be positive")
    if not components:
        raise ValueError("Network contains no active branches")

    max_count = min(max_random_failures, len(components))
    if max_count <= 0:
        raise ValueError("max_random_failures must be positive")
    possible = sum(math.comb(len(components), count) for count in range(1, max_count + 1))
    if num_random_scenarios > possible:
        raise ValueError(
            f"Requested {num_random_scenarios} unique scenarios, but only {possible} are possible"
        )

    rng = np.random.default_rng(seed)
    scenarios: list[tuple[BranchFailure, ...]] = []
    seen: set[tuple[BranchFailure, ...]] = set()
    while len(scenarios) < num_random_scenarios:
        count = int(rng.integers(1, max_count + 1))
        positions = rng.choice(len(components), size=count, replace=False)
        scenario = tuple(sorted(components[int(position)] for position in positions))
        if scenario not in seen:
            seen.add(scenario)
            scenarios.append(scenario)
    return scenarios
