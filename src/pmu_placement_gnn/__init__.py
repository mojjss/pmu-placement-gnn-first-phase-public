"""Reusable PMU placement and GNN research workflows."""

from .placement import coverage_aware_selection, coverage_percent, greedy_pmu_placement
from .power_network import BranchFailure, build_graph, generate_failure_scenarios

__all__ = [
    "BranchFailure",
    "build_graph",
    "coverage_aware_selection",
    "coverage_percent",
    "generate_failure_scenarios",
    "greedy_pmu_placement",
]

__version__ = "0.1.0"

