from __future__ import annotations

from pmu_placement_gnn.power_network import (
    BranchFailure,
    apply_failures,
    build_graph,
    failures_json,
    generate_failure_scenarios,
)


def test_build_graph_preserves_parallel_branches(fake_net):
    graph = build_graph(fake_net)

    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 4
    assert graph.number_of_edges(0, 1) == 2


def test_random_scenarios_are_reproducible_and_count_total_failures(fake_net):
    first = generate_failure_scenarios(
        fake_net,
        mode="random",
        num_random_scenarios=10,
        max_random_failures=3,
        seed=7,
    )
    second = generate_failure_scenarios(
        fake_net,
        mode="random",
        num_random_scenarios=10,
        max_random_failures=3,
        seed=7,
    )

    assert first == second
    assert len(set(first)) == 10
    assert all(1 <= len(scenario) <= 3 for scenario in first)


def test_failure_set_is_replayable(fake_net):
    failures = (BranchFailure("line", 0), BranchFailure("trafo", 0))
    faulted = apply_failures(fake_net, failures)

    assert not bool(faulted.line.at[0, "in_service"])
    assert not bool(faulted.trafo.at[0, "in_service"])
    assert failures_json(failures) == '[{"kind":"line","idx":0},{"kind":"trafo","idx":0}]'

