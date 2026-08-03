# PMU Placement with Graph Neural Networks

[![CI](https://github.com/mojjss/pmu-placement-gnn-first-phase-public/actions/workflows/ci.yml/badge.svg)](https://github.com/mojjss/pmu-placement-gnn-first-phase-public/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21765781.svg)](https://doi.org/10.5281/zenodo.21765781)


This repository is a reproducible research-software project for learning-augmented
Phasor Measurement Unit (PMU) placement. It provides a deterministic greedy
baseline, intact and faulted IEEE test-system datasets, and a PyTorch Geometric
GCN that predicts node-level PMU placements.

The historical notebooks remain available for traceability. The reusable and
portable implementation in `src/pmu_placement_gnn/` is the canonical entry point
for new runs.

This software supports an ongoing research manuscript. Publication citation
details will be added when they become publicly available.

## What is implemented

- Deterministic greedy PMU placement under a one-hop topological observability model.
- N-1 line, two-winding-transformer, and three-winding-transformer outages.
- Seeded, replayable random multi-fault scenarios.
- Portable compressed NPZ datasets with node labels, graph targets, and an index.
- A two-layer PyTorch Geometric GCN with class balancing and coverage-aware losses.
- Coverage-aware top-K decoding and node-feature normalization from training data only.
- Tests and continuous integration for the portable core.

The greedy method is a heuristic and does not certify a minimum PMU count. The
observability model is graph-based; it is not a replacement for an electrical
state-estimation or zero-injection-bus observability study.

## Repository layout

```text
.
├── src/pmu_placement_gnn/  # reusable Python package and CLI
├── tests/                  # regression and data-contract tests
├── notebooks/              # historical exploratory workflows
├── Example results/        # selected outputs from notebook runs
├── docs/                   # methods and reproducibility guidance
├── environment/            # frozen validation environment
├── CITATION.cff            # machine-readable software citation
├── pyproject.toml          # package metadata and dependencies
└── requirements.txt        # full local research environment
```

See [the notebook-to-module map](docs/NOTEBOOK_MAP.md) for the direct relationship
between the older cells and the package.

## Installation

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[ml,notebooks]"
```

Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[ml,notebooks]"
```

`torchvision` and `torchaudio` are not required. GPU users should choose the
PyTorch command appropriate for their CUDA environment before installing the
project extras.

## Quickstart

Generate deterministic IEEE-14 N-1 labels and a GNN dataset:

```bash
pmu-gnn greedy --system IEEE14 --fault-mode n-1 --seed 42 --tag ieee14-n1
```

The command prints the exact run directory. A typical output is:

```text
results/RUN_<UTC>_ieee14-n1/IEEE14/
├── manifest.json
└── dataset/
    ├── index.csv
    ├── metadata.json
    └── samples/*.npz
```

Generate 50 unique random multi-fault scenarios containing at most three total
failed physical branches:

```bash
pmu-gnn greedy --system IEEE14 --fault-mode random \
  --random-scenarios 50 --max-random-failures 3 --seed 42
```

Train the GCN after installing the `ml` extra:

```bash
pmu-gnn train \
  --dataset results/RUN_<UTC>_ieee14-n1/IEEE14/dataset \
  --output results/training/IEEE14/run-01 \
  --epochs 150 --seed 42
```

Every checkpoint records its feature order, training-only normalization
statistics, hyperparameters, and validation metrics.

## Python API

```python
from pmu_placement_gnn.dataset import build_dataset
from pmu_placement_gnn.power_network import generate_failure_scenarios
from pmu_placement_gnn.experiment import load_test_system

net = load_test_system("IEEE14")
scenarios = generate_failure_scenarios(net, mode="n-1", seed=42)
result = build_dataset(
    net,
    scenarios,
    "results/manual/IEEE14/dataset",
    system_name="IEEE14",
    seed=42,
)
print(result.index_path)
```

## Dataset contract

Each sample contains:

- `x`: node features in the order recorded by `metadata.json`.
- `edge_index`: shape `2 x E`; each physical undirected branch appears in both directions.
- `edge_attr`: branch features aligned with `edge_index`.
- `y`: node labels where `1` denotes a greedy reference PMU.
- `bus_ids`: mapping from array position to the original pandapower bus ID.
- `graph_y`: fixed coverage, re-optimized coverage, PMU-count change, and components.
- `scenario_type` and `failures_json`: provenance needed to replay the topology.

Parallel branches are preserved. For a random scenario, `failures_json` contains
the complete failure set rather than only its final outage.

## Verification

Install developer tools and run:

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
python -m build
```

For end-to-end checks and the status of historical artifacts, see
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).
The exact Windows/Python 3.12 environment used for release validation is
recorded in [`environment/`](environment/README.md).

## Citation and archival releases

Use the repository's **Cite this repository** control, generated from
[`CITATION.cff`](CITATION.cff). 

## Contributing and support

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening
a pull request. Report security-sensitive problems using [SECURITY.md](SECURITY.md)
rather than a public issue.

Apache License 2.0. See [LICENSE](LICENSE).

Author: [Mojtaba Sadafi](https://mojsadafi.ir)
