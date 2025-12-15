````md
# Learning-Augmented PMU Placement with Graph Neural Networks

[![CI - C](https://img.shields.io/badge/CI-C-blue)](https://github.com/mojjss/pmu-placement-gnn-first-phase-public/actions)
[![CI - MATLAB](https://img.shields.io/badge/CI-MATLAB-orange)](https://github.com/mojjss/pmu-placement-gnn-first-phase-public/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This project explores **learning-augmented PMU placement** in power networks using **Graph Neural Networks (GNNs)**.

Core workflow:
- Generate “expert” labels with a **classical baseline** (currently greedy observability; **MILP/GA planned**).
- Simulate **intact + faulted** topologies (line/transformer outages) with **pandapower**.
- Export a **PyTorch Geometric (PyG)** dataset.
- Train a **supervised GNN** to predict PMU locations and evaluate **top-K** observability.

**Goal:** keep **full or near-full observability** while reducing optimization/runtime cost and improving robustness under faults.

---

## What’s implemented

### 1) Greedy baseline + robustness
- Greedy PMU placement (observability-based).
- Outage evaluation:
  - `fault_mode="n-1"`: single line/trafo outages (fixed coverage + re-opt via greedy).
  - `fault_mode="random"`: optional random multi-fault scenarios.
- Outputs: CSV metrics, figures, and a **manifest JSON** per run.

### 2) GNN dataset export (NPZ + index.csv)
- Builds a dataset from intact + faulted scenarios (labels from greedy / greedy re-opt).
- Outputs: `samples/*.npz`, `index.csv`, and `dataset_summary.csv`.

### 3) GNN training + top-K evaluation (PyG)
- Node-classification **GCN** for PMU / no-PMU prediction.
- **Top-K inference** with `K = #PMUs(greedy)` (fair comparison) and coverage evaluation.
- Saves curves + checkpoints under `results/`.

---

## Repository entry points

- `Greedy-1.5.*` — baseline, outages, plots, manifest, dataset export  
- `GNN-1.5.*` — dataset loading, training, evaluation, greedy vs GNN top-K

> Notebook-style “cells” (`# %%`). If saved as `.txt`, rename to `.py` for easier execution.

---

## Project layout (typical)

```text
pmu-placement-gnn/
├── Greedy-1.5.*
├── GNN-1.5.*
├── results/
│   ├── figures/
│   ├── metrics/          # baseline/robustness/summary + manifest
│   ├── gnn_dataset/      # NPZ + index.csv
│   ├── figures_gnn/
│   └── gnn_models/
├── requirements.txt
└── README.md
````

---

## Installation

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

python -m pip install --upgrade pip

# PyTorch (CPU example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric

pip install -r requirements.txt
```

---

## Quickstart (end-to-end)

### A) Run greedy baseline + robustness (in `Greedy-1.5.*`)

```python
import pandapower.networks as pn
net = pn.case14()

run_id, manifest = run_suite(
    net,
    system_name="IEEE14",
    fault_mode="n-1",   # or "random"
    top_k=5,
    preview=True,
    run_tag="greedy_n1_14",
)
print("Run ID:", run_id)
```

### B) Export a PyG dataset (still in `Greedy-1.5.*`)

```python
out_root, index_path, summary_path = build_gnn_dataset_from_manifest(
    net,
    manifest,
    max_fault_samples=None,
    extra_random_faults_for_gnn=0,
    max_faults_random=3,
)
print(out_root, index_path, summary_path)
```

```text
results/gnn_dataset/<run_id>/<system>/
├── samples/*.npz
├── index.csv
└── dataset_summary.csv
```

### C) Train + evaluate the GNN (in `GNN-1.5.*`)

* Set dataset paths (e.g., `N1_DATASET_DIRS` / `RANDOM_DATASET_DIRS`)
* Run training to save:

  * curves (`results/figures_gnn/`)
  * checkpoints (`results/gnn_models/`)
  * greedy vs GNN top-K coverage comparisons

---

## Dataset format (concise)

Each `samples/*.npz` includes:

* `x` (`N×5`), `edge_index` (`2×E`), `edge_attr` (`E×6`)
* `y` (`N`, 1=PMU / 0=no-PMU)
* `bus_ids`, `graph_y`, `scenario_type` (`"intact"` / `"faulted"`)

`index.csv` stores per-sample metadata and file pointers.

---

## Outputs

```text
results/
├─ metrics/       # baseline / robustness / summary / manifest
├─ figures/       # baseline + outage plots
├─ gnn_dataset/   # NPZ + index
├─ figures_gnn/   # training curves + comparison visuals
└─ gnn_models/    # .pth checkpoints
```

---

## Roadmap

* Add ILP/MILP and GA baselines.
* Run more IEEE systems (14/30/57/118) and report compact coverage/runtime tables.
* Improve architectures and cross-system generalization.

---

## Citation

Placeholder:

> Sadafi, M. (2025). *Learning-Augmented PMU Placement with Graph Neural Networks.*

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

**Mojtaba Sadafi**
[https://mojsadafi.ir](https://mojsadafi.ir)

```
```

