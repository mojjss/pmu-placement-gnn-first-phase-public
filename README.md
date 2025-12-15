
# Learning-Augmented PMU Placement with Graph Neural Networks

This project investigates the use of **Graph Neural Networks (GNNs)** to optimize
**Phasor Measurement Unit (PMU)** placement in electrical power networks.

The core idea is:

- Use **classical algorithms** (currently a greedy observability heuristic,
  with MILP / GA baselines planned) to generate “expert” PMU placements.
- Simulate **healthy and faulted topologies** (line / transformer outages) on
  IEEE test systems using **pandapower**.
- Build a **GNN-ready dataset** from these scenarios.
- Train a **supervised GNN** to predict where PMUs should be placed.

The long–term goal is to reduce sensor count and computational cost while
maintaining full or near-full observability and good robustness under faults.

This codebase supports the research line for:

> *“Learning-Augmented PMU Placement with Graph Neural Networks.”*


---

## 🧠 Current Capabilities

Right now, the repo contains:

- **Greedy PMU placement baseline**
  - Observability-based greedy algorithm on the bus/line graph.
  - Implemented in `greedy.ipynb`.
- **Random fault & robustness analysis**
  - Random line/transformer outages using pandapower.
  - Rebuilds the graph, recomputes coverage, and re-optimizes PMUs.
  - Saves metrics, figures, and a manifest JSON per run.
- **GNN dataset builder**
  - Converts each (network, placement) pair into:
    - Node features `x`
    - Edge index `edge_index`
    - Edge features `edge_attr`
    - Node labels `y` (PMU = 1, no PMU = 0)
    - Graph-level labels (coverage, Δ#PMUs, components, etc.)
  - Stores everything as `.npz` samples plus an `index.csv` for PyG.
- **GCN-based model & training loop (PyTorch Geometric)**
  - Loads the `.npz` samples into a `PMUGNNDataset`.
  - Trains a small GCN as a **node-classification** model:
    “place a PMU here or not”.

**Planned (not yet implemented in code):**

- MILP / ILP baseline for PMU placement (via PuLP or similar).
- Genetic Algorithm (GA) baseline (via DEAP).
- Cleaner separation into `src/` modules and multiple benchmark systems
  (IEEE-14/30/57/118).


---

## 📁 Project Structure (current)

A typical local layout looks like:

```text
pmu-placement-gnn/
├── greedy.ipynb          # Greedy baseline + robustness + GNN dataset builder
├── GNN-1.0.ipynb         # PyTorch Geometric dataset + GCN training
├── results/
│   ├── figures/          # PNG/SVG plots of placements and outages
│   ├── metrics/          # CSVs + summary + manifests
│   │   ├── baseline/
│   │   ├── robustness/
│   │   ├── summary/
│   │   └── manifest/
│   └── gnn_dataset/      # generated .npz samples + index.csv (per run/system)
├── requirements.txt
└── README.md
````

Some folders (like `results/…` and `results/gnn_dataset/…`) are created
automatically when you run the notebooks.

---

## ⚙️ Installation (CPU-only)

```bash
# Create and activate a virtual environment
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1
# (Linux / macOS: source .venv/bin/activate)

# Upgrade pip
pip install --upgrade pip

# Install CPU-only PyTorch first (from the official index)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric (CPU build)
pip install torch-geometric

# Install the rest of the dependencies
pip install -r requirements.txt
```

> If versions in `requirements.txt` conflict with your local PyTorch install,
> prefer the versions suggested by the official PyTorch / PyG docs and then
> adjust `requirements.txt` accordingly.

---

## 🚀 How to Run the Baseline + Build the GNN Dataset

### 1️⃣ Run greedy baseline + robustness (in `greedy.ipynb`)

1. Open **`greedy.ipynb`**.

2. In **Cell 2**, choose a test system, e.g.:

   ```python
   import pandapower.networks as pn

   net118 = pn.case118()
   run_id_118, manifest_118 = run_suite(
       net118,
       system_name="IEEE118",
       top_k=5,
       preview=True,
       run_tag="greedy_randomfault_118",
   )
   print("IEEE118 Run ID:", run_id_118)
   ```

3. This will:

   * Build the graph from the pandapower network.
   * Compute a **baseline greedy PMU placement**.
   * Simulate random **line/transformer faults** and re-optimize PMUs.
   * Save:

     * CSV metrics in `results/metrics/...`
     * PNG/SVG figures in `results/figures/...`
     * A manifest JSON (paths + summary) under
       `results/metrics/.../manifest/manifest_<system>.json`.

### 2️⃣ Build a PyG-ready GNN dataset (still in `greedy.ipynb`)

After `run_suite(...)` finishes, run **Cell 4**:

```python
net_for_gnn_118 = net118
out_root_118, index_path_118 = build_gnn_dataset_from_manifest(
    net_for_gnn_118,
    manifest_118,
    max_fault_samples=None  # or e.g. 200 to subsample faults
)
print("GNN dataset for IEEE118 stored at:", out_root_118)
print("Index file:", index_path_118)
```

This creates a structure like:

```text
results/gnn_dataset/<run_id_118>/IEEE118/
├── samples/
│   ├── intact.npz
│   ├── line_10_42.npz
│   ├── trafo_3_17.npz
│   └── ...
└── index.csv
```

Each `.npz` sample contains:

* `x`          – node features
* `edge_index` – edge connectivity (2 × E)
* `edge_attr`  – edge features
* `y`          – node labels (1 = PMU, 0 = no PMU)
* `bus_ids`    – original bus indices
* `graph_y`    – graph-level targets (e.g., coverage, ΔPMUs, components)
* `scenario_type` – `"intact"` or `"faulted"`

---

## 📦 What’s in Each GNN Sample?

For a graph with **N** buses and **E** branches:

* **Node features** `x ∈ ℝ^{N×5}`

  1. `voltage_level` – nominal bus voltage (kv)
  2. `degree` – graph degree of the bus
  3. `has_load` – 1 if bus has a load, else 0
  4. `has_gen` – 1 if bus has a generator, else 0
  5. `has_ext` – 1 if bus is connected to an external grid

* **Edge index** `edge_index ∈ {0..N-1}^{2×E}`

  * Standard PyG convention: `edge_index[0, e] = u`, `edge_index[1, e] = v`.

* **Edge features** `edge_attr ∈ ℝ^{E×6}`

  1. `length_km`
  2. `r_ohm_per_km`
  3. `sn_mva`
  4. `impedance` (vk%)
  5. `is_line`  (1 if line, else 0)
  6. `is_trafo` (1 if transformer, else 0)

* **Node labels** `y ∈ {0,1}^N`

  * `y[i] = 1` if bus *i* has a PMU in the reference solution
    (greedy baseline for intact network, greedy re-optimization for faulted ones).

* **Graph-level labels** `graph_y`

  * For the intact system: `[coverage%, #PMUs]`.
  * For each faulted case: `[coverage_fixed%, coverage_reopt%, ΔPMUs, #components]`.

These are exactly what **GNN-1.0.ipynb** expects.

---

## 🧠 Train the GNN (in `GNN-1.0.ipynb`)

1. Open **`GNN-1.0.ipynb`**.

2. Point it to the dataset you just built by setting:

   ```python
   GNN_ROOT = r"D:\my_projects\pmu-placement-gnn\results\gnn_dataset"
   GNN_SYSTEM_DIR = r"<run_id_118>\IEEE118"  # replace with your actual run_id
   DATASET_DIR = os.path.join(GNN_ROOT, GNN_SYSTEM_DIR)

   index_csv = os.path.join(DATASET_DIR, "index.csv")
   ```

3. Run **Cell B/C** to instantiate the `PMUGNNDataset` and create
   train/validation splits.

4. Run **Cells D/E** to:

   * Define `PMUGCN` (a two-layer GCN + classifier).
   * Train for a selected number of epochs.
   * Print **node-level training and validation accuracy** each epoch.

You can then experiment with:

* Different hidden sizes, number of layers, and learning rates.
* Training only on intact graphs and evaluating on faulted ones.
* Training on one IEEE system and testing on another.

---

## 🧩 Algorithms

**Currently implemented**

* **Greedy PMU Placement** (observability-based).
* **Random N-1 style outages** (lines/transformers) + greedy re-optimization.
* **GCN (PyTorch Geometric)** for node-wise PMU placement prediction.

**Planned baselines**

* **MILP/ILP formulation** for optimal PMU placement.
* **Genetic Algorithm (GA)** for heuristic optimization.

These baselines will later be used both:

* As **targets** for GNN training (learning from “expert” placements).
* As **comparators** in the final paper (PMU count, coverage, runtime).

---

## 📊 Outputs

Key outputs appear under `results/`:

* `results/metrics/baseline/…`

  * Nodes, edges, components, base PMU set, coverage, runtimes.
* `results/metrics/robustness/…`

  * One row per outage: kind, idx, components, coverage, re-optimized PMUs, ΔPMUs.
* `results/metrics/summary/…`

  * High-level summary per system (robustness %, avg ΔPMUs, runtimes).
* `results/metrics/manifest/manifest_<system>.json`

  * Machine-readable registry of all files for that run.
* `results/figures/…`

  * Base placement plot, “before/after” critical outage plots, top-K outage bar chart.
* `results/gnn_dataset/<run_id>/<system>/…`

  * The actual **GNN training dataset** (NPZ + index.csv).



---

## 📚 Citation

If you use this work in your research, please cite (placeholder):

> Sadafi, M. (2025). *Learning-Augmented PMU Placement with Graph Neural Networks.*

---

## 👨‍💻 Author

**Mojtaba Sadafi**
🌐 [https://mojsadafi.ir](https://mojsadafi.ir)

```
