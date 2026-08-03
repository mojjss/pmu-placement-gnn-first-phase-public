# Notebook-to-module map

The notebooks are retained as historical, output-bearing workflows. New runs
should use the package because it removes machine-specific state and duplicated
cells.

| Notebook responsibility | Canonical module |
| --- | --- |
| Pandapower graph construction | `power_network.py` |
| Greedy placement and observability | `placement.py` |
| N-1 and random fault creation | `power_network.py` |
| NPZ sample and index generation | `dataset.py` |
| Placement figures and optional EMF export | `plotting.py` |
| `PMUGCN` model | `model.py` |
| Dataset loading, losses, split, training, evaluation | `training.py` |
| Complete IEEE-system run | `experiment.py` and `cli.py` |

The four system-specific Greedy notebooks share the same implementation and vary
mainly in system configuration. The four GNN 1.7 notebooks likewise duplicate the
same model and evaluation logic. The package expresses those variations as CLI
arguments instead of copied source.

## Deliberate corrections

- Random mode now samples a maximum number of failures across all component
  types, not that many lines plus that many transformers.
- The full random failure set is saved and replayed when a dataset sample is built.
- A multigraph retains parallel branches.
- PyG edges are explicitly bidirectional.
- Greedy and decoding ties are deterministic.
- The intact and faulted `graph_y` arrays use one consistent four-value schema.
- Training/validation splits are seeded and shuffled; intact graphs stay in training.
- Feature normalization and class weights use training data only.
