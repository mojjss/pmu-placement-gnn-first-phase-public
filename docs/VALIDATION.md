# Validation record

Date: 2026-08-03  
Platform: Windows, Python 3.12.13

This record documents a clean implementation check. The short training run is a
software smoke test, not a benchmark result.

## Environment

- NumPy 2.4.6
- pandas 2.3.3
- NetworkX 3.6.1
- pandapower 3.5.4
- Matplotlib 3.11.1
- PyTorch 2.9.1+cpu
- PyTorch Geometric 2.8.0.post1

`python -m pip check` reported no broken requirements.

## Automated checks

- `pytest -q`: 8 passed.
- `ruff check src tests`: passed.
- `python -m build`: source distribution and wheel built successfully.
- All notebooks remained valid JSON after metadata-only wording edits.

## IEEE-14 end-to-end checks

N-1 command:

```text
pmu-gnn greedy --system IEEE14 --fault-mode n-1 --seed 42 --tag validation
```

Observed contract:

- 14 nodes and 20 physical intact branches.
- 40 directed PyG edges because every branch is stored in both directions.
- 20 faulted samples plus one intact sample.
- Node feature shape `(14, 5)` and graph-target shape `(4,)`.
- Greedy intact PMUs `[0, 3, 5, 6, 8]`, matching the selected historical output.
- Manifest artifact paths were relative.

A five-scenario random run with seed 42 was regenerated twice. The complete
failure sets matched, and their total failure counts were `[1, 2, 1, 1, 2]`.

## GNN smoke training

The generated N-1 dataset was trained for two CPU epochs with 16 hidden channels,
batch size 4, and seed 42.

- Training samples: 17.
- Validation samples: 4.
- History, metrics, and checkpoint files were written.
- The checkpoint reloaded into `PMUGCN` with its recorded feature order and
  normalization vectors.
- Coverage-aware top-K decoding completed for every validation graph.

