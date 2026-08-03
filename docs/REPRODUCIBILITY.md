# Reproducibility guide

## Reproduction levels

1. **Core regression tests** verify graph conversion, failure generation, dataset
   contracts, deterministic placement, and decoding.
2. **Dataset reproduction** rebuilds labels and NPZ samples from a named IEEE test
   system and explicit seed.
3. **Model reproduction** trains from a generated dataset and stores all settings
   needed to reload the checkpoint.

## Clean run

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[ml,dev]"

pytest
pmu-gnn greedy --system IEEE14 --fault-mode n-1 --seed 42 --tag reproduction
```

Use the run directory printed by the command:

```bash
pmu-gnn train --dataset <run-dir>/dataset --output results/training/reproduction --seed 42
```

On Linux or macOS, activate with `source .venv/bin/activate`.

## Provenance recorded by the package

- UTC run ID and creation timestamp.
- IEEE system name and fault mode.
- Random seed, requested scenario count, and maximum failures.
- Complete physical failure set for every faulted sample.
- Ordered node, edge, and graph-target feature names.
- Original bus identifiers.
- Model hyperparameters, feature normalization, and validation metrics.

Paths inside new manifests and indexes are relative to the run directory.

## Historical notebooks and examples

The notebook files contain outputs from their original interactive sessions. They
are useful for inspection but are not treated as the current executable API.
Several contain machine-specific paths, and one Greedy IEEE-57 notebook has no
stored execution counts. The selected example manifests also preserve absolute
paths from their originating machine. Those paths are provenance strings, not
portable instructions; use the files' current directory layout or generate a new
run with the package.

## Expected numerical variation

Greedy placements and generated scenario identities are deterministic for a fixed
network and seed. Training can still vary slightly across PyTorch versions,
hardware, and low-level kernels. Compare coverage and aggregate metrics with an
appropriate tolerance instead of expecting byte-identical checkpoints.

Before an archival release, run the tests in a clean environment, record
`python --version`, save `python -m pip freeze`, and attach the exact command and
seed used for each released result.

The environment used to validate version 0.1.0 is preserved in
[`environment/`](../environment/README.md), including the Python version, exact
dependency snapshot, validation commands, and reproduction seed.
