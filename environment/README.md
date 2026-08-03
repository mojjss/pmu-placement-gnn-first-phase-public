# Release-validation environment

This directory records the exact environment used to validate version 0.1.0.
It supplements the portable dependency ranges in `pyproject.toml`; it is not a
requirement that other supported environments use these exact transitive
versions.

## Recorded platform

- Date: 2026-08-03
- Operating system: Windows 11, build 26200, AMD64
- Python: 3.12.13
- pip: 25.0.1
- Release version: 0.1.0
- Reproduction seed: 42

## Reconstruct the validation environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install pip==25.0.1
python -m pip install -r environment/pip-freeze-windows-py312.txt `
  --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . --no-deps
```

The frozen file was generated with:

```powershell
python -m pip freeze --exclude-editable
```

## Validation commands

```powershell
python -m pip check
pytest -q
ruff check .
python -m build
pmu-gnn greedy --system IEEE14 --fault-mode n-1 --seed 42 `
  --tag release-validation
```

The IEEE-14 command is expected to produce 21 samples: one intact network and
20 single-component outage scenarios.
