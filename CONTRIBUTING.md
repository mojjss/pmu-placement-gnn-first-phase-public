# Contributing

Thank you for improving this project.

## Development setup

1. Create and activate a Python 3.11 or 3.12 virtual environment.
2. Install the developer environment with `python -m pip install -e ".[dev]"`.
3. Run `pytest`, `ruff check .`, and `python -m build` before submitting changes.

Install the `ml` extra when changing the model or training path. Select a suitable
PyTorch CPU or CUDA wheel first, then run `python -m pip install -e ".[ml,dev]"`.

## Change guidelines

- Keep random behavior behind an explicit seed.
- Preserve the dataset schema or increment its `schema_version`.
- Add a regression test for algorithm or data-contract fixes.
- Keep paths relative in committed manifests and examples.
- Do not commit generated checkpoints or large raw datasets.
- Explain any numerical changes to example outputs.

Open an issue before a large behavioral change so the scope and validation plan
can be agreed in advance.

