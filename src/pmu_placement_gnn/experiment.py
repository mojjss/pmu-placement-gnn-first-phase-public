"""High-level, headless workflows used by the command-line interface."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from .dataset import DatasetBuildResult, build_dataset
from .power_network import generate_failure_scenarios

SUPPORTED_SYSTEMS = ("IEEE14", "IEEE57", "IEEE118", "IEEE300")


@dataclass(frozen=True)
class ExperimentResult:
    """Files created by an end-to-end greedy dataset run."""

    run_id: str
    run_dir: Path
    manifest_path: Path
    dataset: DatasetBuildResult


def load_test_system(system_name: str):
    """Load one of the supported pandapower IEEE test systems."""
    try:
        import pandapower.networks as pn
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("pandapower is required for experiment generation") from exc

    factories = {
        "IEEE14": pn.case14,
        "IEEE57": pn.case57,
        "IEEE118": pn.case118,
        "IEEE300": pn.case300,
    }
    try:
        return factories[system_name.upper()]()
    except KeyError as exc:
        message = f"Unsupported system {system_name!r}; choose from {SUPPORTED_SYSTEMS}"
        raise ValueError(message) from exc


def make_run_id(tag: str | None = None) -> str:
    """Create a filesystem-safe UTC run identifier."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if not tag:
        return f"RUN_{timestamp}"
    safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", tag).strip("-.")
    return f"RUN_{timestamp}_{safe_tag}" if safe_tag else f"RUN_{timestamp}"


def run_experiment(
    *,
    system_name: str,
    output_root: str | Path = "results",
    fault_mode: Literal["n-1", "random"] = "n-1",
    num_random_scenarios: int = 50,
    max_random_failures: int = 3,
    seed: int = 42,
    tag: str | None = None,
) -> ExperimentResult:
    """Generate a complete greedy baseline and replayable GNN dataset."""
    canonical_system = system_name.upper()
    net = load_test_system(canonical_system)
    scenarios = generate_failure_scenarios(
        net,
        mode=fault_mode,
        num_random_scenarios=num_random_scenarios,
        max_random_failures=max_random_failures,
        seed=seed,
    )
    run_id = make_run_id(tag)
    run_dir = Path(output_root).resolve() / run_id / canonical_system
    dataset_result = build_dataset(
        net,
        scenarios,
        run_dir / "dataset",
        system_name=canonical_system,
        seed=seed,
    )

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "system": canonical_system,
        "configuration": {
            "fault_mode": fault_mode,
            "num_random_scenarios": num_random_scenarios if fault_mode == "random" else None,
            "max_random_failures": max_random_failures if fault_mode == "random" else None,
            "seed": seed,
        },
        "artifacts": {
            "dataset_index": dataset_result.index_path.relative_to(run_dir).as_posix(),
            "dataset_metadata": dataset_result.metadata_path.relative_to(run_dir).as_posix(),
        },
        "counts": {"samples": dataset_result.sample_count, "fault_scenarios": len(scenarios)},
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return ExperimentResult(run_id, run_dir, manifest_path, dataset_result)


def result_as_json(result: ExperimentResult) -> str:
    """Serialize CLI output without leaking platform-specific object types."""
    payload = asdict(result)
    payload["run_dir"] = str(result.run_dir)
    payload["manifest_path"] = str(result.manifest_path)
    payload["dataset"] = {
        "root": str(result.dataset.root),
        "index_path": str(result.dataset.index_path),
        "metadata_path": str(result.dataset.metadata_path),
        "sample_count": result.dataset.sample_count,
    }
    return json.dumps(payload, indent=2)
