"""Command-line entry points for reproducible experiments and training."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .experiment import SUPPORTED_SYSTEMS, result_as_json, run_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pmu-gnn", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    greedy = subparsers.add_parser("greedy", help="Build greedy labels and a replayable dataset")
    greedy.add_argument("--system", choices=SUPPORTED_SYSTEMS, default="IEEE14")
    greedy.add_argument("--fault-mode", choices=("n-1", "random"), default="n-1")
    greedy.add_argument("--random-scenarios", type=int, default=50)
    greedy.add_argument("--max-random-failures", type=int, default=3)
    greedy.add_argument("--seed", type=int, default=42)
    greedy.add_argument("--output-root", type=Path, default=Path("results"))
    greedy.add_argument("--tag")

    train = subparsers.add_parser("train", help="Train PMUGCN on a generated dataset")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=150)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--hidden-channels", type=int, default=128)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process status code."""
    args = _parser().parse_args(argv)
    if args.command == "greedy":
        result = run_experiment(
            system_name=args.system,
            output_root=args.output_root,
            fault_mode=args.fault_mode,
            num_random_scenarios=args.random_scenarios,
            max_random_failures=args.max_random_failures,
            seed=args.seed,
            tag=args.tag,
        )
        print(result_as_json(result))
        return 0

    try:
        from .training import TrainingConfig, train_model
    except ImportError as exc:
        raise SystemExit("Training requires: pip install -e \".[ml]\"") from exc

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_channels=args.hidden_channels,
        seed=args.seed,
        device=args.device,
    )
    result = train_model(args.dataset, args.output, config)
    payload = asdict(result)
    print(json.dumps({key: str(value) for key, value in payload.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

