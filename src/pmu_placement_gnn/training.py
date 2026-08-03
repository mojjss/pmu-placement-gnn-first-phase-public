"""Deterministic loading, training, evaluation, and checkpointing for PMUGCN."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .dataset import NODE_FEATURES
from .model import PMUGCN
from .placement import coverage_aware_selection, neighbors_from_edge_index


@dataclass(frozen=True)
class TrainingConfig:
    """Reproducible training hyperparameters."""

    epochs: int = 150
    batch_size: int = 4
    hidden_channels: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.2
    seed: int = 42
    coverage_weight: float = 0.3
    diversity_weight: float = 0.2
    rate_weight: float = 0.05
    device: str = "auto"


@dataclass(frozen=True)
class TrainingResult:
    """Paths created by :func:`train_model`."""

    output_dir: Path
    checkpoint_path: Path
    history_path: Path
    metrics_path: Path


class NPZGraphDataset(Dataset):
    """Read generated samples directly, avoiding stale processed-cache files."""

    def __init__(self, dataset_dir: str | Path):
        self.root = Path(dataset_dir).resolve()
        index_path = self.root / "index.csv"
        if not index_path.is_file():
            raise FileNotFoundError(f"Dataset index not found: {index_path}")
        with index_path.open(newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"Dataset index is empty: {index_path}")
        self.feature_mean: torch.Tensor | None = None
        self.feature_std: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def _sample_path(self, index: int) -> Path:
        path = (self.root / self.rows[index]["file"]).resolve()
        if self.root not in path.parents:
            raise ValueError(f"Sample path escapes dataset root: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Sample file not found: {path}")
        return path

    def raw_features(self, index: int) -> np.ndarray:
        with np.load(self._sample_path(index), allow_pickle=False) as arrays:
            return arrays["x"].astype(np.float32, copy=True)

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feature_mean = mean.detach().cpu()
        self.feature_std = std.detach().cpu()

    def __getitem__(self, index: int) -> Data:
        row = self.rows[index]
        with np.load(self._sample_path(index), allow_pickle=False) as arrays:
            x = torch.as_tensor(arrays["x"].copy(), dtype=torch.float32)
            if self.feature_mean is not None and self.feature_std is not None:
                x = (x - self.feature_mean) / self.feature_std
            data = Data(
                x=x,
                edge_index=torch.as_tensor(arrays["edge_index"].copy(), dtype=torch.long),
                edge_attr=torch.as_tensor(arrays["edge_attr"].copy(), dtype=torch.float32),
                y=torch.as_tensor(arrays["y"].copy(), dtype=torch.long),
            )
            data.bus_ids = torch.as_tensor(arrays["bus_ids"].copy(), dtype=torch.long)
            data.graph_y = torch.as_tensor(arrays["graph_y"].copy(), dtype=torch.float32)
        data.sample_id = row["sample_id"]
        data.scenario_type = row["scenario_type"]
        return data


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(
    rows: list[dict[str, str]], validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Seeded split that keeps intact samples in training and shuffles faults."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    intact = [index for index, row in enumerate(rows) if row["scenario_type"] == "intact"]
    faulted = [index for index, row in enumerate(rows) if row["scenario_type"] == "faulted"]
    if len(faulted) < 2:
        raise ValueError("At least two faulted samples are required for a train/validation split")

    rng = random.Random(seed)
    rng.shuffle(faulted)
    validation_count = max(1, round(validation_fraction * len(faulted)))
    validation_count = min(validation_count, len(faulted) - 1)
    validation = sorted(faulted[:validation_count])
    training = sorted(intact + faulted[validation_count:])
    return training, validation


def feature_statistics(
    dataset: NPZGraphDataset, indices: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate normalization statistics from training nodes only."""
    features = np.concatenate([dataset.raw_features(index) for index in indices], axis=0)
    mean = torch.as_tensor(features.mean(axis=0), dtype=torch.float32)
    std = torch.as_tensor(features.std(axis=0), dtype=torch.float32)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return mean, std


def class_weights(
    dataset: NPZGraphDataset, indices: list[int], device: torch.device
) -> torch.Tensor:
    """Calculate inverse-frequency class weights from training labels only."""
    labels = torch.cat([dataset[index].y for index in indices])
    counts = torch.bincount(labels, minlength=2).float().clamp_min(1.0)
    weights = counts.sum() / (2.0 * counts)
    return weights.to(device)


def _unique_neighbors(edge_index: torch.Tensor, num_nodes: int) -> list[set[int]]:
    neighbors = [set() for _ in range(num_nodes)]
    for source, target in edge_index.detach().cpu().T.tolist():
        if source != target:
            neighbors[source].add(target)
            neighbors[target].add(source)
    return neighbors


def coverage_penalty(batch: Data, logits: torch.Tensor, weight: float) -> torch.Tensor:
    probabilities = F.softmax(logits, dim=-1)[:, 1]
    one_minus = 1.0 - probabilities
    neighbors = _unique_neighbors(batch.edge_index, len(probabilities))
    unobserved = []
    for node, adjacent in enumerate(neighbors):
        indexes = torch.as_tensor([node, *sorted(adjacent)], device=probabilities.device)
        unobserved.append(one_minus[indexes].prod())
    per_node = torch.stack(unobserved)
    losses = []
    for graph_index in range(batch.num_graphs):
        start = int(batch.ptr[graph_index])
        end = int(batch.ptr[graph_index + 1])
        losses.append(per_node[start:end].mean())
    return weight * torch.stack(losses).mean()


def diversity_penalty(batch: Data, logits: torch.Tensor, weight: float) -> torch.Tensor:
    probabilities = F.softmax(logits, dim=-1)[:, 1]
    sources, targets = batch.edge_index
    keep = sources < targets
    if not bool(keep.any()):
        return logits.new_tensor(0.0)
    return weight * (probabilities[sources[keep]] * probabilities[targets[keep]]).mean()


def pmu_rate_penalty(batch: Data, logits: torch.Tensor, weight: float) -> torch.Tensor:
    probabilities = F.softmax(logits, dim=-1)[:, 1]
    losses = []
    for graph_index in range(batch.num_graphs):
        start = int(batch.ptr[graph_index])
        end = int(batch.ptr[graph_index + 1])
        predicted_rate = probabilities[start:end].mean()
        target_rate = batch.y[start:end].float().mean()
        losses.append((predicted_rate - target_rate).square())
    return weight * torch.stack(losses).mean()


def _run_epoch(
    model: PMUGCN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: TrainingConfig,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        if optimizer is not None:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(batch.x, batch.edge_index)
            loss = criterion(logits, batch.y)
            if training:
                loss = (
                    loss
                    + coverage_penalty(batch, logits, config.coverage_weight)
                    + diversity_penalty(batch, logits, config.diversity_weight)
                    + pmu_rate_penalty(batch, logits, config.rate_weight)
                )
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach()) * batch.num_nodes
        total_correct += int((logits.argmax(dim=-1) == batch.y).sum())
        total_nodes += batch.num_nodes
    return total_loss / total_nodes, total_correct / total_nodes


@torch.no_grad()
def _coverage_metrics(
    model: PMUGCN,
    dataset: NPZGraphDataset,
    indices: list[int],
    device: torch.device,
):
    model.eval()
    label_coverages = []
    prediction_coverages = []
    for index in indices:
        data = dataset[index]
        logits = model(data.x.to(device), data.edge_index.to(device))
        probabilities = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        k = int(data.y.sum())
        predicted = coverage_aware_selection(
            probabilities,
            data.edge_index.numpy(),
            k,
            candidate_limit=max(2 * k, 80),
        )
        neighbors = neighbors_from_edge_index(data.edge_index.numpy(), data.num_nodes)

        labels = torch.nonzero(data.y, as_tuple=True)[0].tolist()
        label_coverages.append(_coverage_from_neighbors(neighbors, labels))
        prediction_coverages.append(_coverage_from_neighbors(neighbors, predicted))

    predicted_array = np.asarray(prediction_coverages)
    return {
        "validation_graphs": len(indices),
        "mean_label_coverage_pct": float(np.mean(label_coverages)),
        "mean_predicted_coverage_pct": float(np.mean(prediction_coverages)),
        "predicted_full_coverage_pct": float(np.mean(predicted_array >= 100.0 - 1e-9) * 100.0),
    }


def _coverage_from_neighbors(neighbors: list[set[int]], nodes: list[int]) -> float:
    observed = set(nodes)
    for node in nodes:
        observed.update(neighbors[node])
    return 100.0 * len(observed) / len(neighbors)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train_model(
    dataset_dir: str | Path,
    output_dir: str | Path,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train, validate, and save a self-describing checkpoint."""
    config = config or TrainingConfig()
    if config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    _seed_everything(config.seed)
    device = _resolve_device(config.device)
    dataset = NPZGraphDataset(dataset_dir)
    train_indices, validation_indices = split_indices(
        dataset.rows, config.validation_fraction, config.seed
    )
    mean, std = feature_statistics(dataset, train_indices)
    dataset.set_normalization(mean, std)

    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices), batch_size=config.batch_size, shuffle=False
    )
    model = PMUGCN(len(NODE_FEATURES), config.hidden_channels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(dataset, train_indices, device))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    history = []
    best_validation_loss = float("inf")
    best_state = None
    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = _run_epoch(
            model, train_loader, criterion, device, config, optimizer
        )
        validation_loss, validation_accuracy = _run_epoch(
            model, validation_loader, criterion, device, config, None
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    metrics = _coverage_metrics(model, dataset, validation_indices, device)
    metrics.update(
        {
            "best_validation_loss": best_validation_loss,
            "best_validation_accuracy": max(row["validation_accuracy"] for row in history),
            "training_samples": len(train_indices),
            "validation_samples": len(validation_indices),
            "device": str(device),
        }
    )

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    checkpoint_path = output / "checkpoint.pt"
    torch.save(
        {
            "schema_version": 1,
            "state_dict": best_state,
            "model": {"in_channels": len(NODE_FEATURES), "hidden_channels": config.hidden_channels},
            "feature_names": list(NODE_FEATURES),
            "feature_mean": mean,
            "feature_std": std,
            "training_config": asdict(config),
            "metrics": metrics,
        },
        checkpoint_path,
    )
    return TrainingResult(output, checkpoint_path, history_path, metrics_path)
