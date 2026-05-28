"""
Project configuration for multi-task DDI prediction.
Dataclasses for paths, model, training, and data parameters.
"""

import gc
import os
import torch
import numpy as np
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent


def cpu_workers() -> int:
    return max(1, min((os.cpu_count() or 4) // 2, 8))


def preferred_device(preference: str = "auto") -> torch.device:
    """Select compute device. 'auto' probes MPS > CUDA > CPU."""
    if preference != "auto":
        return torch.device(preference)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int):
    """Reproducibility across CPU, MPS, CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if os.environ.get("TORCH_DETERMINISTIC") == "1":
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            pass


def sync_device(device):
    """Flush device command buffer before checkpoint writes."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def flush_device_cache(device):
    """Force garbage collection and empty device cache."""
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def atomic_write(obj, path):
    """Write torch object via temp file to prevent corruption."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.rename(path)


@dataclass
class Paths:
    root: Path = ROOT

    @property
    def raw(self) -> Path:
        return self.root / "Datasets" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "Datasets" / "processed"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def metrics(self) -> Path:
        return self.root / "results" / "metrics"

    @property
    def plots(self) -> Path:
        return self.root / "results" / "plots"


@dataclass
class AtomConfig:
    features: int = 49
    ring_norm: float = 8.0
    mass_norm: float = 100.0


@dataclass
class GNNConfig:
    hidden_dim: int = 128
    num_layers: int = 3
    heads: int = 4
    dropout: float = 0.2
    edge_dropout: float = 0.15
    pool: str = "mean_max"
    edge_dim: int = 14
    jk_mode: str = "max"
    pk_dim: int = 10
    pk_embed: int = 64
    interaction_dim: int = 512
    severity_classes: int = 4
    mechanism_classes: int = 7


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    epochs: int = 80
    patience: int = 15
    cosine_eta_min: float = 1e-6
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"
    num_workers: int = field(default_factory=cpu_workers)
    checkpoint_every: int = 5
    mechanism_weight: float = 0.5
    neg_ratio: float = 1.0
    neg_oversample: int = 2


@dataclass
class BaselineConfig:
    fp_radius: int = 2
    fp_bits: int = 2048
    cv_folds: int = 5
    rf_trees: int = 300
    rf_depth: int = 20
    mlp_hidden: tuple = (512, 256)
    mlp_epochs: int = 100
    xgb_trees: int = 500
    xgb_depth: int = 6
    xgb_lr: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample: float = 0.6
    mech_rf_trees: int = 200
    mech_rf_depth: int = 15
    mech_mlp_hidden: tuple = (256, 128)
    mech_mlp_epochs: int = 80
    mech_xgb_trees: int = 300
    mech_xgb_depth: int = 6
    mech_xgb_lr: float = 0.05


@dataclass
class DataConfig:
    severity_map: dict = field(default_factory=lambda: {
        "none": 0, "Minor": 1, "Moderate": 2, "Major": 3,
    })
    mechanism_cols: List[str] = field(default_factory=lambda: [
        "cyp_induction", "cyp_inhibition", "qt_prolongation",
        "additive_toxicity", "absorption_interference", "protein_binding",
        "excretion",
    ])


@dataclass
class ProjectConfig:
    paths: Paths = field(default_factory=Paths)
    atom: AtomConfig = field(default_factory=AtomConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)

    def mkdir_tree(self):
        for d in [self.paths.processed, self.paths.models,
                  self.paths.metrics, self.paths.plots]:
            d.mkdir(parents=True, exist_ok=True)