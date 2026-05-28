"""
Ablation study for multi-task DDI prediction.
Variants: GNN-only -no PK, PK-only - no graph, single-task -severity only.
Training only. Run ablation_eval.py for metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch.nn as nn
from rdkit import RDLogger

from config import (
    ProjectConfig, preferred_device, seed_everything, flush_device_cache,
)
from gnn import SharedEncoder, PKBranch, DDIModel, PairClassifier, fused_graph
from train_gnn import train_batches, ddi_step, optimizer_bundle, fit

RDLogger.DisableLog("rdApp.*")
cfg = ProjectConfig()


# Model variants

class GNNOnly(nn.Module):
    """DDIModel without pharmacokinetic branch."""

    def __init__(self, atom_cfg, gnn_cfg):
        super().__init__()
        self.encoder = SharedEncoder(atom_cfg, gnn_cfg)
        self.classifier = PairClassifier(self.encoder.out_dim, gnn_cfg)

    def forward(self, batch_a, batch_b, *_):
        mol_a, mol_b = fused_graph(self.encoder, batch_a, batch_b)
        return self.classifier(mol_a, mol_b)


class PKOnly(nn.Module):
    """Interaction prediction from pharmacokinetic features only."""

    def __init__(self, gnn_cfg):
        super().__init__()
        self.pk_branch = PKBranch(gnn_cfg.pk_dim, gnn_cfg.pk_embed,
                                  gnn_cfg.dropout)
        self.classifier = PairClassifier(gnn_cfg.pk_embed, gnn_cfg)

    def forward(self, pk_a, pk_b):
        return self.classifier(self.pk_branch(pk_a), self.pk_branch(pk_b))


# Step function for PK-only variant

def pk_step(model, batch, device):
    """Forward step using only PK features."""
    preds = model(batch["pk_a"].to(device), batch["pk_b"].to(device))
    return preds, batch["severity"].to(device), batch["mechanism"].to(device)


# Training

def trainable_params(model, mech_weight):
    """Exclude mechanism_head when mechanism_weight is zero."""
    if mech_weight > 0:
        return model.parameters()
    skip = set(id(p) for p in model.classifier.mechanism_head.parameters())
    return [p for p in model.parameters() if id(p) not in skip]


def variant_specs():
    """Return (tag, model_fn, mech_weight, step_fn) for each ablation."""
    return [
        ("gnn_only", lambda: GNNOnly(cfg.atom, cfg.gnn),
         cfg.train.mechanism_weight, ddi_step),
        ("pk_only", lambda: PKOnly(cfg.gnn),
         cfg.train.mechanism_weight, pk_step),
        ("single_task", lambda: DDIModel(cfg.atom, cfg.gnn),
         0.0, ddi_step),
    ]


def fit_variant(tag, model_fn, mw, step_fn, trn, val, device, sev_w, mech_w):
    """Seed, build, train one ablation variant. Returns best val severity."""
    print(f"\n--- {tag} ---")
    seed_everything(cfg.train.seed)
    model = model_fn().to(device)
    optimizer, scheduler = optimizer_bundle(trainable_params(model, mw))
    best = fit(model, optimizer, scheduler, trn, val, device, sev_w, mech_w,
               cfg.paths.models / f"ablation_{tag}.pt",
               cfg.paths.models / f"ablation_{tag}_resume.pt",
               mech_weight=mw, step_fn=step_fn)
    print(f"  Best val - severity: {best:.4f}")
    del model, optimizer, scheduler
    flush_device_cache(device)


# Entry point

def ablations_ddi(only=None):
    """Train ablation variants and save checkpoints."""
    device = preferred_device(cfg.train.device)
    print(f"Device: {device}")
    cfg.mkdir_tree()
    print("DATA LOADING")
    trn, val, sev_w, mech_w = train_batches(
        cfg.train.batch_size, cfg.train.seed, device)
    print(f"Train: {len(trn.dataset)}, Val: {len(val.dataset)}")

    for tag, model_fn, mw, step_fn in variant_specs():
        if only and tag != only:
            continue
        fit_variant(tag, model_fn, mw, step_fn,
                    trn, val, device, sev_w, mech_w)

    print("\n Complete")


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else None
    if variant and variant not in ("gnn_only", "pk_only", "single_task"):
        print("Usage: python ablation.py [gnn_only|pk_only|single_task]")
        sys.exit(1)
    ablations_ddi(only=variant)