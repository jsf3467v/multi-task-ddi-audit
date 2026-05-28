"""
Training loop for multi-task DDI prediction.
Early stopping uses severity-only val loss across all variants for fairness.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from rdkit import RDLogger
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

from config import (
    ProjectConfig, preferred_device, seed_everything,
    sync_device, flush_device_cache, atomic_write,
)
from gnn import DDIModel, ddi_loss, severity_loss
from feature_engineering import (
    ddi_merged, severity_integers, mechanism_matrix,
    negative_pairs, drug_graphs,
)
from stratify import split_pairs

RDLogger.DisableLog("rdApp.*")
cfg = ProjectConfig()



class DDIPairDataset(Dataset):
    """Drug pair dataset with precomputed lookups."""

    def __init__(self, df, graphs, pk_tensor, pk_index):
        self.graphs = graphs
        self.pk_tensor = pk_tensor
        self.pk_index = pk_index

        valid_set = set(graphs.keys()) & set(pk_index.keys())
        mask = df["drug_a"].isin(valid_set) & df["drug_b"].isin(valid_set)
        self.pairs = df.loc[mask].reset_index(drop=True)

        self.drug_a = self.pairs["drug_a"].values
        self.drug_b = self.pairs["drug_b"].values
        self.sev = severity_integers(self.pairs["severity"]).values
        self.mech = torch.from_numpy(mechanism_matrix(self.pairs["mechanisms"]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.drug_a[idx], self.drug_b[idx]
        return {
            "graph_a": self.graphs[a],
            "graph_b": self.graphs[b],
            "pk_a": self.pk_tensor[self.pk_index[a]],
            "pk_b": self.pk_tensor[self.pk_index[b]],
            "severity": self.sev[idx],
            "mechanism": self.mech[idx],
        }


def pair_collate(batch):
    """Collate DDI pairs into batched tensors."""
    return {
        "batch_a": Batch.from_data_list([b["graph_a"] for b in batch]),
        "batch_b": Batch.from_data_list([b["graph_b"] for b in batch]),
        "pk_a": torch.stack([b["pk_a"] for b in batch]),
        "pk_b": torch.stack([b["pk_b"] for b in batch]),
        "severity": torch.tensor([b["severity"] for b in batch],
                                 dtype=torch.long),
        "mechanism": torch.stack([b["mechanism"] for b in batch]),
    }


# Pipeline

def pin_and_workers(device, num_workers):
    """Device-appropriate DataLoader keyword arguments."""
    if device.type == "mps":
        num_workers = 0
    pin = device.type == "cuda"
    return {"num_workers": num_workers, "pin_memory": pin,
            "persistent_workers": num_workers > 0}


def class_weights(train_df):
    """Severity and mechanism inverse-frequency weights from training split."""
    n_cls = cfg.gnn.severity_classes
    sev_counts = severity_integers(train_df["severity"]).value_counts()
    sev_counts = sev_counts.reindex(range(n_cls), fill_value=1)
    sev_w = (1.0 / sev_counts).values.astype(np.float32)
    sev_w = torch.from_numpy(sev_w / sev_w.sum() * n_cls)

    mech_labels = mechanism_matrix(train_df["mechanisms"])
    pos = mech_labels.sum(axis=0).clip(min=1.0)
    neg = (mech_labels.shape[0] - pos).clip(min=1.0)
    mech_w = torch.from_numpy(np.log1p(neg / pos).astype(np.float32))
    return sev_w, mech_w


def merged_data():
    """Merge DDI pairs with SMILES graphs and PK features."""
    pairs, smiles_df, pk_df = ddi_merged()
    graphs = drug_graphs(smiles_df)
    pk_tensor = torch.from_numpy(pk_df.values.astype(np.float32))
    pk_index = {did: i for i, did in enumerate(pk_df.index)}
    if pk_tensor.shape[1] != cfg.gnn.pk_dim:
        raise ValueError(
            f"pk_features.csv has {pk_tensor.shape[1]} columns but "
            f"cfg.gnn.pk_dim={cfg.gnn.pk_dim}. Update config or regenerate CSV."
        )
    valid_ids = set(graphs.keys()) & set(pk_index.keys())
    pairs = pairs[pairs["drug_a"].isin(valid_ids)
                  & pairs["drug_b"].isin(valid_ids)]
    return pairs, graphs, pk_tensor, pk_index, valid_ids


def sampled_splits(pairs, valid_ids, seed):
    """Negative-sample and three-way stratified split."""
    negs = negative_pairs(pairs, valid_ids,
                          ratio=cfg.train.neg_ratio, seed=seed)
    return split_pairs(
        pd.concat([pairs, negs], ignore_index=True), seed=seed)


def pair_batches(df, graphs, pk_tensor, pk_index,
                 batch_size, device, shuffle):
    """Single DataLoader for a DDI split."""
    kw = {"batch_size": batch_size, "collate_fn": pair_collate,
          **pin_and_workers(device, cfg.train.num_workers)}
    return DataLoader(
        DDIPairDataset(df, graphs, pk_tensor, pk_index),
        shuffle=shuffle, **kw)


def train_batches(batch_size, seed, device):
    """Train and val batches and class weights only. No test loader."""
    pairs, graphs, pk_tensor, pk_index, valid_ids = merged_data()
    trn, val, _ = sampled_splits(pairs, valid_ids, seed)
    sev_w, mech_w = class_weights(trn)
    args = (graphs, pk_tensor, pk_index, batch_size, device)
    return (pair_batches(trn, *args, shuffle=True),
            pair_batches(val, *args, shuffle=False),
            sev_w, mech_w)


def eval_batches(batch_size, seed, device):
    """Val and test batches only. Skips train dataset to save memory."""
    pairs, graphs, pk_tensor, pk_index, valid_ids = merged_data()
    _, val, tst = sampled_splits(pairs, valid_ids, seed)
    args = (graphs, pk_tensor, pk_index, batch_size, device)
    return (pair_batches(val, *args, shuffle=False),
            pair_batches(tst, *args, shuffle=False))


# Optimizer

def optimizer_bundle(params):
    """Adam optimizer with cosine annealing scheduler."""
    optimizer = torch.optim.Adam(
        params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs, eta_min=cfg.train.cosine_eta_min)
    return optimizer, scheduler



# Training


def ddi_step(model, batch, device):
    """Move batch to device, run model"""
    ba = batch["batch_a"].to(device)
    bb = batch["batch_b"].to(device)
    preds = model(ba, bb, batch["pk_a"].to(device), batch["pk_b"].to(device))
    return preds, batch["severity"].to(device), batch["mechanism"].to(device)


def train_epoch(model, batches, optimizer, device, sev_w, mech_w,
                mech_weight=None, step_fn=None):
    """Single training epoch"""
    if mech_weight is None:
        mech_weight = cfg.train.mechanism_weight
    if step_fn is None:
        step_fn = ddi_step
    model.train()
    total, n = 0.0, 0
    for batch in batches:
        preds, sev, mech = step_fn(model, batch, device)
        loss = ddi_loss(preds, sev, mech, sev_w, mech_w, mech_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        total += loss.item() * sev.size(0)
        n += sev.size(0)
    return total / max(n, 1)


@torch.no_grad()
def val_epoch(model, batches, device, sev_w, mech_w,
              mech_weight=None, step_fn=None):
    """Returns (joint_loss, severity_loss) means. Stopping uses severity."""
    if mech_weight is None:
        mech_weight = cfg.train.mechanism_weight
    if step_fn is None:
        step_fn = ddi_step
    model.eval()
    joint_total, sev_total, n = 0.0, 0.0, 0
    for batch in batches:
        preds, sev, mech = step_fn(model, batch, device)
        joint = ddi_loss(preds, sev, mech, sev_w, mech_w, mech_weight)
        sev_only = severity_loss(preds, sev, sev_w)
        bs = sev.size(0)
        joint_total += joint.item() * bs
        sev_total += sev_only.item() * bs
        n += bs
    denom = max(n, 1)
    return joint_total / denom, sev_total / denom


# Checkpointing


def snapshot(path, epoch, model, optimizer, scheduler, best_val, wait):
    """Full training state checkpoint."""
    atomic_write({
        "epoch": epoch, "best_val": best_val, "wait": wait,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, path)


def resume_checkpoint(path, model, optimizer, scheduler, device):
    """Restore training state"""
    if not path.exists():
        return 1, float("inf"), 0
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    print(f"Resumed epoch {ckpt['epoch']}, best_val={ckpt['best_val']:.4f}")
    return ckpt["epoch"] + 1, ckpt["best_val"], ckpt["wait"]


# Training

def patience_update(v_sev, best_val, wait, model, device, best_path):
    """Update best checkpoint and patience counter"""
    if v_sev < best_val:
        sync_device(device)
        atomic_write(model.state_dict(), best_path)
        return v_sev, 0
    return best_val, wait + 1


def fit(model, optimizer, scheduler, trn, val, device, sev_w, mech_w,
        best_path, resume_path, mech_weight=None, step_fn=None):
    """Train with severity-only early stopping and crash resume."""
    sev_w, mech_w = sev_w.to(device), mech_w.to(device)
    start, best_val, wait = resume_checkpoint(
        resume_path, model, optimizer, scheduler, device)
    for epoch in range(start, cfg.train.epochs + 1):
        t = train_epoch(model, trn, optimizer, device, sev_w, mech_w,
                        mech_weight, step_fn)
        flush_device_cache(device)
        v_joint, v_sev = val_epoch(model, val, device, sev_w, mech_w,
                                   mech_weight, step_fn)
        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  train={t:.4f}  "
                  f"val_joint={v_joint:.4f}  val_sev={v_sev:.4f}")
        best_val, wait = patience_update(
            v_sev, best_val, wait, model, device, best_path)
        if wait >= cfg.train.patience:
            print(f"  Early stop at epoch {epoch}")
            break
        if epoch % cfg.train.checkpoint_every == 0:
            sync_device(device)
            snapshot(resume_path, epoch, model, optimizer, scheduler,
                     best_val, wait)
        flush_device_cache(device)
    resume_path.unlink(missing_ok=True)
    return best_val


# Entry point

def train_ddi():
    device = preferred_device(cfg.train.device)
    print(f"Device: {device}")
    seed_everything(cfg.train.seed)
    cfg.mkdir_tree()

    print("Data Loading")
    trn, val, sev_w, mech_w = train_batches(
        cfg.train.batch_size, cfg.train.seed, device)
    print(f"Train: {len(trn.dataset)}, Val: {len(val.dataset)}")

    model = DDIModel(cfg.atom, cfg.gnn).to(device)
    optimizer, scheduler = optimizer_bundle(model.parameters())

    best_path = cfg.paths.models / "ddi_best.pt"
    best = fit(model, optimizer, scheduler, trn, val, device, sev_w, mech_w,
               best_path, cfg.paths.models / "ddi_resume.pt")
    print(f"Best val (severity): {best:.4f}, saved: {best_path}")

    del model, optimizer, scheduler
    flush_device_cache(device)


if __name__ == "__main__":
    train_ddi()