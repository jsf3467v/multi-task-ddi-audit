"""
Multi-task DDI prediction model.
SharedEncoder (GATv2 + JumpingKnowledge), PKBranch MLP,
PairClassifier trunk, severity and mechanism heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv, JumpingKnowledge, global_mean_pool, global_max_pool,
)
from torch_geometric.utils import dropout_edge
from config import GNNConfig, AtomConfig


class SharedEncoder(nn.Module):
    """GATv2 molecular encoder with Jumping Knowledge and pooling."""

    def __init__(self, atom_cfg: AtomConfig, gnn_cfg: GNNConfig):
        super().__init__()
        self.edge_dropout = gnn_cfg.edge_dropout
        self.input_proj = nn.Linear(atom_cfg.features, gnn_cfg.hidden_dim)
        self.convs = nn.ModuleList([
            GATv2Conv(gnn_cfg.hidden_dim, gnn_cfg.hidden_dim,
                      heads=gnn_cfg.heads, edge_dim=gnn_cfg.edge_dim,
                      concat=False)
            for _ in range(gnn_cfg.num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(gnn_cfg.hidden_dim)
            for _ in range(gnn_cfg.num_layers)
        ])
        self.drop = nn.Dropout(gnn_cfg.dropout)
        self.jk = JumpingKnowledge(mode=gnn_cfg.jk_mode)
        self.use_mean_max = gnn_cfg.pool == "mean_max"

        jk_dim = (gnn_cfg.hidden_dim * gnn_cfg.num_layers
                  if gnn_cfg.jk_mode == "cat" else gnn_cfg.hidden_dim)
        self.out_dim = jk_dim * 2 if self.use_mean_max else jk_dim

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = self.input_proj(x)
        ei, ea = edge_index, edge_attr
        if self.training and self.edge_dropout > 0:
            ei, mask = dropout_edge(edge_index, p=self.edge_dropout,
                                    training=True)
            ea = edge_attr[mask] if edge_attr is not None else None

        layer_outs = []
        for conv, norm in zip(self.convs, self.norms):
            x = self.drop(F.relu(norm(conv(x, ei, edge_attr=ea)))) + x
            layer_outs.append(x)

        x = self.jk(layer_outs)
        if self.use_mean_max:
            return torch.cat([global_mean_pool(x, batch),
                              global_max_pool(x, batch)], dim=-1)
        return global_mean_pool(x, batch)


class PKBranch(nn.Module):
    """Binary PK vector to dense embedding."""

    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(out_dim, out_dim),
        )

    def forward(self, pk):
        return self.net(pk)


# Shared interaction components

def interaction_vector(h_a, h_b):
    """Pairwise feature vector: [h_a; h_b; h_a*h_b; |h_a-h_b|]."""
    return torch.cat([h_a, h_b, h_a * h_b, torch.abs(h_a - h_b)], dim=-1)


class PairClassifier(nn.Module):
    """Interaction trunk with severity and mechanism heads."""

    def __init__(self, drug_dim, gnn_cfg: GNNConfig):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(drug_dim * 4, gnn_cfg.interaction_dim),
            nn.ReLU(), nn.Dropout(gnn_cfg.dropout),
            nn.Linear(gnn_cfg.interaction_dim, gnn_cfg.interaction_dim // 2),
            nn.ReLU(), nn.Dropout(gnn_cfg.dropout),
        )
        trunk_out = gnn_cfg.interaction_dim // 2
        self.severity_head = nn.Linear(trunk_out, gnn_cfg.severity_classes)
        self.mechanism_head = nn.Linear(trunk_out, gnn_cfg.mechanism_classes)

    def forward(self, h_a, h_b):
        t = self.trunk(interaction_vector(h_a, h_b))
        return {"severity": self.severity_head(t),
                "mechanism": self.mechanism_head(t)}


# Graph encoding

def fused_graph(encoder, batch_a, batch_b):
    """Encode two drug batches in a single GNN pass, split by graph index."""
    n_a = batch_a.num_graphs
    num_nodes_a = batch_a.x.size(0)
    x = torch.cat([batch_a.x, batch_b.x], dim=0)
    ei = torch.cat([batch_a.edge_index,
                    batch_b.edge_index + num_nodes_a], dim=1)
    ea = torch.cat([batch_a.edge_attr, batch_b.edge_attr], dim=0)
    bt = torch.cat([batch_a.batch, batch_b.batch + n_a], dim=0)
    mol_all = encoder(x, ei, bt, edge_attr=ea)
    return mol_all[:n_a], mol_all[n_a:]


# GNN-DDI Model

class DDIModel(nn.Module):
    """
    Encodes both drugs via molecular graph and PK, builds interaction vector.
    Predicts severity and mechanism via shared PairClassifier.
    """

    def __init__(self, atom_cfg: AtomConfig, gnn_cfg: GNNConfig):
        super().__init__()
        self.encoder = SharedEncoder(atom_cfg, gnn_cfg)
        self.pk_branch = PKBranch(gnn_cfg.pk_dim, gnn_cfg.pk_embed,
                                  gnn_cfg.dropout)
        drug_dim = self.encoder.out_dim + gnn_cfg.pk_embed
        self.classifier = PairClassifier(drug_dim, gnn_cfg)

    def forward(self, batch_a, batch_b, pk_a, pk_b):
        mol_a, mol_b = fused_graph(self.encoder, batch_a, batch_b)
        h_a = torch.cat([mol_a, self.pk_branch(pk_a)], dim=-1)
        h_b = torch.cat([mol_b, self.pk_branch(pk_b)], dim=-1)
        return self.classifier(h_a, h_b)


# Loss components

def severity_loss(preds, severity_labels, severity_weights=None):
    """Cross-entropy on severity head only. Used for unified stopping."""
    return F.cross_entropy(preds["severity"], severity_labels,
                           weight=severity_weights)


def ddi_loss(preds, severity_labels, mechanism_labels,
             severity_weights=None, mechanism_pos_weight=None,
             mechanism_weight=0.5):
    """Severity cross-entropy + mechanism BCE, weighted by lambda."""
    sev = severity_loss(preds, severity_labels, severity_weights)
    if mechanism_weight == 0.0:
        return sev
    mech = F.binary_cross_entropy_with_logits(
        preds["mechanism"], mechanism_labels,
        pos_weight=mechanism_pos_weight)
    return sev + mechanism_weight * mech