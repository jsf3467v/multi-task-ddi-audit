"""
Feature engineering for multi-task DDI prediction.
One-hot categorical atom/bond features, severity labels from DDInter,
mechanisms from DrugBank text.
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem

from config import ProjectConfig

cfg = ProjectConfig()

# Onehot encoding tables

ATOM_TYPES = [6, 7, 8, 9, 15, 16, 17, 35, 53]        # C N O F P S Cl Br I
DEGREES = [0, 1, 2, 3, 4, 5]
CHARGES = [-2, -1, 0, 1, 2]
HYBRIDIZATIONS = [
    int(Chem.rdchem.HybridizationType.SP),
    int(Chem.rdchem.HybridizationType.SP2),
    int(Chem.rdchem.HybridizationType.SP3),
    int(Chem.rdchem.HybridizationType.SP3D),
    int(Chem.rdchem.HybridizationType.SP3D2),
]
NUM_HS = [0, 1, 2, 3, 4]
VALENCES = [0, 1, 2, 3, 4, 5, 6]
HETERO_NUMS = frozenset({7, 8, 9, 15, 16, 17, 35, 53})

BOND_TYPES = [1, 2, 3, 12]                            # SINGLE DOUBLE TRIPLE AROMATIC
STEREO_TYPES = [0, 1, 2, 3, 4, 5]                     # NONE ANY Z E CIS TRANS

ATOM_IDX = {v: i for i, v in enumerate(ATOM_TYPES)}
DEGREE_IDX = {v: i for i, v in enumerate(DEGREES)}
CHARGE_IDX = {v: i for i, v in enumerate(CHARGES)}
HYBRID_IDX = {v: i for i, v in enumerate(HYBRIDIZATIONS)}
HS_IDX = {v: i for i, v in enumerate(NUM_HS)}
VAL_IDX = {v: i for i, v in enumerate(VALENCES)}
BOND_IDX = {v: i for i, v in enumerate(BOND_TYPES)}
STEREO_IDX = {v: i for i, v in enumerate(STEREO_TYPES)}


def one_hot(value, index_map, size):
    """One-hot vector with catch-all at final position."""
    vec = [0.0] * (size + 1)
    vec[index_map.get(value, size)] = 1.0
    return vec

# Atom bond features - 49 + 14


def atom_vec(atom, ring_info):
    """49-dim atom feature: one-hot categoricals + scalar properties."""
    idx = atom.GetIdx()
    in_ring = ring_info.NumAtomRings(idx) > 0
    min_rs = next((sz for sz in range(3, 9)
                   if ring_info.IsAtomInRingOfSize(idx, sz)), 0) if in_ring else 0
    return (
        one_hot(atom.GetAtomicNum(), ATOM_IDX, len(ATOM_TYPES))
        + one_hot(atom.GetTotalDegree(), DEGREE_IDX, len(DEGREES))
        + one_hot(atom.GetFormalCharge(), CHARGE_IDX, len(CHARGES))
        + one_hot(int(atom.GetHybridization()), HYBRID_IDX, len(HYBRIDIZATIONS))
        + one_hot(atom.GetTotalNumHs(), HS_IDX, len(NUM_HS))
        + one_hot(atom.GetExplicitValence(), VAL_IDX, len(VALENCES))
        + [float(atom.GetIsAromatic()), float(in_ring),
           float(atom.GetAtomicNum() in HETERO_NUMS),
           min_rs / cfg.atom.ring_norm,
           atom.GetMass() / cfg.atom.mass_norm,
           float(atom.GetNumRadicalElectrons())]
    )


def bond_vec(bond):
    """14-dim bond feature: one-hot type and binary flags and one-hot stereo."""
    return (
        one_hot(int(bond.GetBondType()), BOND_IDX, len(BOND_TYPES))
        + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
        + one_hot(int(bond.GetStereo()), STEREO_IDX, len(STEREO_TYPES))
    )


# Molecular graph


def molecular_graph(smiles):
    """SMILES to PyG Data with one-hot atom and bond features."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    ri = mol.GetRingInfo()
    x = torch.tensor([atom_vec(a, ri) for a in mol.GetAtoms()],
                      dtype=torch.float)
    if mol.GetNumBonds() == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 14), dtype=torch.float)
    else:
        src, dst, attrs = [], [], []
        for b in mol.GetBonds():
            i, j, bv = b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bond_vec(b)
            src += [i, j]; dst += [j, i]; attrs += [bv, bv]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(attrs, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# Data tables

def ddi_merged():
    """DrugBank pairs merged with DDInter severity. Drops 'Unknown'."""
    pairs = pd.read_csv(cfg.paths.processed / "ddi_pairs.csv")
    ddinter = pd.read_csv(cfg.paths.processed / "ddinter_matched.csv")
    ddinter = ddinter[ddinter["Level"] != "Unknown"]

    merged = pairs.drop(columns=["severity"]).merge(
        ddinter[["drug_a", "drug_b", "Level"]],
        on=["drug_a", "drug_b"], how="inner",
    ).rename(columns={"Level": "severity"})

    smiles = pd.read_csv(cfg.paths.processed / "drug_smiles.csv").set_index("drugbank_id")
    pk = pd.read_csv(cfg.paths.processed / "pk_features.csv", index_col=0)
    return merged, smiles, pk


def severity_integers(severity_col):
    """Vectorized severity string to integer label."""
    mapped = severity_col.map(cfg.data.severity_map)
    unmapped = mapped.isna()
    if unmapped.any():
        bad = severity_col[unmapped].unique().tolist()
        raise ValueError(f"Unmapped severity values: {bad}")
    return mapped.astype(int)


def mechanism_matrix(mechanism_col):
    """Pipe-joined strings to binary float32 array. Vectorized per column."""
    safe = mechanism_col.fillna("")
    return np.column_stack([
        safe.str.contains(m, regex=False).astype(np.float32)
        for m in cfg.data.mechanism_cols
    ])


# Negative sampling

def canonical_key(a, b, n):
    """Sorted pair indices to single int64 key."""
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    return lo * n + hi


def positive_keys(df_pos, id_to_idx, n):
    """Sorted positive pair keys for searchsorted filtering."""
    a = df_pos["drug_a"].map(id_to_idx).values.astype(np.int64)
    b = df_pos["drug_b"].map(id_to_idx).values.astype(np.int64)
    return np.sort(canonical_key(a, b, n))


def filtered_negatives(keys, first, pos_keys, n_neg):
    """Filter candidate keys against positive set through searchsorted."""
    if len(pos_keys) == 0:
        return first[:n_neg]
    idx = np.searchsorted(pos_keys, keys).clip(0, len(pos_keys) - 1)
    return first[pos_keys[idx] != keys][:n_neg]


def negative_pairs(df_pos, all_drug_ids, ratio=1.0, *, seed):
    """Single-pass vectorized non-interacting pair sampling."""
    rng = np.random.RandomState(seed)
    ids = np.array(sorted(all_drug_ids))
    id_to_idx = {did: i for i, did in enumerate(ids)}
    n, n_neg = len(ids), int(len(df_pos) * ratio)
    pos_keys = positive_keys(df_pos, id_to_idx, n)

    sz = n_neg * cfg.train.neg_oversample
    a = rng.randint(0, n, sz, dtype=np.int64)
    b = rng.randint(0, n, sz, dtype=np.int64)
    keep = a != b
    a, b = a[keep], b[keep]

    uniq, first = np.unique(canonical_key(a, b, n), return_index=True)
    perm = rng.permutation(len(uniq))
    uniq, first = uniq[perm], first[perm]
    sel = filtered_negatives(uniq, first, pos_keys, n_neg)
    if len(sel) < n_neg:
        print(f"  Negatives requested: {n_neg}, sampled: {len(sel)}")

    lo, hi = np.minimum(a[sel], b[sel]), np.maximum(a[sel], b[sel])
    return pd.DataFrame({"drug_a": ids[lo], "drug_b": ids[hi],
                         "severity": "none", "mechanisms": "none"})


# Drug graph precomputation


def drug_graphs(smiles_df):
    """Dict of drugbank_id to PyG graph."""
    return {did: g for did, smi in zip(smiles_df.index, smiles_df["smiles"])
            if (g := molecular_graph(smi)) is not None}