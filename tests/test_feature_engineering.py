"""Unit tests for featurization and label derivation.
"""
import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

import config
from feature_engineering import (
    one_hot,
    atom_vec,
    bond_vec,
    molecular_graph,
    severity_integers,
    mechanism_matrix,
    canonical_key,
    negative_pairs,
)

cfg = config.ProjectConfig()

BENZENE = "c1ccccc1"
ETHANOL = "CCO"
INVALID = "this_is_not_a_smiles"


# --- one-hot encoding -------------------------------------------------------

def test_one_hot_known_value():
    idx_map = {10: 0, 20: 1, 30: 2}
    v = one_hot(20, idx_map, 3)
    assert v == [0.0, 1.0, 0.0, 0.0]      # size + 1 entries, catch-all last
    assert sum(v) == 1.0


def test_one_hot_unknown_falls_into_catch_all():
    idx_map = {10: 0, 20: 1, 30: 2}
    v = one_hot(99, idx_map, 3)
    assert v[-1] == 1.0                    # unknown -> final catch-all slot
    assert sum(v) == 1.0


# --- graph featurization ----------------------------------------------------

def test_molecular_graph_benzene_shapes():
    """Benzene: 6 atoms, 6 bonds stored both directions (12 edges).
    Atom features are 49-dim, bond features 14-dim — the GNN's input dims."""
    g = molecular_graph(BENZENE)
    assert g is not None
    assert g.x.shape == (6, cfg.atom.features)      # (6, 49)
    assert g.edge_index.shape == (2, 12)
    assert g.edge_attr.shape == (12, cfg.gnn.edge_dim)  # (12, 14)


def test_molecular_graph_invalid_returns_none():
    assert molecular_graph(INVALID) is None


def test_atom_and_bond_vector_widths():
    mol = Chem.MolFromSmiles(ETHANOL)
    ri = mol.GetRingInfo()
    assert len(atom_vec(mol.GetAtomWithIdx(0), ri)) == cfg.atom.features  # 49
    assert len(bond_vec(mol.GetBondWithIdx(0))) == cfg.gnn.edge_dim       # 14


# --- severity labels --------------------------------------------------------

def test_severity_integers_maps_known_levels():
    s = pd.Series(["none", "Minor", "Moderate", "Major"])
    assert severity_integers(s).tolist() == [0, 1, 2, 3]


def test_severity_integers_rejects_unmapped():
    with pytest.raises(ValueError):
        severity_integers(pd.Series(["Moderate", "Catastrophic"]))


# --- mechanism multi-hot ----------------------------------------------------

def test_mechanism_matrix_shape_and_membership():
    cols = cfg.data.mechanism_cols
    s = pd.Series([
        f"{cols[0]}|{cols[2]}",   # first and third mechanisms present
        "",                       # none
        cols[-1],                 # last mechanism only
    ])
    m = mechanism_matrix(s)
    assert m.shape == (3, len(cols))
    assert m.dtype == np.float32
    assert m[0, 0] == 1.0 and m[0, 2] == 1.0 and m[0, 1] == 0.0
    assert m[1].sum() == 0.0
    assert m[2, -1] == 1.0


# --- canonical pair key -----------------------------------------------------

def test_canonical_key_is_order_independent():
    a = np.array([3, 7, 0])
    b = np.array([5, 2, 9])
    n = 100
    assert np.array_equal(
        canonical_key(a, b, n), canonical_key(b, a, n))


def test_canonical_key_value():
    # lo * n + hi for a single pair (2, 5) with n = 10 -> 2*10 + 5 = 25
    assert int(canonical_key(np.array([5]), np.array([2]), 10)[0]) == 25


# --- negative sampling ------------------------------------------------------

def test_negative_pairs_avoids_positives_and_is_well_formed():
    ids = [f"DB{i:03d}" for i in range(20)]
    pos = pd.DataFrame({
        "drug_a": ["DB000", "DB001", "DB002"],
        "drug_b": ["DB005", "DB006", "DB007"],
    })
    neg = negative_pairs(pos, ids, ratio=1.0, seed=0)

    assert set(["drug_a", "drug_b", "severity", "mechanisms"]).issubset(neg.columns)
    assert (neg["severity"] == "none").all()

    # No sampled negative collides with a positive pair (unordered).
    pos_set = {frozenset(p) for p in zip(pos["drug_a"], pos["drug_b"])}
    neg_set = {frozenset(p) for p in zip(neg["drug_a"], neg["drug_b"])}
    assert pos_set.isdisjoint(neg_set)
    # And no self-pairs.
    assert all(len(p) == 2 for p in neg_set)
