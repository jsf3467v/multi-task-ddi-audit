"""Unit tests for the severity-stratified train/val/test split."""
import pandas as pd

from stratify import split_pairs


def _make_df(n_per_class=100):
    rows = []
    for sev in ["none", "Minor", "Moderate", "Major"]:
        for k in range(n_per_class):
            rows.append({"pair_id": f"{sev}_{k}", "severity": sev})
    return pd.DataFrame(rows)


def test_split_is_a_partition():
    df = _make_df(100)                       # 400 rows
    trn, val, tst = split_pairs(df, seed=42)
    total = len(trn) + len(val) + len(tst)
    assert total == len(df)
    ids = set(trn["pair_id"]) | set(val["pair_id"]) | set(tst["pair_id"])
    assert ids == set(df["pair_id"])         # nothing lost or duplicated


def test_split_proportions_are_roughly_80_10_10():
    df = _make_df(100)
    trn, val, tst = split_pairs(df, seed=42)
    n = len(df)
    assert abs(len(trn) / n - 0.8) < 0.02
    assert abs(len(val) / n - 0.1) < 0.02
    assert abs(len(tst) / n - 0.1) < 0.02


def test_split_preserves_every_class_in_each_fold():
    df = _make_df(100)
    trn, val, tst = split_pairs(df, seed=42)
    classes = set(df["severity"])
    for fold in (trn, val, tst):
        assert set(fold["severity"]) == classes


def test_split_is_seed_deterministic():
    df = _make_df(50)
    a = split_pairs(df, seed=7)[0]["pair_id"].tolist()
    b = split_pairs(df, seed=7)[0]["pair_id"].tolist()
    assert a == b
