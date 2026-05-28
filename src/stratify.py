"""
Stratified train / val / test split for DDI pairs.
Preserves severity class proportions across all three splits.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def split_pairs(df, train_frac=0.8, val_frac=0.1, *, seed):
    """Severity-stratified three-way split (80/10/10 default)."""
    test_frac = 1.0 - train_frac - val_frac
    val_of_remaining = val_frac / (val_frac + test_frac)

    trn, rest = train_test_split(
        df, train_size=train_frac, stratify=df["severity"],
        random_state=seed)
    val, tst = train_test_split(
        rest, train_size=val_of_remaining, stratify=rest["severity"],
        random_state=seed)

    return (
        trn.reset_index(drop=True),
        val.reset_index(drop=True),
        tst.reset_index(drop=True),
    )