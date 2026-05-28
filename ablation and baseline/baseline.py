"""
Baseline model training for DDI prediction.
RF, MLP, and XGBoost on Morgan fingerprints + pharmacokinetic features.
5-fold stratified CV on the training split; fixed val/test held out.
Training only. Run baseline_eval.py for metrics.

Usage:
    python baseline.py              # train all three
    python baseline.py rf           # train only RF
    python baseline.py mlp          # train only MLP
    python baseline.py xgb          # train only XGBoost
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

from config import ProjectConfig, seed_everything
from feature_engineering import (
    ddi_merged, severity_integers, mechanism_matrix, negative_pairs,
)
from stratify import split_pairs

RDLogger.DisableLog("rdApp.*")
cfg = ProjectConfig()
bl = cfg.baseline


# Features

def morgan_matrix(smiles_df):
    """SMILES dataframe to contiguous fingerprint array and index dict."""
    radius, nbits = bl.fp_radius, bl.fp_bits
    ids, arrs = [], []
    for did, smi in zip(smiles_df.index, smiles_df["smiles"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        arr = np.zeros(nbits, dtype=np.float32)
        ConvertToNumpyArray(fp, arr)
        ids.append(did)
        arrs.append(arr)
    return np.stack(arrs), {did: i for i, did in enumerate(ids)}


def pair_matrix(df, fp_arr, fp_idx, pk_arr, pk_idx):
    """Pair features and labels and filtered dataframe"""
    valid = set(fp_idx.keys()) & set(pk_idx.keys())
    mask = df["drug_a"].isin(valid) & df["drug_b"].isin(valid)
    df_v = df.loc[mask].reset_index(drop=True)

    ia = df_v["drug_a"].map(fp_idx).values
    ib = df_v["drug_b"].map(fp_idx).values
    fa, fb = fp_arr[ia], fp_arr[ib]

    ja = df_v["drug_a"].map(pk_idx).values
    jb = df_v["drug_b"].map(pk_idx).values
    pa, pb = pk_arr[ja], pk_arr[jb]

    X = np.hstack([fa, fb, fa * fb, np.abs(fa - fb), pa, pb])
    sev = severity_integers(df_v["severity"]).values
    mech = mechanism_matrix(df_v["mechanisms"])
    return X, sev, mech, df_v


def baseline_splits():
    """Fingerprints, PK arrays, and stratified splits."""
    pairs, smiles_df, pk_df = ddi_merged()
    print("CFingerprints")
    fp_arr, fp_idx = morgan_matrix(smiles_df)
    pk_arr = pk_df.values.astype(np.float32)
    pk_idx = {did: i for i, did in enumerate(pk_df.index)}

    valid_ids = set(fp_idx.keys()) & set(pk_idx.keys())
    pairs = pairs[pairs["drug_a"].isin(valid_ids)
                  & pairs["drug_b"].isin(valid_ids)]
    negs = negative_pairs(pairs, valid_ids,
                          ratio=cfg.train.neg_ratio, seed=cfg.train.seed)
    full = pd.concat([pairs, negs], ignore_index=True)
    trn, val, tst = split_pairs(full, seed=cfg.train.seed)
    feats = (fp_arr, fp_idx, pk_arr, pk_idx)
    return trn, val, tst, feats


# Model constructors

def rf_severity():
    base = RandomForestClassifier(
        n_estimators=bl.rf_trees, max_depth=bl.rf_depth,
        n_jobs=-1, random_state=cfg.train.seed)
    return CalibratedClassifierCV(base, method="isotonic", cv=3)


def mlp_severity():
    return MLPClassifier(
        hidden_layer_sizes=bl.mlp_hidden, max_iter=bl.mlp_epochs,
        early_stopping=True, validation_fraction=0.1,
        random_state=cfg.train.seed)


def xgb_severity():
    base = XGBClassifier(
        n_estimators=bl.xgb_trees, max_depth=bl.xgb_depth,
        learning_rate=bl.xgb_lr, subsample=bl.xgb_subsample,
        colsample_bytree=bl.xgb_colsample, tree_method="hist",
        n_jobs=-1, random_state=cfg.train.seed,
        eval_metric="mlogloss", verbosity=0)
    return CalibratedClassifierCV(base, method="isotonic", cv=3)


def rf_mechanism():
    base = RandomForestClassifier(
        n_estimators=bl.mech_rf_trees, max_depth=bl.mech_rf_depth,
        n_jobs=-1, random_state=cfg.train.seed)
    return OneVsRestClassifier(
        CalibratedClassifierCV(base, method="isotonic", cv=3))


def mlp_mechanism():
    return OneVsRestClassifier(
        MLPClassifier(
            hidden_layer_sizes=bl.mech_mlp_hidden,
            max_iter=bl.mech_mlp_epochs, early_stopping=True,
            validation_fraction=0.1, random_state=cfg.train.seed))


def xgb_mechanism():
    base = XGBClassifier(
        n_estimators=bl.mech_xgb_trees, max_depth=bl.mech_xgb_depth,
        learning_rate=bl.mech_xgb_lr, tree_method="hist", n_jobs=-1,
        random_state=cfg.train.seed, eval_metric="logloss",
        verbosity=0)
    return OneVsRestClassifier(
        CalibratedClassifierCV(base, method="isotonic", cv=3))


MODEL_REGISTRY = {
    "rf": (rf_severity, rf_mechanism),
    "mlp": (mlp_severity, mlp_mechanism),
    "xgb": (xgb_severity, xgb_mechanism),
}


# CV training

def fold_models(name, X, sev, mech, fold, train_idx):
    """Fit severity and mechanism for one fold"""
    sev_ctor, mech_ctor = MODEL_REGISTRY[name]
    print(f"  {name} fold {fold}: severity")
    sev_model = sev_ctor()
    sev_model.fit(X[train_idx], sev[train_idx])
    joblib.dump(sev_model,
                cfg.paths.models / f"baseline_{name}_sev_fold{fold}.pkl")

    print(f"  {name} fold {fold}: mechanism")
    mech_model = mech_ctor()
    mech_model.fit(X[train_idx], mech[train_idx])
    joblib.dump(mech_model,
                cfg.paths.models / f"baseline_{name}_mech_fold{fold}.pkl")


def cv_fit(name, X, sev, mech):
    """5-fold stratified CV. Train severity and mechanism per fold."""
    skf = StratifiedKFold(n_splits=bl.cv_folds, shuffle=True,
                          random_state=cfg.train.seed)
    for fold, (train_idx, _) in enumerate(skf.split(X, sev)):
        fold_models(name, X, sev, mech, fold, train_idx)


# Print out

def train_baselines(only=None):
    """Train requested baselines. Each model saved per fold."""
    seed_everything(cfg.train.seed)
    cfg.paths.models.mkdir(parents=True, exist_ok=True)

    trn, _, _, feats = baseline_splits()
    print("Feature matrix")
    X, sev, mech, _ = pair_matrix(trn, *feats)
    print(f"Train: {X.shape[0]}, Features: {X.shape[1]}, "
          f"CV folds: {bl.cv_folds}")

    for name in MODEL_REGISTRY:
        if only and name != only:
            continue
        print(f"\n--- {name} ---")
        cv_fit(name, X, sev, mech)

    print("\n Complete")


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else None
    if variant and variant not in MODEL_REGISTRY:
        print(f"Usage: python baseline.py [{'|'.join(MODEL_REGISTRY)}]")
        sys.exit(1)
    train_baselines(only=variant)