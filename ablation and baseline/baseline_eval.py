"""
Evaluation for baseline models.
Loads 5-fold CV sklearn models and scores each fold on fixed val/test
through the shared probability-native scoring pipeline in evaluate.py.
Also computes the 5-fold ensemble (mean of probs) for cross-model comparison.
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
from rdkit import RDLogger

from config import ProjectConfig
from evaluate import (
    severity_curves, severity_classwise, severity_macro,
    severity_calibration,
    mechanism_thresholds, mechanism_scores, mechanism_macro,
    severity_table, save_predictions,
)
from baseline import baseline_splits, pair_matrix, MODEL_REGISTRY

RDLogger.DisableLog("rdApp.*")
cfg = ProjectConfig()


# Per-fold scoring (probability-native)

def fold_severity(sev_model, X_tst, sev_tst):
    """Severity metrics for one fold on fixed test."""
    probs = sev_model.predict_proba(X_tst)
    curves = severity_curves(probs, sev_tst)
    _, _, accuracy, macro_f1 = severity_classwise(probs, sev_tst)
    macro_auroc, macro_auprc = severity_macro(curves)
    return {
        "accuracy": accuracy, "macro_f1": macro_f1,
        "macro_auroc": macro_auroc, "macro_auprc": macro_auprc,
        "per_class": {k: {"auroc": v["auroc"], "auprc": v["auprc"]}
                      for k, v in curves.items()},
    }


def fold_mechanism(mech_model, X_val, mech_val, X_tst, mech_tst):
    """Mechanism metrics for one fold: thresholds from val, scored on test."""
    val_probs = mech_model.predict_proba(X_val)
    tst_probs = mech_model.predict_proba(X_tst)
    thresholds = mechanism_thresholds(val_probs, mech_val)
    mech = mechanism_scores(tst_probs, mech_tst, thresholds)
    scored = {k: v for k, v in mech.items() if v is not None}
    return {"macro_auroc": mechanism_macro(mech), "per_mechanism": scored}


# Aggregation across folds

def mean_std(values):
    """Mean and std of a list of floats."""
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std()), 4)}


def severity_summary(folds):
    """Mean  std across folds for top-level severity metrics."""
    keys = ["accuracy", "macro_f1", "macro_auroc", "macro_auprc"]
    return {k: mean_std([f[k] for f in folds]) for k in keys}


def per_class_summary(folds):
    """Per-class AUROC/AUPRC mean ± std across folds."""
    names = list(folds[0]["per_class"].keys())
    out = {}
    for name in names:
        aur = [f["per_class"][name]["auroc"] for f in folds]
        aup = [f["per_class"][name]["auprc"] for f in folds]
        out[name] = {"auroc": mean_std(aur), "auprc": mean_std(aup)}
    return out


def mechanism_summary(folds):
    """Mean  std across folds for mechanism macro AUROC."""
    return {"mech_macro_auroc": mean_std([f["macro_auroc"] for f in folds])}


# Ensemble across folds (mean of probs)

def fold_probs(name, fold, X_val, X_tst):
    """Probabilities from one CV fold for severity and mechanism heads."""
    sev = joblib.load(cfg.paths.models / f"baseline_{name}_sev_fold{fold}.pkl")
    mech = joblib.load(cfg.paths.models / f"baseline_{name}_mech_fold{fold}.pkl")
    return {
        "val_sev": sev.predict_proba(X_val),
        "tst_sev": sev.predict_proba(X_tst),
        "val_mech": mech.predict_proba(X_val),
        "tst_mech": mech.predict_proba(X_tst),
    }


def ensemble_probs(name, X_val, X_tst):
    """Mean probabilities across all CV folds."""
    parts = [fold_probs(name, f, X_val, X_tst) for f in range(cfg.baseline.cv_folds)]
    return {k: np.mean([p[k] for p in parts], axis=0)
            for k in ("val_sev", "tst_sev", "val_mech", "tst_mech")}


def ensemble_severity(probs, labels):
    """Per-class P/R/F1 and AUROC/AUPRC and confusion matrix on ensemble probs."""
    per_class, cm, accuracy, macro_f1 = severity_classwise(probs, labels)
    curves = severity_curves(probs, labels)
    macro_auroc, macro_auprc = severity_macro(curves)
    return {
        "accuracy": accuracy, "macro_f1": macro_f1,
        "macro_auroc": macro_auroc, "macro_auprc": macro_auprc,
        "severity_per_class": per_class,
        "severity_curves": {k: {"auroc": v["auroc"], "auprc": v["auprc"]}
                            for k, v in curves.items()},
        "confusion_matrix": cm,
    }


def cache_ensemble(name, ens, sev_val, mech_val, sev_tst, mech_tst, tst_pair_ids):
    """Save ensemble predictions in shared format for cross-model analysis."""
    val = {"sev_probs": ens["val_sev"], "sev_labels": sev_val,
           "mech_probs": ens["val_mech"], "mech_labels": mech_val}
    tst = {"sev_probs": ens["tst_sev"], "sev_labels": sev_tst,
           "mech_probs": ens["tst_mech"], "mech_labels": mech_tst}
    save_predictions(name, val, tst, tst_pair_ids)


# Per-model scoring

def fold_scores(name, X_val, mech_val, X_tst, sev_tst, mech_tst):
    """Per-fold severity and mechanism results. None if any checkpoint missing."""
    folds_sev, folds_mech = [], []
    for fold in range(cfg.baseline.cv_folds):
        sev_path = cfg.paths.models / f"baseline_{name}_sev_fold{fold}.pkl"
        mech_path = cfg.paths.models / f"baseline_{name}_mech_fold{fold}.pkl"
        if not sev_path.exists() or not mech_path.exists():
            print(f"  {name} fold {fold}: missing, skipping model")
            return None, None
        sev_model = joblib.load(sev_path)
        mech_model = joblib.load(mech_path)
        folds_sev.append(fold_severity(sev_model, X_tst, sev_tst))
        folds_mech.append(fold_mechanism(mech_model, X_val, mech_val,
                                          X_tst, mech_tst))
    return folds_sev, folds_mech


def scored_model(name, X_val, sev_val, mech_val,
                 X_tst, sev_tst, mech_tst, tst_pair_ids):
    """Score every fold and the ensemble for one baseline."""
    folds_sev, folds_mech = fold_scores(name, X_val, mech_val,
                                        X_tst, sev_tst, mech_tst)
    if folds_sev is None:
        return None
    ens = ensemble_probs(name, X_val, X_tst)
    cache_ensemble(name, ens, sev_val, mech_val, sev_tst, mech_tst, tst_pair_ids)
    ens_sev = ensemble_severity(ens["tst_sev"], sev_tst)
    ens_thresholds = mechanism_thresholds(ens["val_mech"], mech_val)
    ens_mech = mechanism_scores(ens["tst_mech"], mech_tst, ens_thresholds)
    ens_cal = severity_calibration(ens["tst_sev"], sev_tst)
    return {
        "folds": {"severity": folds_sev, "mechanism": folds_mech},
        "severity": severity_summary(folds_sev),
        "severity_per_class": per_class_summary(folds_sev),
        "mechanism": mechanism_summary(folds_mech),
        "ensemble": {**ens_sev, "calibration": ens_cal,
                     "mechanism": {k: v for k, v in ens_mech.items() if v is not None},
                     "mech_macro_auroc": mechanism_macro(ens_mech)},
    }


# Console output

def model_report(name, result):
    """Mean  std severity, mechanism, and ensemble per-class table."""
    s, m, e = result["severity"], result["mechanism"]["mech_macro_auroc"], result["ensemble"]
    print(f"\n  {name.upper()}  (5-fold CV, mean ± std on fixed test)")
    for k in ["accuracy", "macro_f1", "macro_auroc", "macro_auprc"]:
        d = s[k]
        print(f"    {k:14s}  {d['mean']:.4f} ± {d['std']:.4f}")
    print(f"    {'mech_auroc':14s}  {m['mean']:.4f} ± {m['std']:.4f}")
    print(f"    ensemble:  mech_auroc={e['mech_macro_auroc']:.4f}  "
          f"ECE={e['calibration']['ece']:.4f}")
    severity_table(e["severity_per_class"], e["accuracy"], e["macro_f1"],
                   e["severity_curves"])


def baseline_summary(results):
    """Comparison table across all scored baselines."""
    print("\n" + "=" * 72)
    print("BASELINE SUMMARY - 5-fold CV, mean +/- std)")
    print(f"  {'Model':6s}  {'Acc':>15s}  {'F1':>15s}  "
          f"{'S-AUROC':>15s}  {'M-AUROC':>15s}")
    for name, r in results.items():
        s = r["severity"]
        m = r["mechanism"]["mech_macro_auroc"]
        a, f, u = s['accuracy'], s['macro_f1'], s['macro_auroc']
        print(f"  {name:6s}  "
              f"{a['mean']:.4f}±{a['std']:.4f}  "
              f"{f['mean']:.4f}±{f['std']:.4f}  "
              f"{u['mean']:.4f}±{u['std']:.4f}  "
              f"{m['mean']:.4f}±{m['std']:.4f}")


# Entry point

def evaluate_baselines():
    """Score all registered baselines with shared scoring pipeline."""
    _, val, tst, feats = baseline_splits()
    print("Feature matrices")
    X_val, sev_val, mech_val, _ = pair_matrix(val, *feats)
    X_tst, sev_tst, mech_tst, tst_df = pair_matrix(tst, *feats)
    tst_pair_ids = tst_df[["drug_a", "drug_b"]].values
    print(f"Val: {X_val.shape[0]}, Test: {X_tst.shape[0]}")

    results = {}
    for name in MODEL_REGISTRY:
        r = scored_model(name, X_val, sev_val, mech_val,
                         X_tst, sev_tst, mech_tst, tst_pair_ids)
        if r is not None:
            model_report(name, r)
            results[name] = r

    if not results:
        print("No baseline models found.")
        return
    baseline_summary(results)

    cfg.paths.metrics.mkdir(parents=True, exist_ok=True)
    out = cfg.paths.metrics / "baseline_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    evaluate_baselines()