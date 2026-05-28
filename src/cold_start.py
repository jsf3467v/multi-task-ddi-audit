"""
Drug-level cold-start evaluation.
Partitions test pairs by whether their drugs appear in the training split,
then scores each subset through the shared probability-native pipeline in
evaluate.py. Uses cached predictions and drug IDs to avoid re-running
inference or rebuilding molecular features.

    warm_warm: both drugs appear in training
    warm_cold: exactly one drug appears in training
    cold_cold: neither drug appears in training
"""

import json

import numpy as np
import pandas as pd

from config import ProjectConfig, seed_everything
from feature_engineering import ddi_merged, negative_pairs
from stratify import split_pairs
from evaluate import (
    cached_predictions,
    severity_curves, severity_classwise, severity_macro,
    severity_calibration,
    mechanism_thresholds, mechanism_scores, mechanism_macro,
)

cfg = ProjectConfig()
MIN_SUBSET_N = 10


# Prediction inventory

def predictions_dir():
    return cfg.paths.metrics.parent / "predictions"


def cached_models():
    """Names of models with cached predictions on disk."""
    pdir = predictions_dir()
    if not pdir.exists():
        return []
    return sorted(p.stem for p in pdir.glob("*.npz"))


# Train-drug membership

def train_drug_ids():
    """Drug IDs present in the training split, reproduced from the same seed."""
    pairs, smiles_df, pk_df = ddi_merged()
    valid = set(smiles_df.index) & set(pk_df.index)
    pairs = pairs[pairs["drug_a"].isin(valid) & pairs["drug_b"].isin(valid)]
    negs = negative_pairs(pairs, valid, ratio=cfg.train.neg_ratio,
                          seed=cfg.train.seed)
    full = pd.concat([pairs, negs], ignore_index=True)
    trn, _, _ = split_pairs(full, seed=cfg.train.seed)
    return set(trn["drug_a"]) | set(trn["drug_b"])


def partition_masks(drug_a, drug_b, train_drugs):
    """Boolean masks for warm-warm / warm-cold / cold-cold partitions."""
    a_warm = np.isin(drug_a, list(train_drugs))
    b_warm = np.isin(drug_b, list(train_drugs))
    return {
        "warm_warm": a_warm & b_warm,
        "warm_cold": a_warm ^ b_warm,
        "cold_cold": ~a_warm & ~b_warm,
    }


# Subset scoring

def subset_metrics(tst_d, mask, thresholds):
    """Severity + mechanism + calibration on a boolean-masked subset."""
    n = int(mask.sum())
    if n < MIN_SUBSET_N:
        return {"n": n, "skipped": True}
    sp, sl = tst_d["sev_probs"][mask], tst_d["sev_labels"][mask]
    curves = severity_curves(sp, sl)
    _, _, accuracy, macro_f1 = severity_classwise(sp, sl)
    macro_auroc, macro_auprc = severity_macro(curves)
    cal = severity_calibration(sp, sl)
    out = {
        "n": n, "accuracy": accuracy, "macro_f1": macro_f1,
        "macro_auroc": macro_auroc, "macro_auprc": macro_auprc,
        "ece": cal["ece"],
    }
    if thresholds is not None:
        mp = tst_d["mech_probs"][mask]
        ml = tst_d["mech_labels"][mask]
        out["mech_macro_auroc"] = mechanism_macro(
            mechanism_scores(mp, ml, thresholds))
    return out


def model_partitions(preds, masks):
    """Per-partition metrics for one model."""
    val = preds["val"]
    tst = preds["tst"]
    thresholds = (mechanism_thresholds(val["mech_probs"], val["mech_labels"])
                  if preds["has_mech"] else None)
    return {tag: subset_metrics(tst, mask, thresholds)
            for tag, mask in masks.items()}


# Console output

def partition_table(name, parts):
    """Print partition comparison table for one model."""
    print(f"\n  {name}")
    print(f"    {'partition':12s}  {'n':>6s}  {'Acc':>7s}  {'F1':>7s}  "
          f"{'S-AUROC':>7s}  {'S-AUPRC':>7s}  {'M-AUROC':>7s}  {'ECE':>6s}")
    for tag in ("warm_warm", "warm_cold", "cold_cold"):
        p = parts[tag]
        if p.get("skipped"):
            print(f"    {tag:12s}  {p['n']:>6d}  insufficient (n<{MIN_SUBSET_N})")
            continue
        m_auc = p.get("mech_macro_auroc")
        m_str = f"{m_auc:7.4f}" if m_auc is not None else f"{'N/A':>7s}"
        print(f"    {tag:12s}  {p['n']:>6d}  {p['accuracy']:7.4f}  "
              f"{p['macro_f1']:7.4f}  {p['macro_auroc']:7.4f}  "
              f"{p['macro_auprc']:7.4f}  {m_str}  {p['ece']:6.4f}")


def partition_counts(masks):
    """Print partition sizes once before per-model tables."""
    total = sum(int(m.sum()) for m in masks.values())
    print(f"\nPartition sizes (test n={total})")
    for tag, mask in masks.items():
        n = int(mask.sum())
        print(f"  {tag:12s}  {n:>6d}  ({n / total:>5.1%})")


# Entry point

def stratified_models(names, train_drugs):
    """Per-model partition metrics"""
    results = {}
    first_masks = None
    for name in names:
        preds = cached_predictions(name)
        masks = partition_masks(preds["tst_drug_a"], preds["tst_drug_b"],
                                train_drugs)
        if first_masks is None:
            first_masks = masks
            partition_counts(masks)
        results[name] = model_partitions(preds, masks)
        partition_table(name, results[name])
    return results, first_masks


def cold_start_analysis():
    """Stratify cached predictions by drug-membership partition."""
    seed_everything(cfg.train.seed)

    names = cached_models()
    if not names:
        print(f"No cached predictions in {predictions_dir()}.")
        return
    print(f"Cached models: {', '.join(names)}")

    print("Training drug membership")
    train_drugs = train_drug_ids()
    print(f"Training drugs: {len(train_drugs):,}")

    results, masks = stratified_models(names, train_drugs)

    cfg.paths.metrics.mkdir(parents=True, exist_ok=True)
    out = cfg.paths.metrics / "cold_start_analysis.json"
    with open(out, "w") as f:
        json.dump({"partitions": results,
                   "partition_sizes": {tag: int(m.sum()) for tag, m in masks.items()},
                   "min_subset_n": MIN_SUBSET_N},
                  f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    cold_start_analysis()