"""
TB-specific analysis of DDI model predictions.
Filters test predictions to TB-relevant drug pairs and reports
stratified metrics through evaluate.py's shared probability-native pipeline.

Baselines use ensemble prediction across the 5 CV folds (mean of probs),
yielding single-model tier metrics comparable to the GNN's single checkpoint.

Usage:
    python tb_analysis.py gnn
    python tb_analysis.py mlp
    python tb_analysis.py rf
    python tb_analysis.py xgb
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ablation and baseline"))

import joblib
import numpy as np
from rdkit import RDLogger

from config import (
    ProjectConfig, preferred_device, seed_everything, flush_device_cache,
)
from gnn import DDIModel
from train_gnn import eval_batches
from evaluate import (
    inference_outputs, checkpoint_weights,
    severity_curves, severity_classwise, severity_macro,
    severity_calibration,
    mechanism_thresholds, mechanism_scores, mechanism_macro,
)
from baseline import baseline_splits, pair_matrix

RDLogger.DisableLog("rdApp.*")
cfg = ProjectConfig()

MECH_NAMES = cfg.data.mechanism_cols
SUPPORTED_MODELS = ("gnn", "rf", "mlp", "xgb")
MIN_TIER_N = 10

TB_FIRST_LINE = {
    "DB01045": "Rifampin",
    "DB00951": "Isoniazid",
    "DB00330": "Ethambutol",
}
TB_SECOND_LINE = {
    "DB08903": "Bedaquiline",
    "DB00601": "Linezolid",
    "DB00218": "Moxifloxacin",
    "DB01137": "Levofloxacin",
    "DB00845": "Clofazimine",
    "DB00479": "Amikacin",
}
ARV_COADMIN = {
    "DB00625": "Efavirenz",
    "DB00238": "Nevirapine",
    "DB00503": "Ritonavir",
    "DB01072": "Atazanavir",
    "DB01601": "Lopinavir",
}
COMEDICATIONS = {
    "DB00196": "Fluconazole",
    "DB00582": "Voriconazole",
    "DB00331": "Metformin",
    "DB01067": "Glipizide",
}
TB_TIERS = {
    "First-line": TB_FIRST_LINE,
    "Second-line": TB_SECOND_LINE,
    "ARV co-admin": ARV_COADMIN,
    "Comedications": COMEDICATIONS,
}
TB_ALL = {**TB_FIRST_LINE, **TB_SECOND_LINE, **ARV_COADMIN, **COMEDICATIONS}
RIFAMPIN = "DB01045"


# Drug filtering

def pair_mask(pairs_df, drug_ids):
    """True per pair if either drug is in drug_ids."""
    ids = set(drug_ids)
    return (pairs_df["drug_a"].isin(ids) | pairs_df["drug_b"].isin(ids)).values


def drug_inventory(pairs_df, drug_dict):
    """TB drugs present in the dataset, partitioned into found and missing."""
    all_drugs = set(pairs_df["drug_a"]) | set(pairs_df["drug_b"])
    found = {d: n for d, n in drug_dict.items() if d in all_drugs}
    missing = {d: n for d, n in drug_dict.items() if d not in all_drugs}
    return found, missing


# Subset scoring

def subset_severity(sev_probs, sev_labels, mask):
    """Severity metrics on boolean-masked subset."""
    sp, lab = sev_probs[mask], sev_labels[mask]
    if len(lab) < MIN_TIER_N:
        return None
    curves = severity_curves(sp, lab)
    _, _, accuracy, macro_f1 = severity_classwise(sp, lab)
    macro_auroc, macro_auprc = severity_macro(curves)
    cal = severity_calibration(sp, lab)
    return {
        "n": int(mask.sum()), "accuracy": accuracy,
        "macro_f1": macro_f1, "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
        "ece": cal["ece"],
        "per_class": {k: {"auroc": v["auroc"], "auprc": v["auprc"]}
                      for k, v in curves.items()},
    }


def subset_mechanism(mech_probs, mech_labels, mask, thresholds):
    """Mechanism metrics on boolean-masked subset."""
    mp, lab = mech_probs[mask], mech_labels[mask]
    if len(lab) < MIN_TIER_N:
        return None
    mech = mechanism_scores(mp, lab, thresholds)
    return {
        "n": int(mask.sum()),
        "macro_auroc": mechanism_macro(mech),
        "per_mechanism": {k: v for k, v in mech.items() if v is not None},
    }


# Rifampin CYP analysis

def rifampin_cyp(mech_probs, mech_labels, pairs_df, thresholds):
    """CYP induction precision/recall on rifampin pairs."""
    rif_mask = pair_mask(pairs_df, {RIFAMPIN})
    n_rif = rif_mask.sum()
    if n_rif == 0:
        print("  No rifampin pairs in test set.")
        return None
    cyp_idx = MECH_NAMES.index("cyp_induction")
    probs = mech_probs[rif_mask, cyp_idx]
    labels = mech_labels[rif_mask, cyp_idx]
    preds = (probs >= thresholds[cyp_idx]).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)

    print(f"\n  Rifampin CYP Induction (n={n_rif})")
    print(f"    Actual positives: {int(labels.sum())}  Predicted: {int(preds.sum())}")
    print(f"    Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    return {"n_pairs": int(n_rif), "positives": int(labels.sum()),
            "predicted": int(preds.sum()), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4)}


# Console output

def inventory_report(found, missing):
    """Print which TB drugs were found in the dataset."""
    print(f"\nTB drugs found: {len(found)} of {len(found) + len(missing)}")
    for did, name in sorted(found.items(), key=lambda x: x[1]):
        print(f"  {did}: {name}")
    if missing:
        print(f"  Missing: {', '.join(sorted(missing.values()))}")


def tier_table(tiers):
    """Print tier comparison table."""
    print(f"\n  {'Tier':15s}  {'n':>6s}  {'Acc':>7s}  {'F1':>7s}  "
          f"{'S-AUROC':>7s}  {'S-AUPRC':>7s}  {'M-AUROC':>7s}  {'ECE':>6s}")
    for name, t in tiers.items():
        if t is None or t["sev"] is None:
            print(f"  {name:15s}  insufficient data (n<{MIN_TIER_N})")
            continue
        s = t["sev"]
        m_auc = t["mech"]["macro_auroc"] if t["mech"] else "N/A"
        m_str = f"{m_auc:7.4f}" if isinstance(m_auc, float) else f"{'N/A':>7s}"
        print(f"  {name:15s}  {s['n']:6d}  {s['accuracy']:7.4f}  "
              f"{s['macro_f1']:7.4f}  {s['macro_auroc']:7.4f}  "
              f"{s['macro_auprc']:7.4f}  {m_str}  {s['ece']:6.4f}")


# Tier scoring

def tier_masks(tst_df, found):
    """Boolean masks for full test, TB union, and each TB tier."""
    masks = [("Full Test", np.ones(len(tst_df), dtype=bool)),
             ("TB (any)", pair_mask(tst_df, found))]
    masks.extend((tier, pair_mask(tst_df, drugs))
                 for tier, drugs in TB_TIERS.items())
    return masks


def scored_tiers(tst_d, tst_df, thresholds, found):
    """Severity + mechanism metrics per TB drug tier."""
    tiers = {}
    for name, mask in tier_masks(tst_df, found):
        sev = subset_severity(tst_d["sev_probs"], tst_d["sev_labels"], mask)
        mech = subset_mechanism(tst_d["mech_probs"], tst_d["mech_labels"],
                                mask, thresholds)
        tiers[name] = {"sev": sev, "mech": mech}
    return tiers


# GNN predictions

def gnn_predictions(device):
    """Load GNN, run inference on val and test."""
    val_ldr, tst_ldr = eval_batches(cfg.train.batch_size, cfg.train.seed, device)
    model = DDIModel(cfg.atom, cfg.gnn).to(device)
    checkpoint_weights(model, cfg.paths.models / "ddi_best.pt", device)
    print("Running")
    val_d = inference_outputs(model, val_ldr, device)
    flush_device_cache(device)
    tst_d = inference_outputs(model, tst_ldr, device)
    del model
    flush_device_cache(device)
    return val_d, tst_d, tst_ldr.dataset.pairs


# Baseline predictions - 5-fold CV ensemble

def baseline_fold_probs(model_name, fold, X_val, X_tst):
    """Single-fold severity + mechanism probabilities on fixed val/test."""
    sev = joblib.load(cfg.paths.models / f"baseline_{model_name}_sev_fold{fold}.pkl")
    mech = joblib.load(cfg.paths.models / f"baseline_{model_name}_mech_fold{fold}.pkl")
    return {
        "val_sev": sev.predict_proba(X_val),
        "tst_sev": sev.predict_proba(X_tst),
        "val_mech": mech.predict_proba(X_val),
        "tst_mech": mech.predict_proba(X_tst),
    }


def ensemble_probs(model_name, X_val, X_tst):
    """Mean probabilities across all CV folds."""
    parts = []
    for fold in range(cfg.baseline.cv_folds):
        print(f"  Fold {fold}: loading predictions")
        parts.append(baseline_fold_probs(model_name, fold, X_val, X_tst))
    return {k: np.mean([p[k] for p in parts], axis=0)
            for k in ("val_sev", "tst_sev", "val_mech", "tst_mech")}


def baseline_predictions(model_name):
    """Ensemble CV predictions with same shape as gnn_predictions output."""
    _, val, tst, feats = baseline_splits()
    print("Feature matrices")
    X_val, sev_val, mech_val, _ = pair_matrix(val, *feats)
    X_tst, sev_tst, mech_tst, tst_df = pair_matrix(tst, *feats)
    e = ensemble_probs(model_name, X_val, X_tst)
    val_d = {"sev_probs": e["val_sev"], "sev_labels": sev_val,
             "mech_probs": e["val_mech"], "mech_labels": mech_val}
    tst_d = {"sev_probs": e["tst_sev"], "sev_labels": sev_tst,
             "mech_probs": e["tst_mech"], "mech_labels": mech_tst}
    return val_d, tst_d, tst_df


# Print Out

def tb_json(model_name, tiers, rif, found, missing):
    """Write per-model TB analysis JSON."""
    cfg.paths.metrics.mkdir(parents=True, exist_ok=True)
    out = cfg.paths.metrics / f"tb_analysis_{model_name}.json"
    with open(out, "w") as f:
        json.dump({"model": model_name, "tiers": tiers, "rifampin_cyp": rif,
                   "drugs_found": found, "drugs_missing": missing,
                   "min_tier_n": MIN_TIER_N},
                  f, indent=2, default=str)
    print(f"\nSaved: {out}")


def tb_analysis(model_name):
    """TB-specific evaluation dispatched by model type."""
    seed_everything(cfg.train.seed)

    if model_name == "gnn":
        device = preferred_device(cfg.train.device)
        print(f"Device: {device}")
        print("GNN predictions")
        val_d, tst_d, tst_df = gnn_predictions(device)
    else:
        print(f" {model_name.upper()} predictions - 5-fold ensemble ")
        val_d, tst_d, tst_df = baseline_predictions(model_name)

    print(f"Test pairs: {len(tst_df)}")
    found, missing = drug_inventory(tst_df, TB_ALL)
    inventory_report(found, missing)

    thresholds = mechanism_thresholds(val_d["mech_probs"], val_d["mech_labels"])
    tiers = scored_tiers(tst_d, tst_df, thresholds, found)

    print("\n" + "=" * 65)
    print(f"TB STRATIFIED ANALYSIS ({model_name.upper()})")
    tier_table(tiers)
    rif = rifampin_cyp(tst_d["mech_probs"], tst_d["mech_labels"],
                       tst_df, thresholds)
    tb_json(model_name, tiers, rif, found, missing)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in SUPPORTED_MODELS:
        print(f"Usage: python tb_analysis.py [{'|'.join(SUPPORTED_MODELS)}]")
        sys.exit(1)
    tb_analysis(sys.argv[1])