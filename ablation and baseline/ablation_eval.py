"""
Evaluation for ablation study.
Loads saved models and computes metrics through the shared probability-native
pipeline in evaluate.py.
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from rdkit import RDLogger

from config import (
    ProjectConfig, preferred_device, seed_everything, flush_device_cache,
)
from gnn import DDIModel
from train_gnn import eval_batches, ddi_step
from evaluate import (
    inference_outputs, checkpoint_weights, save_predictions,
    severity_curves, severity_classwise, severity_macro,
    severity_calibration,
    mechanism_thresholds, mechanism_scores, mechanism_macro,
    severity_table, mechanism_table, calibration_table,
)
from ablation import GNNOnly, PKOnly, pk_step

RDLogger.DisableLog("rdApp.*")
cfg = ProjectConfig()

VARIANTS = [
    ("gnn_only", lambda: GNNOnly(cfg.atom, cfg.gnn), ddi_step),
    ("pk_only", lambda: PKOnly(cfg.gnn), pk_step),
    ("single_task", lambda: DDIModel(cfg.atom, cfg.gnn), ddi_step),
]


# Variant inference and scoring

def variant_inference(model, val_ldr, tst_ldr, device, step_fn):
    """Run inference on val and test, return both prediction dicts."""
    val_d = inference_outputs(model, val_ldr, device, step_fn)
    flush_device_cache(device)
    tst_d = inference_outputs(model, tst_ldr, device, step_fn)
    flush_device_cache(device)
    return val_d, tst_d


def variant_metrics(tag, val_d, tst_d, skip_mechanism=False):
    """Severity and mechanism metrics for one variant on test, val for thresholds."""
    curves = severity_curves(tst_d["sev_probs"], tst_d["sev_labels"])
    per_class, _, accuracy, macro_f1 = severity_classwise(
        tst_d["sev_probs"], tst_d["sev_labels"])
    macro_auroc, macro_auprc = severity_macro(curves)
    cal = severity_calibration(tst_d["sev_probs"], tst_d["sev_labels"])

    if skip_mechanism:
        mech, mech_macro = {}, "N/A"
    else:
        thresholds = mechanism_thresholds(
            val_d["mech_probs"], val_d["mech_labels"])
        mech_full = mechanism_scores(
            tst_d["mech_probs"], tst_d["mech_labels"], thresholds)
        mech_macro = mechanism_macro(mech_full)
        mech = {k: v for k, v in mech_full.items() if v is not None}

    return {
        "accuracy": accuracy, "macro_f1": macro_f1,
        "macro_auroc": macro_auroc, "macro_auprc": macro_auprc,
        "mech_macro_auroc": mech_macro,
        "severity_per_class": per_class,
        "severity_curves": {k: {"auroc": v["auroc"], "auprc": v["auprc"]}
                            for k, v in curves.items()},
        "calibration": cal,
        "mechanism": mech,
    }


# Console output

def variant_report(tag, m):
    """Print severity, mechanism, calibration tables for one variant."""
    print(f"\n  {tag}")
    severity_table(m["severity_per_class"], m["accuracy"],
                   m["macro_f1"], m["severity_curves"])
    if isinstance(m["mech_macro_auroc"], float):
        mechanism_table(m["mechanism"])
        print(f"  Mechanism macro-AUROC: {m['mech_macro_auroc']:.4f}")
    else:
        print("  Mechanism: N/A")
    calibration_table(m["calibration"])


def ablation_summary(results):
    """Comparison table and write JSON."""
    cfg.paths.metrics.mkdir(parents=True, exist_ok=True)
    out = cfg.paths.metrics / "ablation_results.json"
    print("\n" + "=" * 72)
    print("ABLATION SUMMARY")
    print(f"  {'Variant':15s}  {'Acc':>7s}  {'F1':>7s}  {'S-AUROC':>7s}  "
          f"{'S-AUPRC':>7s}  {'M-AUROC':>7s}  {'ECE':>6s}")
    for name, r in results.items():
        m_auc = r["mech_macro_auroc"]
        m_str = (f"{m_auc:7.4f}" if isinstance(m_auc, float)
                 else f"{'N/A':>7s}")
        ece = r["calibration"]["ece"]
        print(f"  {name:15s}  {r['accuracy']:7.4f}  {r['macro_f1']:7.4f}  "
              f"{r['macro_auroc']:7.4f}  {r['macro_auprc']:7.4f}  {m_str}  "
              f"{ece:6.4f}")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out}")


# Variant evaluation

def scored_variant(tag, model_fn, step_fn, val, tst, device):
    """Load checkpoint, evaluate, cache predictions."""
    path = cfg.paths.models / f"ablation_{tag}.pt"
    if not path.exists():
        print(f"  {tag}: checkpoint not found at {path}")
        return None
    model = model_fn().to(device)
    checkpoint_weights(model, path, device)
    print(f"  {tag}: loaded {path}")
    val_d, tst_d = variant_inference(model, val, tst, device, step_fn)
    skip_mech = (tag == "single_task")
    tst_pair_ids = tst.dataset.pairs[["drug_a", "drug_b"]].values
    save_predictions(f"ablation_{tag}", val_d, tst_d, tst_pair_ids,
                     save_mechanism=not skip_mech)
    metrics = variant_metrics(tag, val_d, tst_d, skip_mechanism=skip_mech)
    variant_report(tag, metrics)
    del model
    flush_device_cache(device)
    return metrics


# Output

def evaluate_ablations(only=None):
    """Evaluate all available ablation checkpoints."""
    device = preferred_device(cfg.train.device)
    print(f"Device: {device}")
    seed_everything(cfg.train.seed)

    print("DATA LOADING")
    val, tst = eval_batches(cfg.train.batch_size, cfg.train.seed, device)
    print(f"Val: {len(val.dataset)}, Test: {len(tst.dataset)}")

    results = {}
    for tag, model_fn, step_fn in VARIANTS:
        if only and tag != only:
            continue
        metrics = scored_variant(tag, model_fn, step_fn, val, tst, device)
        if metrics is not None:
            results[tag] = metrics

    if not results:
        print("No checkpoints found.")
        return
    ablation_summary(results)


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else None
    if variant and variant not in ("gnn_only", "pk_only", "single_task"):
        print("Usage: python ablation_eval.py "
              "[gnn_only|pk_only|single_task]")
        sys.exit(1)
    evaluate_ablations(only=variant)