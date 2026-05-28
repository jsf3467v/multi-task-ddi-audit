"""
Evaluation for multi-task DDI prediction.
Probability-native scoring shared across GNN, ablations, and baselines.
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from rdkit import RDLogger
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix, f1_score,
)

from config import (
    ProjectConfig, preferred_device, seed_everything, flush_device_cache,
)
from gnn import DDIModel
from train_gnn import eval_batches, ddi_step

RDLogger.DisableLog("rdApp.*")
plt.style.use("seaborn-v0_8-whitegrid")

cfg = ProjectConfig()
SEV_NAMES = list(cfg.data.severity_map.keys())
MECH_NAMES = cfg.data.mechanism_cols


# Model restoration

def checkpoint_weights(model, path, device):
    """Saved checkpoint weights to model."""
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)


# Activations

def softmax(logits):
    """Numerically stable softmax over axis 1."""
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def sigmoid(logits):
    """Element-wise sigmoid, numerically stable."""
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -88, 88)))


# Inference

@torch.no_grad()
def inference_outputs(model, batches, device, step_fn=None):
    """Run inference, return numpy probabilities and labels."""
    if step_fn is None:
        step_fn = ddi_step
    model.eval()
    sev_logits, sev_labels, mech_logits, mech_labels = [], [], [], []
    for batch in batches:
        preds, sev, mech = step_fn(model, batch, device)
        sev_logits.append(preds["severity"])
        sev_labels.append(sev)
        mech_logits.append(preds["mechanism"])
        mech_labels.append(mech)
    sl = torch.cat(sev_logits).cpu().numpy()
    ml = torch.cat(mech_logits).cpu().numpy()
    return {
        "sev_probs": softmax(sl),
        "sev_labels": torch.cat(sev_labels).cpu().numpy(),
        "mech_probs": sigmoid(ml),
        "mech_labels": torch.cat(mech_labels).cpu().numpy(),
    }


# Safety check

def scorable(y):
    """True when y has both positive and negative samples."""
    pos = y.sum()
    return 0 < pos < len(y)


# Severity metrics - probability-native

def severity_curves(probs, labels):
    """Per-class ROC and PR curve data + AUROC/AUPRC."""
    n_classes = len(SEV_NAMES)
    one_hot = np.eye(n_classes)[labels]
    curves = {}
    for i, name in enumerate(SEV_NAMES):
        if not scorable(one_hot[:, i]):
            continue
        fpr, tpr, _ = roc_curve(one_hot[:, i], probs[:, i])
        prec, rec, _ = precision_recall_curve(one_hot[:, i], probs[:, i])
        curves[name] = {
            "fpr": fpr, "tpr": tpr, "prec": prec, "rec": rec,
            "auroc": round(roc_auc_score(one_hot[:, i], probs[:, i]), 4),
            "auprc": round(average_precision_score(one_hot[:, i], probs[:, i]), 4),
        }
    return curves


def severity_macro(curves):
    """Macro AUROC and AUPRC from per-class curves."""
    aurocs = [c["auroc"] for c in curves.values()]
    auprcs = [c["auprc"] for c in curves.values()]
    return (round(np.mean(aurocs), 4) if aurocs else 0.0,
            round(np.mean(auprcs), 4) if auprcs else 0.0)


def severity_classwise(probs, labels):
    """Per-class metrics from confusion matrix."""
    n_classes = len(SEV_NAMES)
    preds = probs.argmax(axis=1)
    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    accuracy = round(float((preds == labels).mean()), 4)
    macro_f1 = round(float(
        f1_score(labels, preds, average="macro", zero_division=0)), 4)

    tp = np.diag(cm)
    row_sums, col_sums = cm.sum(axis=1), cm.sum(axis=0)
    fn, fp = row_sums - tp, col_sums - tp
    tn = cm.sum() - tp - fn - fp
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    spec = tn / np.maximum(tn + fp, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-8)

    per_class = {name: {"precision": round(float(prec[i]), 4),
                        "sensitivity": round(float(rec[i]), 4),
                        "specificity": round(float(spec[i]), 4),
                        "f1": round(float(f1[i]), 4),
                        "support": int(row_sums[i])}
                 for i, name in enumerate(SEV_NAMES)}
    return per_class, cm.tolist(), accuracy, macro_f1


# Calibration

def severity_calibration(probs, labels, n_bins=10):
    """Multiclass calibration via predicted-class confidence binning and ECE."""
    pred_class = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = (pred_class == labels).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)

    bins = []
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        avg_conf = float(confidence[mask].mean())
        accuracy = float(correct[mask].mean())
        bins.append({"bin": b, "lower": float(edges[b]), "upper": float(edges[b + 1]),
                     "n": n, "confidence": round(avg_conf, 4),
                     "accuracy": round(accuracy, 4)})
        ece += (n / len(probs)) * abs(accuracy - avg_conf)

    return {"bins": bins, "ece": round(float(ece), 4), "n_total": int(len(probs))}


# Mechanism metrics (probability-native)

def mechanism_thresholds(probs, labels):
    """Per-mechanism F1-maximizing thresholds via vectorized sweep."""
    candidates = np.arange(0.05, 0.96, 0.05)
    n_mech = labels.shape[1]
    best = np.full(n_mech, 0.5)
    for i in range(n_mech):
        if not scorable(labels[:, i]):
            continue
        preds = (probs[:, i:i + 1] >= candidates[np.newaxis, :]).astype(np.float32)
        y = labels[:, i].astype(np.float32)
        tp = (preds * y[:, np.newaxis]).sum(axis=0)
        fp = (preds * (1.0 - y[:, np.newaxis])).sum(axis=0)
        fn = ((1.0 - preds) * y[:, np.newaxis]).sum(axis=0)
        f1 = 2.0 * tp / np.maximum(2.0 * tp + fp + fn, 1e-8)
        best[i] = candidates[f1.argmax()]
    return best


def mechanism_scores(probs, labels, thresholds=None):
    """Per-mechanism AUROC, AUPRC, F1, precision, recall, specificity."""
    if thresholds is None:
        thresholds = np.full(labels.shape[1], 0.5)
    preds = (probs >= thresholds[np.newaxis, :]).astype(np.int64)
    y = labels.astype(np.int64)
    tp = (preds * y).sum(axis=0)
    fp = (preds * (1 - y)).sum(axis=0)
    fn = ((1 - preds) * y).sum(axis=0)
    tn = ((1 - preds) * (1 - y)).sum(axis=0)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    spec = tn / np.maximum(tn + fp, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-8)
    support = y.sum(axis=0)

    per_mech = {}
    for i, name in enumerate(MECH_NAMES):
        if not scorable(labels[:, i]):
            per_mech[name] = None
            continue
        per_mech[name] = {
            "auroc": round(roc_auc_score(labels[:, i], probs[:, i]), 4),
            "auprc": round(average_precision_score(labels[:, i], probs[:, i]), 4),
            "precision": round(float(prec[i]), 4), "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4), "specificity": round(float(spec[i]), 4),
            "threshold": round(float(thresholds[i]), 2), "support": int(support[i])}
    return per_mech


def mechanism_macro(mech_results):
    """Macro AUROC over scorable mechanisms only."""
    aurocs = [v["auroc"] for v in mech_results.values() if v is not None]
    return round(float(np.mean(aurocs)), 4) if aurocs else 0.0


# Plots

def severity_confusion(cm, path):
    """Severity confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(len(SEV_NAMES)), yticks=range(len(SEV_NAMES)),
           xlabel="Predicted", ylabel="True", title="Severity Confusion Matrix")
    ax.set_xticklabels(SEV_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(SEV_NAMES)
    threshold = np.max(cm) / 2
    for i in range(len(SEV_NAMES)):
        for j in range(len(SEV_NAMES)):
            color = "white" if cm[i][j] > threshold else "black"
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def severity_roc(curves, path):
    """Per-class ROC curves."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, c in curves.items():
        ax.plot(c["fpr"], c["tpr"], label=f"{name} ({c['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="Severity ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def severity_pr(curves, path):
    """Per-class precision-recall curves."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, c in curves.items():
        ax.plot(c["rec"], c["prec"], label=f"{name} ({c['auprc']:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision",
           title="Severity Precision-Recall Curves")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def mechanism_bars(mech_results, path):
    """Mechanism AUROC and AUPRC grouped bar chart."""
    names, aurocs, auprcs = [], [], []
    for name, vals in mech_results.items():
        if vals is None:
            continue
        names.append(name.replace("_", " "))
        aurocs.append(vals["auroc"])
        auprcs.append(vals["auprc"])
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, aurocs, w, label="AUROC")
    ax.bar(x + w / 2, auprcs, w, label="AUPRC")
    ax.set(xticks=x, ylabel="Score",
           title="Mechanism Prediction Performance", ylim=(0, 1.05))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def calibration_plot(cal, path):
    """Reliability diagram for severity predicted-class confidence."""
    if not cal["bins"]:
        return
    confs = np.array([b["confidence"] for b in cal["bins"]])
    accs = np.array([b["accuracy"] for b in cal["bins"]])
    sizes = np.array([b["n"] for b in cal["bins"]])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax.scatter(confs, accs, s=np.sqrt(sizes) * 8, alpha=0.7,
               label=f"ECE = {cal['ece']:.4f}")
    ax.set(xlabel="Predicted Confidence", ylabel="Empirical Accuracy",
           title="Severity Calibration", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# Console output

def severity_table(per_class, accuracy, macro_f1, curves):
    """Print severity table."""
    print(f"\n Severity (Accuracy: {accuracy}, Macro-F1: {macro_f1}) ")
    print(f"  {'class':12s}  {'P':>6s}  {'R/Sens':>6s}  {'Spec':>6s}  "
          f"{'F1':>6s}  {'AUROC':>6s}  {'AUPRC':>6s}  {'n':>6s}")
    for name in SEV_NAMES:
        c = per_class[name]
        auc = curves.get(name, {})
        print(f"  {name:12s}  {c['precision']:6.3f}  {c['sensitivity']:6.3f}"
              f"  {c['specificity']:6.3f}  {c['f1']:6.3f}  "
              f"{auc.get('auroc', 'N/A'):>6}  {auc.get('auprc', 'N/A'):>6}"
              f"  {c['support']:6d}")


def mechanism_table(mech_results):
    """Print mechanism table."""
    print(f"\n  Mechanism ")
    print(f"  {'mechanism':24s}  {'AUROC':>6s}  {'AUPRC':>6s}  {'P':>6s}  "
          f"{'R':>6s}  {'Spec':>6s}  {'F1':>6s}  {'Thr':>5s}  {'n':>6s}")
    for name, vals in mech_results.items():
        if vals is None:
            print(f"  {name:24s}  no positive samples")
            continue
        print(f"  {name:24s}  {vals['auroc']:6.3f}  {vals['auprc']:6.3f}  "
              f"{vals['precision']:6.3f}  {vals['recall']:6.3f}  "
              f"{vals['specificity']:6.3f}  {vals['f1']:6.3f}  "
              f"{vals['threshold']:5.2f}  {vals['support']:6d}")


def confusion_table(cm):
    """Print confusion matrix."""
    print("\n Confusion Matrix ")
    print(f"  {'':12s}  " + "  ".join(f"{n:>8s}" for n in SEV_NAMES))
    for i, row in enumerate(cm):
        print(f"  {SEV_NAMES[i]:12s}  " + "  ".join(f"{v:8d}" for v in row))


def calibration_table(cal):
    """Print calibration bin summary."""
    print(f"\n Calibration (ECE: {cal['ece']:.4f}, n: {cal['n_total']:,})")
    print(f"  {'bin':>5s}  {'range':>14s}  {'n':>7s}  {'conf':>6s}  {'acc':>6s}")
    for b in cal["bins"]:
        rng = f"[{b['lower']:.2f},{b['upper']:.2f}]"
        print(f"  {b['bin']:>5d}  {rng:>14s}  {b['n']:>7d}  "
              f"{b['confidence']:>6.3f}  {b['accuracy']:>6.3f}")


# Prediction caching for downstream analysis

def save_predictions(name, val_data, tst_data, tst_pair_ids, save_mechanism=True):
    """Cache val/test probabilities, labels, and test pair drug IDs.
    Set save_mechanism=False when the mechanism head is untrained
    (e.g. single_task ablation) to keep junk arrays out of downstream analyses.
    """
    out_dir = cfg.paths.metrics.parent / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "val_sev_probs": val_data["sev_probs"], "val_sev_labels": val_data["sev_labels"],
        "tst_sev_probs": tst_data["sev_probs"], "tst_sev_labels": tst_data["sev_labels"],
        "tst_drug_a": tst_pair_ids[:, 0], "tst_drug_b": tst_pair_ids[:, 1],
    }
    if save_mechanism:
        arrays.update({
            "val_mech_probs": val_data["mech_probs"], "val_mech_labels": val_data["mech_labels"],
            "tst_mech_probs": tst_data["mech_probs"], "tst_mech_labels": tst_data["mech_labels"],
        })
    np.savez(out_dir / f"{name}.npz", **arrays)
    print(f"Predictions cached: {out_dir / (name + '.npz')}")


def cached_predictions(name):
    """Load cached val and test arrays. Mechanism keys absent if not saved."""
    data = np.load(cfg.paths.metrics.parent / "predictions" / f"{name}.npz",
                   allow_pickle=True)
    has_mech = "tst_mech_probs" in data.files
    val = {"sev_probs": data["val_sev_probs"], "sev_labels": data["val_sev_labels"]}
    tst = {"sev_probs": data["tst_sev_probs"], "sev_labels": data["tst_sev_labels"]}
    if has_mech:
        val["mech_probs"] = data["val_mech_probs"]
        val["mech_labels"] = data["val_mech_labels"]
        tst["mech_probs"] = data["tst_mech_probs"]
        tst["mech_labels"] = data["tst_mech_labels"]
    return {"val": val, "tst": tst, "has_mech": has_mech,
            "tst_drug_a": data["tst_drug_a"], "tst_drug_b": data["tst_drug_b"]}


# Output orchestration

def plot_files(curves, cm, mech, cal):
    """Write all evaluation plots."""
    cfg.paths.plots.mkdir(parents=True, exist_ok=True)
    severity_confusion(cm, cfg.paths.plots / "confusion_matrix.png")
    severity_roc(curves, cfg.paths.plots / "roc_curves.png")
    severity_pr(curves, cfg.paths.plots / "pr_curves.png")
    mechanism_bars(mech, cfg.paths.plots / "mechanism_performance.png")
    calibration_plot(cal, cfg.paths.plots / "calibration.png")
    print(f"\nPlots saved: {cfg.paths.plots}")


def metric_files(per_class, curves, cm, accuracy, macro_f1, mech, cal):
    """Write metrics JSON."""
    cfg.paths.metrics.mkdir(parents=True, exist_ok=True)
    results = {
        "accuracy": accuracy, "macro_f1": macro_f1,
        "severity_per_class": per_class,
        "severity_auroc_auprc": {k: {"auroc": v["auroc"], "auprc": v["auprc"]}
                                 for k, v in curves.items()},
        "confusion_matrix": cm,
        "mechanism": {k: v for k, v in mech.items() if v is not None},
        "mech_macro_auroc": mechanism_macro(mech),
        "calibration": cal,
    }
    out_path = cfg.paths.metrics / "test_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved: {out_path}")


def test_metrics(val_data, tst_data):
    """Thresholds from val, all metrics from test."""
    thresholds = mechanism_thresholds(val_data["mech_probs"], val_data["mech_labels"])
    print("\nOptimal thresholds (val): "
          + ", ".join(f"{n}={t:.2f}" for n, t in zip(MECH_NAMES, thresholds)))
    curves = severity_curves(tst_data["sev_probs"], tst_data["sev_labels"])
    per_class, cm, accuracy, macro_f1 = severity_classwise(
        tst_data["sev_probs"], tst_data["sev_labels"])
    mech = mechanism_scores(
        tst_data["mech_probs"], tst_data["mech_labels"], thresholds)
    cal = severity_calibration(tst_data["sev_probs"], tst_data["sev_labels"])
    return curves, per_class, cm, accuracy, macro_f1, mech, cal


# Print out

def evaluate_ddi():
    device = preferred_device(cfg.train.device)
    seed_everything(cfg.train.seed)

    val, tst = eval_batches(cfg.train.batch_size, cfg.train.seed, device)
    print(f"Val: {len(val.dataset)}, Test: {len(tst.dataset)}")

    model = DDIModel(cfg.atom, cfg.gnn).to(device)
    path = cfg.paths.models / "ddi_best.pt"
    checkpoint_weights(model, path, device)
    print(f"Loaded: {path}")

    print("Running")
    val_data = inference_outputs(model, val, device)
    flush_device_cache(device)
    tst_data = inference_outputs(model, tst, device)
    del model
    flush_device_cache(device)

    tst_pair_ids = tst.dataset.pairs[["drug_a", "drug_b"]].values
    save_predictions("gnn", val_data, tst_data, tst_pair_ids)
    curves, per_class, cm, accuracy, macro_f1, mech, cal = test_metrics(
        val_data, tst_data)
    severity_table(per_class, accuracy, macro_f1, curves)
    confusion_table(cm)
    mechanism_table(mech)
    calibration_table(cal)
    print(f"\nMechanism macro-AUROC: {mechanism_macro(mech):.4f}")
    plot_files(curves, cm, mech, cal)
    metric_files(per_class, curves, cm, accuracy, macro_f1, mech, cal)


if __name__ == "__main__":
    evaluate_ddi()