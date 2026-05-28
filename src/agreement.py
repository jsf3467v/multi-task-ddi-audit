"""
Pairwise model agreement analysis on severity predictions.
Reports the four-cell agreement matrix
plus per-class breakdown of who is right when models disagree.
"""

import json

import numpy as np

from config import ProjectConfig
from evaluate import cached_predictions

cfg = ProjectConfig()
SEV_NAMES = list(cfg.data.severity_map.keys())


# Prediction loading

def predictions_dir():
    return cfg.paths.metrics.parent / "predictions"


def cached_models():
    """Names of models with cached predictions on disk."""
    pdir = predictions_dir()
    if not pdir.exists():
        return []
    return sorted(p.stem for p in pdir.glob("*.npz"))


def severity_arrays(name):
    """Predicted classes and labels for a single model."""
    preds = cached_predictions(name)
    return preds["tst"]["sev_probs"].argmax(axis=1), preds["tst"]["sev_labels"]


# Agreement statistics

def agreement_matrix(a_pred, b_pred, labels):
    """Four-cell breakdown: both correct, only A, only B, both wrong."""
    a_ok = a_pred == labels
    b_ok = b_pred == labels
    n = len(labels)
    return {
        "both_correct": int(np.sum(a_ok & b_ok)),
        "a_only_correct": int(np.sum(a_ok & ~b_ok)),
        "b_only_correct": int(np.sum(~a_ok & b_ok)),
        "both_wrong": int(np.sum(~a_ok & ~b_ok)),
        "n_total": int(n),
        "agreement_rate": round(float(np.mean(a_pred == b_pred)), 4),
    }


def disagreement_by_class(a_pred, b_pred, labels):
    """When A and B disagree, who is right per true severity class."""
    disagree = a_pred != b_pred
    out = {}
    for i, name in enumerate(SEV_NAMES):
        mask = disagree & (labels == i)
        n = int(mask.sum())
        if n == 0:
            out[name] = None
            continue
        a_wins = int(np.sum(a_pred[mask] == labels[mask]))
        b_wins = int(np.sum(b_pred[mask] == labels[mask]))
        out[name] = {"n_disagreement": n,
                     "a_correct": a_wins,
                     "b_correct": b_wins,
                     "neither_correct": n - a_wins - b_wins}
    return out


# Console output

def matrix_table(name_a, name_b, m, by_class):
    """Print agreement matrix and per-class disagreement breakdown."""
    print(f"\n  {name_a} vs {name_b}  (n={m['n_total']})")
    print(f"    both correct:        {m['both_correct']:>6d}  "
          f"({m['both_correct'] / m['n_total']:>5.1%})")
    print(f"    {name_a} only correct: {m['a_only_correct']:>6d}  "
          f"({m['a_only_correct'] / m['n_total']:>5.1%})")
    print(f"    {name_b} only correct: {m['b_only_correct']:>6d}  "
          f"({m['b_only_correct'] / m['n_total']:>5.1%})")
    print(f"    both wrong:          {m['both_wrong']:>6d}  "
          f"({m['both_wrong'] / m['n_total']:>5.1%})")
    print(f"    overall agreement:   {m['agreement_rate']:>6.1%}")
    print(f"\n    Disagreement breakdown by true class:")
    print(f"    {'class':12s}  {'n':>6s}  {name_a + ' wins':>10s}  "
          f"{name_b + ' wins':>10s}  {'neither':>8s}")
    for cls, d in by_class.items():
        if d is None:
            print(f"    {cls:12s}  {'0':>6s}  no disagreements")
            continue
        print(f"    {cls:12s}  {d['n_disagreement']:>6d}  "
              f"{d['a_correct']:>10d}  {d['b_correct']:>10d}  "
              f"{d['neither_correct']:>8d}")


# Print out

def model_agreement():
    """Pairwise agreement analysis across all cached predictions."""
    names = cached_models()
    if len(names) < 2:
        print(f"Need at least 2 cached models, found {len(names)}.")
        return
    print(f"Cached models: {', '.join(names)}")

    preds = {n: severity_arrays(n) for n in names}
    results = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            a_pred, labels = preds[a]
            b_pred, _ = preds[b]
            matrix = agreement_matrix(a_pred, b_pred, labels)
            by_class = disagreement_by_class(a_pred, b_pred, labels)
            matrix_table(a, b, matrix, by_class)
            results[f"{a} vs {b}"] = {"matrix": matrix,
                                      "by_class": by_class}

    cfg.paths.metrics.mkdir(parents=True, exist_ok=True)
    out = cfg.paths.metrics / "agreement_analysis.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    model_agreement()