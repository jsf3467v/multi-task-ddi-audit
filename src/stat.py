"""
Statistical significance testing for cross-model comparison.
McNemar's test on paired severity predictions, bootstrap CIs on
headline metrics, and per-class CIs for severity P/R/F1.
Predictions written by evaluate.py and baseline_eval.py
to avoid re-running inference.
"""

import json
from math import comb, erf, sqrt

import numpy as np

from config import ProjectConfig, seed_everything
from evaluate import (
    cached_predictions,
    severity_curves, severity_classwise, severity_macro, mechanism_macro,
    mechanism_thresholds, mechanism_scores,
)

cfg = ProjectConfig()
N_BOOTSTRAP = 1000
ALPHA = 0.05
SEV_NAMES = list(cfg.data.severity_map.keys())
PER_CLASS_METRICS = ("precision", "recall", "f1")


# Prediction inventory

def predictions_dir():
    return cfg.paths.metrics.parent / "predictions"


def cached_models():
    """Names of models with cached predictions on disk."""
    pdir = predictions_dir()
    if not pdir.exists():
        return []
    return sorted(p.stem for p in pdir.glob("*.npz"))


# McNemar's test on severity correctness

def mcnemar_pvalue(b, c):
    """Two-sided McNemar p-value (exact for small counts, normal otherwise)."""
    n = b + c
    if n == 0:
        return 1.0
    if n < 25:
        k = min(b, c)
        tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
        return min(1.0, 2 * tail)
    chi2 = (abs(b - c) - 1) ** 2 / n
    return 1.0 - erf(sqrt(chi2 / 2))


def mcnemar_severity(name_a, name_b, preds):
    """Pairwise McNemar's test on severity correctness."""
    a = preds[name_a]["tst"]
    b = preds[name_b]["tst"]
    a_correct = a["sev_probs"].argmax(axis=1) == a["sev_labels"]
    b_correct = b["sev_probs"].argmax(axis=1) == b["sev_labels"]
    a_only = int(np.sum(a_correct & ~b_correct))
    b_only = int(np.sum(b_correct & ~a_correct))
    p = mcnemar_pvalue(a_only, b_only)
    return {"a_only_correct": a_only, "b_only_correct": b_only,
            "n_total": int(len(a_correct)), "p_value": round(float(p), 6)}


# Bootstrap CI

def bootstrap_indices(n, n_boot, rng):
    """Resample indices for bootstrap. Returns (n_boot, n) integer matrix."""
    return rng.integers(0, n, size=(n_boot, n))


def severity_on_indices(probs, labels, idx):
    """Severity accuracy, F1, macro-AUROC on a bootstrap sample."""
    sp, sl = probs[idx], labels[idx]
    curves = severity_curves(sp, sl)
    _, _, accuracy, macro_f1 = severity_classwise(sp, sl)
    macro_auroc, _ = severity_macro(curves)
    return accuracy, macro_f1, macro_auroc


def class_triplet_on_indices(probs, labels, idx):
    """Per-class precision, recall, F1 dict on a bootstrap sample."""
    per_class, _, _, _ = severity_classwise(probs[idx], labels[idx])
    return {c: (per_class[c]["precision"],
                per_class[c]["sensitivity"],
                per_class[c]["f1"]) for c in SEV_NAMES}


def mech_on_indices(mech_probs, mech_labels, idx, thresholds):
    """Mechanism macro-AUROC on a bootstrap sample."""
    mp, ml = mech_probs[idx], mech_labels[idx]
    return mechanism_macro(mechanism_scores(mp, ml, thresholds))


def bootstrap_ci(values, alpha):
    """Percentile CI from bootstrap distribution."""
    lo = float(np.percentile(values, 100 * alpha / 2))
    hi = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return {"mean": round(float(np.mean(values)), 4),
            "lo": round(lo, 4), "hi": round(hi, 4)}


def class_samples_init():
    """Empty per-class metric collector."""
    return {c: {m: [] for m in PER_CLASS_METRICS} for c in SEV_NAMES}


def bootstrap_samples(tst, has_mech, thresholds, idx_matrix):
    """Collect bootstrap samples of macros, per-class triplets, and mechanism."""
    accs, f1s, sevs, mechs = [], [], [], []
    class_samples = class_samples_init()
    for idx in idx_matrix:
        a, f, s = severity_on_indices(tst["sev_probs"], tst["sev_labels"], idx)
        accs.append(a); f1s.append(f); sevs.append(s)
        for c, (p, r, f1) in class_triplet_on_indices(
                tst["sev_probs"], tst["sev_labels"], idx).items():
            class_samples[c]["precision"].append(p)
            class_samples[c]["recall"].append(r)
            class_samples[c]["f1"].append(f1)
        if has_mech:
            mechs.append(mech_on_indices(
                tst["mech_probs"], tst["mech_labels"], idx, thresholds))
    return accs, f1s, sevs, mechs, class_samples


def bootstrap_metrics(name, preds, thresholds, n_boot, rng):
    """Bootstrap CIs for severity macros, per-class P/R/F1, and mechanism."""
    tst = preds[name]["tst"]
    has_mech = preds[name]["has_mech"]
    idx_matrix = bootstrap_indices(len(tst["sev_labels"]), n_boot, rng)
    accs, f1s, sevs, mechs, class_samples = bootstrap_samples(
        tst, has_mech, thresholds, idx_matrix)
    out = {
        "accuracy": bootstrap_ci(accs, ALPHA),
        "macro_f1": bootstrap_ci(f1s, ALPHA),
        "severity_macro_auroc": bootstrap_ci(sevs, ALPHA),
        "severity_per_class": {c: {m: bootstrap_ci(v, ALPHA)
                                    for m, v in d.items()}
                               for c, d in class_samples.items()},
    }
    if has_mech:
        out["mechanism_macro_auroc"] = bootstrap_ci(mechs, ALPHA)
    return out


# Paired bootstrap on accuracy difference

def paired_bootstrap_accuracy(name_a, name_b, preds, n_boot, rng):
    """Paired bootstrap CI on (accuracy_a - accuracy_b). Excludes 0 -> significant."""
    a_pred = preds[name_a]["tst"]["sev_probs"].argmax(axis=1)
    b_pred = preds[name_b]["tst"]["sev_probs"].argmax(axis=1)
    labels = preds[name_a]["tst"]["sev_labels"]
    n = len(labels)
    idx_matrix = bootstrap_indices(n, n_boot, rng)
    diffs = np.array([
        (a_pred[idx] == labels[idx]).mean() - (b_pred[idx] == labels[idx]).mean()
        for idx in idx_matrix
    ])
    ci = bootstrap_ci(diffs, ALPHA)
    ci["significant"] = bool(ci["lo"] > 0 or ci["hi"] < 0)
    return ci


# Console output

def cell(d):
    return f"{d['mean']:.4f} [{d['lo']:.4f},{d['hi']:.4f}]"


def model_table(metrics):
    """Print bootstrap CI summary table per model."""
    print(f"\n  {'Model':18s}  {'Accuracy':>22s}  {'Macro-F1':>22s}  "
          f"{'S-AUROC':>22s}  {'M-AUROC':>22s}")
    for name, m in metrics.items():
        sev = [cell(m[k]) for k in
               ("accuracy", "macro_f1", "severity_macro_auroc")]
        mech = (cell(m["mechanism_macro_auroc"])
                if "mechanism_macro_auroc" in m else "N/A")
        cells = sev + [mech]
        print(f"  {name:18s}  " + "  ".join(f"{c:>22s}" for c in cells))


def per_class_table(metrics):
    """Print per-class P/R/F1 with bootstrap CIs for each model."""
    print(f"\n  Per-class severity (95% CI from {N_BOOTSTRAP} bootstrap resamples)")
    for name, m in metrics.items():
        print(f"\n  {name}")
        print(f"    {'class':10s}  {'Precision':>22s}  {'Recall':>22s}  {'F1':>22s}")
        for c in SEV_NAMES:
            row = m["severity_per_class"][c]
            cells = [cell(row[k]) for k in PER_CLASS_METRICS]
            print(f"    {c:10s}  " + "  ".join(f"{x:>22s}" for x in cells))


def comparison_table(comparisons):
    """Print pairwise comparison table."""
    print(f"\n  {'Pair':25s}  {'Δ accuracy (95% CI)':>28s}  {'McNemar p':>10s}  "
          f"{'Significant':>11s}")
    for key, c in comparisons.items():
        d = c["accuracy_difference"]
        diff_str = f"{d['mean']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]"
        sig = "yes" if c["mcnemar"]["p_value"] < ALPHA else "no"
        print(f"  {key:25s}  {diff_str:>28s}  {c['mcnemar']['p_value']:>10.6f}  {sig:>11s}")


# Print out

def per_model_bootstrap(names, preds, rng):
    """Bootstrap CIs for each model. Threshold derived per-model from val."""
    out = {}
    for name in names:
        thresholds = (mechanism_thresholds(
            preds[name]["val"]["mech_probs"], preds[name]["val"]["mech_labels"])
            if preds[name]["has_mech"] else None)
        print(f"  Bootstrapping {name} ({N_BOOTSTRAP} resamples)")
        out[name] = bootstrap_metrics(name, preds, thresholds,
                                       N_BOOTSTRAP, rng)
    return out


def pairwise_tests(names, preds, rng):
    """McNemar + paired bootstrap CI for every model pair."""
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            print(f"  Paired test: {a} vs {b}")
            out[f"{a} vs {b}"] = {
                "mcnemar": mcnemar_severity(a, b, preds),
                "accuracy_difference": paired_bootstrap_accuracy(
                    a, b, preds, N_BOOTSTRAP, rng),
            }
    return out


def stat_tests():
    """Bootstrap CIs per model, per-class CIs, and pairwise McNemar and accuracy CI."""
    seed_everything(cfg.train.seed)
    rng = np.random.default_rng(cfg.train.seed)

    names = cached_models()
    if not names:
        print(f"No cached predictions in {predictions_dir()}.")
        return
    print(f"Cached models: {', '.join(names)}")
    preds = {n: cached_predictions(n) for n in names}

    per_model = per_model_bootstrap(names, preds, rng)
    model_table(per_model)
    per_class_table(per_model)
    comparisons = pairwise_tests(names, preds, rng)
    comparison_table(comparisons)

    cfg.paths.metrics.mkdir(parents=True, exist_ok=True)
    out = cfg.paths.metrics / "stat_tests.json"
    with open(out, "w") as f:
        json.dump({"per_model": per_model, "pairwise": comparisons,
                   "n_bootstrap": N_BOOTSTRAP, "alpha": ALPHA},
                  f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    stat_tests()