[![CI](https://github.com/jsf3467v/multi-task-ddi-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/jsf3467v/multi-task-ddi-audit/actions/workflows/ci.yml)

# Multi-task DDI Benchmark Audit for Tuberculosis-HIV Pharmacotherapy

A methodology audit of a multi-task drug-drug interaction (DDI) prediction benchmark constructed from DrugBank and DDInter, focused on tuberculosis-HIV pharmacotherapy. Seven model variants (a full GNN+PK model, three GNN ablations, and three feature-based baselines) serve as instruments to surface three patterns by which standard benchmark assembly choices systematically disadvantage drugs central to global health.

## About this project

Final capstone project for AI in Healthcare at Johns Hopkins University, Baltimore, MD (Spring 2026). The paper is formatted in the PMLR / MLHC 2026 template because the course assignment specified that format. **The work was not submitted to MLHC and has not been peer-reviewed.** The repository is shared as a portfolio artifact demonstrating multi-task GNN and baseline implementation, statistical evaluation rigor (bootstrap CIs, McNemar pairwise tests, calibration analysis), and an honest methodological audit — including discovery and correction of a label-leakage issue in the input features.

## Findings

**1. Cross-database name resolution silently drops drugs labeled by WHO international nonproprietary names.** An exact-string DDInter → DrugBank merge removed 1,825 documented interactions (1.5% of DDInter records), including all 284 pairs involving rifampin — the most clinically important CYP3A4 inducer in TB pharmacotherapy. The asymmetry is structural: DDInter uses WHO INN (`rifampicin`, `salbutamol`), DrugBank uses American generic names (`rifampin`, `albuterol`). An alias map recovered three matchable mismatches (rifampicin, salbutamol, interferon alfa-2a variants). Drugs registered by WHO INN are disproportionately affected.

**2. Random pair-level splits with broad-pool negative sampling saturate the drug space and make cold-start evaluation structurally impossible.** Of 13,002 test pairs, 12,997 (99.96%) are warm-warm (both drugs appear in training), 5 are warm-cold, and 0 are cold-cold. The benchmark cannot answer whether a model trained on it generalizes to a novel drug. Newer agents in current WHO TB and HIV guidelines (delamanid, pretomanid, dolutegravir) lack sufficient test coverage under this design.

**3. Aggregate accuracy masks architecture-specific failure modes on rare severity classes.** The full GNN+PK and the MLP ensemble achieve different F1 on the rare Minor class (0.51 vs 0.76) through *opposite* mechanisms. GNN+PK has high recall and low precision on Minor (R = 0.88, P = 0.36); the MLP has high precision and lower recall (P = 0.87, R = 0.68). Bootstrap 95% CIs on per-class precision do not overlap, confirming the two architectures are statistically distinct in their failure modes. In clinical deployment, they would generate qualitatively different alert behavior despite comparable benchmark scores.

**Plus a methodology contribution:** indirect label leakage in the pharmacokinetic feature vector was identified and corrected during the audit. CYP-inducer and CYP-inhibitor flags carried 2.5× and 2.2× lift on the corresponding mechanism labels because both originate from the same DrugBank curation process. 20 of the original 30 PK columns were removed; all reported results use the 10-dimensional leakage-corrected vector.

## Headline results

Single fixed test set (n = 13,002). Baseline rows are 5-fold CV ensemble means with bootstrap 95% CIs. GNN rows are single best-validation checkpoints trained on the leakage-corrected PK vector.

| Model                  | Accuracy             | Macro-F1             | Sev AUROC | Mech AUROC | ECE       |
| ---------------------- | -------------------- | -------------------- | --------- | ---------- | --------- |
| RF (5-fold ensemble)   | 0.849 [0.843, 0.855] | 0.707 [0.689, 0.722] | 0.959     | 0.960      | 0.017     |
| **MLP (5-fold ensemble)** | **0.955 [0.951, 0.958]** | **0.896 [0.884, 0.907]** | **0.991** | **0.984**  | 0.011     |
| XGB (5-fold ensemble)  | 0.890 [0.885, 0.895] | 0.781 [0.766, 0.795] | 0.973     | 0.974      | 0.014     |
| GNN+PK (full)          | 0.862 [0.856, 0.868] | 0.748 [0.736, 0.759] | 0.973     | 0.948      | **0.008** |
| GNN-only               | 0.849                | 0.742                | 0.970     | 0.947      | 0.010     |
| PK-only                | 0.635                | 0.472                | 0.828     | 0.823      | 0.106     |
| Single-task            | 0.851                | 0.727                | 0.971     | N/A        | 0.009     |

The MLP outperforms the GNN+PK by 9.3 accuracy points and 14.8 macro-F1 points on the full test set. The gap is significant under McNemar (p < 10⁻⁶) with a paired-bootstrap 95% CI of [0.087, 0.099]. The full manuscript expands this into per-tier TB results, where calibration degrades sharply across all models on the small TB-relevant subgroups — GNN+PK ECE goes from 0.008 on the full test to 0.091 on first-line TB pairs and 0.135 on ARV co-administration pairs.

## Reproducing the results

The repository has two source directories: `src/` holds the GNN, the shared scoring pipeline, and the audit modules; `ablation and baseline/` holds the baseline classifiers and the GNN ablations. Each script resolves paths relative to itself, so commands below assume your shell is in the script's own directory.

### Setup

Tested on Python 3.10+ with PyTorch 2.x. Apple Silicon (M-series) is the primary supported device via the MPS backend; CUDA and CPU also work via `cfg.train.device = "auto"` in `src/config.py`.

```bash
pip install -r requirements.txt
```

### Data

DrugBank and DDInter must be obtained separately:

- **DrugBank** (version 5.1.15): register at <https://go.drugbank.com/> and download the full XML release. Place under `Datasets/raw/`.
- **DDInter**: download from <http://ddinter.scbdd.com/>. Place under `Datasets/raw/`.

Then run `EDA/EDA.ipynb` end-to-end. This produces four processed tables in `Datasets/processed/`:

- `ddi_pairs.csv`
- `ddinter_matched.csv` (post alias-correction merge)
- `drug_smiles.csv`
- `pk_features.csv` (the 10-dimensional leakage-corrected PK vector)

### Train

All training uses seed 42, set in `src/config.py`.

In `ablation and baseline/`:

```bash
python baseline.py        # RF, MLP, XGB with 5-fold stratified CV (~30 min on M4 Max)
python ablation.py        # GNN-only, PK-only, single-task ablations (~3-5 hrs total)
```

In `src/`:

```bash
python train_gnn.py       # full multi-task GNN+PK with early stopping (~1-2 hrs)
```

Each script accepts a positional argument to train a single variant (e.g., `python baseline.py rf`, `python ablation.py pk_only`).

### Per-model evaluation

These produce JSON metrics in `results/metrics/` and cached prediction arrays in `results/predictions/`. The cross-model audit scripts read from the latter.

In `ablation and baseline/`:

```bash
python baseline_eval.py
python ablation_eval.py
```

In `src/`:

```bash
python evaluate.py        # GNN+PK, writes test_metrics.json + plots
```

### Cross-model audit analyses

These read cached predictions from `results/predictions/`. Run from `src/`:

```bash
python agreement.py       # four-cell agreement matrix + class-stratified disagreement
python cold_start.py      # warm-warm / warm-cold / cold-cold partition metrics
python stat.py            # McNemar pairwise tests + paired bootstrap CIs
```

### TB-specific analysis

Stratifies test pairs into first-line / second-line / ARV co-administration / comedications tiers, reports per-tier accuracy and calibration, and runs the rifampin CYP-induction precision/recall probe (n = 16). Run from `src/`:

```bash
python tb_analysis.py gnn
python tb_analysis.py mlp
python tb_analysis.py rf
python tb_analysis.py xgb
```

## Repository layout

```
.
├── .gitignore
├── README.md
├── requirements.txt
├── manuscript.pdf                     # Audit write-up (MLHC template, not submitted)
├── EDA/
│   └── EDA.ipynb                      # Data preprocessing notebook (raw → processed)
├── src/                               # Core model + audit modules
│   ├── config.py                      # ProjectConfig dataclasses
│   ├── feature_engineering.py         # SMILES → PyG graph, severity/mechanism labels
│   ├── gnn.py                         # GATv2 encoder, PK branch, pair classifier
│   ├── stratify.py                    # Severity-stratified 80/10/10 split
│   ├── train_gnn.py                   # Multi-task training loop
│   ├── evaluate.py                    # GNN evaluation + shared scoring pipeline
│   ├── agreement.py                   # Cross-model agreement analysis
│   ├── cold_start.py                  # Warm/cold partition analysis
│   ├── stat.py                        # Statistical significance tests
│   └── tb_analysis.py                 # TB drug-tier stratified analysis
├── ablation and baseline/             # Baseline + ablation training and eval
│   ├── baseline.py                    # RF / MLP / XGB training
│   ├── baseline_eval.py               # Baseline 5-fold CV evaluation + ensemble
│   ├── ablation.py                    # GNN-only / PK-only / single-task training
│   └── ablation_eval.py               # Ablation evaluation
└── results/
    ├── metrics/                       # JSON outputs from audit runs (committed for reference)
    │   ├── ablation_results.json
    │   ├── agreement_analysis.json
    │   ├── baseline_results.json
    │   ├── cold_start_analysis.json
    │   ├── stat_tests.json
    │   ├── tb_analysis_{gnn,mlp,rf,xgb}.json
    │   └── test_metrics.json
    └── plots/                         # Generated figures (committed for reference)
        ├── calibration.png
        ├── confusion_matrix.png
        ├── mechanism_performance.png
        ├── pr_curves.png
        ├── roc_curves.png
        └── eda/
            ├── lipinski_properties.png
            ├── mechanism_distribution.png
            └── severity_distribution.png
```

**Not committed** (regenerated locally; see `.gitignore`):

- `Datasets/raw/` — DrugBank + DDInter source files (obtain separately, license-restricted)
- `Datasets/processed/` — generated by `EDA/EDA.ipynb` from raw
- `models/` — model checkpoints (`.pt` for GNN/ablations, `.pkl` for baseline folds)
- `results/predictions/` — cached probability arrays (`.npz`) used by the cross-model audit scripts

## Limitations

This audit evaluates a single benchmark (DrugBank + DDInter). Cross-benchmark validation is needed before generalizing the findings. Mechanism labels are derived via regex pattern matching against DrugBank's free-text descriptions; 19.1% of positive pairs match no keyword and are excluded from mechanism training, so the reported mechanism-head performance reflects the regex labels rather than ground-truth pharmacology.

Treating unlabeled drug pairs as negatives during training assumes a positive-unlabeled framing. Because DrugBank records only known interactions, any pair without a recorded interaction is treated as non-interacting. This may underestimate false negatives, particularly for less-studied drugs.

The TB cohort includes 18 drugs that passed the alias-corrected merge with sufficient test coverage. Newer agents such as delamanid, pretomanid, and dolutegravir are excluded due to insufficient test pairs — itself a manifestation of finding 2.

The GNN encoder uses one-hot atom (49-dim) and bond (14-dim) features with no pretraining. Variants with pretrained encoders might shift accuracy and AUROC values but are unlikely to overturn the audit's three main conclusions, since the fundamental representational asymmetry — Morgan fingerprints encode 2,048 bits of substructural information directly, while the GNN must learn it from 104,011 training pairs — remains.

## Manuscript

The full write-up is in `manuscript.pdf`. Section 4 documents the data pipeline along with the alias-correction and PK-leakage discoveries. Section 5 presents the three audit findings with per-tier and per-class breakdowns. Section 6 discusses implications for clinical decision support and regulatory frameworks.

## Citation

This is unpublished course work. If you reference it, please cite as:

```bibtex
@misc{keith_ddi_audit_2026,
  author = {Arlene Keith},
  title  = {Validation Gaps in Drug-Drug Interaction Prediction Benchmarks for Tuberculosis-HIV Pharmacotherapy},
  year   = {2026},
  note   = {Capstone project, AI in Healthcare, Johns Hopkins University, Spring 2026, unpublished},
  url    = {https://github.com/jsf3467v/multi-task-ddi-audit}
}
```

## Acknowledgements

DrugBank (Wishart et al., 2018) and DDInter (Xiong et al., 2022) are the data sources. The audit framing builds on prior clinical ML benchmark critiques by Wong et al. (2021), Kapoor and Narayanan (2023), Huang et al. (2021), and Shen et al. (2025). Full references are in the manuscript.
