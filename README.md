[![CI](https://github.com/jsf3467v/multi-task-ddi-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/jsf3467v/multi-task-ddi-audit/actions/workflows/ci.yml)

# Multi-task DDI Benchmark Audit for Tuberculosis-HIV Pharmacotherapy

This project audits a multi-task drug-drug interaction (DDI) prediction benchmark for tuberculosis-HIV pharmacotherapy, built from DrugBank and DDInter. Across seven model variants, including a full GNN+PK model, three GNN ablations, and three feature-based baselines, the audit reveals three ways that standard benchmark assembly choices systematically disadvantage drugs central to global health.

## About this project

This is a final capstone project in AI in Healthcare at Johns Hopkins University, Baltimore, MD (Spring 2026). The paper uses the PMLR / MLHC 2026 template because the course assignment required it. **The work was not submitted to MLHC and has not been peer-reviewed.** The repository is shared as a portfolio artifact demonstrating a multi-task GNN and baseline implementation, rigorous statistical evaluation (bootstrap CIs, McNemar pairwise tests, calibration analysis), and an honest methodological audit that includes the discovery and correction of a label-leakage issue in the input features.

## Findings

**1. Cross-database name resolution silently omits drugs labeled with WHO international nonproprietary names.** Merging DDInter with DrugBank based on exact string matches removed 1,824 documented interactions, about 0.82% of the 222,383 DDInter records. This included all 284 pairs involving rifampin, the most clinically significant CYP3A4 inducer used in TB treatment. The discrepancy stems from structural differences: DDInter uses WHO INN names (`rifampicin`, `salbutamol'), whereas DrugBank uses American generic names (`rifampin`, `albuterol`). An alias mapping recovered four matchable mismatches with rifampicin, salbutamol, interferon alfa-2a, and a COVID-19 vaccine name variant—resolving all but 2 of the interactions initially dropped. Drugs registered by WHO INN are more likely to be affected.

**2. Random pair-level splits with broad-pool negative sampling saturate the drug space and make cold-start evaluation structurally impossible.** Of 13,002 test pairs, 12,997 (99.96%) are warm-warm (both drugs appear in training), 5 are warm-cold, and 0 are cold-cold. The benchmark cannot answer whether a model trained on it generalizes to a novel drug. Newer agents in current WHO TB and HIV guidelines (delamanid, pretomanid, dolutegravir) lack sufficient test coverage under this design.

**3. Aggregate accuracy masks reveal architecture-specific failure modes in rare severity classes.** The complete GNN+PK model and the MLP ensemble display different F1 scores on the rare Minor class (0.51 vs 0.76), due to *opposite* mechanisms. GNN+PK features high recall but low precision for Minor (R = 0.88, P = 0.36), whereas the MLP exhibits high precision but lower recall (P = 0.87, R = 0.68). Bootstrap 95% confidence intervals for per-class precision do not overlap, indicating statistically distinct failure modes between the two architectures. In clinical use, they would produce qualitatively different alert behaviors despite similar benchmark performance.

**A methodological note.** During the audit, indirect label leakage in the pharmacokinetic feature vector was recognized and addressed. The CYP-inducer and CYP-inhibitor flags showed 2.2× and 2.5× increases in their respective mechanism labels, as both are derived from the same DrugBank curation. Twenty of the original thirty PK columns were eliminated, and all reported outcomes now utilize the 10-dimensional leakage-corrected vector.

## Headline results

Single fixed test set (n = 13,002). Baseline results represent the 5-fold cross-validation ensemble means with bootstrap 95% confidence intervals. GNN results are based on the single best-validation checkpoints trained on the leakage-corrected PK vector.

| Model                     | Accuracy                 | Macro-F1                 | Sev AUROC | Mech AUROC | ECE       |
| ------------------------- | ------------------------ | ------------------------ | --------- | ---------- | --------- |
| RF (5-fold ensemble)      | 0.849 [0.843, 0.855]     | 0.707 [0.689, 0.722]     | 0.959     | 0.960      | 0.017     |
| **MLP (5-fold ensemble)** | **0.955 [0.951, 0.958]** | **0.896 [0.884, 0.907]** | **0.991** | **0.984**  | 0.011     |
| XGB (5-fold ensemble)     | 0.890 [0.885, 0.895]     | 0.781 [0.766, 0.795]     | 0.973     | 0.974      | 0.014     |
| GNN+PK (full)             | 0.862 [0.856, 0.868]     | 0.748 [0.736, 0.759]     | 0.973     | 0.948      | **0.008** |
| GNN-only                  | 0.849                    | 0.742                    | 0.970     | 0.947      | 0.010     |
| PK-only                   | 0.635                    | 0.472                    | 0.828     | 0.823      | 0.106     |
| Single-task               | 0.851                    | 0.727                    | 0.971     | N/A        | 0.009     |

The MLP outperforms GNN+PK by 9.3 accuracy points and 14.8 macro-F1 points on the full test set. The difference is highly significant under McNemar (p < 10⁻⁶), with a paired-bootstrap 95% confidence interval of [0.087, 0.099]. The complete paper further details this with per-tier TB results, showing a sharp decline in calibration across all models on the small TB-relevant subgroups. Specifically, GNN+PK's ECE increases from 0.008 on the full test to 0.091 on first-line TB pairs and 0.135 on ARV co-administration pairs.

## Reproducing the results

The repository has two source directories. `src/` contains the GNN, the shared scoring pipeline, and the audit modules, while `ablation and baseline/` contains the baseline classifiers and the GNN ablations. Each script resolves paths relative to itself, so the commands below assume your shell is in the script's directory.

### Setup

Tested on Python 3.10+ with PyTorch 2.x. Apple Silicon (M-series) is the primary supported device via the MPS backend, and CUDA and CPU also work via `cfg.train.device = "auto"` in `src/config.py`.

```
pip install -r requirements.txt
```

### Data

DrugBank and DDInter must be obtained separately.

- **DrugBank** (version 5.1.15). Register at <https://go.drugbank.com/> and download the full XML release, then place it under `Datasets/raw/`.
- **DDInter**. Download from <http://ddinter.scbdd.com/> and place it under `Datasets/raw/`.

Then run `EDA/EDA.ipynb` end to end. This produces four processed tables in `Datasets/processed/`.

- `ddi_pairs.csv`
- `ddinter_matched.csv` (post alias-correction merge)
- `drug_smiles.csv`
- `pk_features.csv` (the 10-dimensional leakage-corrected PK vector)

### Train

All training uses seed 42, set in `src/config.py`. Run from `ablation and baseline/`.

```
python baseline.py        # RF, MLP, XGB with 5-fold stratified CV (~30 min on M4 Max)
python ablation.py        # GNN-only, PK-only, single-task ablations (~3-5 hrs total)
```

Run from `src/`.

```
python train_gnn.py       # full multi-task GNN+PK with early stopping (~1-2 hrs)
```

Each script accepts a positional argument to train a single variant (for example, `python baseline.py rf` or `python ablation.py pk_only`).

### Per-model evaluation

These produce JSON metrics in `results/metrics/` and cached prediction arrays in `results/predictions/`. The cross-model audit scripts read from the latter. Run from `ablation and baseline/`.

```
python baseline_eval.py
python ablation_eval.py
```

Run from `src/`.

```
python evaluate.py        # GNN+PK, writes test_metrics.json + plots
```

### Cross-model audit analyses

These read cached predictions from `results/predictions/`. Run them from `src/`.

```
python agreement.py       # four-cell agreement matrix + class-stratified disagreement
python cold_start.py      # warm-warm / warm-cold / cold-cold partition metrics
python stat.py            # McNemar pairwise tests + paired bootstrap CIs
```

### TB-specific analysis

Stratifies test pairs into first-line, second-line, ARV co-administration, and comedications tiers, reports per-tier accuracy and calibration, and runs the rifampin CYP-induction precision/recall probe (n = 16). Run from `src/`.

```
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
├── paper.pdf                          # Audit write-up (MLHC template, not submitted)
├── EDA/
│   └── EDA.ipynb                      # Data preprocessing notebook (raw to processed)
├── src/                               # Core model + audit modules
│   ├── config.py                      # ProjectConfig dataclasses
│   ├── feature_engineering.py         # SMILES to PyG graph, severity/mechanism labels
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

**Not committed**, regenerated locally (see `.gitignore`).

- `Datasets/raw/`, the DrugBank and DDInter source files, obtained separately and license-restricted.
- `Datasets/processed/`, generated by `EDA/EDA.ipynb` from raw.
- `models/`, model checkpoints (`.pt` for GNN and ablations, `.pkl` for baseline folds).
- `results/predictions/`, cached probability arrays (`.npz`) used by the cross-model audit scripts.

## Limitations

This audit evaluates a single benchmark (DrugBank + DDInter). Cross-benchmark validation is needed before generalizing the findings. Mechanism labels are derived via regex pattern matching against DrugBank's free-text descriptions, and 19.1% of positive pairs match no keyword and are excluded from mechanism training, so the reported mechanism-head performance reflects the regex labels rather than ground-truth pharmacology.

Treating unlabeled drug pairs as negatives during training assumes a positive-unlabeled framing. Because DrugBank records only known interactions, any pair without a recorded interaction is treated as non-interacting. This may underestimate false negatives, particularly for less-studied drugs.

The TB cohort includes 18 drugs that passed the alias-corrected merge with sufficient test coverage. Newer agents such as delamanid, pretomanid, and dolutegravir are excluded due to insufficient test pairs, which is itself a manifestation of finding 2.

The GNN encoder uses one-hot atom (49-dim) and bond (14-dim) features with no pretraining. Variants with pretrained encoders might shift accuracy and AUROC values but are unlikely to overturn the audit's three main conclusions. The fundamental representational asymmetry remains, because Morgan fingerprints encode 2,048 bits of substructural information directly, while the GNN must learn it from 104,011 training pairs.

## Paper

The full write-up is in `paper.pdf`. Section 4 documents the data pipeline along with the alias-correction and PK-leakage discoveries. Section 5 presents the three audit findings with per-tier and per-class breakdowns. Section 6 discusses implications for clinical decision support and regulatory frameworks.

## Citation

This is unpublished course work. If you reference it, please use the following citation.

```
@misc{keith_ddi_audit_2026,
  author = {Arlene Keith},
  title  = {Validation Gaps in Drug-Drug Interaction Prediction Benchmarks for Tuberculosis-HIV Pharmacotherapy},
  year   = {2026},
  note   = {Capstone project, AI in Healthcare, Johns Hopkins University, Spring 2026, unpublished},
  url    = {https://github.com/jsf3467v/multi-task-ddi-audit}
}
```

## Acknowledgements

DrugBank (Wishart et al., 2018) and DDInter (Xiong et al., 2022) are the data sources. The audit framing builds on prior clinical ML benchmark critiques by Wong et al. (2021), Kapoor and Narayanan (2023), Huang et al. (2021), and Shen et al. (2025). Full references are in the paper.
