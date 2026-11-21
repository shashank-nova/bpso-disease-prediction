# BPSO Feature Selection for Disease Prediction — Research Summary

Everything needed to write the paper, drawn from the code and the completed five-dataset run
(`results_experiment/`, config `configs/all_datasets.yaml`). All numbers below come from
the generated files in `paper/tables/`; all figures are in `paper/figures/` (PNG + PDF).
LaTeX versions of every table are next to the CSVs (`paper/tables/*.tex`).

> **Research use only.** Public benchmark data; the models are not clinically validated.

---

## 1. One-paragraph overview (draft abstract)

Wrapper feature selection can shrink clinical prediction models to a handful of measurements,
but its benefit over cheap filter methods is rarely tested under a rigorous protocol. We
evaluate Binary Particle Swarm Optimization (BPSO) for feature selection on five public binary
disease-classification benchmarks (breast cancer, chronic kidney disease, heart disease, liver
disease, diabetes). BPSO minimises a weighted objective that trades inner-cross-validated
accuracy against subset size. It is compared with nine alternatives — all features, a binary
genetic algorithm, mutual-information and ANOVA-F filters (with *k* fixed to BPSO's size or
tuned by CV), recursive feature elimination, L1-regularised logistic regression and random
subsets — across five classifiers, using 5 × repeated stratified 10-fold outer cross-validation
(50 outer folds per dataset) with all selection and tuning done inside the training folds.
BPSO removed 38–58 % of features on every dataset while keeping F1 within about 0.015 of the
all-feature model, and was significantly better than random subsets of the same size on two
datasets. However, across 25 dataset–classifier settings it did not outperform the simpler
alternatives: its average Friedman rank was 7.20 of 10 (Friedman χ² = 76.7, p < 10⁻¹²), on par
with the genetic algorithm (7.28) and behind all-features (3.68) and CV-tuned filters
(3.80–3.88). The main value of BPSO here is compact feature sets, not higher accuracy.

---

## 2. Research question and contributions

**Question.** Does BPSO wrapper feature selection produce smaller feature sets for disease
prediction *without* losing predictive performance, and how does it compare with standard
filter, embedded and evolutionary alternatives under a leakage-free protocol?

**Contributions you can claim**
1. A reproducible BPSO feature-selection pipeline with a size-penalised fitness function,
   fitness caching and an empty-subset repair rule.
2. A nested, leakage-free benchmark: every selector and every hyper-parameter search is fitted
   only on the outer training fold.
3. A comparison against 9 alternatives × 5 classifiers × 5 datasets, with paired Wilcoxon
   tests (Holm-corrected), Friedman/Nemenyi ranking, stability (Jaccard) and runtime analysis.
4. A sensitivity study of the accuracy/size trade-off weight α.
5. An open-source package, inference API and dashboard (deployed on Render).

---

## 3. Datasets  (`paper/tables/dataset_summary.*`)

| Dataset | Samples *n* | Features *d* | Disease cases (%) | Missing cells | Source |
|---|---:|---:|---:|---:|---|
| Breast Cancer (WDBC) | 569 | 30 | 212 (37.3) | 0 | UCI / scikit-learn |
| Chronic Kidney Disease | 400 | 24 | 250 (62.5) | 1013 | UCI |
| Heart Disease (Cleveland) | 303 | 13 | 139 (45.9) | 6 | UCI |
| Indian Liver Patient (ILPD) | 583 | 10 | 416 (71.4) | 4 | UCI |
| Pima Indians Diabetes | 768 | 8 | 268 (34.9) | 652 | UCI / NIDDK |

Notes for the paper:
- Pima: physiologically impossible zeros (e.g. glucose, BMI, insulin = 0) are treated as missing,
  which is why it has 652 missing cells.
- Heart: the original 0–4 target is binarised (0 = no disease, 1–4 = disease).
- CKD: the UCI source is a `.rar` archive; the ARFF file was extracted manually into `data/raw/`.
- Raw files and their source URLs are in `data/raw/*.source`.

---

## 4. Method

### 4.1 Pre-processing (inside every training fold)
- Numeric features: median imputation → standardisation (z-score).
- Categorical features: most-frequent imputation → ordinal encoding → standardisation.
- Pre-processing is part of the scikit-learn pipeline, so it is re-fitted on each training fold
  (no information from test folds).

### 4.2 Fitness function  (`src/bpso_fs/fitness.py`)

For a binary mask *z* ∈ {0,1}^d, BPSO **minimises**

$$F(z) = \alpha\,\bigl(1 - P(z)\bigr) + (1-\alpha)\,\frac{|z|}{d}$$

- *P(z)* = mean 5-fold stratified inner-CV **accuracy** of a logistic-regression wrapper trained
  on the selected features, computed on the outer training fold only.
- |z| = number of selected features, *d* = total features.
- α = 0.99 (performance dominates; size breaks near-ties). An empty subset gets F = 1.
- Fitness values are cached per unique mask, so repeated subsets cost nothing.

### 4.3 Binary PSO  (`src/bpso_fs/selectors/bpso.py`)

Each particle *i* has a binary position *x_i* (a feature mask) and a real velocity *v_i*.
For iteration *t*:

$$v_{ij} \leftarrow w(t)\,v_{ij} + c_1 r_1 (p_{ij} - x_{ij}) + c_2 r_2 (g_j - x_{ij}),\qquad v_{ij}\in[-v_{\max}, v_{\max}]$$

$$x_{ij} = \begin{cases}1 & \text{if } u < S(v_{ij}) = \dfrac{1}{1+e^{-v_{ij}}}\\ 0 & \text{otherwise}\end{cases}\qquad u\sim U(0,1)$$

with *p_i* the particle's personal best, *g* the global best, and linearly decreasing inertia
*w(t)* = w_max − (w_max − w_min)·t/T. When a particle's mask is empty, one random feature is
switched on (repair). When two solutions have equal fitness, the one with fewer features wins.

**Hyper-parameters**  (`paper/tables/bpso_hyperparameters.*`)

| Parameter | Value |
|---|---|
| Swarm size | 20 particles |
| Iterations *T* | 50 |
| Inertia *w* | linear 0.9 → 0.4 |
| c₁, c₂ | 2.0, 2.0 |
| Velocity clamp v_max | 6.0 |
| Transfer function | S-shaped (sigmoid) |
| Initial bit probability | 0.5 |
| Empty subset | repair |
| α | 0.99 |
| Wrapper / inner CV | logistic regression, 5-fold, accuracy |

### 4.4 Comparison methods

| Label | Description |
|---|---|
| All features | No selection (reference) |
| Binary GA | Genetic algorithm with the **same fitness** as BPSO; tournament size 3, crossover 0.9, 1 elite, mutation 1/d |
| MI (k = BPSO) | Top-*k* mutual information, *k* = median BPSO subset size on that dataset |
| MI (k by CV) | Top-*k* mutual information, *k* chosen by inner CV |
| ANOVA-F (k = BPSO) | Top-*k* ANOVA F-score, *k* = BPSO size |
| ANOVA-F (k by CV) | Top-*k* ANOVA F-score, *k* chosen by inner CV |
| RFE (k = BPSO) | Recursive feature elimination to BPSO size |
| L1-LogReg | Embedded selection: non-zero coefficients of L1 logistic regression (C by inner CV) |
| Random (k = BPSO) | Random subset of BPSO size (sanity baseline) |

The "k = BPSO" variants isolate the question *"are BPSO's features better than other features
of the same count?"*

### 4.5 Classifiers (tuned by inner grid-search CV on the training fold)

| Classifier | Grid |
|---|---|
| Logistic regression | C ∈ {0.01, 0.1, 1, 10} |
| SVM (RBF) | C ∈ {0.1, 1, 10}, γ ∈ {scale, 0.01, 0.1} |
| Random forest | 200 trees, max_depth ∈ {None, 5, 10}, min_samples_leaf ∈ {1, 3} |
| k-NN | n_neighbors ∈ {3, 5, 7, 11} |
| Gaussian Naive Bayes | — |

(XGBoost is supported but was not installed in this run.)

### 4.6 Evaluation protocol
- **Outer loop:** stratified 10-fold CV repeated with 5 seeds → **50 outer folds per dataset**,
  identical folds for every method (paired comparisons).
- **Inside each training fold:** pre-processing → feature selection → classifier grid search.
  The test fold is used once, for scoring only.
- **Metrics:** accuracy, precision, recall (sensitivity), specificity, F1, ROC-AUC, PR-AUC,
  number of selected features, selection time.
- **Statistics:**
  - Paired two-sided Wilcoxon signed-rank tests, BPSO vs each method, per dataset, Holm
    correction; wins/ties/losses over the 50 folds.
  - Friedman test over 25 blocks (5 datasets × 5 classifiers, F1) with Nemenyi critical
    difference (CD = 2.71 at α = 0.05, k = 10 methods).
  - Selection stability: mean pairwise Jaccard similarity of selected subsets across folds.
- **Headline tables** use one classifier per dataset: the one with the best mean ROC-AUC for
  BPSO (WDBC → SVM, CKD → RF, Heart/ILPD/Pima → logistic regression). All 5 classifiers are
  reported in `classifier_comparison.*` and used for the Friedman test.
- **α sensitivity:** α ∈ {0.80, 0.90, 0.95, 0.99}, 2 seeds × 10 folds = 20 runs per value.
- **Scale of the run:** 2,500 outer-fold evaluations per dataset (10 methods × 5 classifiers ×
  50 folds) — 12,500 in total plus the α sweep; ~51 minutes wall time with 2 parallel jobs.

---

## 5. Results

### 5.1 Headline: BPSO vs all features (reported classifier, mean ± std over 50 folds)
(`paper/tables/main_results_<dataset>.*`)

| Dataset (classifier) | Features: all → BPSO | Reduction | F1 all | F1 BPSO | ROC-AUC all | ROC-AUC BPSO |
|---|---|---:|---:|---:|---:|---:|
| WDBC (SVM) | 30 → 14.7 ± 2.2 | 51 % | 0.963 ± 0.024 | 0.956 ± 0.029 | 0.996 ± 0.008 | 0.994 ± 0.009 |
| CKD (RF) | 24 → 12.2 ± 2.0 | 49 % | 0.994 ± 0.011 | **0.996 ± 0.011** | 1.000 ± 0.001 | 1.000 ± 0.001 |
| Heart (LogReg) | 13 → 8.0 ± 1.6 | 38 % | 0.820 ± 0.074 | 0.805 ± 0.065 | 0.907 ± 0.051 | 0.900 ± 0.052 |
| ILPD (LogReg) | 10 → 4.2 ± 1.2 | 58 % | 0.826 ± 0.024 | **0.830 ± 0.024** | 0.751 ± 0.060 | 0.726 ± 0.062 |
| Pima (LogReg) | 8 → 4.8 ± 1.0 | 40 % | 0.626 ± 0.081 | 0.614 ± 0.081 | 0.833 ± 0.052 | 0.824 ± 0.053 |

**Reading:** BPSO roughly halves the feature set; F1 changes by −0.015 to +0.004. None of the
F1 differences vs all features is significant after Holm correction (all p ≥ 0.06), but
ROC-AUC is significantly lower on WDBC (p = 0.006), ILPD (p < 0.001) and Pima (p = 0.003).

### 5.2 Paired Wilcoxon tests, BPSO vs each method (F1, Holm-corrected)
(`paper/tables/statistical_tests.*`) — significant results only:

| Dataset | BPSO vs | ΔF1 | W/T/L | p (Holm) |
|---|---|---:|---|---:|
| WDBC | MI (k = BPSO) | +0.028 | 32/12/6 | 0.0006 ✔ BPSO better |
| WDBC | ANOVA-F (k = BPSO) | +0.028 | 32/12/6 | 0.0006 ✔ BPSO better |
| WDBC | MI (k by CV) | −0.010 | 6/22/22 | 0.026 ✘ |
| WDBC | ANOVA-F (k by CV) | −0.008 | 9/20/21 | 0.040 ✘ |
| CKD | Random (k = BPSO) | +0.011 | 22/25/3 | 0.021 ✔ |
| Pima | Random (k = BPSO) | +0.068 | 33/3/14 | 0.0014 ✔ |
| Pima | MI (k = BPSO) | −0.017 | 13/7/30 | 0.039 ✘ |

All other BPSO comparisons (including every comparison with the genetic algorithm) are not
significant. On WDBC, BPSO's 15 features beat the 15 top-ranked MI/ANOVA features — the
clearest evidence that the wrapper picks complementary rather than individually strong features.

### 5.3 Overall ranking across 25 settings  (`friedman_ranks.*`, Fig. `cd_diagram`)

Friedman χ² = 76.68, p = 7.3 × 10⁻¹³; Nemenyi CD = 2.71.

| Rank | Method | Avg. rank |
|---:|---|---:|
| 1 | All features | 3.68 |
| 2 | ANOVA-F (k by CV) | 3.80 |
| 3 | MI (k by CV) | 3.88 |
| 4 | L1-LogReg | 4.08 |
| 5 | RFE (k = BPSO) | 4.84 |
| 6 | ANOVA-F (k = BPSO) | 5.70 |
| 7 | MI (k = BPSO) | 5.78 |
| 8 | **BPSO (proposed)** | **7.20** |
| 9 | Binary GA | 7.28 |
| 10 | Random (k = BPSO) | 8.76 |

BPSO is statistically indistinguishable from GA, RFE, the fixed-*k* filters and random subsets,
and significantly behind all-features, CV-tuned filters and L1 (rank gap > CD). Methods that
keep more features rank higher — expected, because none of these datasets is high-dimensional.

### 5.4 Subset size and stability  (`subset_sizes.*`, Fig. `subset_size_*`)

| Dataset | BPSO selected | Reduction | BPSO Jaccard | GA Jaccard | L1 Jaccard |
|---|---|---:|---:|---:|---:|
| WDBC | 14.68 ± 2.17 | 51.1 % | 0.369 | 0.343 | 0.668 |
| CKD | 12.22 ± 2.04 | 49.1 % | 0.482 | 0.525 | 0.847 |
| Heart | 8.04 ± 1.55 | 38.2 % | 0.528 | 0.539 | 0.845 |
| ILPD | 4.22 ± 1.18 | 57.8 % | 0.474 | 0.491 | 0.772 |
| Pima | 4.82 ± 1.04 | 39.8 % | 0.606 | 0.601 | 0.753 |

BPSO and GA are the least stable non-random selectors (Jaccard 0.37–0.61): the wrapper search
finds different, near-equivalent subsets on different folds. Filters and L1 are far more stable.

### 5.5 Most frequently selected features by BPSO  (`top_features.*`, Fig. `feature_frequency_*`)

| Dataset | Top features (selection frequency across 50 folds) |
|---|---|
| WDBC | worst smoothness (0.90), worst symmetry (0.84), area error (0.76), worst concavity (0.72), mean texture (0.64) |
| CKD | specific gravity (1.00), haemoglobin (1.00), appetite (0.88), diabetes mellitus (0.88), hypertension (0.84) |
| Heart | ca – major vessels (1.00), thal (1.00), chest-pain type (0.82), max heart rate (0.76), sex (0.68) |
| ILPD | age (1.00), SGPT/ALT (0.86), alkaline phosphatase (0.56), total bilirubin (0.42) |
| Pima | glucose (1.00), BMI (1.00), diabetes pedigree (0.84), insulin (0.62), age (0.54) |

These agree with known clinical markers (e.g. glucose/BMI for diabetes, specific gravity and
haemoglobin for CKD, vessel count and thalassemia for heart disease) — useful for the
discussion section.

### 5.6 Runtime  (`runtime.*`)

| Dataset | BPSO selection time (s / fold) | Unique subsets evaluated | GA time (s) |
|---|---:|---:|---:|
| WDBC | 8.90 ± 0.57 | 1020 | 5.58 |
| CKD | 7.29 ± 0.21 | 1019 | 4.17 |
| Heart | 5.46 ± 0.48 | 786 | 2.54 |
| ILPD | 3.03 ± 0.22 | 425 | 1.77 |
| Pima | 1.48 ± 0.10 | 190 | 1.12 |

Filters take 0.01–0.6 s. BPSO costs 1.3–2× the GA because it evaluates more unique subsets
(20 particles × 50 iterations = 1000 evaluations max; caching saves the rest on small *d*).

### 5.7 Sensitivity to α  (`alpha_sensitivity.*`, Fig. `alpha_sensitivity_*`)

| Dataset (clf) | α = 0.80 | α = 0.90 | α = 0.95 | α = 0.99 |
|---|---|---|---|---|
| WDBC (SVM) | 5.0 f, F1 0.942 | 6.8 f, 0.953 | 9.9 f, 0.960 | 14.3 f, 0.961 |
| CKD (RF) | 3.9 f, 0.977 | 6.3 f, 0.987 | 7.7 f, 0.995 | 12.2 f, 0.999 |
| Heart (LR) | 3.3 f, 0.783 | 4.0 f, 0.790 | 5.0 f, 0.796 | 7.7 f, 0.808 |
| ILPD (LR) | 1.0 f, 0.832 | 1.4 f, 0.833 | 2.8 f, 0.831 | 4.2 f, 0.832 |
| Pima (LR) | 1.0 f, 0.574 | 2.1 f, 0.602 | 2.7 f, 0.609 | 4.6 f, 0.616 |

(f = mean number of selected features.) Lowering α gives much smaller models at a modest cost:
on WDBC, α = 0.90 keeps F1 at 0.953 with only ~7 of 30 features (77 % reduction); on CKD,
α = 0.95 keeps F1 0.995 with ~8 of 24. On ILPD, a single feature gives the same F1 (but lower
ROC-AUC: 0.61 vs 0.72), showing F1 is not informative there because of class imbalance.

---

## 6. Discussion — what to say honestly

1. **BPSO is a compression method here, not an accuracy booster.** It halves the input set with
   a small, mostly non-significant F1 cost. On low-dimensional clinical tables (8–30 features),
   removing features rarely improves accuracy, so "all features" ranks first.
2. **Wrapper vs filter at equal size.** BPSO beats same-size MI/ANOVA filters on WDBC (correlated
   features where univariate ranking picks redundant ones), but not elsewhere.
3. **BPSO ≈ GA.** With the same fitness, the two metaheuristics give practically identical
   accuracy, size and stability; BPSO is slower.
4. **Fitness mismatch.** BPSO optimises inner *accuracy* with logistic regression; the evaluation
   uses F1/ROC-AUC and five different classifiers. This likely explains the ROC-AUC drops on
   imbalanced ILPD and Pima. Optimising ROC-AUC or balanced accuracy is a natural next step.
5. **Stability is a weakness** (Jaccard ≈ 0.4–0.6); for clinical interpretability, report
   selection *frequency* (Section 5.5) rather than a single subset.
6. **α is the key knob:** α ≈ 0.90–0.95 offers a much better size/accuracy trade-off than the
   default 0.99 on WDBC and CKD.

## 7. Limitations (put these in the paper)
- Five small, low-dimensional UCI datasets; results may differ for high-dimensional data
  (genomics, imaging features) where wrapper selection usually matters more.
- Reduced budget (20 particles × 50 iterations, 5 seeds); `configs/default.yaml` is the full
  protocol and was not run.
- Headline classifier per dataset is chosen by BPSO's best ROC-AUC, which slightly favours
  BPSO in Section 5.1; the Friedman analysis over all classifiers avoids this.
- No external validation cohort; benchmark data only; not clinically validated.
- CKD and Pima have many missing values handled by simple median/mode imputation.

## 8. Future work
Fitness on ROC-AUC / balanced accuracy; V-shaped transfer functions; multi-objective
(Pareto) BPSO; stability-aware fitness; high-dimensional datasets; cost-sensitive selection
(cheap vs expensive clinical tests); external validation.

---

## 9. Figures for the paper  (`paper/figures/`, each as .png and .pdf)

| File | What it shows | Suggested caption |
|---|---|---|
| `cd_diagram` | Critical-difference diagram | Average Friedman ranks of the 10 methods over 25 dataset–classifier settings (F1); methods joined by a bar are not significantly different (Nemenyi, CD = 2.71). |
| `method_comparison_<ds>` | Box plots of F1 and ROC-AUC per method | Distribution of F1 and ROC-AUC over 50 outer folds on *ds*; diamonds mark means. |
| `convergence_<ds>` | BPSO/GA best fitness per iteration | Convergence of the best fitness for BPSO and GA (mean ± band over folds). |
| `subset_size_<ds>` | Subset size of the best solution per iteration, BPSO vs GA | Size of the best-so-far feature subset over iterations (mean ± band over folds). |
| `feature_frequency_<ds>` | How often each feature was selected | Selection frequency of each feature by BPSO across 50 outer folds. |
| `roc_<ds>` | ROC curves | Mean ROC curve from pooled out-of-fold predictions per seed (± std across seeds). |
| `confusion_bpso_<ds>` | Confusion matrix | Row-normalised confusion matrix for BPSO, summed over all 50 outer folds. |
| `alpha_sensitivity_<ds>` | F1/AUC and subset size vs α | Effect of the trade-off weight α on subset size and performance. |

`<ds>` ∈ {breast_cancer, ckd, heart_cleveland, ilpd, pima_diabetes}. Suggested main-text set:
`cd_diagram`, one `method_comparison`, one `feature_frequency`, one `alpha_sensitivity`;
the rest go to supplementary material.

## 10. Reproducibility and software

```powershell
pip install -e ".[serve,dev]"
python scripts/run_experiments.py --config configs/all_datasets.yaml --dataset all --n-jobs 2 --results-dir results_experiment
python scripts/make_tables.py  --results-dir results_experiment --tables-dir paper/tables
python scripts/make_figures.py --results-dir results_experiment --figures-dir paper/figures
```

- Seeds are fixed per (seed, fold), so runs are repeatable; per-fold metrics, predictions,
  selections and BPSO/GA histories are saved in `results_experiment/<dataset>/`.
- Stack: Python 3.10+, scikit-learn, NumPy, SciPy, pandas, matplotlib/seaborn; FastAPI +
  Streamlit for the demo; Docker; GitHub Actions CI.
- Live demo: dashboard https://bpso-showcase.onrender.com, API docs
  https://bpso-api.onrender.com/docs (free tier; first load may take ~1 min).
- Code: https://github.com/shashank-nova/bpso-disease-prediction

## 11. Core references (check details against the originals before citing)
- Kennedy, J. & Eberhart, R. C. (1997). A discrete binary version of the particle swarm algorithm. *IEEE SMC*.
- Kennedy, J. & Eberhart, R. (1995). Particle swarm optimization. *Proc. IEEE ICNN*.
- Shi, Y. & Eberhart, R. (1998). A modified particle swarm optimizer (inertia weight). *IEEE CEC*.
- Mirjalili, S. & Lewis, A. (2013). S-shaped versus V-shaped transfer functions for binary PSO. *Swarm and Evolutionary Computation*, 9.
- Xue, B., Zhang, M., Browne, W. N. & Yao, X. (2016). A survey on evolutionary computation approaches to feature selection. *IEEE Trans. Evolutionary Computation*, 20(4).
- Guyon, I. & Elisseeff, A. (2003). An introduction to variable and feature selection. *JMLR*, 3.
- Kohavi, R. & John, G. H. (1997). Wrappers for feature subset selection. *Artificial Intelligence*, 97.
- Guyon, I., Weston, J., Barnhill, S. & Vapnik, V. (2002). Gene selection for cancer classification using SVMs (RFE). *Machine Learning*, 46.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *JRSS-B*, 58.
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *JMLR*, 7.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scand. J. Statistics*, 6.
- Kuncheva, L. I. (2007). A stability index for feature selection. *AIAP*.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12.
- Datasets: Wolberg, Street & Mangasarian (WDBC, 1995); Smith et al. (1988) Pima/ADAP; Detrano et al. (1989) Cleveland heart; Ramana & Venkateswarlu (ILPD, UCI 2012); Rubini, Soundarapandian & Eswaran (CKD, UCI 2015); Dua & Graff, UCI Machine Learning Repository.
