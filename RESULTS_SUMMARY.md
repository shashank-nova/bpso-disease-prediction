# Results summary

Everything below is read off the generated files in `paper/tables/` (`*.md`/`*.csv`) and `results/*/`.

## What was run

* **Config:** `configs/reduced.yaml`. This is the same protocol as the paper config (`configs/default.yaml`), at a smaller
  scale so it could finish on a 2-core machine:
  * 5 seeds × stratified 10-fold outer CV (50 paired test folds per method), instead of 30 seeds;
  * swarm of 20 particles × 50 iterations, instead of 30 × 100;
  * α sweep with 2 seeds × 10 folds.
  Fitness α = 0.99, inner 5-fold CV, logistic-regression wrapper, accuracy as P(z), linear inertia 0.9 → 0.4,
  c1 = c2 = 2, v_max = 6, sigmoid transfer.
* **Datasets:** Breast Cancer (WDBC, 569 × 30), Pima Diabetes (768 × 8), Heart Disease Cleveland (303 × 13).
  The UCI server could not be reached from the build machine, so Cleveland was loaded from the ISLR `Heart.csv` copy of
  the same 303 records (`results/heart_cleveland/run_metadata.json` → `data_source`). On a normal network the loader
  uses UCI first.
* **Classifiers:** LR, SVM-RBF, RF, k-NN, Gaussian NB, XGBoost (default hyper-parameters, no tuning).
* **Headline classifier** (best mean BPSO ROC-AUC per dataset): SVM for Breast Cancer, LR for Pima and Heart.

**Re-run `configs/default.yaml` before quoting numbers in the paper.** These results are from the reduced run.

## Per dataset (headline classifier)

| Dataset | d | BPSO subset (mean ± std) | Reduction | F1: all → BPSO | ROC-AUC: all → BPSO | Significant? (Wilcoxon, Holm) |
|---|---|---|---|---|---|---|
| Breast Cancer (SVM) | 30 | 14.68 ± 2.17 | 51.1 % | 0.9633 → 0.9558 | 0.9959 → 0.9943 | F1: no (p = 0.062); ROC-AUC: **yes, all features better** (p = 0.006) |
| Pima Diabetes (LR) | 8 | 4.82 ± 1.04 | 39.8 % | 0.6260 → 0.6143 | 0.8328 → 0.8235 | F1: no (p = 0.38); ROC-AUC: **yes, all features better** (p = 0.003) |
| Heart Disease (LR) | 13 | 8.04 ± 1.55 | 38.2 % | 0.8199 → 0.8054 | 0.9073 → 0.8995 | F1: no (p = 0.59); ROC-AUC: no (p = 0.41) |

## Main findings

1. **BPSO reduced the number of features substantially**: about half of the features on Breast Cancer and about 40 %
   on Pima and Heart.
2. **It did not improve predictive performance over using all features.** Across all 18 dataset × classifier
   combinations, BPSO's mean F1 was lower than all-features in every one (differences of 0.35 to 2.8 percentage points), and its
   mean ROC-AUC was lower in 16 of 18. Of the 36 BPSO-vs-all-features tests (18 combinations × F1/ROC-AUC), 11 were
   significant after Holm correction, all in favour of all features; none favoured BPSO.
3. **Against other selectors at the same subset size:** BPSO was significantly better than a random subset of the same
   size in all 14 significant comparisons, and better than MI and ANOVA-F with k = the BPSO size in 10 of 13 and 10 of
   14 significant comparisons respectively. All 20 of these wins are on Breast Cancer, where MI and ANOVA-F select the
   same 15 highly correlated size/concavity features in every fold. RFE at the same k, L1-logistic regression, and MI
   or ANOVA-F with k tuned by CV were never significantly worse than BPSO. When they differed significantly
   (5 to 6 tests each), BPSO was the worse method.
4. **BPSO vs. binary GA** with the same fitness and the same budget: no significant difference in any of the
   36 tests. GA reached a lower final fitness on Breast Cancer, where it selected fewer features (12.0 vs. 14.7), and
   it was faster (8.8 s vs. 12.0 s per fold).
5. **Friedman test** over 18 dataset × classifier blocks (F1): χ² = 81.87, p = 6.9 × 10⁻¹⁴. Average ranks (1 = best):
   all features 3.06, L1 3.11, MI-CV 3.56, ANOVA-CV 3.89, RFE 4.94, ANOVA-k 5.83, MI-k 6.28, **BPSO 7.22**, GA 7.89,
   random 9.22. The Nemenyi CD is 3.19 (`paper/figures/cd_diagram.pdf`).
6. **α trade-off works as intended** (`alpha_sensitivity`): lowering α gives smaller subsets at a modest cost in
   score. For example, Breast Cancer goes from 14.3 features at α = 0.99 to 4.95 at α = 0.80 (SVM F1 0.961 → 0.942), and
   Pima goes to a single feature (Glucose) at α = 0.80.
7. **Most consistently selected features** (`top_features`):
   * Pima: Glucose and BMI (100 % of runs), then DiabetesPedigreeFunction (84 %).
   * Heart: `ca` and `thal` (100 %), then `cp` (82 %) and `thalach` (76 %).
   * Breast Cancer: worst smoothness (90 %), worst symmetry (84 %) and area error (76 %). Subset stability is low here
     (Jaccard 0.37) because many WDBC features are near-duplicates.
8. **Cost:** BPSO needs 1.9 to 12 s per training fold here, against well under 1 s for the filter and embedded methods
   (`runtime`).

## Honest reading for the paper

On these three benchmarks, with these settings, BPSO is a working way to obtain **compact** subsets, roughly halving
the inputs at a small loss of accuracy. It is **not** shown to be more accurate than using all features, or than
simple filter or embedded selectors. A defensible claim is about the size/performance trade-off and interpretability,
not about higher accuracy. Points to check in the full 30-seed run before writing conclusions:

* whether the losses against all features stay significant with more repetitions;
* whether a larger swarm and more iterations (30 × 100) narrow the gap to GA (the BPSO swarm-mean fitness plateaus
  after about 10 iterations; see `convergence_*.pdf`);
* whether a V-shaped transfer function (`transfer_function: v_shaped`) or a lower α changes the picture.
