# BPSO Feature Selection for Disease Prediction

Reproducible research code for the paper **"Swarm Intelligence (BPSO) Feature Selection for Disease Prediction"**.
Binary Particle Swarm Optimization (BPSO) searches for a compact subset of clinical features; the subset is then
used to train and evaluate standard classifiers under repeated nested cross-validation, against all-features and
seven other feature-selection baselines.

> **Every number in `paper/tables/`, `paper/figures/` and `RESULTS_SUMMARY.md` is produced by running this code.**
> Nothing is typed in by hand. If you change the config, re-run the pipeline and the tables/figures update themselves.

---

## How the code maps to the paper

| Paper section | Code |
|---|---|
| Problem statement: $z \in \{0,1\}^d$, $X_z = \{x_j \mid z_j = 1\}$ | `selectors/base.py` (`get_support`, `transform`) |
| Objective $F(z) = \alpha(1-P(z)) + (1-\alpha)\lvert z\rvert/d$ | `src/bpso_fs/fitness.py` |
| BPSO (velocity, sigmoid transfer, position update, inertia schedule) | `src/bpso_fs/selectors/bpso.py` |
| Baselines: all features, MI, ANOVA-F, RFE, L1-LogReg, binary GA, random | `src/bpso_fs/selectors/` |
| Data preparation (imputation, encoding, scaling, inside CV only) | `src/bpso_fs/data/` |
| Classifiers (LR, SVM-RBF, RF, k-NN, Gaussian NB, XGBoost) | `src/bpso_fs/models.py` |
| Evaluation protocol, metrics, Wilcoxon/Holm, Friedman/Nemenyi, stability | `src/bpso_fs/experiment.py`, `src/bpso_fs/evaluation.py` |
| Tables (LaTeX booktabs + CSV + Markdown) | `src/bpso_fs/reporting.py` → `paper/tables/` |
| Figures (PDF + PNG, IEEE column widths) | `src/bpso_fs/plotting.py` → `paper/figures/` |
| Equations + pseudo-code for the Methodology section | `paper/methodology_snippets.tex` |

## Installation

Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
make setup                                              # = pip install -r requirements.txt && pip install -e .
pip install xgboost                                     # optional: enables the XGBoost classifier
```

## Reproduce everything

```bash
make data                                   # download + cache datasets, write data summaries
make test                                   # unit tests + end-to-end smoke test (~2 min)
make all QUICK=1                            # smoke run (2 seeds, 3 folds, 10x10 swarm) -> results_quick/ only
make all CONFIG=configs/reduced.yaml        # the run shipped in this repo (5 seeds x 10 folds, 20x50 swarm)
make all                                    # full paper protocol: configs/default.yaml (30 seeds x 10 folds, 30x100 swarm)
```

Or step by step:

```bash
python scripts/run_experiments.py --config configs/default.yaml --dataset all --seeds 30 --n-jobs -1
python scripts/make_figures.py      # -> paper/figures/*.pdf, *.png
python scripts/make_tables.py       # -> paper/tables/*.tex, *.csv, *.md
```

Useful flags for `run_experiments.py`: `--dataset breast_cancer,heart_cleveland`, `--seeds N`, `--folds K`,
`--quick`, `--no-alpha-sweep`, `--results-dir DIR`, `--n-jobs N`.

**Resume:** each (dataset, method, seed, fold) result is saved to `results/<dataset>/runs/<method>/s<seed>_f<fold>.json`
together with a hash of the settings it depends on. Re-running skips everything already done, so an interrupted run can
simply be restarted. Changing a relevant setting (e.g. `alpha`, swarm size, classifier list) invalidates only the
affected runs.

## Experimental protocol

* **Outer loop:** stratified 10-fold CV repeated over `n_seeds` seeds. All methods use the same splits, so results are
  paired by (seed, fold) for the statistical tests.
* **Inside each outer training fold only:** imputation (median / most frequent) and standardisation are fitted; the
  selector is run; each classifier is trained on the selected original columns (nominal features are one-hot encoded
  in the final model). The outer test fold is touched once, for scoring. `tests/test_no_leakage.py` checks this.
* **BPSO fitness:** $P(z)$ = mean accuracy of logistic regression over an inner stratified 5-fold CV on the outer
  training fold; $\alpha = 0.99$. Every distinct subset is scored once (memoised).
* **Baselines** (same splits and seeds):
  all features; mutual information and ANOVA-F `SelectKBest` with *k = median BPSO subset size* and with *k tuned by
  inner CV*; RFE (logistic regression, same k); L1-logistic regression (`C` by inner CV); binary GA with the same
  fitness, population and evaluation budget as BPSO; a random subset of the same size (sanity check).
* **Metrics:** accuracy, precision, recall (sensitivity), specificity, F1, ROC-AUC, PR-AUC, number of selected features,
  selection time and fit time, for every (dataset, method, classifier, seed, fold).
* **Statistics:** paired Wilcoxon signed-rank test (BPSO vs. each baseline, Holm-corrected); Friedman test over
  dataset × classifier blocks with Nemenyi critical difference; subset stability (mean pairwise Jaccard; Kuncheva index
  for fixed-size methods); per-feature selection frequency.
* **α sensitivity:** BPSO re-run with α ∈ {0.80, 0.90, 0.95, 0.99}.

### Implementation notes

* Feature selection works on the **original** features: selectors see a 1:1 numeric view (nominal features as ordinal
  codes), and the mask is mapped back to column names. The final models then one-hot encode the selected nominal
  columns (only relevant for Heart Disease: `cp`, `restecg`, `slope`, `thal`).
* Inside the BPSO/GA fitness, the inner CV runs on the outer-training-fold view (imputer and scaler fitted on the whole
  outer training fold). This affects only model selection inside the training fold; the outer test fold is never seen.
* With scikit-learn ≥ 1.9, `SVC(probability=True)` is deprecated; the code then uses its documented replacement,
  `CalibratedClassifierCV(SVC(), method="sigmoid", ensemble=False)`, so the SVM still outputs probabilities.
* The labels are always 1 = disease. For WDBC this means scikit-learn's target is flipped (1 = malignant).

## Results and outputs

```
results/<dataset>/
  data_summary.json            shape, class balance, missing values
  raw_results.csv              tidy: one row per (method, classifier, seed, fold) with every metric
  selections.csv               selected subset of every run (mask + names)
  history.csv                  BPSO/GA per-iteration best/mean fitness and subset size
  alpha_sweep_results.csv      α sensitivity runs
  predictions.csv.gz           out-of-fold scores (ROC curves)
  run_metadata.json            full config, git commit, library versions, timestamp
paper/tables/   *.tex (\input into IEEEtran), *.csv, *.md
paper/figures/  *.pdf (for LaTeX), *.png
```

Tables: `dataset_summary`, `bpso_hyperparameters`, `main_results_<dataset>`, `classifier_comparison`, `subset_sizes`,
`top_features`, `statistical_tests`, `friedman_ranks`, `runtime`, `alpha_sensitivity`.
They need `\usepackage{booktabs}`. The headline classifier of each `main_results_<dataset>` table is chosen
automatically as the classifier with the best mean ROC-AUC for BPSO on that dataset
(`paper/tables/report_classifiers.json`); override with `--report-classifier logreg`. Results for every classifier are
in `classifier_comparison` and `results_all_classifiers.csv`.

Figures: `convergence_<ds>`, `subset_size_<ds>`, `feature_frequency_<ds>`, `roc_<ds>`, `method_comparison_<ds>`
(two-column width), `alpha_sensitivity_<ds>`, `confusion_bpso_<ds>`, `cd_diagram` (two-column width).

The results committed in this repository come from **`configs/reduced.yaml`** (see `RESULTS_SUMMARY.md`). For the final
paper, re-run with `configs/default.yaml` and rebuild the tables and figures.

## Runtime

Measured on the build machine (2 CPU cores, Python 3.13, scikit-learn 1.9):

| Run | Time |
|---|---|
| `pytest -q` (all tests incl. end-to-end smoke run) | about 1.5 min |
| `--quick` on one dataset | about 1 min |
| `configs/reduced.yaml`, 3 datasets (shipped results) | about 76 CPU-minutes of task time, so about 40–45 min wall time on 2 cores |
| `configs/default.yaml`, 3 datasets | **not run here.** An extrapolation from the reduced run (3× the fitness evaluations per BPSO/GA run, 6× the outer folds, 5× the α-sweep seeds) gives roughly 17–18 CPU-hours, i.e. about 2–2.5 h on 8 cores |

BPSO costs per outer training fold (20 × 50 swarm): 12 s on Breast Cancer, 7 s on Heart and 1.9 s on Pima
(`paper/tables/runtime.md`). Use `--n-jobs` to parallelise over (seed, fold) jobs.

## Using your own dataset

Any CSV with a binary target works:

```bash
python scripts/run_experiments.py --config configs/default.yaml --csv data/raw/my.csv --target Outcome \
       --positive-label 1 --name my_dataset
```

or add a YAML file to `configs/datasets/` (see `custom_example.yaml`) and put its `name` in `experiment.datasets`.
Text columns are treated as nominal categorical features unless you list `categorical:` explicitly; ID columns can be
removed with `drop:`.

## Datasets and how to cite them

| Name | Size | Source used by the loader |
|---|---|---|
| `breast_cancer` | 569 × 30 | `sklearn.datasets.load_breast_cancer` |
| `pima_diabetes` | 768 × 8 | public CSV mirror (raw.githubusercontent.com/jbrownlee/Datasets) |
| `heart_cleveland` | 303 × 13 | UCI `processed.cleveland.data`; if UCI is unreachable, the ISLR `Heart.csv` copy of the same 303 records (converted to UCI coding) |
| `ilpd` (optional) | 583 × 10 | UCI |
| `ckd` (optional) | 400 × 24 | UCI (.rar archive: extract `chronic_kidney_disease_full.arff` to `data/raw/chronic_kidney_disease.arff`) |

If a download fails, the loader prints the URLs it tried and the exact file name to save manually in `data/raw/`.
Data are never synthesised.

* W. Wolberg, O. Mangasarian, N. Street, W. Street, "Breast Cancer Wisconsin (Diagnostic)," UCI Machine Learning Repository, 1995, doi:10.24432/C5DW2B.
* J. W. Smith, J. E. Everhart, W. C. Dickson, W. C. Knowler, R. S. Johannes, "Using the ADAP learning algorithm to forecast the onset of diabetes mellitus," *Proc. Annu. Symp. Comput. Appl. Med. Care*, pp. 261–265, 1988.
* A. Janosi, W. Steinbrunn, M. Pfisterer, R. Detrano, "Heart Disease," UCI Machine Learning Repository, 1988, doi:10.24432/C52P4X.
* B. Ramana, N. Venkateswarlu, "ILPD (Indian Liver Patient Dataset)," UCI Machine Learning Repository, 2012, doi:10.24432/C5D02C.
* L. Rubini, P. Soundarapandian, P. Eswaran, "Chronic Kidney Disease," UCI Machine Learning Repository, 2015, doi:10.24432/C5G020.

Method references: J. Kennedy and R. C. Eberhart, "A discrete binary version of the particle swarm algorithm," *IEEE
SMC*, 1997; S. Mirjalili and A. Lewis, "S-shaped versus V-shaped transfer functions for binary particle swarm
optimization," *Swarm Evol. Comput.*, 2013; J. Demšar, "Statistical comparisons of classifiers over multiple data
sets," *JMLR*, 2006; F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*, 2011.

## Tests

```bash
pytest -q
```

* `test_bpso.py`: positions stay binary, velocities stay within ±v_max, empty-subset repair, same seed → same result,
  global best never gets worse, BPSO recovers ≥ 4 of 5 informative features among 25 noise features, GA budget.
* `test_fitness.py`: hand-computed fitness values, α = 1 ignores subset size, caching.
* `test_no_leakage.py`: the imputer/scaler and the selector are fitted on training indices only.
* `test_pipeline_smoke.py`: `--quick` end-to-end on breast cancer, resume, tables and figures.

## Repository layout

```
configs/            default.yaml (paper), reduced.yaml (shipped run), datasets/*.yaml
src/bpso_fs/        data/, selectors/, fitness.py, models.py, evaluation.py, experiment.py, plotting.py, reporting.py
scripts/            download_data.py, run_experiments.py, make_figures.py, make_tables.py
paper/              figures/, tables/, methodology_snippets.tex
notebooks/          exploration.ipynb (EDA only)
tests/
```
