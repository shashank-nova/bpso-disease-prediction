<div align="center">
  <h1>🩺 BPSO Disease Prediction</h1>
  <p><strong>Research-grade feature selection · Reproducible evaluation · Interactive ML showcase</strong></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10 or newer">
    <img src="https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn 1.2 or newer">
    <img src="https://img.shields.io/badge/FastAPI-inference-009688?logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/Streamlit-showcase-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
    <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" alt="Docker Compose">
    <img src="https://img.shields.io/badge/Experiment-5%20datasets-2ea44f" alt="Five-dataset experiment">
    <img src="https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white" alt="GitHub Actions workflow configured">
  </p>
</div>

**BPSO Disease Prediction** evaluates Binary Particle Swarm Optimization (BPSO) for selecting compact feature subsets in binary disease classification. It combines a reproducible cross-validation pipeline, baseline comparisons, statistical reporting, and a local web/API showcase.

> **Research use only.** These public benchmark datasets and models are not clinically validated and must not be used to diagnose, treat, or make decisions about patients.

## Project at a glance

| Capability | Implementation |
| --- | --- |
| Feature selection | Binary PSO with cached wrapper-fitness evaluations |
| Comparators | All features, mutual information, ANOVA-F, RFE, L1 logistic regression, genetic algorithm, and random subsets |
| Evaluation | Repeated stratified outer cross-validation with inner-fold model selection |
| Classifiers | Logistic regression, RBF SVM, random forest, k-NN, Gaussian Naive Bayes; XGBoost is optional |
| Reporting | CSV and LaTeX tables; PDF and PNG figures |
| Showcase | FastAPI inference API and Streamlit interface, launched with Docker Compose |

## Portfolio snapshot

The checked-in showcase summary is generated from the completed reduced experiment and is displayed in the web app even when the full local run directory is not present.

| Evidence | Completed run |
| --- | ---: |
| Public benchmark datasets | 5 |
| Feature-selection methods | 10 |
| Classifiers used | 5 |
| Repeated outer-CV folds | 50 per dataset |
| Aggregated metric rows in the app | 3,850 |
| Observed experiment runtime | 51 min 28 sec, `--n-jobs 2` |

The downloadable raw predictions, checkpoints, and per-fold records stay local and are excluded from Git. The compact [`assets/experiment_summary.csv`](assets/experiment_summary.csv) and [`assets/experiment_provenance.json`](assets/experiment_provenance.json) are regenerated from those raw outputs with [`scripts/build_showcase_summary.py`](scripts/build_showcase_summary.py).

**Resume summary:** Built a reproducible BPSO feature-selection benchmark across five public datasets, ten selection methods, and five classifiers using repeated stratified cross-validation; packaged the results in an interactive Streamlit dashboard and FastAPI service.

The BPSO objective balances predictive performance and subset size:

\[
F(z) = \alpha\bigl(1-P(z)\bigr) + (1-\alpha)\frac{|z|}{d},
\]

where `z` is a binary feature mask, `P(z)` is the inner-CV score, `d` is the number of input features, and `α` controls the trade-off. The implementation is in [`src/bpso_fs/fitness.py`](src/bpso_fs/fitness.py) and [`src/bpso_fs/selectors/bpso.py`](src/bpso_fs/selectors/bpso.py).

## Quick start: local showcase

Requirements: Docker Desktop with Docker Compose. From the repository root, run:

```powershell
docker compose up --build -d
```

On first startup, Compose builds the image and trains a reduced BPSO model on the bundled Breast Cancer dataset. The trained artifact is stored in a Docker-managed volume.

| Service | URL | Purpose |
| --- | --- | --- |
| Streamlit showcase | [http://localhost:8501](http://localhost:8501) | Review model metadata, inspect experimental results, and submit CSV records |
| FastAPI | [http://localhost:8000/docs](http://localhost:8000/docs) | Interactive API documentation |
| Readiness | [http://localhost:8000/health](http://localhost:8000/health) | Confirms the prediction model is loaded |

Stop the services with `docker compose down`. This stops containers without deleting the model volume.

The showcase model is a separate full-data fit for inference. Its BPSO search score is **not** an unbiased estimate of generalization performance. Use the repeated-CV experiment below for evaluation claims.

### Professor walkthrough

1. Open **Evaluation results**, select **Main experiment**, then choose a dataset, classifier, and metric. Explain that the table compares BPSO with nine baselines over 50 outer-fold runs per dataset.
2. Use **Methodology & provenance** to show the run configuration, source dataset, and how the dashboard summary was built from the experiment CSVs.
3. Use **Try a prediction** only to demonstrate the separate inference API. The prediction model was fit on the complete Breast Cancer benchmark, so its demo predictions are not test-set evidence.

The performance numbers shown in **Evaluation results** are genuine outputs from the completed five-dataset reduced run, aggregated by [`scripts/build_showcase_summary.py`](scripts/build_showcase_summary.py). They are not manually entered scores. The summary includes mean and standard deviation across outer folds; present the paired statistical tests from the generated report when discussing significance. Do not describe the BPSO inner fitness/search score as held-out accuracy.

## Reproduce the five-dataset experiment

The research job compares methods on shared stratified folds and saves per-fold metrics, predictions, feature selections, BPSO/GA histories, and run metadata. Results are written to `results_experiment/` and surfaced by the app's **Experimental evaluation** section.

### Dataset setup

Four datasets are downloaded or loaded automatically. CKD is distributed as a RAR archive; extract its ARFF file to:

```text
data/raw/chronic_kidney_disease.arff
```

The loader supports the UCI archive's `wbcc`/`rbcc` names and maps them to the project's `wc`/`rc` schema. The Compose job uses `--require-all-datasets`, so it fails clearly if any requested dataset is missing.

### Reduced run

The reduced configuration uses 5 seeds × 10 outer folds, a 20-particle × 50-iteration swarm, the configured baselines/classifiers, and an α sensitivity sweep:

```powershell
docker compose --profile research run --build --rm research --config configs/all_datasets.yaml --dataset all --n-jobs 2 --require-all-datasets --results-dir /results
```

When the run completes, refresh the versioned dashboard summary from the full per-fold outputs:

```powershell
docker compose --profile research run --rm --entrypoint python research scripts/build_showcase_summary.py --results-dir /results --output /app/assets/experiment_summary.csv
```

Then rebuild/restart the app and refresh the browser:

```powershell
docker compose up --build -d
```

The evaluation panel summarizes outer-fold scores by dataset, classifier, and metric. It reports means and standard deviations; these are not independent confidence intervals.

### Full protocol

[`configs/all_datasets_full.yaml`](configs/all_datasets_full.yaml) uses 30 seeds and a 30-particle × 100-iteration swarm across all five datasets. It is substantially more expensive than the reduced run:

```powershell
docker compose --profile research run --build --rm research --config configs/all_datasets_full.yaml --dataset all --n-jobs 2 --require-all-datasets --results-dir /results
```

### Observed runtime

One reduced five-dataset run completed on an Intel Core Ultra 5 225H with `--n-jobs 2` in about **51 minutes 28 seconds**, plus about **105 seconds** to build the container image. Treat this as a reference, not a guarantee; power mode, cooling, Docker resource limits, and data download time affect results.

### Generate publication artifacts

After the experiment completes, generate tables and figures from the saved CSV outputs:

```powershell
docker compose --profile research run --rm --entrypoint python research scripts/make_tables.py --results-dir /results --tables-dir /paper/tables
docker compose --profile research run --rm --entrypoint python research scripts/make_figures.py --results-dir /results --figures-dir /paper/figures
```

The generated report artifacts are written to `paper/tables/` and `paper/figures/`. Review them alongside the per-run metadata before citing or presenting results.

## Run experiments with Python

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,serve]"
```

Run one dataset or the default three-dataset paper configuration:

```powershell
python scripts/run_experiments.py --config configs/reduced.yaml --dataset breast_cancer --n-jobs 2
python scripts/run_experiments.py --config configs/default.yaml --dataset all --n-jobs 2
```

Useful options include `--seeds`, `--folds`, `--quick`, `--no-alpha-sweep`, `--results-dir`, and `--require-all-datasets`. The runner checkpoints each `(dataset, method, seed, fold)` task and can resume compatible work.

To train a final inference artifact independently of the evaluation run:

```powershell
python scripts/train_model.py --config configs/reduced.yaml --dataset breast_cancer --output models/disease_model.joblib
```

## Evaluation protocol

1. Repeated stratified outer folds partition each dataset. Every method uses the same folds for paired comparisons.
2. Selectors operate on a one-column-per-feature numeric view. The final classifier pipeline handles imputation, scaling, and categorical encoding using the selected original columns.
3. BPSO and the genetic algorithm optimize the configured inner-CV fitness on the outer training partition. The outer test partition is reserved for scoring.
4. Metrics include accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC, PR-AUC, selected-feature count, selection time, and fit time.
5. Statistical outputs include paired Wilcoxon tests with Holm correction, Friedman/Nemenyi comparisons, subset stability, and feature-selection frequencies.

The inner wrapper fitness is a model-selection score, not the final evaluation score. For conclusions, use the outer-fold results and the paired statistical analyses. See [`src/bpso_fs/experiment.py`](src/bpso_fs/experiment.py), [`src/bpso_fs/evaluation.py`](src/bpso_fs/evaluation.py), and [`tests/test_no_leakage.py`](tests/test_no_leakage.py).

## Datasets

| Dataset key | Samples × features | Source | Positive class |
| --- | ---: | --- | --- |
| `breast_cancer` | 569 × 30 | scikit-learn Wisconsin Diagnostic Breast Cancer | Malignant |
| `pima_diabetes` | 768 × 8 | Pima Indians Diabetes benchmark CSV | Diabetes |
| `heart_cleveland` | 303 × 13 | UCI Cleveland Heart Disease; ISLR mirror fallback | Disease present |
| `ilpd` | 583 × 10 | UCI Indian Liver Patient Dataset | Liver patient |
| `ckd` | 400 × 24 | UCI Chronic Kidney Disease | CKD |

Data source details and citations are maintained by the loaders in [`src/bpso_fs/data/loaders.py`](src/bpso_fs/data/loaders.py). Downloaded raw files and model artifacts are local-only and excluded from Git.

## API contract

`POST /predict` accepts a JSON object with a `records` array. Each record must include all input features listed by `GET /metadata`; batches are limited to 1,000 records. The response includes a binary label and positive-class probability for each row. `GET /health` is ready only after the artifact loads. Set `BPSO_MODEL` to change the model artifact path.

The API and UI are configured for local research demonstrations. Before exposing them outside a trusted environment, add the authentication, TLS termination, access controls, monitoring, and data-retention policies required by the deployment platform.

## Publish a shareable demo

The current deployment is a three-container Compose stack (one-time model trainer, FastAPI, and Streamlit). The simplest way to publish this exact setup is an Ubuntu VPS; Streamlit Community Cloud deploys a Streamlit entrypoint from GitHub and does not run this Compose stack as-is.

1. Push this repository to a GitHub repository. Keep the repository public if you want the code and experiment evidence to be reviewable. The large per-fold files under `results_experiment/`, downloaded datasets, and local model artifacts are intentionally ignored; the compact, real experiment summary and provenance under `assets/` are included.
2. Create an Ubuntu 24.04 VPS. For this small research showcase, choose at least 2 vCPU and 4 GB RAM. Add a cloud firewall rule for SSH (port 22, restricted to your IP). For a temporary IP-based demo, allow TCP 8501 for Streamlit and TCP 8000 for FastAPI.
3. Install Docker Engine and the Compose plugin using [Docker's official Ubuntu instructions](https://docs.docker.com/engine/install/ubuntu/). Connect over SSH, clone your GitHub repository, and enter its directory:

   ```bash
   git clone https://github.com/<your-user>/bpso-disease-prediction.git
   cd bpso-disease-prediction
   ```

4. Start the application:

   ```bash
   docker compose up --build -d
   docker compose ps
   docker compose logs -f trainer api showcase
   ```

   The first start builds the image and fits the separate Breast Cancer inference model into a persistent Docker volume. It does not rerun the five-dataset research experiment. The results tab reads the checked-in aggregate summary when the ignored local raw result directory is absent.
5. Set the API docs URL in the ignored `.env` file on the VPS, replacing the example IP:

   ```bash
   echo 'BPSO_API_DOCS_URL=http://YOUR_VPS_IP:8000/docs' > .env
   docker compose up --build -d
   ```

   Then share `http://YOUR_VPS_IP:8501`.

   For a polished long-lived link, point a domain to the VPS and put a TLS reverse proxy in front of the app and API, exposing only HTTPS publicly. The raw-IP route is suitable for a short classroom demo, not sensitive records. Do not upload identifiable patient information; this is a research demonstration, not a diagnostic service.

Useful operations:

```bash
docker compose logs -f showcase api
docker compose restart showcase api
docker compose down
```

## Repository layout

```text
configs/                 Experiment configurations and dataset declarations
data/raw/                Downloaded source files (ignored by Git)
paper/figures/           Generated report figures
paper/tables/            Generated report tables
scripts/                 Experiment, training, and report entry points
src/bpso_fs/             Loaders, selectors, fitness, evaluation, API
tests/                   Unit, leakage, and pipeline smoke tests
results/                 Checked-in summaries; large run artifacts are ignored
results_experiment/      Local all-dataset run outputs (ignored by Git)
```

## Quality checks

```powershell
python -m pytest -q
python -m ruff check src scripts tests app.py
python -m ruff format --check src scripts tests app.py
```

GitHub Actions runs tests, lint/format checks, and a Docker image build on pushes and pull requests.

## Data and method references

- Wolberg et al., *Breast Cancer Wisconsin (Diagnostic)*, UCI Machine Learning Repository, doi:10.24432/C5DW2B.
- Smith et al., “Using the ADAP learning algorithm to forecast the onset of diabetes mellitus,” *SCAMC*, 1988.
- Janosi et al., *Heart Disease*, UCI Machine Learning Repository, doi:10.24432/C52P4X.
- Ramana and Venkateswarlu, *ILPD*, UCI Machine Learning Repository, doi:10.24432/C5D02C.
- Rubini et al., *Chronic Kidney Disease*, UCI Machine Learning Repository, doi:10.24432/C5G020.
- Kennedy and Eberhart, “A discrete binary version of the particle swarm algorithm,” *IEEE SMC*, 1997.
- Demšar, “Statistical comparisons of classifiers over multiple data sets,” *JMLR*, 2006.
