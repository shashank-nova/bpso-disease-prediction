"""Streamlit portfolio interface for BPSO experiments and inference."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("BPSO_API_URL", "http://127.0.0.1:8000")
API_DOCS_URL = os.getenv("BPSO_API_DOCS_URL", f"{API_URL}/docs")
RESULTS_DIR = Path(os.getenv("BPSO_RESULTS_DIR", "results_experiment"))
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
SUMMARY_FILE = ASSETS_DIR / "experiment_summary.csv"
PROVENANCE_FILE = ASSETS_DIR / "experiment_provenance.json"

METHOD_LABELS = {
    "all": "All features",
    "bpso": "BPSO",
    "ga": "Genetic algorithm",
    "mi_k": "Mutual information (fixed k)",
    "mi_cv": "Mutual information (CV tuned)",
    "anova_k": "ANOVA-F (fixed k)",
    "anova_cv": "ANOVA-F (CV tuned)",
    "rfe_k": "RFE (fixed k)",
    "l1": "L1 logistic regression",
    "random_k": "Random subset",
}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall / sensitivity",
    "specificity": "Specificity",
    "f1": "F1 score",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "n_selected": "Selected features",
    "selection_time": "Selection time (seconds)",
    "fit_time": "Classifier fit time (seconds)",
    "total_time": "Total fit time (seconds)",
}

st.set_page_config(
    page_title="BPSO Disease Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
      .block-container {max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem;}
      [data-testid="stMetric"] {background:#f4f7fb; border:1px solid #e2e8f0;
        padding:1rem 1.1rem; border-radius:12px;}
      [data-testid="stMetricLabel"] p {font-size:.9rem;}
      div[data-testid="stTabs"] button {font-weight:600;}
      .small-muted {color:#64748b; font-size:.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def load_results() -> tuple[pd.DataFrame, bool]:
    """Prefer local raw run CSVs; fall back to the checked-in aggregate for clean clones."""
    frames = []
    if RESULTS_DIR.exists():
        for dataset_dir in sorted(path for path in RESULTS_DIR.iterdir() if path.is_dir()):
            for filename, study in (
                ("raw_results.csv", "cross_validation"),
                ("alpha_sweep_results.csv", "alpha_sweep"),
            ):
                result_file = dataset_dir / filename
                if result_file.is_file():
                    try:
                        frame = pd.read_csv(result_file)
                        frame["study"] = study
                        frames.append(frame)
                    except (OSError, pd.errors.ParserError):
                        continue
    if frames:
        return pd.concat(frames, ignore_index=True), True
    if SUMMARY_FILE.is_file():
        return pd.read_csv(SUMMARY_FILE), False
    return pd.DataFrame(), False


def load_provenance() -> dict:
    """Load run provenance from local run metadata, then packaged metadata."""
    if RESULTS_DIR.exists():
        dataset_metadata = []
        for dataset_dir in sorted(path for path in RESULTS_DIR.iterdir() if path.is_dir()):
            path = dataset_dir / "run_metadata.json"
            if path.is_file():
                try:
                    dataset_metadata.append(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError):
                    continue
        if dataset_metadata:
            return {
                "config": dataset_metadata[0].get("config", {}),
                "datasets": dataset_metadata,
            }
    if PROVENANCE_FILE.is_file():
        try:
            return json.loads(PROVENANCE_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def method_name(method: str) -> str:
    if method.startswith("bpso_alpha"):
        return f"BPSO (α={method.removeprefix('bpso_alpha')})"
    return METHOD_LABELS.get(method, method.replace("_", " ").title())


def format_metric(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def main() -> None:
    st.title("🩺 BPSO Disease Prediction")
    st.caption("Feature selection research · Five public benchmarks · Reproducible evaluation")

    try:
        response = requests.get(f"{API_URL}/metadata", timeout=3)
        response.raise_for_status()
        model_metadata = response.json()
    except requests.RequestException:
        st.error(
            "The prediction service is not ready. Start it with `docker compose up --build -d`."
        )
        st.stop()

    results, results_are_raw = load_results()
    provenance = load_provenance()
    datasets = sorted(results["dataset"].dropna().unique()) if not results.empty else []
    main_results = (
        results[results["study"] == "cross_validation"] if not results.empty else pd.DataFrame()
    )
    classifiers = (
        sorted(main_results["classifier"].dropna().unique()) if not main_results.empty else []
    )
    methods_count = main_results["method"].nunique() if not main_results.empty else 0
    experiment_config = provenance.get("config", {}).get("experiment", {})
    outer_runs = int(experiment_config.get("n_seeds", 0)) * int(
        experiment_config.get("outer_folds", 0)
    )

    with st.sidebar:
        st.success("Prediction API connected")
        st.link_button("Open API documentation", API_DOCS_URL, use_container_width=True)
        st.divider()
        st.markdown("**Inference model**")
        st.write(model_metadata["dataset"].replace("_", " ").title())
        st.caption(f"Classifier: {model_metadata['classifier']}")
        st.caption(
            "This model is a separate full-data fit; experiment results are shown in the Evaluation tab."
        )

    overview_tab, evaluation_tab, predict_tab, methodology_tab = st.tabs(
        ["Overview", "Evaluation results", "Try a prediction", "Methodology & provenance"]
    )

    with overview_tab:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Benchmarks evaluated", len(datasets) if datasets else "—")
        m2.metric("Methods", methods_count if datasets else "—")
        m3.metric("Classifiers", len(classifiers) if classifiers else "—")
        m4.metric("Outer-fold runs / dataset", outer_runs if datasets else "—")

        st.subheader("What this project does")
        st.write(
            "BPSO searches for compact feature subsets. The experiment compares it with "
            "nine feature-selection baselines using the same repeated stratified folds. "
            "The evaluation table reports outer-fold results; the live prediction model "
            "is a separate artifact trained on the full Breast Cancer benchmark."
        )
        left, right = st.columns(2)
        with left:
            st.markdown("#### Experimental evidence")
            st.write("Choose a dataset, classifier, and metric in **Evaluation results**.")
            st.write(
                "The bundled summary is generated from the experiment CSVs, not typed into the UI."
            )
        with right:
            st.markdown("#### Live inference")
            st.write("Upload a feature-only CSV in **Try a prediction** to score records.")
            st.warning(
                "The displayed BPSO search score is for model selection, not independent test performance."
            )

        if not datasets:
            st.info(
                "Bundled experiment results are not available. See the README to run the five-dataset experiment."
            )

    with evaluation_tab:
        st.subheader("Repeated cross-validation results")
        if results.empty:
            st.info(
                "No experiment summary is available. Run the research workflow described in the README."
            )
        else:
            studies = sorted(results["study"].dropna().unique())
            study = st.selectbox(
                "Evaluation set",
                studies,
                format_func=lambda value: (
                    "Main experiment" if value == "cross_validation" else "α sensitivity"
                ),
                help="Main experiment compares feature-selection methods; α sensitivity compares BPSO trade-offs.",
            )
            study_results = results[results["study"] == study]
            c1, c2, c3 = st.columns(3)
            dataset = c1.selectbox("Dataset", sorted(study_results["dataset"].dropna().unique()))
            dataset_results = study_results[study_results["dataset"] == dataset]
            classifier = c2.selectbox(
                "Classifier", sorted(dataset_results["classifier"].dropna().unique())
            )
            classifier_results = dataset_results[dataset_results["classifier"] == classifier]
            if results_are_raw:
                metric_choices = [
                    metric for metric in METRIC_LABELS if metric in classifier_results.columns
                ]
            else:
                available_metrics = set(classifier_results["metric"].dropna().unique())
                metric_choices = [metric for metric in METRIC_LABELS if metric in available_metrics]
            metric = c3.selectbox(
                "Metric",
                metric_choices,
                format_func=format_metric,
                help="Predictive scores are higher-is-better. Time and feature-count metrics are lower-is-better.",
            )

            if results_are_raw:
                summary = (
                    classifier_results.groupby("method")[metric]
                    .agg(mean="mean", std="std", runs="count")
                    .reset_index()
                )
            else:
                summary = classifier_results[classifier_results["metric"] == metric][
                    ["method", "mean", "std", "n_runs"]
                ].rename(columns={"n_runs": "runs"})

            lower_is_better = metric in {"n_selected", "selection_time", "fit_time", "total_time"}
            summary = summary.sort_values("mean", ascending=lower_is_better)
            summary["method_label"] = summary["method"].map(method_name)
            summary["mean ± SD"] = summary.apply(
                lambda row: (
                    f"{row['mean']:.3f} ± {row['std']:.3f}"
                    if pd.notna(row["std"])
                    else f"{row['mean']:.3f}"
                ),
                axis=1,
            )

            best = summary.iloc[0]
            score_unit = "s" if metric.endswith("time") else ""
            v1, v2, v3 = st.columns(3)
            v1.metric("Top method for this view", best["method_label"])
            v2.metric(format_metric(metric), f"{best['mean']:.3f}{score_unit}")
            v3.metric("Outer-fold runs", int(best["runs"]))

            chart_data = summary.set_index("method_label")[["mean"]]
            st.bar_chart(chart_data, horizontal=True, height=360)
            shown = summary[["method_label", "mean", "std", "runs"]].rename(
                columns={
                    "method_label": "Method",
                    "mean": "Mean",
                    "std": "Standard deviation",
                    "runs": "Fold runs",
                }
            )
            st.dataframe(shown, use_container_width=True, hide_index=True)
            st.download_button(
                "Download this comparison",
                shown.to_csv(index=False),
                file_name=f"{dataset}_{study}_{classifier}_{metric}.csv",
                mime="text/csv",
            )
            st.caption(
                "Values are means and standard deviations across outer-fold runs. Repeated-CV folds overlap and are not independent confidence intervals."
            )

            source = None
            for entry in provenance.get("datasets", []):
                if entry.get("dataset") == dataset:
                    source = entry.get("data_source")
                    break
            if source:
                st.caption(f"Dataset source recorded for this run: `{source}`")

    with predict_tab:
        st.subheader("Score records with the inference model")
        st.info(
            f"This endpoint uses the {model_metadata['dataset'].replace('_', ' ').title()} "
            f"{model_metadata['classifier']} model. It was fit on {model_metadata['training_rows']} "
            "public benchmark rows; its predictions are a demonstration, not independent evaluation."
        )
        st.caption(
            "The probability is for class 1. In this Breast Cancer dataset, class 1 means malignant."
        )
        with st.expander("Required CSV columns", expanded=False):
            st.code(",".join(model_metadata["features"]), language="text")
        template = pd.DataFrame(columns=model_metadata["features"])
        st.download_button(
            "Download empty CSV template",
            template.to_csv(index=False),
            file_name=f"{model_metadata['dataset']}_input_template.csv",
            mime="text/csv",
        )
        with st.form("prediction_form"):
            uploaded = st.file_uploader(
                "Upload a feature-only CSV (up to 1,000 rows)", type=["csv"]
            )
            submitted = st.form_submit_button("Score records", type="primary")
        if submitted:
            if uploaded is None:
                st.error("Choose a CSV file first.")
            else:
                try:
                    frame = pd.read_csv(uploaded)
                except (UnicodeDecodeError, pd.errors.ParserError) as exc:
                    st.error(f"The file could not be read as CSV: {exc}")
                else:
                    missing = [
                        name for name in model_metadata["features"] if name not in frame.columns
                    ]
                    if frame.empty:
                        st.error("The CSV has a header but no data rows.")
                    elif len(frame) > 1000:
                        st.error("Upload at most 1,000 rows per request.")
                    elif missing:
                        st.error(f"Missing required columns: {', '.join(missing)}")
                    else:
                        extra = [
                            name for name in frame.columns if name not in model_metadata["features"]
                        ]
                        if extra:
                            st.caption(f"Extra columns will be ignored: {', '.join(extra)}")
                        records = (
                            frame[model_metadata["features"]]
                            .astype(object)
                            .where(pd.notna(frame[model_metadata["features"]]), None)
                        )
                        try:
                            prediction_response = requests.post(
                                f"{API_URL}/predict",
                                json={"records": records.to_dict("records")},
                                timeout=30,
                            )
                            prediction_response.raise_for_status()
                            output = pd.DataFrame(prediction_response.json()["predictions"])
                            output.insert(0, "row", range(1, len(output) + 1))
                            output = output.rename(
                                columns={
                                    "label": "Predicted class",
                                    "probability": "Class 1 probability",
                                }
                            )
                            st.success(f"Scored {len(output)} records.")
                            st.dataframe(output, use_container_width=True, hide_index=True)
                            st.download_button(
                                "Download predictions",
                                output.to_csv(index=False),
                                file_name="bpso_predictions.csv",
                                mime="text/csv",
                            )
                        except requests.RequestException as exc:
                            st.error(f"Prediction request failed: {exc}")

    with methodology_tab:
        st.subheader("How to interpret the numbers")
        st.markdown(
            "The main experiment trains/selects within each outer training fold and evaluates "
            "on that fold's held-out partition. The packaged dashboard summary is calculated "
            "from `raw_results.csv` and `alpha_sweep_results.csv` by "
            "`scripts/build_showcase_summary.py`."
        )
        st.latex(r"F(z)=\alpha(1-P(z))+(1-\alpha)\frac{|z|}{d}")
        st.markdown(
            "`P(z)` is the inner-CV wrapper score; the displayed outer-CV metrics are the "
            "experimental results. The model shown in **Try a prediction** is a separate final fit."
        )
        with st.expander("Run configuration and dataset provenance"):
            if provenance:
                config = provenance.get("config", {})
                experiment = config.get("experiment", {})
                bpso = config.get("bpso", {})
                st.write(
                    f"Seeds: {experiment.get('n_seeds', '—')} · Outer folds: "
                    f"{experiment.get('outer_folds', '—')} · BPSO swarm: "
                    f"{bpso.get('n_particles', '—')} particles × {bpso.get('n_iterations', '—')} iterations"
                )
                st.json(provenance.get("datasets", []), expanded=False)
            else:
                st.info(
                    "Run metadata is not bundled. Regenerate it from the raw experiment outputs."
                )
        st.warning(
            "These are public benchmark experiments, not prospective clinical validation. "
            "Do not use the predictions for patient care."
        )


main()
