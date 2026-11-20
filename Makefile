# Usage:
#   make setup                 install the package and dependencies
#   make all                   full pipeline with configs/default.yaml (slow; see README)
#   make all QUICK=1           smoke test (2 seeds, 3 folds, 10 particles, 10 iterations)
#   make all CONFIG=configs/reduced.yaml
#   make test / make lint

PYTHON ?= python
CONFIG ?= configs/default.yaml
DATASET ?= all
NJOBS ?= -1
QUICK ?=
QUICK_FLAG := $(if $(QUICK),--quick,)
# quick runs go to results_quick/ so they never overwrite the real results, tables or figures
RESULTS ?= $(if $(QUICK),results_quick,results)
FIGURES ?= $(if $(QUICK),results_quick/figures,paper/figures)
TABLES ?= $(if $(QUICK),results_quick/tables,paper/tables)

.PHONY: setup data experiments figures tables test lint format all clean model serve showcase

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e ".[serve,dev]"

data:
	$(PYTHON) scripts/download_data.py

experiments:
	$(PYTHON) scripts/run_experiments.py --config $(CONFIG) --dataset $(DATASET) --n-jobs $(NJOBS) --results-dir $(RESULTS) $(QUICK_FLAG)

figures:
	$(PYTHON) scripts/make_figures.py --results-dir $(RESULTS) --figures-dir $(FIGURES)

tables:
	$(PYTHON) scripts/make_tables.py --results-dir $(RESULTS) --tables-dir $(TABLES)

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check src scripts tests
	black --check src scripts tests

format:
	ruff check --fix src scripts tests
	black src scripts tests

all: data experiments figures tables

model:
	$(PYTHON) scripts/train_model.py --config configs/reduced.yaml --dataset breast_cancer --output models/disease_model.joblib

serve:
	$(PYTHON) -m uvicorn bpso_fs.api:app --host 0.0.0.0 --port 8000

showcase:
	$(PYTHON) -m streamlit run app.py

clean:
	rm -rf $(RESULTS)/*/runs
