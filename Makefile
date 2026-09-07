PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.DEFAULT_GOAL := help
.PHONY: help setup data panel audit docs walkforward tune leaderboard results demo-data demo analysis serve setup-web web verify test lint format typecheck check clean clean-cache

help:
	@echo "setup      create .venv and install the pinned dependencies"
	@echo "data       download the configured universe into the local cache"
	@echo "panel      build the pooled feature and target panel"
	@echo "audit      run the leakage, truncation and scale audits"
	@echo "docs       regenerate docs/FEATURES.md from the feature registry"
	@echo "tune       hyperparameter search on the earliest folds only"
	@echo "walkforward  run walk-forward evaluation and record the results"
	@echo "leaderboard  compare recorded runs against their baselines"
	@echo "results    regenerate docs/RESULTS.md from the recorded runs"
	@echo "demo-data  rebuild the compact artefacts shipped with the repo"
	@echo "demo       regenerate one formulation end to end, about 20 minutes"
	@echo "analysis   regenerate docs/ANALYSIS.md (regimes, SHAP, errors, friction)"
	@echo "serve      run the local API on 127.0.0.1:8000"
	@echo "setup-web  install the frontend dependencies"
	@echo "web        run the dashboard on localhost:3000 (needs serve in another terminal)"
	@echo "verify     quick end-to-end check: fetch, audit, and a short run"
	@echo "test       run the test suite"
	@echo "lint       ruff check"
	@echo "format     ruff format"
	@echo "typecheck  mypy"
	@echo "check      lint, typecheck and test"

setup:
	@command -v brew >/dev/null 2>&1 && brew list libomp >/dev/null 2>&1 || \
		echo "note: xgboost and lightgbm need OpenMP on macOS (brew install libomp)"
	python3 -m venv .venv
	$(PIP) install --quiet --upgrade pip setuptools wheel
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e . --no-deps
	@echo "done: activate with 'source .venv/bin/activate'"

data:
	$(PYTHON) scripts/fetch_data.py

panel:
	$(PYTHON) scripts/build_panel.py

audit:
	$(PYTHON) scripts/audit_leakage.py

docs:
	$(PYTHON) scripts/feature_docs.py

tune:
	$(PYTHON) scripts/tune_hyperparams.py

walkforward:
	$(PYTHON) scripts/run_walkforward.py --groups dev --experiment main

leaderboard:
	$(PYTHON) scripts/leaderboard.py

results:
	$(PYTHON) scripts/make_results.py

demo-data:
	$(PYTHON) scripts/build_demo_data.py

# Regenerates the shipped prediction history for one formulation from scratch. Useful if
# you want to confirm the committed artefacts came out of the pipeline in this repository.
demo:
	$(PYTHON) scripts/run_walkforward.py --groups dev --targets excess_direction \
		--horizons 1 --experiment demo
	$(PYTHON) scripts/build_demo_data.py --experiment demo

analysis:
	$(PYTHON) scripts/make_analysis.py

serve:
	$(PYTHON) scripts/serve.py

setup-web:
	cd frontend && npm install

web:
	cd frontend && npm run dev

verify:
	$(PYTHON) scripts/fetch_data.py --groups dev
	$(PYTHON) scripts/audit_leakage.py --sample 1
	$(PYTHON) scripts/build_panel.py --groups dev
	$(PYTHON) scripts/run_walkforward.py --groups dev --targets direction --horizons 1 \
		--max-folds 3 --experiment verify

test:
	$(PYTHON) -m pytest tests/ -q

lint:
	.venv/bin/ruff check src tests scripts

format:
	.venv/bin/ruff format src tests scripts

typecheck:
	.venv/bin/mypy

check: lint typecheck test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find src tests scripts -name __pycache__ -type d -exec rm -rf {} +

clean-cache:
	rm -rf data/cache artifacts
