PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.DEFAULT_GOAL := help
.PHONY: help setup data panel audit docs test lint format typecheck check clean clean-cache

help:
	@echo "setup      create .venv and install the pinned dependencies"
	@echo "data       download the configured universe into the local cache"
	@echo "panel      build the pooled feature and target panel"
	@echo "audit      run the leakage, truncation and scale audits"
	@echo "docs       regenerate docs/FEATURES.md from the feature registry"
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
