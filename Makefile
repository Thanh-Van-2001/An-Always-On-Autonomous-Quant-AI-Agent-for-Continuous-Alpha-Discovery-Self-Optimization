# =============================================================================
# AQAA - Makefile
# =============================================================================
# Common commands for development and reproducing KDD 2026 paper experiments.
# =============================================================================

.DEFAULT_GOAL := help
PYTHON       ?= python
PIP          ?= pip

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

.PHONY: install
install: ## Install the package in editable mode with all optional deps
	$(PIP) install -e ".[all,dev]"

.PHONY: install-core
install-core: ## Install only core dependencies (no LLM/live/dev)
	$(PIP) install -e .

.PHONY: install-dev
install-dev: ## Install core + dev dependencies
	$(PIP) install -e ".[dev]"

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

.PHONY: test
test: ## Run the test suite
	$(PYTHON) -m pytest tests/ -v --tb=short

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=aqaa --cov-report=term-missing --cov-report=html

.PHONY: lint
lint: ## Run linters (ruff + mypy)
	$(PYTHON) -m ruff check aqaa/ tests/
	$(PYTHON) -m mypy aqaa/

.PHONY: format
format: ## Auto-format code with black and ruff
	$(PYTHON) -m black aqaa/ tests/ scripts/
	$(PYTHON) -m ruff check --fix aqaa/ tests/

.PHONY: check
check: lint test ## Run all quality checks (lint + test)

# ---------------------------------------------------------------------------
# Paper experiments
# ---------------------------------------------------------------------------

.PHONY: run-backtest
run-backtest: ## Run the full backtest pipeline (Table 1 reproduction)
	$(PYTHON) -m aqaa.experiments.backtest \
		--config configs/default.yaml \
		--output results/backtest/

.PHONY: run-ablation
run-ablation: ## Run the ablation study (Table 2 reproduction)
	$(PYTHON) -m aqaa.experiments.ablation \
		--config configs/default.yaml \
		--output results/ablation/

.PHONY: run-tif-analysis
run-tif-analysis: ## Run TIF leakage-detection analysis (Table 3 reproduction)
	$(PYTHON) -m aqaa.experiments.tif_analysis \
		--config configs/default.yaml \
		--output results/tif/

.PHONY: run-cross-market
run-cross-market: ## Run cross-market generalization experiments
	$(PYTHON) -m aqaa.experiments.cross_market \
		--configs configs/markets/ \
		--output results/cross_market/

.PHONY: run-all
run-all: run-backtest run-ablation run-tif-analysis run-cross-market ## Run all experiments

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

.PHONY: clean
clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .mypy_cache .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
