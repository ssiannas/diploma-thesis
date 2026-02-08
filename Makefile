.PHONY: help install install-dev test test-verbose test-coverage clean format lint type-check docs run-tests setup venv demo train

# Default target
help:
	@echo "Available commands:"
	@echo "  make install         - Install package in editable mode"
	@echo "  make install-dev     - Install package with dev dependencies"
	@echo "  make venv            - Create fresh virtual environment"
	@echo "  make setup           - Full setup: venv + install-dev"
	@echo "  make demo            - Run baseline model demo"
	@echo "  make train           - Train baseline model"
	@echo "  make test            - Run tests with pytest"
	@echo "  make test-verbose    - Run tests with verbose output"
	@echo "  make test-coverage   - Run tests with coverage report"
	@echo "  make format          - Format code with black and isort"
	@echo "  make lint            - Run linting with flake8 and pylint"
	@echo "  make type-check      - Run type checking with mypy"
	@echo "  make clean           - Remove build artifacts and cache"
	@echo "  make docs            - Build documentation"

# Python and virtual environment
PYTHON := python3
VENV := .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
ACTIVATE := . $(VENV)/bin/activate

# Create virtual environment
venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Upgrading pip, setuptools, and wheel..."
	$(VENV_PIP) install --upgrade pip setuptools wheel
	@echo "Virtual environment created at $(VENV)"
	@echo "Activate with: source $(VENV)/bin/activate"

# Install package in editable mode
install:
	@echo "Installing package in editable mode..."
	pip install -e .

# Install with development dependencies
install-dev:
	@echo "Installing package with development dependencies..."
	pip install -e ".[dev,jupyter]"
	pip install -r requirements-dev.txt
	@echo "Installing pre-commit hooks..."
	pre-commit install

# Full setup from scratch
setup: venv
	@echo "Installing all dependencies..."
	$(VENV_PIP) install -e ".[dev,jupyter]"
	$(VENV_PIP) install -r requirements-dev.txt
	@echo "Installing pre-commit hooks..."
	$(VENV)/bin/pre-commit install
	@echo "Setup complete! Activate with: source $(VENV)/bin/activate"

# Testing
test:
	@echo "Running tests..."
	pytest tests/

test-verbose:
	@echo "Running tests with verbose output..."
	pytest -v -s tests/

test-coverage:
	@echo "Running tests with coverage..."
	pytest --cov=pcml --cov-report=term-missing --cov-report=html tests/
	@echo "Coverage report generated in htmlcov/index.html"

# Code quality
format:
	@echo "Formatting code with black..."
	black pcml/ tests/ scripts/
	@echo "Sorting imports with isort..."
	isort pcml/ tests/ scripts/

lint:
	@echo "Running flake8..."
	flake8 --max-line-length=88 --extend-ignore=E203,W503,E402,F401,F841,F541 pcml/ tests/ scripts/
	@echo "Running black check..."
	black --check pcml/ tests/ scripts/
	@echo "Running isort check..."
	isort --check-only pcml/ tests/ scripts/

type-check:
	@echo "Running mypy type checker..."
	mypy pcml/

# Documentation
docs:
	@echo "Building documentation..."
	cd docs && make html
	@echo "Documentation built in docs/_build/html/index.html"

# Demo and Training
demo:
	@echo "Running baseline model demo..."
	$(VENV_PYTHON) scripts/demo_baseline.py

train:
	@echo "Training baseline model..."
	$(VENV_PYTHON) scripts/train_baseline.py

# Cleaning
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "Clean complete!"

# Development workflow shortcuts
check: format lint type-check test
	@echo "All checks passed!"

# Quick test for data loaders
test-data:
	@echo "Testing data loaders..."
	pytest -v tests/test_data.py

# Quick test for metrics
test-metrics:
	@echo "Testing metrics..."
	pytest -v tests/test_metrics.py
