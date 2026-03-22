PAPER_TITLE := Rate-Conditioned Post-Processing for Learned Point Cloud Compression
PAPER_DIR   := paper
VENV        := .venv
SRC_DIRS    := pcml/ postprocessing/ tests/

.PHONY: help install test test-verbose test-coverage clean format lint type-check paper check

# Default target
help:
	@echo "Available commands:"
	@echo "  make install         - Install packages in editable mode"
	@echo "  make test            - Run tests with pytest"
	@echo "  make test-verbose    - Run tests with verbose output"
	@echo "  make test-coverage   - Run tests with coverage report"
	@echo "  make format          - Format code with black and isort"
	@echo "  make lint            - Lint with ruff, black check, isort check"
	@echo "  make type-check      - Run type checking with mypy"
	@echo "  make paper           - Build paper PDF"
	@echo "  make clean           - Remove build artifacts and cache"
	@echo "  make check           - Run format + lint + type-check + test"

# Install packages in editable mode
install:
	$(VENV)/bin/pip install -e .
	$(VENV)/bin/pip install -r requirements-dev.txt

# Testing
test:
	$(VENV)/bin/pytest tests/ --override-ini="addopts="

test-verbose:
	$(VENV)/bin/pytest -v -s tests/ --override-ini="addopts="

test-coverage:
	$(VENV)/bin/pytest --cov=pcml --cov=postprocessing --cov-report=term-missing --cov-report=html tests/ --override-ini="addopts="
	@echo "Coverage report generated in htmlcov/index.html"

# Code quality
format:
	$(VENV)/bin/black $(SRC_DIRS)
	$(VENV)/bin/isort $(SRC_DIRS)

lint:
	$(VENV)/bin/ruff check $(SRC_DIRS)
	$(VENV)/bin/black --check $(SRC_DIRS)
	$(VENV)/bin/isort --check-only $(SRC_DIRS)

type-check:
	$(VENV)/bin/mypy pcml/ postprocessing/

# Paper
paper:
	cd $(PAPER_DIR) && \
	pdflatex -interaction=nonstopmode main.tex && \
	bibtex main && \
	pdflatex -interaction=nonstopmode main.tex && \
	pdflatex -interaction=nonstopmode main.tex && \
	cp main.pdf "$(PAPER_TITLE).pdf"
	@echo "Built: $(PAPER_DIR)/$(PAPER_TITLE).pdf"

# Cleaning
clean:
	rm -rf build/ dist/ *.egg-info htmlcov/ .coverage .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Full check
check: format lint type-check test
