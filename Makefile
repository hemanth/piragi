.PHONY: install test format lint type-check clean build publish help

help:
	@echo "Ragi Development Commands"
	@echo "========================="
	@echo "install      - Install package in development mode"
	@echo "test         - Run tests"
	@echo "test-cov     - Run tests with coverage"
	@echo "format       - Format code with black"
	@echo "lint         - Lint code with ruff"
	@echo "type-check   - Check types with mypy"
	@echo "clean        - Remove build artifacts"
	@echo "build        - Build package"
	@echo "publish      - Publish to PyPI"

install:
	pip install -e ".[dev]"

test:
	pytest

test-cov:
	pytest --cov=piragi --cov-report=term-missing --cov-report=html

format:
	black src/ tests/ examples/

lint:
	ruff check src/ tests/ examples/

type-check:
	mypy src/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

publish: build
	python -m twine upload dist/*
