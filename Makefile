.PHONY: help setup lint fmt test cov api ui bench clean

help:  ## Show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-8s %s\n", $$1, $$2}'

setup:  ## Create the conda env, hooks and GitHub repo
	./scripts/setup.sh

lint:  ## Ruff lint + format check
	ruff check .
	ruff format --check .

fmt:  ## Auto-fix lint issues and format
	ruff check --fix .
	ruff format .

test:  ## Run the test suite
	python -m pytest -q

cov:  ## Tests with coverage
	python -m pytest --cov=goftar --cov-report=term-missing

api:  ## Run the REST API on 127.0.0.1:8000
	goftar serve

ui:  ## Run the web UI on 127.0.0.1:7860
	goftar ui

bench:  ## Benchmark on bench/dataset (MODELS=small,medium)
	goftar bench bench/dataset --models $${MODELS:-small}

clean:  ## Remove caches and build output
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml build dist src/*.egg-info .cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
