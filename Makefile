# Makefile

# PyPI Build

build-for-pypi:
	@pip install --verbose build wheel twine
	@python -m build --sdist --wheel --outdir dist/ .
	@twine upload dist/*
.PHONY: build-for-pypi

push-to-pypi: build-for-pypi
	@twine upload dist/*
.PHONY: push-to-pypi

# Static Checks

format:
	@black .
	@ruff format
.PHONY: format

static-checks:
	@black --diff --check .
	@ruff check
	@mkdir -p .mypy_cache
	@mypy --install-types --non-interactive .
.PHONY: lint

# Unit tests

test:
	python -m pytest
.PHONY: test
