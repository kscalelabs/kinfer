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

# Define Python files to process (excluding ref/ and proto/)
PYTHON_FILES := $(shell find . -name "*.py" ! -path "./ref/*" ! -path "*/proto/*" ! -path "./build/*")

format:
	@black $(PYTHON_FILES)
	@ruff format $(PYTHON_FILES)
	@cargo fmt --all
.PHONY: format

static-checks:
	@black --diff --check $(PYTHON_FILES)
	@ruff check $(PYTHON_FILES)
	@mkdir -p .mypy_cache
	@mypy --install-types --non-interactive $(PYTHON_FILES)
	@cargo clippy --all-targets --all-features -- -D warnings
.PHONY: static-checks

# Unit tests

test:
	python -m pytest
.PHONY: test
