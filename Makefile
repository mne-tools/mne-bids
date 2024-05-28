.PHONY: all clean-pyc clean-so clean-build clean-ctags clean-cache clean-e clean inplace test ruff-check ruff-format pep build-doc dist-build

all: clean inplace pep test build-doc dist-build

clean-pyc:
	find . -name "*.pyc" | xargs rm -f

clean-so:
	find . -name "*.so" | xargs rm -f
	find . -name "*.pyd" | xargs rm -f

clean-build:
	rm -rf _build

clean-ctags:
	rm -f tags

clean-cache:
	find . -name "__pycache__" | xargs rm -rf

clean-e:
	find . -name "*-e" | xargs rm -rf

clean: clean-build clean-pyc clean-so clean-ctags clean-cache clean-e

inplace:
	@python -m pip install -e ".[dev]"

test:
	@echo "Running tests"
	@python -m pytest . \
	--doctest-modules \
	--cov=mne_bids mne_bids/tests/ mne_bids/commands/tests/ \
	--cov-report=xml \
	--cov-config=pyproject.toml \
	--verbose \
	--ignore mne-python \
	--ignore examples

ruff-format:
	@echo "Running ruff format"
	@ruff format mne_bids/
	@ruff format examples/

ruff-check:
	@echo "Running ruff check"
	@ruff check mne_bids/
	@ruff check examples/ --ignore=D103,D400,D205

pep: ruff-check ruff-format

build-doc:
	@echo "Building documentation"
	make -C doc/ clean
	make -C doc/ html
	cd doc/ && make view

dist-build:
	@echo "Building dist"
	rm -rf dist
	@python -m pip install build twine
	@python -m build
	@python -m twine check --strict dist/*
