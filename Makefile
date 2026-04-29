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
	pytest mne_bids -v

build-doc:
	@echo "Building documentation"
	make -C doc/ clean
	make -C doc/ html
	@if [ -z "$$CI" ]; then cd doc/ && make view; fi

dist-build:
	@echo "Building dist"
	rm -rf dist
	@python -m pip install build twine
	@python -m build
	@python -m twine check --strict dist/*
