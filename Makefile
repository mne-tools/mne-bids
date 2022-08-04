.PHONY: all clean-pyc clean-so clean-build clean-ctags clean-cache clean-e inplace test check-manifest flake pydocstyle pep build-doc

all: clean inplace pep test build-doc

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
	python setup.py develop

test:
	@echo "Running tests"
	@python -m pytest . \
	--doctest-modules \
	--cov=mne_bids mne_bids/tests/ mne_bids/commands/tests/ \
	--cov-report=xml \
	--cov-config=setup.cfg \
	--verbose \
	--ignore mne-python \
	--ignore examples

check-manifest:
	@echo "Checking MANIFEST.in"
	@check-manifest .

flake:
	@echo "Running flake8"
	@flake8 --count mne_bids examples

pydocstyle:
	@echo "Running pydocstyle"
	@pydocstyle .

pep: flake pydocstyle check-manifest

build-doc:
	@echo "Building documentation"
	make -C doc/ clean
	make -C doc/ html
	cd doc/ && make view

dist-build:
	@echo "Building dist"
	rm -rf dist
	@python -m build
