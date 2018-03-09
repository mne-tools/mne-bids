# simple makefile to simplify repetetive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
PYTESTS ?= py.test
CTAGS ?= ctags
CODESPELL_SKIPS ?= "*.fif,*.eve,*.gz,*.tgz,*.zip,*.mat,*.stc,*.label,*.w,*.bz2,*.annot,*.sulc,*.log,*.local-copy,*.orig_avg,*.inflated_avg,*.gii,*.pyc,*.doctree,*.pickle,*.inv,*.png,*.edf,*.touch,*.thickness,*.nofix,*.volume,*.defect_borders,*.mgh,lh.*,rh.*,COR-*,FreeSurferColorLUT.txt,*.examples,.xdebug_mris_calc,bad.segments,BadChannels,*.hist,empty_file,*.orig,*.js,*.map,*.ipynb,searchindex.dat"
CODESPELL_DIRS ?= mne_bids/ doc/ examples/
all: clean inplace test # test-doc

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

clean: clean-build clean-pyc clean-so clean-ctags clean-cache

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

test: in
	rm -f .coverage
	$(PYTESTS) mne_bids

test-doc:
	$(PYTESTS) --doctest-modules --doctest-ignore-import-errors --doctest-glob='*.rst' ./doc/

test-coverage: testing_data
	rm -rf coverage .coverage
	$(PYTESTS) --cov=mne_bids --cov-report html:coverage
# whats the difference with test-no-sample-with-coverage?

trailing-spaces:
	find . -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) -R *

upload-pipy:
	python setup.py sdist bdist_egg register upload

flake:
	@if command -v flake8 > /dev/null; then \
		echo "Running flake8"; \
		flake8 --count mne_bids examples; \
	else \
		echo "flake8 not found, please install it!"; \
		exit 1; \
	fi;
	@echo "flake8 passed"

codespell:  # running manually
	@codespell -w -i 3 -q 3 -S $(CODESPELL_SKIPS) -D ./dictionary.txt $(CODESPELL_DIRS)

codespell-error:  # running on travis
	@codespell -i 0 -q 7 -S $(CODESPELL_SKIPS) -D ./dictionary.txt $(CODESPELL_DIRS)

pydocstyle:
	@echo "Running pydocstyle"
	@pydocstyle

pep:
	@$(MAKE) -k flake pydocstyle docstring codespell-error

build-doc-dev:
	cd doc; make clean
	cd doc; DISPLAY=:1.0 xvfb-run -n 1 -s "-screen 0 1280x1024x24 -noreset -ac +extension GLX +render" make html_dev

build-doc-stable:
	cd doc; make clean
	cd doc; DISPLAY=:1.0 xvfb-run -n 1 -s "-screen 0 1280x1024x24 -noreset -ac +extension GLX +render" make html_stable

docstyle:
	@pydocstyle
