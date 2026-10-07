# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## What this is

MNE-BIDS reads and writes MEG, EEG, iEEG, EMG, NIRS, and related data (plus anatomical
MRIs) according to the [BIDS specification](https://bids-specification.readthedocs.io):
`BIDSPath` for building and matching filenames, `write_raw_bids` and friends for
converting to BIDS, `read_raw_bids` for reading back into MNE-Python objects, plus
sidecar updating, anonymization, reports, and the `mne_bids` command-line tools.
MNE-Python owns the data containers (`Raw`, `Epochs`, `Info`) and the file readers; this
package maps them to and from BIDS.

## Follow MNE-Python's conventions

This package is a subsidiary of MNE-Python. Unless something below says otherwise, follow
[MNE-Python's AGENTS.md](https://github.com/mne-tools/mne-python/blob/main/AGENTS.md)
(read it rather than guessing): keep changes small, naming, numpydoc style, imports,
deprecation policy, compact tests, license rules for adapted code, and in particular its
[policy on AI assistance](https://github.com/mne-tools/mne-python/blob/main/CONTRIBUTING.md#policy-on-ai-assistance-in-contributions):

- Work test-first: write (or extend) a test that fails for the right reason, then make it
  pass. Promote anything a throwaway script caught into a real test.
- Do not open pull requests, push, or commit unless explicitly asked; the human submitting
  the change must review, understand, and disclose AI use in the PR description.
- Keep changes minimal and scoped to the request; mention, don't silently fix, unrelated
  problems you notice.

What does *not* carry over from MNE-Python:

- There are no towncrier fragments. The changelog is `doc/whats_new.rst`: add a bullet
  under the right heading of the unreleased version, in the existing format (ending in
  ``by `Name`_.``), and add first-time contributors to `doc/authors.rst`, the authors list
  at the top of that version, and `CITATION.cff` (before Alexandre Gramfort and Mainak
  Jas). See `CONTRIBUTING.md`.
- There are no lazy `__init__.pyi` stubs: `mne_bids/__init__.py` imports its public names
  directly, and new public API must also be added to `doc/api.rst`.
- There is no `docdict`. Functions taking `verbose` use `@verbose` from `mne.utils`, which
  `mne_bids/tests/test_verbose.py` enforces.
- PR titles start with `[MRG]` once ready for review (see the PR template), and the
  default branch is `main`.

## Things to know

- Correctness here means "valid BIDS", not just "round-trips through MNE-BIDS". Check
  behavior against the BIDS specification version in `mne_bids/config.py`
  (`BIDS_VERSION`), not against what the code currently does, and run the validator
  (below) on anything a change writes.
- `mne_bids/config.py` holds the tables that drive most behavior (allowed datatypes,
  entities and their order, suffixes, extensions, conversion formats, coordinate frames,
  unit maps). Adding support for something new usually starts there; grep for how an
  existing entry is used before adding one.
- `BIDSPath` (`mne_bids/path.py`) is used everywhere, and its matching and entity
  parsing are subtle; changes there need tests covering the entity and datatype
  combinations they affect.
- Supported MNE-Python versions are the current and previous stable releases (minimum in
  `pyproject.toml`), and CI also tests MNE-Python `main`. Code that needs a newer
  MNE-Python gates on `mne.utils.check_version` (grep for existing examples). Private
  MNE-Python helpers (`_validate_type`, `_check_option`, ...) are used freely, so they can
  break when MNE-Python's `main` changes.
- Optional dependencies (pandas, nibabel, pybv, edfio, eeglabio, ...) are in the `full`
  extra and must be imported lazily inside the functions that use them.

## Tests and docs

`make test` runs the unit tests (`pytest mne_bids` with pytest-xdist; `make test JOBS=0
ARGS="--pdb"` to debug in one process). Most tests need the MNE testing dataset
(`python -c 'import mne; mne.datasets.testing.data_path(verbose=True)'`) and are
decorated with `@testing.requires_testing_data`; CI also runs the suite without it, so
new tests that use it must be decorated too. Tests that take the `_bids_validate` fixture
run the BIDS validator, which needs `deno` (or `bids-validator-deno` from PyPI) on the
`PATH`; `BIDS_VALIDATOR_VERSION` selects `stable`, `dev`, or a specific version. Warnings
are errors (`filterwarnings` in `pyproject.toml`): catch warnings a test expects with
`pytest.warns`, and add a narrowly scoped `ignore` line only for third-party noise.

The gallery examples in `examples/` are not unit tests: they run when the docs (Sphinx +
sphinx-gallery) are built on CircleCI with `make build-doc` (or `make -C doc html-noplot`
locally for a quick check without running them). Run `pre-commit run --all-files` (ruff,
ruff-format, toml-sort, and others) before handing work back.
