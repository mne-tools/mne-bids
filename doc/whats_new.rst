:orphan:

.. _whats_new:

.. currentmodule:: mne_bids

What's new?
===========

.. _changes_0_15:

Version 0.15 (unreleased)
-------------------------

👩🏽‍💻 Authors
~~~~~~~~~~~~~~~

The following authors contributed for the first time. Thank you so much! 🤩

* `Daniel McCloy`_

The following authors had contributed before. Thank you for sticking around! 🤘

* `Laetitia Fesselier`_
* `Richard Höchenberger`_
* `Stefan Appelhoff`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

🚀 Enhancements
^^^^^^^^^^^^^^^

- nothing yet

🧐 API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- The experimental support for running MNE-BIDS examples from your browser using Binder has
  been removed, by `Stefan Appelhoff`_ (:gh:`1202`)

🛠 Requirements
^^^^^^^^^^^^^^^

- MNE-BIDS now requires Python 3.9 or higher.
- MNE-BIDS now requires MNE-Python 1.5.0 or higher.
- ``edfio`` replaces ``EDFlib-Python`` for export to EDF with MNE-Python >= 1.7.0.
- Installing ``mne-bids[full]`` will now also install ``defusedxml`` on all platforms.
- Version requirements for optional dependency packages have been bumped up, see installation instructions.

🪲 Bug fixes
^^^^^^^^^^^^

- nothing yet

⚕️ Code health
^^^^^^^^^^^^^^

- The configuration of MNE-BIDS has been consolidated from several files (e.g., ``setup.cfg``,
  ``setup.py``, ``requirements.txt``) and is now specified in a standard ``pyproject.toml``
  file, by `Stefan Appelhoff`_ (:gh:`1202`)
- Linting and code formatting is now done entirely using ``ruff``. Previously used tools
  (e.g., ``flake8``, ``black``) have been fully replaced, by `Stefan Appelhoff`_ (:gh:`1203`)
- The package build backend has been switched from ``setuptools`` to ``hatchling``. This
  only affects users who build and install MNE-BIDS from source, and should not lead to
  changed runtime behavior, by `Richard Höchenberger`_ (:gh:`1204`)
- Display of the version number on the website is now truncated for over-long version strings,
  by `Daniel McCloy`_ (:gh:`1206`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
