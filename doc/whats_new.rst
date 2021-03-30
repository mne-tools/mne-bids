:orphan:

.. _whats_new:

.. currentmodule:: mne_bids

What's new?
===========

.. currentmodule:: mne_bids
.. _changes_0_8:

Version 0.8 (unreleased)
------------------------

...

Notable changes
~~~~~~~~~~~~~~~

- ...

Authors
~~~~~~~

* `Alex Rockhill`_
* `Richard Höchenberger`_
* `Adam Li`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

Enhancements
^^^^^^^^^^^^

- Updated the "Read BIDS datasets" example to use data from `OpenNeuro <https://openneuro.org>`_, by `Alex Rockhill`_ (:gh:`753`)

API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^

- Writing datasets via :func:`write_raw_bids`, will now never overwrite ``dataset_description.json`` file, by `Adam Li`_ (:gh:`765`)

Requirements
^^^^^^^^^^^^

- For downloading `OpenNeuro <https://openneuro.org>`_ datasets, ``openneuro-py`` is now required to run the examples and build the documentation, by `Alex Rockhill`_ (:gh:`753`)

Bug fixes
^^^^^^^^^

- :func:`mne_bids.make_report` now (1) detects male/female sex and left/right handedness irrespective of letter case, (2) will parse a ``gender`` column if no ``sex`` column is found in ``participants.tsv``, and (3) reports sex as male/female instead of man/woman, by `Alex Rockhill`_ (:gh:`755`)
- The :class:`mne.Annotations` ``BAD_ACQ_SKIP`` – added by the acquisition system to ``FIFF`` files – will now be preserved when reading raw data, even if these time periods are **not** explicitly included in ``*_events.tsv``, by `Richard Höchenberger`_ (:gh:`xxx`)
- :func:`mne_bids.write_raw_bids` will handle different cased extensions for EDF files, such as `.edf` and `.EDF` by `Adam Li`_ (:gh: `765`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
