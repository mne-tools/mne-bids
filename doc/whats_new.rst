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

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

Enhancements
^^^^^^^^^^^^

- Updated the "Read BIDS datasets" example to use data from `OpenNeuro <https://openneuro.org>`_, by `Alex Rockhill`_ (:gh:`753`)

API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^

- ...

Requirements
^^^^^^^^^^^^

- For downloading `OpenNeuro <https://openneuro.org>`_ datasets, ``openneuro-py`` is now required to run the examples and build the documentation, by `Alex Rockhill`_ (:gh:`753`)

Bug fixes
^^^^^^^^^

- :func:`mne_bids.make_report` now (1) detects male/female sex and left/right handedness irrespective of letter case, (2) will parse a ``gender`` column if no ``sex`` column is found in ``participants.tsv``, and (3) reports sex as male/female instead of man/woman, by `Alex Rockhill`_ (:gh:`755`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
