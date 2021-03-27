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

- ...

API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^

- ...

Requirements
^^^^^^^^^^^^

- ...

Bug fixes
^^^^^^^^^

- :func:`mne_bids.make_report` now (1) detects male/female sex and left/right handedness irrespective of letter case, (2) will parse a ``gender`` column if no ``sex`` column is found in ``participants.tsv``, and (3) reports sex as male/female instead of man/woman, by `Alex Rockhill`_ (:gh:`755`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
