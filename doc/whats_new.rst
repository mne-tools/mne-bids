:orphan:

.. _whats_new:

.. currentmodule:: mne_bids

What's new?
===========

.. _changes_0_14:

Version 0.14 (2023-11-16)
-------------------------

👩🏽‍💻 Authors
~~~~~~~~~~~~~~~

The following authors contributed:

* `Eric Larson`_
* `Laetitia Fesselier`_
* `Mathieu Scheltienne`_
* `Richard Höchenberger`_
* `Stefan Appelhoff`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

🚀 Enhancements
^^^^^^^^^^^^^^^

- Enable exporting to the EEGLAB data format (``.set``), by `Laetitia Fesselier`_ and `Stefan Appelhoff`_ (:gh:`1187`)

🪲 Bug fixes
^^^^^^^^^^^^

- Fix reading when the channel order differs between ``*_channels.tsv`` and the raw data file, which would previously throw an error, by `Richard Höchenberger`_ (:gh:`1171`)
- Fix bug with writing crosstalk and calibration files when subject is ``"emptyroom"``, by `Eric Larson`_ (:gh:`1189`)
- Make ``recording`` entity available for :func:`mne_bids.get_entity_vals`, by `Stefan Appelhoff`_ (:gh:`1182`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
