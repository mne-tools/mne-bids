:orphan:

.. _whats_new:

.. currentmodule:: mne_bids

What's new?
===========

.. _changes_0_16:

Version 0.16 (unreleased)
-------------------------

👩🏽‍💻 Authors
~~~~~~~~~~~~~~~

The following authors contributed for the first time. Thank you so much! 🤩

* `Amaia Benitez`_
* `Daniel McCloy`_
* `Mara Wolter`_


The following authors had contributed before. Thank you for sticking around! 🤘

* `Daniel McCloy`_
* `Eric Larson`_
* `Stefan Appelhoff`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

🚀 Enhancements
^^^^^^^^^^^^^^^

- nothing yet

🧐 API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :func:`mne_bids.read_raw_bids` no longer warns about unit changes in channels upon reading, as that information is taken from ``channels.tsv`` and judged authorative, by `Stefan Appelhoff`_ (:gh:`1282`)
- MEG OPM channels are now experimentally included, by `Amaia Benitez`_ (:gh:`1222`)

🛠 Requirements
^^^^^^^^^^^^^^^

- MNE-BIDS now requires MNE-Python 1.6.0 or higher.

🪲 Bug fixes
^^^^^^^^^^^^

- The datatype in the dataframe returned by :func:`mne_bids.stats.count_events` is now
  ``pandas.Int64Dtype`` instead of ``float64``, by `Eric Larson`_ (:gh:`1227`)
- The :func:`mne_bids.copyfiles.copyfile_ctf` now accounts for files with ``.{integer}_meg4`` extension, instead of only .meg4,
  when renaming the files of a .ds folder, by `Mara Wolter`_ (:gh:`1230`)
- We fixed handling of time zones when reading ``*_scans.tsv`` files; specifically, non-UTC timestamps are now processed correctly,
  by `Stefan Appelhoff`_ and `Richard Höchenberger`_  (:gh:`1240`)
- When anonymizing the date of a recording, MNE-BIDS will no longer error during `~mne_bids.write_raw_bids` if passing a `~mne.io.Raw` instance to ``empty_room``, by `Daniel McCloy`_ (:gh:`1270`)

⚕️ Code health
^^^^^^^^^^^^^^

- Keep MNE-BIDS up to date with recent changes on participant birthday date handling in MNE-Python, by `Eric Larson`_ (gh:1278:)
- Make rules for linting more strict, make quality assessment exceptions less permissive, by `Stefan Appelhoff`_ (gh:1283:)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
