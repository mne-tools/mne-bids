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

* nobody yet

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

- nothing yet

🛠 Requirements
^^^^^^^^^^^^^^^

- MNE-BIDS now requires MNE-Python 1.6.0 or higher.

🪲 Bug fixes
^^^^^^^^^^^^

- When anonymizing the date of a recording, MNE-BIDS will no longer error during `~mne_bids.write_raw_bids` if passing a `~mne.io.Raw` instance to ``empty_room``, by `Daniel McCloy`_ (:gh:`1270`)

⚕️ Code health
^^^^^^^^^^^^^^

- Keep MNE-BIDS up to date with recent changes on participant birthday date handling in MNE-Python, by `Eric Larson`_ (gh:1278:)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
