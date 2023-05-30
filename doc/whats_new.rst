:orphan:

.. _whats_new:

.. currentmodule:: mne_bids

What's new?
===========

.. _changes_0_13:

Version 0.13 (unreleased)
-------------------------

...

📝 Notable changes
~~~~~~~~~~~~~~~~~~

- ...

👩🏽‍💻 Authors
~~~~~~~~~~~~~~~

The following authors contributed for the first time. Thank you so much! 🤩

* `Laetitia Fesselier`_
* `Jonathan Vanhoecke`_

The following authors had contributed before. Thank you for sticking around! 🤘

* `Richard Höchenberger`_
* `Eric Larson`_
* `Stefan Appelhoff`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

🚀 Enhancements
^^^^^^^^^^^^^^^

- :class:`~mne_bids.BIDSPath` now supports the new ``"sessions"`` suffix, by `Jonathan Vanhoecke`_ and `Richard Höchenberger`_ (:gh:`1137`)

🧐 API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- When writing events, we now also create an ``*_events.json`` file in addition to ``*_events.tsv``. This ensures compatibility with the upcoming release of BIDS 1.9, by `Richard Höchenberger`_ (:gh:`1132`)
- We silenced warnings about missing ``events.tsv`` files when reading empty-room or resting-state data, by `Richard Höchenberger`_ (:gh:`1133`)

🛠 Requirements
^^^^^^^^^^^^^^^

- MNE-BIDS now requires MNE-Python 1.3 or newer.

🪲 Bug fixes
^^^^^^^^^^^^

- Amending a dataset now works in cases where the newly-written data contains additional participant properties (new columns in ``participants.tsv``) not found in the existing dataset, by `Richard Höchenberger`_ (:gh:`1113`)
- Fix ``raw_to_bids`` CLI tool to properly recognize boolean and numeric values for the ``line_freq`` and ``overwrite`` parameters, by `Stefan Appelhoff`_ (:gh:`1125`)
- Fix :func:`~mne_bids.copyfiles.copyfile_eeglab` to prevent data type conversion leading to an ``eeg_checkset`` failure when trying to load the file in EEGLAB, by `Laetitia Fesselier`_ (:gh:`1126`)
- Improve compatibility with latest MNE-Python, by `Eric Larson`_ (:gh:`1128`)
- Working with :class:`~mne_bids.BIDSPath` would sometimes inadvertently create new directories, contaminating the BIDS dataset, by `Richard Höchenberger`_ (:gh:`1139`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
