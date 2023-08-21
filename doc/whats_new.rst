:orphan:

.. _whats_new:

.. currentmodule:: mne_bids

What's new?
===========

.. _changes_0_13:

Version 0.13 (2023-08-21)
-------------------------

👩🏽‍💻 Authors
~~~~~~~~~~~~~~~

The following authors contributed for the first time. Thank you so much! 🤩

* `Ford McDonald`_
* `Jonathan Vanhoecke`_
* `Laetitia Fesselier`_
* `Pierre Guetschel`_

The following authors had contributed before. Thank you for sticking around! 🤘

* `Adam Li`_
* `Alexandre Gramfort`_
* `Eric Larson`_
* `Richard Höchenberger`_
* `Stefan Appelhoff`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

🚀 Enhancements
^^^^^^^^^^^^^^^

- :class:`~mne_bids.BIDSPath` now supports the new ``"sessions"`` suffix, by `Jonathan Vanhoecke`_ and `Richard Höchenberger`_ (:gh:`1137`)
- The :func:`~mne_bids.BIDSPath.rm` method will safely delete all the files compatible with that path and update the ``scans.tsv`` and ``participants.tsv`` files accordingly, by `Pierre Guetschel`_ and `Adam Li`_ (:gh:`1149`)

🧐 API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- When writing events, we now also create an ``*_events.json`` file in addition to ``*_events.tsv``. This ensures compatibility with the upcoming release of BIDS 1.9, by `Richard Höchenberger`_ (:gh:`1132`)
- We silenced warnings about missing ``events.tsv`` files when reading empty-room or resting-state data, by `Richard Höchenberger`_ (:gh:`1133`)
- BTi 'processed data files' (pdf) will no longer be renamed to ``c,rf*`` by default. Instead, they will be copied over without name changes. When reading BTi data via ``mne-bids``, more informative log and error messages will help choosing the right course of action, by `Stefan Appelhoff`_ (:gh:`1160`)

🛠 Requirements
^^^^^^^^^^^^^^^

- MNE-BIDS now requires MNE-Python 1.4 or newer.

🪲 Bug fixes
^^^^^^^^^^^^

- Amending a dataset now works in cases where the newly-written data contains additional participant properties (new columns in ``participants.tsv``) not found in the existing dataset, by `Richard Höchenberger`_ (:gh:`1113`)
- Fix ``raw_to_bids`` CLI tool to properly recognize boolean and numeric values for the ``line_freq`` and ``overwrite`` parameters, by `Stefan Appelhoff`_ (:gh:`1125`)
- Fix :func:`~mne_bids.copyfiles.copyfile_eeglab` to prevent data type conversion leading to an ``eeg_checkset`` failure when trying to load the file in EEGLAB, by `Laetitia Fesselier`_ (:gh:`1126`)
- Improve compatibility with latest MNE-Python, by `Eric Larson`_ (:gh:`1128`)
- Working with :class:`~mne_bids.BIDSPath` would sometimes inadvertently create new directories, contaminating the BIDS dataset, by `Richard Höchenberger`_ (:gh:`1139`)
- Fix thrown error if the ``BIDSVersion`` defined in ``dataset_description.json`` file does not match the MNE-BIDS compliant ``BIDSVersion``, ensuring backwards compatibility across BIDS complient tools, by `Ford McDonald`_ (:gh:`1147`)
- Copying BTi files without a headshape file will no longer raise an error, the file will simply be copied, and the missing headshape file will be ignored, by `Stefan Appelhoff`_ (:gh:`1158`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
