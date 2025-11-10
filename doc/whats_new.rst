:orphan:

.. _whats_new:

.. currentmodule:: mne_bids

.. include:: authors.rst

What's new?
===========

.. _changes_0_18:

Version 0.18 (unreleased)
-------------------------

👩🏽‍💻 Authors
~~~~~~~~~~~~~~~

The following authors contributed for the first time. Thank you so much! 🤩

* `Julius Welzel`_
* `Alex Lopez Marquez`_
* `Bruno Aristimunha`_
* `Kalle Mäkelä`_

The following authors had contributed before. Thank you for sticking around! 🤘

* `Stefan Appelhoff`_


Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

🚀 Enhancements
^^^^^^^^^^^^^^^

- :func:`mne_bids.write_raw_bids()` has a new parameter `electrodes_tsv_task` which allows adding the `task` entity to the `electrodes.tsv` filepath, by `Alex Lopez Marquez`_ (:gh:`1424`)
- Extended the configuration to recognise `motion` as a valid BIDS datatype by `Julius Welzel`_ (:gh:`1430`)
- Better control of verbosity in several functions, by `Bruno Aristimunha`_ (:gh:`1449`)
- Added parallel reading and writing in all the `open` operations using `_open_lock` mechanism from mne, creating a small wrapper to manage the locks by `Bruno Aristimunha`_ (:gh:`1451`)
- :meth:`mne_bids.BIDSPath.match()` now short-circuits root directory scans when ``subject``, ``session``, or ``datatype`` entities are known, reducing lookup time on large datasets, by `Bruno Aristimunha`_ and `Maximilien Chaumon`_ (:gh:`1450`)

🧐 API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  `tracksys` accepted as argument in :class:`mne_bids.BIDSPath()` by `Julius Welzel`_ (:gh:`1430`)
- :func:`mne_bids.read_raw_bids()` has a new parameter ``on_ch_mismatch`` that controls behaviour when there is a mismatch between channel names in ``channels.tsv`` and the raw data; accepted values are ``'raise'`` (default), ``'reorder'``, and ``'rename'``, by `Kalle Mäkelä`_.
- Column names read from ``.tsv`` are now converted from `numpy.str_` class to built-in `str` class, by `Daniel McCloy`_.

🛠 Requirements
^^^^^^^^^^^^^^^
- Including ``filelock`` as a dependency to handle atomic file writing and parallel processing support by `Bruno Aristimunha`_ (:gh:`1451`)

- None yet

🪲 Bug fixes
^^^^^^^^^^^^

- Fixed a bug that modified the name and help message of some of the available commands, by `Alex Lopez Marquez`_ (:gh:`1441`)
- Updated MEG/iEEG writers to satisfy the stricter checks in the latest BIDS validator releases: BTi/4D run folders now retain their ``.pdf`` suffix (falling back to the legacy naming when an older validator is detected), KIT marker files encode the run via the ``acq`` entity instead of ``run``, datasets lacking iEEG montages receive placeholder ``electrodes.tsv``/``coordsystem.json`` files, and the ``AssociatedEmptyRoom`` entry stores dataset-relative paths  by `Bruno Aristimunha`_ (:gh:`1449`)
- Made the lock helpers skip reference counting when the optional ``filelock`` dependency is missing, preventing spurious ``AttributeError`` crashes during reads, by `Bruno Aristimunha`_ (:gh:`1469`)
- Fixed a bug in :func:`mne_bids.read_raw_bids` that caused it to fail when reading BIDS datasets where the acquisition time was specified in local time rather than UTC only in Windows, by `Bruno Aristimunha`_ (:gh:`1452`)

⚕️ Code health
^^^^^^^^^^^^^^

- Made :func:`mne_bids.copyfiles.copyfile_brainvision` output more meaningful error messages when encountering problematic files, by `Stefan Appelhoff`_ (:gh:`1444`)
- Raised the minimum ``edfio`` requirement to ``0.4.10``,  eeglabio to ``0.1.0``  by `Bruno Aristimunha`_ (:gh:`1449`)
- Relaxed EDF padding warnings in the test suite to accommodate upstream changes by `Bruno Aristimunha`_ (:gh:`1449`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`
