:orphan:

.. _whats_new:

.. currentmodule:: mne_bids

.. include:: authors.rst

What's new?
===========

.. _changes_0_19:

Version 0.19 (unreleased)
-------------------------

👩🏽‍💻 Authors
~~~~~~~~~~~~~~~

The following authors contributed for the first time. Thank you so much! 🤩

* None yet

The following authors had contributed before. Thank you for sticking around! 🤘

* `Bruno Aristimunha`_


Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

🚀 Enhancements
^^^^^^^^^^^^^^^

- None yet

🧐 API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- None yet

🛠 Requirements
^^^^^^^^^^^^^^^

- None yet

🪲 Bug fixes
^^^^^^^^^^^^

- Fix :func:`mne_bids.BIDSPath.find_matching_sidecar` to search for sidecar files at the dataset root level per the BIDS inheritance principle, by `Bruno Aristimunha`_ (:gh:`1508`)
- Reinstate the requirement for ``coordsystem.json`` whenever ``electrodes.tsv`` is present (including EMG), by `Bruno Aristimunha`_ (:gh:`1508`)

⚕️ Code health
^^^^^^^^^^^^^^

- None yet

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`
