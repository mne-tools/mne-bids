:orphan:

.. _whats_new:

.. currentmodule:: mne_bids

.. include:: authors.rst

What's new?
===========

.. _changes_0_21:

Version 0.21 (unreleased)
-------------------------

👩🏽‍💻 Authors
~~~~~~~~~~~~~~~

The following authors contributed for the first time. Thank you so much! 🤩

* `Hamza Abdelhedi`_
* `Shubham Padkonde`_

The following authors had contributed before. Thank you for sticking around! 🤘

* `Bruno Aristimunha`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

🚀 Enhancements
^^^^^^^^^^^^^^^

- Expose MRI defacing function to public API as :func:`mne_bids.deface_mri` by `Erica Peterson`_ (:gh:`1544`)
- Add :func:`mne_bids.open_lock`, the cross-process file lock mne-bids uses for shared files such as ``participants.tsv``, so downstream writers can share it, by `Bruno Aristimunha`_ (:gh:`1675`)
- :func:`mne_bids.read_epochs_bids` now reads epoched FIF files (``RecordingType: "epoched"``) with :func:`mne.read_epochs`, and :func:`mne_bids.read_raw_bids` points to it for such files, by `Hamza Abdelhedi`_ (:gh:`1679`)

🧐 API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- None yet

🛠 Requirements
^^^^^^^^^^^^^^^

- None yet

🪲 Bug fixes
^^^^^^^^^^^^

- Fix :func:`find_matching_paths` omitting files with a ``tracksys`` entity and returning no matches when ``tracking_systems`` is specified, by `Shubham Padkonde`_.

⚕️ Code health
^^^^^^^^^^^^^^

- None yet

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`
