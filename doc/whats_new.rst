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

🧐 API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Reading sidecar files no longer takes a file lock: files written by MNE-BIDS are now replaced atomically (written to a temporary file next to the target, then moved into place), so a reader always sees either the old or the new file. A plain overwrite no longer takes a lock either, and a sidecar whose content would not change is no longer rewritten, so its modification time stays as it was, by `Hamza Abdelhedi`_ (:gh:`1680`)

🛠 Requirements
^^^^^^^^^^^^^^^

- None yet

🪲 Bug fixes
^^^^^^^^^^^^

- Fix a reader being able to see an empty or partially written sidecar file while another process was rewriting it. This happened when file locking was not active (``filelock`` missing or older than the required version); sidecars are now replaced atomically, by `Hamza Abdelhedi`_ (:gh:`1680`)
- Fix :func:`mne_bids.write_raw_bids` updating ``.bidsignore`` without a lock when writing BTi/4D data, so that writers running in parallel could overwrite each other's changes, by `Hamza Abdelhedi`_ (:gh:`1680`)
- Fix :func:`find_matching_paths` omitting files with a ``tracksys`` entity and returning no matches when ``tracking_systems`` is specified, by `Shubham Padkonde`_.

⚕️ Code health
^^^^^^^^^^^^^^

- Sped up reading on network filesystems, where taking a lock for every sidecar was a large part of :func:`mne_bids.read_raw_bids`, and stopped the lock warnings on read-only datasets. :func:`mne_bids.write_raw_bids` is faster as well, alone and in parallel, because sidecars that are already up to date are neither locked nor rewritten, by `Hamza Abdelhedi`_ (:gh:`1680`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`
