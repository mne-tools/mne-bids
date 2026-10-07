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

* None yet

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

🚀 Enhancements
^^^^^^^^^^^^^^^

- Expose MRI defacing function to public API as :func:`mne_bids.deface_mri` by `Erica Peterson`_ (:gh:`1544`)

🧐 API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- None yet

🛠 Requirements
^^^^^^^^^^^^^^^

- None yet

🪲 Bug fixes
^^^^^^^^^^^^

- Fix :func:`find_matching_paths` and :meth:`BIDSPath.match` returning hidden files, and files inside hidden directories such as ``.git``, when a whole dataset is searched without ``datatypes`` and without ``ignore_nosub``; the other searches already left them out, by `Hamza Abdelhedi`_ (:gh:`XXXX`)
- Fix :func:`find_matching_paths` omitting files with a ``tracksys`` entity and returning no matches when ``tracking_systems`` is specified, by `Shubham Padkonde`_.

⚕️ Code health
^^^^^^^^^^^^^^

- Sped up :meth:`mne_bids.BIDSPath.match` and :func:`mne_bids.find_matching_paths` when searching from the dataset root for given datatypes: the tree is now walked once with :func:`os.scandir`, taking file types from the directory listing instead of one ``stat`` call per file, which matters most on network filesystems. The returned paths and their order are unchanged, by `Hamza Abdelhedi`_ (:gh:`XXXX`)
- :func:`mne_bids.read_raw_bids` now lists each directory once when it looks up the sidecar files of a recording, instead of asking the filesystem about every candidate file, which saves calls on network filesystems, by `Hamza Abdelhedi`_ (:gh:`XXXX`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`
