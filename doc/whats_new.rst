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

- None yet

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

- Sped up :meth:`mne_bids.BIDSPath.match` and :func:`mne_bids.find_matching_paths` when searching from the dataset root for given datatypes: the tree is now walked once with :func:`os.scandir`, taking file types from the directory listing instead of one ``stat`` call per file, which matters most on network filesystems. The returned paths and their order are unchanged, by `Hamza Abdelhedi`_ (:gh:`XXXX`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`
