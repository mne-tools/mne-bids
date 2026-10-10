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

- None yet

🛠 Requirements
^^^^^^^^^^^^^^^

- None yet

🪲 Bug fixes
^^^^^^^^^^^^

- :func:`find_matching_paths` and :meth:`BIDSPath.match` now follow symbolic links to directories, and no longer return hidden files (or files inside hidden directories), which could happen when a dataset was searched without ``datatypes`` and without ``ignore_nosub``, by `Hamza Abdelhedi`_ (:gh:`1678`)
- Fix :func:`find_matching_paths` omitting files with a ``tracksys`` entity and returning no matches when ``tracking_systems`` is specified, by `Shubham Padkonde`_.

⚕️ Code health
^^^^^^^^^^^^^^

- :func:`mne_bids.read_raw_bids` now parses ``participants.tsv`` once per version of the file instead of on every call, so reading every recording of a large dataset no longer gets slower as the number of subjects grows, by `Hamza Abdelhedi`_ (:gh:`1681`)
- Dropping rows from a TSV table no longer deep-copies the table first, by `Hamza Abdelhedi`_ (:gh:`1681`)
- Improvements to directory searching that speed up :meth:`BIDSPath.match`, :func:`find_matching_paths`, and :func:`read_raw_bids`, especially when searching network file systems, by `Hamza Abdelhedi`_ (:gh:`1678`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`
