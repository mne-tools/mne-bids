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

- Sidecar files are now written to a temporary file and moved into place, so reading them no longer takes a file lock, and a sidecar whose content would not change is no longer rewritten, by `Hamza Abdelhedi`_ (:gh:`1680`)

🛠 Requirements
^^^^^^^^^^^^^^^

- None yet

🪲 Bug fixes
^^^^^^^^^^^^

- Fix readers seeing a partially written sidecar while another process was rewriting it, by `Hamza Abdelhedi`_ (:gh:`1680`)
- Fix parallel :func:`mne_bids.write_raw_bids` calls for BTi/4D data overwriting each other's ``.bidsignore`` changes, by `Hamza Abdelhedi`_ (:gh:`1680`)
- Fix :func:`find_matching_paths` omitting files with a ``tracksys`` entity and returning no matches when ``tracking_systems`` is specified, by `Shubham Padkonde`_.
- Allow ``acquisition="calibration"`` or ``"crosstalk"`` together with a ``task`` in :class:`BIDSPath` when ``datatype`` is not ``"meg"``, since these labels are reserved only for MEG fine-calibration and crosstalk files, by `Bruno Aristimunha`_ (:gh:`1673`)

⚕️ Code health
^^^^^^^^^^^^^^

- :func:`mne_bids.read_raw_bids` now parses ``participants.tsv`` once per version of the file instead of on every call, so reading every recording of a large dataset no longer gets slower as the number of subjects grows, by `Hamza Abdelhedi`_ (:gh:`1681`)
- Dropping rows from a TSV table no longer deep-copies the table first, by `Hamza Abdelhedi`_ (:gh:`1681`)
- Sped up :func:`mne_bids.read_raw_bids` and :func:`mne_bids.write_raw_bids` on network filesystems by taking fewer file locks, by `Hamza Abdelhedi`_ (:gh:`1680`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`
