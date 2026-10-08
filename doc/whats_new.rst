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

* `Shubham Padkonde`_

The following authors had contributed before. Thank you for sticking around! 🤘

* `Bruno Aristimunha`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

🚀 Enhancements
^^^^^^^^^^^^^^^

- Expose MRI defacing function to public API as :func:`mne_bids.deface_mri` by `Erica Peterson`_ (:gh:`1544`)
- Add an ``extra_fields`` parameter to :func:`make_dataset_description` to write additional top-level keys to ``dataset_description.json``, by `Bruno Aristimunha`_ (:gh:`1674`)

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
