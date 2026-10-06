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
- Write ``MISCChannelCount`` in ``*_eeg.json`` sidecars per the BIDS specification (previously written as ``MiscChannelCount``). The legacy misspelled key is kept alongside the spec-conformant key for one deprecation cycle, by `Bruno Aristimunha`_.

⚕️ Code health
^^^^^^^^^^^^^^

- None yet

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`
