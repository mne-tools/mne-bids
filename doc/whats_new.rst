:orphan:

.. _whats_new:

.. currentmodule:: mne_bids

What's new?
===========

.. _changes_0_17:

Version 0.17 (unreleased)
-------------------------

👩🏽‍💻 Authors
~~~~~~~~~~~~~~~

The following authors contributed for the first time. Thank you so much! 🤩

* `Christian O'Reilly`_
* `Berk Gerçek`_
* `Arne Gottwald`_
* `Matthias Dold`_

The following authors had contributed before. Thank you for sticking around! 🤘

* `Stefan Appelhoff`_
* `Daniel McCloy`_
* `Scott Huberty`_
* `Pierre Guetschel`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

🚀 Enhancements
^^^^^^^^^^^^^^^

- :func:`mne_bids.write_raw_bids()` can now handle mne `Raw` objects with `eyegaze` and `pupil` channels, by `Christian O'Reilly`_ (:gh:`1344`)
- :func:`mne_bids.get_entity_vals()` has a new parameter ``ignore_suffixes`` to easily ignore sidecar files, by `Daniel McCloy`_ (:gh:`1362`)
- Empty-room matching now preferentially finds recordings in the subject directory tagged as `task-noise` before looking in the `sub-emptyroom` directories. This adds support for a part of the BIDS specification for ER recordings, by `Berk Gerçek`_ (:gh:`1364`)
- Path matching is now implemenented in a more efficient manner within :meth:`mne_bids.BIDSPath.match()` and :func:`mne_bids.find_matching_paths()`, by `Arne Gottwald` (:gh:`1355`)
- :func:`mne_bids.get_entity_vals()` has a new parameter ``include_match`` to prefilter item matching and ignore non-matched items from begin of directory scan, by `Arne Gottwald` (:gh:`1355`)
- `_handle_events_reading()` was refactored to now exposed the filtering of the `events.tsv` data as a separate function :func:`mne_bids.events_file_to_annotation_kwargs`, by `Matthias Dold` (:gh:`1389`)


🧐 API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :func:`mne_bids.make_dataset_description` will now auto-generate basic ``GeneratedBy`` fields if ``generated_by=None``. To suppress the auto-generated fields, pass an empty list. By `Daniel McCloy`_ (:gh:`1384`)

🛠 Requirements
^^^^^^^^^^^^^^^

- MNE-BIDS now requires ``mne`` 1.8 or higher.

🪲 Bug fixes
^^^^^^^^^^^^

- :func:`mne_bids.read_raw_bids` can optionally return an ``event_id`` dictionary suitable for use with :func:`mne.events_from_annotations`, and if a ``values`` column is present in ``events.tsv`` it will be used as the source of the integer event ID codes, by `Daniel McCloy`_ (:gh:`1349`)
- BIDS dictates that the recording entity should be displayed as "_recording-" in the filename. This PR makes :class:`mne_bids.BIDSPath`  correctly display "_recording-" (instead of "_rec-") in BIDSPath.fpath. By `Scott Huberty`_ (:gh:`1348`)
- :func:`mne_bids.make_dataset_description` now correctly encodes the dataset description as UTF-8 on disk, by `Scott Huberty`_ (:gh:`1357`)
- Corrects extension when filtering filenames in :meth:`mne_bids.BIDSPath.match()` and :func:`mne_bids.find_matching_paths()`, by `Arne Gottwald` (:gh:`1355`)
- Fix :class:`mne_bids.BIDSPath` partially matching a value, by `Pierre Guetschel` (:gh:`1388`)

⚕️ Code health
^^^^^^^^^^^^^^

- Tests that were adding or deleting files to/from a session-scoped dataset now properly clean up after themselves, by `Daniel McCloy`_ (:gh:`1347`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
