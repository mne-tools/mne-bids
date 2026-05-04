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
* `Pierre Guetschel`_
* `Alexandre Gramfort`_
* `Marijn van Vliet`_


Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

🚀 Enhancements
^^^^^^^^^^^^^^^

- Add support for reading and writing MEF3 (Multiscale Electrophysiology Format) iEEG data with the ``.mefd`` extension. Requires MNE-Python 1.12 or later, by `Bruno Aristimunha`_ (:gh:`1511`)
- Save ``Annotations.extras`` fields in events.tsv files when writing events, by `Pierre Guetschel`_ (:gh:`1502`)
- Added support for ``EEGLAB`` and ``EEGLAB-HJ`` coordinate systems as defined in the BIDS specification. Both use ALS orientation (identical to CTF) and map to MNE's ``ctf_head`` coordinate frame, by `Bruno Aristimunha`_ (:gh:`1514`)
- :func:`mne_bids.read_raw_bids` now reads channel units from ``channels.tsv`` and sets them on the raw object. This includes support for units like ``rad`` (radians), ``V``, ``µV``, ``mV``, ``T``, ``T/m``, ``S``, ``oC``, ``M``, and ``px``. The write path was also updated to correctly write ``rad`` units to ``channels.tsv``, by `Alexandre Gramfort`_ (:gh:`1509`)
- Added support for hashing ``BIDSPath`` objects so they can be used in caching and other contexts that require hashable objects, by `Eric Larson`_ (:gh:`1563`)
- Speed up :func:`mne_bids.get_datatypes` by restricting filesystem traversal to ``bids_root/sub-*/(ses-*/)<datatype>`` directories, by `Eric Larson`_ (:gh:`1563`)
- Speed up :meth:`mne_bids.BIDSPath.find_matching_sidecar` by searching most likely file locations first, by `Eric Larson`_ (:gh:`1565`)
- Add support for CHPI channels and gracefully handle incorrect channel definition ``MEGGRAD``, by `Eric Larson`_ (:gh:`1578`)

🧐 API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Expected format conversions notices are now logged at ``info`` instead of ``warn`` level, by `Bruno Aristimunha`_ (:gh:`1589`)
- Add ``readme`` parameter to :func:`mne_bids.write_raw_bids`; pass ``readme=False`` to leave any existing ``README`` untouched and skip creating one, by `Bruno Aristimunha`_ (:gh:`1550`)
- :func:`mne_bids.make_dataset_description` preserves BIDS-spec keys it does not model when merging with an existing ``dataset_description.json``, by `Bruno Aristimunha`_ (:gh:`1548`)

🛠 Requirements
^^^^^^^^^^^^^^^

- MEF3 (``.mefd``) file format support requires MNE-Python 1.12 or later, by `Bruno Aristimunha`_ (:gh:`1511`)

🪲 Bug fixes
^^^^^^^^^^^^

- Fix :func:`mne_bids.BIDSPath.find_matching_sidecar` to search for sidecar files at the dataset root level per the BIDS inheritance principle, by `Bruno Aristimunha`_ (:gh:`1508`)
- Reinstate the requirement for ``coordsystem.json`` whenever ``electrodes.tsv`` is present (including EMG), by `Bruno Aristimunha`_ (:gh:`1508`)
- Fix :func:`read_raw_bids` ignoring ``electrodes.tsv`` when ``EEGCoordinateUnits`` is ``"n/a"`` by inferring the unit from coordinate magnitudes, and synthesize approximate fiducials for ``ctf_head`` montages to enable the coordinate transform to ``head`` frame, by `Bruno Aristimunha`_ (:gh:`1506`)
- Improve :func:`mne_bids.read_raw_bids` handling when ``electrodes.tsv`` exists without ``coordsystem.json``: keep strict failure for iEEG, and for EEG/MEG emit a warning and continue without applying a montage, by `Bruno Aristimunha`_
- Allow ``task=None`` in :func:`mne_bids.read_raw_bids` for BIDS paths without a task entity (e.g. datasets that omit task in the path), by `Aman Jaiswal`_
- Fix :func:`mne_bids.read_raw_bids` and related read paths failing with ``PermissionError`` on datalad/git-annex datasets by keeping the file lock next to the symlink instead of its (read-only) target, and gracefully continuing without a lock when one cannot be created, by `Bruno Aristimunha`_ (:gh:`1569`)
- Avoid modifying calibration files by making :func:`mne_bids.write_meg_calibration` copy instead of parsing and rewriting, by `Marijn van Vliet`_ (:gh:`1576`)
- Fix bug with :meth:`mne_bids.BIDSPath.find_matching_sidecar` not searching parent directories properly, by `Eric Larson`_ (:gh:`1565`)
- Fix :func:`mne_bids.events_file_to_annotation_kwargs` to drop rows with invalid ``onset`` values (``n/a``, ``nan``, ``NaN``, empty string) before float conversion. This makes reading real-world OpenNeuro datasets (e.g. ``ds004841``, ``ds004842``, ``ds004843``) succeed instead of either raising or returning ``NaN`` onsets, by `Bruno Aristimunha`_ (:gh:`1547`)
- Fix :func:`mne_bids.events_file_to_annotation_kwargs` to fall back to the ``value`` column when ``trial_type`` is entirely ``n/a`` but ``value`` contains trigger codes, instead of dropping all events, by `Bruno Aristimunha`_ (:gh:`947`)
- :func:`mne_bids.read_raw_bids` now tolerates malformed ``scans.tsv`` entries and ISO 8601 ``acq_time`` variants, by `Bruno Aristimunha`_ (:gh:`1591`)
- :func:`mne_bids.read_raw_bids` now skips channel metadata with a warning when ``channels.tsv`` is empty or has no ``name`` column, and deduplicates duplicate channel names with ``-0/-1/...`` suffixes, by `Bruno Aristimunha`_

⚕️ Code health
^^^^^^^^^^^^^^

- None yet

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`
