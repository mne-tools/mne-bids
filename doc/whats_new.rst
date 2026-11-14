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
* `Eric Larson`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

🚀 Enhancements
^^^^^^^^^^^^^^^

- Add :func:`mne_bids.read_participants` to read ``participants.tsv`` into rows keyed by subject label, also from the file's bytes for files that are not on a local disk, by `Bruno Aristimunha`_ (:gh:`1706`)
- Expose MRI defacing function to public API as :func:`mne_bids.deface_mri` by `Erica Peterson`_ (:gh:`1544`)
- Add an ``extra_fields`` parameter to :func:`make_dataset_description` to write additional top-level keys to ``dataset_description.json``, by `Bruno Aristimunha`_ (:gh:`1674`)
- Add :func:`mne_bids.open_lock`, the cross-process file lock mne-bids uses for shared files such as ``participants.tsv``, so downstream writers can share it, by `Bruno Aristimunha`_ (:gh:`1675`)
- :func:`mne_bids.read_epochs_bids` now reads epoched FIF files (``RecordingType: "epoched"``) with :func:`mne.read_epochs`, and :func:`mne_bids.read_raw_bids` points to it for such files, by `Hamza Abdelhedi`_ (:gh:`1679`)
- :func:`mne_bids.read_raw_bids` now accepts path-like inputs (``str`` or :class:`pathlib.Path`), converting them via :func:`mne_bids.get_bids_path_from_fname`, by `Bruno Aristimunha`_ (:gh:`1695`)
- Add ``fnames`` parameter to :func:`mne_bids.find_matching_paths` to filter pre-collected file paths without walking the filesystem, by `Bruno Aristimunha`_ (:gh:`1698`)
- Add :func:`mne_bids.find_inherited_sidecars` to choose, from a list of file paths, the sidecar files that apply to a recording under the BIDS inheritance principle, ordered from the dataset root down, without accessing the files, so it also works with URLs, by `Bruno Aristimunha`_ (:gh:`1708`)
- :class:`mne_bids.BIDSPath` and :func:`mne_bids.get_entities_from_fname` now know every entity, datatype and raw-data suffix of the BIDS schema (from ``bidsschematools``; for example ``echo``, ``reconstruction``, ``tracer``, and the ``func``, ``dwi``, ``fmap``, ``perf``, ``pet``, ``mrs`` and ``micr`` datatypes), so the names of such MRI, PET, MRS and microscopy files can be parsed, and built when their extension is one MNE-BIDS accepts (such as ``.nii.gz`` and ``.json``); with ``check=True``, extensions such as ``.bval``, ``.bvec``, ``.ome.tif`` and ``.ome.zarr`` are still rejected, by `Bruno Aristimunha`_ (:gh:`1715`)

🧐 API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :func:`mne_bids.get_entities_from_fname` with ``on_error="warn"`` now keeps an entity that is not in the BIDS schema, as ``"ignore"`` does, so ``BIDSPath(**get_entities_from_fname(fname, on_error="warn"))`` can raise a ``TypeError``, and :attr:`BIDSPath.entities` and :func:`mne_bids.get_entities_from_fname` now return all 35 BIDS entities, so the short keys ``rec`` and ``dir`` of a filename are returned as the ``reconstruction`` and ``direction`` entities, by `Bruno Aristimunha`_ (:gh:`1715`)
- Sidecar files are now written to a temporary file and moved into place, so reading them no longer takes a file lock, and a sidecar whose content would not change is no longer rewritten, by `Hamza Abdelhedi`_ (:gh:`1680`)

🛠 Requirements
^^^^^^^^^^^^^^^

- MNE-BIDS now requires ``bidsschematools`` (>=1.2), the official package of the BIDS schema, by `Bruno Aristimunha`_ (:gh:`1715`)

🪲 Bug fixes
^^^^^^^^^^^^

- :func:`mne_bids.read_raw_bids` no longer fails with an ``AttributeError`` when an ``events.json`` sidecar is not a JSON object (for example ``[]``), by `Bruno Aristimunha`_ (:gh:`1703`)
- Fix readers seeing a partially written sidecar while another process was rewriting it, by `Hamza Abdelhedi`_ (:gh:`1680`)
- Fix parallel :func:`mne_bids.write_raw_bids` calls for BTi/4D data overwriting each other's ``.bidsignore`` changes, by `Hamza Abdelhedi`_ (:gh:`1680`)
- :func:`find_matching_paths` and :meth:`BIDSPath.match` now follow symbolic links to directories, and no longer return hidden files (or files inside hidden directories), which could happen when a dataset was searched without ``datatypes`` and without ``ignore_nosub``, by `Hamza Abdelhedi`_ (:gh:`1678`)
- Fix :func:`find_matching_paths` omitting files with a ``tracksys`` entity and returning no matches when ``tracking_systems`` is specified, by `Shubham Padkonde`_.
- Fix :func:`find_matching_paths` and :meth:`BIDSPath.match` omitting every file with an entity they did not know, such as ``echo`` or ``rec`` in MRI filenames; :func:`find_matching_paths` matches these entities with its new ``entities`` parameter, by `Bruno Aristimunha`_ (:gh:`1715`)
- Allow ``acquisition="calibration"`` or ``"crosstalk"`` together with a ``task`` in :class:`BIDSPath` when ``datatype`` is not ``"meg"``, since these labels are reserved only for MEG fine-calibration and crosstalk files, by `Bruno Aristimunha`_ (:gh:`1673`)
- Fix :func:`find_matching_paths` and :meth:`BIDSPath.match` raising an error when the dataset contains a file that cannot be represented as a :class:`BIDSPath`, such as FreeSurfer output in ``derivatives``; such files are now skipped, by `Eric Larson`_ (:gh:`1699`)
- Fix :func:`mne_bids.get_entity_vals` with ``include_match`` returning no values when the dataset itself is inside a hidden directory, by `Eric Larson`_ (:gh:`1699`)
- Fix :func:`mne_bids.get_bids_path_from_fname` raising an ``UnboundLocalError`` instead of a clear error for a ``scans.tsv`` file name without subject or session entities, by `Eric Larson`_ (:gh:`1699`)
- :func:`mne_bids.read_raw_bids` no longer sets fNIRS channels whose ``channels.tsv`` type is ``NIRSCWHBO``, ``NIRSCWHBR`` or ``NIRSCWOPTICALDENSITY`` to ``misc``: they get the MNE types ``hbo``, ``hbr`` and ``fnirs_od``, and ``channels.tsv`` is written with these BIDS types for such channels instead of raising a ``KeyError``, by `Bruno Aristimunha`_ (:gh:`1701`)
- :func:`mne_bids.find_matching_paths` and :meth:`mne_bids.BIDSPath.match` now return MEF3 ``.mefd`` recordings, which are directories, as they do CTF ``.ds`` directories, by `Bruno Aristimunha`_ (:gh:`1707`)
- :func:`mne_bids.read_raw_bids` now matches fNIRS channels on the ``source``, ``detector`` and ``wavelength_nominal`` columns of ``channels.tsv`` when their names differ from the names MNE gives SNIRF channels (for example ``S1-D1-690.0`` in OpenNeuro ds004830), instead of raising a channel mismatch error, by `Bruno Aristimunha`_ (:gh:`1704`)

⚕️ Code health
^^^^^^^^^^^^^^

- :func:`mne_bids.read_raw_bids` now parses ``participants.tsv`` once per version of the file instead of on every call, so reading every recording of a large dataset no longer gets slower as the number of subjects grows, by `Hamza Abdelhedi`_ (:gh:`1681`)
- Dropping rows from a TSV table no longer deep-copies the table first, by `Hamza Abdelhedi`_ (:gh:`1681`)
- Sped up :func:`mne_bids.read_raw_bids` and :func:`mne_bids.write_raw_bids` on network filesystems by taking fewer file locks, by `Hamza Abdelhedi`_ (:gh:`1680`)
- Improvements to directory searching that speed up :meth:`BIDSPath.match`, :func:`find_matching_paths`, and :func:`read_raw_bids`, especially when searching network file systems, by `Hamza Abdelhedi`_ (:gh:`1678`)
- Creating and updating :class:`BIDSPath` objects is faster, which speeds up :meth:`BIDSPath.match` and :func:`find_matching_paths` on datasets with many files, by `Eric Larson`_ (:gh:`1699`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`
