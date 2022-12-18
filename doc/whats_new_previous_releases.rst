:orphan:

.. _whats_new_in_previous_releases:

.. currentmodule:: mne_bids

What was new in previous releases?
==================================

.. _changes_0_12:

Version 0.12 (2022-12-18)
-------------------------

This release includes a number of bug fixes as well as several smaller enhancements.
Please note some updated requirements, as listed in the details below.

📝 Notable changes
~~~~~~~~~~~~~~~~~~

- Nothing out of the ordinary, see below.

👩🏽‍💻 Authors
~~~~~~~~~~~~~~~

The following authors contributed for the first time. Thank you so much! 🤩

* `Moritz Gerster`_

The following authors had contributed before. Thank you for sticking around! 🤘

* `Alexandre Gramfort`_
* `Eric Larson`_
* `Richard Höchenberger`_
* `Stefan Appelhoff`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

🚀 Enhancements
^^^^^^^^^^^^^^^

- Speed up :func:`mne_bids.read_raw_bids` when lots of events are present by `Alexandre Gramfort`_ (:gh:`1079`)
- Add :meth:`mne_bids.BIDSPath.get_empty_room_candidates` to get the candidate empty-room files that could be used by :meth:`mne_bids.BIDSPath.find_empty_room` by `Eric Larson`_ (:gh:`1083`, :gh:`1093`)
- Add :meth:`mne_bids.BIDSPath.find_matching_sidecar` to find the sidecar file associated with a given file path by `Eric Larson`_ (:gh:`1093`)
- When writing data via :func:`~mne_bids.write_raw_bids`, it is now possible to specify a custom mapping of :class:`mne.Annotations` descriptions to event codes via the ``event_id`` parameter. Previously, passing this parameter would always require to also pass ``events``, and using a custom event code mapping for annotations was impossible, by `Richard Höchenberger`_ (:gh:`1084`)
- Improve error message when :obj:`~mne_bids.BIDSPath.fpath` cannot be uniquely resolved by `Eric Larson`_ (:gh:`1097`)
- Add :func:`mne_bids.find_matching_paths` to retrieve all `BIDSPaths` matching user-specified entities. The functionality partially overlaps with what's offered through :meth:`mne_bids.BIDSPath.match()`, but is more versatile, by `Moritz Gerster`_ (:gh:`1103`)

🧐 API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- nothing new!

🛠 Requirements
^^^^^^^^^^^^^^^

- MNE-BIDS now requires Python 3.8 or higher, because Python 3.7 is soon reaching its end of life.
- MNE-BIDS now requires MNE-Python 1.2.0 or higher.

🪲 Bug fixes
^^^^^^^^^^^^

- When writing data containing :class:`mne.Annotations` **and** passing events to :func:`~mne_bids.write_raw_bids`, previously, annotations whose description did not appear in ``event_id`` were silently dropped. We now raise an exception and request users to specify mappings between descriptions and event codes in this case. It is still possible to omit ``event_id`` if no ``events`` are passed, by `Richard Höchenberger`_ (:gh:`1084`)
- When working with NIRS data, raise the correct error message when a faulty ``format`` argument is passed to :func:`~mne_bids.write_raw_bids`, by `Stefan Appelhoff`_ (:gh:`1092`)
- Fixed writing data preloaded from a format that's not supported by BIDS, by `Richard Höchenberger`_ (:gh:`1101`)
- Fixed writing data from neuromag system having 122 sensors, by `Alexandre Gramfort`_ (:gh:`1109`)

.. _changes_0_11:

Version 0.11 (2022-10-08)
-------------------------

This release includes a number of bug fixes as well as several smaller enhancements.
Please note some behavior changes and updated requirements, as listed in the details below.

📝 Notable changes
~~~~~~~~~~~~~~~~~~

- Support for new channel types is available: temperature and galvanic skin response

- MNE-BIDS now supports the BIDS "description" entity (``desc``)

- It's now possible to store Curry (CDT) files and EGI files with MNE-BIDS.


👩🏽‍💻 Authors
~~~~~~~~~~~~~~~

The following authors contributed for the first time. Thank you so much! 🤩

* `Anand Saini`_
* `Bruno Hebling Vieira`_
* `Daniel McCloy`_
* `Denis Engemann`_
* `Mathieu Scheltienne`_
* `Scott Huberty`_

The following authors had contributed before. Thank you for sticking around! 🤘

* `Adam Li`_
* `Alex Rockhill`_
* `Alexandre Gramfort`_
* `Dominik Welke`_
* `Eduard Ort`_
* `Eric Larson`_
* `Richard Höchenberger`_
* `Robert Luke`_
* `Stefan Appelhoff`_
* `Teon Brooks`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

🚀 Enhancements
^^^^^^^^^^^^^^^

- You can now write raw data and an associated empty-room recording with just a single call to :func:`mne_bids.write_raw_bids`: the ``empty_room`` parameter now also accepts an :class:`mne.io.Raw` data object. The empty-room session name will be derived from the recording date automatically, by `Richard Höchenberger`_ (:gh:`998`)

- :func:`~mne_bids.write_raw_bids` now stores participant weight and height in ``participants.tsv``, by `Richard Höchenberger`_ (:gh:`1031`)

- :func:`~mne_bids.write_raw_bids` now supports EGI format, by `Anand Saini`_, `Scott Huberty`_ and `Mathieu Scheltienne`_ (:gh:`1006`)

- When a given subject cannot be found, valid suggestions are now printed, by `Eric Larson`_ (:gh:`1066`)

- TSV files that are empty (i.e., only a header row is present) are now handled more robustly and a warning is issued, by `Stefan Appelhoff`_ (:gh:`1038`)

- :class:`~mne_bids.BIDSPath` now supports the BIDS "description" entity ``desc``, used in derivative data, by `Richard Höchenberger`_ (:gh:`1049`)

- Added support for ``GSR`` (galvanic skin response / electrodermal activity, EDA) and ``TEMP`` (temperature) channel types, by `Richard Höchenberger`_ (:gh:`1059`)

- Added support for reading EEG files in Curry 8 format ('.cdt' extension) by `Denis Engemann`_ (:gh:`1072`)

🧐 API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :func:`~mne_bids.write_raw_bids` now expects all but the first four parameters to be passed as keyword arguments, by `Richard Höchenberger`_ (:gh:`1054`)

- The ``events_data`` parameter of :func:`~mne_bids.write_raw_bids` has been deprecated in favor of a new parameter named ``events``. This ensures more consistency between the MNE-BIDS and MNE-Python APIs. You may continue using the ``events_data`` parameter for now, but a ``FutureWarning`` will be raised. ``events_data`` will be removed in MNE-BIDS 0.14, by `Richard Höchenberger`_ (:gh:`1054`)

- In many places, we used to infer the ``datatype`` of a :class:`~mne_bids.BIDSPath` from the ``suffix``, if not explicitly provided. However, this has lead to trouble in certain edge cases. In an effort to reduce the amount of implicit behavior in MNE-BIDS, we now require users to explicitly specify a ``datatype`` whenever the invoked functions or methods expect one, by `Richard Höchenberger`_ (:gh:`1030`)

- :func:`mne_bids.make_dataset_description` now accepts keyword arguments only, and can now also write the following metadata: ``HEDVersion``, ``EthicsApprovals``, ``GeneratedBy``, and ``SourceDatasets``, by `Stefan Appelhoff`_ (:gh:`406`)

- The deprecated function ``mne_bids.mark_bad_channels`` has been removed in favor of :func:`mne_bids.mark_channels`, by `Richard Höchenberger`_ (:gh:`1009`)

- :func:`mne_bids.print_dir_tree` now raises a :py:class:`FileNotFoundError` instead of a :py:class:`ValueError` if the directory does not exist, by `Richard Höchenberger`_ (:gh:`1013`)

- Passing only one of ``events`` and ``event_id`` to :func:`~mne_bids.write_raw_bids` now raises a ``ValueError`` instead of a ``RuntimeError``, by `Richard Höchenberger`_ (:gh:`1054`)

- Until now, :class:`mne_bids.BIDSPath` prepends extensions with a period "." automatically. We intend to remove this undocumented side-effect and now emit a ``FutureWarning`` if an ``extension`` that does not start with a ``.`` is provided. Starting with MNE-BIDS 0.12, an exception will be raised in this case, by `Richard Höchenberger`_ (:gh:`1061`)

- Provide a more helpful error message when trying to write non-preloaded concatenated data, by `Richard Höchenberger`_ (:gh:`1075`)

🛠 Requirements
^^^^^^^^^^^^^^^

- MNE-BIDS now requires MNE-Python 1.0 or newer.

- Writing BrainVision files now requires ``pybv`` version ``0.7.3``, by `Stefan Appelhoff`_ (:gh:`1011`)

🪲 Bug fixes
^^^^^^^^^^^^

- Fix ACPC in ``surface RAS`` instead of ``scanner RAS`` in :ref:`ieeg-example` and add convenience functions :func:`mne_bids.convert_montage_to_ras` and :func:`mne_bids.convert_montage_to_mri` to help, by `Alex Rockhill`_ (:gh:`990`)

- Suppress superfluous warnings about MaxShield in many functions when handling Elekta/Neuromag/MEGIN data, by `Richard Höchenberger`_ (:gh:`1000`)

- The MNE-BIDS Inspector didn't work if ``mne-qt-browser`` was installed and used as the default plotting backend, as the Inspector currently only supports the Matplotlib backend, by `Richard Höchenberger`_ (:gh:`1007`)

- :func:`~mne_bids.copyfiles.copyfile_brainvision` can now deal with ``.dat`` file extension, by `Dominik Welke`_ (:gh:`1008`)

- :func:`~mne_bids.print_dir_tree` now correctly expands ``~`` to the user's home directory, by `Richard Höchenberger`_ (:gh:`1013`)

- :func:`~mne_bids.write_raw_bids` now correctly excludes stim channels when writing to electrodes.tsv, by `Scott Huberty`_ (:gh:`1023`)

- :func:`~mne_bids.read_raw_bids` doesn't populate ``raw.info['subject_info']`` with invalid values anymore, preventing users from writing the data to disk again, by `Richard Höchenberger`_ (:gh:`1031`)

- Writing EEGLAB files was sometimes broken when ``.set`` and ``.fdt`` pairs were supplied. This is now fixed in :func:`~mne_bids.copyfiles.copyfile_eeglab`, by `Stefan Appelhoff`_ (:gh:`1039`)

- Writing and copying CTF files now works on Windows when files already exist (``overwrite=True``), by `Stefan Appelhoff`_ (:gh:`1035`)

- Instead of deleting files and raising cryptic errors, an intentional error message is now sent when calling :func:`~mne_bids.write_raw_bids` with the source file identical to the destination file, unless ``format`` is specified, by `Adam Li`_ and `Stefan Appelhoff`_ (:gh:`889`)

- Internal helper function to :func:`~mne_bids.read_raw_bids` would reject BrainVision data if ``_scans.tsv`` listed a ``.eeg`` file instead of ``.vhdr``, by `Teon Brooks`_ (:gh:`1034`)

- Whenever :func:`~mne_bids.read_raw_bids` encounters a channel type that currently doesn't translate into an appropriate MNE channel type, the channel type will now be set to ``'misc``. Previously, seemingly arbitrary channel types would be applied, e.g. ``'eeg'`` for GSR and temperature channels, by `Richard Höchenberger`_ (:gh:`1052`)

- Fix the incorrect setting of the fields ``ContinuousHeadLocalization`` and ``HeadCoilFrequency`` for Neuromag MEG recordings, by `Eduard Ort`_ (:gh:`1067`)

.. _changes_0_10:

Version 0.10 (2022-03-14)
-------------------------

This release brings experimental fNIRS suppot, improvements in coordinate frame
handling, and various enhancements regarding MRI fiducials.

📝 Notable changes
~~~~~~~~~~~~~~~~~~

- We now have **experimental** support for fNIRS data (SNIRF format). This is
  still super fresh, and the respective BIDS enhancement proposal (BEP) has not
  yet been finalized & accepted into the standard. However, we're excitied to
  be able to do this first step towards fNIRS support!

- Numerous improvements have been added to enhance our support for various
  coordinate frames, including those that are not yet supported by MNE-BIDS.
  These changes are mostly relevant to iEEG users. Please see the detailed list
  of changes below.

- We have added support for storing and reading multiple anatomical landmarks
  ("fiducials") for the same participant. This makes it possible, for example,
  to store different sets of landmarks for each recording session.

- It's now possible to store Neuroscan (CNT) files with MNE-BIDS.

👩🏽‍💻 Authors
~~~~~~~~~~~~~~~

The following authors contributed for the first time. Thank you so much! 🤩

* `Simon Kern`_
* `Swastika Gupta`_
* `Yorguin Mantilla`_

The following authors had contributed before. Thank you for sticking around! 🤘

* `Adam Li`_
* `Alex Rockhill`_
* `Alexandre Gramfort`_
* `Eric Larson`_
* `Richard Höchenberger`_
* `Robert Luke`_
* `Stefan Appelhoff`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

🚀 Enhancements
^^^^^^^^^^^^^^^

- Add experimental support for fNIRS (SNIRF) files in :func:`mne_bids.write_raw_bids`, by `Robert Luke`_ (:gh:`406`)

- Add support for CNT (Neuroscan) files in :func:`mne_bids.write_raw_bids`, by `Yorguin Mantilla`_ (:gh:`924`)

- Add the ability to write multiple landmarks with :func:`mne_bids.write_anat` (e.g. to have separate landmarks for different sessions) via the new ``kind`` parameter, by `Alexandre Gramfort`_ (:gh:`955`)

- Similarly, :func:`mne_bids.get_head_mri_trans` and :func:`mne_bids.update_anat_landmarks` gained a new ``kind`` parameter to specify which of multiple landmark sets to operate on, by `Alexandre Gramfort`_ and `Richard Höchenberger`_ (:gh:`955`, :gh:`957`)

- Add support for iEEG data in the coordinate frame ``Pixels``; although MNE-Python does not recognize this coordinate frame and so it will be set to ``unknown`` in the montage, MNE-Python can still be used to analyze this kind of data, by `Alex Rockhill`_ (:gh:`976`)

- Add an explanation in :ref:`ieeg-example` of why it is better to have intracranial data in individual rather than template coordinates, by `Alex Rockhill`_ (:gh:`975`)

- :func:`mne_bids.update_anat_landmarks` can now directly work with fiducials saved from the MNE-Python coregistration GUI or :func:`mne.io.write_fiducials`, by `Richard Höchenberger`_ (:gh:`977`)

- All non-MNE-Python BIDS coordinate frames are now set to ``'unknown'`` on reading, by `Alex Rockhill`_ (:gh:`979`)

- :func:`mne_bids.write_raw_bids` can now write to template coordinates by `Alex Rockhill`_ (:gh:`980`)

- Add :func:`mne_bids.template_to_head` to transform channel locations in BIDS standard template coordinate systems to ``head`` and also provides a ``trans``, by `Alex Rockhill`_ (:gh:`983`)

🧐 API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :func:`mne_bids.update_anat_landmarks` will now by default raise an exception if the requested MRI landmarks do not already exist. Use the new ``on_missing`` parameter to control this behavior, by `Richard Höchenberger`_ (:gh:`957`)

- :func:`mne_bids.get_head_mri_trans` now raises a warning if ``datatype`` or ``suffix`` of the provided electrophysiological :class:`mne_bids.BIDSPath` are not set. In the future, this will raise an exception, by `Richard Höchenberger`_(:gh:`969`)

- Passing ``fs_subject=None`` to :func:`get_head_mri_trans` has been deprecated. Please pass the FreeSurfer subject name explicitly, by Richard Höchenberger`_ (:gh:`977`)

- Corrupted or missing fiducials in ``head`` coordinates now raise an error instead of warning in :func:`mne_bids.write_raw_bids` by `Alex Rockhill`_ (:gh:`980`)

🛠 Requirements
^^^^^^^^^^^^^^^

- MNE-BIDS now requires Jinja2 to work with MNE-Python 0.24.

🪲 Bug fixes
^^^^^^^^^^^^

- Forcing EDF conversion in :func:`mne_bids.write_raw_bids` properly uses the ``overwrite`` parameter now, by `Adam Li`_ (:gh:`930`)

- :func:`mne_bids.make_report` now correctly handles ``participant.tsv`` files that only contain a ``participant_id`` column, by `Simon Kern`_ (:gh:`912`)

- :func:`mne_bids.write_raw_bids` doesn't store age, handedness, and sex in ``participants.tsv`` anymore for empty-room recordings, by `Richard Höchenberger`_ (:gh:`935`)

- When :func:`mne_bids.read_raw_bids` automatically creates new hierarchical event names based on event values (in cases where the same ``trial_type`` was assigned to different ``value``s in ``*_events.tsv``), ``'n/a'`` values will now be converted to ``'na'``, by `Richard Höchenberger`_ (:gh:`937`)

- Avoid ``DeprecationWarning`` in :func:`mne_bids.inspect_dataset` with the upcoming MNE-Python 1.0 release, by `Richard Höchenberger`_ (:gh:`942`)

- Avoid modifying the instance of :class:`mne_bids.BIDSPath` if validation fails when calling :meth:`mne_bids.BIDSPath.update`, by `Alexandre Gramfort`_ (:gh:`950`)

- :func:`mne_bids.get_head_mri_trans` now respects ``datatype`` and ``suffix`` of the provided electrophysiological :class:`mne_bids.BIDSPath`, simplifying e.g. reading of derivaties, by `Richard Höchenberger`_ (:gh:`969`)

- Do not convert unknown coordinate frames to ``head``, by `Alex Rockhill`_ (:gh:`976`)

.. _changes_0_9:

Version 0.9 (2021-11-23)
------------------------

This release brings compatibility with MNE-Python 0.24 and some new convenience
functions and speedups of existing code to help you be more productive! 👩🏽‍💻
And, of course, plenty of bug fixes. 🐞

Notable changes
~~~~~~~~~~~~~~~

- 🧠 Compatibility with MNE-Python 0.24!
- 👻 Anonymize an entire BIDS dataset via :func:`mne_bids.anonymize_dataset`!
- 🏝 Conveniently turn a path into a :class:`BIDSPath` via
  :func:`get_bids_path_from_fname`!
- 🏎 :func:`mne_bids.stats.count_events` and :meth:`mne_bids.BIDSPath.match`
  are operating **much** faster now!
- 🔍 :func:`write_raw_bids` now stores the names of the input files in the
  ``source`` column of ``*_scans.tsv``, making it easier for you to
  *go back to the source* should you ever need to!

Authors
~~~~~~~

People who contributed to this release (in alphabetical order):

* `Adam Li`_
* `Alex Rockhill`_
* `Alexandre Gramfort`_
* `Clemens Brunner`_
* `Franziska von Albedyll`_
* `Julia Guiomar Niso Galán`_
* `Mainak Jas`_
* `Marijn van Vliet`_
* `Richard Höchenberger`_
* `Stefan Appelhoff`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

Enhancements
^^^^^^^^^^^^

- :func:`mne_bids.get_anat_landmarks` now accepts a :class:`mne_bids.BIDSPath` as ``image`` parameter, by `Alex Rockhill`_ (:gh:`852`)

- :func:`mne_bids.write_raw_bids` now accepts ``'EDF'`` as a ``'format'`` value to force conversion to EDF files, by `Adam Li`_ (:gh:`866`)

- :func:`mne_bids.write_raw_bids` now adds ``SpatialCompensation`` information to the JSON sidecar for MEG data, by `Julia Guiomar Niso Galán`_ (:gh:`885`)

- Modify iEEG tutorial to use MNE ``raw`` object, by `Alex Rockhill`_ (:gh:`859`)

- Add :func:`mne_bids.search_folder_for_text` to find specific metadata entries (e.g. all ``"n/a"`` sidecar data fields, or to check that "60 Hz" was written properly as the power line frequency), by `Alex Rockhill`_ (:gh: `870`)

- Add :func:`mne_bids.get_bids_path_from_fname` to return a :class:`mne_bids.BIDSPath` from a file path, by `Adam Li`_ (:gh:`883`)

- Great performance improvements in :func:`mne_bids.stats.count_events` and :meth:`mne_bids.BIDSPath.match`, significantly reducing processing time, by `Richard Höchenberger`_ (:gh:`888`)

- The command ``mne_bids count_events`` gained new parameters: ``--output`` to direct the output into a CSV file; ``--overwrite`` to overwrite an existing file; and  ``--silent`` to suppress output of the event counts to the console, by `Richard Höchenberger`_ (:gh:`888`)

- The new function :func:`mne_bids.anonymize_dataset` can be used to anonymize an entire BIDS dataset, by `Richard Höchenberger`_ (:gh:`893`, :gh:`914`, :gh:`917`)

- :meth:`mne_bids.BIDSPath.find_empty_room` gained a new parameter ``use_sidecar_only`` to limit empty-room search to the metadata stored in the sidecar files, by `Richard Höchenberger`_ (:gh:`893`)

- :meth:`mne_bids.BIDSPath.find_empty_room` gained a new parameter ``verbose`` to limit verbosity of the output, by `Richard Höchenberger`_ (:gh:`893`)

- :func:`mne_bids.write_raw_bids` can now write the source filename to ``scans.tsv`` in a new column, ``source``, by `Adam Li`_ (:gh:`890`)

API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^

- ``mne_bids.mark_bad_channels`` deprecated in favor of :func:`mne_bids.mark_channels`, which allows specifying the status to change channels to by `Adam Li`_ (:gh:`882`)

- :func:`mne_bids.get_entities_from_fname` does not return ``suffix`` anymore as that is not considered a BIDS entity, by `Adam Li`_ (:gh:`883`)

- Reading BIDS data with ``"HeadCoilFrequency"`` and ``"PowerLineFrequency"`` data specified in JSON sidecars will only "warn" in case of mismatches between Raw and JSON data, by `Franziska von Albedyll`_ (:gh:`855`)

- Accessing :attr:`mne_bids.BIDSPath.fpath` emit a warning anymore if the path does not exist. This behavior was unreliable and yielded confusing error messages in certain use cases. Use `mne_bids.BIDSPath.fpath.exists()` to check whether the path exists in the file system, by `Richard Höchenberger`_ (:gh:`904`)

- :func:`mne_bids.get_entity_vals` gained a new parameter, ``ignore_dirs``, to exclude directories from the search, by `Adam Li`_ and `Richard Höchenberger`_ (:gh:`899`, :gh:`908`)

- In :func:`mne_bids.write_anat`, the deprecated parameters ``raw``, ``trans``, and ``t1w`` have been removed, by `Richard Höchenberger`_ (:gh:`909`)

- In :func:`mne_bids.write_raw_bids`, any EDF output is always stored with lower-case extension (``.edf``), by `Adam Li`_ (:gh:`906`)

Requirements
^^^^^^^^^^^^

- MNE-BIDS now requires MNE-Python 0.24 or newer.

- Writing BrainVision files now requires ``pybv`` version 0.6, by `Stefan Appelhoff`_ (:gh:`880`)

Bug fixes
^^^^^^^^^

- Fix writing Ricoh/KIT data that comes without an associated ``.mrk``, ``.elp``, or ``.hsp`` file using :func:`mne_bids.write_raw_bids`, by `Richard Höchenberger`_ (:gh:`850`)

- Properly support CTF MEG data with 2nd-order gradient compensation, by `Mainak Jas`_ (:gh:`858`)

- Fix writing and reading EDF files with upper-case extension (``.EDF``), by `Adam Li`_ (:gh:`868`)

- Fix reading of TSV files with only a single column, by `Marijn van Vliet`_ (:gh:`886`)

- Fix erroneous measurement date check in :func:`mne_bids.write_raw_bids` when requesting to anonymize empty-room data, by `Richard Höchenberger`_ (:gh:`893`)

- :func:`mne_bids.write_raw_bids` now raises an exception if the provided :class:`mne_bids.BIDSPath` doesn't contain ``subject`` and ``task`` entities, which are required for neurophysiological data, by `Richard Höchenberger`_ (:gh:`903`)

- :func:`mne_bids.read_raw_bids` now handles datasets with multiple electrophysiological data types correctly, by `Richard Höchenberger`_ (:gh:`910`, :gh`916`)

- More robust handling of situations where :func:`mne_bids.read_raw_bids` tries to read a file that does not exist, by `Richard Höchenberger`_ (:gh:`904`)

.. _changes_0_8:

Version 0.8 (2021-07-15)
------------------------

This release brings numerous improvements and fixes based on feedback from our
users, including those working with very large datasets. MNE-BIDS now handles
previously-overlooked edge cases, offers a much more efficient way to
store data on macOS and Linux (using symbolic links), and lays the groundwork
for supporting BIDS derivatives, i.e., storing modified data.

Notable changes
~~~~~~~~~~~~~~~

- You can now write preloaded and potentially modified data with
  :func:`mne_bids.write_raw_bids` by passing ``allow_preload=True``. This is
  a first step towards supporting derivative files.
- `mne_bids.BIDSPath` now has property getters and setters for all BIDS
  entities. What this means is that you can now do things like
  ``bids_path.subject = '01'`` instead of ``bids_path.update(subject='01')``.
- We now support Deep Brain Stimulation (DBS) data.
- The way we handle anatomical landmarks was greatly revamped to ensure we're
  always using the correct coordinate systems. A new function,
  `mne_bids.get_anat_landmarks`, helps with extracting fiducial points from
  anatomical scans.
- When creating a BIDS dataset from FIFF files on macOS and Linux, MNE-BIDS
  can now optionally generate symbolic links to the original files instead of
  copies. Simply pass ``symlink=True`` to
  :func:`mne_bids.write_raw_bids`. This can massively reduce the storage space
  requirements.

Authors
~~~~~~~

* `Adam Li`_
* `Alex Rockhill`_
* `Alexandre Gramfort`_
* `Clemens Brunner`_
* `Eduard Ort`_
* `Eric Larson`_
* `Jean-Rémi King`_ (new contributor)
* `Julia Guiomar Niso Galán`_ (new contributor)
* `Mainak Jas`_
* `Richard Höchenberger`_
* `Richard Köhler`_ (new contributor)
* `Robert Luke`_ (new contributor)
* `Sin Kim`_ (new contributor)
* `Stefan Appelhoff`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

Enhancements
^^^^^^^^^^^^

- The fields "DigitizedLandmarks" and "DigitizedHeadPoints" in the json sidecar of Neuromag data are now set to ``true`` or ``false`` depending on whether any landmarks (NAS, RPA, LPA) or extra points are found in ``raw.info['dig']``, by `Eduard Ort`_ (:gh:`772`)
- Updated the "Read BIDS datasets" example to use data from `OpenNeuro <https://openneuro.org>`_, by `Alex Rockhill`_ (:gh:`753`)
- :func:`mne_bids.get_head_mri_trans` is now more lenient when looking for the fiducial points (LPA, RPA, and nasion) in the MRI JSON sidecar file, and accepts a larger variety of landmark names (upper- and lowercase letters; ``'nasion'`` instead of only ``'NAS'``), by `Richard Höchenberger`_ (:gh:`769`)
- :func:`mne_bids.get_head_mri_trans` gained a new keyword argument ``t1_bids_path``, allowing for the MR scan to be stored in a different session or even in a different BIDS dataset than the electrophysiological recording, by `Richard Höchenberger`_ (:gh:`771`)
- Add writing simultaneous EEG-iEEG recordings via :func:`mne_bids.write_raw_bids`. The desired output datatype must be specified in the :class:`mne_bids.BIDSPath` object, by `Richard Köhler`_ (:gh:`774`)
- :func:`mne_bids.write_raw_bids` gained a new keyword argument ``symlink``, which allows to create symbolic links to the original data files instead of copying them over. Currently works for ``FIFF`` files on macOS and Linux, by `Richard Höchenberger`_ (:gh:`778`)
- :class:`mne_bids.BIDSPath` now has property getter and setter methods for all BIDS entities, i.e., you can now do things like ``bids_path.subject = 'foo'`` and don't have to resort to ``bids_path.update()``. This also ensures you'll get proper completion suggestions from your favorite Python IDE, by `Richard Höchenberger`_ (:gh:`786`)
- :func:`mne_bids.write_raw_bids` now stores information about continuous head localization measurements (e.g., Elekta/Neuromag cHPI) in the MEG sidecar file, by `Richard Höchenberger`_ (:gh:`794`)
- :func:`mne_bids.write_raw_bids` gained a new parameter ``empty_room`` that allows to specify an associated empty-room recording when writing an MEG data file. This information will be stored in the ``AssociatedEmptyRoom`` field of the MEG JSON sidecar file, by `Richard Höchenberger`_ (:gh:`795`)
- Added support for the new channel type ``'dbs'`` (Deep Brain Stimulation), which was introduced in MNE-Python 0.23, by `Richard Köhler`_ (:gh:`800`)
- :func:`mne_bids.read_raw_bids` now warns in many situations when it encounters a mismatch between the channels in ``*_channels.tsv`` and the raw data, by `Richard Höchenberger`_ (:gh:`823`)
- MNE BIDS now accepts ``.mrk`` head digitization files used in the KIT/Yokogawa/Ricoh MEG system, by `Jean-Rémi King`_ and `Stefan Appelhoff`_ (:gh:`842`)

API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^

- Writing datasets via :func:`write_raw_bids`, will now never overwrite ``dataset_description.json`` file, by `Adam Li`_ (:gh:`765`)
- When writing BIDS datasets, MNE-BIDS now tags them as BIDS 1.6.0 (we previously tagged them as BIDS 1.4.0), by `Richard Höchenberger`_ (:gh:`782`)
- :func:`mne_bids.read_raw_bids` now passes ``allow_maxshield=True`` to the MNE-Python reader function by default when reading FIFF files. Previously, ``extra_params=dict(allow_maxshield=True)`` had to be passed explicitly, by `Richard Höchenberger`_ (:gh:`787`)
- The ``raw_to_bids`` command has lost its ``--allow_maxshield`` parameter. If writing a FIFF file, we will now always assume that writing data before applying a Maxwell filter is fine, by `Richard Höchenberger`_ (:gh:`787`)
- :meth:`mne_bids.BIDSPath.find_empty_room` now first looks for an ``AssociatedEmptyRoom`` field in the MEG JSON sidecar file to retrieve the empty-room recording; only if this information is missing, it will proceed to try and find the best-matching empty-room recording based on measurement date (i.e., fall back to the previous behavior), by `Richard Höchenberger`_ (:gh:`795`)
- If :func:`mne_bids.read_raw_bids` encounters raw data with the ``STI 014`` stimulus channel and this channel is not explicitly listed in ``*_channels.tsv``, it is now automatically removed upon reading, by `Richard Höchenberger`_ (:gh:`823`)
- :func:`mne_bids.get_anat_landmarks` was added to clarify and simplify the process of generating landmarks that now need to be passed to :func:`mne_bids.write_anat`; this deprecates the arguments ``raw``, ``trans`` and ``t1w`` of :func:`mne_bids.write_anat`, by `Alex Rockhill`_ and `Alexandre Gramfort`_ (:gh:`827`)
- :func:`write_raw_bids` now accepts preloaded raws as input with some caveats if the new parameter ``allow_preload`` is explicitly set to ``True``. This enables some preliminary support for items such as uncommon file formats, generated data, and processed derivatives, by `Sin Kim`_ (:gh:`819`)
- MNE-BIDS now writes all TSV data files with a newline character at the end of the file, complying with UNIX/POSIX standards, by `Stefan Appelhoff`_ (:gh:`831`)

Requirements
^^^^^^^^^^^^

- For downloading `OpenNeuro <https://openneuro.org>`_ datasets, ``openneuro-py`` is now required to run the examples and build the documentation, by `Alex Rockhill`_ (:gh:`753`)
- MNE-BIDS now depends on `setuptools <https://setuptools.readthedocs.io>`_. This package is normally installed by your Python distribution automatically, so we don't expect any users to be affected by this change, by `Richard Höchenberger`_ (:gh:`794`)
- MNE-BIDS now requires Python 3.7 or higher, because Python 3.6 is soon reaching its end of life.

Bug fixes
^^^^^^^^^

- :func:`mne_bids.make_report` now (1) detects male/female sex and left/right handedness irrespective of letter case, (2) will parse a ``gender`` column if no ``sex`` column is found in ``participants.tsv``, and (3) reports sex as male/female instead of man/woman, by `Alex Rockhill`_ (:gh:`755`)
- The :class:`mne.Annotations` ``BAD_ACQ_SKIP`` – added by the acquisition system to ``FIFF`` files – will now be preserved when reading raw data, even if these time periods are **not** explicitly included in ``*_events.tsv``, by `Richard Höchenberger`_ and `Alexandre Gramfort`_ (:gh:`754` and :gh:`762`)
- :func:`mne_bids.write_raw_bids` will handle different cased extensions for EDF files, such as `.edf` and `.EDF` by `Adam Li`_ (:gh:`765`)
- :func:`mne_bids.inspect_dataset` didn't handle certain filenames correctly on some systems, by `Richard Höchenberger`_ (:gh:`769`)
- :func:`mne_bids.write_raw_bids` now works across data types with ``overwrite=True``, by `Alexandre Gramfort`_ (:gh:`791`)
- :func:`mne_bids.read_raw_bids` didn't always replace all traces of the measurement date and time stored in the raw data with the date found in `*_scans.tsv`, by `Richard Höchenberger`_ (:gh:`812`, :gh:`815`)
- :func:`mne_bids.read_raw_bids` crashed when the (optional) ``acq_time`` column was missing in ``*_scans.tsv``, by `Alexandre Gramfort`_ (:gh:`814`)
- :func:`mne_bids.write_raw_bids` doesn't crash anymore if the designated output directory contains the string ``"tsv"``, by `Richard Höchenberger`_ (:gh:`833`)
- :func:`mne_bids.get_head_mri_trans` gave incorrect results when the T1 image was not in LIA format, now all formats function properly, by `Alex Rockhill`_ and `Alexandre Gramfort`_ (:gh:`827`)
- :func:`mne_bids.get_head_mri_trans` and :func:`mne_bids.write_anat` used a T1w image but depended specifically on the freesurfer T1w image. Now the FreeSurfer subjects directory is used, by `Alex Rockhill`_ and `Alexandre Gramfort`_ (:gh:`827`)

.. _changes_0_7:

Version 0.7 (2021-03-22)
------------------------

This release brings numerous enhancements and bug fixes that enhance reading
and writing BIDS data, and improve compatibility with the latest BIDS
specifications.

Notable changes
~~~~~~~~~~~~~~~

- Channel names in ``*_channels.tsv`` and ``*_electrodes.tsv`` files now always
  take precedence over the names stored in the raw files.
- When reading data where the same trial type refers to different trigger
  values, we will now automatically create hierarchical event names in the
  form of ``trial_type/value1``, `trial_type/value2`` etc.
- :func:`mne_bids.write_raw_bids` now allows users to specify a format
  conversion via the new ``format`` parameter.
- Various improvements to data reading and :class:`mne_bids.BIDSPath` make
  working with real-life data easier.
- Many bug fixes in :func:`mne_bids.write_raw_bids` and in the MNE-BIDS
  Inspector.

Authors
~~~~~~~

* `Adam Li`_
* `Alexandre Gramfort`_
* `Austin Hurst`_
* `Diego Lozano-Soldevilla`_
* `Eduard Ort`_
* `Maximilien Chaumon`_
* `Richard Höchenberger`_
* `Stefan Appelhoff`_
* `Tom Donoghue`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

Enhancements
^^^^^^^^^^^^

- Some datasets out in the real world have a non-standard ``stim_type`` instead of a ``trial_type`` column in ``*_events.tsv``. :func:`mne_bids.read_raw_bids` now makes use of this column, and emits a warning, encouraging users to rename it, by `Richard Höchenberger`_ (:gh:`680`)
- When reading data where the same event name or trial type refers to different event or trigger values, we will now create a hierarchical event name in the form of ``trial_type/value``, e.g. ``stimulus/110``, by `Richard Höchenberger`_ (:gh:`688`)
- When reading data via :func:`mne_bids.read_raw_bids`, the channel names specified in the BIDS ``*_channels.tsv`` and ``*_electrodes.tsv`` files now always take precedence over (and do not need to match) the channel names stored in the raw files anymore, by `Adam Li`_ and `Richard Höchenberger`_ (:gh:`691`, :gh:`704`)
- Improve the ``Convert iEEG data to BIDS`` tutorial to include a note on how BIDS and MNE-Python coordinate frames are handled, by `Adam Li`_ (:gh:`717`)
- More detailed error messages when trying to write modified data via :func:`mne_bids.write_raw_bids`, by `Richard Höchenberger`_ (:gh:`719`)
- If ``check=True``, :class:`mne_bids.BIDSPath` now checks the ``space`` entity to be valid according to BIDS specification Appendix VIII, by `Stefan Appelhoff`_ (:gh:`724`)
- Data types that are currently unsupported by MNE-BIDS (e.g. ``dwi``, ``func``) can now be used in :class:`mne_bids.BIDSPath` by setting ``check=False``, by `Adam Li`_ (:gh:`744`)
- Arbitrary file names can now be represented as a `BIDSPath`` by passing the entire name as ``suffix`` and setting ``check=False``, by `Adam Li`_ (:gh:`729`)
- Add support for MNE's flux excitation channel (``exci``), by `Maximilien Chaumon`_ (:gh:`728`)
- :meth:`mne_bids.BIDSPath.match` gained a new parameter ``check``; when setting ``check=True``, ``match()`` will only return paths that conform to BIDS, by `Richard Höchenberger`_ (:gh:`726`)
- ``BIDSPath.root`` now automatically expands ``~`` to the user's home directory, by `Richard Höchenberger`_ (:gh:`725`)

API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^

- Add ``format`` kwarg to :func:`mne_bids.write_raw_bids` that allows users to specify if they want to force conversion to ``BrainVision`` or ``FIF`` file format, by `Adam Li`_ (:gh:`672`)
- :func:`mne_bids.read_raw_bids` now stores the ``participant_id`` value from ``participants.tsv`` in ``raw.info['subject_info']['his_id']``, not in ``raw.info['subject_info']['participant_id']`` anymore, by `Richard Höchenberger`_ (:gh:`745`)

Requirements
^^^^^^^^^^^^

- For writing BrainVision files, ``pybv`` version 0.5 is now required to allow writing of non-voltage channels, by `Adam Li`_ (:gh:`670`)

Bug fixes
^^^^^^^^^

- Fix writing MEGIN Triux files, by `Alexandre Gramfort`_ (:gh:`674`)
- Anonymization of EDF files in :func:`write_raw_bids` will now convert recording date to ``01-01-1985 00:00:00`` if anonymization takes place, while setting the recording date in the ``scans.tsv`` file to the anonymized date, thus making the file EDF/EDFBrowser compliant, by `Adam Li`_ (:gh:`669`)
- :func:`mne_bids.write_raw_bids` will not overwrite an existing ``coordsystem.json`` anymore, unless explicitly requested, by `Adam Li`_ (:gh:`675`)
- :func:`mne_bids.read_raw_bids` now properly handles datasets without event descriptions, by `Richard Höchenberger`_ (:gh:`680`)
- :func:`mne_bids.stats.count_events` now handles files without a ``trial_type`` or ``stim_type`` column gracefully, by `Richard Höchenberger`_ (:gh:`682`)
- :func:`mne_bids.read_raw_bids` now correctly treats ``coordsystem.json`` as optional for EEG and MEG data, by `Diego Lozano-Soldevilla`_ (:gh:`691`)
- :func:`mne_bids.read_raw_bids` now ignores ``exclude`` parameters passed via ``extra_params``, by `Richard Höchenberger`_ (:gh:`703`)
- :func:`mne_bids.write_raw_bids` now retains original event IDs in the ``value`` column of ``*_events.tsv``, by `Richard Höchenberger`_ (:gh:`708`)
- Fix writing correct ``iEEGCoordinateSystemDescription``, by `Stefan Appelhoff`_ (:gh:`706`)
- FIF files that were split due to filesize limitations (using the ``_split-<label>`` entity), are now all listed in ``scans.tsv``, as recommended by BIDS, by `Eduard Ort`_ (:gh:`710`)
- The ``mne_bids inspect`` command now automatically tries to discover flat channels by default; this should have been the case all along, but the default parameter was set incorrectly, by `Richard Höchenberger`_ (:gh:`726`)
- :func:`mne_bids.inspect_dataset` would sometimes open the same file multiple times, by `Richard Höchenberger`_ (:gh:`726`)
- :func:`mne_bids.inspect_dataset` would try to open the SSP projector selection window for non-MEG data, by `Richard Höchenberger`_ (:gh:`726`)

.. _changes_0_6:

Version 0.6 🎄 (2020-12-16)
---------------------------

These are challenging days for many of us, and to make your lives
ever so slightly easier, we've been working hard to deliver this early
Christmas present 🎁 And even if you do not celebrate Christmas, we are quite
certain you will like what we got for you! So – what are you waiting for? It's
time to unwrap!

Notable changes
~~~~~~~~~~~~~~~

- The new Inspector, which can be invoked via :func:`mne_bids.inspect_dataset`,
  allows you to interactively explore your raw data, change
  the bad channels selection, and edit :class:`mne.Annotations`. It also
  performs automated detection of flat data segments or channels, to assist
  you during visual inspection. The capabilities of the inspector will be
  further expanded in upcoming releases of MNE-BIDS.

- To further assist you during data inspection, we have added a function to
  summarize all events present in a dataset,
  :func:`mne_bids.stats.count_events`.

- Sidecar JSON files can now be updated using a template via
  :func:`mne_bids.update_sidecar_json`.

- You can now read and write FLASH MRI images using
  :func:`mne_bids.write_anat`. We also fixed some issues with MRI defacing
  along the way.

- Event durations are now preserved upon reading and writing data (we used to
  set all event durations to zero before).

Authors
~~~~~~~

* `Stefan Appelhoff`_
* `Adam Li`_
* `Alex Rockhill`_
* `Alexandre Gramfort`_
* `Austin Hurst`_
* `Ethan Knights`_  (new contributor)
* `Mainak Jas`_
* `Richard Höchenberger`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

Enhancements
^^^^^^^^^^^^

- The function :func:`mne_bids.print_dir_tree` has a new parameter ``return_str`` which allows it to return a str of the dir tree instead of printing it, by `Stefan Appelhoff`_ (:gh:`600`)
- The function :func:`mne_bids.print_dir_tree` has a new parameter ``return_str`` which allows it to return a str of the dir tree instead of printing it, by `Stefan Appelhoff`_ (:gh:`600`)
- :func:`mne_bids.write_raw_bids` now preserves event durations when writing :class:`mne.Annotations` to ``*_events.tsv`` files, and :func:`mne_bids.read_raw_bids` restores these durations upon reading, by `Richard Höchenberger`_ (:gh:`603`)
- Writing BrainVision data via :func:`mne_bids.write_raw_bids` will now set the unit of EEG channels to µV for enhanced interoperability with other software, by `Alexandre Gramfort`_, `Stefan Appelhoff`_, and `Richard Höchenberger`_ (:gh:`610`)
- New function :func:`mne_bids.stats.count_events` to easily summarize all the events present in a dataset, by `Alexandre Gramfort`_ (:gh:`629`)
- Add :func:`mne_bids.update_sidecar_json` to allow updating sidecar JSON files with a template, by `Adam Li`_ and `Austin Hurst`_ (:gh:`601`)
- Add support for anonymizing EDF and BDF files without converting to BrainVision format, by `Austin Hurst`_ (:gh:`636`)
- Add support for writing FLASH MRI data with :func:`mne_bids.write_anat`, by `Alexandre Gramfort`_ (:gh:`641`)
- Add interactive data inspector :func:`mne_bids.inspect_dataset`, by `Richard Höchenberger`_ (:gh:`561`)
- Do not complain about missing events if it's likely we're dealing with resting-state data in :func:`mne_bids.write_raw_bids`, **by new contributor** `Ethan Knights`_ (:gh:`631`)

API changes
^^^^^^^^^^^

- When passing event IDs to :func:`mne_bids.write_raw_bids` via ``events_data`` without an accompanying event description in ``event_id``, we will now raise a `ValueError`. This ensures that accidentally un-described events won't get written unnoticed, by `Richard Höchenberger`_ (:gh:`603`)
- The :func:`mne_bids.get_head_mri_trans` now has a parameter ``extra_params`` to allow passing arguments specific to a file format, by `Mainak Jas`_ (:gh:`638`)
- The first parameter of :func:`mne_bids.write_anat` is now called ``image`` and not ``t1w``, by `Alexandre Gramfort`_ (:gh:`641`)

Requirements
^^^^^^^^^^^^

- Writing BrainVision data now requires ``pybv`` 0.4 or later.

Bug fixes
^^^^^^^^^

- Make sure large FIF files with splits are handled transparently on read and write, by `Alexandre Gramfort`_ (:gh:`612`)
- The function :func:`mne_bids.write_raw_bids` now outputs ``*_electrodes.tsv`` and ``*_coordsystem.json`` files for EEG/iEEG data that are BIDS-compliant (only contain subject, session, acquisition, and space entities), by `Adam Li`_ (:gh:`601`)
- Make sure writing empty-room data with anonymization shifts the session back in time, by `Alexandre Gramfort`_ (:gh:`611`)
- Fix a bug in :func:`mne_bids.write_raw_bids`, where passing raw data with :class:`mne.Annotations` set and the ``event_id`` dictionary not containing the :class:`mne.Annotations` descriptions as keys would raise an error, by `Richard Höchenberger`_ (:gh:`603`)
- Fix a bug in :func:`mne_bids.write_raw_bids` when passing raw MEG data with Internal Active Shielding (IAS) from Triux system, by `Alexandre Gramfort`_ (:gh:`616`)
- Fix a bug in :func:`mne_bids.write_raw_bids`, where original format of data was not kept when writing to FIFF, by `Alexandre Gramfort`_, `Stefan Appelhoff`_, and `Richard Höchenberger`_ (:gh:`610`)
- Fix a bug where conversion to BrainVision format was done even when non-Volt channel types were present in the data (BrainVision conversion is done by ``pybv``, which currently only supports Volt channel types), by `Stefan Appelhoff`_ (:gh:`619`)
- Ensure sidecar files (`.tsv` and `.json`) are always read and written in UTF-8, by `Richard Höchenberger`_ (:gh:`625`)
- Fix a bug where ``participants.tsv`` was not being appended to correctly when it contained a subset of ``hand``, ``age`` and ``sex``, by `Adam Li`_ (:gh:`648`)
- :func:`mne_bids.copyfiles.copyfile_eeglab` didn't handle certain EEGLAB files correctly, by `Richard Höchenberger`_ (:gh:`653`)
- Fix bug where images with different orientations than the T1 used to define the landmarks were defaced improperly, by `Alex Rockhill`_ (:gh:`651`)


.. _changes_0_5:

Version 0.5 (2020-10-22)
------------------------

This is a **big** release with lots of changes, many of them breaking existing
code. But do not fear: migration is easy, and you will **love** what we have
been cooking for you!


Notable changes
~~~~~~~~~~~~~~~
- We introduce `mne_bids.BIDSPath`, a new class for all BIDS file and folder
  operations. All functions in MNE-BIDS that previously accepted filenames
  and folder locations (e.g. ``bids_root``) have been updated to work with
  ``BIDSPath``. Others have been removed.
  Consequently, you will need to update your existing code, too.
  See the ``API changes`` section for an overview of which functions have
  changed or have been removed, and follow
  :ref:`this introduction<bidspath-intro>` and our
  to learn about the basics of ``BIDSPath``. Don't worry – it's going to be a
  breeze! 🤗

- MNE-BIDS now requires MNE-Python 0.21.

- The new function :func:`mne_bids.make_report` can help you populate a
  paragraph of your next paper's methods section!

- You can now interactively mark channels as bad using
  ``mne_bids.mark_bad_channels``.

- Elekta/Neuromag/MEGIN fine-calibration and crosstalk files can now be stored
  in your BIDS dataset via :func:`mne_bids.write_meg_calibration` and
  :func:`mne_bids.write_meg_crosstalk`.

- When writing a raw file that contains annotations, these will now be
  converted to and stored as events by :func:`mne_bids.write_raw_bids`.


Authors
~~~~~~~
The following people have contributed to this release of MNE-BIDS:

- `Adam Li`_
- `Alexandre Gramfort`_
- `Alex Rockhill`_
- `Austin Hurst`_
- `Evgenii Kalenkovich`_
- `Mainak Jas`_
- `Richard Höchenberger`_
- `Robert Luke`_
- `Stefan Appelhoff`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

Enhancements
^^^^^^^^^^^^

- Introduce :class:`mne_bids.BIDSPath`, the new universal MNE-BIDS working horse for file operations, by `Adam Li`_, `Alex Rockhill`_, and `Richard Höchenberger`_ (:gh:`496`, :gh:`507`, :gh:`511`, :gh:`514`, :gh:`542`)
- The new function :func:`mne_bids.make_report` and its corresponding CLI function, ``make_report``, produce human-readable summary of the BIDS dataset, by `Adam Li`_ (:gh:`457`)
- :func:`read_raw_bids` now reads ``participants.tsv`` data, by `Adam Li`_ (:gh:`392`)
- :func:`mne_bids.get_entity_vals` has gained ``ignore_*`` keyword arguments to exclude specific values from the list of results, e.g. the entities for a particular subject or task, by `Richard Höchenberger`_ (:gh:`409`)
- :func:`mne_bids.write_raw_bids` now uses the ``space`` BIDS entity when writing iEEG electrodes and coordinate frames, by `Adam Li`_ (:gh:`390`)
- :code:`convert_ieeg_to_bids` to now use sample ECoG EDF data, by `Adam Li`_ (:gh:`390`)
- :func:`mne_bids.write_raw_bids` now writes ``CoordinateSystemDescription`` as specified in BIDS Specification if CoordinateSystem is MNE-compatible, by `Adam Li`_ (:gh:`416`)
- :func:`mne_bids.write_raw_bids` and :func:`mne_bids.read_raw_bids` now handle scalp EEG if Captrak coordinate system and NAS/LPA/RPA landmarks are present, by `Adam Li`_ (:gh:`416`)
- :func:`write_raw_bids` now adds citations to the ``README``, by `Alex Rockhill`_ (:gh:`463`)
- :func:`make_dataset_description` now has an additional parameter ``dataset_type`` to set the recommended field ``DatasetType`` (defaults to ``"raw"``), by `Stefan Appelhoff`_ (:gh:`472`)
- :func:`mne_bids.copyfiles.copyfile_brainvision` now has an ``anonymize`` parameter to control anonymization, by `Stefan Appelhoff`_ (:gh:`475`)
- :func:`mne_bids.read_raw_bids` and :func:`mne_bids.write_raw_bids` now map respiratory (``RESP``) channel types, by `Richard Höchenberger`_ (:gh:`482`)
- When impedance values are available from a ``raw.impedances`` attribute, MNE-BIDS will now write an ``impedance`` column to ``*_electrodes.tsv`` files, by `Stefan Appelhoff`_ (:gh:`484`)
- :func:`mne_bids.write_raw_bids` writes out status_description with ``'n/a'`` values into the channels.tsv sidecar file, by `Adam Li`_ (:gh:`489`)
- Added a new function ``mne_bids.mark_bad_channels`` and command line interface ``mark_bad_channels`` which allows updating of the channel status (bad, good) and description of an existing BIDS dataset, by `Richard Höchenberger`_ (:gh:`491`)
- :func:`mne_bids.read_raw_bids` correctly maps all specified ``handedness`` and ``sex`` options to MNE-Python, instead of only an incomplete subset, by `Richard Höchenberger`_ (:gh:`550`)
- :func:`mne_bids.write_raw_bids` only writes a ``README`` if it does not already exist, by `Adam Li`_ (:gh:`489`)
- Allow :func:`mne_bids.write_raw_bids` to write EEG/iEEG files from Persyst using ``mne.io.read_raw_persyst`` function, by `Adam Li`_ (:gh:`546`)
- :func:`mne_bids.print_dir_tree` now works if a ``pathlib.Path`` object is passed, by `Adam Li`_ (:gh:`555`)
- Allow writing of Elekta/Neuromag/MEGIN fine-calibration and crosstalk data via the new functions :func:`mne_bids.write_meg_calibration` and :func:`mne_bids.write_meg_crosstalk`, and retrieval of the file locations via :attr:`BIDSPath.meg_calibration_fpath` and :attr:`BIDSPath.meg_crosstalk_fpath`, by `Richard Höchenberger`_ (:gh:`562`)
- Allow :func:`mne_bids.write_raw_bids` to write EEG/iEEG files from Nihon Kohden using ``mne.io.read_raw_nihon`` function, by `Adam Li`_ (:gh:`567`)
- Allow :func:`mne_bids.get_entity_vals` to ignore datatypes using ``ignore_datatypes`` kwarg, by `Adam Li`_ (:gh:`578`)
- Add ``with_key`` keyword argument to :func:`mne_bids.get_entity_vals` to allow returning the full entity strings, by `Adam Li`_ (:gh:`578`)
- :func:`mne_bids.write_raw_bids` now also writes ``mne.io.Raw.annotations`` to ``*_events.tsv``, by `Adam Li`_ and `Richard Höchenberger`_ (:gh:`582`)
- BIDS conformity: The ``_part-%d`` entity is now called ``_split-`` throughout BIDS, MNE, and MNE-BIDS, by `Stefan Appelhoff`_ (:gh:`417`)

Bug fixes
^^^^^^^^^

- Fix bug in :func:`write_raw_bids` where raw.info['subject_info'] can be ``None``, by `Adam Li`_ (:gh:`392`)
- :func:`read_raw_bids` will now read all channels from ``electrodes.tsv``. Channels with coordinates ``'n/a'`` will also be included but their location set to ``np.nan`` in the ``raw`` object, by `Adam Li`_ (:gh:`393`)
- Do not change an events array passed to :func:`write_raw_bids` via the ``events_data`` keyword argument, by `Alexandre Gramfort`_ (:gh:`402`)
- Fix :func:`mne_bids.read_raw_bids` to correctly scale coordinate to meters in ``electrodes.tsv``, and also read possible iEEG coordinate frames via the 'space' BIDs-entity, by `Adam Li`_ (:gh:`390`)
- Fix coordystem reading in :func:`mne_bids.read_raw_bids` for cases where the ``acq`` is undefined, by `Stefan Appelhoff`_ (:gh:`440`)
- Calling :func:`write_raw_bids` with ``overwrite==True`` will preserve existing entries in ``participants.tsv`` and ``participants.json`` if the **new** dataset does not contain these entries, by `Adam Li`_ (:gh:`442`)
- BIDS entity ``recording`` should be represented as ``rec`` in filenames, by `Adam Li`_ (:gh:`446`)
- Fix :func:`write_raw_bids` when ``info['dig']`` is ``None``, by `Alexandre Gramfort`_ (:gh:`452`)
- :func:`mne_bids.write_raw_bids` now applies ``verbose`` to the functions that read events, by `Evgenii Kalenkovich`_ (:gh:`453`)
- Fix ``raw_to_bids`` CLI tool to work with non-FIFF files, by `Austin Hurst`_ (:gh:`456`)
- Fix :func:`mne_bids.write_raw_bids` to output BTI and CTF data in the ``scans.tsv`` according to the BIDS specification, by `Adam Li`_ (:gh:`465`)
- :func:`mne_bids.read_raw_bids` now populates the list of bad channels based on ``*_channels.tsv`` if (and only if) a ``status`` column is present, ignoring similar metadata stored in raw file (which will still be used if **no** ``status`` column is present in ``*_channels.tsv``), by `Richard Höchenberger`_ (:gh:`499`)
- Ensure that ``Raw.info['bads']`` returned by :func:`mne_bids.read_raw_bids` is always a list, by `Richard Höchenberger`_ (:gh:`501`)
- :func:`mne_bids.write_raw_bids` now ensures that **all** parts of the :class:`mne.io.Raw` instance stay in sync when using anonymization to shift dates, e.g. ``raw.annotations``, by `Richard Höchenberger`_ (:gh:`504`)
- Fix :func:`mne_bids.write_raw_bids` failed BIDS validator for ``raw.info['dig'] = []``, by `Alex Rockhill`_ (:gh:`505`)
- Ensure :func:`mne_bids.print_dir_tree` prints files and directories in alphabetical order, by `Richard Höchenberger`_ (:gh:`563`)
- :func:`mne_bids.write_raw_bids` now writes the correct coordinate system names to the JSON sidecars, by `Richard Höchenberger`_ (:gh:`585`)

API changes
^^^^^^^^^^^

In the transition to using `mne_bids.BIDSPath`, the following functions have been updated:

- :func:`mne_bids.write_anat` now accepts a :class:`mne_bids.BIDSPath` instead of entities as keyword arguments, by `Adam Li`_ (:gh:`575`)
- In :func:`mne_bids.write_raw_bids`, :func:`mne_bids.read_raw_bids`, and :func:`mne_bids.get_head_mri_trans`, the ``bids_basename`` and ``bids_root`` keyword arguments have been removed. The functions now expect ``bids_path``, an instance of :class:`mne_bids.BIDSPath`, by `Adam Li`_ (:gh:`525`)

The following functions have been removed:

- ``mne_bids.make_bids_basename`` has been removed. Use :class:`mne_bids.BIDSPath` directly, by `Adam Li`_ (:gh:`511`)
- ``mne_bids.get_matched_empty_room`` has been removed. Use :meth:`mne_bids.BIDSPath.find_empty_room` instead, by `Richard Höchenberger`_ (:gh:`421`, :gh:`535`)
- ``mne_bids.make_bids_folders`` has been removed. Use :meth:`mne_bids.BIDSPath.mkdir` instead, by `Adam Li`_ (:gh:`543`)

Further API changes:

- The functions :func:`mne_bids.write_anat`, :func:`mne_bids.make_report`, :func:`mne_bids.get_entity_vals` and :func:`mne_bids.get_datatypes` use now expect a ``root`` keyword argument instead of ``bids_root``, `Adam Li`_ (:gh:`556`)
- Added namespace :code:`mne_bids.path` which hosts path-like functionality for MNE-BIDS, by `Adam Li`_ (:gh:`483`)
- The ``datasets.py`` module was removed from ``MNE-BIDS`` and its utility was replaced by ``mne.datasets``, by `Stefan Appelhoff`_ (:gh:`471`)
- :func:`mne_bids.make_dataset_description` now accepts the argument ``overwrite``, which will reset all fields if ``True``. If ``False``, user-provided fields will no longer be overwritten by :func:`mne_bids.write_raw_bids` when its ``overwrite`` argument is ``True``, unless new values are supplied, by `Alex Rockhill`_ (:gh:`478`)
- A function for retrieval of BIDS entity values from a filename, :func:`mne_bids.get_entities_from_fname`, is now part of the public API (it used to be a private function called ``mne_bids.path._parse_bids_filename``), by `Richard Höchenberger`_ and `Adam Li`_ (:gh:`487`, :gh:`496`)
- Entity names passed to :func:`mne_bids.get_entity_vals` must now be in the "long" format, e.g. ``subject`` instead of ``sub`` etc., by `Richard Höchenberger`_ (:gh:`501`)
- It is now required to specify the Power Line Frequency to use :func:`write_raw_bids`, while in 0.4 it could be estimated, by `Alexandre Gramfort`_ and `Alex Rockhill`_ (:gh:`506`)
- Rename ``mne_bids.get_modalities`` to :func:`mne_bids.get_datatypes` for getting data types from a BIDS dataset, by `Alexandre Gramfort`_ (:gh:`253`)

.. _changes_0_4:

Version 0.4 (2020-04-04)
------------------------

Changelog
~~~~~~~~~

- Added automatic conversion of FIF to BrainVision format with warning for EEG only data and conversion to FIF for meg non-FIF data, by `Alex Rockhill`_ (:gh:`237`)
- Add possibility to pass raw readers parameters (e.g. `allow_maxshield`) to :func:`read_raw_bids` to allow reading BIDS-formatted data before applying maxfilter, by  `Sophie Herbst`_
- New feature in :code:`mne_bids.write.write_anat` for shear deface of mri, by `Alex Rockhill`_ (:gh:`271`)
- Added option to anonymize by shifting measurement date with `anonymize` parameter, in accordance with BIDS specifications, by `Alex Rockhill`_ (:gh:`280`)
- Added ``mne_bids.get_matched_empty_room`` to get empty room filename matching a data file, by `Mainak Jas`_ (:gh:`290`)
- Add ability for :func:`mne_bids.get_head_mri_trans` to read fiducial points from fif data without applying maxfilter, by `Maximilien Chaumon`_ (:gh:`291`)
- Added landmark argument to :func:`write_anat` to pass landmark location directly for deface, by `Alex Rockhill`_ (:gh:`292`)
- Standardized `bids_root` and `output_path` arguments in :func:`read_raw_bids`, :func:`write_raw_bids`, and ``make_bids_folders`` to just `bids_root`, by `Adam Li`_ (:gh:`303`)
- Add existence check for :func:`write_raw_bids` before :func:`make_dataset_description` is called, by `Adam Li`_ (:gh:`331`)
- Update scans.tsv writer to adhere to MNE-Python v0.20+ where `meas_date` is stored as a `datetime` object, by `Adam Li`_ (:gh:`344`)
- :func:`read_raw_bids` now reads in sidecar json files to set, or estimate Power Line Frequency, by `Adam Li`_ (:gh:`341`)
- Allow FIF files with multiple parts to be read using :func:`read_raw_bids`, by `Teon Brooks`_ (:gh:`346`)
- Added handedness to participant files, by `Dominik Welke`_ (:gh:`354`)
- MNE-BIDS can now handle a paths that are :class:`pathlib.Path` objects (in addition to strings), by `Richard Höchenberger`_ (:gh:`362`)
- The documentation is now available for all MNE-BIDS versions via a dropdown menu, by `Stefan Appelhoff`_ (:gh:`370`)

Bug
~~~

- Fixed broken event timing broken by conversion to BV in :func:`write_raw_bids`, by `Alex Rockhill`_ (:gh:`294`)
- Support KIT related files .elp and .hsp BIDS conversion in :func:`write_raw_bids`, by `Fu-Te Wong`_ (:gh:`323`)
- Enforce that the name arg is a required field for :func:`mne_bids.make_dataset_description`, by `Teon Brooks`_ and `Stefan Appelhoff`_ (:gh:`342`)
- Fix writing to scans.tsv file when anonymization is turned on, by `Adam Li`_ (:gh:`352`)
- Fix :func:`read_raw_bids` to properly read in sidecar json even if a similarly named copy lives inside "derivatives/" sub-folder, by `Adam Li`_  (:gh:`350`)
- Fix :func:`read_raw_bids` to properly read in events.tsv sidecar if the 'trial_type' description has a "#" character in it, by `Adam Li`_ (:gh:`355`)
- Avoid cases in which NumPy would raise a `FutureWarning` when populating TSV files, by `Richard Höchenberger`_ (:gh:`372`)
- Remove events with an unknown onset, and assume unknown event durations to be zero, when reading BIDS data via :func:`read_raw_bids`, by `Richard Höchenberger`_ (:gh:`375`)

Authors
~~~~~~~

People who contributed to this release (in alphabetical order):

* `Adam Li`_
* `Alex Rockhill`_
* `Alexandre Gramfort`_
* `Ariel Rokem`_
* `Dominik Welke`_
* `Fu-Te Wong`_
* `Mainak Jas`_
* `Maximilien Chaumon`_
* `Richard Höchenberger`_
* `Sophie Herbst`_
* `Stefan Appelhoff`_
* `Teon Brooks`_

.. _changes_0_3:

Version 0.3 (2019-12-17)
------------------------

Changelog
~~~~~~~~~

- New function ``mne_bids.get_modalities`` for getting data types from a BIDS dataset, by `Stefan Appelhoff`_ (:gh:`253`)
- New function :func:`mne_bids.get_entity_vals` allows to get a list of instances of a certain entity in a BIDS directory, by `Mainak Jas`_ and `Stefan Appelhoff`_ (:gh:`252`)
- :func:`mne_bids.print_dir_tree` now accepts an argument :code:`max_depth` which can limit the depth until which the directory tree is printed, by `Stefan Appelhoff`_ (:gh:`245`)
- New command line function exposed :code:`cp` for renaming/copying files including automatic doc generation "CLI", by `Stefan Appelhoff`_ (:gh:`225`)
- :func:`read_raw_bids` now also reads channels.tsv files accompanying a raw BIDS file and sets the channel types accordingly, by `Stefan Appelhoff`_ (:gh:`219`)
- Add example :code:`convert_mri_and_trans` for using :func:`get_head_mri_trans` and :func:`write_anat`, by `Stefan Appelhoff`_ (:gh:`211`)
- :func:`get_head_mri_trans` allows retrieving a :code:`trans` object from a BIDS dataset that contains MEG and T1 weighted MRI data, by `Stefan Appelhoff`_ (:gh:`211`)
- :func:`write_anat` allows writing T1 weighted MRI scans for subjects and optionally creating a T1w.json sidecar from a supplied :code:`trans` object, by `Stefan Appelhoff`_ (:gh:`211`)
- :func:`read_raw_bids` will return the the raw object with :code:`raw.info['bads']` already populated, whenever a :code:`channels.tsv` file is present, by `Stefan Appelhoff`_ (:gh:`209`)
- :func:`read_raw_bids` is now more likely to find event and channel sidecar json files, by `Marijn van Vliet`_ (:gh:`233`)
- Enhanced :func:`read_raw_bids` and :func:`write_raw_bids` for iEEG coordinates along with example and unit test, by `Adam Li`_ (:gh:`335`)

Bug
~~~

- Fixed bug in ``mne_bids.datasets.fetch_faces_data`` where downloading multiple subjects was impossible, by `Stefan Appelhoff`_ (:gh:`262`)
- Fixed bug where :func:`read_raw_bids` would throw a ValueError upon encountering strings in the "onset" or "duration" column of events.tsv files, by `Stefan Appelhoff`_ (:gh:`234`)
- Allow raw data from KIT systems to have two marker files specified, by `Matt Sanderson`_ (:gh:`173`)

API
~~~

- :func:`read_raw_bids` no longer optionally returns :code:`events` and :code:`event_id` but returns the raw object with :code:`mne.Annotations`, whenever an :code:`events.tsv` file is present, by `Stefan Appelhoff`_ (:gh:`209`)


Authors
~~~~~~~

People who contributed to this release (in alphabetical order):

* `Alexandre Gramfort`_
* `Mainak Jas`_
* `Marijn van Vliet`_
* `Matt Sanderson`_
* `Stefan Appelhoff`_
* `Teon Brooks`_

.. _changes_0_2:

Version 0.2 (2019-04-26)
------------------------

Changelog
~~~~~~~~~

- Add a reader for BIDS compatible raw files, by `Mainak Jas`_ (:gh:`135`)

Bug
~~~

- Normalize the length of the branches in :func:`mne_bids.print_dir_tree` by the length of the root path, leading to more adequate visual display, by `Stefan Appelhoff`_ (:gh:`192`)
- Assert a minimum required MNE-version, by `Dominik Welke`_ (:gh:`166`)
- Add function in mne_bids.utils to copy and rename CTF files :code:`mne_bids.utils.copyfile_ctf`, by `Romain Quentin`_ (:gh:`162`)
- Encoding of BrainVision .vhdr/.vmrk files is checked to prevent encoding/decoding errors when modifying, by `Dominik Welke`_ (:gh:`155`)
- The original units present in the raw data will now correctly be written to channels.tsv files for BrainVision, EEGLAB, and EDF, by `Stefan Appelhoff`_ (:gh:`125`)
- Fix logic with inferring unknown channel types for CTF data, by `Mainak Jas`_ (:gh:`129`)
- Fix the file naming for FIF files to only expose the part key-value pair when files are split, by `Teon Brooks`_ (:gh:`137`)
- Allow files with no stim channel, which could be the case for example in resting state data, by `Mainak Jas`_ (:gh:`167`)
- Better handling of unicode strings in TSV files, by `Mainak Jas`_ (:gh:`172`)
- Fix separator in scans.tsv to always be `/`, by `Matt Sanderson`_ (:gh:`176`)
- Add seeg to :code:`mne_bids.utils._handle_datatype` when determining the kind of ieeg data, by `Ezequiel Mikulan`_ (:gh:`180`)
- Fix problem in copying CTF files on Network File System due to a bug upstream in Python, by `Mainak Jas`_ (:gh:`174`)
- Fix problem in copying BTi files. Now, a utility function ensures that all the related files
  such as config and headshape are copied correctly, by `Mainak Jas`_ (:gh:`135`)
- Fix name of "sample" and "value" columns on events.tsv files, by `Ezequiel Mikulan`_ (:gh:`185`)
- Update function to copy KIT files to the `meg` directory, by `Matt Sanderson`_ (:gh:`187`)

API
~~~

- :func:`make_dataset_description` is now available from `mne_bids` main namespace, all copyfile functions are available from `mne_bids.copyfiles` namespace, by `Stefan Appelhoff`_ (:gh:`196`)
- Add support for non maxfiltered .fif files, by `Maximilien Chaumon`_ (:gh:`171`)
- Remove support for Neuroscan ``.cnt`` data because its support is no longer planned in BIDS, by `Stefan Appelhoff`_ (:gh:`142`)
- Remove support for Python 2 because it is no longer supported in MNE-Python, by `Teon Brooks`_ (:gh:`141`)
- Remove Pandas requirement to reduce number of dependencies, by `Matt Sanderson`_ (:gh:`122`)
- Use more modern API of event_from_annotations in MNE for extracting events in .vhdr and .set files, by `Mainak Jas`_ (:gh:`167`)

Authors
~~~~~~~

People who contributed to this release (in alphabetical order):

* `Alexandre Gramfort`_
* `Chris Holdgraf`_
* `Clemens Brunner`_
* `Dominik Welke`_
* `Ezequiel Mikulan`_
* `Mainak Jas`_
* `Matt Sanderson`_
* `Maximilien Chaumon`_
* `Romain Quentin`_
* `Stefan Appelhoff`_
* `Teon Brooks`_

.. _changes_0_1:

Version 0.1 (2018-11-05)
------------------------

Changelog
~~~~~~~~~

- Add example for how to rename BrainVision file triplets: `rename_brainvision_files.py`, by `Stefan Appelhoff`_ (:gh:`104`)
- Add function to fetch BrainVision testing data ``mne_bids.datasets.fetch_brainvision_testing_data``, by `Stefan Appelhoff`_ (:gh:`104`)
- Add support for EEG and a corresponding example: `make_eeg_bids.py`, by `Stefan Appelhoff`_ (:gh:`78`)
- Update :code:`mne_bids.raw_to_bids` to work for KIT and BTi systems, by `Teon Brooks`_ (:gh:`16`)
- Add support for iEEG and add ``mne_bids.make_bids_folders`` and ``mne_bids.make_bids_folders``, by `Chris Holdgraf`_ (:gh:`28` and :gh:`37`)
- Add command line interface, by `Teon Brooks`_ (:gh:`31`)
- Add :func:`mne_bids.print_dir_tree` for visualizing directory structures and restructuring package to be more
  open towards integration of other modalities (iEEG, EEG), by `Stefan Appelhoff`_ (:gh:`55`)
- Automatically generate participants.tsv, by `Matt Sanderson`_ (:gh:`70`)
- Add support for KIT marker files to be exported with raw data, by `Matt Sanderson`_ (:gh:`114`)

Bug
~~~

- Correctly handle the case when measurement date is not available, by `Mainak Jas`_ (:gh:`23`)
- Fix counting of miscellaneous channels, by `Matt Sanderson`_ (:gh:`49`)
- The source data is now copied over to the new BIDS directory. Previously this was only true for FIF data, by `Stefan Appelhoff`_ (:gh:`55`)
- Fix ordering of columns in scans.tsv, by `Matt Sanderson`_ (:gh:`68`)
- Fix bug in how artificial trigger channel STI014 is handled in channels.tsv for KIT systems, by `Matt Sanderson`_ (:gh:`72`)
- Fix channel types for KIT system in channels.tsv, by `Matt Sanderson`_ (:gh:`76`)
- Fix the way FIF files are named to satisfy the BIDS part parameters of the filename construction, `Teon Brooks`_ (:gh:`102`)
- Fix how overwriting of data is handled, by `Matt Sanderson`_ (:gh:`99`)

Authors
~~~~~~~

People who contributed to this release (in alphabetical order):

* `Alexandre Gramfort`_
* `Chris Holdgraf`_
* `Kambiz Tavabi`_
* `Mainak Jas`_
* `Matt Sanderson`_
* `Romain Quentin`_
* `Stefan Appelhoff`_
* `Teon Brooks`_


.. include:: authors.rst
