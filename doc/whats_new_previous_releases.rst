:orphan:

.. _whats_new_in_previous_releases:

.. currentmodule:: mne_bids

What was new in previous releases?
==================================

.. contents:: Contents
   :local:
   :depth: 3

.. currentmodule:: mne_bids
.. _changes_0_5:

Version 0.5 (2020-10-22)
------------------------

This is a **big** release with lots of changes, many of them breaking existing
code. But do not fear: migration is easy, and you will **love** what we have
been cooking for you!

.. contents:: Contents
   :local:
   :depth: 3

Notable changes
~~~~~~~~~~~~~~~
- We introduce `mne_bids.BIDSPath`, a new class for all BIDS file and folder
  operations. All functions in MNE-BIDS that previously accepted filenames
  and folder locations (e.g. ``bids_root``) have been updated to work with
  ``BIDSPath``. Others have been removed.
  Consequently, you will need to update your existing code, too.
  See the `API changes`_ section for an overview of which functions have
  changed or have been removed, and follow
  :ref:`this introduction<bidspath-intro>` and our 
  to learn about the basics of ``BIDSPath``. Don't worry – it's going to be a
  breeze! 🤗

- MNE-BIDS now requires MNE-Python 0.21.

- The new function :func:`mne_bids.make_report` can help you populate a
  paragraph of your next paper's methods section!

- You can now interactively mark channels as bad using
  :func:`mne_bids.mark_bad_channels`.

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

- Introduce :class:`mne_bids.BIDSPath`, the new universal MNE-BIDS working horse for file operations, by `Adam Li`_, `Alex Rockhill`_, and `Richard Höchenberger`_ (`#496 <https://github.com/mne-tools/mne-bids/pull/496>`_, `#507 <https://github.com/mne-tools/mne-bids/pull/507>`_, `#511 <https://github.com/mne-tools/mne-bids/pull/511>`_, `#514 <https://github.com/mne-tools/mne-bids/pull/514>`_, `#542 <https://github.com/mne-tools/mne-bids/pull/542>`_)
- The new function :func:`mne_bids.make_report` and its corresponding CLI function, ``make_report``, produce human-readable summary of the BIDS dataset, by `Adam Li`_ (`#457 <https://github.com/mne-tools/mne-bids/pull/457>`_)
- :func:`read_raw_bids` now reads ``participants.tsv`` data, by `Adam Li`_ (`#392 <https://github.com/mne-tools/mne-bids/pull/392>`_)
- :func:`mne_bids.get_entity_vals` has gained ``ignore_*`` keyword arguments to exclude specific values from the list of results, e.g. the entities for a particular subject or task, by `Richard Höchenberger`_ (`#409 <https://github.com/mne-tools/mne-bids/pull/409>`_)
- :func:`mne_bids.write_raw_bids` now uses the ``space`` BIDS entity when writing iEEG electrodes and coordinate frames, by `Adam Li`_ (`#390 <https://github.com/mne-tools/mne-bids/pull/390>`_)
- :code:`convert_ieeg_to_bids` to now use sample ECoG EDF data, by `Adam Li`_ (`#390 <https://github.com/mne-tools/mne-bids/pull/390>`_)
- :func:`mne_bids.write_raw_bids` now writes ``CoordinateSystemDescription`` as specified in BIDS Specification if CoordinateSystem is MNE-compatible, by `Adam Li`_ (`#416 <https://github.com/mne-tools/mne-bids/pull/416>`_)
- :func:`mne_bids.write_raw_bids` and :func:`mne_bids.read_raw_bids` now handle scalp EEG if Captrak coordinate system and NAS/LPA/RPA landmarks are present, by `Adam Li`_ (`#416 <https://github.com/mne-tools/mne-bids/pull/416>`_)
- :func:`write_raw_bids` now adds citations to the ``README``, by `Alex Rockhill`_ (`#463 <https://github.com/mne-tools/mne-bids/pull/463>`_)
- :func:`make_dataset_description` now has an additional parameter ``dataset_type`` to set the recommended field ``DatasetType`` (defaults to ``"raw"``), by `Stefan Appelhoff`_ (`#472 <https://github.com/mne-tools/mne-bids/pull/472>`_)
- :func:`mne_bids.copyfiles.copyfile_brainvision` now has an ``anonymize`` parameter to control anonymization, by `Stefan Appelhoff`_ (`#475 <https://github.com/mne-tools/mne-bids/pull/475>`_)
- :func:`mne_bids.read_raw_bids` and :func:`mne_bids.write_raw_bids` now map respiratory (``RESP``) channel types, by `Richard Höchenberger`_ (`#482 <https://github.com/mne-tools/mne-bids/pull/482>`_)
- When impedance values are available from a ``raw.impedances`` attribute, MNE-BIDS will now write an ``impedance`` column to ``*_electrodes.tsv`` files, by `Stefan Appelhoff`_ (`#484 <https://github.com/mne-tools/mne-bids/pull/484>`_)
- :func:`mne_bids.write_raw_bids` writes out status_description with ``'n/a'`` values into the channels.tsv sidecar file, by `Adam Li`_ (`#489 <https://github.com/mne-tools/mne-bids/pull/489>`_)
- Added a new function :func:`mne_bids.mark_bad_channels` and command line interface ``mark_bad_channels`` which allows updating of the channel status (bad, good) and description of an existing BIDS dataset, by `Richard Höchenberger`_ (`#491 <https://github.com/mne-tools/mne-bids/pull/491>`_)
- :func:`mne_bids.read_raw_bids` correctly maps all specified ``handedness`` and ``sex`` options to MNE-Python, instead of only an incomplete subset, by `Richard Höchenberger`_ (`#550 <https://github.com/mne-tools/mne-bids/pull/550>`_)
- :func:`mne_bids.write_raw_bids` only writes a ``README`` if it does not already exist, by `Adam Li`_ (`#489 <https://github.com/mne-tools/mne-bids/pull/489>`_)
- Allow :func:`mne_bids.write_raw_bids` to write EEG/iEEG files from Persyst using ``mne.io.read_raw_persyst`` function, by `Adam Li`_ (`#546 <https://github.com/mne-tools/mne-bids/pull/546>`_)
- :func:`mne_bids.print_dir_tree` now works if a ``pathlib.Path`` object is passed, by `Adam Li`_ (`#555 <https://github.com/mne-tools/mne-bids/pull/555>`_)
- Allow writing of Elekta/Neuromag/MEGIN fine-calibration and crosstalk data via the new functions :func:`mne_bids.write_meg_calibration` and :func:`mne_bids.write_meg_crosstalk`, and retrieval of the file locations via :attr:`BIDSPath.meg_calibration_fpath` and :attr:`BIDSPath.meg_crosstalk_fpath`, by `Richard Höchenberger`_ (`#562 <https://github.com/mne-tools/mne-bids/pull/562>`_)
- Allow :func:`mne_bids.write_raw_bids` to write EEG/iEEG files from Nihon Kohden using ``mne.io.read_raw_nihon`` function, by `Adam Li`_ (`#567 <https://github.com/mne-tools/mne-bids/pull/567>`_)
- Allow :func:`mne_bids.get_entity_vals` to ignore datatypes using ``ignore_datatypes`` kwarg, by `Adam Li`_ (`#578 <https://github.com/mne-tools/mne-bids/pull/578>`_)
- Add ``with_key`` keyword argument to :func:`mne_bids.get_entity_vals` to allow returning the full entity strings, by `Adam Li`_ (`#578 <https://github.com/mne-tools/mne-bids/pull/578>`_)
- :func:`mne_bids.write_raw_bids` now also writes :attr:`mne.io.Raw.annotations` to ``*_events.tsv``, by `Adam Li`_ and `Richard Höchenberger`_ (`#582 <https://github.com/mne-tools/mne-bids/pull/582>`_)
- BIDS conformity: The ``_part-%d`` entity is now called ``_split-`` throughout BIDS, MNE, and MNE-BIDS, by `Stefan Appelhoff`_ (`#417 <https://github.com/mne-tools/mne-bids/pull/417>`_)

Bug fixes
^^^^^^^^^

- Fix bug in :func:`write_raw_bids` where raw.info['subject_info'] can be ``None``, by `Adam Li`_ (`#392 <https://github.com/mne-tools/mne-bids/pull/392>`_)
- :func:`read_raw_bids` will now read all channels from ``electrodes.tsv``. Channels with coordinates ``'n/a'`` will also be included but their location set to ``np.nan`` in the ``raw`` object, by `Adam Li`_ (`#393 <https://github.com/mne-tools/mne-bids/pull/393>`_)
- Do not change an events array passed to :func:`write_raw_bids` via the ``events_data`` keyword argument, by `Alexandre Gramfort`_ (`#402 <https://github.com/mne-tools/mne-bids/pull/402>`_)
- Fix :func:`mne_bids.read_raw_bids` to correctly scale coordinate to meters in ``electrodes.tsv``, and also read possible iEEG coordinate frames via the 'space' BIDs-entity by `Adam Li`_ (`#390 <https://github.com/mne-tools/mne-bids/pull/390>`_)
- Fix coordystem reading in :func:`mne_bids.read_raw_bids` for cases where the ``acq`` is undefined, by `Stefan Appelhoff`_ (`#440 <https://github.com/mne-tools/mne-bids/pull/440>`_)
- Calling :func:`write_raw_bids` with ``overwrite==True`` will preserve existing entries in ``participants.tsv`` and ``participants.json`` if the **new** dataset does not contain these entries, by `Adam Li`_ (`#442 <https://github.com/mne-tools/mne-bids/pull/442>`_)
- BIDS entity ``recording`` should be represented as ``rec`` in filenames, by `Adam Li`_ (`#446 <https://github.com/mne-tools/mne-bids/pull/446>`_)
- Fix :func:`write_raw_bids` when ``info['dig']`` is ``None``, by `Alexandre Gramfort`_ (`#452 <https://github.com/mne-tools/mne-bids/pull/452>`_)
- :func:`mne_bids.write_raw_bids` now applies ``verbose`` to the functions that read events, by `Evgenii Kalenkovich`_ (`#453 <https://github.com/mne-tools/mne-bids/pull/453>`_)
- Fix ``raw_to_bids`` CLI tool to work with non-FIFF files, by `Austin Hurst`_ (`#456 <https://github.com/mne-tools/mne-bids/pull/456>`_)
- Fix :func:`mne_bids.write_raw_bids` to output BTI and CTF data in the ``scans.tsv`` according to the BIDS specification, by `Adam Li`_ (`#465 <https://github.com/mne-tools/mne-bids/pull/465>`_)
- :func:`mne_bids.read_raw_bids` now populates the list of bad channels based on ``*_channels.tsv`` if (and only if) a ``status`` column is present, ignoring similar metadata stored in raw file (which will still be used if **no** ``status`` column is present in ``*_channels.tsv``), by `Richard Höchenberger`_ (`#499 <https://github.com/mne-tools/mne-bids/pull/499>`_)
- Ensure that ``Raw.info['bads']`` returned by :func:`mne_bids.read_raw_bids` is always a list, by `Richard Höchenberger`_ (`#501 <https://github.com/mne-tools/mne-bids/pull/501>`_)
- :func:`mne_bids.write_raw_bids` now ensures that **all** parts of the :class:`mne.io.Raw` instance stay in sync when using anonymization to shift dates, e.g. ``raw.annotations``, by `Richard Höchenberger`_ (`#504 <https://github.com/mne-tools/mne-bids/pull/504>`_)
- Fix :func:`mne_bids.write_raw_bids` failed BIDS validator for ``raw.info['dig'] = []``, by `Alex Rockhill`_ (`#505 <https://github.com/mne-tools/mne-bids/pull/505>`_)
- Ensure :func:`mne_bids.print_dir_tree` prints files and directories in alphabetical order, by `Richard Höchenberger`_ (`#563 <https://github.com/mne-tools/mne-bids/pull/563>`_)
- :func:`mne_bids.write_raw_bids` now writes the correct coordinate system names to the JSON sidecars, by `Richard Höchenberger`_ (`#585 <https://github.com/mne-tools/mne-bids/pull/585>`_)

API changes
^^^^^^^^^^^

In the transition to using `mne_bids.BIDSPath`, the following functions have been updated:

- :func:`mne_bids.write_anat` now accepts a :class:`mne_bids.BIDSPath` instead of entities as keyword arguments, by `Adam Li`_ (`#575 <https://github.com/mne-tools/mne-bids/pull/575>`_)
- In :func:`mne_bids.write_raw_bids`, :func:`mne_bids.read_raw_bids`, and :func:`mne_bids.get_head_mri_trans`, the ``bids_basename`` and ``bids_root`` keyword arguments have been removed. The functions now expect ``bids_path``, an instance of :class:`mne_bids.BIDSPath`, by `Adam Li`_ (`#525 <https://github.com/mne-tools/mne-bids/pull/525>`_)

The following functions have been removed:

- ``mne_bids.make_bids_basename`` has been removed. Use :class:`mne_bids.BIDSPath` directly, by `Adam Li`_ (`#511 <https://github.com/mne-tools/mne-bids/pull/511>`_)
- ``mne_bids.get_matched_empty_room`` has been removed. Use :meth:`mne_bids.BIDSPath.find_empty_room` instead, by `Richard Höchenberger`_ (`#421 <https://github.com/mne-tools/mne-bids/pull/421>`_, `#535 <https://github.com/mne-tools/mne-bids/pull/535>`_)
- ``mne_bids.make_bids_folders`` has been removed. Use :meth:`mne_bids.BIDSPath.mkdir` instead, by `Adam Li`_ (`#543 <https://github.com/mne-tools/mne-bids/pull/543>`_)

Further API changes:

- The functions :func:`mne_bids.write_anat`, :func:`mne_bids.make_report`, :func:`mne_bids.get_entity_vals` and :func:`mne_bids.get_datatypes` use now expect a ``root`` keyword argument instead of ``bids_root``, `Adam Li`_ (`#556 <https://github.com/mne-tools/mne-bids/pull/556>`_)
- Added namespace :code:`mne_bids.path` which hosts path-like functionality for MNE-BIDS, by `Adam Li`_ (`#483 <https://github.com/mne-tools/mne-bids/pull/483>`_)
- The ``datasets.py`` module was removed from ``MNE-BIDS`` and its utility was replaced by ``mne.datasets``, by `Stefan Appelhoff`_ (`#471 <https://github.com/mne-tools/mne-bids/pull/471>`_)
- :func:`mne_bids.make_dataset_description` now accepts the argument ``overwrite``, which will reset all fields if ``True``. If ``False``, user-provided fields will no longer be overwritten by :func:`mne_bids.write_raw_bids` when its ``overwrite`` argument is ``True``, unless new values are supplied, by `Alex Rockhill`_ (`#478 <https://github.com/mne-tools/mne-bids/pull/478>`_)
- A function for retrieval of BIDS entity values from a filename, :func:`mne_bids.get_entities_from_fname`, is now part of the public API (it used to be a private function called ``mne_bids.path._parse_bids_filename``), by `Richard Höchenberger`_ and `Adam Li`_ (`#487 <https://github.com/mne-tools/mne-bids/pull/487>`_, `#496 <https://github.com/mne-tools/mne-bids/pull/496>`_)
- Entity names passed to :func:`mne_bids.get_entity_vals` must now be in the "long" format, e.g. ``subject`` instead of ``sub`` etc., by `Richard Höchenberger`_ (`#501 <https://github.com/mne-tools/mne-bids/pull/501>`_)
- It is now required to specify the Power Line Frequency to use :func:`write_raw_bids`, while in 0.4 it could be estimated, by `Alexandre Gramfort`_ and `Alex Rockhill`_ (`#506 <https://github.com/mne-tools/mne-bids/pull/506>`_)
- Rename ``mne_bids.get_modalities`` to :func:`mne_bids.get_datatypes` for getting data types from a BIDS dataset, by `Alexandre Gramfort`_ (`#253 <https://github.com/mne-tools/mne-bids/pull/253>`_)

.. _changes_0_4:

Version 0.4
-----------

Changelog
~~~~~~~~~

- Added automatic conversion of FIF to BrainVision format with warning for EEG only data and conversion to FIF for meg non-FIF data, by `Alex Rockhill`_ (`#237 <https://github.com/mne-tools/mne-bids/pull/237>`_)
- Add possibility to pass raw readers parameters (e.g. `allow_maxshield`) to :func:`read_raw_bids` to allow reading BIDS-formatted data before applying maxfilter, by  `Sophie Herbst`_
- New feature in :code:`mne_bids.write.write_anat` for shear deface of mri, by `Alex Rockhill`_ (`#271 <https://github.com/mne-tools/mne-bids/pull/271>`_)
- Added option to anonymize by shifting measurement date with `anonymize` parameter, in accordance with BIDS specifications, by `Alex Rockhill`_ (`#280 <https://github.com/mne-tools/mne-bids/pull/280>`_)
- Added ``mne_bids.get_matched_empty_room`` to get empty room filename matching a data file, by `Mainak Jas`_ (`#290 <https://github.com/mne-tools/mne-bids/pull/290>`_)
- Add ability for :func:`mne_bids.get_head_mri_trans` to read fiducial points from fif data without applying maxfilter, by `Maximilien Chaumon`_ (`#291 <https://github.com/mne-tools/mne-bids/pull/291>`_)
- Added landmark argument to :func:`write_anat` to pass landmark location directly for deface, by `Alex Rockhill`_ (`#292 <https://github.com/mne-tools/mne-bids/pull/292>`_)
- Standardized `bids_root` and `output_path` arguments in :func:`read_raw_bids`, :func:`write_raw_bids`, and ``make_bids_folders`` to just `bids_root`, by `Adam Li`_ (`#303 <https://github.com/mne-tools/mne-bids/pull/303>`_)
- Add existence check for :func:`write_raw_bids` before :func:`make_dataset_description` is called, by `Adam Li`_ (`#331 <https://github.com/mne-tools/mne-bids/pull/331>`_)
- Update scans.tsv writer to adhere to MNE-Python v0.20+ where `meas_date` is stored as a `datetime` object, by `Adam Li`_ (`#344 <https://github.com/mne-tools/mne-bids/pull/344>`_)
- :func:`read_raw_bids` now reads in sidecar json files to set, or estimate Power Line Frequency, by `Adam Li`_ (`#341 <https://github.com/mne-tools/mne-bids/pull/341>`_)
- Allow FIF files with multiple parts to be read using :func:`read_raw_bids`, by `Teon Brooks`_ (`#346 <https://github.com/mne-tools/mne-bids/pull/346>`_)
- Added handedness to participant files, by `Dominik Welke`_ (`#354 <https://github.com/mne-tools/mne-bids/pull/354>`_)
- MNE-BIDS can now handle a paths that are :class:`pathlib.Path` objects (in addition to strings), by `Richard Höchenberger`_ (`#362 <https://github.com/mne-tools/mne-bids/pull/362>`_)
- The documentation is now available for all MNE-BIDS versions via a dropdown menu, by `Stefan Appelhoff`_ (`#370 <https://github.com/mne-tools/mne-bids/pull/370>`_)

Bug
~~~

- Fixed broken event timing broken by conversion to BV in :func:`write_raw_bids`, by `Alex Rockhill`_ (`#294 <https://github.com/mne-tools/mne-bids/pull/294>`_)
- Support KIT related files .elp and .hsp BIDS conversion in :func:`write_raw_bids`, by `Fu-Te Wong`_ (`#323 <https://github.com/mne-tools/mne-bids/pull/323>`_)
- Enforce that the name arg is a required field for :func:`mne_bids.make_dataset_description`, by `Teon Brooks`_ and `Stefan Appelhoff`_ (`#342 <https://github.com/mne-tools/mne-bids/pull/342>`_)
- Fix writing to scans.tsv file when anonymization is turned on, by `Adam Li`_ (`#352 <https://github.com/mne-tools/mne-bids/pull/352>`_)
- Fix :func:`read_raw_bids` to properly read in sidecar json even if a similarly named copy lives inside "derivatives/" sub-folder, by `Adam Li`_  (`#350 <https://github.com/mne-tools/mne-bids/pull/350>`_)
- Fix :func:`read_raw_bids` to properly read in events.tsv sidecar if the 'trial_type' description has a "#" character in it by `Adam Li`_ (`#355 <https://github.com/mne-tools/mne-bids/pull/355>`_)
- Avoid cases in which NumPy would raise a `FutureWarning` when populating TSV files, by `Richard Höchenberger`_ (`#372 <https://github.com/mne-tools/mne-bids/pull/372>`_)
- Remove events with an unknown onset, and assume unknown event durations to be zero, when reading BIDS data via :func:`read_raw_bids`, by `Richard Höchenberger`_ (`#375 <https://github.com/mne-tools/mne-bids/pull/375>`_)

Authors
~~~~~~~

People who contributed to this release (in alphabetical order):

* Adam Li
* Alex Rockhill
* Alexandre Gramfort
* Ariel Rokem
* Dominik Welke
* Fu-Te Wong
* Mainak Jas
* Maximilien Chaumon
* Richard Höchenberger
* Sophie Herbst
* Stefan Appelhoff
* Teon Brooks

.. _changes_0_3:

Version 0.3
-----------

Changelog
~~~~~~~~~

- New function ``mne_bids.get_modalities`` for getting data types from a BIDS dataset, by `Stefan Appelhoff`_ (`#253 <https://github.com/mne-tools/mne-bids/pull/253>`_)
- New function :func:`mne_bids.get_entity_vals` allows to get a list of instances of a certain entity in a BIDS directory, by `Mainak Jas`_ and `Stefan Appelhoff`_ (`#252 <https://github.com/mne-tools/mne-bids/pull/252>`_)
- :func:`mne_bids.print_dir_tree` now accepts an argument :code:`max_depth` which can limit the depth until which the directory tree is printed, by `Stefan Appelhoff`_ (`#245 <https://github.com/mne-tools/mne-bids/pull/245>`_)
- New command line function exposed :code:`cp` for renaming/copying files including automatic doc generation "CLI", by `Stefan Appelhoff`_ (`#225 <https://github.com/mne-tools/mne-bids/pull/225>`_)
- :func:`read_raw_bids` now also reads channels.tsv files accompanying a raw BIDS file and sets the channel types accordingly, by `Stefan Appelhoff`_ (`#219 <https://github.com/mne-tools/mne-bids/pull/219>`_)
- Add example :code:`convert_mri_and_trans` for using :func:`get_head_mri_trans` and :func:`write_anat`, by `Stefan Appelhoff`_ (`#211 <https://github.com/mne-tools/mne-bids/pull/211>`_)
- :func:`get_head_mri_trans` allows retrieving a :code:`trans` object from a BIDS dataset that contains MEG and T1 weighted MRI data, by `Stefan Appelhoff`_ (`#211 <https://github.com/mne-tools/mne-bids/pull/211>`_)
- :func:`write_anat` allows writing T1 weighted MRI scans for subjects and optionally creating a T1w.json sidecar from a supplied :code:`trans` object, by `Stefan Appelhoff`_ (`#211 <https://github.com/mne-tools/mne-bids/pull/211>`_)
- :func:`read_raw_bids` will return the the raw object with :code:`raw.info['bads']` already populated, whenever a :code:`channels.tsv` file is present, by `Stefan Appelhoff`_ (`#209 <https://github.com/mne-tools/mne-bids/pull/209>`_)
- :func:`read_raw_bids` is now more likely to find event and channel sidecar json files, by `Marijn van Vliet`_ (`#233 <https://github.com/mne-tools/mne-bids/pull/233>`_)
- Enhanced :func:`read_raw_bids` and :func:`write_raw_bids` for iEEG coordinates along with example and unit test, by `Adam Li`_ (`#335 <https://github.com/mne-tools/mne-bids/pull/335/>`_)

Bug
~~~

- Fixed bug in ``mne_bids.datasets.fetch_faces_data`` where downloading multiple subjects was impossible, by `Stefan Appelhoff`_ (`#262 <https://github.com/mne-tools/mne-bids/pull/262>`_)
- Fixed bug where :func:`read_raw_bids` would throw a ValueError upon encountering strings in the "onset" or "duration" column of events.tsv files, by `Stefan Appelhoff`_ (`#234 <https://github.com/mne-tools/mne-bids/pull/234>`_)
- Allow raw data from KIT systems to have two marker files specified, by `Matt Sanderson`_ (`#173 <https://github.com/mne-tools/mne-bids/pull/173>`_)

API
~~~

- :func:`read_raw_bids` no longer optionally returns :code:`events` and :code:`event_id` but returns the raw object with :code:`mne.Annotations`, whenever an :code:`events.tsv` file is present, by `Stefan Appelhoff`_ (`#209 <https://github.com/mne-tools/mne-bids/pull/209>`_)


Authors
~~~~~~~

People who contributed to this release (in alphabetical order):

* Alexandre Gramfort
* Mainak Jas
* Marijn van Vliet
* Matt Sanderson
* Stefan Appelhoff
* Teon Brooks

.. _changes_0_2:

Version 0.2
-----------

Changelog
~~~~~~~~~

- Add a reader for BIDS compatible raw files, by `Mainak Jas`_ (`#135 <https://github.com/mne-tools/mne-bids/pull/135>`_)

Bug
~~~

- Normalize the length of the branches in :func:`mne_bids.print_dir_tree` by the length of the root path, leading to more adequate visual display, by `Stefan Appelhoff`_ (`#192 <https://github.com/mne-tools/mne-bids/pull/192>`_)
- Assert a minimum required MNE-version, by `Dominik Welke`_ (`#166 <https://github.com/mne-tools/mne-bids/pull/166>`_)
- Add function in mne_bids.utils to copy and rename CTF files :code:`mne_bids.utils.copyfile_ctf`, by `Romain Quentin`_ (`#162 <https://github.com/mne-tools/mne-bids/pull/162>`_)
- Encoding of BrainVision .vhdr/.vmrk files is checked to prevent encoding/decoding errors when modifying, by `Dominik Welke`_ (`#155 <https://github.com/mne-tools/mne-bids/pull/155>`_)
- The original units present in the raw data will now correctly be written to channels.tsv files for BrainVision, EEGLAB, and EDF, by `Stefan Appelhoff`_ (`#125 <https://github.com/mne-tools/mne-bids/pull/125>`_)
- Fix logic with inferring unknown channel types for CTF data, by `Mainak Jas`_ (`#129 <https://github.com/mne-tools/mne-bids/pull/16>`_)
- Fix the file naming for FIF files to only expose the part key-value pair when files are split, by `Teon Brooks`_ (`#137 <https://github.com/mne-tools/mne-bids/pull/137>`_)
- Allow files with no stim channel, which could be the case for example in resting state data, by `Mainak Jas`_ (`#167 <https://github.com/mne-tools/mne-bids/pull/167/files>`_)
- Better handling of unicode strings in TSV files, by `Mainak Jas`_ (`#172 <https://github.com/mne-tools/mne-bids/pull/172/files>`_)
- Fix separator in scans.tsv to always be `/`, by `Matt Sanderson`_ (`#176 <https://github.com/mne-tools/mne-bids/pull/176>`_)
- Add seeg to :code:`mne_bids.utils._handle_datatype` when determining the kind of ieeg data, by `Ezequiel Mikulan`_ (`#180 <https://github.com/mne-tools/mne-bids/pull/180/files>`_)
- Fix problem in copying CTF files on Network File System due to a bug upstream in Python by `Mainak Jas`_ (`#174 <https://github.com/mne-tools/mne-bids/pull/174/files>`_)
- Fix problem in copying BTi files. Now, a utility function ensures that all the related files
  such as config and headshape are copied correctly, by `Mainak Jas`_ (`#135 <https://github.com/mne-tools/mne-bids/pull/135>`_)
- Fix name of "sample" and "value" columns on events.tsv files, by `Ezequiel Mikulan`_ (`#185 <https://github.com/mne-tools/mne-bids/pull/185>`_)
- Update function to copy KIT files to the `meg` directory, by `Matt Sanderson`_ (`#187 <https://github.com/mne-tools/mne-bids/pull/187>`_)

API
~~~

- :func:`make_dataset_description` is now available from `mne_bids` main namespace, all copyfile functions are available from `mne_bids.copyfiles` namespace by `Stefan Appelhoff`_ (`#196 <https://github.com/mne-tools/mne-bids/pull/196>`_)
- Add support for non maxfiltered .fif files, by `Maximilien Chaumon`_ (`#171 <https://github.com/mne-tools/mne-bids/pull/171>`_)
- Remove support for Neuroscan ``.cnt`` data because its support is no longer planned in BIDS, by `Stefan Appelhoff`_ (`#142 <https://github.com/mne-tools/mne-bids/pull/142>`_)
- Remove support for Python 2 because it is no longer supported in MNE-Python, by `Teon Brooks`_ (`#141 <https://github.com/mne-tools/mne-bids/pull/141>`_)
- Remove Pandas requirement to reduce number of dependencies, by `Matt Sanderson`_ (`#122 <https://github.com/mne-tools/mne-bids/pull/122>`_)
- Use more modern API of event_from_annotations in MNE for extracting events in .vhdr and .set files, by `Mainak Jas`_ (`#167 <https://github.com/mne-tools/mne-bids/pull/167/files>`_)

Authors
~~~~~~~

People who contributed to this release (in alphabetical order):

* Alexandre Gramfort
* Chris Holdgraf
* Clemens Brunner
* Dominik Welke
* Ezequiel Mikulan
* Mainak Jas
* Matt Sanderson
* Maximilien Chaumon
* Romain Quentin
* Stefan Appelhoff
* Teon Brooks

.. _changes_0_1:

Version 0.1
-----------

Changelog
~~~~~~~~~

- Add example for how to rename BrainVision file triplets: `rename_brainvision_files.py` by `Stefan Appelhoff`_ (`#104 <https://github.com/mne-tools/mne-bids/pull/104>`_)
- Add function to fetch BrainVision testing data ``mne_bids.datasets.fetch_brainvision_testing_data`` by `Stefan Appelhoff`_ (`#104 <https://github.com/mne-tools/mne-bids/pull/104>`_)
- Add support for EEG and a corresponding example: `make_eeg_bids.py` by `Stefan Appelhoff`_ (`#78 <https://github.com/mne-tools/mne-bids/pull/78>`_)
- Update :code:`mne_bids.raw_to_bids` to work for KIT and BTi systems, by `Teon Brooks`_ (`#16 <https://github.com/mne-tools/mne-bids/pull/16>`_)
- Add support for iEEG and add ``mne_bids.make_bids_folders`` and ``mne_bids.make_bids_folders``, by `Chris Holdgraf`_ (`#28 <https://github.com/mne-tools/mne-bids/pull/28>`_ and `#37 <https://github.com/mne-tools/mne-bids/pull/37>`_)
- Add command line interface by `Teon Brooks`_ (`#31 <https://github.com/mne-tools/mne-bids/pull/31>`_)
- Add :func:`mne_bids.print_dir_tree` for visualizing directory structures and restructuring package to be more
  open towards integration of other modalities (iEEG, EEG), by `Stefan Appelhoff`_ (`#55 <https://github.com/mne-tools/mne-bids/pull/55>`_)
- Automatically generate participants.tsv, by `Matt Sanderson`_ (`#70 <https://github.com/mne-tools/mne-bids/pull/70>`_)
- Add support for KIT marker files to be exported with raw data, by `Matt Sanderson`_ (`#114 <https://github.com/mne-tools/mne-bids/pull/114>`_)

Bug
~~~

- Correctly handle the case when measurement date is not available, by `Mainak Jas`_ (`#23 <https://github.com/mne-tools/mne-bids/pull/23>`_)
- Fix counting of miscellaneous channels, by `Matt Sanderson`_ (`#49 <https://github.com/mne-tools/mne-bids/pull/49>`_)
- The source data is now copied over to the new BIDS directory. Previously this was only true for FIF data, by `Stefan Appelhoff`_ (`#55 <https://github.com/mne-tools/mne-bids/pull/55>`_)
- Fix ordering of columns in scans.tsv, by `Matt Sanderson`_ (`#68 <https://github.com/mne-tools/mne-bids/pull/68>`_)
- Fix bug in how artificial trigger channel STI014 is handled in channels.tsv for KIT systems, by `Matt Sanderson`_ (`#72 <https://github.com/mne-tools/mne-bids/pull/72>`_)
- Fix channel types for KIT system in channels.tsv, by `Matt Sanderson`_ (`#76 <https://github.com/mne-tools/mne-bids/pull/76>`_)
- Fix the way FIF files are named to satisfy the BIDS part parameters of the filename construction, `Teon Brooks`_ (`#102 <https://github.com/mne-tools/mne-bids/pull/102>`_)
- Fix how overwriting of data is handled, by `Matt Sanderson`_ (`#99 <https://github.com/mne-tools/mne-bids/pull/99>`_)

Authors
~~~~~~~

People who contributed to this release (in alphabetical order):

* Alexandre Gramfort
* Chris Holdgraf
* Kambiz Tavabi
* Mainak Jas
* Matt Sanderson
* Romain Quentin
* Stefan Appelhoff
* Teon Brooks


.. include:: authors.rst
