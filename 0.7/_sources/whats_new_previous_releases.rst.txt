:orphan:

.. _whats_new_in_previous_releases:

.. currentmodule:: mne_bids

What was new in previous releases?
==================================

.. currentmodule:: mne_bids
.. _changes_0_6:

Version 0.6 🎄
--------------

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
- Added a new function :func:`mne_bids.mark_bad_channels` and command line interface ``mark_bad_channels`` which allows updating of the channel status (bad, good) and description of an existing BIDS dataset, by `Richard Höchenberger`_ (:gh:`491`)
- :func:`mne_bids.read_raw_bids` correctly maps all specified ``handedness`` and ``sex`` options to MNE-Python, instead of only an incomplete subset, by `Richard Höchenberger`_ (:gh:`550`)
- :func:`mne_bids.write_raw_bids` only writes a ``README`` if it does not already exist, by `Adam Li`_ (:gh:`489`)
- Allow :func:`mne_bids.write_raw_bids` to write EEG/iEEG files from Persyst using ``mne.io.read_raw_persyst`` function, by `Adam Li`_ (:gh:`546`)
- :func:`mne_bids.print_dir_tree` now works if a ``pathlib.Path`` object is passed, by `Adam Li`_ (:gh:`555`)
- Allow writing of Elekta/Neuromag/MEGIN fine-calibration and crosstalk data via the new functions :func:`mne_bids.write_meg_calibration` and :func:`mne_bids.write_meg_crosstalk`, and retrieval of the file locations via :attr:`BIDSPath.meg_calibration_fpath` and :attr:`BIDSPath.meg_crosstalk_fpath`, by `Richard Höchenberger`_ (:gh:`562`)
- Allow :func:`mne_bids.write_raw_bids` to write EEG/iEEG files from Nihon Kohden using ``mne.io.read_raw_nihon`` function, by `Adam Li`_ (:gh:`567`)
- Allow :func:`mne_bids.get_entity_vals` to ignore datatypes using ``ignore_datatypes`` kwarg, by `Adam Li`_ (:gh:`578`)
- Add ``with_key`` keyword argument to :func:`mne_bids.get_entity_vals` to allow returning the full entity strings, by `Adam Li`_ (:gh:`578`)
- :func:`mne_bids.write_raw_bids` now also writes :attr:`mne.io.Raw.annotations` to ``*_events.tsv``, by `Adam Li`_ and `Richard Höchenberger`_ (:gh:`582`)
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

Version 0.4
-----------

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

* Alexandre Gramfort
* Chris Holdgraf
* Kambiz Tavabi
* Mainak Jas
* Matt Sanderson
* Romain Quentin
* Stefan Appelhoff
* Teon Brooks


.. include:: authors.rst
