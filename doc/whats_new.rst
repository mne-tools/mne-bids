:orphan:

.. _whats_new:

.. currentmodule:: mne_bids

What's new?
===========

.. currentmodule:: mne_bids

.. _changes_0_5:

Version 0.5 (2020-10-xx)
------------------------

This is a **big** release with lots of changes, many of them breaking existing
code. But do not fear: migration is easy, and you will **love** what we have
been cooking for you!

.. contents:: Contents
   :local:
   :depth: 3

Notable changes
~~~~~~~~~~~~~~~
xxx

Authors
~~~~~~~
The following people have contributed to this release of MNE-BIDS:

- `Adam Li`_
- `Alexandre Gramfort`_
- `Alex Rockhill`_
- `Richard Höchenberger`_ 
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
- ``mne_bids.get_matched_empty_room`` now implements an algorithm for discovering empty-room recordings that do not have the recording date set as their session, by `Richard Höchenberger`_ (`#421 <https://github.com/mne-tools/mne-bids/pull/421>`_)
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
- ``mne_bids.get_matched_empty_room`` has been removed. Use :meth:`mne_bids.BIDSPath.find_empty_room` instead, by `Richard Höchenberger`_ (`#535 <https://github.com/mne-tools/mne-bids/pull/535>`_)
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


:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
