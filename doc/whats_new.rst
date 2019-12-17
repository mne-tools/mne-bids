:orphan:

.. _whats_new:


What's new?
===========

Here we list a changelog of MNE-BIDS.

.. contents:: Contents
   :local:
   :depth: 3

.. currentmodule:: mne_bids

.. _current:

Current
-------

Changelog
~~~~~~~~~

- Standardized `bids_root` and `output_path` arguments in :func:`read_raw_bids`, :func:`write_raw_bids`, and :func:`make_bids_folder` to just `bids_root`, by `Adam Li`_ (`#303 <https://github.com/mne-tools/mne-bids/pull/303>`_)
- Added landmark argument to :func:`write_anat` to pass landmark location directly for deface, by `Alex Rockhill`_ (`#292 <https://github.com/mne-tools/mne-bids/pull/292>`_)
- Added option to anonymize by shifting measurement date with `anonymize` parameter, in accordance with BIDS specifications, by `Alex Rockhill`_ (`#280 <https://github.com/mne-tools/mne-bids/pull/280>`_)
- Added automatic conversion of FIF to BrainVision format with warning for EEG only data and conversion to FIF for meg non-FIF data, by `Alex Rockhill`_ (`#237 <https://github.com/mne-tools/mne-bids/pull/237>`_)
- Add possibility to pass raw readers parameters (e.g. `allow_maxshield`) to :func:`read_raw_bids` to allow reading BIDS-formatted data before applying maxfilter, by  `Sophie Herbst`_
- New feature in :func`mne_bids.write.write_anat` for shear deface of mri, by `Alex Rockhill`_ (`#271 <https://github.com/mne-tools/mne-bids/pull/271>`_)
- Add ability for :func:`mne_bids.get_head_mri_trans` to read fiducial points from fif data without applying maxfilter, by `Maximilien Chaumon`_

Bug
~~~

- Fixed broken event timing broken by conversion to BV in :func:`write_raw_bids`, by `Alex Rockhill`_ (`#294 <https://github.com/mne-tools/mne-bids/pull/294>`_)

- Support KIT related files .elp and .hsp BIDS conversion in :func:`write_raw_bids`, by `foucault`_ (`#323 <https://github.com/mne-tools/mne-bids/pull/323>`_)

API
~~~

.. _changes_0_3:

Version 0.3
-----------

Changelog
~~~~~~~~~

- New function :func:`mne_bids.utils.get_kinds` for getting data types from a BIDS dataset, by `Stefan Appelhoff`_ (`#253 <https://github.com/mne-tools/mne-bids/pull/253>`_)
- New function :func:`mne_bids.utils.get_entity_vals` allows to get a list of instances of a certain entity in a BIDS directory, by `Mainak Jas`_ and `Stefan Appelhoff`_ (`#252 <https://github.com/mne-tools/mne-bids/pull/252>`_)
- :func:`mne_bids.utils.print_dir_tree` now accepts an argument :code:`max_depth` which can limit the depth until which the directory tree is printed, by `Stefan Appelhoff`_ (`#245 <https://github.com/mne-tools/mne-bids/pull/245>`_)
- New command line function exposed :code:`cp` for renaming/copying files including automatic doc generation "CLI", by `Stefan Appelhoff`_ (`#225 <https://github.com/mne-tools/mne-bids/pull/225>`_)
- :func:`read_raw_bids` now also reads channels.tsv files accompanying a raw BIDS file and sets the channel types accordingly, by `Stefan Appelhoff`_ (`#219 <https://github.com/mne-tools/mne-bids/pull/219>`_)
- Add example :code:`convert_mri_and_trans` for using :func:`get_head_mri_trans` and :func:`write_anat`, by `Stefan Appelhoff`_ (`#211 <https://github.com/mne-tools/mne-bids/pull/211>`_)
- :func:`get_head_mri_trans` allows retrieving a :code:`trans` object from a BIDS dataset that contains MEG and T1 weighted MRI data, by `Stefan Appelhoff`_ (`#211 <https://github.com/mne-tools/mne-bids/pull/211>`_)
- :func:`write_anat` allows writing T1 weighted MRI scans for subjects and optionally creating a T1w.json sidecar from a supplied :code:`trans` object, by `Stefan Appelhoff`_ (`#211 <https://github.com/mne-tools/mne-bids/pull/211>`_)
- :func:`read_raw_bids` will return the the raw object with :code:`raw.info['bads']` already populated, whenever a :code:`channels.tsv` file is present, by `Stefan Appelhoff`_ (`#209 <https://github.com/mne-tools/mne-bids/pull/209>`_)
- :func:`read_raw_bids` is now more likely to find event and channel sidecar json files, by `Marijn van Vliet`_ (`#233 <https://github.com/mne-tools/mne-bids/pull/233>`_)

Bug
~~~

- Fixed bug in :func:`mne_bids.datasets.fetch_faces_data` where downloading multiple subjects was impossible, by `Stefan Appelhoff`_ (`#262 <https://github.com/mne-tools/mne-bids/pull/262>`_)
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

- Normalize the length of the branches in :func:`print_dir_tree` by the length of the root path, leading to more adequate visual display, by `Stefan Appelhoff`_ (`#192 <https://github.com/mne-tools/mne-bids/pull/192>`_)
- Assert a minimum required MNE-version, by `Dominik Welke`_ (`#166 <https://github.com/mne-tools/mne-bids/pull/166>`_)
- Add function in mne_bids.utils to copy and rename CTF files :func:`mne_bids.utils.copyfile_ctf`, by `Romain Quentin`_ (`#162 <https://github.com/mne-tools/mne-bids/pull/162>`_)
- Encoding of BrainVision .vhdr/.vmrk files is checked to prevent encoding/decoding errors when modifying, by `Dominik Welke`_ (`#155 <https://github.com/mne-tools/mne-bids/pull/155>`_)
- The original units present in the raw data will now correctly be written to channels.tsv files for BrainVision, EEGLAB, and EDF, by `Stefan Appelhoff`_ (`#125 <https://github.com/mne-tools/mne-bids/pull/125>`_)
- Fix logic with inferring unknown channel types for CTF data, by `Mainak Jas`_ (`#129 <https://github.com/mne-tools/mne-bids/pull/16>`_)
- Fix the file naming for FIF files to only expose the part key-value pair when files are split, by `Teon Brooks`_ (`#137 <https://github.com/mne-tools/mne-bids/pull/137>`_)
- Allow files with no stim channel, which could be the case for example in resting state data, by `Mainak Jas`_ (`#167 <https://github.com/mne-tools/mne-bids/pull/167/files>`_)
- Better handling of unicode strings in TSV files, by `Mainak Jas`_ (`#172 <https://github.com/mne-tools/mne-bids/pull/172/files>`_)
- Fix separator in scans.tsv to always be `/`, by `Matt Sanderson`_ (`#176 <https://github.com/mne-tools/mne-bids/pull/176>`_)
- Add seeg to :func:`mne_bids.utils._handle_kind` when determining the kind of ieeg data, by `Ezequiel Mikulan`_ (`#180 <https://github.com/mne-tools/mne-bids/pull/180/files>`_)
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
- Add function to fetch BrainVision testing data :func:`mne_bids.datasets.fetch_brainvision_testing_data` `Stefan Appelhoff`_ (`#104 <https://github.com/mne-tools/mne-bids/pull/104>`_)
- Add support for EEG and a corresponding example: `make_eeg_bids.py` by `Stefan Appelhoff`_ (`#78 <https://github.com/mne-tools/mne-bids/pull/78>`_)
- Update :func:`mne_bids.raw_to_bids` to work for KIT and BTi systems, by `Teon Brooks`_ (`#16 <https://github.com/mne-tools/mne-bids/pull/16>`_)
- Add support for iEEG and add :func:`mne_bids.make_bids_folders` and :func:`mne_bids.make_bids_folders`, by `Chris Holdgraf`_ (`#28 <https://github.com/mne-tools/mne-bids/pull/28>`_ and `#37 <https://github.com/mne-tools/mne-bids/pull/37>`_)
- Add command line interface by `Teon Brooks`_ (`#31 <https://github.com/mne-tools/mne-bids/pull/31>`_)
- Add :func:`mne_bids.utils.print_dir_tree` for visualizing directory structures and restructuring package to be more
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

.. _Mainak Jas: https://perso.telecom-paristech.fr/mjas/
.. _Teon Brooks: https://teonbrooks.github.io/
.. _Chris Holdgraf: https://bids.berkeley.edu/people/chris-holdgraf
.. _Matt Sanderson: https://github.com/monkeyman192
.. _Stefan Appelhoff: http://stefanappelhoff.com/
.. _Romain Quentin: https://github.com/romquentin
.. _Dominik Welke: https://github.com/dominikwelke
.. _Maximilien Chaumon: https://github.com/dnacombo
.. _Ezequiel Mikulan: https://github.com/ezemikulan
.. _Marijn van Vliet: https://github.com/wmvanvliet
.. _Alex Rockhill: http://github.com/alexrockhill
.. _Sophie Herbst: http://github.com/SophieHerbst
.. _Adam Li: https://github.com/adam2392
