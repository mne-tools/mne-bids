:orphan:

.. _whats_new:

What's new?
===========

.. currentmodule:: mne_bids

.. _current:

Current
-------

Changelog
~~~~~~~~~

Bug
~~~

- The original units present in the raw data will now correctly be written to channels.tsv files for BrainVision, EEGLAB, and EDF, by `Stefan Appelhoff`_ (`#125 <https://github.com/mne-tools/mne-bids/pull/125>`_)
- Fix logic with inferring unknown channel types for CTF data, by `Mainak Jas`_ (`#129 <https://github.com/mne-tools/mne-bids/pull/16>`_)
- Fix the file naming for FIF files to only expose the part key-value pair when files are split, by `Teon Brooks`_ (`#137 <https://github.com/mne-tools/mne-bids/pull/137>`_)

API
~~~

- Remove support for Neuroscan ``.cnt`` data because its support is no longer planned in BIDS, by `Stefan Appelhoff`_ (`#142 <https://github.com/mne-tools/mne-bids/pull/142>`_)
- Remove support for Python 2 because it is no longer supported in MNE-Python, by `Teon Brooks`_ (`#141 <https://github.com/mne-tools/mne-bids/pull/141>`_)

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

People who contributed to this release  (in alphabetical order):

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
