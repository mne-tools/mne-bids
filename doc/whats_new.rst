:orphan:

.. _whats_new:

.. currentmodule:: mne_bids

What's new?
===========

.. _changes_0_9:

Version 0.9 (unreleased)
------------------------

...

Notable changes
~~~~~~~~~~~~~~~

- ...

Authors
~~~~~~~

* `Alex Rockhill`_
* `Richard Höchenberger`_
* `Mainak Jas`_
* `Adam Li`_
* `Stefan Appelhoff`_
* `Franziska von Albedyll`_

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

- The new function :func:`mne_bids.anonymize_dataset` can be used to anonymize an entire BIDS dataset, by `Richard Höchenberger`_ (:gh:`893`)

- :meth:`mne_bids.BIDSPath.find_empty_room` gained a new parameter ``use_sidecar_only`` to limit empty-room search to the metadata stored in the sidecar files, by `Richard Höchenberger`_ (:gh:`893`)

- :meth:`mne_bids.BIDSPath.find_empty_room` gained a new parameter ``verbose`` to limit verbosity of the output, by `Richard Höchenberger`_ (:gh:`893`)

API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^

- ``mne_bids.mark_bad_channels`` deprecated in favor of :func:`mne_bids.mark_channels`, which allows specifying the status to change channels to by `Adam Li`_ (:gh:`882`)

- :func:`mne_bids.get_entities_from_fname` does not return ``suffix`` anymore as that is not considered a BIDS entity, by `Adam Li`_ (:gh:`883`)

- Reading BIDS data with ``"HeadCoilFrequency"`` and ``"PowerLineFrequency"`` data specified in JSON sidecars will only "warn" in case of mismatches between Raw and JSON data, by `Franziska von Albedyll`_ (:gh:`885`)

- Accessing :attr:`mne_bids.BIDSPath.fpath` emit a warning anymore if the path does not exist. This behavior was unreliable and yielded confusing error messages in certain use cases. Use `mne_bids.BIDSPath.fpath.exists()` to check whether the path exists in the file system, by `Richard Höchenberger`_ (:gh:`904`)

- :func:`mne_bids.get_entity_vals` gained a new parameter, ``ignore_dirs``, to exclude directories from the search, by `Adam Li`_ and `Richard Höchenberger`_ (:gh:`899`, :gh:`908`)

Requirements
^^^^^^^^^^^^

- Writing BrainVision files now requires ``pybv`` version 0.6, by `Stefan Appelhoff`_ (:gh:`880`)


Bug fixes
^^^^^^^^^

- Fix writing Ricoh/KIT data that comes without an associated ``.mrk``, ``.elp``, or ``.hsp`` file using :func:`mne_bids.write_raw_bids`, by `Richard Höchenberger`_ (:gh:`850`)

- Properly support CTF MEG data with 2nd-order gradient compensation, by `Mainak Jas`_ (:gh:`858`)

- Fix writing and reading EDF files with upper-case extension (``.EDF``), by `Adam Li`_ (:gh:`868`)

- Fix reading of EDF files with lower/upper case extension, by `Adam Li`_ (:gh:`875`)

- Fix reading of TSV files with only a single column, by `Marijn van Vliet`_ (:gh:`886`)

- Fix erroneous measurement date check in :func:`mne_bids.write_raw_bids` when requesting to anonymize empty-room data, by `Richard Höchenberger`_ (:gh:`893`)

- More robust handling of situations where :func:`mne_bids.read_raw_bids` tries to read a file that does not exist, by `Richard Höchenberger`_ (:gh:`904`)


:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
