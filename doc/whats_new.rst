:orphan:

.. _whats_new:

.. currentmodule:: mne_bids

What's new?
===========

.. currentmodule:: mne_bids
.. _changes_0_8:

Version 0.8 (unreleased)
------------------------

...

Notable changes
~~~~~~~~~~~~~~~

- ...

Authors
~~~~~~~

* `Alex Rockhill`_
* `Richard Höchenberger`_
* `Adam Li`_
* `Richard Köhler`_ (new contributor)

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

Enhancements
^^^^^^^^^^^^
- The fields "DigitizedLandmarks" and "DigitizedHeadPoints" in the json sidecar of Neuromag data are now set to True/False depending on whether any landmarks (NAS, RPA, LPA) or extra points are found in raw.info['dig'], by `Eduard Ort`_ (:gh:`772`)
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

API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^

- Writing datasets via :func:`write_raw_bids`, will now never overwrite ``dataset_description.json`` file, by `Adam Li`_ (:gh:`765`)
- When writing BIDS datasets, MNE-BIDS now tags them as BIDS 1.6.0 (we previously tagged them as BIDS 1.4.0), by `Richard Höchenberger`_ (:gh:`782`)
- :func:`mne_bids.read_raw_bids` now passes ``allow_maxshield=True`` to the MNE-Python reader function by default when reading FIFF files. Previously, ``extra_params=dict(allow_maxshield=True)`` had to be passed explicitly, by `Richard Höchenberger`_ (:gh:`787`)
- The ``raw_to_bids`` command has lost its ``--allow_maxshield`` parameter. If writing a FIFF file, we will now always assume that writing data before applying a Maxwell filter is fine, by `Richard Höchenberger`_ (:gh:`787`)
- :meth:`mne_bids.BIDSPath.find_empty_room` now first looks for an ``AssociatedEmptyRoom`` field in the MEG JSON sidecar file to retrieve the empty-room recording; only if this information is missing, it will proceed to try and find the best-matching empty-room recording based on measurement date (i.e., fall back to the previous behavior), by `Richard Höchenberger`_ (:gh:`795`)
- If :func:`mne_bids.read_raw_bids` encounters raw data with the ``STI 014`` stimulus channel and this channel is not explicitly listed in ``*_channels.tsv``, it is now automatically removed upon reading, by `Richard Höchenberger`_ (:gh:`823`)
- :func:`mne_bids.get_anat_landmarks` was added to clarify and simplify the process of generating landmarks that now need to be passed to :func:`mne_bids.write_anat`; this depreciates the arguments ``raw``, ``trans`` and ``t1w`` of :func:`mne_bids.write_anat`, by `Alex Rockhill`_ and `Alexandre Gramfort`_ (:gh:`827`)
- :func:`write_raw_bids` now accepts preloaded raws as input with some caveats if the new parameter ``allow_preload`` is explicitly set to ``True``. This enables some preliminary support for uncommon file formats, generated data, processed derivatives etc., by `Sin Kim`_ (:gh:`819`)

Requirements
^^^^^^^^^^^^

- For downloading `OpenNeuro <https://openneuro.org>`_ datasets, ``openneuro-py`` is now required to run the examples and build the documentation, by `Alex Rockhill`_ (:gh:`753`)
- MNE-BIDS now depends on `setuptools <https://setuptools.readthedocs.io>`_. This package is normally installed by your Python distribution automatically, so we don't expect any users to be affected by this change, by `Richard Höchenberger`_ (:gh:`794`)

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
- :func:`mne_bids.get_head_mri_trans` and :func:`mne_bids.write_anat` used a T1w image but depended specifically on the freesurfer T1w image. Now the freesurfer subject directory is used, by `Alex Rockhill`_ and `Alexandre Gramfort`_ (:gh:`827`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
