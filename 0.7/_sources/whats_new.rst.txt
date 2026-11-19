:orphan:

.. _whats_new:

.. currentmodule:: mne_bids

What's new?
===========

.. currentmodule:: mne_bids
.. _changes_0_7:

Version 0.7
-----------

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

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
