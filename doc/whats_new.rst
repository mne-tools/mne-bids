:orphan:

.. _whats_new:

.. currentmodule:: mne_bids

What's new?
===========

.. currentmodule:: mne_bids
.. _changes_0_7:

Version 0.7 (unreleased)
------------------------

xxx

Notable changes
~~~~~~~~~~~~~~~

- xxx

Authors
~~~~~~~

* `Adam Li`_
* `Richard Höchenberger`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

Enhancements
^^^^^^^^^^^^

- Some datasets out in the real world have a non-standard ``stim_type`` instead of a ``trial_type`` column in ``*_events.tsv``. :func:`mne_bids.read_raw_bids` now makes use of this column, and emits a warning, encouraging users to rename it, by `Richard Höchenberger`_ (:gh:`680`)
- When reading data where the same event name or trial type refers to different event or trigger values, we will now create a hierarchical event name in the form of ``trial_type/value``, e.g. ``stimulus/110``  by `Richard Höchenberger`_ (:gh:`688`)

API changes
^^^^^^^^^^^

- Add ``format`` kwarg to :func:`write_raw_bids` that allows users to specify if they want to force conversion to ``BrainVision`` or ``FIF`` file format, by `Adam Li`_ (:gh:`672`)

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

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
