:orphan:

.. _whats_new:

.. currentmodule:: mne_bids

What's new?
===========

.. currentmodule:: mne_bids
.. _changes_0_6:

Version 0.6 (unreleased)
------------------------
xxx

.. contents:: Contents
   :local:
   :depth: 3

Notable changes
~~~~~~~~~~~~~~~
xxx

Authors
~~~~~~~
* `Stefan Appelhoff`_
* `Adam Li`_
* `Richard Höchenberger`_
* `Austin Hurst`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

Enhancements
^^^^^^^^^^^^

- The function :func:`mne_bids.print_dir_tree` has a new parameter ``return_str`` which allows it to return a str of the dir tree instead of printing it, by `Stefan Appelhoff`_ (`#600 <https://github.com/mne-tools/mne-bids/pull/600>`_)
- :func:`mne_bids.write_raw_bids` now preserves event durations when writing :class:`mne.Annotations` to ``*_events.tsv`` files, and :func:`mne_bids.read_raw_bids` restores these durations upon reading, by `Richard Höchenberger`_ (`#603 <https://github.com/mne-tools/mne-bids/pull/603>`_)
- Writing BrainVision data via :func:`mne_bids.write_raw_bids` will now set the unit of EEG channels to µV for enhanced interoperability with other software, by `Alexandre Gramfort`_, `Stefan Appelhoff`_, and `Richard Höchenberger`_ (`#610 <https://github.com/mne-tools/mne-bids/pull/610>`_)
- Add :func:`mne_bids.update_sidecar_json` to allow updating sidecar JSON files with a template JSON by `Adam Li`_ and `Austin Hurst`_ (`#601 <https://github.com/mne-tools/mne-bids/pull/601>`_)

Bug fixes
^^^^^^^^^
- Make sure large FIF files with splits are handled transparently on read and write, by `Alexandre Gramfort`_ (`#612 <https://github.com/mne-tools/mne-bids/pull/612>`_)
- The function :func:`mne_bids.write_raw_bids` now outputs ``*_electrodes.tsv`` and ``*_coordsystem.json`` files for EEG/iEEG data that are BIDS-compliant (only contain subject, session, acquisition, and space entities), by `Adam Li`_ (`#601 <https://github.com/mne-tools/mne-bids/pull/601>`_)
- Make sure writing empty-room data with anonymization shifts the session back in time, by `Alexandre Gramfort`_ (`#611 <https://github.com/mne-tools/mne-bids/pull/611>`_)
- Fix a bug in :func:`mne_bids.write_raw_bids`, where passing raw data with :class:`mne.Annotations` set and the ``event_id`` dictionary not containing the :class:`mne.Annotations` descriptions as keys would raise an error, by `Richard Höchenberger`_ (`#603 <https://github.com/mne-tools/mne-bids/pull/603>`_)
- Fix a bug in :func:`mne_bids.write_raw_bids` when passing raw MEG data with Internal Active Shielding (IAS) from Triux system, by `Alexandre Gramfort`_ (`#616 <https://github.com/mne-tools/mne-bids/pull/616>`_)
- Fix a bug in :func:`mne_bids.write_raw_bids`, where original format of data was not kept when writing to FIFF, by `Alexandre Gramfort`_, `Stefan Appelhoff`_, and `Richard Höchenberger`_ (`#610 <https://github.com/mne-tools/mne-bids/pull/610>`_)
- Fix a bug where conversion to BrainVision format was done even when non-Volt channel types were present in the data (BrainVision conversion is done by ``pybv``, which currently only supports Volt channel types), by `Stefan Appelhoff`_ (`#619 <https://github.com/mne-tools/mne-bids/pull/619>`_)

API changes
^^^^^^^^^^^
- When passing event IDs to :func:`mne_bids.write_raw_bids` via ``events_data`` without an accompanying event description in ``event_id``, we will now raise a `ValueError`. This ensures that accidentally un-described events won't get written unnoticed, by `Richard Höchenberger`_ (`#603 <https://github.com/mne-tools/mne-bids/pull/603>`_)


Requirements
^^^^^^^^^^^^
- Writing BrainVision data now requires ``pybv`` 0.3 or later.

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
