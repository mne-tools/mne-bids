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
* `Richard Höchenberger`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

Enhancements
^^^^^^^^^^^^
- The function :func:`mne_bids.print_dir_tree` has a new parameter ``return_str`` which allows it to return a str of the dir tree instead of printing it, by `Stefan Appelhoff`_ (`#600 <https://github.com/mne-tools/mne-bids/pull/600>`_)
- :func:`mne_bids.write_raw_bids` now preserves event durations when writing :class:`mne.Annotations` to ``*_events.tsv`` files, and :func:`mne_bids.read_raw_bids` restores these durations upon reading, by `Richard Höchenberger`_ (`#603 <https://github.com/mne-tools/mne-bids/pull/603>`_)

Bug fixes
^^^^^^^^^
- Fix a bug in :func:`mne_bids.write_raw_bids`, where passing raw data with :class:`mne.Annotations` set and ``event_id`` dictionary not containing the :class:`mne.Annotations` descriptions as keys would raise an error, by `Richard Höchenberger`_ (`#603 <https://github.com/mne-tools/mne-bids/pull/603>`_)

API changes
^^^^^^^^^^^
- When passing event IDs to :func:`mne_bids.write_raw_bids` via ``events_data`` without an accompanying event description in ``event_id``, we will now raise a `ValueError`. This ensures that accidentally un-described events won't get written unnoticed, by `Richard Höchenberger`_ (`#603 <https://github.com/mne-tools/mne-bids/pull/603>`_)


:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
