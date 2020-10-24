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

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

Enhancements
^^^^^^^^^^^^
- The function :func:`mne_bids.print_dir_tree` has a new parameter ``return_str`` which allows it to return a str of the dir tree instead of printing it, by `Stefan Appelhoff`_ (`#600 <https://github.com/mne-tools/mne-bids/pull/600>`_)
- The function :func:`mne_bids.write_raw_bids` now outputs `electrodes.tsv` and `coordsystem.json` files for EEG/iEEG data that are BIDS compliant (only contain subject, session, acquisition, and space entities), by `Adam Li`_ (`#601 <https://github.com/mne-tools/mne-bids/pull/601>`_)

Bug fixes
^^^^^^^^^
xxx

API changes
^^^^^^^^^^^
xxx


:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
