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

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

Enhancements
^^^^^^^^^^^^

- xxx

API changes
^^^^^^^^^^^

- xxx

Requirements
^^^^^^^^^^^^

- For writing BrainVision files, ``pybv`` version 0.5 is now required to allow writing of non-voltage channels, by `Adam Li`_ (:gh:`670`)

Bug fixes
^^^^^^^^^

- Anonymization of EDF files in :func:`write_raw_bids` will now convert recording date to ``01-01-1985 00:00:00``, while setting the recording date in the ``scans.tsv`` file to an anonymized date < 1925, thus making the file EDF/EDFBrowser compliant, by `Adam Li`_ (:gh:`600`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
