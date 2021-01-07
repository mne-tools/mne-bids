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

- Add ``format`` kwarg to :func:`write_raw_bids` that allows users to specify if they want to force conversion to ``BrainVision`` or ``FIF`` file format, by `Adam Li`_ (:gh:`672`)

Requirements
^^^^^^^^^^^^

- For writing BrainVision files, ``pybv`` version 0.5 is now required to allow writing of non-voltage channels, by `Adam Li`_ (:gh:`670`)

Bug fixes
^^^^^^^^^

- Fix writing MEGIN Triux files, by `Alexandre Gramfort`_ (:gh:`674`)
- Anonymization of EDF files in :func:`write_raw_bids` will now convert recording date to ``01-01-1985 00:00:00`` if anonymization takes place, while setting the recording date in the ``scans.tsv`` file to the anonymized date, thus making the file EDF/EDFBrowser compliant, by `Adam Li`_ (:gh:`669`)
- Add fix to MEG ``coordsystem.json`` file writing, where an error will be thrown if ``coordsystem.json`` file path exists, but contents differ, by `Adam Li`_ (:gh:`675`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
