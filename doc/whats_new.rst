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

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

Enhancements
^^^^^^^^^^^^

- :func:`mne_bids.get_anat_landmarks` now accepts a :class:`mne_bids.BIDSPath` as ``image`` parameter, by `Alex Rockhill`_ (:gh:`852`)

- :func:`mne_bids.write_raw_bids` now accepts ``'EDF'`` as a ``'format'`` value to force conversion to EDF files, by `Adam Li`_ (:gh:`866`)

- Modify iEEG tutorial to use MNE ``raw`` object, by `Alex Rockhill`_ (:gh:`859`)

- Add :func:`mne_bids.search_folder_for_text` to find specific metadata entries (e.g. all ``"n/a"`` sidecar data fields, or to check that "60 Hz" was written properly as the power line frequency), by `Alex Rockhill`_ (:gh: `870`)

API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^

- ...

Requirements
^^^^^^^^^^^^

- Writing BrainVision files now requires ``pybv`` version 0.6, by `Stefan Appelhoff`_ (:gh:`880`)


Bug fixes
^^^^^^^^^

- Fix writing Ricoh/KIT data that comes without an associated ``.mrk``, ``.elp``, or ``.hsp`` file using :func:`mne_bids.write_raw_bids`, by `Richard Höchenberger`_ (:gh:`850`)

- Properly support CTF MEG data with 2nd-order gradient compensation, by `Mainak Jas`_ (:gh:`858`)

- Fix writing and reading EDF files with upper-case extension (``.EDF``), by `Adam Li`_ (:gh:`868`)

- Fix reading of EDF files with lower/upper case extension, by `Adam Li`_ (:gh:`875`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
