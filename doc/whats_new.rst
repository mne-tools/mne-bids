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

* `Richard Höchenberger`_
* `Mainak Jas`_
* `Adam Li`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

Enhancements
^^^^^^^^^^^^

- :func:`mne_bids.get_anat_landmarks` now accepts a :class:`mne_bids.BIDSPath` as ``image`` parameter, by `Alex Rockhill`_ (:gh:`852`)
- :func:`mne_bids.write_raw_bids` now accepts 'EDF' as a 'format' key word to force conversion to EDF files, by `Adam Li`_ (:gh:``)

API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^

- ...

Requirements
^^^^^^^^^^^^

- ...

Bug fixes
^^^^^^^^^

- Fix writing Ricoh/KIT data that comes without an associated ``.mrk``, ``.elp``, or ``.hsp`` file using :func:`mne_bids.write_raw_bids`, by `Richard Höchenberger`_ (:gh:`850`)

- Properly support CTF MEG data with 2nd-order gradient compensation, by `Mainak Jas`_ (:gh:`858`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
