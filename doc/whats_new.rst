:orphan:

.. _whats_new:

.. currentmodule:: mne_bids

What's new?
===========

.. _changes_0_10:

Version 0.10 (unreleased)
-------------------------

...

Notable changes
~~~~~~~~~~~~~~~

- ...

Authors
~~~~~~~

People who contributed to this release (in alphabetical order):

* `Simon Kern`_
* `Adam Li`_
* `Alex Rockhill`_
* `Alexandre Gramfort`_
* `Mainak Jas`_
* `Richard Höchenberger`_
* `Stefan Appelhoff`_
* `Yorguin Mantilla`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

Enhancements
^^^^^^^^^^^^

- Add support for CNT (Neuroscan) files in :func:`mne_bids.write_raw_bids`, by `Yorguin Mantilla`_ (:gh:`924`)

- Add the ability to write multiple landmarks with :func:`mne_bids.write_anat` (e.g. to have separate landmarks for different sessions) via the new ``kind`` parameter, by `Alexandre Gramfort`_ (:gh:`955`)

- :func:`mne_bids.get_head_mri_trans` and :func:`mne_bids.update_anat_landmarks` gained a new``kind`` parameter to specify which of multiple landmark sets to retrieve, by `Alexandre Gramfort`_ and `Richard Höchenberger`_ (:gh:`955`, :gh:`957`)

API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^

- :func:`mne_bids.update_anat_landmarks` will now by default raise an exception if the requested MRI landmarks do not already exist. Use the new ``on_missing`` parameter to control this behavior, by `Richard Höchenberger`_ (:gh:`957`)

Requirements
^^^^^^^^^^^^

- MNE-BIDS now requires Jinja2 to work with MNE-Python 0.24.

Bug fixes
^^^^^^^^^

- Forcing EDF conversion in :func:`mne_bids.write_raw_bids` properly uses the ``overwrite`` parameter now, by `Adam Li`_ (:gh:`930`)

- :func:`mne_bids.make_report` now correctly handles ``participant.tsv`` files that only contain a ``participant_id`` column, by `Simon Kern`_ (:gh:`912`)

- :func:`mne_bids.write_raw_bids` doesn't store age, handedness, and sex in ``participants.tsv`` anymore for empty-room recordings, by `Richard Höchenberger`_ (:gh:`935`)

- When :func:`mne_bids.read_raw_bids` automatically creates new hierarchical event names based on event values (in cases where the same ``trial_type`` was assigned to different ``value``s in ``*_events.tsv``), ``'n/a'`` values will now be converted to ``'na'``, by `Richard Höchenberger`_ (:gh:`937`)

- Avoid ``DeprecationWarning`` in :func:`mne_bids.inspect_dataset` with the upcoming MNE-Python 1.0 release, by `Richard Höchenberger`_ (:gh:`942`)

- Avoid modifying the instance of :class:`mne_bids.BIDSPath` if validation fails when calling :meth:`mne_bids.BIDSPath.update`, by `Alexandre Gramfort`_ (:gh:`950`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
