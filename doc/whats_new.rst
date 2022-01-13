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

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

Enhancements
^^^^^^^^^^^^

- ...

API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^

- ...

Requirements
^^^^^^^^^^^^

- ...

Bug fixes
^^^^^^^^^

- Forcing EDF conversion in :func:`mne_bids.write_raw_bids` properly uses the ``overwrite`` parameter now, by `Adam Li`_ (:gh:`930`)

- :func:`mne_bids.make_report` now correctly handles ``participant.tsv`` files that only contain a ``participant_id`` column, by `Simon Kern`_ (:gh:`912`)

- :func:`mne_bids.write_raw_bids` doesn't store age, handedness, and sex in ``participants.tsv`` anymore for empty-room recordings, by `Richard Höchenberger`_ (:gh:`935`)

- When :func:`mne_bids.read_raw_bids` automatically creates new hierarchical event names based on event values (in cases where the same ``trial_type`` was assigned to different ``value``s in ``*_events.tsv``), ``'n/a'`` values will now be converted to ``'na'``, by `Richard Höchenberger`_ (:gh:`937`)

- Avoid ``DeprecationWarning`` in :func:`mne_bids.inspect_dataset` with the upcoming MNE-Python 1.0 release, by `Richard Höchenberger`_ (:gh:`942`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
