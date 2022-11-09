:orphan:

.. _whats_new:

.. currentmodule:: mne_bids

What's new?
===========

.. _changes_0_12:

Version 0.12 (unreleased)
-------------------------

...

📝 Notable changes
~~~~~~~~~~~~~~~~~~

- ...

👩🏽‍💻 Authors
~~~~~~~~~~~~~~~

The following authors contributed for the first time. Thank you so much! 🤩

* ...

The following authors had contributed before. Thank you for sticking around! 🤘

* `Alexandre Gramfort`_
* `Eric Larson`_
* `Richard Höchenberger`_
* `Stefan Appelhoff`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

🚀 Enhancements
^^^^^^^^^^^^^^^

- Speed up :func:`mne_bids.read_raw_bids` when lots of events are present by `Alexandre Gramfort`_ (:gh:`1079`)
- Add :meth:`mne_bids.BIDSPath.get_empty_room_candidates` to get the candidate empty-room files that could be used by :meth:`mne_bids.BIDSPath.find_empty_room` by `Eric Larson`_ (:gh:`1083`, :gh:`1093`)
- Add :meth:`mne_bids.BIDSPath.find_matching_sidecar` to find the sidecar file associated with a given file path by `Eric Larson`_ (:gh:`1093`)
- When writing data via :func:`~mne_bids.write_raw_bids`, it is now possible to specify a custom mapping of :class:`mne.Annotations` descriptions to event codes via the ``event_id`` parameter. Previously, passing this parameter would always require to also pass ``events``, and using a custom event code mapping for annotations was impossible, by `Richard Höchenberger`_ (:gh:`1084`)
- Improve error message when :obj:`~mne_bids.BIDSPath.fpath` cannot be uniquely resolved by `Eric Larson`_ (:gh:`1097`)

🧐 API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ...

🛠 Requirements
^^^^^^^^^^^^^^^

- ...

🪲 Bug fixes
^^^^^^^^^^^^

- When writing data containing :class:`mne.Annotations` **and** passing events to :func:`~mne_bids.write_raw_bids`, previously, annotations whose description did not appear in ``event_id`` were silently dropped. We now raise an exception and request users to specify mappings between descriptions and event codes in this case. It is still possible to omit ``event_id`` if no ``events`` are passed, by `Richard Höchenberger`_ (:gh:`1084`)
- When working with NIRS data, raise the correct error message when a faulty ``format`` argument is passed to :func:`~mne_bids.write_raw_bids`, by `Stefan Appelhoff`_ (:gh:`1092`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
