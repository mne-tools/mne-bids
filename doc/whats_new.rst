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

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

🚀 Enhancements
^^^^^^^^^^^^^^^

- Speed up :func:`mne_bids.read_raw_bids` when lots of events are present by `Alexandre Gramfort`_ (:gh:`1079`)
- Add the option ``return_candidates`` to :meth:`mne_bids.BIDSPath.find_empty_room` by `Eric Larson`_ (:gh:`1083`)
- When writing data via :func:`~mne_bids.write_raw_bids`, it is now possible to specify a custom mapping of :class:`mne.Annotations` descriptions to event codes via the ``event_id`` parameter. Previously, passing this parameter would always require to also pass ``events``, and using a custom event code mapping for annotations was impossible, by `Richard Höchenberger`_ (:gh:`1084`)

🧐 API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ...

🛠 Requirements
^^^^^^^^^^^^^^^

- ...

🪲 Bug fixes
^^^^^^^^^^^^

- When writing data containing :class:`mne.Annotations` **and** passing events to :func:`~mne_bids.write_raw_bids`, previously, annotations whose description did not appear in ``event_id`` were silently dropped. We now raise an exception and request users to specify mappings between descriptions and event codes in this case. It is still possible to omit ``event_id`` if no ``events`` are passed, by `Richard Höchenberger`_ (:gh:`1084`)


- Fix the ``basename`` of :class:`~mne_bids.BIDSPath` when the path only consists of a filename extension. Previously, the ``basename`` would be empty, by `Richard Höchenberger`_ (:gh:`1062`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
