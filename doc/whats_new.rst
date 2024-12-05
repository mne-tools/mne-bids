:orphan:

.. _whats_new:

.. currentmodule:: mne_bids

What's new?
===========

.. _changes_0_17:

Version 0.17 (unreleased)
-------------------------

👩🏽‍💻 Authors
~~~~~~~~~~~~~~~

The following authors contributed for the first time. Thank you so much! 🤩

* `Christian O'Reilly`_

The following authors had contributed before. Thank you for sticking around! 🤘

* `Stefan Appelhoff`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

🚀 Enhancements
^^^^^^^^^^^^^^^

- :func:`mne_bids.write_raw_bids()` can now handle mne `Raw` objects with `eyegaze` and `pupil` channels, by `Christian O'Reilly`_ (:gh:`1344`)


🧐 API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Nothing yet

🛠 Requirements
^^^^^^^^^^^^^^^

- Nothing yet

🪲 Bug fixes
^^^^^^^^^^^^

- :func:`mne_bids.read_raw_bids` can optionally return an ``event_id`` dictionary suitable for use with :func:`mne.events_from_annotations`, and if a ``values`` column is present in ``events.tsv`` it will be used as the source of the integer event ID codes, by `Daniel McCloy`_ (:gh:`1349`)

⚕️ Code health
^^^^^^^^^^^^^^

- Nothing yet

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
