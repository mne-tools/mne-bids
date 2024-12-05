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
* `Daniel McCloy`_

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

- Nothing yet

⚕️ Code health
^^^^^^^^^^^^^^

- Tests that were adding or deleting files to/from a session-scoped dataset now properly clean up after themselves, by `Daniel McCloy`_ (:gh:`1347`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
