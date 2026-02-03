:orphan:

.. _whats_new:

.. currentmodule:: mne_bids

.. include:: authors.rst

What's new?
===========

.. _changes_0_19:

Version 0.19 (unreleased)
-------------------------

👩🏽‍💻 Authors
~~~~~~~~~~~~~~~

The following authors contributed for the first time. Thank you so much! 🤩

* None Yet

The following authors had contributed before. Thank you for sticking around! 🤘

* `Pierre Guetschel`_
* `Alexandre Gramfort`_


Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

🚀 Enhancements
^^^^^^^^^^^^^^^

- Save ``Annotations.extras`` fields in events.tsv files when writing events, by `Pierre Guetschel`_ (:gh:`1502`)
- :func:`mne_bids.read_raw_bids` now reads channel units from ``channels.tsv`` and sets them on the raw object. This includes support for units like ``rad`` (radians), ``V``, ``µV``, ``mV``, ``T``, ``T/m``, ``S``, ``oC``, ``M``, and ``px``. The write path was also updated to correctly write ``rad`` units to ``channels.tsv``, by `Alexandre Gramfort`_ (:gh:`1509`)

🧐 API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- None yet

🛠 Requirements
^^^^^^^^^^^^^^^

- None yet

🪲 Bug fixes
^^^^^^^^^^^^

- None yet

⚕️ Code health
^^^^^^^^^^^^^^

- None yet

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`
