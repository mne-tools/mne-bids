:orphan:

.. _whats_new:

.. currentmodule:: mne_bids

What's new?
===========

.. _changes_0_13:

Version 0.13 (unreleased)
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

* `Richard Höchenberger`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

🚀 Enhancements
^^^^^^^^^^^^^^^

- ...

🧐 API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ...

🛠 Requirements
^^^^^^^^^^^^^^^

- ...

🪲 Bug fixes
^^^^^^^^^^^^

- Amending a dataset now works in cases where the newly-written data contains additional participant properties (new columns in ``participants.tsv``) not found in the existing dataset, by `Richard Höchenberger`_ (:gh:`1113`)
- Fix ``raw_to_bids`` CLI tool to properly recognize boolean and numeric values for the ``line_freq`` and ``overwrite`` parameters, by `Stefan Appelhoff`_ (:gh:`1125`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
