:orphan:

.. _whats_new:

.. currentmodule:: mne_bids

What's new?
===========

.. _changes_0_14:

Version 0.14 (unreleased)
-------------------------

👩🏽‍💻 Authors
~~~~~~~~~~~~~~~

The following authors contributed for the first time. Thank you so much! 🤩

* nobody yet

The following authors had contributed before. Thank you for sticking around! 🤘

* `Richard Höchenberger`_
* `Stefan Appelhoff`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

🚀 Enhancements
^^^^^^^^^^^^^^^

- nothing yet

🧐 API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- nothing yet

🛠 Requirements
^^^^^^^^^^^^^^^

- nothing yet

🪲 Bug fixes
^^^^^^^^^^^^

- Fix reading when the channel order differs between ``*_channels.tsv`` and the raw data file, which would previously throw an error, by `Richard Höchenberger`_ (:gh:`1171`)
- Make ``recording`` entity available for :func:`mne_bids.get_entity_vals`, by `Stefan Appelhoff`_ (:gh:`1182`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
