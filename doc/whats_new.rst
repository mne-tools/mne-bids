:orphan:

.. _whats_new:

.. currentmodule:: mne_bids

.. include:: authors.rst

What's new?
===========

.. _changes_0_18:

Version 0.18 (unreleased)
-------------------------

👩🏽‍💻 Authors
~~~~~~~~~~~~~~~

The following authors contributed for the first time. Thank you so much! 🤩

* `Julius Welzel`_
* `Alex Lopez Marquez`_


The following authors had contributed before. Thank you for sticking around! 🤘

* `Stefan Appelhoff`_


Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

🚀 Enhancements
^^^^^^^^^^^^^^^

- :func:`mne_bids.write_raw_bids()` has a new parameter `electrodes_tsv_task` which allows adding the `task` entity to the `electrodes.tsv` filepath, by `Alex Lopez Marquez`_ (:gh:`1424`)
- Extended the configuration to recognise `motion` as a valid BIDS datatype by `Julius Welzel`_ (:gh:`1430`)

🧐 API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  `tracksys` accepted as argument in :class:`mne_bids.BIDSPath()` by `Julius Welzel`_ (:gh:`1430`)

🛠 Requirements
^^^^^^^^^^^^^^^

- None yet

🪲 Bug fixes
^^^^^^^^^^^^

- Fixed a bug that modified the name and help message of some of the available commands, by `Alex Lopez Marquez`_ (:gh:`1441`)

⚕️ Code health
^^^^^^^^^^^^^^

- Made :func:`mne_bids.copyfiles.copyfile_brainvision` output more meaningful error messages when encountering problematic files, by `Stefan Appelhoff`_ (:gh:`1444`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`
