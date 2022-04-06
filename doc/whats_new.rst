:orphan:

.. _whats_new:

.. currentmodule:: mne_bids

What's new?
===========

.. _changes_0_11:

Version 0.11 (unreleased)
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

* `Adam Li`_
* `Alex Rockhill`_
* `Alexandre Gramfort`_
* `Eric Larson`_
* `Richard Höchenberger`_
* `Robert Luke`_
* `Stefan Appelhoff`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

🚀 Enhancements
^^^^^^^^^^^^^^^

- ...

🧐 API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :func:`mne_bids.make_dataset_description` now accepts keyword arguments only, and can now also write the following metadata: ``HEDVersion``, ``EthicsApprovals``, ``GeneratedBy``, and ``SourceDatasets``, by `Stefan Appelhoff`_ (:gh:`406`)

🛠 Requirements
^^^^^^^^^^^^^^^

- ...

🪲 Bug fixes
^^^^^^^^^^^^

- Fix ACPC in ``surface RAS`` instead of ``scanner RAS`` in :ref:`ieeg-example` and add convenience functions :func:`mne_bids.convert_montage_to_ras` and :func:`mne_bids.convert_montage_to_mri` to help, by `Alex Rockhill`_ :gh:`990`

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
