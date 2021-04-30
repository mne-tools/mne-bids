:orphan:

.. _whats_new:

.. currentmodule:: mne_bids

What's new?
===========

.. currentmodule:: mne_bids
.. _changes_0_8:

Version 0.8 (unreleased)
------------------------

...

Notable changes
~~~~~~~~~~~~~~~

- ...

Authors
~~~~~~~

* `Alex Rockhill`_
* `Richard Höchenberger`_
* `Adam Li`_
* `Richard Köhler`_ (new contributor)

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

Enhancements
^^^^^^^^^^^^
- The fields "DigitizedLandmarks" and "DigitizedHeadPoints" in the json sidecar of Neuromag data are now set to True/False depending on whether any landmarks (NAS, RPA, LPA) or extra points are found in raw.info['dig'], by `Eduard Ort`_ (:gh:`772`)
- Updated the "Read BIDS datasets" example to use data from `OpenNeuro <https://openneuro.org>`_, by `Alex Rockhill`_ (:gh:`753`)
- :func:`mne_bids.get_head_mri_trans` is now more lenient when looking for the fiducial points (LPA, RPA, and nasion) in the MRI JSON sidecar file, and accepts a larger variety of landmark names (upper- and lowercase letters; ``'nasion'`` instead of only ``'NAS'``), by `Richard Höchenberger`_ (:gh:`769`)
- :func:`mne_bids.get_head_mri_trans` gained a new keyword argument ``t1_bids_path``, allowing for the MR scan to be stored in a different session or even in a different BIDS dataset than the electrophysiological recording, by `Richard Höchenberger`_ (:gh:`771`)
- Add writing simultaneous EEG-iEEG recordings via :func:`mne_bids.write_raw_bids`. The desired output datatype must be specified in the :class:`mne_bids.BIDSPath` object, by `Richard Köhler`_ (:gh:`774`)
- :func:`mne_bids.write_raw_bids` gained a new keyword argument ``symlink``, which allows to create symbolic links to the original data files instead of copying them over. Currently works for ``FIFF`` files on macOS and Linux, by `Richard Höchenberger`_ (:gh:`778`)
- :class:`mne_bids.BIDSPath` now has property getter and setter methods for all BIDS entities, i.e., you can now do things like ``bids_path.subject = 'foo'`` and don't have to resort to ``bids_path.update()``. This also ensures you'll get proper completion suggestions from your favorite Python IDE, by `Richard Höchenberger`_ (:gh:`786`)

API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^

- Writing datasets via :func:`write_raw_bids`, will now never overwrite ``dataset_description.json`` file, by `Adam Li`_ (:gh:`765`)

Requirements
^^^^^^^^^^^^

- For downloading `OpenNeuro <https://openneuro.org>`_ datasets, ``openneuro-py`` is now required to run the examples and build the documentation, by `Alex Rockhill`_ (:gh:`753`)

Bug fixes
^^^^^^^^^

- :func:`mne_bids.make_report` now (1) detects male/female sex and left/right handedness irrespective of letter case, (2) will parse a ``gender`` column if no ``sex`` column is found in ``participants.tsv``, and (3) reports sex as male/female instead of man/woman, by `Alex Rockhill`_ (:gh:`755`)
- The :class:`mne.Annotations` ``BAD_ACQ_SKIP`` – added by the acquisition system to ``FIFF`` files – will now be preserved when reading raw data, even if these time periods are **not** explicitly included in ``*_events.tsv``, by `Richard Höchenberger`_ and `Alexandre Gramfort`_ (:gh:`754` and :gh:`762`)
- :func:`mne_bids.write_raw_bids` will handle different cased extensions for EDF files, such as `.edf` and `.EDF` by `Adam Li`_ (:gh: `765`)
- :func:`mne_bids.inspect_dataset` didn't handle certain filenames correctly on some systems, by `Richard Höchenberger`_ (:gh:`769`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
