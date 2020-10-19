:orphan:

.. _api_documentation:

=================
API Documentation
=================

Here we list the Application Programming Interface (API) for MNE-BIDS.

.. contents:: Contents
   :local:
   :depth: 2


MNE BIDS
========

:py:mod:`mne_bids`:

.. automodule:: mne_bids
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne_bids

.. autosummary::
   :toctree: generated/

   write_raw_bids
   read_raw_bids
   write_meg_calibration
   write_meg_crosstalk
   BIDSPath
   make_dataset_description
   make_report
   write_anat
   mark_bad_channels
   get_head_mri_trans
   get_anonymization_daysback
   print_dir_tree
   get_entities_from_fname
   get_entity_vals
   get_datatypes

Copyfiles
=========

:py:mod:`mne_bids.copyfiles`:

.. automodule:: mne_bids.copyfiles
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne_bids.copyfiles

.. autosummary::
   :toctree: generated/

   copyfile_brainvision
   copyfile_eeglab
   copyfile_ctf
   copyfile_bti
   copyfile_kit
