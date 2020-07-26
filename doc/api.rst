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
   make_bids_folders
   make_bids_basename
   make_dataset_description
   make_report
   write_anat
   get_head_mri_trans
   get_matched_empty_room
   get_anonymization_daysback


Path
====

:py:mod:`mne_bids.path`:

.. automodule:: mne_bids.path
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne_bids.path

.. autosummary::
   :toctree: generated/

   print_dir_tree
   get_entity_vals
   get_kinds
   BIDSPath


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


Utils
=====

:py:mod:`mne_bids.utils`:

.. automodule:: mne_bids.utils
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne_bids.utils

.. autosummary::
   :toctree: generated/

   get_anonymization_daysback