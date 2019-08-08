:orphan:

.. _api_documentation:

=================
API Documentation
=================

Here we list the Application Programming Interface (API) for MNE-BIDS.

.. contents:: Contents
   :local:
   :depth: 2

MNE BIDS (:py:mod:`mne_bids`)
=============================

.. currentmodule:: mne_bids

.. autosummary::
   :toctree: generated/

   write_raw_bids
   read_raw_bids
   make_bids_folders
   make_bids_basename
   make_dataset_description
   write_anat
   get_head_mri_trans

Utils (:py:mod:`mne_bids.utils`)
================================

.. currentmodule:: mne_bids.utils

.. autosummary::
   :toctree: generated/

   print_dir_tree
   get_values_for_key
   get_kinds

Copyfiles (:py:mod:`mne_bids.copyfiles`)
========================================

.. currentmodule:: mne_bids.copyfiles

.. autosummary::
   :toctree: generated/

   copyfile_brainvision
   copyfile_eeglab
   copyfile_ctf
   copyfile_bti

Datasets (:py:mod:`mne_bids.datasets`)
======================================

.. currentmodule:: mne_bids.datasets

.. autosummary::
    :toctree: generated/

    fetch_faces_data
    fetch_brainvision_testing_data
