"""
=========================================
07. Read and load datasets in BIDS format
=========================================

When working with neural data in the BIDS format, we usually have
varying types of data, which can be loaded in via `read_raw_bids` function.

- MEG
- EEG (scalp)
- iEEG (ECoG and SEEG)
- the anatomical MRI scan of a study participant

In order to check if your dataset is compliant with BIDS, you can first
run the BIDS validator online.

In this tutorial, we show how `read_raw_bids`
can be used to load any BIDS format data,
and to display the data that is saved within the accompanying sidecars.

.. note:: For this example you will need to install ``matplotlib`` and
          ``nilearn`` on top of your usual ``mne-bids`` installation.

"""
# Authors: Adam Li <https://github.com/adam2392>
# License: BSD (3-clause)

###############################################################################
# We are importing everything we need for this example:

import os

import mne
from mne.datasets import eegbci

from mne_bids import (make_bids_basename, write_raw_bids, read_raw_bids)
from mne_bids.utils import print_dir_tree

import shutil as sh
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# We will be using the `MNE sample data <mne_sample_data_>`_ and write a basic
# BIDS dataset. For more information, you can checkout the respective
# :ref:`example <ex-convert-mne-sample>`.

# get MNE directory w/ example data
mne_dir = mne.get_config('MNE_DATASETS_SAMPLE_PATH')

###############################################################################
# Step 1: Download/Get a BIDS dataset
# -----------------------------------
#
# Let's start by formatting a few subjects to test. This code is
# copied over from the `convert_eeg_to_bids` example.
bids_root = os.path.join(mne_dir, 'eegmmidb_bids')

# Define which tasks we want to download.
tasks = [2,  # This is 2 minutes of eyes closed rest
         4,  # This is run #1 of imagining to close left or right fist
         12]  # This is run #2 of imagining to close left or right fist

# Download the data for subjects 1 and 2
for subj_idx in [1, 2]:
    eegbci.load_data(subject=subj_idx,
                     runs=tasks,
                     path=mne_dir,
                     update_path=True)
data_dir = os.path.join(mne_dir, 'MNE-eegbci-data')

# Some initial information that we found in the PhysioBank documentation
task_names = {2: 'resteyesclosed',
              4: 'imaginefists',  # run 1
              12: 'imaginefists'  # run 2
              }

run_mapping = {2: None,  # for resting eyes closed task, there was only one run
               4: '1',
               12: '2'
               }

# Brief description of the event markers present in the data. This will become
# the `trial_type` column in our BIDS `events.tsv`. We know about the event
# meaning from the documentation on PhysioBank
trial_type = {'rest': 0, 'imagine left fist': 1, 'imagine right fist': 2}

# Now go over subjects and *bidsify*
for subj_idx in [1, 2]:
    for task_idx in [2, 4, 12]:
        # Load the data
        edf_path = eegbci.load_data(subject=subj_idx, runs=task_idx)[0]

        raw = mne.io.read_raw_edf(edf_path, preload=False, stim_channel=None)
        annot = mne.read_annotations(edf_path)
        raw.set_annotations(annot)
        events, event_id = mne.events_from_annotations(raw)

        bids_basename = make_bids_basename(subject='{:03}'.format(subj_idx),
                                           task=task_names[task_idx],
                                           run=run_mapping[task_idx])

        write_raw_bids(raw, bids_basename, bids_root, event_id=trial_type,
                       events_data=events, overwrite=True)

###############################################################################
# Print the directory tree
print_dir_tree(bids_root)

###############################################################################
# Step 2: Read a BIDS dataset
# ---------------------------
#
# Let's read in the dataset and show off a few features of the
# loading function `read_raw_bids`.
bids_fname = bids_basename + "_eeg.edf"
raw = read_raw_bids(bids_fname, bids_root, verbose=True)

###############################################################################
# `raw.info` has the basic subject metadata
print(raw.info['subject_info'])

# `raw.info` has the basic channel metadata
print(raw.info['dig'])

# `raw.info` has the PowerLineFrequency loaded in
print(raw.info['line_freq'])

###############################################################################
# Let's modify data in some of the sidecar files and re-load the data
# and see how the `raw.info` has changed.


###############################################################################
# `raw.info` has the basic subject metadata
print(raw.info['subject_info'])

# `raw.info` has the basic channel metadata
print(raw.info['dig'])

# `raw.info` has the PowerLineFrequency loaded in
print(raw.info['line_freq'])

# Plot it
# fig, ax = plt.subplots()
# plot_anat(t1_nii_fname, axes=ax, title='Defaced')
# plt.show()

###############################################################################
# .. LINKS
#
# .. _mne_sample_data:
#    https://martinos.org/mne/stable/manual/sample_dataset.html
#
