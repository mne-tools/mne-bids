"""
=========================================
02. Read and load datasets in BIDS format
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

"""
# Authors: Adam Li <adam2392@gmail.com>
# License: BSD (3-clause)

###############################################################################
# We are importing everything we need for this example:
import os

import mne
from mne.datasets import somato

from mne_bids import make_bids_basename, read_raw_bids
from mne_bids.utils import print_dir_tree

###############################################################################
# We will be using the `MNE somato data <mne_somato_data_>`_, which
# is already stored in BIDS format.
# For more information, you can checkout the
# respective :ref:`example <ex-convert-mne-sample>`.

# get MNE directory w/ example data
mne_dir = mne.get_config('MNE_DATASETS_SAMPLE_PATH')

###############################################################################
# Step 1: Download/Get a BIDS dataset
# -----------------------------------
#
# Get the MNE somato data
bids_root = somato.data_path()
somato_raw_fname = os.path.join(bids_root, 'sub-01', 'meg',
                                'sub-01_task-somato_meg.fif')

subject_id = '01'
task = 'somato'
kind = "meg"

bids_basename = make_bids_basename(subject=subject_id, task=task)

# bids basename is nicely formatted
print(bids_basename)

###############################################################################
# Print the directory tree
print_dir_tree(bids_root)

###############################################################################
# Step 2: Read a BIDS dataset
# ---------------------------
#
# Let's read in the dataset and show off a few features of the
# loading function `read_raw_bids`. Note, this is just one line of code.
bids_fname = bids_basename + "_{}.fif".format(kind)
raw = read_raw_bids(bids_fname, bids_root, verbose=True)

###############################################################################
# `raw.info` has the basic subject metadata
print(raw.info['subject_info'])

# `raw.info` has the PowerLineFrequency loaded in, which should be 50 Hz here
print(raw.info['line_freq'])

# `raw.info` has the sampling frequency loaded in
print(raw.info['sfreq'])

# annotations
print(raw.annotations)

###############################################################################
# .. LINKS
#
# .. _mne_somato_data:
#    https://mne.tools/dev/generated/mne.datasets.somato.data_path.html
#
