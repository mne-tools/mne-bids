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
from mne.datasets import somato

from mne_bids import (BIDSPath, read_raw_bids,
                      print_dir_tree)

###############################################################################
# We will be using the `MNE somato data <mne_somato_data_>`_, which
# is already stored in BIDS format.
# For more information, you can checkout the
# respective :ref:`example <ex-convert-mne-sample>`.

###############################################################################
# Step 1: Download/Get a BIDS dataset
# -----------------------------------
#
# Get the MNE somato data
bids_root = somato.data_path()
subject_id = '01'
task = 'somato'
datatype = 'meg'

bids_path = BIDSPath(subject=subject_id, task=task,
                     datatype=datatype, suffix=datatype,
                     root=bids_root)

# bids basename is nicely formatted
print(bids_path)

###############################################################################
# Print the directory tree
print_dir_tree(bids_root)

###############################################################################
# Step 2: Read a BIDS dataset
# ---------------------------
#
# Let's read in the dataset and show off a few features of the
# loading function `read_raw_bids`. Note, this is just one line of code.
raw = read_raw_bids(bids_path=bids_path, verbose=True)

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
