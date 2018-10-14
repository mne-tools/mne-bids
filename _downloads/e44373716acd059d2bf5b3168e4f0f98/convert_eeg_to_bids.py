"""
===============================
Convert EEG data to BIDS format
===============================

In this example, we use MNE-BIDS to create a BIDS-compatible directory of EEG
data. Specifically, we will follow these steps:

1. Download some EEG data from the
   `PhysioBank database <https://physionet.org/physiobank/database>`_.

2. Load the data, extract information, and save in a new BIDS directory

3. Check the result and compare it with the standard

"""

# Authors: Stefan Appelhoff <stefan.appelhoff@mailbox.org>
# License: BSD (3-clause)

###############################################################################
# We are importing everything we need for this example:
import os
import shutil as sh

from mne.datasets import eegbci
from mne.io import read_raw_edf

from mne_bids import raw_to_bids
from mne_bids.utils import print_dir_tree

###############################################################################
# Step 1: Download the data
# -------------------------
#
# First, we need some data to work with. We will use the
# `EEG Motor Movement/Imagery Dataset <https://doi.org/10.13026/C28G6P>`_
# available on the PhysioBank database.
#
# The data consists of 109 volunteers performing 14 experimental runs each. For
# each subject, there were two baseline tasks (1. eyes open, 2. eyes closed) as
# well as four different motor imagery tasks. For the present example, we will
# show how to format the data for two subjects and selected tasks to comply
# with the Brain Imaging Data Structure
# (`BIDS <http://bids.neuroimaging.io/>`).
#
# Conveniently, there is already a data loading function available with
# MNE-Python:

# Make a directory to save the data to
home = os.path.expanduser('~')
mne_dir = os.path.join(home, 'mne_data')
if not os.path.exists(mne_dir):
    os.makedirs(mne_dir)

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

###############################################################################
# Let's see whether the data has been downloaded using a quick visualization
# of the directory tree.

data_dir = os.path.join(mne_dir, 'MNE-eegbci-data')
print_dir_tree(data_dir)

###############################################################################
# The data are in the `European Data Format <https://www.edfplus.info/>`_
# '.edf', which is good for us because next to the BrainVision format, EDF is
# one of the recommended file formats for EEG BIDS. However, apart from the
# data format, we need to build a directory structure and supply meta data
# files to properly *bidsify* this data.

###############################################################################
# Step 2: Formatting as BIDS
# --------------------------
#
# Let's start by formatting a single subject. We are reading the data using
# MNE-Python's io module and the `read_raw_edf` function.
data_dir = os.path.join(data_dir, 'physiobank', 'database', 'eegmmidb')
edf_path = os.path.join(data_dir, 'S001', 'S001R02.edf')
raw = read_raw_edf(edf_path, preload=True)
print(raw)

###############################################################################
# With this simple step we have everything to start a new BIDS directory using
# our data. To do that, we can use the high level function `raw_to_bids`, which
# is the core of MNE-BIDS. Generally, `raw_to_bids` tries to extract as much
# meta data as possible from the raw data and then formats it in a BIDS
# compatible way. `raw_to_bids` takes a bunch of inputs, most of which are
# however optional. The required inputs are:
#
# * subject_id
# * task
# * raw_file
# * output_path
#
# ... as you can see in the docstring:
print(raw_to_bids.__doc__)

###############################################################################
# We loaded 'S001R02.edf', which corresponds to subject 1 in the second task.
# The second task was to rest with closed eyes.
subject_id = '001'  # zero padding to account for >100 subjects in this dataset
task = 'resteyesclosed'
raw_file = raw
output_path = os.path.join(home, 'mne_data', 'eegmmidb_bids')

# Clean up existing directories for the sake of the example
if os.path.exists(output_path):
    sh.rmtree(output_path)

###############################################################################
# Now we just need to specify a few more EEG details to get something sensible:

# First, tell `MNE-BIDS` that it is working with EEG data:
kind = 'eeg'

# Brief description of the event markers present in the data. This will become
# the `trial_type` column in our BIDS `events.tsv`. We know about the event
# meaning from the documentation on PhysioBank
trial_type = {'rest': 0, 'imagine left fist': 1, 'imagine right fist': 2}

# Now convert our data to be in a new BIDS dataset.
raw_to_bids(subject_id=subject_id, task=task, raw_file=raw_file,
            output_path=output_path, kind=kind, event_id=trial_type)

###############################################################################
# What does our fresh BIDS directory look like?
print_dir_tree(output_path)

###############################################################################
# Looks good so far, let's convert the data for all tasks and subjects.

# Start with a clean directory
sh.rmtree(output_path)

# Some initial information that we found in the PhysioBank documentation
task_names = {2: 'resteyesclosed',
              4: 'imaginefists',  # run 1
              12: 'imaginefists'  # run 2
              }

run_mapping = {2: None,  # for resting eyes closed task, there was only one run
               4: '1',
               12: '2'
               }

# Now go over subjects and *bidsify*
for subj_idx in [1, 2]:
    for task_idx in [2, 4, 12]:
        # Load the data
        edf_path = os.path.join(data_dir,
                                'S{:03}'.format(subj_idx),
                                'S{:03}R{:02}.edf'.format(subj_idx, task_idx))
        raw = read_raw_edf(edf_path, preload=True)

        # `kind` and `trial_type` were already defined above
        raw_to_bids(subject_id='{:03}'.format(subj_idx),
                    task=task_names[task_idx],
                    run=run_mapping[task_idx],
                    raw_file=raw,
                    output_path=output_path,
                    kind=kind,
                    event_id=trial_type,
                    overwrite=False
                    )

###############################################################################
# Step 3: Check and compare with standard
# ---------------------------------------
# Now we have written our BIDS directory.
print_dir_tree(output_path)

###############################################################################
# MNE-BIDS has created a suitable directory structure for us, and among other
# meta data files, it started an `events.tsv` and `channels.tsv` and made an
# initial `dataset_description` on top!
#
# Now it's time to manually check the BIDS directory and the meta files to add
# all the information that MNE-BIDS could not infer. These places are marked
# with "n/a".
#
# Remember that there is a convenient javascript tool to validate all your BIDS
# directories called the "BIDS-validator", available as a web version and a
# command line tool:
#
# Web version: https://bids-standard.github.io/bids-validator/
#
# Command line tool: https://www.npmjs.com/package/bids-validator
