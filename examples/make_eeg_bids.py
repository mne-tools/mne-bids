"""
=========================
Use MNE-Bids for EEG Data
=========================

In this example, we use MNE-Bids to create a BIDS-compatible directory of EEG
data. Specifically, we will follow these steps:

1. Download some EEG data from the Open Science Framework (https://osf.io)
2. Load the data, extract information, and save in a new BIDS directory
3. Check the result and compare it with the standard

"""

# Authors: Stefan Appelhoff <stefan.appelhoff@mailbox.org>
# License: BSD (3-clause)

# We are importing everything we need for this example:
import os

from mne import find_events
from mne.io import read_raw_brainvision
from mne.utils import _TempDir

from mne_bids import raw_to_bids
from mne_bids.datasets import download_matchingpennies_subj
from mne_bids.utils import print_dir_tree

###############################################################################
# Step 1: Download the data
# -------------------------
#
# First, we need some data to work with. We will use the example data that is
# provided with the report on the BIDS extension for EEF data: The "Matching
# Pennies" dataset. See here for more information: https://osf.io/cj2dr/
#
# For convenience, MNE-Bids has a function to download this data.

# make a temporary directory to save the data to
data_dir = _TempDir()

###############################################################################
# Now, we download data for participant sub-05 (about 350MB). We could also
# download the whole dataset by specifying `url_dict` as an input to
# `download_matchingpennies_subj`, see the function's docstring for more
# information.
download_matchingpennies_subj(directory=data_dir)

###############################################################################
# Let's see whether the data has been downloaded using a quick visualization
print_dir_tree(data_dir)

###############################################################################
# The data are part of the example suit for the EEG specification of BIDS, so
# they are already in BIDS format as we can see from the file names (also, see
# the `channels.tsv` and `events.tsv` files). However, we want to start from
# scratch and will only use the "pure" EEG data in its BrainVision format with
# a '.eeg' for binary data, a '.vhdr' for header information, and a '.vmrk' for
# specifying the event markers within the data.

###############################################################################
# Step 2: Formatting as BIDS
# --------------------------
#
# We are reading the data using MNE-Python's io module and the
# `read_raw_brainvision` function, which takes a `.vhdr` file as an input.
vhdr_path = os.path.join(data_dir, 'sub-05_task-matchingpennies_eeg.vhdr')
raw = read_raw_brainvision(vhdr_path)
print(raw)

###############################################################################
# By reading the raw data, MNE-Python has also read the `.vmrk` file in which
# the events of the EEG recording are stored. It has automatically created a
# new "stimulus channel" containing the event pulses and appended this to our
# data:
print(raw.ch_names)  # Should be called something like "STI"

###############################################################################
# For BIDS, we specify all events in a dedicated `events.tsv` file. Let's read
# the events from our stimulus channel using `find_events`
events = find_events(raw)

###############################################################################
# Now we have everything to start a new BIDS directory using our data.
# To do that, we can use the high level function `raw_to_bids`, which is the
# core of MNE-BIDS. Generally, `raw_to_bids` tries to extract as much meta data
# as possible from the raw data and then format it in a BIDS compatible way.
# `raw_to_bids` takes a bunch of inputs, most of which are however optional.
# The required inputs are described in the docstring:
#
# .. code-block:: python
#    """
#    Walk over a folder of files and create BIDS compatible folder.
#
#    Parameters
#    ----------
#    subject_id : str
#        The subject name in BIDS compatible format ('01', '02', etc.)
#    task : str
#        Name of the task the data is based on.
#    raw_file : str | instance of mne.Raw
#        The raw data. If a string, it is assumed to be the path to the raw
#        data file. Otherwise it must be an instance of mne.Raw
#    output_path : str
#        The path of the BIDS compatible folder
#     """
# Let's assume the previous file name `sub-05_task-matchingpennies_eeg`
# contained an error and that this was actually `sub-01` and we also want to
# have another task name `MatchingPennies`.
subject_id = '01'
task = 'MatchingPennies'
raw_file = raw  # We can specify the raw file, or the path to the .vhdr file.
output_path = _TempDir()

###############################################################################
# Now we just need to specify a few more EEG details to get something sensible:

# First, tell `MNE-BIDS` that it is working with EEG data:
kind = 'eeg'

# The EEG reference used for the recording
eeg_reference = 'Fz'

# brief description of the event markers present in the data. This will become
# the `trial_type` column in our BIDS `events.tsv`.
event_id = {'left': 1, 'right': 2}

# Now convert our data to be in a new BIDS dataset.
raw_to_bids(subject_id=subject_id, task=task, raw_file=raw_file,
            output_path=output_path, kind=kind,
            eeg_reference=eeg_reference, event_id=event_id)

###############################################################################
# Step 3: Check and compare with standard
# ---------------------------------------
# Now we have written our BIDS directory. Having read the BIDS specification,
# we know that the structure should approximately look like this:
#
# .. code-block:: python
#
#    ---------- CHANGES
#    ---------- dataset_description.json
#    ---------- participants.tsv
#    ---------- README
#    ---------- sub-xx
#        ---------- modality
#            ---------- sub-xx_task-yy_events.tsv
#            ---------- sub-xx_task-yy_modality.ext
#            ---------- sub-xx_task-yy_modality.json
#
#
# Now, what does it actually look like?
print_dir_tree(output_path)

###############################################################################
# The three actual data files `.eeg`, `.vhdr`, and `.vmrk` have their new names
# and the internal pointers within each data file have been updated
# accordingly. In addition to that, MNE-BIDS has created a suitable directory
# structure for us, started an `events.tsv` and `channels.tsv` and made an
# initial `dataset_description` on top!
#
# That's nice and it has saved us a lot of work. However, a few things are not
# yet covered by MNE-BIDS for EEG data and will have to be added by hand. Most
# importantly, the `README` file at the root of the directory, but also
# `CHANGES`, `participants.tsv`, and some more specific entries.
#
# Remember that there is a convenient javascript tool to validate all your BIDS
# directories called the "BIDS-validator", available as a web version and a
# command line tool:
#
# Web version: https://incf.github.io/bids-validator/
#
# Command line tool: https://www.npmjs.com/package/bids-validator
