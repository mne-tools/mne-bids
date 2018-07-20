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

from mne.utils import _TempDir

from mne_bids.datasets import download_matchingpennies_subject

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
# `download_matchingpennies_subject`, see the function's docstring for more
# information.
download_matchingpennies_subject(directory=data_dir)

###############################################################################
# Let's see whether the data has been downloaded:
os.listdir(data_dir)

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


###############################################################################
# Step 3: Check and compare with standard
# ---------------------------------------
#
