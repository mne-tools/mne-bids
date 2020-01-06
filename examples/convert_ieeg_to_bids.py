"""
===================================
07. Convert iEEG data to BIDS format
===================================

In this example, we use MNE-BIDS to create a BIDS-compatible directory of iEEG
data. Specifically, we will follow these steps:

1. Download some iEEG data from the MNE-Python API
   `PhysioBank database <https://physionet.org/physiobank/database>`_.
    `https://mne.tools/stable/auto_tutorials/misc/plot_ecog.html#sphx-glr-auto-tutorials-misc-plot-ecog-py`_.

2. Load the data, extract information, and save in a new BIDS directory

3. Check the result and compare it with the standard

The iEEG data will be pretty similar to the iEEG data with
the addition of extra elements in the electrodes.tsv and
coord_system.json files.
"""

# Authors: Adam Li <adam2392@gmail.com>
# License: BSD (3-clause)

import os
import tempfile

# need to install visualization
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.io import loadmat

from mne_bids import write_raw_bids, make_bids_basename, read_raw_bids
from mne_bids.utils import print_dir_tree

###############################################################################
# Step 1: Download the data
# -------------------------
#
# First, we need some data to work with. We will use the
# data downloaded via MNE-Python's API.
#
# `https://mne.tools/stable/generated/mne.datasets.misc.data_path.html#mne.datasets.misc.data_path`_.
#
# Conveniently, there is already a data loading function available with
# MNE-Python:

# get MNE directory w/ example data
mne_dir = mne.get_config('MNE_DATASETS_SAMPLE_PATH')

mat = loadmat(mne.datasets.misc.data_path() + '/ecog/sample_ecog.mat')
ch_names = mat['ch_names'].tolist()
ch_names = [x.strip() for x in ch_names]
elec = mat['elec']  # electrode positions given in meters
# Now we make a montage stating that the iEEG contacts are in MRI
# coordinate system.
montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, elec)),
                                        coord_frame='mri')
print('Created %s channel positions' % len(ch_names))
print(ch_names)

###############################################################################
# The electrode data are in the Matlab format: '.mat'.
# This is easy to read in with
# `scipy.io.loadmat` function. We also need to get some
# sample EEG data, so we will just generate random data
# from white noise.
# Here is where you would use your own data if you had it.
eegdata = np.random.rand(len(ch_names), 1000)

# However, apart from the data format, we need to build
# a directory structure and supply meta data
# files to properly *bidsify* this data.
info = mne.create_info(ch_names, 1000., 'ecog', montage=montage)
raw = mne.io.RawArray(eegdata, info)

###############################################################################
# Step 2: Formatting as BIDS
# --------------------------
#
# Let's start by formatting a single subject.
###############################################################################
# With this step, we have everything to start a new BIDS directory using
# our data. To do that, we can use :func:`write_raw_bids`
# Generally, :func:`write_raw_bids` tries to extract as much
# meta data as possible from the raw data and then formats it in a BIDS
# compatible way. :func:`write_raw_bids` takes a bunch of inputs, most of
# which are however optional. The required inputs are:
#
# * :code:`raw`
# * :code:`bids_basename`
# * :code:`bids_root`
#
# ... as you can see in the docstring:
print(write_raw_bids.__doc__)

###############################################################################
# Let us initialize some of the necessary data for the subject
# There is a subject, and specific task for the dataset
subject_id = '001'  # zero padding to account for >100 subjects in this dataset
task = 'testresteyes'
bids_root = os.path.join(mne_dir, 'eegmmidb_bids')

###############################################################################
# Now we just need to specify a few more EEG details to get something sensible:
# Brief description of the event markers present in the data. This will become
# the `trial_type` column in our BIDS `events.tsv`. We know about the event
# meaning from the documentation on PhysioBank

# # Now convert our data to be in a new BIDS dataset.
bids_basename = make_bids_basename(subject=subject_id,
                                   task=task,
                                   acquisition="ecog")

# need to set the filenames if we are initializing data from array
with tempfile.TemporaryDirectory() as tmp_root:
    tmp_fpath = os.path.join(tmp_root, "test_raw.fif")
    raw.save(tmp_fpath)
    raw = mne.io.read_raw_fif(tmp_fpath)

    write_raw_bids(raw, bids_basename,
                   bids_root,
                   overwrite=True)

###############################################################################
# Step 3: Check and compare with standard
# ---------------------------------------
# Now we have written our BIDS directory.
print_dir_tree(bids_root)

###############################################################################
# MNE-BIDS has created a suitable directory structure for us, and among other
# meta data files, it started an `events.tsv` and `channels.tsv` and made an
# initial `dataset_description` on top!
#
# Now it's time to manually check the BIDS directory and the meta files to add
# all the information that MNE-BIDS could not infer. For instance, you must
# describe EEGReference and EEGGround yourself. It's easy to find these by
# searching for "n/a" in the sidecar files.
#
# Remember that there is a convenient javascript tool to validate all your BIDS
# directories called the "BIDS-validator", available as a web version and a
# command line tool:
#
# Web version: https://bids-standard.github.io/bids-validator/
#
# Command line tool: https://www.npmjs.com/package/bids-validator

###############################################################################
# Step 4: Plot output channels and check that they match!
# -------------------------------------------------------
# Now we have written our BIDS directory.
bids_fname = bids_basename + "_ieeg.vhdr"
raw = read_raw_bids(bids_fname, bids_root=bids_root)

fig = plt.figure()
ax2d = fig.add_subplot(121)
raw.plot_sensors(ch_type='ecog', axes=ax2d)
# import mpl_toolkits
# ax3d = fig.add_subplot(122, projection='3d')
# raw.plot_sensors(ch_type="ecog", axes=ax3d, kind='3d')
plt.show()
