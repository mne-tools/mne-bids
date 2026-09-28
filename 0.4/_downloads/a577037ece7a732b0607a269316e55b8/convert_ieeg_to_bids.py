"""
====================================
08. Convert iEEG data to BIDS format
====================================

In this example, we use MNE-BIDS to create a BIDS-compatible directory of iEEG
data. Specifically, we will follow these steps:

1. Download some iEEG data from the
   `PhysioBank database <https://physionet.org/physiobank/database>`_
   and the
   `MNE-ECoG ex <https://mne.tools/stable/auto_tutorials/misc/plot_ecog>`_.

2. Load the data, extract information, and save in a new BIDS directory.

3. Check the result and compare it with the standard.

4. Confirm that written iEEG coordinates are the
   same before :func:`write_raw_bids` was called.

The iEEG data will be written by :func:`write_raw_bids` with
the addition of extra metadata elements in the following files:

    * sidecar.json
    * electrodes.tsv
    * coord_system.json
    * events.tsv
    * channels.tsv

Compared to EEG data, the main differences are within the
coord_system and electrodes files.
For more information on these files,
refer to the iEEG-BIDS specification.
"""

# Authors: Adam Li <adam2392@gmail.com>
# License: BSD (3-clause)

import os
import tempfile
from pprint import pprint

import numpy as np
from scipy.io import loadmat

import mne
from mne_bids import write_raw_bids, make_bids_basename, read_raw_bids
from mne_bids.utils import print_dir_tree

###############################################################################
# Step 1: Download the data
# -------------------------
#
# First, we need some data to work with. We will use the
# data downloaded via MNE-Python's API.
#
# `<https://mne.tools/stable/generated/mne.datasets.misc.data_path>`_.
#
# Conveniently, there is already a data loading function available with
# MNE-Python:

# get MNE directory w/ example data
mne_dir = mne.get_config('MNE_DATASETS_SAMPLE_PATH')

# The electrode coords data are in the Matlab format: '.mat'.
# This is easy to read in with :func:`scipy.io.loadmat` function.
mat = loadmat(mne.datasets.misc.data_path() + '/ecog/sample_ecog.mat')
ch_names = mat['ch_names'].tolist()
ch_names = [x.strip() for x in ch_names]
elec = mat['elec']  # electrode positions given in meters

###############################################################################
# Now we make a montage stating that the iEEG contacts are in MRI
# coordinate system.
montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, elec)),
                                        coord_frame='mri')
print('Created %s channel positions' % len(ch_names))
print(dict(zip(ch_names, elec)))

###############################################################################
# We also need to get some sample iEEG data,
# so we will just generate random data from white noise.
# Here is where you would use your own data if you had it.
ieegdata = np.random.rand(len(ch_names), 1000)

# We will create a :func:`mne.io.RawArray` object and
# use the montage we created.
info = mne.create_info(ch_names, 1000., 'ecog')
raw = mne.io.RawArray(ieegdata, info)
raw.set_montage(montage)

###############################################################################
# Let us confirm what our channel coordinates look like.

# make a plot of the sensors in 2D plane
raw.plot_sensors(ch_type='ecog')

# Get the first 5 channels and show their locations.
picks = mne.pick_types(raw.info, ecog=True)
dig = [raw.info['dig'][pick] for pick in picks]
chs = [raw.info['chs'][pick] for pick in picks]
pos = np.array([ch['r'] for ch in dig[:5]])
ch_names = np.array([ch['ch_name'] for ch in chs[:5]])
print("The channel coordinates before writing into BIDS: ")
pprint([x for x in zip(ch_names, pos)])

###############################################################################
# Step 2: Formatting as BIDS
# --------------------------
#
# Now, let us format the `Raw` object into BIDS.

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
# There is a subject, and specific task for the dataset.
subject_id = '001'  # zero padding to account for >100 subjects in this dataset
task = 'testresteyes'

# There is the root directory for where we will write our data.
bids_root = os.path.join(mne_dir, 'ieegmmidb_bids')

###############################################################################
# Now we just need to specify a few iEEG details to make things work:
# We need the basename of the dataset. In addition, write_raw_bids
# requires a `filenames` of the Raw object to be non-empty, so since we
# initialized the dataset from an array, we need to do a hack where we
# temporarily save the data to disc before reading it back in.

# Now convert our data to be in a new BIDS dataset.
bids_basename = make_bids_basename(subject=subject_id,
                                   task=task,
                                   acquisition="ecog")

# need to set the filenames if we are initializing data from array
with tempfile.TemporaryDirectory() as tmp_root:
    tmp_fpath = os.path.join(tmp_root, "test_raw.fif")
    raw.save(tmp_fpath)
    raw = mne.io.read_raw_fif(tmp_fpath)

    # write `raw` to BIDS and anonymize it into BrainVision format
    write_raw_bids(raw, bids_basename, bids_root=bids_root,
                   anonymize=dict(daysback=30000), overwrite=True)

###############################################################################
# Step 3: Check and compare with standard
# ---------------------------------------

# Now we have written our BIDS directory.
print_dir_tree(bids_root)

###############################################################################
# MNE-BIDS has created a suitable directory structure for us, and among other
# meta data files, it started an `events.tsv` and `channels.tsv` and made an
# initial `dataset_description.json` on top!
#
# Now it's time to manually check the BIDS directory and the meta files to add
# all the information that MNE-BIDS could not infer. For instance, you must
# describe iEEGReference and iEEGGround yourself. It's easy to find these by
# searching for "n/a" in the sidecar files.
#
# `$ grep -i 'n/a' <bids_root>`
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
#
# Now we have written our BIDS directory. We can use
# :func:`read_raw_bids` to read in the data.

# read in the BIDS dataset and plot the coordinates
bids_fname = bids_basename + "_ieeg.vhdr"
raw = read_raw_bids(bids_fname, bids_root=bids_root)

# get the first 5 channels and show their locations
# this should match what was printed earlier.
picks = mne.pick_types(raw.info, ecog=True)
dig = [raw.info['dig'][pick] for pick in picks]
chs = [raw.info['chs'][pick] for pick in picks]
pos = np.array([ch['r'] for ch in dig[:5]])
ch_names = np.array([ch['ch_name'] for ch in chs[:5]])
print("The channel coordinates after writing into BIDS: ")
pprint([x for x in zip(ch_names, pos)])

# make a plot of the sensors in 2D plane
raw.plot_sensors(ch_type='ecog')
