"""
.. currentmodule:: mne_bids

====================================
08. Convert iEEG data to BIDS format
====================================

In this example, we use MNE-BIDS to create a BIDS-compatible directory of iEEG
data. Specifically, we will follow these steps:

1. Download some iEEG data from the
   `MNE-ECoG ex <https://mne.tools/stable/auto_tutorials/misc/plot_ecog>`_.

2. Load the data, extract information, and save in a new BIDS directory.

3. Check the result and compare it with the standard.

4. Cite MNE-BIDS

5. Confirm that written iEEG coordinates are the
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

import os.path as op
import shutil
from pprint import pprint
from collections import OrderedDict

import numpy as np

import mne
from mne_bids import (write_raw_bids, BIDSPath,
                      read_raw_bids, print_dir_tree)

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

misc_path = mne.datasets.misc.data_path(force_update=True)


# The electrode coords data are in the tsv file format
# which is easily read in using numpy
fname = misc_path + '/ecog/sample_ecog_electrodes.tsv'
data = np.loadtxt(fname, dtype=str, delimiter='\t',
                  comments=None, encoding='utf-8')
column_names = data[0, :]
info = data[1:, :]
electrode_tsv = OrderedDict()
for i, name in enumerate(column_names):
    electrode_tsv[name] = info[:, i].tolist()

# load in channel names
ch_names = electrode_tsv['name']
# load in the xyz coordinates as a float
elec = np.empty(shape=(len(ch_names), 3))
for ind, axis in enumerate(['x', 'y', 'z']):
    elec[:, ind] = list(map(float, electrode_tsv[axis]))

###############################################################################
# Now we make a montage stating that the iEEG contacts are in MRI
# coordinate system.
montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, elec)),
                                        coord_frame='mri')
print('Created %s channel positions' % len(ch_names))
print(dict(zip(ch_names, elec)))

###############################################################################
# We will load a :class:`mne.io.Raw` object and
# use the montage we created.
info = mne.create_info(ch_names, 1000., 'ecog')
raw = mne.io.read_raw_edf(misc_path + '/ecog/sample_ecog.edf')
raw.info['line_freq'] = 60  # specify power line frequency as required by BIDS
raw.set_channel_types({ch: 'ecog' for ch in raw.ch_names})

# set the bad channels
raw.info['bads'].extend(['BTM1', 'BTM2', 'BTM3', 'BTM4', 'BTM5', 'BTM6',
                         'BTP1', 'BTP2', 'BTP3', 'BTP4', 'BTP5', 'BTP6',
                         'EKGL', 'EKGR'])

# set montage
raw.set_montage(montage, on_missing='warn')

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

# get MNE directory w/ example data
mne_data_dir = mne.get_config('MNE_DATASETS_MISC_PATH')

# There is the root directory for where we will write our data.
bids_root = op.join(mne_data_dir, 'ieegmmidb_bids')

###############################################################################
# To ensure the output path doesn't contain any leftover files from previous
# tests and example runs, we simply delete it.
#
# .. warning:: Do not delete directories that may contain important data!
#

if op.exists(bids_root):
    shutil.rmtree(bids_root)

###############################################################################
# Now we just need to specify a few iEEG details to make things work:
# We need the basename of the dataset. In addition, write_raw_bids
# requires a `filenames` of the Raw object to be non-empty, so since we
# initialized the dataset from an array, we need to do a hack where we
# temporarily save the data to disc before reading it back in.

# Now convert our data to be in a new BIDS dataset.
bids_path = BIDSPath(subject=subject_id,
                     task=task, acquisition="ecog", root=bids_root)

# write `raw` to BIDS and anonymize it into BrainVision format
write_raw_bids(raw, bids_path, anonymize=dict(daysback=30000),
               overwrite=True)

###############################################################################
# Step 3: Check and compare with standard
# ---------------------------------------

# Now we have written our BIDS directory.
print_dir_tree(bids_root)

###############################################################################
# Step 4: Cite mne-bids
# ---------------------
# We can see that the appropriate citations are already written in the README.
# If you are preparing a manuscript, please make sure to also cite MNE-BIDS
# there.
readme = op.join(bids_root, 'README')
with open(readme, 'r', encoding='utf-8-sig') as fid:
    text = fid.read()
print(text)

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
# Step 5: Plot output channels and check that they match!
# -------------------------------------------------------
#
# Now we have written our BIDS directory. We can use
# :func:`read_raw_bids` to read in the data.

# read in the BIDS dataset and plot the coordinates
raw = read_raw_bids(bids_path=bids_path)

# get the first 5 channels and show their locations
# this should match what was printed earlier.
picks = mne.pick_types(raw.info, ecog=True)
dig = [raw.info['dig'][pick] for pick in picks]
chs = [raw.info['chs'][pick] for pick in picks]
pos = np.array([ch['r'] for ch in dig[:5]])
ch_names = np.array([ch['ch_name'] for ch in chs[:5]])

print("The channel montage after writing into BIDS: ")
pprint(dig[0:5])
print("The channel coordinates after writing into BIDS: ")
pprint([x for x in zip(ch_names, pos)])

# make a plot of the sensors in 2D plane
raw.plot_sensors(ch_type='ecog')
