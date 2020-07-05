"""
===================================
04. Convert EEG data to BIDS format
===================================

In this example, we use MNE-BIDS to create a BIDS-compatible directory of EEG
data. Specifically, we will follow these steps:

1. Download some EEG data from the
   `PhysioBank database <https://physionet.org/physiobank/database>`_.

2. Load the data, extract information, and save in a new BIDS directory ...
   once for a single subject, and then while looping over different subjects.

3. Check the result and compare it with the standard

4. Cite ``mne-bids``

.. currentmodule:: mne_bids
"""

# Authors: Stefan Appelhoff <stefan.appelhoff@mailbox.org>
# License: BSD (3-clause)

###############################################################################
# We are importing everything we need for this example:
import os
import shutil as sh

import mne
from mne.datasets import eegbci

from mne_bids import write_raw_bids, make_bids_basename
from mne_bids.utils import print_dir_tree

###############################################################################
# Step 1: Download the data
# -------------------------
#
# First, we need some data to work with. We will use the
# `EEG Motor Movement/Imagery Dataset <https://doi.org/10.13026/C28G6P>`_
# available on the PhysioBank database.
#
# The data consists of 109 volunteers performing 14 experimental runs each.
# For each subject, there were two baseline tasks (i) eyes open, (ii) eyes
# closed, as well as four different motor imagery tasks.
# For the present example, we will show how to format the data for two subjects
# and selected tasks to comply with the Brain Imaging Data Structure
# (`BIDS <http://bids.neuroimaging.io/>`_).
#
# Conveniently, there is already a data loading function available with
# MNE-Python:

# Define which tasks we want to download.
tasks = [
    2,  # This is 2 minutes of eyes closed rest
    4,  # This is run #1 of imagining to close left or right fist
    12  # This is run #2 of imagining to close left or right fist
]

# Download the data for subjects 1 and 2
subjects = [1, 2]
for subj in subjects:
    eegbci.load_data(subject=subj, runs=tasks, update_path=True)

###############################################################################
# Let's see whether the data has been downloaded using a quick visualization
# of the directory tree.

# get MNE directory with example data
mne_data_dir = mne.get_config('MNE_DATASETS_EEGBCI_PATH')
data_dir = os.path.join(mne_data_dir, 'MNE-eegbci-data')

print_dir_tree(data_dir)

###############################################################################
# The data are in the `European Data Format <https://www.edfplus.info/>`_ with
# the ``.edf`` extension, which is good for us because next to the
# `BrainVision format`_, EDF is one of the recommended file formats for EEG
# data in BIDS format.
# However, apart from the data format, we need to build a directory structure
# and supply meta data files to properly *bidsify* this data.

###############################################################################
# Step 2: Formatting as BIDS
# --------------------------
#
# Let's start by formatting a single subject. We are reading the data using
# MNE-Python's io module and the :func:`mne.io.read_raw_edf` function. Note
# that we must use `preload=False`, the default in MNE-Python. It prevents the
# data from being loaded and modified when converting to BIDS.

# Load the data from "2 minutes eyes closed rest"
edf_path = eegbci.load_data(subject=1, runs=2)[0]
raw = mne.io.read_raw_edf(edf_path, preload=False)

# For converting the data to BIDS, we need to convert the the annotations
# stored in the file to a 2D numpy array of events.
events, event_id = mne.events_from_annotations(raw)

###############################################################################
# For the sake of the example we will also pretend that we have the electrode
# coordinates for the data recordings.
# We will use a coordinates file from the MNE testing data in `CapTrak`_
# format.
#
# .. note:: The ``*electrodes.tsv`` and ``*coordsystem.json`` files in BIDS are
#           intended to carry information about digitized (i.e., *measured*)
#           electrode positions on the scalp of the research subject. Do *not*
#           (!) use these files to store "template" or "idealized" electrode
#           positions, like those that can be obtained from
#           :func:`mne.channels.make_standard_montage`!
#

# Get the electrode coordinates
testing_data = mne.datasets.testing.data_path()
captrak_path = os.path.join(testing_data, 'montage', 'captrak_coords.bvct')
montage = mne.channels.read_dig_captrak(captrak_path)

# Rename the montage channel names only for this example, because as said
# before, coordinate and EEG data were not actually collected together
# Do *not* do this for your own data.
montage.rename_channels(dict(zip(montage.ch_names, raw.ch_names)))

# "attach" the electrode coordinates to the `raw` object
raw.set_montage(montage)

# show the electrode positions
raw.plot_sensors()

###############################################################################
# With these steps, we have everything to start a new BIDS directory using
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
# We loaded ``S001R02.edf``, which corresponds to subject 1 in the second task.
# The second task was to rest with closed eyes.
subject_id = '001'  # zero padding to account for >100 subjects in this dataset
task = 'resteyesclosed'
bids_root = os.path.join(mne_data_dir, 'eegmmidb_bids')

# Start with a clean directory in case the directory existed beforehand
sh.rmtree(bids_root, ignore_errors=True)

###############################################################################
# Now we just need to specify a few more EEG details to get something sensible:

# Brief description of the event markers present in the data. This will become
# the `trial_type` column in our BIDS `events.tsv`. We know about the event
# meaning from the documentation on PhysioBank.
trial_type = {'rest': 0, 'imagine left fist': 1, 'imagine right fist': 2}

# Now convert our data to be in a new BIDS dataset.
bids_basename = make_bids_basename(subject=subject_id, task=task)
write_raw_bids(raw, bids_basename, bids_root, event_id=trial_type,
               events_data=events, overwrite=True)

###############################################################################
# What does our fresh BIDS directory look like?
print_dir_tree(bids_root)

###############################################################################
# Looks good so far, let's convert the data for all tasks and subjects.

# Start with a clean directory
sh.rmtree(bids_root)

# Some initial information that we found in the PhysioBank documentation
task_names = {
    2: 'resteyesclosed',
    4: 'imaginefists',  # run 1
    12: 'imaginefists'  # run 2
}

run_mapping = {
    2: None,  # for resting eyes closed task, there was only one run
    4: '1',
    12: '2'
}

# The electrode coordinate files would be different for each subject, but in
# our example they are all the same
subj2electrode_files = {subj: captrak_path for subj in subjects}

# Now go over subjects and *bidsify*
for subj in subjects:
    for task in tasks:
        # Load the data
        edf_path = eegbci.load_data(subject=subj, runs=task)[0]
        raw = mne.io.read_raw_edf(edf_path, preload=False, stim_channel=None)

        # get events
        events, event_id = mne.events_from_annotations(raw)

        # attach electrode coordinates depending on subject
        montage = mne.channels.read_dig_captrak(subj2electrode_files[subj])
        montage.rename_channels(dict(zip(montage.ch_names, raw.ch_names)))
        raw.set_montage(montage)

        # convert
        bids_basename = make_bids_basename(subject='{:03}'.format(subj),
                                           task=task_names[task],
                                           run=run_mapping[task])
        write_raw_bids(raw, bids_basename, bids_root, event_id=trial_type,
                       events_data=events, overwrite=True)

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
readme = os.path.join(bids_root, 'README')
with open(readme, 'r') as fid:
    text = fid.read()
print(text)

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
#
# .. _BrainVision format: https://www.brainproducts.com/productdetails.php?id=21&tab=5
# .. _CapTrak: http://www.fieldtriptoolbox.org/faq/how_are_the_different_head_and_mri_coordinate_systems_defined/#details-of-the-captrak-coordinate-system
