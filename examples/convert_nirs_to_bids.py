"""
====================================
13. Convert NIRS data to BIDS format
====================================

In this example, we use MNE-BIDS to create a BIDS-compatible directory of NIRS
data. Specifically, we will follow these steps:

1. Download some NIRS data from a github repository

2. Load the data, extract information, and save it in a new BIDS directory.

3. Check the result and compare it with the standard.

4. Cite ``mne-bids``.

.. currentmodule:: mne_bids

.. _BrainVision format: https://www.brainproducts.com/productdetails.php?id=21&tab=5
.. _CapTrak: https://www.fieldtriptoolbox.org/faq/coordsys/#details-of-the-captrak-coordinate-system

"""  # noqa: E501

# Authors: Robert Luke <code@robertluke.net>
#
# License: BSD-3-Clause

# %%
# We are importing everything we need for this example:
import os.path as op
import shutil
import tempfile

import pooch
import mne

from mne_bids import write_raw_bids, BIDSPath, print_dir_tree
from mne_bids.stats import count_events

# %%
# Download the data
# -----------------
#
# First, we need some data to work with. We will use the
# `Finger Tapping Dataset <https://github.com/rob-luke/BIDS-NIRS-Tapping>`_
# available on GitHub.
#
# The data consists of 109 volunteers performing 14 experimental runs each.
# For each subject, there were two baseline tasks (i) eyes open, (ii) eyes
# closed, as well as four different motor imagery tasks.
#
# In this example, we will download the data for a single subject doing the
# baseline task "eyes closed" and format it to the Brain Imaging Data Structure
# (`BIDS <https://bids.neuroimaging.io/>`_).
#

# Create a temporary directory to store the data in
data_dir = tempfile.mkdtemp()

# Download the data for subject 1.

url = ('https://github.com/rob-luke/BIDS-NIRS-Tapping/raw/'
       '01-Raw-to-SNIRF/sub-01/nirs/sub-01_task-tapping_nirs.snirf')
hash = '627eea5777f387512dc48ec8e2d329132bfdc943b21ebfa7b432141b0b0dbf59'

file_path = pooch.retrieve(
    url=url,
    known_hash=f'sha256:{hash}',
    fname='sub-01_task-tapping_nirs.snirf',
    path=data_dir,
    progressbar=True
)

# Let's see whether the data has been downloaded using a quick visualization
# of the directory tree.
print_dir_tree(data_dir)

# %%
# The data are in the `European Data Format <https://www.edfplus.info/>`_ with
# the ``.edf`` extension, which is good for us because next to the
# `BrainVision format`_, EDF is one of the recommended file formats for EEG
# data in BIDS format.
#
# However, apart from the data format, we need to build a directory structure
# and supply meta data files to properly *bidsify* this data.
#
# We will do exactly that in the next step.

# %%
# Convert to BIDS
# ---------------
#
# Let's start with loading the data and extracting the events.
# We are reading the data using MNE-Python's ``io`` module and the
# :func:`mne.io.read_raw_edf` function.
# Note that we must use the ``preload=False`` parameter, which is the default
# in MNE-Python.
# It prevents the data from being loaded and modified when converting to BIDS.

# Load the data from "2 minutes eyes closed rest"
raw = mne.io.read_raw_snirf(file_path, preload=False)
raw.info['line_freq'] = 50  # specify power line frequency as required by BIDS

# %%
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


# Rename the montage channel names only for this example, because as said
# before, coordinate and EEG data were not actually collected together
# Do *not* do this for your own data.

# "attach" the electrode coordinates to the `raw` object
# Note that this only works for some channel types (EEG/sEEG/ECoG/DBS/fNIRS)

# show the electrode positions
raw.plot_sensors()

# %%
# I also like to rename the triggers to something meaningful and
# set the duration of each stimulus

trigger_info = {'1.0': 'Control',
                '2.0': 'Tapping/Left',
                '3.0': 'Tapping/Right'}
raw.annotations.rename(trigger_info)
raw.annotations.set_durations(5.0)


# %%
# With these steps, we have everything to start a new BIDS directory using
# our data.
#
# To do that, we can use :func:`write_raw_bids`
#
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

# %%
# We loaded ``S001R02.edf``, which corresponds to subject 1 in the second run.
# In the second run of the experiment, the task was to rest with closed eyes.

# zero padding to account for >100 subjects in this dataset
subject_id = '01'

# define a task name and a directory where to save the data to
task = 'Tapping'
bids_root = tempfile.mkdtemp()

# %%
# To ensure the output path doesn't contain any leftover files from previous
# tests and example runs, we simply delete it.
#
# .. warning:: Do not delete directories that may contain important data!
#

if op.exists(bids_root):
    shutil.rmtree(bids_root)

# %%
# The data contains annotations; which will be converted to events
# automatically by MNE-BIDS when writing the BIDS data:

print(raw.annotations)

# %%
# Finally, let's write the BIDS data!

bids_path = BIDSPath(subject=subject_id, task=task, root=bids_root)
write_raw_bids(raw, bids_path, overwrite=True)

# %%
# What does our fresh BIDS directory look like?
print_dir_tree(bids_root)

# %%
# Finally let's get an overview of the events on the whole dataset

counts = count_events(bids_root)
counts

# %%
# We can see that MNE-BIDS wrote several important files related to subject 1
# for us:
#
# * ``electrodes.tsv`` containing the electrode coordinates and
#   ``coordsystem.json``, which contains the metadata about the electrode
#   coordinates.
# * The actual EDF data file (now with a proper BIDS name) and an accompanying
#   ``*_eeg.json`` file that contains metadata about the EEG recording.
# * The ``*scans.json`` file lists all data recordings with their acquisition
#   date. This file becomes more handy once there are multiple sessions and
#   recordings to keep track of.
# * And finally, ``channels.tsv`` and ``events.tsv`` which contain even further
#   metadata.
#
# Next to the subject specific files, MNE-BIDS also created several experiment
# specific files. However, we will not go into detail for them in this example.
#
# Cite mne-bids
# -------------
# After a lot of work was done by MNE-BIDS, it's fair to cite the software
# when preparing a manuscript and/or a dataset publication.
#
# We can see that the appropriate citations are already written in the
# ``README`` file.
#
# If you are preparing a manuscript, please make sure to also cite MNE-BIDS
# there.
readme = op.join(bids_root, 'README')
with open(readme, 'r', encoding='utf-8-sig') as fid:
    text = fid.read()
print(text)


# %%
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
