"""
====================================
13. Convert NIRS data to BIDS format
====================================

In this example, we use MNE-BIDS to create a BIDS-compatible directory of NIRS
data. Specifically, we will follow these steps:

1. Download some NIRS data

2. Load the data, extract information, and save it in a new BIDS directory.

3. Check the result and compare it with the standard.

4. Cite ``mne-bids``.

.. currentmodule:: mne_bids

"""  # noqa: E501

# Authors: Robert Luke <code@robertluke.net>
#
# License: BSD-3-Clause

# %%
# We are importing everything we need for this example:
import os.path as op
import pathlib
import shutil

import mne
from mne_nirs import datasets  # For convenient downloading of example data

from mne_bids import write_raw_bids, BIDSPath, print_dir_tree
from mne_bids.stats import count_events

# %%
# Download the data
# -----------------
#
# First, we need some data to work with. We will use the
# `Finger Tapping Dataset <https://github.com/rob-luke/BIDS-NIRS-Tapping>`_
# available on GitHub.
# We will use the MNE-NIRS package which includes convenient functions to
# download openly available datasets.

data_dir = pathlib.Path(datasets.fnirs_motor_group.data_path())

# Let's see whether the data has been downloaded using a quick visualization
# of the directory tree.
print_dir_tree(data_dir)

# %%
# The data are already in BIDS format. However, we will just use one of the
# SNIRF files and demonstrate how this could be used to generate a new BIDS
# compliant dataset from this single file.

# Specify file to use as input to BIDS generation process
file_path = data_dir / "sub-01" / "nirs" / "sub-01_task-tapping_nirs.snirf"

# %%
# Convert to BIDS
# ---------------
#
# Let's start with loading the data and updating the annotations.
# We are reading the data using MNE-Python's ``io`` module and the
# :func:`mne.io.read_raw_snirf` function.
# Note that we must use the ``preload=False`` parameter, which is the default
# in MNE-Python.
# It prevents the data from being loaded and modified when converting to BIDS.

# Load the data
raw = mne.io.read_raw_snirf(file_path, preload=False)
raw.info['line_freq'] = 50  # specify power line frequency as required by BIDS

# Sanity check, show the optode positions
raw.plot_sensors()

# %%
# I also like to rename the annotations to something meaningful and
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

# zero padding to account for >100 subjects in this dataset
subject_id = '01'

# define a task name and a directory where to save the data to
task = 'Tapping'
bids_root = data_dir.with_name(data_dir.name + '-bids')
print(bids_root)

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
# * ``optodes.tsv`` containing the optode coordinates and
#   ``coordsystem.json``, which contains the metadata about the optode
#   coordinates.
# * The actual SNIRF data file (with a proper BIDS name) and an accompanying
#   ``*_nirs.json`` file that contains metadata about the NIRS recording.
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
# describe Authors.
#
# Remember that there is a convenient javascript tool to validate all your BIDS
# directories called the "BIDS-validator", available as a web version and a
# command line tool:
#
# Web version: https://bids-standard.github.io/bids-validator/
#
# Command line tool: https://www.npmjs.com/package/bids-validator
