"""
=====================================
06. Rename BrainVision EEG data files
=====================================

The BrainVision file format is one of the recommended formats to store EEG data
within a BIDS directory. To organize EEG data in BIDS format, it is often
necessary to rename the files. In the case of BrainVision files, we would have
to rename multiple files for each dataset instance (i.e., once per recording):

1. A text header file (``.vhdr``) containing meta data
2. A text marker file (``.vmrk``) containing information about events in the
   data
3. A binary data file (``.eeg``) containing the voltage values of the EEG

The problem is that the three files contain internal links that guide a
potential data reading software. If we just rename the three files without also
adjusting the internal links, we corrupt the file format.

In this example, we use MNE-BIDS to rename BrainVision data files including a
repair of the internal file pointers.

For the command line version of this tool, see the :code:`cp` tool in the docs
for the :ref:`Python Command Line Interface <python_cli>`.

"""

# Authors: Stefan Appelhoff <stefan.appelhoff@mailbox.org>
# License: BSD (3-clause)

###############################################################################
# We are importing everything we need for this example:
import os.path as op

from numpy.testing import assert_array_equal
import mne

from mne_bids.copyfiles import copyfile_brainvision

###############################################################################
# Step 1: Download some example data
# ----------------------------------
# To demonstrate the MNE-BIDS functions, we need some testing data. Here, we
# will use the MNE-Python testing data. Feel free to use your own BrainVision
# data.
#
# .. warning:: This will download 1.6 GB of data!

data_path = mne.datasets.testing.data_path()
examples_dir = op.join(data_path, 'Brainvision')

###############################################################################
# Step 2: Rename the recording
# ----------------------------------
# Above, at the top of the example, we imported
# :func:`mne_bids.utils.copyfile_brainvision` from
# the MNE-BIDS ``utils.py`` module. This function takes two arguments as
# input: First, the path to the existing .vhdr file. And second, the path to
# the future .vhdr file.
#
# :func:`mne_bids.utils.copyfile_brainvision` will then create three new files
# (.vhdr, .vmrk, and .eeg) with the new names as provided with the second
# argument.
#
# Here, we rename a test file name:

vhdr_file = op.join(examples_dir, 'test_NO.vhdr')
vhdr_file_renamed = op.join(examples_dir, 'test_renamed.vhdr')
copyfile_brainvision(vhdr_file, vhdr_file_renamed)

###############################################################################
# Step 3: Assert that the renamed data can be read by a software
# --------------------------------------------------------------
# Finally, let's use MNE-Python to read in both, the original BrainVision data
# as well as the renamed data. They should be the same.
raw = mne.io.read_raw_brainvision(vhdr_file)
raw_renamed = mne.io.read_raw_brainvision(vhdr_file_renamed)

assert_array_equal(raw.get_data(), raw_renamed.get_data())

###############################################################################
# Further information
# -------------------
# There are alternative options to rename your BrainVision files. You could
# for example check out the
# `BVARENAMER <https://github.com/stefanSchinkel/bvarenamer>`_ by Stefan
# Schinkel.
#
# Lastly, there is a tool to check the integrity of your BrainVision files.
# For that, see the `BrainVision Validator <bv-validator_>`_
#
# .. LINKS
#
# .. _`bv-validator`: https://github.com/sappelhoff/brainvision-validator
