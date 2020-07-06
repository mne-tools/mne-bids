"""
======================================================
06. Rename and/or anonymize BrainVision EEG data files
======================================================

According to the EEG extension to BIDS [1]_, the `BrainVision data format`_ is
one of the recommended formats to store EEG data within a BIDS directory.

To organize EEG data in BIDS format, it is often necessary to rename the files.
In the case of BrainVision files, we would have to rename multiple files for
each dataset instance (i.e., once per recording):

1. A text header file (``.vhdr``) containing meta data
2. A text marker file (``.vmrk``) containing information about events in the
   data
3. A binary data file (``.eeg``) containing the voltage values of the EEG

.. Note:: The three files contain internal links that guide the data reading
          software. Renaming the three files without adjusting the internal
          links will corrupt the file format! But relax, MNE-BIDS can take care
          of this for you.

Next to renaming files for BIDS, we often will also have to anonymize the data.

In this example, we use MNE-BIDS to:

1. Rename BrainVision data files including a repair of the internal file links
2. Anonymize (overwrite) all recording dates

For the command line version of this tool, see the :code:`cp` tool in the docs
for the :ref:`Python Command Line Interface <python_cli>`.

References
----------
.. [1] Pernet, C.R., Appelhoff, S., Gorgolewski, K.J. et al. EEG-BIDS, an
       extension to the brain imaging data structure for
       electroencephalography. Sci Data 6, 103 (2019).
       https://doi.org/10.1038/s41597-019-0104-8
.. _BrainVision data format: https://www.brainproducts.com/productdetails.php?id=21&tab=5
"""

# Authors: Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)

###############################################################################
# We are importing everything we need for this example:
import os.path as op
import datetime

from numpy.testing import assert_array_equal
import mne

from mne_bids.copyfiles import copyfile_brainvision

###############################################################################
# Download some example data
# --------------------------
# To demonstrate the MNE-BIDS functions, we need some testing data. Here, we
# will use the MNE-Python testing data. Feel free to use your own BrainVision
# data.
#
# .. warning:: This will download 1.6 GB of data!

data_path = mne.datasets.testing.data_path()
examples_dir = op.join(data_path, 'Brainvision')

###############################################################################
# Rename the recording
# --------------------
# Above, at the top of the example, we imported
# :func:`mne_bids.copyfiles.copyfile_brainvision` from
# the MNE-BIDS :mod:`copyfiles.py` module. This function takes two main inputs:
# First, the path to the existing ``.vhdr`` file. And second, the path to
# the future ``.vhdr`` file.
#
# With the optional ``verbose`` parameter you can furthermore determine how
# much information you want to get during the procedure.
#
# :func:`mne_bids.copyfiles.copyfile_brainvision` will then create three new
# files (``.vhdr``, ``.vmrk``, and ``.eeg``) with the new names as provided
# with the second argument.
#
# Here, we rename a test file name:

# Rename the file
vhdr_file = op.join(examples_dir, 'Analyzer_nV_Export.vhdr')
vhdr_file_renamed = op.join(examples_dir, 'test_renamed.vhdr')
copyfile_brainvision(vhdr_file, vhdr_file_renamed, verbose=True)

# Check that MNE-Python can read in both, the original as well as the renamed
# data. They should be the same.
raw = mne.io.read_raw_brainvision(vhdr_file)
raw_renamed = mne.io.read_raw_brainvision(vhdr_file_renamed)

# If we can run the following without error, everything is fine
assert_array_equal(raw.get_data(), raw_renamed.get_data())

###############################################################################
# Anonymize the recording
# -----------------------
# We successfully renamed our file. But we can use
# :func:`mne_bids.copyfiles.copyfile_brainvision` also to anonymize the
# recording.
#
# In the example below, we overwrite our previously renamed file. But this time
# we specify the ``anonymize`` and ``date`` parameters, to overwrite the
# existing dates with our new one.

# Get the original recording date
original_date = raw.info['meas_date']
print(f'The original date was {original_date}\n')

# Make up a new date to anonymize the original one. 1900-05-06 at 00:00:00
# seems fine
new_date = datetime.datetime(1924, 5, 6)

# Now anonymize using `new_date`
copyfile_brainvision(vhdr_file, vhdr_file_renamed, anonymize=True,
                     date=new_date, verbose=True)

# Check that the freshly renamed and anonymized file has the new date
raw_renamed = mne.io.read_raw_brainvision(vhdr_file_renamed)
anonymized_date = raw_renamed.info['meas_date']
print(f'After anonymization, the date is now {anonymized_date}.\n')

###############################################################################
# Further information
# -------------------
# You can also anonymize BrainVision files without specifying the ``date``
# parameter. If you just specify ``anonymize=True``, MNE-BIDS will overwrite
# all dates with a default datetime of 1924-01-01T00:00:00.
#
# For converting data files, or writing new data to the BrainVision format, you
# can use the `pybv`_ Python package.
#
# There is node JS tool to check the integrity of your BrainVision files.
# For that, see the `BrainVision Validator <bv-validator_>`_
#
# .. _`pybv`: https://github.com/bids-standard/pybv
# .. _`bv-validator`: https://github.com/sappelhoff/brainvision-validator
