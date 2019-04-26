"""
==========================================
Creating BIDS-compatible folders and files
==========================================

The Brain Imaging Data Structure (BIDS) has standard conventions for file
names and folder hierarchy. MNE-BIDS comes with convenience functions if you
wish to create these files/folders on your own.

.. note::

   You may automatically convert Raw objects to BIDS-compatible files with
   `write_raw_bids`. This example is for manually creating files/folders.
"""

# Authors: Chris Holdgraf <choldgraf@berkeley.edu>
# License: BSD (3-clause)

###############################################################################
# We'll import the relevant functions from the write module

from mne_bids import make_bids_folders, make_bids_basename

###############################################################################
# Creating file names for BIDS
# ----------------------------
#
# BIDS requires a specific ordering and structure for metadata fields in
# file paths, the function `make_bids_basename` allows you to specify many such
# pieces of metadata, ensuring that they are in the correct order in the
# final file path. Omitted keys will not be included in the file path.

bids_basename = make_bids_basename(subject='test', session='two',
                                   task='mytask', suffix='data.csv')
print(bids_basename)

###############################################################################
# You may also omit the suffix, which will result in *only* a prefix for a
# file name. This could then prepended to many more files.

bids_basename = make_bids_basename(subject='test', task='mytask')
print(bids_basename)

###############################################################################
# Creating folders
# ----------------
#
# You can also use MNE-BIDS to create folder hierarchies.

path_folder = make_bids_folders('sub_01', session='mysession',
                                kind='meg', output_path='path/to/project',
                                make_dir=False)
print(path_folder)

# Note that passing `make_dir=True` will create the folder hierarchy, ignoring
# errors if the folder already exists.
