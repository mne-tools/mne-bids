"""
==========================================
Creating BIDS-compatible folders and files
==========================================

The Brain Imaging Data Structure (BIDS) has standard conventions for file
names and folder hierarchy. MNE-BIDS comes with convenience functions if you
wish to create these files/folders on your own.

.. note::

   You may automatically convert Raw files to a BIDS-compatible folder with
   `raw_to_bids`. This example is for manually creating files/folders.
"""

# Authors: Chris Holdgraf <choldgraf@berkeley.edu>
# License: BSD (3-clause)

###############################################################################
# We'll import the relevant functions from the utils module

from mne_bids import create_folders, filename_bids

###############################################################################
# Creating file names for BIDS
# ----------------------------
#
# BIDS requires a specific ordering and structure for metadata fields in
# file paths, the function `filename_bids` allows you to specify many such
# pieces of metadata, ensuring that they are in the correct order in the
# final file path. Omitted keys will not be included in the file path.

my_name = filename_bids(subject='test', session='two', task='mytask',
                        suffix='data.csv')
print(my_name)

###############################################################################
# You may also omit the suffix, which will result in *only* a prefix for a
# file name. This could then prepended to many more files.

my_name = filename_bids(subject='test', task='mytask')
print(my_name)

###############################################################################
# Creating folders
# ----------------
#
# You can also use MNE-BIDS to create folder hierarchies.

path_folder = create_folders('sub_01', session='my_session',
                             kind='meg', root='path/to/project', create=False)
print(path_folder)

# Note that passing `create=True` will create the folder hierarchy, ignoring
# errors if the folder already exists.
