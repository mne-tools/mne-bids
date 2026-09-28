"""
=======================================================
11. Creating BIDS-compatible folder names and filenames
=======================================================

The Brain Imaging Data Structure (BIDS) has standard conventions for file
names and folder hierarchy. MNE-BIDS comes with convenience functions if you
wish to create these files/folders on your own.

.. note::

   You may automatically convert Raw objects to BIDS-compatible files with
   ``write_raw_bids``. This example is for manually creating files/folders.
"""

# Authors: Chris Holdgraf <choldgraf@berkeley.edu>
#
# License: BSD-3-Clause

# %%
# First we will import the relevant functions

from mne_bids import BIDSPath

# %%
# Creating file names for BIDS
# ----------------------------
#
# BIDS requires a specific ordering and structure for metadata fields in
# file paths, the class `BIDSPath` allows you to specify many such
# pieces of metadata, ensuring that they are in the correct order in the
# final file path. Omitted keys will not be included in the file path.

bids_path = BIDSPath(subject='test', session='two', task='mytask',
                     suffix='events', extension='.tsv')
print(bids_path)

# %%
# You may also omit the suffix, which will result in *only* a prefix for a
# file name. This could then prepended to many more files.

bids_path = BIDSPath(subject='test', task='mytask')
print(bids_path)

# %%
# Creating folders
# ----------------
#
# You can also use MNE-BIDS to create folder hierarchies.

bids_path = BIDSPath(subject='01', session='mysession',
                     datatype='meg', root='path/to/project').mkdir()
print(bids_path.directory)
