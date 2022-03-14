"""
==========================
12. Updating BIDS datasets
==========================

When working with electrophysiological data in the BIDS format, we usually
do not have all the metadata stored in the ``Raw`` mne-python object.
We can update the BIDS sidecar files via the ``update_sidecar_json`` function.

In this tutorial, we show how ``update_sidecar_json`` can be used to update and
modify BIDS-formatted data.
"""
# Authors: Adam Li <adam2392@gmail.com>
#          mne-bids developers
#
# License: BSD-3-Clause

# %%
# Imports
# -------
# We are importing everything we need for this example:
from mne.datasets import somato

from mne_bids import (BIDSPath, read_raw_bids,
                      print_dir_tree, make_report, update_sidecar_json)

# %%
# We will be using the `MNE somato data <mne_somato_data_>`_, which
# is already stored in BIDS format.
# For more information, you can check out the
# respective :ref:`example <ex-convert-mne-sample>`.
#
# Download the ``somato`` BIDS dataset
# ------------------------------------
#
# Download the data if it hasn't been downloaded already, and return the path
# to the download directory. This directory is the so-called `root` of this
# BIDS dataset.
bids_root = somato.data_path()

# %%
# Explore the dataset contents
# ----------------------------
#
# We can use MNE-BIDS to print a tree of all
# included files and folders. We pass the ``max_depth`` parameter to
# `mne_bids.print_dir_tree` to the output to three levels of folders, for
# better readability in this example.

print_dir_tree(bids_root, max_depth=3)

# We can generate a report of the existing dataset
print(make_report(bids_root))

# %%
# Update the sidecar JSON dataset contents
# ----------------------------------------
#
# We can use MNE-BIDS to update all sidecar files for a matching
# ``BIDSPath`` object. We then pass in a dictionary (or JSON file) to update
# all matching metadata fields within the BIDS dataset.

# create a BIDSPath object
bids_root = somato.data_path()
datatype = 'meg'
subject = '01'
task = 'somato'
suffix = 'meg'

bids_path = BIDSPath(subject=subject, task=task, suffix=suffix,
                     datatype=datatype, root=bids_root)
sidecar_path = bids_path.copy().update(extension='.json')

# We can now retrieve a list of all MEG-related files in the dataset:
# we will specifically now update the sidecar json file.
print(bids_path.match())

# Define a sidecar update as a dictionary
entries = {
    'PowerLineFrequency': 60,
    'Manufacturer': "MEGIN",
    'InstitutionName': "Martinos Center"
}

# Note: ``update_sidecar_json`` will perform essentially a
# dictionary update to your sidecar json file, so be absolutely sure
# that the ``entries`` are defined with the proper fields specified
# by BIDS. For example, if you are updating the ``coordsystem.json``
# file, then you don't want to include ``PowerLineFrequency`` in
# ``entries``.
#
# Now update all sidecar fields according to our updating dictionary
update_sidecar_json(bids_path=sidecar_path, entries=entries)

# %%
# Read the updated dataset
# ------------------------

# new line frequency is now 60 Hz
raw = read_raw_bids(bids_path=bids_path)
print(raw.info['line_freq'])

# %%
# Generate a new report based on the updated metadata.

# The manufacturer was changed to ``MEGIN``
print(make_report(bids_root))

# %%
# We can revert the changes by updating the sidecar again.

# update the sidecar data to have a new PowerLineFrequency
entries['Manufacturer'] = "Elekta"
entries['PowerLineFrequency'] = 50
update_sidecar_json(bids_path=sidecar_path, entries=entries)

# %%
# Now let us inspect the dataset again by generating the report again. Now that
# ``update_sidecar_json`` was called, the metadata will be updated.

# The power line frequency should now change back to 50 Hz
raw = read_raw_bids(bids_path=bids_path)
print(raw.info['line_freq'])

# Generate the report with updated fields
print(make_report(bids_root))

# %%
# .. LINKS
#
# .. _mne_somato_data:
#    https://mne.tools/dev/generated/mne.datasets.somato.data_path.html
#
