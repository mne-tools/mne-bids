"""
========================
11. Update BIDS datasets
========================

When working with electrophysiological data in the BIDS format, we usually
do not have all the metadata stored in the ``Raw`` mne-python object.
We can update the BIDS sidecar files via the ``update_sidecars`` function.

In this tutorial, we show how ``update_sidecars`` can be used to update and
modify BIDS-formatted data.

"""
# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)

###############################################################################
# .. contents:: Contents
#    :local:
#    :depth: 1
#
# Imports
# -------
# We are importing everything we need for this example:
import os.path as op
from mne.datasets import somato
import tempfile
import json
from mne_bids import (BIDSPath, read_raw_bids,
                      print_dir_tree, make_report)
from mne_bids.sidecar_updates import update_sidecars

###############################################################################
# We will be using the `MNE somato data <mne_somato_data_>`_, which
# is already stored in BIDS format.
# For more information, you can check out the
# respective :ref:`example <ex-convert-mne-sample>`.

###############################################################################
# Download the ``somato`` BIDS dataset
# ------------------------------------
#
# Download the data if it hasn't been downloaded already, and return the path
# to the download directory. This directory is the so-called `root` of this
# BIDS dataset.
bids_root = somato.data_path()

###############################################################################
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

###############################################################################
# Update the dataset contents
# ---------------------------
#
# We can use MNE-BIDS to update all sidecar files for a matching
# ``BIDSPath`` object. We then pass in a sidecar template to update
# all matching metadata fields within the BIDS dataset.
###############################################################################

# create a BIDSPath object
bids_root = somato.data_path()
datatype = 'meg'
subject = '01'
task = 'somato'
suffix = 'meg'

bids_path = BIDSPath(subject=subject, task=task, suffix=suffix,
                     datatype=datatype, root=bids_root)

# We can now retrieve a list of all MEG-related files in the dataset:
print(bids_path.match())

# Define a sidecar template as a dictionary
sidecar_template = {
    'PowerLineFrequency': 60,
    'Manufacturer': 'Captrak',
    'InstitutionName': 'Martinos Center'
}

# Now update all sidecar fields according to the template
update_sidecars(bids_path=bids_path, sidecar_template=sidecar_template)

###############################################################################
# Read the updated dataset
# ------------------------
#
# We can use MNE-BIDS to update all sidecar files for a matching
# ``BIDSPath`` object. We then pass in a sidecar template to update
# all matching metadata fields within the BIDS dataset.
###############################################################################

# new line frequency is now 60 Hz
raw = read_raw_bids(bids_path=bids_path)
print(raw.info['line_freq'])

###############################################################################
# Generate a new report based on the updated metadata.

# The manufacturer was changed to ``Captrak``
print(make_report(bids_root))

###############################################################################
# We can also update sidecars by passing in a template JSON file.

# update the template to have a new PowerLineFrequency
sidecar_template['Manufacturer'] = 'Elekta'
sidecar_template['PowerLineFrequency'] = 50

# create a temporary JSON file that contains the updated template
with tempfile.TemporaryDirectory() as tempdir:
    sidecar_template_fpath = op.join(tempdir, 'template.json')
    with open(sidecar_template_fpath, 'w') as fout:
        json.dump(sidecar_template, fout)

    # Update sidecar files via a template defined as a JSON file.
    update_sidecars(bids_path=bids_path,
                    sidecar_template=sidecar_template_fpath)

###############################################################################
# Now let us inspect the dataset again by generating the report again. Now that
# ``update_sidecars`` was called, the metadata will be updated.

# The power line frequency should now change back
raw = read_raw_bids(bids_path=bids_path)
print(raw.info['line_freq'])

# Generate the report with updated fields
print(make_report(bids_root))
