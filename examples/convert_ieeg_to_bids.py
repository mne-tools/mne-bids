"""
.. currentmodule:: mne_bids

====================================
08. Convert iEEG data to BIDS format
====================================

In this example, we use MNE-BIDS to create a BIDS-compatible directory of iEEG
data. Specifically, we will follow these steps:

1. Download some iEEG data.

2. Load the data, extract information, and save in a new BIDS directory.

3. Check the result and compare it with the standard.

4. Cite MNE-BIDS

5. Confirm that written iEEG coordinates are the
   same before :func:`write_raw_bids` was called.

The iEEG data will be written by :func:`write_raw_bids` with
the addition of extra metadata elements in the following files:

- the sidecar file ``ieeg.json``
- ``electrodes.tsv``
- ``coordsystem.json``
- ``events.tsv``
- ``channels.tsv``

Compared to EEG data, the main differences are within the
``coordsystem.json`` and ``electrodes.tsv`` files.
For more information on these files,
refer to the `iEEG part of the BIDS specification`_.

.. _iEEG part of the BIDS specification: https://bids-specification.readthedocs.io/en/latest/04-modality-specific-files/04-intracranial-electroencephalography.html
.. _appendix VIII: https://bids-specification.readthedocs.io/en/stable/99-appendices/08-coordinate-systems.html
.. _background on FreeSurfer: https://mne.tools/dev/auto_tutorials/source-modeling/plot_background_freesurfer_mne
.. _MNE-Python coordinate frames: https://mne.tools/dev/auto_tutorials/source-modeling/plot_source_alignment.html

"""  # noqa: E501

# Authors: Adam Li <adam2392@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD-3-Clause

# %%

import os.path as op
import shutil

import matplotlib.pyplot as plt

import mne
from mne_bids import (write_raw_bids, BIDSPath,
                      read_raw_bids, print_dir_tree)

# %%
# Step 1: Download the data
# -------------------------
#
# First, we need some data to work with. We will use the
# data downloaded via MNE-Python's ``datasets`` API:
# :func:`mne.datasets.misc.data_path`
misc_path = mne.datasets.misc.data_path()

# The electrode coords data are in the tsv file format
# which is easily read in using numpy
raw = mne.io.read_raw_fif(op.join(
    misc_path, 'seeg', 'sample_seeg_ieeg.fif'))
raw.info['line_freq'] = 60  # specify power line frequency as required by BIDS
subjects_dir = op.join(misc_path, 'seeg')  # Freesurfer recon-all directory

# %%
# Now we make a montage in ACPC space. MNE stores channel locations
# in the "head" coordinate frame, which has the origin at the center
# between the left and right auricular points. BIDS requires that iEEG data
# be in ACPC space so we need to convert from "head" to "mni_tal" which is
# an `ACPC-aligned coordinate system
# <https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems>`_.
# The MNI Talairach coordinate system is very useful for group analysis as
# shown in `Working with SEEG
# <https://mne.tools/stable/auto_tutorials/misc/plot_seeg.html>`_.
# The ``fsaverage`` template brain is in MNI space so we can use the Talairach
# transform to get to this space from "mri" or Freesurfer surface RAS space.
# First we have to assign, or estimate the fiducial points (see
# `Locating Intracranial Electrode Contacts
# <https://mne.tools/dev/auto_tutorials/clinical/10_ieeg_localize.html>`_).
# Second, we create a Transformation object from ``head`` to ``mri``
# coordinates using the fiducial points. Finally, we can use the
# Freesurfer Talairach transform to get to MNI space.

# estimate the transformation from "head" to "mri" space
trans = mne.coreg.estimate_head_mri_t('sample_seeg', subjects_dir)

# get Talairach transform
mri_mni_t = mne.read_talxfm('sample_seeg', subjects_dir)

# %%
# Now let's convert the montage to MNI Talairach ("mni_tal").
montage = raw.get_montage()
montage.apply_trans(trans)  # head->mri
montage.apply_trans(mri_mni_t)
# a bit of a hack here; MNE will transform the coordinates to "head"
# when you set the montage if there are fiducials and we don't want
# that so we have to get rid of them
montage.dig = montage.dig[3:]
# warns that identity transformation to "head" is assumed which is what we want
raw.set_montage(montage, verbose='error')

# %%
# Let's plot to check what our starting channel coordinates look like.


def plot_3D_montage(montage):
    positions = montage.get_positions()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    cmap = plt.get_cmap('rainbow')
    colors = dict()
    for name, pos in positions['ch_pos'].items():
        name = ''.join([letter for letter in name if
                        not letter.isdigit() and letter != ' '])
        if name in colors:
            color = colors[name]
        else:
            color = cmap(15 * (len(colors) + 1))
            colors[name] = color
        ax.scatter(*pos, color=color, label=name)
    fig.show()


plot_3D_montage(montage)

# %%
# BIDS vs MNE-Python Coordinate Systems
# -------------------------------------
#
# BIDS has many acceptable coordinate systems for iEEG, which can be viewed in
# `appendix VIII`_ of the BIDS specification.
# However, MNE-BIDS depends on MNE-Python and MNE-Python does not support all
# these coordinate systems (yet).
#
# MNE-Python has a few tutorials on this topic:
#
# - `background on FreeSurfer`_
# - `MNE-Python coordinate frames`_
#
# Currently, MNE-Python supports the ``mni_tal`` coordinate frame, which
# corresponds to the ``fsaverage`` BIDS coordinate system. All other coordinate
# frames in MNE-Python if written with :func:`mne_bids.write_raw_bids` are
# written with coordinate system ``'Other'``. Note, then we suggest using
# :func:`mne_bids.update_sidecar_json` to update the sidecar
# ``*_coordsystem.json`` file to add additional information.
#
# Step 2: Formatting as BIDS
# --------------------------
#
# Now, let us format the `Raw` object into BIDS.
#
# With this step, we have everything to start a new BIDS directory using
# our data. To do that, we can use :func:`write_raw_bids`
# Generally, :func:`write_raw_bids` tries to extract as much
# meta data as possible from the raw data and then formats it in a BIDS
# compatible way. :func:`write_raw_bids` takes a bunch of inputs, most of
# which are however optional. The required inputs are:
#
# - :code:`raw`
# - :code:`bids_basename`
# - :code:`bids_root`
#
# ... as you can see in the docstring:
print(write_raw_bids.__doc__)

# %%
# Let us initialize some of the necessary data for the subject.

# There is a subject, and specific task for the dataset.
subject_id = '1'
task = 'motor'

# get MNE-Python directory w/ example data
mne_data_dir = mne.get_config('MNE_DATASETS_MISC_PATH')

# There is the root directory for where we will write our data.
bids_root = op.join(mne_data_dir, 'ieeg_bids')

# %%
# To ensure the output path doesn't contain any leftover files from previous
# tests and example runs, we simply delete it.
#
# .. warning:: Do not delete directories that may contain important data!
#

if op.exists(bids_root):
    shutil.rmtree(bids_root)

# %%
# Now we just need to specify a few iEEG details to make things work:
# We need the basename of the dataset. In addition, :func:`write_raw_bids`
# requires the ``.filenames`` attribute of the Raw object to be non-empty,
# so since we
# initialized the dataset from an array, we need to do a hack where we
# temporarily save the data to disc before reading it back in.

# Now convert our data to be in a new BIDS dataset.
bids_path = BIDSPath(subject=subject_id, task=task, root=bids_root)

# write `raw` to BIDS and anonymize it into BrainVision format
write_raw_bids(raw, bids_path, anonymize=dict(daysback=30000),
               overwrite=True)

# %%
# Step 3: Check and compare with standard
# ---------------------------------------

# Now we have written our BIDS directory.
print_dir_tree(bids_root)

# %%
# Step 4: Cite mne-bids
# ---------------------
# We can see that the appropriate citations are already written in the README.
# If you are preparing a manuscript, please make sure to also cite MNE-BIDS
# there.
readme = op.join(bids_root, 'README')
with open(readme, 'r', encoding='utf-8-sig') as fid:
    text = fid.read()
print(text)

# %%
# MNE-BIDS has created a suitable directory structure for us, and among other
# meta data files, it started an ``events.tsv``` and ``channels.tsv`` file,
# and created an initial ``dataset_description.json`` file on top!
#
# Now it's time to manually check the BIDS directory and the meta files to add
# all the information that MNE-BIDS could not infer. For instance, you must
# describe ``iEEGReference`` and ``iEEGGround`` yourself.
# It's easy to find these by searching for ``"n/a"`` in the sidecar files.
#
# ``$ grep -i 'n/a' <bids_root>```
#
# Remember that there is a convenient JavaScript tool to validate all your BIDS
# directories called the "BIDS-validator", available as a web version and a
# command line tool:
#
# Web version: https://bids-standard.github.io/bids-validator/
#
# Command line tool: https://www.npmjs.com/package/bids-validator

# %%
# Step 5: Plot output channels and check that they match!
# -------------------------------------------------------
#
# Now we have written our BIDS directory. We can use
# :func:`read_raw_bids` to read in the data.

# read in the BIDS dataset to plot the coordinates
raw = read_raw_bids(bids_path=bids_path)

# %%
# Now we have to go back to "head" coordinates. We do this with ``fsaverage``
# fiducials which are in MNI space.
#
# .. note:: If you were downloading this from ``OpenNeuro``, you would
#           have to run the Freesurfer ``recon-all`` to get the transforms.

montage = raw.get_montage()
montage.add_mni_fiducials(subjects_dir=subjects_dir)
raw.set_montage(montage)

# %%
# Finally, we can plot the result to ensure that the data was correctly
# formatted for the round trip.

plot_3D_montage(montage)
