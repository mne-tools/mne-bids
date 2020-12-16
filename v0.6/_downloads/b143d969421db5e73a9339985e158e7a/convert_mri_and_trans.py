"""
==============================================================================
07. Save and load T1-weighted MRI scan along with anatomical landmarks in BIDS
==============================================================================

When working with MEEG data in the domain of source localization, we usually
have to deal with aligning several coordinate systems, such as the coordinate
systems of ...

- the head of a study participant
- the recording device (in the case of MEG)
- the anatomical MRI scan of a study participant

The process of aligning these frames is also called coregistration, and is
performed with the help of a transformation matrix, called ``trans`` in MNE.

In this tutorial, we show how ``MNE-BIDS`` can be used to save a T1 weighted
MRI scan in BIDS format, and to encode all information of the ``trans`` object
in a BIDS compatible way.

Finally, we will automatically reproduce our ``trans`` object from a BIDS
directory.

See the documentation pages in the MNE docs for more information on
`source alignment and coordinate frames <mne_source_coords_>`_

.. note:: For this example you will need to install ``matplotlib`` and
          ``nilearn`` on top of your usual ``mne-bids`` installation.

"""
# Authors: Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Alex Rockhill <aprockhill206@gmail.com>
#          Alex Gramfort <alexandre.gramfort@inria.fr>
# License: BSD (3-clause)

###############################################################################
# We are importing everything we need for this example:

import os.path as op
import shutil

import numpy as np
import matplotlib.pyplot as plt

import nibabel as nib
from nilearn.plotting import plot_anat

import mne
from mne.datasets import sample
from mne.source_space import head_to_mri

from mne_bids import (write_raw_bids, BIDSPath, write_anat,
                      get_head_mri_trans, print_dir_tree)

###############################################################################
# We will be using the `MNE sample data <mne_sample_data_>`_ and write a basic
# BIDS dataset. For more information, you can checkout the respective
# :ref:`example <ex-convert-mne-sample>`.

data_path = sample.data_path()
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
            'Visual/Right': 4, 'Smiley': 5, 'Button': 32}
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
events_data = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw-eve.fif')
output_path = op.abspath(op.join(data_path, '..', 'MNE-sample-data-bids'))

###############################################################################
# To ensure the output path doesn't contain any leftover files from previous
# tests and example runs, we simply delete it.
#
# .. warning:: Do not delete directories that may contain important data!
#

if op.exists(output_path):
    shutil.rmtree(output_path)

###############################################################################
# Read the input data and store it as BIDS data.

raw = mne.io.read_raw_fif(raw_fname)
raw.info['line_freq'] = 60  # specify power line frequency as required by BIDS

sub = '01'
ses = '01'
task = 'audiovisual'
run = '01'
bids_path = BIDSPath(subject=sub, session=ses, task=task,
                     run=run, root=output_path)
write_raw_bids(raw, bids_path, events_data=events_data,
               event_id=event_id, overwrite=True)

###############################################################################
# Print the directory tree
print_dir_tree(output_path)

###############################################################################
# Writing T1 image
# ----------------
#
# Now let's assume that we have also collected some T1 weighted MRI data for
# our subject. And furthermore, that we have already aligned our coordinate
# frames (using e.g., the `coregistration GUI`_) and obtained a transformation
# matrix :code:`trans`.

# Get the path to our MRI scan
t1_mgh_fname = op.join(data_path, 'subjects', 'sample', 'mri', 'T1.mgz')

# Load the transformation matrix and show what it looks like
trans_fname = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_raw-trans.fif')
trans = mne.read_trans(trans_fname)
print(trans)

###############################################################################
# We can save the MRI to our existing BIDS directory and at the same time
# create a JSON sidecar file that contains metadata, we will later use to
# retrieve our transformation matrix :code:`trans`. The metadata will here
# consist of the coordinates of three anatomical landmarks (LPA, Nasion and
# RPA (=left and right preauricular points) expressed in voxel coordinates
# w.r.t. the T1 image.

# First create the BIDSPath object.
t1w_bids_path = \
    BIDSPath(subject=sub, session=ses, root=output_path, suffix='T1w')

# We use the write_anat function
t1w_bids_path = write_anat(
    image=t1_mgh_fname,  # path to the MRI scan
    bids_path=t1w_bids_path,
    raw=raw,  # the raw MEG data file connected to the MRI
    trans=trans,  # our transformation matrix
    verbose=True  # this will print out the sidecar file
)
anat_dir = t1w_bids_path.directory

###############################################################################
# Let's have another look at our BIDS directory
print_dir_tree(output_path)

###############################################################################
# Our BIDS dataset is now ready to be shared. We can easily estimate the
# transformation matrix using ``MNE-BIDS`` and the BIDS dataset.
estim_trans = get_head_mri_trans(bids_path=bids_path)

###############################################################################
# Finally, let's use the T1 weighted MRI image and plot the anatomical
# landmarks Nasion, LPA, and RPA onto the brain image. For that, we can
# extract the location of Nasion, LPA, and RPA from the MEG file, apply our
# transformation matrix :code:`trans`, and plot the results.

# Get Landmarks from MEG file, 0, 1, and 2 correspond to LPA, NAS, RPA
# and the 'r' key will provide us with the xyz coordinates. The coordinates
# are expressed here in MEG Head coordinate system.
pos = np.asarray((raw.info['dig'][0]['r'],
                  raw.info['dig'][1]['r'],
                  raw.info['dig'][2]['r']))

# We now use the ``head_to_mri`` function from MNE-Python to convert MEG
# coordinates to MRI scanner RAS space. For the conversion we use our
# estimated transformation matrix and the MEG coordinates extracted from the
# raw file. `subjects` and `subjects_dir` are used internally, to point to
# the T1-weighted MRI file: `t1_mgh_fname`. Coordinates are is mm.
mri_pos = head_to_mri(pos=pos,
                      subject='sample',
                      mri_head_t=estim_trans,
                      subjects_dir=op.join(data_path, 'subjects'))

# Our MRI written to BIDS, we got `anat_dir` from our `write_anat` function
t1_nii_fname = op.join(anat_dir, 'sub-01_ses-01_T1w.nii.gz')

# Plot it
fig, axs = plt.subplots(3, 1, figsize=(7, 7), facecolor='k')
for point_idx, label in enumerate(('LPA', 'NAS', 'RPA')):
    plot_anat(t1_nii_fname, axes=axs[point_idx],
              cut_coords=mri_pos[point_idx, :],
              title=label, vmax=160)
plt.show()

###############################################################################
# Writing FLASH MRI image
# -----------------------
#
# We can write another types of MRI data such as FLASH images for BEM models

flash_mgh_fname = \
    op.join(data_path, 'subjects', 'sample', 'mri', 'flash', 'mef05.mgz')

flash_bids_path = \
    BIDSPath(subject=sub, session=ses, root=output_path, suffix='FLASH')

write_anat(
    image=flash_mgh_fname,
    bids_path=flash_bids_path,
    verbose=True
)

###############################################################################
# Writing defaced and anonymized T1 image
# ---------------------------------------
#
# We can deface the MRI for anonymization by passing ``deface=True``.
t1w_bids_path = write_anat(
    image=t1_mgh_fname,  # path to the MRI scan
    bids_path=bids_path,
    raw=raw,  # the raw MEG data file connected to the MRI
    trans=trans,  # our transformation matrix
    deface=True,
    overwrite=True,
    verbose=True  # this will print out the sidecar file
)
anat_dir = t1w_bids_path.directory

# Our MRI written to BIDS, we got `anat_dir` from our `write_anat` function
t1_nii_fname = op.join(anat_dir, 'sub-01_ses-01_T1w.nii.gz')

# Plot it
fig, ax = plt.subplots()
plot_anat(t1_nii_fname, axes=ax, title='Defaced', vmax=160)
plt.show()

###############################################################################
# Writing defaced and anonymized FLASH MRI image
# ----------------------------------------------
#
# Defacing the FLASH is a bit more complicated because it uses different
# coordinates than the T1. Since, in the example dataset, we used the head
# surface (which was reconstructed by freesurfer from the T1) to align the
# digitization points, the points are relative to the T1-defined coordinate
# system (called surface or TkReg RAS). Thus, you can you can provide the T1
# or you can find the fiducials in FLASH voxel space or scanner RAS coordinates
# using your favorite 3D image view (e.g. freeview). You can also read the
# fiducial coordinates from the `raw` and apply the `trans` yourself.
# Let's explore the different options to do this.

###############################################################################
# Option 1 : Pass `t1w` with `raw` and `trans`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
flash_bids_path = write_anat(
    image=flash_mgh_fname,  # path to the MRI scan
    bids_path=flash_bids_path,
    raw=raw,
    trans=trans,
    t1w=t1_mgh_fname,
    deface=True,
    overwrite=True,
    verbose=True  # this will print out the sidecar file
)

# Our MRI written to BIDS, we got `anat_dir` from our `write_anat` function
flash_nii_fname = op.join(anat_dir, 'sub-01_ses-01_FLASH.nii.gz')

# Plot it
fig, ax = plt.subplots()
plot_anat(flash_nii_fname, axes=ax, title='Defaced', vmax=700)
plt.show()

###############################################################################
# Option 2 : Use manual landmarks coordinates in scanner RAS for FLASH image
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# You can find such landmarks with a 3D image viewer (e.g. freeview).
# Note that, in freeview, this is called "RAS" and not "TkReg RAS"
flash_ras_landmarks = \
    np.array([[-74.53102838, 19.62854953, -52.2888194],
              [-1.89454315, 103.69850925, 4.97120376],
              [72.01200673, 21.09274883, -57.53678375]]) / 1e3  # mm -> m

landmarks = mne.channels.make_dig_montage(
    lpa=flash_ras_landmarks[0],
    nasion=flash_ras_landmarks[1],
    rpa=flash_ras_landmarks[2],
    coord_frame='ras'
)

flash_bids_path = write_anat(
    image=flash_mgh_fname,  # path to the MRI scan
    bids_path=flash_bids_path,
    landmarks=landmarks,
    deface=True,
    overwrite=True,
    verbose=True  # this will print out the sidecar file
)

# Plot it
fig, ax = plt.subplots()
plot_anat(flash_nii_fname, axes=ax, title='Defaced', vmax=700)
plt.show()

###############################################################################
# Option 3 : Compute the landmarks in scanner RAS or mri voxel space from trans
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Get Landmarks from MEG file, 0, 1, and 2 correspond to LPA, NAS, RPA
# and the 'r' key will provide us with the xyz coordinates.
#
# .. note::
#
#   We can use in the head_to_mri function based on T1 as the T1 and FLASH
#   images are already registered.
head_pos = np.asarray((raw.info['dig'][0]['r'],
                       raw.info['dig'][1]['r'],
                       raw.info['dig'][2]['r']))

ras_pos = head_to_mri(pos=head_pos,
                      subject='sample',
                      mri_head_t=trans,
                      subjects_dir=op.join(data_path, 'subjects')) / 1e3

montage_ras = mne.channels.make_dig_montage(
    lpa=ras_pos[0],
    nasion=ras_pos[1],
    rpa=ras_pos[2],
    coord_frame='ras'
)

# pass FLASH scanner RAS coordinates
flash_bids_path = write_anat(
    image=flash_mgh_fname,  # path to the MRI scan
    bids_path=flash_bids_path,
    landmarks=montage_ras,
    deface=True,
    overwrite=True,
    verbose=True  # this will print out the sidecar file
)

# Plot it
fig, ax = plt.subplots()
plot_anat(flash_nii_fname, axes=ax, title='Defaced', vmax=700)
plt.show()

##############################################################################
# Let's now pass it in voxel coordinates
flash_mri_hdr = nib.load(flash_mgh_fname).header
flash_vox_pos = mne.transforms.apply_trans(
    flash_mri_hdr.get_ras2vox(), ras_pos * 1e3)

montage_flash_vox = mne.channels.make_dig_montage(
    lpa=flash_vox_pos[0],
    nasion=flash_vox_pos[1],
    rpa=flash_vox_pos[2],
    coord_frame='mri_voxel'
)

# pass FLASH voxel coordinates
flash_bids_path = write_anat(
    image=flash_mgh_fname,  # path to the MRI scan
    bids_path=flash_bids_path,
    landmarks=montage_flash_vox,
    deface=True,
    overwrite=True,
    verbose=True  # this will print out the sidecar file
)

# Plot it
fig, ax = plt.subplots()
plot_anat(flash_nii_fname, axes=ax, title='Defaced', vmax=700)
plt.show()

###############################################################################
# .. LINKS
#
# .. _coregistration GUI:
#    https://mne.tools/stable/auto_tutorials/source-modeling/plot_source_alignment.html#defining-the-headmri-trans-using-the-gui
# .. _mne_source_coords:
#    https://mne.tools/stable/auto_tutorials/source-modeling/plot_source_alignment.html
# .. _mne_sample_data:
#    https://mne.tools/stable/overview/datasets_index.html#sample
#
