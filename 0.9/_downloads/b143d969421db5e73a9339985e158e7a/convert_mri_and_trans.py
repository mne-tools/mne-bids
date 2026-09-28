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
#          Alex Rockhill <aprockhill@mailbox.org>
#          Alex Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause

# %%
# Let's import everything we need for this example:

import os.path as op
import shutil

import numpy as np
import matplotlib.pyplot as plt

from nilearn.plotting import plot_anat

import mne
from mne.datasets import sample
from mne import head_to_mri

from mne_bids import (write_raw_bids, BIDSPath, write_anat, get_anat_landmarks,
                      get_head_mri_trans, print_dir_tree)

# %%
# We will be using the `MNE sample data <mne_sample_data_>`_ and write a basic
# BIDS dataset. For more information, you can checkout the respective
# :ref:`example <ex-convert-mne-sample>`.

data_path = sample.data_path()
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
            'Visual/Right': 4, 'Smiley': 5, 'Button': 32}
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
events_data = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw-eve.fif')
output_path = op.abspath(op.join(data_path, '..', 'MNE-sample-data-bids'))
fs_subjects_dir = op.join(data_path, 'subjects')  # FreeSurfer subjects dir

# %%
# To ensure the output path doesn't contain any leftover files from previous
# tests and example runs, we simply delete it.
#
# .. warning:: Do not delete directories that may contain important data!
#

if op.exists(output_path):
    shutil.rmtree(output_path)

# %%
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

# %%
# Print the directory tree
print_dir_tree(output_path)

# %%
# Writing T1 image
# ----------------
#
# Now let's assume that we have also collected some T1 weighted MRI data for
# our subject. And furthermore, that we have already aligned our coordinate
# frames (using e.g., the `coregistration GUI`_) and obtained a transformation
# matrix :code:`trans`.

# Get the path to our MRI scan
t1_fname = op.join(fs_subjects_dir, 'sample', 'mri', 'T1.mgz')

# Load the transformation matrix and show what it looks like
trans_fname = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_raw-trans.fif')
trans = mne.read_trans(trans_fname)
print(trans)

# %%
# We can save the MRI to our existing BIDS directory and at the same time
# create a JSON sidecar file that contains metadata, we will later use to
# retrieve our transformation matrix :code:`trans`. The metadata will here
# consist of the coordinates of three anatomical landmarks (LPA, Nasion and
# RPA (=left and right preauricular points) expressed in voxel coordinates
# w.r.t. the T1 image.

# First create the BIDSPath object.
t1w_bids_path = BIDSPath(subject=sub, session=ses, root=output_path,
                         suffix='T1w')

# use ``trans`` to transform landmarks from the ``raw`` file to
# the voxel space of the image
landmarks = get_anat_landmarks(
    t1_fname,  # path to the MRI scan
    info=raw.info,  # the MEG data file info from the same subject as the MRI
    trans=trans,  # our transformation matrix
    fs_subject='sample',  # FreeSurfer subject
    fs_subjects_dir=fs_subjects_dir,  # FreeSurfer subjects directory
)

# We use the write_anat function
t1w_bids_path = write_anat(
    image=t1_fname,  # path to the MRI scan
    bids_path=t1w_bids_path,
    landmarks=landmarks,  # the landmarks in MRI voxel space
    verbose=True  # this will print out the sidecar file
)
anat_dir = t1w_bids_path.directory

# %%
# Let's have another look at our BIDS directory
print_dir_tree(output_path)

# %%
# Our BIDS dataset is now ready to be shared. We can easily estimate the
# transformation matrix using ``MNE-BIDS`` and the BIDS dataset.
# This function converts the anatomical landmarks stored in the T1 sidecar
# file into FreeSurfer surface RAS space, and aligns the landmarks in the
# electrophysiology data with them. This way your electrophysiology channel
# locations can be transformed to surface RAS space using the ``trans`` which
# is crucial for source localization and other uses of the FreeSurfer surfaces.
#
# .. note:: If this dataset were shared with you, you would first have to use
#           the T1 image as input for the FreeSurfer recon-all, see
#           :ref:`tut-freesurfer-mne`.
estim_trans = get_head_mri_trans(bids_path=bids_path, fs_subject='sample',
                                 fs_subjects_dir=fs_subjects_dir)

# %%
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
                      subjects_dir=fs_subjects_dir)

# Our MRI written to BIDS, we got `anat_dir` from our `write_anat` function
t1_nii_fname = op.join(anat_dir, 'sub-01_ses-01_T1w.nii.gz')

# Plot it
fig, axs = plt.subplots(3, 1, figsize=(7, 7), facecolor='k')
for point_idx, label in enumerate(('LPA', 'NAS', 'RPA')):
    plot_anat(t1_nii_fname, axes=axs[point_idx],
              cut_coords=mri_pos[point_idx, :],
              title=label, vmax=160)
plt.show()

# %%
# Writing FLASH MRI image
# -----------------------
#
# We can write another types of MRI data such as FLASH images for BEM models

flash_fname = op.join(fs_subjects_dir, 'sample', 'mri', 'flash', 'mef05.mgz')

flash_bids_path = \
    BIDSPath(subject=sub, session=ses, root=output_path, suffix='FLASH')

write_anat(
    image=flash_fname,
    bids_path=flash_bids_path,
    verbose=True
)

# %%
# Writing defaced and anonymized T1 image
# ---------------------------------------
#
# We can deface the MRI for anonymization by passing ``deface=True``.
t1w_bids_path = write_anat(
    image=t1_fname,  # path to the MRI scan
    bids_path=bids_path,
    landmarks=landmarks,
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

# %%
# Writing defaced and anonymized FLASH MRI image
# ----------------------------------------------
#
# Defacing the FLASH works just like the T1 as long as they are aligned.

# use ``trans`` to transform landmarks from the ``raw`` file to
# the voxel space of the image
landmarks = get_anat_landmarks(
    flash_fname,  # path to the FLASH scan
    info=raw.info,  # the MEG data file info from the same subject as the MRI
    trans=trans,  # our transformation matrix
    fs_subject='sample',  # freesurfer subject
    fs_subjects_dir=fs_subjects_dir,  # freesurfer subjects directory
)

flash_bids_path = write_anat(
    image=flash_fname,  # path to the MRI scan
    bids_path=flash_bids_path,
    landmarks=landmarks,
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

# %%
# Using manual landmark coordinates in scanner RAS
# ------------------------------------------------
#
# You can also find landmarks with a 3D image viewer (e.g. FreeView) if you
# have not aligned the channel locations (including fiducials) using the
# coregistration GUI or if this is just more convenient.
#
# .. note:: In FreeView, you need to use "RAS" and not "TkReg RAS" for this.
#           You can also use voxel coordinates but, in FreeView, they
#           are integers and so not as precise as the "RAS" decimal numbers.
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
    image=flash_fname,  # path to the MRI scan
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

# %%
# .. LINKS
#
# .. _coregistration GUI:
#    https://mne.tools/stable/auto_tutorials/forward/20_source_alignment.html#defining-the-headmri-trans-using-the-gui
# .. _mne_source_coords:
#    https://mne.tools/stable/auto_tutorials/source-modeling/plot_source_alignment.html
# .. _mne_sample_data:
#    https://mne.tools/stable/overview/datasets_index.html#sample
#
