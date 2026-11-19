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
# License: BSD (3-clause)

###############################################################################
# We are importing everything we need for this example:

import os.path as op
import shutil as sh

import numpy as np
import matplotlib.pyplot as plt

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
if op.exists(output_path):
    sh.rmtree(output_path)
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
# retrieve our transformation matrix :code:`trans`.

# First create the BIDSPath object.
t1w_bids_path = BIDSPath(subject=sub, session=ses, root=output_path)

# We use the write_anat function
t1w_bids_path = write_anat(
    t1w=t1_mgh_fname,  # path to the MRI scan
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
# landmarks Nasion, LPA, and RPA (=left and right preauricular points) onto
# the brain image. For that, we can extract the location of Nasion, LPA, and
# RPA from the MEG file, apply our transformation matrix :code:`trans`, and
# plot the results.

# Get Landmarks from MEG file, 0, 1, and 2 correspond to LPA, NAS, RPA
# and the 'r' key will provide us with the xyz coordinates
pos = np.asarray((raw.info['dig'][0]['r'],
                  raw.info['dig'][1]['r'],
                  raw.info['dig'][2]['r']))


# We use a function from MNE-Python to convert MEG coordinates to MRI space
# for the conversion we use our estimated transformation matrix and the
# MEG coordinates extracted from the raw file. `subjects` and `subjects_dir`
# are used internally, to point to the T1-weighted MRI file: `t1_mgh_fname`
mri_pos = head_to_mri(pos=pos,
                      subject='sample',
                      mri_head_t=estim_trans,
                      subjects_dir=op.join(data_path, 'subjects')
                      )

# Our MRI written to BIDS, we got `anat_dir` from our `write_anat` function
t1_nii_fname = op.join(anat_dir, 'sub-01_ses-01_T1w.nii.gz')

# Plot it
fig, axs = plt.subplots(3, 1)
for point_idx, label in enumerate(('LPA', 'NAS', 'RPA')):
    plot_anat(t1_nii_fname, axes=axs[point_idx],
              cut_coords=mri_pos[point_idx, :],
              title=label)
plt.show()

###############################################################################
# We can deface the MRI for anonymization by passing ``deface=True``.
t1w_bids_path = write_anat(
    t1w=t1_mgh_fname,  # path to the MRI scan
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
plot_anat(t1_nii_fname, axes=ax, title='Defaced')
plt.show()

###############################################################################
# .. LINKS
#
# .. _coregistration GUI:
#    https://martinos.org/mne/stable/auto_tutorials/source-modeling/plot_source_alignment.html#defining-the-headmri-trans-using-the-gui  # noqa: E501
# .. _mne_source_coords:
#    https://www.martinos.org/mne/stable/auto_tutorials/source-modeling/plot_source_alignment.html  # noqa: E501
# .. _mne_sample_data:
#    https://martinos.org/mne/stable/manual/sample_dataset.html
#
