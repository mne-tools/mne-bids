"""
================================
Add fiducial points for Defacing
================================

In this example, we will add fiducial points so that mne-bids
knows where the face is for the defacing anonymization algorithm.
"""

# Authors: Alex Rockhill <aprockhill206@gmail.com>
# License: BSD (3-clause)

###############################################################################
# We are importing everything we need for this example:
import os
import os.path as op
import importlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from nilearn.plotting import plot_anat

import mne
from mne.datasets import sample

from mne_bids import write_anat

###############################################################################
# Load in the raw data to which the dig montage will be added.

data_path = sample.data_path()

# load in the raw data for the fiducials
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(raw_fname)

# Get Landmarks from the file. 0, 1, and 2 correspond to LPA, NAS, RPA
# and the 'r' key will provide us with the xyz coordinates
pos = np.asarray((raw.info['dig'][0]['r'],
                  raw.info['dig'][1]['r'],
                  raw.info['dig'][2]['r']))
del raw

# load in the T1
t1_mgh_fname = op.join(data_path, 'subjects', 'sample', 'mri', 'T1.mgz')

# load in ECoG data
mat = loadmat(mne.datasets.misc.data_path() + '/ecog/sample_ecog.mat')
ch_names = mat['ch_names'].tolist()
elec = mat['elec']  # electrode positions given in meters

###############################################################################
# Create unaligned datapoints in a dig montage and then align them to the MRI.

# Now we make a montage stating that the sEEG contacts are in head
# coordinate system (although they are in MRI). This is compensated
# by the fact that below we do not specicty a trans file so the Head<->MRI
# transform is the identity.
montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, elec)),
                                        coord_frame='head',
                                        lpa=np.zeros((3,)),
                                        nasion=np.zeros((3,)),
                                        rpa=np.zeros((3,)))
# I was going to use pos[0:2] but interestingly zeros work also
# possibly due to an automicatic fiduical detection algorithm?

# Since human heads are relatively similar in size, aligning the
# fiducial points from the example data will work pretty well.
info = mne.create_info(ch_names, 1000., 'ecog', montage=montage)

# Use random data as a placeholder
raw = mne.io.RawArray(np.random.random((len(ch_names), 1000)), info)

raw.save('sample_ecog.fif')  # bug with not saving out if raw passed directly

# Now align the fiducial points and save out the aligned trans file.
# See https://mne.tools/stable/generated/mne.gui.coregistration.html
# for instructions on how to do this. Briefly, click on the LPA
# checkbox, click on the LPA on the head, then do the same for
# nasion and RPA. Then click save in the same menu box.
mne.gui.coregistration(subject='sample',
                       subjects_dir=op.join(data_path, 'subjects'),
                       inst='sample_ecog.fif')

os.remove('sample_ecog.fif')

# The changed fiducials were saved as sample-fiducials.fif in this case.
fids, coord_frame = mne.io.read_fiducials(
    op.join(data_path, 'subjects', 'sample', 'bem', 'sample-fiducials.fif'))

montage2 = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, elec)),
                                         coord_frame='head',
                                         lpa=fids[0]['r'],
                                         nasion=fids[1]['r'],
                                         rpa=fids[2]['r'])

raw.set_montage(montage2)

# No trans specified thus the identity matrix is our transform
trans = mne.transforms.Transform(fro='head', to='mri')

###############################################################################
# Deface the MRI for anonymization

# io parameters
output_path = op.abspath(op.join(data_path, '..', 'MNE-sample-data-bids'))
sub = '01'
ses = '01'

anat_dir = write_anat(bids_root=output_path,  # the BIDS dir we wrote earlier
                      subject=sub,
                      t1w=t1_mgh_fname,  # path to the MRI scan
                      session=ses,
                      raw=raw,  # the raw MEG data file connected to the MRI
                      trans=trans,  # our transformation matrix
                      deface=True,
                      overwrite=True,
                      verbose=True  # this will print out the sidecar file
                      )

# Our MRI written to BIDS, we got `anat_dir` from our `write_anat` function
t1_nii_fname = op.join(anat_dir, 'sub-01_ses-01_T1w.nii.gz')

# Plot it
importlib.reload(plt)  # bug due to mne.gui
fig, ax = plt.subplots()
plot_anat(t1_nii_fname, axes=ax, title='Defaced')
plt.show()
