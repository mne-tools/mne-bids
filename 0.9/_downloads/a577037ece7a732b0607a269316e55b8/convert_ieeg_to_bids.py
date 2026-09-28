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

4. Cite MNE-BIDS.

5. Repeat the process for the ``fsaverage`` template coordinate frame.

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

from nilearn.plotting import plot_anat

import mne
from mne_bids import (BIDSPath, write_raw_bids, write_anat,
                      get_anat_landmarks, read_raw_bids,
                      search_folder_for_text, print_dir_tree)

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
# When the locations of the channels in this dataset were found
# in `Locating Intracranial Electrode Contacts
# <https://mne.tools/dev/auto_tutorials/clinical/10_ieeg_localize.html>`_,
# the T1 was aligned to ACPC. So, this montage is in an
# `ACPC-aligned coordinate system
# <https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems>`_.
# We can either save the channel positions in the subject's anatomical
# space (from their T1 image) or we can transform to a template space
# such as ``fsaverage``. To save them in the individual space, it is
# required that the T1 have been aligned to ACPC and then the channel positions
# be in terms of that coordinate system. Automated alignment to ACPC has not
# been implemented in MNE yet, so if the channel positions are not in
# an ACPC-aligned coordinate system, using a template (like ``fsaverage``)
# is the best option.

# estimate the transformation from "head" to "mri" space
trans = mne.coreg.estimate_head_mri_t('sample_seeg', subjects_dir)

# %%
# Now let's convert the montage to "mri"
montage = raw.get_montage()
montage.apply_trans(trans)  # head->mri

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
# Currently, MNE-Python supports the ``mni_tal`` and ``mri`` coordinate frames,
# corresponding to the ``fsaverage`` and ``ACPC`` (for an ACPC-aligned T1) BIDS
# coordinate systems respectively. All other coordinate coordinate frames in
# MNE-Python if written with :func:`mne_bids.write_raw_bids` are written with
# coordinate system ``'Other'``. Note, then we suggest using
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
# Now we just need make a :class:`mne_bids.BIDSPath` to save the data.
#
# .. warning:: By passing ``acpc_aligned=True``, we are affirming that
#              the T1 in this dataset is aligned to ACPC. This is very
#              difficult to check with a computer which is why this
#              step is required.

# Now convert our data to be in a new BIDS dataset.
bids_path = BIDSPath(subject=subject_id, task=task, root=bids_root)

# plot T1 to show that it is ACPC-aligned
# note that the origin is centered on the anterior commissure (AC)
# with the y-axis passing through the posterior commissure (PC)
T1_fname = op.join(subjects_dir, 'sample_seeg', 'mri', 'T1.mgz')
fig = plot_anat(T1_fname, cut_coords=(0, 0, 0))
fig.axes['x'].ax.annotate('AC', (2., -2.), (30., -40.), color='w',
                          arrowprops=dict(facecolor='w', alpha=0.5))
fig.axes['x'].ax.annotate('PC', (-31., -2.), (-80., -40.), color='w',
                          arrowprops=dict(facecolor='w', alpha=0.5))

# write ACPC-aligned T1
landmarks = get_anat_landmarks(T1_fname, raw.info, trans,
                               'sample_seeg', subjects_dir)
T1_bids_path = write_anat(T1_fname, bids_path, deface=True,
                          landmarks=landmarks)

# write `raw` to BIDS and anonymize it (converts to BrainVision format)
#
# we need to pass the `montage` argument for coordinate frames other than
# "head" which is what MNE uses internally in the `raw` object
#
# `acpc_aligned=True` affirms that our MRI is aligned to ACPC
# if this is not true, convert to `fsaverage` (see below)!
write_raw_bids(raw, bids_path, anonymize=dict(daysback=30000),
               montage=montage, acpc_aligned=True, overwrite=True)

# check our output
print_dir_tree(bids_root)

# %%
# MNE-BIDS has created a suitable directory structure for us, and among other
# meta data files, it started an ``events.tsv``` and ``channels.tsv`` file,
# and created an initial ``dataset_description.json`` file on top!
#
# Now it's time to manually check the BIDS directory and the meta files to add
# all the information that MNE-BIDS could not infer. For instance, you must
# describe ``iEEGReference`` and ``iEEGGround`` yourself.
# It's easy to find these by searching for ``"n/a"`` in the sidecar files.

search_folder_for_text('n/a', bids_root)

# Remember that there is a convenient JavaScript tool to validate all your BIDS
# directories called the "BIDS-validator", available as a web version and a
# command line tool:
#
# Web version: https://bids-standard.github.io/bids-validator/
#
# Command line tool: https://www.npmjs.com/package/bids-validator

# %%
# Step 3: Load channels from BIDS-formatted dataset and compare
# -------------------------------------------------------------
#
# Now we have written our BIDS directory. We can use
# :func:`read_raw_bids` to read in the data.

# read in the BIDS dataset to plot the coordinates
raw = read_raw_bids(bids_path=bids_path)

# compare with standard
print('Recovered coordinate: {recovered}\n'
      'Saved coordinate:     {saved}'.format(
          recovered=raw.info['chs'][0]['loc'][:3],
          saved=montage.dig[0]['r']))

# %%
# Now we have to go back to "head" coordinates with the head->mri transform.
#
# .. note:: If you were downloading this from ``OpenNeuro``, you would
#           have to run the Freesurfer ``recon-all`` to get the transforms.

montage = raw.get_montage()
# this uses Freesurfer recon-all subject directory
montage.add_estimated_fiducials('sample_seeg', subjects_dir=subjects_dir)
# now the montage is properly in "head" and ready for analysis in MNE
raw.set_montage(montage)

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
# Step 5: Store coordinates in a template space
# ---------------------------------------------
# Alternatively, if your T1 is not aligned to ACPC-space or you prefer to
# store the coordinates in a template space (e.g. ``fsaverage``) for another
# reason, you can also do that.
#
# Here we'll use the MNI Talairach transform to get to ``fsaverage`` space
# from "mri" aka surface RAS space.
# ``fsaverage`` is very useful for group analysis as shown in
# `Working with SEEG
# <https://mne.tools/stable/auto_tutorials/misc/plot_seeg.html>`_.

# ensure the output path doesn't contain any leftover files from previous
# tests and example runs
if op.exists(bids_root):
    shutil.rmtree(bids_root)

# load our raw data again
raw = mne.io.read_raw_fif(op.join(
    misc_path, 'seeg', 'sample_seeg_ieeg.fif'))
raw.info['line_freq'] = 60  # specify power line frequency as required by BIDS

# get Talairach transform
mri_mni_t = mne.read_talxfm('sample_seeg', subjects_dir)

# %%
# Now let's convert the montage to MNI Talairach ("mni_tal").
montage = raw.get_montage()
montage.apply_trans(trans)  # head->mri
montage.apply_trans(mri_mni_t)

# write to BIDS, this time with a template coordinate system
write_raw_bids(raw, bids_path, anonymize=dict(daysback=30000),
               montage=montage, overwrite=True)

# read in the BIDS dataset
raw = read_raw_bids(bids_path=bids_path)

# check that we can recover the coordinates
print('Recovered coordinate: {recovered}\n'
      'Saved coordinate:     {saved}'.format(
          recovered=raw.info['chs'][0]['loc'][:3],
          saved=montage.dig[0]['r']))

# %%
# Now we should go back to "head" coordinates. We do this with ``fsaverage``
# fiducials which are in MNI space. In this case, you would not need to run
# the Freesurfer ``recon-all`` for the subject, you would just need a
# ``subjects_dir`` with ``fsaverage`` in it, which is accessible using
# :func:`mne.datasets.fetch_fsaverage`.

montage = raw.get_montage()
# add fiducials for "mni_tal" (which is the coordinate frame fsaverage is in)
# so that it can properly be set to "head"
montage.add_mni_fiducials(subjects_dir=subjects_dir)

# Many other templates are included in the Freesurfer installation,
# so, for those, the fiducials can be estimated with
# ``montage.add_estimated_fiducials(template, os.environ['FREESURFER_HOME'])``
# where ``template`` maybe be ``cvs_avg35_inMNI152`` for instance
raw.set_montage(montage)
