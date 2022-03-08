"""
.. _ieeg-example:

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

5. Repeat the process for the ``fsaverage`` template coordinate space.

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
import numpy as np
import shutil

import nibabel as nib
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
# MNE-Python supports uses ``mni_tal`` and ``mri`` coordinate frames,
# corresponding to the ``fsaverage`` and ``ACPC`` (for an ACPC-aligned T1) BIDS
# coordinate systems respectively. All other coordinate coordinate frames in
# MNE-Python, if written with :func:`mne_bids.write_raw_bids`, must have
# an :attr:`mne_bids.BIDSPath.space` specified, and will be read in with
# the montage channel locations set to the coordinate frame 'unknown'.
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
write_raw_bids(raw, bids_path, anonymize=dict(daysback=40000),
               montage=montage, acpc_aligned=True, overwrite=True)

# check our output
print_dir_tree(bids_root)

# %%
# MNE-BIDS has created a suitable directory structure for us, and among other
# meta data files, it started an ``events.tsv`` and ``channels.tsv`` file,
# and created an initial ``dataset_description.json`` file on top!
#
# Now it's time to manually check the BIDS directory and the meta files to add
# all the information that MNE-BIDS could not infer. For instance, you must
# describe ``iEEGReference`` and ``iEEGGround`` yourself.
# It's easy to find these by searching for ``"n/a"`` in the sidecar files.

search_folder_for_text('n/a', bids_root)

# %%
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
raw2 = read_raw_bids(bids_path=bids_path)

# %%
# Now we have to go back to "head" coordinates with the head->mri transform.
#
# .. note:: If you were downloading this from ``OpenNeuro``, you would
#           have to run the Freesurfer ``recon-all`` to get the transforms.

montage2 = raw2.get_montage()

# this uses Freesurfer recon-all subject directory
montage2.add_estimated_fiducials('sample_seeg', subjects_dir=subjects_dir)

# get head->mri trans, invert from mri->head
trans2 = mne.transforms.invert_transform(
    mne.channels.compute_native_head_t(montage2))

# now the montage is properly in "head" and ready for analysis in MNE
raw2.set_montage(montage2)

# get the monage, apply the trans and make sure it's the same
# note: the head coordinates may differ because they are defined by
# the fiducials which are estimated; as long as the head->mri trans
# is computed with the same fiducials, the coordinates will be the same
# in ACPC space which is what matters
montage2 = raw.get_montage()  # get montage in 'head' coordinates
montage2.apply_trans(trans2)

# compare with standard
print('Recovered coordinate: {recovered}\n'
      'Saved coordinate:     {saved}'.format(
          recovered=montage2.get_positions()['ch_pos']['LENT 1'],
          saved=montage.get_positions()['ch_pos']['LENT 1']))

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
# :ref:`tut-working-with-seeg`. Note, this is only a linear transform and so
# one loses quite a bit of accuracy relative to the needs of intracranial
# researchers so it is quite suboptimal. A better option is to use a
# symmetric diffeomorphic transform to create a one-to-one mapping of brain
# voxels from the individual's brain to the template as shown in
# :ref:`tut-ieeg-localize`. Even so, it's better to provide the coordinates
# in the individual's brain space, as was done above, so that the researcher
# who uses the coordinates has the ability to tranform them to a template
# of their choice.

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
write_raw_bids(raw, bids_path, anonymize=dict(daysback=40000),
               montage=montage, overwrite=True)

# read in the BIDS dataset
raw2 = read_raw_bids(bids_path=bids_path)

# %%
# Now we should go back to "head" coordinates. We do this with ``fsaverage``
# fiducials which are in MNI space. In this case, you would not need to run
# the Freesurfer ``recon-all`` for the subject, you would just need a
# ``subjects_dir`` with ``fsaverage`` in it, which is accessible using
# :func:`mne.datasets.fetch_fsaverage`.

montage2 = raw2.get_montage()
# add fiducials for "mni_tal" (which is the coordinate frame fsaverage is in)
# so that it can properly be set to "head"
montage2.add_mni_fiducials(subjects_dir=subjects_dir)

# get the new head->mri (in this case mri == mni because fsavearge is in MNI)
mni_head_t = mne.channels.compute_native_head_t(montage2)

# set the montage transforming to the "head" coordinate frame
raw2.set_montage(montage2)

# check that we can recover the coordinates
print('Recovered coordinate head: {recovered}\n'
      'Saved coordinate head:     {saved}'.format(
          recovered=raw2.info['chs'][0]['loc'][:3],
          saved=raw.info['chs'][0]['loc'][:3]))

# check difference in trans
print('Recovered trans:\n{recovered}\n'
      'Original trans:\n{saved}'.format(
          recovered=mni_head_t['trans'].round(3),
          # combine head->mri with mri->mni to get head->mni
          # and then invert to get mni->head
          saved=np.linalg.inv(np.dot(trans['trans'], mri_mni_t['trans'])
                              ).round(3)))

# ensure that the data in MNI coordinates is exactly the same
# (within numerical precision)
montage2 = raw2.get_montage()  # get montage after transformed back to head
montage2.apply_trans(mne.transforms.invert_transform(mni_head_t))
print('Recovered coordinate: {recovered}\n'
      'Saved coordinate:     {saved}'.format(
          recovered=montage2.get_positions()['ch_pos']['LENT 1'],
          saved=montage.get_positions()['ch_pos']['LENT 1']))

# %%
# As you can see the coordinates stored in the ``raw`` object are slightly off.
# This is because the ``head`` coordinate frame is defined by the fiducials
# (nasion, left and right pre-auricular points), and, in the first case,
# the fiducials were found on the individual anatomy and then transformed
# to MNI space, whereas, in the second case, they were found directly on
# the template brain (this was done once for the template so that we could
# just load it from a file). This difference means that there are slightly
# different head->mri transforms. Once these transforms are applied, however,
# the positions are the same in MNI coordinates which is what is important.
#
# As a final step, let's go over how to assign the fiducials for a template
# brain where they are not found for you. Many template coordinate systems
# are allowed by BIDS but are not used in MNE-Python.
#
# .. note::
#     As of this writing, BIDS accepts channel coordinates in reference to the
#     the following template spaces: ``ICBM452AirSpace``,
#     ``ICBM452Warp5Space``, ``IXI549Space``, ``fsaverage``, ``fsaverageSym``,
#     ``fsLR``, ``MNIColin27``, ``MNI152Lin``,
#     ``MNI152NLin2009[a-c][Sym|Asym]``, ``MNI152NLin6Sym``,
#     ``MNI152NLin6ASym``, ``MNI305``, ``NIHPD``, ``OASIS30AntsOASISAnts``,
#     ``OASIS30Atropos``, ``Talairach`` and ``UNCInfant``. As discussed above,
#     it is recommended to share the coordinates in the individual subject's
#     anatomical reference frame so that researchers who use the data can
#     transform the coordinates to any of these templates that they choose.

pos = montage.get_positions()
# fiducial points are included in the sample data but could be found using
# Freeview (note the coordinates are in MNE "mri" coordinates which is the
# same as Freesurfers TkRegRAS but in meters not millimeters --
# divide the TkRegRAS values by 1000)
# or fiducial points could also be found with the MNE coregistration GUI
nas = pos['nasion']
lpa = pos['lpa']
rpa = pos['rpa']

print('Fiducial points determined from the template head anatomy:\n'
      f'nasion: {nas}\nlpa:    {lpa}\nrpa:    {rpa}')

# read raw in again to start over
raw2 = read_raw_bids(bids_path=bids_path)
montage2 = raw2.get_montage()

# note: for fsaverage, the montage will be in the coordinate frame
# 'mni_tal' because it is recognized by MNE but other templates will be
# in the 'unknown' coordinate frame because they are not recognized by MNE
pos2 = montage2.get_positions()

# here we will set the coordinate frame to be 'mri' because our channel
# positions and fiducials are in the Freesurfer surface RAS coordinate
# frame of the template T1 MRI (in this case fsaverage)
montage2 = mne.channels.make_dig_montage(  # add fiducials
    ch_pos=pos2['ch_pos'],
    nasion=nas,
    lpa=lpa,
    rpa=rpa,
    coord_frame='mri')

# get head->mri trans, invert from mri->head
trans2 = mne.transforms.invert_transform(
    mne.channels.compute_native_head_t(montage2))

# set the montage to transform back to 'head'
raw2.set_montage(montage2)

# check that the coordinates were recovered
montage2 = raw2.get_montage()  # get montage after transformed back to head
montage2.apply_trans(trans2)
print('Recovered coordinate: {recovered}\n'
      'Saved coordinate:     {saved}'.format(
          recovered=montage2.get_positions()['ch_pos']['LENT 1'],
          saved=montage.get_positions()['ch_pos']['LENT 1']))

# %%
# We showed how to add the fiducials from ``montage.add_mni_fiducials``
# yourself, which allows us to use any coordinate frame, even if it's not MNI,
# but we still assumed that the coordinates were in Freesurfer surface RAS.
# Unfortunately, this is not guranteed to be the case because the BIDS
# template descriptions only specify the anatomical space (as defined by the
# template T1 MRI) not the coordinate frame of the space; the coordinates
# could be in terms of surface RAS, voxels of the template MRI or scanner RAS
# MRI coordinates. See :ref:`tut-source-alignment` for a tutorial explaining
# the different coordinate frames. If the coordinates are in voxels or scanner
# RAS, we'll have to find the fiducials in those same coordinates.

# get the template T1 (must be mgz to have surface RAS transforms
# the Freesurfer command `mri_convert` can be used to convert
subjects_dir = op.join(mne.datasets.sample.data_path(), 'subjects')
template_T1 = nib.load(op.join(subjects_dir, 'fsaverage', 'mri', 'T1.mgz'))

# get vox->mri transform
vox_mri_t = template_T1.header.get_vox2ras_tkr()

# transform the channel data to voxels just to demonstrate how to transform it
# back (this is the case where the BIDS-formatted data in the template space is
# in voxel coordinates so it would already be transformed when read in)
raw = mne.io.read_raw_fif(op.join(  # load our raw data again
    misc_path, 'seeg', 'sample_seeg_ieeg.fif'))
montage = raw.get_montage()  # get the original montage
montage.apply_trans(trans)  # head->mri
mri_vox_trans = mne.transforms.Transform(
    fro='mri',
    to='mri_voxel',
    trans=np.linalg.inv(vox_mri_t)
)
scale_t = np.eye(4)
scale_t[:3, :3] *= 1000  # m->mm
montage.apply_trans(mne.transforms.Transform('mri', 'mri', scale_t))
montage.apply_trans(mri_vox_trans)  # transform our original montage

# print transformed fiducials, these could be found in the voxel locator in
# Freesurfer's `freeview` based on the template MRI since we wouldn't have
# them found for us for a template other than fsaverage
pos = montage.get_positions()
nas = pos['nasion']
lpa = pos['lpa']
rpa = pos['rpa']

print('Fiducial points determined from the template head anatomy in voxels:\n'
      f'nasion: {nas}\nlpa:    {lpa}\nrpa:    {rpa}')

# read raw in again to start over
raw2 = read_raw_bids(bids_path=bids_path)

# %%
# Now, it's the same as if we just got the montage from the raw in voxels
# i.e. if we would do ``montage = raw.get_montage()`` and get the positions
# directly in voxels instead of doing the transform to voxels first
montage2 = mne.channels.make_dig_montage(  # add fiducials
    ch_pos=pos['ch_pos'],
    nasion=nas,
    lpa=lpa,
    rpa=rpa,
    coord_frame='mri_voxel')
vox_mri_trans = mne.transforms.Transform(
    fro='mri_voxel',
    to='mri',
    trans=vox_mri_t
)
montage2.apply_trans(vox_mri_trans)
scale_t = np.eye(4)
scale_t[:3, :3] /= 1000  # mm->m
montage2.apply_trans(mne.transforms.Transform('mri', 'mri', scale_t))

# get head->mri trans, invert from mri->head
trans2 = mne.transforms.invert_transform(
    mne.channels.compute_native_head_t(montage2))

# set the montage to transform back to 'head'
raw2.set_montage(montage2)

# check that the coordinates were recovered
montage = raw.get_montage()  # get the original montage
montage.apply_trans(trans)  # head->mri
montage2 = raw2.get_montage()  # get montage after transformed back to head
montage2.apply_trans(trans2)
print('Recovered coordinate: {recovered}\n'
      'Saved coordinate:     {saved}'.format(
          recovered=montage2.get_positions()['ch_pos']['LENT 1'],
          saved=montage.get_positions()['ch_pos']['LENT 1']))

# %%
# Finally, the template could also be in scanner RAS:

# transform the channel data to scanner RAS just to demonstrate how to
# transform it back (this is the case where the BIDS-formatted data in the
# template space is in scanner RAS coordinates so it would already be
# transformed when read in)
raw = mne.io.read_raw_fif(op.join(  # load our raw data again
    misc_path, 'seeg', 'sample_seeg_ieeg.fif'))
montage = raw.get_montage()  # get the original montage
montage.apply_trans(trans)  # head->mri
scale_t = np.eye(4)
scale_t[:3, :3] *= 1000  # m->mm
montage.apply_trans(mne.transforms.Transform('mri', 'mri', scale_t))
montage.apply_trans(mri_vox_trans)  # transform our original montage
vox_ras_t = template_T1.header.get_vox2ras()  # no tkr for scanner RAS
vox_ras_trans = mne.transforms.Transform(
    fro='mri_voxel',
    to='ras',
    trans=vox_ras_t
)
montage.apply_trans(vox_ras_trans)

# print transformed fiducials, these could be found in the RAS locator in
# Freesurfer's `freeview` based on the template MRI
# note in this case, they are in mm and should not be transformed
pos = montage.get_positions()
nas = pos['nasion']
lpa = pos['lpa']
rpa = pos['rpa']

print('Fiducial points determined from the template head anatomy in '
      f'scanner RAS:\nnasion: {nas}\nlpa:    {lpa}\nrpa:    {rpa}')

# read raw in again to start over
raw2 = read_raw_bids(bids_path=bids_path)

# %%
# Now, it's the same as if we just got the montage from the raw in scanner RAS
# i.e. we would use ``montage = raw.get_montage()`` instead of having to do
# the transforms above
montage2 = mne.channels.make_dig_montage(  # add fiducials
    ch_pos=pos['ch_pos'],
    nasion=nas,
    lpa=lpa,
    rpa=rpa,
    coord_frame='ras')
ras_vox_t = template_T1.header.get_ras2vox()
vox_mri_t = template_T1.header.get_vox2ras_tkr()
ras_mri_trans = mne.transforms.Transform(
    fro='ras',
    to='mri',
    trans=np.dot(ras_vox_t, vox_mri_t)  # combine ras->vox and vox->mri to get
)                                       # ras->mri
montage2.apply_trans(ras_mri_trans)
scale_t = np.eye(4)
scale_t[:3, :3] /= 1000  # mm->m
montage2.apply_trans(mne.transforms.Transform('mri', 'mri', scale_t))

# get head->mri trans, invert from mri->head
trans2 = mne.transforms.invert_transform(
    mne.channels.compute_native_head_t(montage2))

# set the montage to transform back to 'head'
raw2.set_montage(montage2)

# check that the coordinates were recovered exactly this time
montage = raw.get_montage()  # get the original montage
montage.apply_trans(trans)  # head->mri
montage2 = raw2.get_montage()  # get montage after transformed back to head
montage2.apply_trans(trans2)
print('Recovered coordinate: {recovered}\n'
      'Saved coordinate:     {saved}'.format(
          recovered=montage2.get_positions()['ch_pos']['LENT 1'],
          saved=montage.get_positions()['ch_pos']['LENT 1']))

# %%
# In summary, as we saw, these standard template spaces that are allowable by
# BIDS are quite complicated. We therefore only cover these cases because
# datasets are allowed to be in these coordinate systems, and we want to be
# able to analyze them with MNE-Python. Because the coordinate space doesn't
# specify a coordinate frame within that space for the standard templates
# and because saving the raw data in the individual's ACPC space allows the
# person analyzing the data to transform the positions to whatever template
# they want, we recommend if at all possible, saving BIDS iEEG data in ACPC
# space.
