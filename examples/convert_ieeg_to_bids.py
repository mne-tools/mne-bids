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

.. _iEEG part of the BIDS specification: https://bids-specification.readthedocs.io/en/latest/modality-specific-files/intracranial-electroencephalography.html
.. _appendix VIII: https://bids-specification.readthedocs.io/en/stable/appendices/coordinate-systems.html
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
                      search_folder_for_text, print_dir_tree,
                      template_to_head, convert_montage_to_ras,
                      convert_montage_to_mri)

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
# When the locations of the channels in this dataset were found in
# :ref:`Locating Intracranial Electrode Contacts <tut-ieeg-localize>`,
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
# Now let's convert the montage to "ras"
montage = raw.get_montage()
montage.apply_trans(trans)  # head->mri
convert_montage_to_ras(montage, 'sample_seeg', subjects_dir)  # mri->ras

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
# MNE-Python supports using ``mni_tal`` and ``mri`` coordinate frames,
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
# Now we have to go back to "head" coordinates.
#
# .. note:: If you were downloading this from ``OpenNeuro``, you would
#           have to run the Freesurfer ``recon-all`` to get the transforms.

montage2 = raw2.get_montage()

# we need to go from scanner RAS back to surface RAS (requires recon-all)
convert_montage_to_mri(montage2, 'sample_seeg', subjects_dir=subjects_dir)

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
montage = raw.get_montage()  # the original montage in 'head' coordinates
montage.apply_trans(trans)
montage2 = raw2.get_montage()  # the recovered montage in 'head' coordinates
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
#
# .. note::
#
#     For ``fsaverage``, the template coordinate system was defined
#     so that ``scanner RAS`` is equivalent to ``surface RAS``.
#     BIDS requires that template data be in ``scanner RAS`` so for
#     coordinate frames where this is not the case, the coordinates
#     must be converted (see below).

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
# MNE-Python uses ``head`` coordinates with a ``head -> mri`` ``trans`` so we
# need to make sure to get our data in this form. As shown below, the montage
# is in the ``mni_tal`` coordinate frame but doesn't have fiducials. The
# ``head`` coordinate frame is defined based on the fiducial points so we need
# to add these. Fortunately, there is a convenient function
# (:func:`mne_bids.template_to_head`) that loads stored fiducials and takes
# care of the transformations. Once this function is applied, you can use
# the ``raw`` object and the ``trans`` as in any MNE example
# (e.g. :ref:`tut-working-with-seeg`).

# use `coord_frame='mri'` to indicate that the montage is in surface RAS
# and `unit='m'` to indicate that the units are in meters
trans2 = template_to_head(
    raw2.info, space='fsaverage', coord_frame='mri', unit='m')[1]
# this a bit confusing since we transformed from mri->mni and now we're
# saying we're back in 'mri' but that is because we were in the surface RAS
# coordinate frame of `sample_seeg` and transformed to 'mni_tal', which is the
# surface RAS coordinate frame for `fsaverage`: since MNE denotes surface RAS
# as 'mri', both coordinate frames are 'mri', it's just that 'mni_tal' is 'mri'
# when the subject is 'fsaverage'

# %%
# Let's check that we can recover the original coordinates from the BIDS
# dataset now that we are working in the ``head`` coordinate frame with a
# ``head -> mri`` ``trans`` which is the setup MNE-Python is designed around.

# check that we can recover the coordinates
print('Recovered coordinate head: {recovered}\n'
      'Original coordinate head:  {original}'.format(
          recovered=raw2.info['chs'][0]['loc'][:3],
          original=raw.info['chs'][0]['loc'][:3]))

# check difference in trans
print('Recovered trans:\n{recovered}\n'
      'Original trans:\n{original}'.format(
          recovered=trans2['trans'].round(3),
          # combine head->mri with mri->mni to get head->mni
          # and then invert to get mni->head
          original=np.linalg.inv(np.dot(trans['trans'], mri_mni_t['trans'])
                                 ).round(3)))

# ensure that the data in MNI coordinates is exactly the same
# (within computer precision)
montage2 = raw2.get_montage()  # get montage after transformed back to head
montage2.apply_trans(trans2)
print('Recovered coordinate: {recovered}\n'
      'Original coordinate:  {original}'.format(
          recovered=montage2.get_positions()['ch_pos']['LENT 1'],
          original=montage.get_positions()['ch_pos']['LENT 1']))

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
# As a final step, let's go over how to assign coordinate systems that are
# not recognized by MNE-Python. Many template coordinate systems are allowed by
# BIDS but are not used in MNE-Python. For these templates, the fiducials have
# been found and the transformations have been pre-computed so that we can
# get our coordinates in the ``head`` coordinate frame that MNE-Python uses.
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
#
# BIDS requires that the template be stored in ``scanner RAS`` coordinates
# so first we'll convert our original data to ``scanner RAS`` and then
# convert it back. Just in case the template electrode coordinates are
# provided in voxels or the unit is not specified, these options are able
# to be overridden in :func:`mne_bids.template_to_head` for ease of use.
#
# .. warning::
#
#     If no coordinate frame is passed to :func:`mne_bids.template_to_head`
#     it will infer ``voxels`` if the coordinates are only positive and
#     ``scanner RAS`` otherwise. Be sure not to use the wrong coordinate
#     frame! ``surface RAS`` and ``scanner RAS`` are quite similar which
#     is especially confusing, but, fortunately, in most of the Freesurfer
#     template coordinate systems ``surface RAS`` is identical to
#     ``scanner RAS``. ``surface RAS`` is a Freesurfer coordinate frame so
#     it is most likely to be used with Freesurfer template coordinate
#     systems). This is the case for ``fsaverage``, ``MNI305`` and
#     ``fsaverageSym`` but not ``fsLR``.

# %%
# The template should be in scanner RAS:

# ensure the output path doesn't contain any leftover files from previous
# tests and example runs
if op.exists(bids_root):
    shutil.rmtree(bids_root)

# get a template mgz image to transform the montage to voxel coordinates
subjects_dir = op.join(mne.datasets.sample.data_path(), 'subjects')
template_T1 = nib.load(op.join(subjects_dir, 'fsaverage', 'mri', 'T1.mgz'))

# get voxels to surface RAS and scanner RAS transforms
vox_mri_t = template_T1.header.get_vox2ras_tkr()  # surface RAS
vox_ras_t = template_T1.header.get_vox2ras()  # scanner RAS

raw = mne.io.read_raw_fif(op.join(  # load our raw data again
    misc_path, 'seeg', 'sample_seeg_ieeg.fif'))
montage = raw.get_montage()  # get the original montage
montage.apply_trans(trans)  # head->mri
montage.apply_trans(mri_mni_t)  # mri->mni
pos = montage.get_positions()
ch_pos = np.array(list(pos['ch_pos'].values()))  # get an array of positions
# mri -> vox and m -> mm
ch_pos = mne.transforms.apply_trans(np.linalg.inv(vox_mri_t), ch_pos * 1000)
ch_pos = mne.transforms.apply_trans(vox_ras_t, ch_pos)

montage_ras = mne.channels.make_dig_montage(
    ch_pos=dict(zip(pos['ch_pos'].keys(), ch_pos)), coord_frame='ras')

# specify our standard template coordinate system space
bids_path.update(datatype='ieeg', space='fsaverage')

# write to BIDS, this time with a template coordinate system in voxels
write_raw_bids(raw, bids_path, anonymize=dict(daysback=40000),
               montage=montage_ras, overwrite=True)

# %%
# Now, let's load our data and convert our montage to ``head``.

raw2 = read_raw_bids(bids_path=bids_path)
trans2 = template_to_head(  # unit='auto' automatically determines it's in mm
    raw2.info, space='fsaverage', coord_frame='ras', unit='auto')[1]

# %%
# Let's check to make sure again that the original coordinates from the BIDS
# dataset were recovered.

montage2 = raw2.get_montage()  # get montage after transformed back to head
montage2.apply_trans(trans2)  # apply trans to go back to 'mri'
print('Recovered coordinate: {recovered}\n'
      'Original coordinate:  {original}'.format(
          recovered=montage2.get_positions()['ch_pos']['LENT 1'],
          original=montage.get_positions()['ch_pos']['LENT 1']))

# %%
# In summary, as we saw, these standard template spaces that are allowable by
# BIDS are quite complicated. We therefore only cover these cases because
# datasets are allowed to be in these coordinate systems, and we want to be
# able to analyze them with MNE-Python. BIDS data in a template coordinate
# space doesn't allow you to convert to a template of your choosing so it is
# better to save the raw data in the individual's ACPC space. Thus, we
# recommend, if at all possible, saving BIDS iEEG data in ACPC coordinate space
# corresponding to the individual subject's brain, not in a template
# coordinate frame.
