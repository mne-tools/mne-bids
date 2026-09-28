"""
==============================
13. Anonymizing a BIDS dataset
==============================

Consider the following scenario:

- You've created a BIDS dataset.
- Now you want to make this dataset available to the public.
- Therefore, all personally identifying information must be removed.

While :func:`mne_bids.write_raw_bids` and :func:`mne_bids.write_anat` can be
used to store anonymized copies of data (by passing the ``anonymize`` and
``deface`` keyword arguments, respectively), using these functions to anonymize
an entire existing dataset can be cumbersome and error-prone.

MNE-BIDS provides a dedicated function, :func:`mne_bids.anonymize_dataset`,
to do the heavy lifting for you, automatically.
"""

# Authors: Richard Höchenberger <richard.hoechenberger@gmail.com>
# License: BSD-3-Clause

# %%
import shutil
from pathlib import Path
import mne
from mne_bids import (
    BIDSPath, write_raw_bids, write_anat, write_meg_calibration,
    write_meg_crosstalk, anonymize_dataset, print_dir_tree
)

data_path = Path(mne.datasets.sample.data_path())
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
            'Visual/Right': 4, 'Smiley': 5, 'Button': 32}

raw_path = data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
raw_er_path = data_path / 'MEG' / 'sample' / 'ernoise_raw.fif'  # empty-room
events_path = data_path / 'MEG' / 'sample' / 'sample_audvis_raw-eve.fif'
cal_path = data_path / 'SSS' / 'sss_cal_mgh.dat'
ct_path = data_path / 'SSS' / 'ct_sparse_mgh.fif'
t1w_path = data_path / 'subjects' / 'sample' / 'mri' / 'T1.mgz'

bids_root = data_path.parent / 'MNE-sample-data-bids'
bids_root_anon = data_path.parent / 'MNE-sample-data-bids-anon'

# %%
# To ensure the output paths don't contain any leftover files from previous
# tests and example runs, we simply delete it.
#
# .. warning:: Do not delete directories that may contain important data!
#

if bids_root.exists():
    shutil.rmtree(bids_root)
if bids_root_anon.exists():
    shutil.rmtree(bids_root_anon)

# %%
bids_path = BIDSPath(
    subject='ABC123', task='audiovisual', root=bids_root, datatype='meg'
)
bids_path_er = bids_path.copy().update(
    subject='emptyroom', task='noise', session='20021206'
)

raw = mne.io.read_raw_fif(raw_path, verbose=False)
raw_er = mne.io.read_raw_fif(raw_er_path, verbose=False)
# specify power line frequency as required by BIDS
raw.info['line_freq'] = 60
raw_er.info['line_freq'] = 60

# Write empty-room data
write_raw_bids(raw=raw_er, bids_path=bids_path_er, verbose=False)

# Write experimental MEG data, fine-calibration and crosstalk files
write_raw_bids(
    raw=raw, bids_path=bids_path, events_data=events_path, event_id=event_id,
    empty_room=bids_path_er, verbose=False
)
write_meg_calibration(cal_path, bids_path=bids_path, verbose=False)
write_meg_crosstalk(ct_path, bids_path=bids_path, verbose=False)

# Write anatomical scan
# We pass the MRI landmark coordinates, which will later be required for
# automated defacing
mri_landmarks = mne.channels.make_dig_montage(
    lpa=[66.08580, 51.33362, 46.52982],
    nasion=[41.87363, 32.24694, 74.55314],
    rpa=[17.23812, 53.08294, 47.01789],
    coord_frame='mri_voxel'
)
bids_path.datatype = 'anat'
write_anat(
    image=t1w_path, bids_path=bids_path, landmarks=mri_landmarks, verbose=False
)

# %%
# Basic anonymization
# -------------------
# Now we're ready to anonymize the dataset!

anonymize_dataset(bids_root_in=bids_root, bids_root_out=bids_root_anon)

# %%
# That's it! Let's have a look at directory structure of the anonymized
# dataset.
print_dir_tree(bids_root_anon)

# %%
# You can see that the subject ID was changed to a number (in this case, the
# digit ``1```), and the recording dates have been shifted backward in time (as
# indicated by the ``emptyroom`` session name). Anonymized IDs are zero-padded
# numbers ranging from 1 to :math:`N`, where :math:`N` is the total number of
# participants (excluding the ``emptyroom`` pseudo-subject).
#
# Limiting to specific data types
# -------------------------------
# By default, :func:`mne_bids.anonymize_dataset` will anonymize
# electrophysiological data and anatomical MR scans (T1-weighted and FLASH).
# You can limit which data types to convert using the ``datatypes`` keyword
# argument. The parameter can be a string (e.g., ``'meg'``, ``'eeg'``,
# ``'anat'``) or a list of such strings.

shutil.rmtree(bids_root_anon)
anonymize_dataset(
    bids_root_in=bids_root,
    bids_root_out=bids_root_anon,
    datatypes='anat'  # Only anatomical data
)
print_dir_tree(bids_root_anon)

# %%
# Specifying time shift
# ---------------------
# Anonymization involves shifting the recording dates back in time. MNE-BIDS
# will try to automatically choose a suitable time shift. You may also
# explicitly specify by how many days you wish to shift the recording dates
# back in time via the ``daysback`` parameter. To avoid the time shift, pass
# ``daysback=0``.

shutil.rmtree(bids_root_anon)
anonymize_dataset(
    bids_root_in=bids_root,
    bids_root_out=bids_root_anon,
    datatypes='meg',  # Only MEG data
    daysback=10
)
print_dir_tree(bids_root_anon / 'sub-emptyroom')  # Easy to see effects here

# %%
# Specifying subject IDs
# ----------------------
# Anonymized subject IDs are automatically generated as unique numbers in
# ascending order. You can control this behavior via the ``subject_mapping``
# parameter. Set it to ``None`` to avoid changing the subject IDs, e.g., in
# case they're already anonymized. You can pass a dictionary that maps original
# subject IDs to the anonymize IDs. Lastly, you can also pass a function that
# accepts a list of original IDs and returns such a dictionary.

shutil.rmtree(bids_root_anon)

subject_mapping = {
    'ABC123': 'anonymous',
    'emptyroom': 'emptyroom'
}

anonymize_dataset(
    bids_root_in=bids_root,
    bids_root_out=bids_root_anon,
    datatypes='meg',
    subject_mapping=subject_mapping
)
print_dir_tree(bids_root_anon)

# %%
# Reproducibility
# ---------------
# Every time you run this function, the automatically-generated subject IDs and
# the timeshift may differ (unless you excplicitly specify them as described
# above), as they are determined randomly.
#
# To ensure results are reproducible across runs, you can pass the
# ``random_state`` parameter, causing the random number generator to produce
# the same results every time you execute the function. This may come in handy
# e.g. in situations where you discover a problem with the data while working
# with the anonymized dataset, fix the issue in the original dataset, and
# run anonymization again.
#
# (Note that throughout this example, we only had a single subject in our
# dataset, meaning it will always be assigned the anonymized ID ``1``. Only
# in a dataset with multiple subjects will the effects of randomly-picked IDs
# become apparent.)
#
# .. note::
#    Passing ``random_state`` merely guarantees that subject IDs and time shift
#    remain the same across anonymization runs if the original dataset
#    remained unchanged. It does **not** allow you to incrementally add data
#    (e.g., a new participant) to an anonymized dataset: If the original
#    dataset changes and you want the changes anonymized, you will need to
#    anonymize the entire dataset again.

for i in range(2):
    print(f'\n\nRun {i+1}\n')
    shutil.rmtree(bids_root_anon)
    anonymize_dataset(
        bids_root_in=bids_root,
        bids_root_out=bids_root_anon,
        datatypes='meg',
        random_state=42
    )
    print_dir_tree(bids_root_anon)
