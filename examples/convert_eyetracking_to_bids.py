"""
.. currentmodule:: mne_bids

.. _ex-convert-eyetracking-to-bids:

=======================================
Convert eyetracking data to BIDS Format
=======================================

This example shows how to convert Eyelink eyetracking data to BIDS using
MNE-BIDS.

.. seealso::

   | `Working with eyetracking data in MNE-Python <https://mne.tools/stable/auto_tutorials/preprocessing/90_eyetracking_data.html>`_
   | `The Eyetracking BIDS specification`_
"""  # noqa: D400

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
import json
import shutil
from pprint import pprint

import mne
from mne.datasets import testing
from mne.datasets.eyelink import data_path as eyelink_data_path
from mne.preprocessing.eyetracking import read_eyelink_calibration

from mne_bids import BIDSPath, print_dir_tree, read_raw_bids, write_raw_bids
from mne_bids.physio import read_eyetrack_calibration, write_eyetrack_calibration

# %%
# Load example eyetracking data
# -----------------------------
#
# Here we use an Eyelink file from the MNE-Python testing data.

data_path = testing.data_path(download=False)
eyetrack_fpath = data_path / "eyetrack" / "test_eyelink.asc"
raw = mne.io.read_raw_eyelink(eyetrack_fpath)
cals = read_eyelink_calibration(eyetrack_fpath)
raw

# %%
raw.plot(scalings="auto")

# %%
# Where are BIDS compliant eyetracking files stored?
# --------------------------------------------------
#
# Eyetracking-only data is stored in the ``'beh'`` modality directory. Eyetracking data
# that was collected alongside another modality (``eeg``, ``meg``, etc) will be stored
# in the same directory as that modality (``'eeg'``, ``'meg'``, etc). As such, when
# defining a BIDSPath instance to read or write eyetracking data, pass
# ``datatype="beh"`` for eyetracking-only data. If the data were collected alongside
# EEG data, then you would pass ``datatype='eeg'``. Either way, you should also pass
# ``suffix="physio"``, and ``recording='eye1'`` to the BIDSPath constructor
# (even for binocular data, where there is also a ``<match>_recording-eye2.tsv.gz``
# file. MNE-BIDS will handle reading and writing of ``eye2`` data for us.)

# %%
bids_root = data_path.parent / "MNE-eyetrack-data-bids-example"
if bids_root.exists():
    shutil.rmtree(bids_root)

bids_path = BIDSPath(
    root=bids_root,
    datatype="beh",
    subject="01",
    session="01",
    task="eyetrack",
    run="01",
    recording="eye1",
    suffix="physio",
    extension=".tsv.gz",
)

# %%
# Write BIDS eyetracking files
# ----------------------------
#
# MNE-BIDS will write one ``*_physio.tsv.gz`` + ``*_physio.json`` pair per eye,
# and matching ``*_physioevents.tsv.gz`` files. Additionally, we are going to convert
# our eyetracking eyegaze channels from pixels-on-screen to radians-of-visual-angle, to
# demonstrate how BIDS stores the units.

cal = cals[0]
cal["screen_resolution"] = (1920, 1080)
cal["screen_size"] = (0.53, 0.3)
cal["screen_distance"] = 0.9
mne.preprocessing.eyetracking.convert_units(raw, calibration=cal, to="radians")


write_raw_bids(raw=raw, bids_path=bids_path, allow_preload=True, overwrite=True)

# %%
# Add calibration metadata to the eyetracking sidecar
# ---------------------------------------------------
#
# We can update the ``*_physio.json`` sidecar with calibration eyetracking
# calibration information.

# %%
write_eyetrack_calibration(bids_path, cals)

# %%
# Inspect the generated BIDS directory tree.

# %%
print_dir_tree(bids_root)

# %%
# Inspect one sidecar JSON file.
# ------------------------------
# Notice 1) that the calibration information was written to this physio.json file, and
# 2) that the units for the eyegaze channels are ``'rad'``, meaning "radians of visual
# angle."

# %%
eye1_json = bids_path.fpath.with_suffix("").with_suffix(".json")
print(f"Filepath: {eye1_json}")
pprint(json.loads(eye1_json.read_text()), indent=2)

# %%
# Read the eyetracking data back from BIDS
# ----------------------------------------
#

# %%
raw_in = read_raw_bids(bids_path=bids_path)
raw_in

# %%
raw_in.plot(scalings=dict(pupil="auto"))

# %%
cals_in = read_eyetrack_calibration(bids_path)

# %%
# Convert simultaneous EEG + eyetracking data to BIDS
# ---------------------------------------------------
#
# When eyetracking data is collected simultaneously with another BIDS modality, then the
# eyetracking files will be written to that modality folder. In other words, instead of
# being written to a ``beh`` directory, as the stand-alone eyetracking data that we just
# used was, the dataset below will be written alongside the EEG data in the ``'eeg'``
# directory. Additionally, unlike the previous example, where we converted our eyegaze
# channel units from pixels-on-screen to radians-of-visual-angle, in this example we'll
# keep the data as pixels-on-screen, and this will be reflected in the BIDS metadata.

eyelink_root = eyelink_data_path()
et_fpath = eyelink_root / "eeg-et" / "sub-01_task-plr_eyetrack.asc"
eeg_fpath = eyelink_root / "eeg-et" / "sub-01_task-plr_eeg.mff"

raw_et = mne.io.read_raw_eyelink(et_fpath)
raw_eeg = mne.io.read_raw_egi(eeg_fpath, events_as_annotations=True).load_data()

# %%
# Interpolate NaN blink periods prior to merging with EEG data

# %%
mne.preprocessing.eyetracking.interpolate_blinks(
    raw_et, buffer=(0.05, 0.2), interpolate_gaze=True
)

# %%
# Merge with EEG data
et_events = mne.find_events(raw_et, min_duration=0.01, shortest_event=1, uint_cast=True)
eeg_events = mne.find_events(raw_eeg, stim_channel="DIN3")
# Convert event onsets from samples to seconds
et_flash_times = et_events[:, 0] / raw_et.info["sfreq"]
eeg_flash_times = eeg_events[:, 0] / raw_eeg.info["sfreq"]
# Align the data
mne.preprocessing.realign_raw(
    raw_et, raw_eeg, et_flash_times, eeg_flash_times, verbose="error"
)

# %%
# Add EEG channels to the eye-tracking raw object

# %%
raw_et.add_channels([raw_eeg], force_update_info=True)
del raw_eeg  # free up some memory

# %%
# Write the merged EEG + eyetracking recording.

# %%
bids_root_simultaneous = eyelink_root.parent / "MNE-eyetrack-eeg-bids-example"
if bids_root_simultaneous.exists():
    shutil.rmtree(bids_root_simultaneous)

bids_path_eeg = BIDSPath(
    root=bids_root_simultaneous,
    subject="01",
    session="01",
    run="01",
    task="plr",
    datatype="eeg",
    suffix="eeg",
)
write_raw_bids(
    raw_et, bids_path_eeg, allow_preload=True, format="BrainVision", verbose="error"
)

# %%
# Inspect the generated dataset. Besides EEG files, MNE-BIDS will create
# eyetracking ``*_physio`` files in the same modality folder.

# %%
print_dir_tree(bids_root_simultaneous)

# %%
# Again, let's inspect the saved metadata for one eye. Note that the units for our
# eyegaze channels are 'pixel', meaining these data are 'pixel-on-screen' coordinates.
# %%
eye1_json = bids_path_eeg.find_matching_sidecar(suffix="physio", extension=".json")
print(f"Filepath: {eye1_json}")
pprint(json.loads(eye1_json.read_text()), indent=2)

# %%
# Read back one eye's eyetracking recording from the simultaneous dataset.
# ------------------------------------------------------------------------
# Note that we will have to read the eyetracking data back into MNE-Python on its own.
# In other words, we will need to also read the EEG data back in and merge the two
# modalities as we did before (but won't repeat those steps here for the sake of
# brevity.)

# %%
bids_path_eye1 = bids_path_eeg.copy().update(
    suffix="physio",
    extension=".tsv.gz",
    recording="eye1",
)
raw_eye1 = read_raw_bids(bids_path_eye1)
raw_eye1
