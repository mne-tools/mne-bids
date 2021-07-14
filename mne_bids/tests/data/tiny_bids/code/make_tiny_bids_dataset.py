"""Code used to generate the tiny_bids dataset."""
# %%
import json
import os
import os.path as op

import mne
import numpy as np

import mne_bids
from mne_bids import BIDSPath, write_raw_bids

data_path = mne.datasets.testing.data_path()
vhdr_fname = op.join(data_path, "montage", "bv_dig_test.vhdr")
captrak_path = op.join(data_path, "montage", "captrak_coords.bvct")

mne_bids_root = os.sep.join(mne_bids.__file__.split("/")[:-2])
tiny_bids = op.join(mne_bids_root, "mne_bids", "tests", "data", "tiny_bids")
os.makedirs(tiny_bids, exist_ok=True)

bids_path = BIDSPath(subject="01", task="rest", session="eeg", root=tiny_bids)

# %%
raw = mne.io.read_raw_brainvision(vhdr_fname)
montage = mne.channels.read_dig_captrak(captrak_path)

raw.set_channel_types(dict(ECG="ecg", HEOG="eog", VEOG="eog"))
raw.set_montage(montage)
raw.info["line_freq"] = 50

raw.info["subject_info"] = {
    "id": 1,
    "last_name": "Musterperson",
    "first_name": "Maxi",
    "middle_name": "Luka",
    "birthday": (1970, 10, 20),
    "sex": 2,
    "hand": 3,
}

raw.set_annotations(None)
events = np.array([[0, 0, 1], [1000, 0, 2]])
event_id = {"start_experiment": 1, "show_stimulus": 2}

# %%

write_raw_bids(
    raw, bids_path, events_data=events, event_id=event_id, overwrite=True
)

# %%

dataset_description_json = op.join(tiny_bids, "dataset_description.json")
with open(dataset_description_json, "r") as fin:
    ds_json = json.load(fin)

ds_json["Name"] = "tiny_bids"
ds_json["Authors"] = ["MNE-BIDS Developers", "And Friends"]

with open(dataset_description_json, "w") as fout:
    json.dump(ds_json, fout, indent=4)
    fout.write("\n")
