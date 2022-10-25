"""Code used to generate the tiny_bids dataset."""

# %%
import json
from pathlib import Path

import mne
import numpy as np

import mne_bids
from mne_bids import BIDSPath, write_raw_bids

data_path = mne.datasets.testing.data_path(download=False)
assert mne.datasets.has_dataset('testing'), 'Download testing data'
vhdr_path = data_path / "montage" / "bv_dig_test.vhdr"
captrak_path = data_path / "montage" / "captrak_coords.bvct"

mne_bids_root = Path(mne_bids.__file__).parents[1]
tiny_bids_root = mne_bids_root / "mne_bids" / "tests" / "data" / "tiny_bids"
tiny_bids_root.mkdir(exist_ok=True)

bids_path = BIDSPath(
    subject="01", task="rest", session="eeg", suffix="eeg", extension=".vhdr",
    datatype="eeg", root=tiny_bids_root
)

# %%
raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)
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

# %%
# Add GSR and temperature channels
if 'GSR' not in raw.ch_names and 'Temperature' not in raw.ch_names:
    gsr_data = np.array([2.1e-6] * len(raw.times))
    temperature_data = np.array([36.5] * len(raw.times))

    gsr_and_temp_data = np.concatenate([
        np.atleast_2d(gsr_data),
        np.atleast_2d(temperature_data),
    ])
    gsr_and_temp_info = mne.create_info(
        ch_names=["GSR", "Temperature"],
        sfreq=raw.info["sfreq"],
        ch_types=["gsr", "temperature"],
    )
    gsr_and_temp_info["line_freq"] = raw.info["line_freq"]
    gsr_and_temp_info["subject_info"] = raw.info["subject_info"]
    with gsr_and_temp_info._unlock():
        gsr_and_temp_info["lowpass"] = raw.info["lowpass"]
        gsr_and_temp_info["highpass"] = raw.info["highpass"]
    gsr_and_temp_raw = mne.io.RawArray(
        data=gsr_and_temp_data,
        info=gsr_and_temp_info,
        first_samp=raw.first_samp,
    )
    raw.add_channels([gsr_and_temp_raw])
    del gsr_and_temp_raw, gsr_and_temp_data, gsr_and_temp_info

# %%
raw.set_annotations(None)
events = np.array([
    [0, 0, 1],
    [1000, 0, 2]
])
event_id = {
    "start_experiment": 1,
    "show_stimulus": 2
}

# %%
write_raw_bids(
    raw, bids_path, events=events, event_id=event_id, overwrite=True,
    allow_preload=True, format="BrainVision",
)
mne_bids.mark_channels(
    bids_path=bids_path,
    ch_names=['C3', 'C4', 'PO10', 'GSR', 'Temperature'],
    status=['good', 'good', 'bad', 'good', 'good'],
    descriptions=['resected', 'resected', 'continuously flat',
                  'left index finger', 'left ear']
)

# %%
dataset_description_json_path = tiny_bids_root / "dataset_description.json"
ds_json = json.loads(
    dataset_description_json_path.read_text(encoding="utf-8")
)

ds_json["Name"] = "tiny_bids"
ds_json["Authors"] = ["MNE-BIDS Developers", "And Friends"]

with open(dataset_description_json_path, "w", encoding="utf-8") as fout:
    json.dump(ds_json, fout, indent=4)
    fout.write("\n")
