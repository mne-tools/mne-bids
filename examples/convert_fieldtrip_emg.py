"""
===============================
Convert EMG data to BIDS format
===============================

In this example, we use MNE-BIDS to create a BIDS-compatible directory of EMG data.
The data comes from the FieldTrip download server, and was recorded through an EEG
amplifier in BrainVision format. Because of that, we'll need to explicitly change the
channel type from EEG to EMG; this may not be necessary for your data.

.. currentmodule:: mne_bids
"""  # noqa: D205 D400

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from warnings import filterwarnings

import mne
import pooch

import mne_bids

# %%
# There are sometimes a lot of warnings when writing BIDS datasets. Here we use
# :func:`warnings.filterwarnings` to ignore some warnings that are expected given what
# we (don't) know about this dataset.

known_warnings = {
    "The `doi` field in dataset_description should be": "mne_bids",  # our DOI is n/a
    "Converting data files to EDF format": "mne_bids",  # BIDS requires EDF/BDF for EMG
    "EDF format requires equal-length data blocks": "mne",  # zero-padding the output
    "No electrode location info found": "mne_bids",  # EMG sensor locations unknown
    r"Channels contain different (low|high)pass filters\.": "mne",
}
for msg, mod in known_warnings.items():
    filterwarnings(
        action="ignore",
        message=msg,
        category=RuntimeWarning,
        module=mod,
    )

# %%
# First we'll get the data. We'll use :mod:`pooch` here, but the :mod:`requests` module
# would work just as well (or you could manually download the data of course).

remote_folder = "https://download.fieldtriptoolbox.org/example/bids_emg/original/"
remote_files_and_hashes = {
    "nt05_EMG_SESS1_GABA_RAW.eeg": "d4c62cec8abb57c64304685884e0181dffa03ba9ec7730730d4f7e5d5254a345",  # noqa E501
    "nt05_EMG_SESS1_GABA_RAW.vhdr": "d1094625f9e80ff66c04dcc355e7afa929d496edcc8f206eab1c88fa05ebe7a3",  # noqa E501
    "nt05_EMG_SESS1_GABA_RAW.vmrk": "73ab8da32c5f8938004fdee4145133f61843f69137d90ddfcdf9d8fc0380c7cd",  # noqa E501
    "nt05_EMG_SESS1_RS_RAW.eeg": "016a3b1ad93b0c2c9d2c0aed67668127e98c76e63f7abe65c67839c2affcc2f0",  # noqa E501
    "nt05_EMG_SESS1_RS_RAW.vhdr": "73d1d893635d6bec91e07dc2714a420d9b8abdb211cd2ec1fa3dc977b66ca6aa",  # noqa E501
    "nt05_EMG_SESS1_RS_RAW.vmrk": "bdce8497ffd0ee65aed535c22afef988d41fdcba199f0b04cb5792acba10b22c",  # noqa E501
}
download_dir = Path("~/mne_data/MNE-fieldtrip_emg-data").expanduser()
download_dir.mkdir(exist_ok=True)
for fname, known_hash in remote_files_and_hashes.items():
    pooch.retrieve(
        remote_folder + fname, known_hash=known_hash, fname=fname, path=download_dir
    )

# %%
# Next we'll set up the directory where we'll store the BIDS-formatted dataset.

target_dir = download_dir / "bidsified"
target_dir.mkdir(exist_ok=True)
bids_path = mne_bids.BIDSPath(
    root=target_dir, datatype="emg", subject="nt05", task="rest"
)

# %%
# We don't know a lot about this data, so the dataset description file will have a lot
# of ``n/a`` fields.

mne_bids.make_dataset_description(
    path=bids_path.root,
    name="EMG example",
    dataset_type="raw",
    data_license="n/a",
    authors="n/a",
    funding="n/a",
    acknowledgements="n/a",
    references_and_links="n/a",
    doi="n/a",
)

# %%
# Next we prepare the metadata that is common to both acquisition runs,
# then prepare the info about what is *different* between them.

emg_metadata = dict(
    Manufacturer="BrainProducts",
    ManufacturersModelName="BrainAmp MR Plus",
    EMGChannelCount=13,
    EMGGround="unclear where the ground electrode was placed on the body",
    EMGPlacementScheme="Other",
    EMGPlacementSchemeDescription="electrode pairs were placed over various muscles",
    EMGReference="bipolar",
)
task_description = (
    "The subject was lying in the MRI scanner during a {} scan, "
    "while EMG was being recorded."
)
# distinguish the acquisitions
acqs = dict(
    GABA=dict(descr="GABA MRS", entity_value="GabaMrs"),
    RS=dict(descr="resting-state BOLD", entity_value="Bold"),
)

# %%
# Now it's time to write out the data:

for acq, acq_dict in acqs.items():
    # read the data
    bids_path.update(acquisition=acq_dict["entity_value"])
    print(bids_path)
    source_fpath = download_dir / f"nt05_EMG_SESS1_{acq}_RAW.vhdr"
    raw = mne.io.read_raw_brainvision(source_fpath)
    # convert channel types
    raw.set_channel_types({ch: "emg" for ch in raw.ch_names})
    # write to bids folder tree
    mne_bids.write_raw_bids(
        raw=raw, bids_path=bids_path, format="EDF", emg_placement="Other"
    )
    # update the sidecar with acquisition-specific info
    bp = bids_path.copy().update(suffix="emg", extension="json")
    print(bp)
    mne_bids.update_sidecar_json(
        bids_path=bp,
        entries=dict(
            TaskDescription=task_description.format(acq_dict["descr"]), **emg_metadata
        ),
    )

# %%
# Note that in this dataset, only the BOLD acquisition had event markers in the raw
# data file, so the GABA MRS run lacks ``*_events.tsv`` and ``*_events.json`` files:

mne_bids.print_dir_tree(bids_path.root)
