"""Tests for read_epochs_bids."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

import json
import shutil

import mne
import pytest
from mne.datasets import testing
from numpy.testing import assert_allclose

from mne_bids import BIDSPath, read_epochs_bids, read_raw_bids

data_path = testing.data_path(download=False)
eeglab_epochs_set = data_path / "EEGLAB" / "test_epochs.set"
eeglab_epochs_fdt = data_path / "EEGLAB" / "test_epochs.fdt"

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:At least one epoch has multiple events.*:RuntimeWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:Data file name in EEG.data .* is incorrect.*:RuntimeWarning"
    ),
]


def _make_epoched_eeglab_dataset(bids_root):
    (bids_root / "sub-01" / "eeg").mkdir(parents=True)
    bids_path = BIDSPath(
        subject="01",
        task="test",
        datatype="eeg",
        suffix="eeg",
        extension=".set",
        root=bids_root,
    )
    shutil.copy(eeglab_epochs_set, bids_path.fpath)
    shutil.copy(eeglab_epochs_fdt, bids_path.fpath.with_suffix(".fdt"))

    template = mne.io.read_epochs_eeglab(eeglab_epochs_set, verbose=False)
    sidecar = {
        "TaskName": "test",
        "PowerLineFrequency": 60,
        "SamplingFrequency": float(template.info["sfreq"]),
        "RecordingType": "epoched",
        "EEGChannelCount": len(template.ch_names),
    }
    bids_path.copy().update(extension=".json").fpath.write_text(json.dumps(sidecar))
    channels = bids_path.copy().update(suffix="channels", extension=".tsv")
    channels.fpath.write_text(
        "name\ttype\tunits\n" + "".join(f"{ch}\tEEG\tµV\n" for ch in template.ch_names),
        encoding="utf-8",
    )
    (bids_root / "dataset_description.json").write_text(
        json.dumps({"Name": "x", "BIDSVersion": "1.8.0"})
    )
    (bids_root / "participants.tsv").write_text("participant_id\nsub-01\n")
    return bids_path, template


@testing.requires_testing_data
def test_read_epochs_bids_eeglab(tmp_path):
    """read_epochs_bids loads an EEGLAB epoched dataset and applies sidecars."""
    bids_path, expected = _make_epoched_eeglab_dataset(tmp_path / "ds")

    epochs = read_epochs_bids(bids_path)

    assert isinstance(epochs, mne.BaseEpochs)
    assert epochs.ch_names == expected.ch_names
    assert epochs.info["line_freq"] == 60
    assert_allclose(epochs.get_data(copy=False), expected.get_data(copy=False))

    with pytest.raises(RuntimeError, match="read_epochs_bids"):
        read_raw_bids(bids_path)
