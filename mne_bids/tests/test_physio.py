"""Test physiological data I/O."""

import json

import numpy as np
import pytest

# from mne_bids.physio.generic import _read_json
from mne_bids import get_bids_path_from_fname, read_raw_bids
from mne_bids.physio.generic import _read_json
from mne_bids.tsv_handler import _from_tsv, _to_tsv
from mne_bids.utils import _write_json

# Our toy dataset doesnt create a participants.tsv
pytestmark = pytest.mark.filterwarnings("ignore:participants.tsv:RuntimeWarning")


@pytest.fixture(scope="module")
def physio_dataset(tmp_path_factory, _bids_validate):
    """On-disk Toy BIDS dataset with Physio data."""
    root = tmp_path_factory.mktemp("bids")
    datatype = "beh"
    sub = "sub-01"
    task = "task-nback"

    fpath = root / sub / datatype / f"{sub}_{task}_physio.tsv.gz"
    fpath.parent.mkdir(parents=True)

    fpath_json = fpath.with_suffix("").with_suffix(".json")

    events_fname = f"{fpath.with_suffix('').with_suffix('').stem}events.tsv.gz"
    fpath_events = fpath.parent / events_fname
    fpath_events_json = fpath_events.with_suffix("").with_suffix(".json")

    description = {"Name": "toy", "BIDSVersion": "1.11"}

    data = dict(
        timestamp=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
        cardiac=[10.1, 10.0, 9.5, 9.2, 9.0, 10.2, 10.3, 10.1],
    )

    metadata = {
        "PhysioType": "generic",
        "SamplingFrequency": 100.0,
        "StartTime": 0.0,
        "Columns": ["timestamp", "cardiac"],
        "cardiac": {"Description": "continuous pulse measurement", "Units": "V"},
        "timestamp": {
            "Description": "Sampling time in seconds",
            "Units": "s",
            "Origin": "System startup",
        },
    }

    physioevents_data = {"onset": [0.01, 0.03, 0.05], "value": ["foo", "bar", "baz"]}

    physioevents_metadata = {
        "Columns": ["onset", "trial_type"],
        "Description": "Messages logged by the measurement device",
        "OnsetSource": "timestamp",
    }

    encoding = "utf-8"
    (root / "dataset_description.json").write_text(
        json.dumps(description), encoding=encoding
    )
    _to_tsv(data, fpath, compress=True)
    _write_json(fpath_json, metadata)
    _to_tsv(physioevents_data, fpath_events, compress=True)
    _write_json(fpath_events_json, physioevents_metadata)

    _bids_validate(root)
    yield root


def test_read_physio(physio_dataset):
    """Read a <match>_physio.tsv.gz file."""
    bpath = get_bids_path_from_fname(
        physio_dataset / "sub-01" / "beh" / "sub-01_task-nback_physio.tsv.gz"
    )
    assert bpath.fpath.exists()
    json_fpath = bpath.find_matching_sidecar("physio", ".json")
    metadata = _read_json(json_fpath)
    raw = read_raw_bids(bpath)
    assert raw.ch_names == metadata["Columns"] == ["timestamp", "cardiac"]
    assert raw.get_channel_types() == ["misc", "ecg"]


def test_read_physioevents(physio_dataset):
    """Read a <match>_physioevents.tsv.gz file."""
    bpath = get_bids_path_from_fname(
        physio_dataset / "sub-01" / "beh" / "sub-01_task-nback_physio.tsv.gz"
    )
    ev_fpath = bpath.find_matching_sidecar(suffix="physioevents", extension=".tsv.gz")
    ev_data = _from_tsv(ev_fpath)
    raw = read_raw_bids(bpath)
    np.testing.assert_array_equal(ev_data["trial_type"], raw.annotations.description)
    np.testing.assert_allclose(
        list(map(float, ev_data["onset"])), raw.annotations.onset
    )
    np.testing.assert_allclose(raw.annotations.duration, 0)
