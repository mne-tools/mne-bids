"""Tests for I/O of BIDS-compliant eyetracking data (BEP 020)."""

import json

import mne
import numpy as np
import pytest
from mne.datasets import testing
from mne.io import RawArray, read_raw_egi, read_raw_eyelink

from mne_bids import BIDSPath, read_raw_bids, write_raw_bids
from mne_bids.physio import _get_eyetrack_annotation_inds, write_eyetrack_calibration


@pytest.fixture(scope="module")
def eyelink_fpath():
    """Get path to MNE testing Eyelink file."""
    return testing.data_path(download=False) / "eyetrack" / "test_eyelink.asc"


@pytest.fixture(scope="module")
def raw_eye_and_cals(eyelink_fpath):
    """Get re-usable raw eyetracking object and calibrations."""
    raw = read_raw_eyelink(eyelink_fpath)
    cals = mne.preprocessing.eyetracking.read_eyelink_calibration(eyelink_fpath)
    return raw, cals


@pytest.fixture
def eyetrack_bpath(tmp_path):
    """Get fresh base BIDSPath for eyetracking-only datasets."""
    return BIDSPath(
        root=tmp_path / "bids",
        datatype="beh",
        subject="01",
        session="01",
        task="foo",
        run="01",
        recording="eye1",
        suffix="physio",
        extension=".tsv.gz",
    )


def _assert_roundtrip_raw(raw_in, raw):
    """Assert basic roundtrip equivalence for raw objects."""
    assert raw_in.get_channel_types() == raw.get_channel_types()
    assert raw_in.info["sfreq"] == raw.info["sfreq"]
    for ch_orig, ch_in in zip(raw.info["chs"], raw_in.info["chs"]):
        np.testing.assert_array_equal(ch_orig["loc"], ch_in["loc"])


def test_get_eyetrack_annotation_inds():
    """Test selecting annotations tied to eyetracking channels."""
    info = mne.create_info(
        ch_names=["xpos_left", "pupil_left", "eeg1"],
        sfreq=100,
        ch_types=["eyegaze", "pupil", "eeg"],
    )
    raw = RawArray(np.zeros((3, 400)), info)
    raw.set_annotations(
        mne.Annotations(
            onset=[0.0, 1.0, 2.0, 3.0],
            duration=[0.1, 0.1, 0.1, 0.1],
            description=["fixation", "stim", "blink", "misc"],
            ch_names=[("xpos_left",), ("eeg1",), ("pupil_left",), ()],
        )
    )

    got = _get_eyetrack_annotation_inds(raw)
    want = np.array([0, 2])
    np.testing.assert_array_equal(got, want)


def test_write_eyetracking_calibration(tmp_path, eyetrack_bpath):
    """Calibration writer should add calibration keys to the right eye files."""
    bpath = eyetrack_bpath.copy().update(extension=".json")
    eye1_json = bpath.fpath
    eye2_json = bpath.copy().update(recording="eye2").fpath

    eye1_json.parent.mkdir(parents=True, exist_ok=True)
    eye1_json.write_text(json.dumps({"PhysioType": "eyetrack", "RecordedEye": "left"}))
    eye2_json.write_text(json.dumps({"PhysioType": "eyetrack", "RecordedEye": "right"}))

    calibrations = [
        {
            "eye": "left",
            "avg_error": 0.1,
            "max_error": 0.2,
            "model": "HV3",
            "positions": np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]),
            "screen_distance": 0.6,
        },
        {
            "eye": "right",
            "avg_error": 0.3,
            "max_error": 0.5,
            "model": "HV3",
            "positions": np.array([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]]),
            "screen_distance": 0.6,
        },
    ]
    updated = write_eyetrack_calibration(eyetrack_bpath, calibrations)

    assert set(updated) == {eye1_json, eye2_json}
    eye1 = json.loads(eye1_json.read_text())
    eye2 = json.loads(eye2_json.read_text())

    assert eye1["CalibrationCount"] == 1
    assert eye1["AverageCalibrationError"] == 0.1
    assert eye1["MaximalCalibrationError"] == 0.2
    assert eye1["CalibrationType"] == "HV3"
    assert eye1["CalibrationDistance"] == 0.6

    assert eye2["CalibrationCount"] == 1
    assert eye2["AverageCalibrationError"] == 0.3
    assert eye2["MaximalCalibrationError"] == 0.5

    # If no BIDS dataset on disk, should raise
    dupe_bpath = eyetrack_bpath.update(root=tmp_path)
    with pytest.raises(FileNotFoundError, match="Eyetracking sidecar not found"):
        write_eyetrack_calibration(dupe_bpath, calibrations)


@testing.requires_testing_data
def test_eyetracking_io_roundtrip(_bids_validate, raw_eye_and_cals, eyetrack_bpath):
    """Test eyetracking-only BIDS write/read roundtrip."""
    raw, _ = raw_eye_and_cals

    write_raw_bids(
        raw,
        eyetrack_bpath,
        allow_preload=True,
        format="auto",
        overwrite=False,
    )
    raw_in = read_raw_bids(eyetrack_bpath)

    want_names = [
        "x_coordinate_left",
        "y_coordinate_left",
        "pupil_size_left",
        "x_coordinate_right",
        "y_coordinate_right",
        "pupil_size_right",
    ]
    assert raw_in.ch_names == want_names
    _assert_roundtrip_raw(raw_in, raw)
    assert len(raw_in.ch_names) == len(set(raw_in.ch_names))
    assert "x_coordinate_left" in raw_in.ch_names
    assert "y_coordinate_left" in raw_in.ch_names
    assert "pupil_size_left" in raw_in.ch_names

    eye1_json = json.loads(
        eyetrack_bpath.copy().update(extension=".json").fpath.read_text()
    )
    assert "x_coordinate" in eye1_json["Columns"]
    assert "y_coordinate" in eye1_json["Columns"]

    # The Physioevents TSV should be headerless
    phys_ev_fpath = (
        eyetrack_bpath.copy().update(suffix="physioevents", check=False).fpath
    )

    phys_ev = np.loadtxt(phys_ev_fpath, encoding="utf-8-sig", dtype=str, delimiter="\t")
    first_line = phys_ev[0]
    assert "onset" not in first_line
    # Only ocular events should be in physioevents
    trial_types = set(phys_ev[:, 2])
    assert trial_types == {"blink", "fixation", "saccade"}

    # Eyetracking only Data should not have a *_channels.tsv file
    with pytest.raises(RuntimeError, match="Did not find any"):
        eyetrack_bpath.find_matching_sidecar(suffix="channels", extension=".tsv")


@testing.requires_testing_data
def test_write_raw_bids_does_not_mutate_raw(
    _bids_validate, raw_eye_and_cals, eyetrack_bpath
):
    """write_raw_bids should not mutate source raw object.

    Writing Eyetracking BIDS involves copying the Raw object, then deleting channels and
    Annotations. So this is a safeguard to make sure that our code does not mutate the
    input Raw object.
    """
    raw, _ = raw_eye_and_cals
    ch_names_before = raw.ch_names.copy()
    desc_before = raw.annotations.description.copy()

    write_raw_bids(
        raw,
        eyetrack_bpath,
        allow_preload=True,
        format="auto",
        overwrite=False,
    )

    assert raw.ch_names == ch_names_before
    np.testing.assert_array_equal(raw.annotations.description, desc_before)


@testing.requires_testing_data
@pytest.mark.filterwarnings("ignore:Converting data:RuntimeWarning")
@pytest.mark.filterwarnings(
    "ignore:Encountered unsupported non-voltage units:UserWarning"
)
def test_eeg_eyetracking_io_roundtrip(_bids_validate, tmp_path, eyetrack_bpath):
    """Test simultaneous EEG+eyetracking write and readback."""
    eyetrack_fpath = testing.data_path(download=False) / "eyetrack" / "test_eyelink.asc"
    egi_fpath = testing.data_path(download=False) / "EGI" / "test_egi.mff"
    raw_eye = read_raw_eyelink(eyetrack_fpath)
    raw_egi = read_raw_egi(egi_fpath).load_data()
    cals = mne.preprocessing.eyetracking.read_eyelink_calibration(eyetrack_fpath)

    # Hack together the raws
    raw_eye.crop(tmax=raw_egi.times[-1]).resample(100, method="polyphase")
    raw_egi.resample(100)
    raw_eye.set_meas_date(None)
    raw_egi.set_meas_date(None)

    raw = raw_egi.copy().add_channels([raw_eye], force_update_info=True)
    raw.set_annotations(raw.annotations + raw_eye.annotations)

    eyetrack_bpath.update(datatype="eeg")
    eeg_bpath = eyetrack_bpath.copy().update(
        recording=None, datatype="eeg", suffix="eeg", extension=".vhdr"
    )

    write_raw_bids(raw, eeg_bpath, allow_preload=True, format="BrainVision")
    write_eyetrack_calibration(eyetrack_bpath, cals)

    assert eyetrack_bpath.fpath.parent.name == "eeg"

    # e.g. in EEG-eytracking, the recording-eye{1,2} entity is only for the physio files
    for suffix, ext in zip(("channels", "eeg", "events"), (".tsv", ".json", ".tsv")):
        sidecar_fname = eeg_bpath.find_matching_sidecar(suffix=suffix, extension=ext)
        assert sidecar_fname.exists()
        assert "recording-eye" not in str(sidecar_fname.name)

    eye1_json = json.loads(
        eyetrack_bpath.fpath.with_suffix("").with_suffix(".json").read_text()
    )
    assert eye1_json["RecordedEye"] == "left"

    raw_eye_in = read_raw_bids(eyetrack_bpath)
    _assert_roundtrip_raw(raw_eye_in, raw_eye)
    assert len(raw_eye_in.ch_names) == len(set(raw_eye_in.ch_names))
