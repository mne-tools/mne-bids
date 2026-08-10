"""Tests for I/O of BIDS-compliant eyetracking data (BEP 020)."""

import json

import mne
import numpy as np
import pytest
from mne.datasets import testing
from mne.io import RawArray, read_raw_egi, read_raw_eyelink

import mne_bids
from mne_bids import BIDSPath, write_raw_bids
from mne_bids.physio import _get_eyetrack_annotation_inds, write_eyetrack_calibration


@pytest.fixture(scope="module")
def eyelink_fpath():
    """Get path to MNE testing Eyelink file."""
    return testing.data_path(download=False) / "eyetrack" / "test_eyelink.asc"


@pytest.fixture(scope="function")
def raw_eye_and_cals(eyelink_fpath):
    """Get re-usable raw eyetracking object and calibrations."""
    raw = read_raw_eyelink(eyelink_fpath)
    cals = mne.preprocessing.eyetracking.read_eyelink_calibration(eyelink_fpath)
    cals = _add_screen_metadata(cals)
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


def _add_screen_metadata(calibrations):
    """Add BIDS-required screen metadata to eyetracking calibrations."""
    calibrations = [cal.copy() for cal in calibrations]
    for cal in calibrations:
        cal["screen_distance"] = 0.9
        cal["screen_origin"] = ["top", "left"]
        cal["screen_resolution"] = [1920, 1080]
        cal["screen_size"] = [0.53, 0.3]
    return calibrations


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
            "screen_origin": ["top", "left"],
            "screen_resolution": [1920, 1080],
            "screen_size": [0.53, 0.3],
        },
        {
            "eye": "right",
            "avg_error": 0.3,
            "max_error": 0.5,
            "model": "HV3",
            "positions": np.array([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]]),
            "screen_distance": 0.6,
            "screen_origin": ["top", "left"],
            "screen_resolution": [1920, 1080],
            "screen_size": [0.53, 0.3],
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
def test_write_eyetracking_bino(_bids_validate, raw_eye_and_cals, eyetrack_bpath):
    """Test writing eyetracking-only data to BIDS."""
    raw, cals = raw_eye_and_cals

    write_raw_bids(
        raw,
        eyetrack_bpath,
        allow_preload=True,
        format="auto",
        eyetrack_calibration=cals,
        overwrite=False,
    )
    _bids_validate(eyetrack_bpath.root)


@testing.requires_testing_data
@pytest.mark.parametrize("eye", ["left", "right"], ids=["left", "right"])
def test_write_eyetracking_mono(_bids_validate, raw_eye_and_cals, eyetrack_bpath, eye):
    """Test writing monocular eyetracking data."""
    raw, cals = raw_eye_and_cals
    raw.drop_channels([f"xpos_{eye}", f"ypos_{eye}", f"pupil_{eye}"])
    cal = cals[0] if eye == "left" else cals[1]
    write_raw_bids(
        raw,
        eyetrack_bpath,
        allow_preload=True,
        format="auto",
        eyetrack_calibration=cal,
        overwrite=False,
    )
    _bids_validate(eyetrack_bpath.root)


@testing.requires_testing_data
def test_write_eyetrack_without_annotations(
    _bids_validate, raw_eye_and_cals, eyetrack_bpath
):
    """Ensure ET data without annotations can be written and is complaint."""
    raw, cals = raw_eye_and_cals
    eye_annot_inds = [
        ii
        for ii, names in enumerate(raw.annotations.ch_names)
        if (
            names == ("xpos_left", "ypos_left", "pupil_left")
            or names == ("xpos_right", "ypos_right", "pupil_right")
        )
    ]
    raw.annotations.delete(eye_annot_inds)

    with pytest.warns(match="No eyetracking annotations found."):
        write_raw_bids(
            raw,
            eyetrack_bpath,
            allow_preload=True,
            format="auto",
            eyetrack_calibration=cals,
            overwrite=False,
        )
    _bids_validate(eyetrack_bpath.root)


@testing.requires_testing_data
@pytest.mark.filterwarnings("ignore:Converting data:RuntimeWarning")
@pytest.mark.filterwarnings(
    "ignore:Encountered unsupported non-voltage units:UserWarning"
)
def test_write_eeg_eyetracking(_bids_validate, tmp_path, eyetrack_bpath):
    """Test writing simultaneous EEG+eyetracking data to BIDS."""
    eyetrack_fpath = testing.data_path(download=False) / "eyetrack" / "test_eyelink.asc"
    egi_fpath = testing.data_path(download=False) / "EGI" / "test_egi.mff"
    raw_eye = read_raw_eyelink(eyetrack_fpath)
    raw_egi = read_raw_egi(egi_fpath, events_as_annotations=False).load_data()
    cals = mne.preprocessing.eyetracking.read_eyelink_calibration(eyetrack_fpath)
    cals = _add_screen_metadata(cals)

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

    write_raw_bids(
        raw,
        eeg_bpath,
        allow_preload=True,
        format="BrainVision",
        eyetrack_calibration=cals,
    )
    _bids_validate(eeg_bpath.root)


@testing.requires_testing_data
def test_write_raises(raw_eye_and_cals, eyetrack_bpath):
    """Ensure that malformed raw objects hit our error messages when writing."""
    # 1. We need the loc array to be properly set for eyetracking channels.
    raw, cals = raw_eye_and_cals
    orig_coord = raw.info["chs"][0]["loc"][4].copy()
    raw.info["chs"][0]["loc"][4] = 4
    with pytest.raises(match="Eyegaze channels must set"):
        write_raw_bids(
            raw,
            eyetrack_bpath,
            allow_preload=True,
            format="auto",
            eyetrack_calibration=cals,
            overwrite=False,
        )
    raw.info["chs"][0]["loc"][4] = orig_coord

    # 2: BIDS can't handle e.g. two x-coordinate channels for the left eye..
    new_data = raw.get_data(picks="xpos_left").copy()
    new_info = mne.create_info(
        ch_names=["xpos_left_2"], sfreq=raw.info["sfreq"], ch_types=["eyegaze"]
    )
    new_channel = mne.io.RawArray(new_data, new_info)
    new_channel = mne.preprocessing.eyetracking.set_channel_types_eyetrack(
        new_channel, mapping={"xpos_left_2": ("eyegaze", "px", "left", "x")}
    )
    raw.add_channels([new_channel])
    with pytest.raises(match="this will result in duplicate BIDS names"):
        write_raw_bids(
            raw,
            eyetrack_bpath,
            allow_preload=True,
            format="auto",
            eyetrack_calibration=cals,
            overwrite=False,
        )
    raw.drop_channels("xpos_left_2")

    # 3 The user needs to specify the datatype
    eyetrack_bpath_bad = eyetrack_bpath.copy().update(datatype=None)
    with pytest.raises(match="datatype must be specified"):
        mne_bids.physio.eyetracking._write_eyetrack_tsvs(
            raw=raw,
            bids_path=eyetrack_bpath_bad,
            overwrite=False,
        )
    del eyetrack_bpath_bad

    # 4. Again, the loc array must be properly set for eyetrack channels
    orig_eye = raw.info["chs"][0]["loc"][3].copy()
    raw.info["chs"][0]["loc"][3] = 999.0
    with pytest.raises(match="must specify the eye"):
        mne_bids.physio.eyetracking._write_eyetrack_tsvs(
            raw=raw,
            bids_path=eyetrack_bpath,
            overwrite=False,
        )
    raw.info["chs"][0]["loc"][3] = orig_eye

    # 4. Calibraiton objects must contain 'left' or 'right' in their 'eye' key
    cal_bad = cals[0].copy()
    cal_bad["eye"] = "foo"
    with pytest.raises(match="'left' or 'right' in its 'eye' key."):
        write_raw_bids(
            raw,
            eyetrack_bpath,
            allow_preload=True,
            format="auto",
            eyetrack_calibration=cal_bad,
            overwrite=False,
        )
    del cal_bad
