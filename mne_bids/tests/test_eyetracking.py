import numpy as np
import pytest
from mne.datasets import testing
from mne.io import read_raw_egi, read_raw_eyelink

from mne_bids import BIDSPath, read_raw_bids, write_raw_bids


@testing.requires_testing_data
def test_eyetracking_io(_bids_validate, tmp_path):
    eyetrack_fpath = testing.data_path(download=False) / "eyetrack" / "test_eyelink.asc"
    raw = read_raw_eyelink(eyetrack_fpath)
    root = tmp_path / "bids"

    bpath = BIDSPath(
        root=root,
        datatype="beh",
        subject="01",
        session="01",
        task="foo",
        run="01",
        recording="eye1",
        suffix="physio",
        extension=".tsv"
    )
    write_raw_bids(
        raw,
        bpath,
        allow_preload=True,
        format="auto",
        overwrite=False,
        )
    raw_in = read_raw_bids(
        bpath,
        eyetrack_ch_types={
            "xpos_left": "eyegaze",
            "ypos_left": "eyegaze",
            "pupil_left": "pupil",
            "xpos_right": "eyegaze",
            "ypos_right": "eyegaze",
            "pupil_right": "pupil",
        },
        )

    assert raw_in.ch_names == raw.ch_names
    assert raw_in.get_channel_types() == raw.get_channel_types()
    assert raw_in.info['sfreq'] == raw.info['sfreq']
    for ch_orig, ch_in in zip(raw.info["chs"], raw_in.info["chs"]):
        np.testing.assert_array_equal(ch_orig["loc"], ch_in["loc"])
    assert 1 == 0


@testing.requires_testing_data
@pytest.mark.filterwarnings("ignore:Converting data:RuntimeWarning")
@pytest.mark.filterwarnings(
    "ignore:Encountered unsupported non-voltage units:UserWarning"
    )
def test_eeg_eyetracking_io(_bids_validate, tmp_path):
    """Test writing eyetracking data into the EEG datatype sub-folder."""
    # Let's Hack together an EEG-Eyetracking dataset
    eyetrack_fpath = testing.data_path(download=False) / "eyetrack" / "test_eyelink.asc"
    egi_fpath = testing.data_path(download=False) / "EGI" / "test_egi.mff"
    raw_eye = read_raw_eyelink(eyetrack_fpath, )
    raw_egi = read_raw_egi(egi_fpath).load_data()

    # Make the two recordings the same length
    raw_eye.crop(tmax=raw_egi.times[-1]).resample(100, method="polyphase")
    raw_egi.resample(100)

    raw_eye.set_meas_date(None)
    raw_egi.set_meas_date(None) # (raw_eye.info["meas_date"])

    # Combine
    raw = raw_egi.copy().add_channels([raw_eye], force_update_info=True)
    raw.set_annotations(raw.annotations + raw_eye.annotations)

    # Set up BIDSPath
    root = tmp_path / "bids"

    bpath = BIDSPath(
        root=root,
        subject="01",
        session="01",
        run="01",
        task="bar",
        datatype="eeg",
        suffix="eeg",
    )

    write_raw_bids(
        raw,
        bpath,
        allow_preload=True,
        format="BrainVision",
        )

    raw_egi_in = read_raw_bids(bpath)
    bpath_et = bpath.copy().update(
        suffix="physio", extension=".tsv", recording="eye1",
        )
    raw_eye_in = read_raw_bids(
            bpath_et,
            eyetrack_ch_types={
                "xpos_left": "eyegaze",
                "ypos_left": "eyegaze",
                "pupil_left": "pupil",
                "xpos_right": "eyegaze",
                "ypos_right": "eyegaze",
                "pupil_right": "pupil",
        },
     )
    assert raw_eye_in.ch_names == raw_eye.ch_names
    assert raw_eye_in.get_channel_types() == raw_eye.get_channel_types()
    assert raw_eye_in.info['sfreq'] == raw_eye.info['sfreq']
    for ch_orig, ch_in in zip(raw_eye_in.info["chs"], raw_eye.info["chs"]):
        np.testing.assert_array_equal(ch_orig["loc"], ch_in["loc"])
    assert 1 == 0
