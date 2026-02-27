"""Testing utilities for file io."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import json
import logging
import multiprocessing as mp
import os
import os.path as op
import re
import shutil as sh
from collections import OrderedDict
from contextlib import nullcontext
from datetime import UTC, date, datetime
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest
from mne.datasets import testing
from mne.io.constants import FIFF
from mne.utils import assert_dig_allclose, check_version, object_diff
from numpy.testing import assert_almost_equal

import mne_bids.write
from mne_bids import BIDSPath
from mne_bids.config import (
    BIDS_SHARED_COORDINATE_FRAMES,
    BIDS_TO_MNE_FRAMES,
    MNE_STR_TO_FRAME,
)
from mne_bids.path import _find_matching_sidecar
from mne_bids.read import (
    _handle_channels_reading,
    _handle_events_reading,
    _handle_scans_reading,
    _read_raw,
    events_file_to_annotation_kwargs,
    get_head_mri_trans,
    read_raw_bids,
)
from mne_bids.sidecar_updates import _update_sidecar
from mne_bids.tsv_handler import _from_tsv, _to_tsv
from mne_bids.utils import _write_json
from mne_bids.write import get_anat_landmarks, write_anat, write_raw_bids

subject_id = "01"
session_id = "01"
run = "01"
acq = "01"
task = "testing"

sample_data_event_id = {
    "Auditory/Left": 1,
    "Auditory/Right": 2,
    "Visual/Left": 3,
    "Visual/Right": 4,
    "Smiley": 5,
    "Button": 32,
}

_bids_path = BIDSPath(
    subject=subject_id, session=session_id, run=run, acquisition=acq, task=task
)

_bids_path_minimal = BIDSPath(subject=subject_id, task=task)

# Get the MNE testing sample data - USA
data_path = testing.data_path(download=False)
raw_fname = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif"

# Data with cHPI info
raw_fname_chpi = op.join(data_path, "SSS", "test_move_anon_raw.fif")

# Tiny BIDS testing dataset
mne_bids_root = Path(mne_bids.__file__).parents[1]
tiny_bids_root = mne_bids_root / "mne_bids" / "tests" / "data" / "tiny_bids"

warning_str = dict(
    channel_unit_changed="ignore:The unit for chann*.:RuntimeWarning:mne",
    meas_date_set_to_none="ignore:.*'meas_date' set to None:RuntimeWarning:mne",
    nasion_not_found="ignore:.*nasion not found:RuntimeWarning:mne",
    maxshield="ignore:.*Internal Active Shielding:RuntimeWarning:mne",
    synthetic_fiducials="ignore:No fiducial points found:RuntimeWarning:mne_bids",
)


def _wrap_read_raw(read_raw):
    def fn(fname, *args, **kwargs):
        raw = read_raw(fname, *args, **kwargs)
        raw.info["line_freq"] = 60
        return raw

    return fn


_read_raw_fif = _wrap_read_raw(mne.io.read_raw_fif)
_read_raw_ctf = _wrap_read_raw(mne.io.read_raw_ctf)
_read_raw_edf = _wrap_read_raw(mne.io.read_raw_edf)


def _make_parallel_raw(subject, *, seed=None):
    """Generate a lightweight Raw instance for parallel-reading tests."""
    rng_seed = seed if seed is not None else sum(ord(ch) for ch in subject)
    rng = np.random.default_rng(rng_seed)
    info = mne.create_info(["MEG0113"], 100, ch_types="mag")
    data = rng.standard_normal((1, 100)) * 1e-12
    raw = mne.io.RawArray(data, info)
    raw.set_meas_date(datetime(2020, 1, 1, tzinfo=UTC))
    raw.info["line_freq"] = 60
    raw.info["subject_info"] = {
        "his_id": subject,
        "sex": 1,
        "hand": 2,
        "birthday": date(1990, 1, 1),
    }
    return raw


def _write_parallel_dataset(root, *, subject, run):
    """Write a minimal dataset using write_raw_bids."""
    root = Path(root)
    raw = _make_parallel_raw(subject)
    bids_path = BIDSPath(
        subject=subject, task="rest", run=run, datatype="meg", root=root
    )
    write_raw_bids(raw, bids_path, allow_preload=True, format="FIF", verbose=False)


def _parallel_read_participants(root, expected_ids):
    """Read participants.tsv in a multiprocessing worker."""
    participants_path = Path(root) / "participants.tsv"
    participants = _from_tsv(participants_path)
    assert set(participants["participant_id"]) == set(expected_ids)


def _parallel_read_scans(root, expected_filenames):
    """Read scans.tsv in a multiprocessing worker."""
    scans_path = BIDSPath(subject="01", root=root, suffix="scans", extension=".tsv")
    scans = _from_tsv(scans_path.fpath)
    filenames = {str(filename) for filename in scans["filename"]}
    assert filenames == set(expected_filenames)


def test_read_raw():
    """Test the raw reading."""
    # Use a file ending that does not exist
    f = "file.bogus"
    with pytest.raises(ValueError, match="file name extension must be one of"):
        _read_raw(f)


def test_not_implemented(tmp_path):
    """Test the not yet implemented data formats raise an adequate error."""
    for not_implemented_ext in [".nwb"]:
        raw_fname = tmp_path / f"test{not_implemented_ext}"
        with open(raw_fname, "w", encoding="utf-8"):
            pass
        with pytest.raises(
            ValueError, match=("there is no IO support for this file format yet")
        ):
            _read_raw(raw_fname)


def test_mefd_requires_supported_mne(tmp_path, monkeypatch):
    """Test that reading .mefd requires a registered reader implementation."""
    import mne_bids.read as read_module

    mefd_path = tmp_path / "test.mefd"
    mefd_path.mkdir()
    monkeypatch.delitem(read_module.reader, ".mefd", raising=False)

    with pytest.raises(ValueError, match="MEF3 support requires MNE-Python >= 1.12"):
        _read_raw(mefd_path)


def test_mefd_read_uses_reader_registry(tmp_path, monkeypatch):
    """Test that reading .mefd uses the registered reader from config."""
    import mne_bids.read as read_module

    mefd_path = tmp_path / "test.mefd"
    mefd_path.mkdir()
    sentinel = object()

    def _fake_mefd_reader(path, verbose=None, **kwargs):
        assert path == mefd_path
        assert isinstance(path, Path)
        assert verbose is None
        assert kwargs == {"preload": False}
        return sentinel

    monkeypatch.setitem(read_module.reader, ".mefd", _fake_mefd_reader)

    assert _read_raw(mefd_path, preload=False) is sentinel


def test_read_correct_inputs():
    """Test that inputs of read functions are correct."""
    bids_path = "sub-01_ses-01_meg.fif"
    with pytest.raises(RuntimeError, match='"bids_path" must be a BIDSPath object'):
        read_raw_bids(bids_path)

    with pytest.raises(RuntimeError, match='"bids_path" must be a BIDSPath object'):
        get_head_mri_trans(bids_path)

    with pytest.raises(RuntimeError, match='"bids_path" must contain `root`'):
        bids_path = BIDSPath(root=bids_path)
        read_raw_bids(bids_path)


@pytest.mark.filterwarnings(
    "ignore:No events found or provided:RuntimeWarning",
    "ignore:Found no extension for raw file.*:RuntimeWarning",
)
def test_parallel_participants_multiprocess(tmp_path):
    """Ensure parallel reads keep all participants entries visible."""
    bids_root = tmp_path / "parallel_multiprocess"
    subjects = [f"{i:02d}" for i in range(1, 50)]

    for subject in subjects:
        _write_parallel_dataset(str(bids_root), subject=subject, run="01")

    expected_ids = [f"sub-{subject}" for subject in subjects]
    processes = []
    for _ in range(len(subjects) // 10):  # spawn a few processes
        proc = mp.Process(
            target=_parallel_read_participants, args=(str(bids_root), expected_ids)
        )
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()
        assert proc.exitcode == 0

    participants_path = bids_root / "participants.tsv"
    assert participants_path.exists()
    participants = _from_tsv(participants_path)
    assert set(participants["participant_id"]) == set(expected_ids)
    sh.rmtree(bids_root, ignore_errors=True)


@pytest.mark.filterwarnings(
    "ignore:No events found or provided:RuntimeWarning",
    "ignore:Found no extension for raw file.*:RuntimeWarning",
)
def test_parallel_scans_multiprocessing(tmp_path):
    """Ensure multiprocessing reads see all runs in scans.tsv."""
    bids_root = tmp_path / "parallel_multiprocessing"
    runs = [f"{i:02d}" for i in range(1, 50)]

    for run in runs:
        _write_parallel_dataset(str(bids_root), subject="01", run=run)

    expected = {f"meg/sub-01_task-rest_run-{run}_meg.fif" for run in runs}
    processes = []
    for _ in range(4):
        proc = mp.Process(target=_parallel_read_scans, args=(str(bids_root), expected))
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()
        assert proc.exitcode == 0

    scans_path = BIDSPath(
        subject="01", root=bids_root, suffix="scans", extension=".tsv"
    )
    assert scans_path.fpath.exists()
    scans = _from_tsv(scans_path.fpath)
    filenames = {str(filename) for filename in scans["filename"]}
    assert filenames == expected
    sh.rmtree(bids_root, ignore_errors=True)


@pytest.mark.filterwarnings(warning_str["channel_unit_changed"])
@testing.requires_testing_data
def test_read_participants_data(tmp_path):
    """Test reading information from a BIDS sidecar.json file."""
    bids_path = _bids_path.copy().update(root=tmp_path, datatype="meg")
    raw = _read_raw_fif(raw_fname, verbose=False)

    # if subject info was set, we don't roundtrip birthday
    # due to possible anonymization in mne-bids
    subject_info = {"hand": 1, "sex": 2, "weight": 70.5, "height": 180.5}
    raw.info["subject_info"] = subject_info
    write_raw_bids(raw, bids_path, overwrite=True, verbose=False)
    raw = read_raw_bids(bids_path=bids_path)
    assert raw.info["subject_info"]["hand"] == 1
    assert raw.info["subject_info"]["weight"] == 70.5
    assert raw.info["subject_info"]["height"] == 180.5
    assert raw.info["subject_info"].get("birthday", None) is None
    assert raw.info["subject_info"]["his_id"] == f"sub-{bids_path.subject}"
    assert "participant_id" not in raw.info["subject_info"]

    # if modifying participants tsv, then read_raw_bids reflects that
    participants_tsv_fpath = tmp_path / "participants.tsv"
    participants_tsv = _from_tsv(participants_tsv_fpath)
    participants_tsv["hand"][0] = "n/a"
    participants_tsv["outcome"] = ["good"]  # e.g. clinical tutorial from MNE-Python
    _to_tsv(participants_tsv, participants_tsv_fpath)
    with pytest.warns(RuntimeWarning, match="Unable to map"):
        raw = read_raw_bids(bids_path=bids_path)
    assert raw.info["subject_info"]["hand"] == 0
    assert raw.info["subject_info"]["sex"] == 2
    assert raw.info["subject_info"]["weight"] == 70.5
    assert raw.info["subject_info"]["height"] == 180.5
    assert raw.info["subject_info"].get("birthday", None) is None

    # make sure things are read even if the entries don't make sense
    participants_tsv = _from_tsv(participants_tsv_fpath)
    participants_tsv["hand"][0] = "righty"
    participants_tsv["sex"][0] = "malesy"
    # 'n/a' values should get omitted
    participants_tsv["weight"] = ["n/a"]
    participants_tsv["height"] = ["tall"]
    del participants_tsv["outcome"]

    _to_tsv(participants_tsv, participants_tsv_fpath)
    with pytest.warns(RuntimeWarning, match="Unable to map"):
        raw = read_raw_bids(bids_path=bids_path)

    assert "hand" not in raw.info["subject_info"]
    assert "sex" not in raw.info["subject_info"]
    assert "weight" not in raw.info["subject_info"]
    assert "height" not in raw.info["subject_info"]

    # test reading if participants.tsv is missing
    raw = _read_raw_fif(raw_fname, verbose=False)
    write_raw_bids(raw, bids_path, overwrite=True, verbose=False)
    participants_tsv_fpath.unlink()
    with pytest.warns(RuntimeWarning, match="participants.tsv file not found"):
        raw = read_raw_bids(bids_path=bids_path)

    assert raw.info["subject_info"] == dict()


@pytest.mark.parametrize(
    ("hand_bids", "hand_mne", "sex_bids", "sex_mne"),
    [
        ("Right", 1, "Female", 2),
        ("RIGHT", 1, "FEMALE", 2),
        ("R", 1, "F", 2),
        ("left", 2, "male", 1),
        ("l", 2, "m", 1),
    ],
)
@pytest.mark.filterwarnings(warning_str["channel_unit_changed"])
@testing.requires_testing_data
def test_read_participants_handedness_and_sex_mapping(
    hand_bids, hand_mne, sex_bids, sex_mne, tmp_path
):
    """Test we're correctly mapping handedness and sex between BIDS and MNE."""
    bids_path = _bids_path.copy().update(root=tmp_path, datatype="meg")
    participants_tsv_fpath = tmp_path / "participants.tsv"
    raw = _read_raw_fif(raw_fname, verbose=False)

    # Avoid that we end up with subject information stored in the raw data.
    raw.info["subject_info"] = {}
    write_raw_bids(raw, bids_path, overwrite=True, verbose=False)

    participants_tsv = _from_tsv(participants_tsv_fpath)
    participants_tsv["hand"][0] = hand_bids
    participants_tsv["sex"][0] = sex_bids
    _to_tsv(participants_tsv, participants_tsv_fpath)

    raw = read_raw_bids(bids_path=bids_path)
    assert raw.info["subject_info"]["hand"] is hand_mne
    assert raw.info["subject_info"]["sex"] is sex_mne


@pytest.mark.filterwarnings(warning_str["channel_unit_changed"])
@testing.requires_testing_data
def test_get_head_mri_trans(tmp_path):
    """Test getting a trans object from BIDS data."""
    nib = pytest.importorskip("nibabel")

    events_fname = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw-eve.fif"
    subjects_dir = op.join(data_path, "subjects")

    # Drop unknown events.
    events = mne.read_events(events_fname)
    events = events[events[:, 2] != 0]

    # Write it to BIDS
    raw = _read_raw_fif(raw_fname)
    bids_path = _bids_path.copy().update(root=tmp_path, datatype="meg", suffix="meg")
    write_raw_bids(
        raw, bids_path, events=events, event_id=sample_data_event_id, overwrite=False
    )

    # We cannot recover trans if no MRI has yet been written
    with pytest.raises(FileNotFoundError, match="Did not find"):
        estimated_trans = get_head_mri_trans(
            bids_path=bids_path, fs_subject="sample", fs_subjects_dir=subjects_dir
        )

    # Write some MRI data and supply a `trans` so that a sidecar gets written
    trans = mne.read_trans(str(raw_fname).replace("_raw.fif", "-trans.fif"))

    # Get the T1 weighted MRI data file ... test write_anat with a nibabel
    # image instead of a file path
    t1w_mgh = op.join(data_path, "subjects", "sample", "mri", "T1.mgz")
    t1w_mgh = nib.load(t1w_mgh)

    landmarks = get_anat_landmarks(
        t1w_mgh, raw.info, trans, fs_subject="sample", fs_subjects_dir=subjects_dir
    )
    t1w_bids_path = bids_path.copy().update(datatype="anat", suffix="T1w")
    t1w_bids_path = write_anat(
        t1w_mgh, bids_path=t1w_bids_path, landmarks=landmarks, verbose=True
    )
    anat_dir = t1w_bids_path.directory

    # Try to get trans back through fitting points
    estimated_trans = get_head_mri_trans(
        bids_path=bids_path, fs_subject="sample", fs_subjects_dir=subjects_dir
    )

    assert trans["from"] == estimated_trans["from"]
    assert trans["to"] == estimated_trans["to"]
    assert_almost_equal(trans["trans"], estimated_trans["trans"])

    # provoke an error by introducing NaNs into MEG coords
    raw.info["dig"][0]["r"] = np.full(3, np.nan)
    sh.rmtree(anat_dir)
    bad_landmarks = get_anat_landmarks(
        t1w_mgh, raw.info, trans, "sample", op.join(data_path, "subjects")
    )
    write_anat(t1w_mgh, bids_path=t1w_bids_path, landmarks=bad_landmarks)
    with pytest.raises(RuntimeError, match="AnatomicalLandmarkCoordinates"):
        estimated_trans = get_head_mri_trans(
            bids_path=t1w_bids_path, fs_subject="sample", fs_subjects_dir=subjects_dir
        )

    # test raw with no fiducials to provoke error
    t1w_bids_path = write_anat(  # put back
        t1w_mgh, bids_path=t1w_bids_path, landmarks=landmarks, overwrite=True
    )
    montage = raw.get_montage()
    montage.remove_fiducials()
    raw_test = raw.copy()
    raw_test.set_montage(montage)
    raw_test.save(bids_path.fpath, overwrite=True)

    ctx = nullcontext()
    with ctx:
        get_head_mri_trans(
            bids_path=bids_path, fs_subject="sample", fs_subjects_dir=subjects_dir
        )

    # test we are permissive for different casings of landmark names in the
    # sidecar, and also accept "nasion" instead of just "NAS"
    raw = _read_raw_fif(raw_fname)
    write_raw_bids(
        raw, bids_path, events=events, event_id=sample_data_event_id, overwrite=True
    )  # overwrite with new acq
    t1w_bids_path = write_anat(
        t1w_mgh, bids_path=t1w_bids_path, landmarks=landmarks, overwrite=True
    )

    t1w_json_fpath = t1w_bids_path.copy().update(extension=".json").fpath
    with t1w_json_fpath.open("r", encoding="utf-8") as f:
        t1w_json = json.load(f)

    coords = t1w_json["AnatomicalLandmarkCoordinates"]
    coords["lpa"] = coords["LPA"]
    coords["Rpa"] = coords["RPA"]
    coords["Nasion"] = coords["NAS"]
    del coords["LPA"], coords["RPA"], coords["NAS"]

    _write_json(t1w_json_fpath, t1w_json, overwrite=True)

    estimated_trans = get_head_mri_trans(
        bids_path=bids_path, fs_subject="sample", fs_subjects_dir=subjects_dir
    )
    assert_almost_equal(trans["trans"], estimated_trans["trans"])

    # Test t1_bids_path parameter
    #
    # Case 1: different BIDS roots
    meg_bids_path = _bids_path.copy().update(
        root=tmp_path / "meg_root", datatype="meg", suffix="meg"
    )
    t1_bids_path = _bids_path.copy().update(
        root=tmp_path / "mri_root", task=None, run=None
    )
    raw = _read_raw_fif(raw_fname)

    write_raw_bids(raw, bids_path=meg_bids_path)
    landmarks = get_anat_landmarks(
        t1w_mgh, raw.info, trans, fs_subject="sample", fs_subjects_dir=subjects_dir
    )
    write_anat(t1w_mgh, bids_path=t1_bids_path, landmarks=landmarks)
    read_trans = get_head_mri_trans(
        bids_path=meg_bids_path,
        t1_bids_path=t1_bids_path,
        fs_subject="sample",
        fs_subjects_dir=subjects_dir,
    )
    assert np.allclose(trans["trans"], read_trans["trans"])

    # Case 2: different sessions
    raw = _read_raw_fif(raw_fname)
    meg_bids_path = _bids_path.copy().update(
        root=tmp_path / "session_test", session="01", datatype="meg", suffix="meg"
    )
    t1_bids_path = meg_bids_path.copy().update(
        session="02", task=None, run=None, datatype="anat", suffix="T1w"
    )

    write_raw_bids(raw, bids_path=meg_bids_path)
    write_anat(t1w_mgh, bids_path=t1_bids_path, landmarks=landmarks)
    read_trans = get_head_mri_trans(
        bids_path=meg_bids_path,
        t1_bids_path=t1_bids_path,
        fs_subject="sample",
        fs_subjects_dir=subjects_dir,
    )
    assert np.allclose(trans["trans"], read_trans["trans"])

    # Test that incorrect subject directory throws error
    with pytest.raises(ValueError, match="Could not find"):
        estimated_trans = get_head_mri_trans(
            bids_path=bids_path, fs_subject="bad", fs_subjects_dir=subjects_dir
        )

    # Case 3: write with suffix for kind
    landmarks2 = landmarks.copy()
    landmarks2.dig[0]["r"] *= -1
    landmarks2.save(tmp_path / "landmarks2-dig.fif")
    landmarks2 = tmp_path / "landmarks2-dig.fif"
    write_anat(
        t1w_mgh,
        bids_path=t1_bids_path,
        overwrite=True,
        deface=True,
        landmarks={"coreg": landmarks, "deface": landmarks2},
    )
    read_trans1 = get_head_mri_trans(
        bids_path=meg_bids_path,
        t1_bids_path=t1_bids_path,
        fs_subject="sample",
        fs_subjects_dir=subjects_dir,
        kind="coreg",
    )
    assert np.allclose(trans["trans"], read_trans1["trans"])
    read_trans2 = get_head_mri_trans(
        bids_path=meg_bids_path,
        t1_bids_path=t1_bids_path,
        fs_subject="sample",
        fs_subjects_dir=subjects_dir,
        kind="deface",
    )
    assert not np.allclose(trans["trans"], read_trans2["trans"])

    # Test we're respecting existing suffix & data type
    # The following path is supposed to mimic a derivative generated by the
    # MNE-BIDS-Pipeline.
    #
    # XXX We MAY want to revise this once the BIDS-Pipeline produces more
    # BIDS-compatible output, e.g. including `channels.tsv` files for written
    # Raw data etc.
    raw = _read_raw_fif(raw_fname)
    deriv_root = tmp_path / "derivatives" / "mne-bids-pipeline"
    electrophys_path = (
        deriv_root / "sub-01" / "eeg" / "sub-01_task-av_proc-filt_raw.fif"
    )
    electrophys_path.parent.mkdir(parents=True)
    raw.save(electrophys_path)

    electrophys_bids_path = BIDSPath(
        subject="01",
        task="av",
        datatype="eeg",
        processing="filt",
        suffix="raw",
        extension=".fif",
        root=deriv_root,
        check=False,
    )
    t1_bids_path = _bids_path.copy().update(
        root=tmp_path / "mri_root", task=None, run=None
    )
    with (
        pytest.warns(RuntimeWarning, match="Did not find any channels.tsv"),
        pytest.warns(RuntimeWarning, match=r"Did not find any eeg\.json"),
        pytest.warns(RuntimeWarning, match=r"participants\.tsv file not found"),
    ):
        get_head_mri_trans(
            bids_path=electrophys_bids_path,
            t1_bids_path=t1_bids_path,
            fs_subject="sample",
            fs_subjects_dir=subjects_dir,
        )

    # bids_path without datatype is deprecated
    bids_path = electrophys_bids_path.copy().update(datatype=None)
    # defaut location is all wrong!
    with (
        pytest.raises(FileNotFoundError),
        pytest.warns(DeprecationWarning, match="did not have a datatype"),
    ):
        get_head_mri_trans(
            bids_path=bids_path,
            t1_bids_path=t1_bids_path,
            fs_subject="sample",
            fs_subjects_dir=subjects_dir,
        )

    # bids_path without suffix is deprecated
    bids_path = electrophys_bids_path.copy().update(suffix=None)
    # defaut location is all wrong!
    with (
        pytest.raises(FileNotFoundError),
        pytest.warns(DeprecationWarning, match="did not have a suffix"),
    ):
        get_head_mri_trans(
            bids_path=bids_path,
            t1_bids_path=t1_bids_path,
            fs_subject="sample",
            fs_subjects_dir=subjects_dir,
        )

    # Should fail for an unsupported coordinate frame
    raw = _read_raw_fif(raw_fname)
    bids_root = tmp_path / "unsupported_coord_frame"
    bids_path = BIDSPath(
        subject="01",
        task="av",
        datatype="meg",
        suffix="meg",
        extension=".fif",
        root=bids_root,
    )
    t1_bids_path = _bids_path.copy().update(
        root=tmp_path / "mri_root", task=None, run=None
    )
    write_raw_bids(raw=raw, bids_path=bids_path, verbose=False)


@testing.requires_testing_data
@pytest.mark.parametrize("with_extras", [False, True])
def test_handle_events_reading(tmp_path, with_extras):
    """Test reading events from a BIDS events.tsv file."""
    # We can use any `raw` for this
    raw = _read_raw_fif(raw_fname)

    # Create an arbitrary events.tsv file, to test we can deal with 'n/a'
    # make sure we can deal w/ "#" characters
    events = {
        "onset": [11, 12, "n/a"],
        "duration": ["n/a", "n/a", "n/a"],
        "trial_type": ["rec start", "trial #1", "trial #2!"],
    }
    if with_extras:
        events["foo"] = ["a", "b", "c"]
    events_fname = tmp_path / "bids1" / "sub-01_task-test_events.json"
    events_fname.parent.mkdir()
    _to_tsv(events, events_fname)

    with (
        pytest.warns(
            RuntimeWarning,
            match=re.escape(
                "The version of MNE-Python you are using (<1.10) "
                "does not support the extras argument in mne.Annotations. "
                "The extra column(s) ['foo'] will be ignored."
            ),
        )
        if with_extras and not check_version("mne", "1.10")
        else contextlib.nullcontext()
    ):
        raw, event_id = _handle_events_reading(events_fname, raw)

    ev_arr, ev_dict = mne.events_from_annotations(raw)
    assert list(ev_dict.values()) == [1, 2]  # auto-assigned
    want = len(events["onset"]) - 1  # one onset was n/a
    assert want == len(raw.annotations) == len(ev_arr) == len(ev_dict)
    if with_extras and check_version("mne", "1.10"):
        for d, v in zip(raw.annotations.extras, "abc"):
            assert "foo" in d
            assert d["foo"] == v

    # Test with a `stim_type` column instead of `trial_type`.
    events = {
        "onset": [11, 12, "n/a"],
        "duration": ["n/a", "n/a", "n/a"],
        "stim_type": ["rec start", "trial #1", "trial #2!"],
    }
    events_fname = tmp_path / "bids2" / "sub-01_task-test_events.json"
    events_fname.parent.mkdir()
    _to_tsv(events, events_fname)

    with pytest.warns(RuntimeWarning, match="This column should be renamed"):
        raw, _ = _handle_events_reading(events_fname, raw)
    events, event_id = mne.events_from_annotations(raw)

    # Test with only a `value` column.
    events = {
        "onset": [11, 12, 13, 14, 15],
        "duration": ["n/a", "n/a", 0.1, 0.1, "n/a"],
        "value": [3, 1, 1, 3, "n/a"],
    }
    events_fname = tmp_path / "bids3" / "sub-01_task-test_events.json"
    events_fname.parent.mkdir()
    _to_tsv(events, events_fname)

    raw, event_id = _handle_events_reading(events_fname, raw)
    ev_arr, ev_dict = mne.events_from_annotations(raw, event_id=event_id)
    assert len(ev_arr) == len(events["value"]) - 1  # one value was n/a
    assert {"1": 1, "3": 3} == event_id == ev_dict

    # Test with same `trial_type` referring to different `value`:
    # The events should be renamed automatically
    events = {
        "onset": [11, 12, 13, 14, 15],
        "duration": ["n/a", "n/a", "n/a", "n/a", "n/a"],
        "trial_type": ["event1", "event1", "event2", "event3", "event3"],
        "value": [1, 2, 3, 4, "n/a"],
    }
    events_fname = tmp_path / "bids4" / "sub-01_task-test_events.json"
    events_fname.parent.mkdir()
    _to_tsv(events, events_fname)

    raw, event_id = _handle_events_reading(events_fname, raw)
    ev_arr, ev_dict = mne.events_from_annotations(raw)
    # `event_id` will exclude the last event, as its value is `n/a`, but `ev_dict` won't
    # exclude it (it's made from annotations, which don't know about missing `value`s)
    assert len(event_id) == len(ev_dict) - 1
    # check the renaming
    assert len(ev_arr) == 5
    assert "event1/1" in ev_dict
    assert "event1/2" in ev_dict
    assert "event3/4" in ev_dict
    assert "event3/na" in ev_dict  # 'n/a' value should become 'na'
    assert "event2" in ev_dict  # has unique value mapping; should not be renamed

    # Test without any kind of event description.
    events = {"onset": [11, 12, "n/a"], "duration": ["n/a", "n/a", "n/a"]}
    events_fname = tmp_path / "bids5" / "sub-01_task-test_events.json"
    events_fname.parent.mkdir()
    _to_tsv(events, events_fname)

    raw, event_id = _handle_events_reading(events_fname, raw)
    ev_arr, ev_dict = mne.events_from_annotations(raw)
    assert event_id == ev_dict == {"n/a": 1}  # fallback behavior

    # Test with only a (non-numeric) `value` column
    events = {"onset": [10, 15], "duration": [1, 1], "value": ["A", "B"]}
    events_fname = tmp_path / "bids6" / "sub-01_task-test_events.tsv"
    events_fname.parent.mkdir()
    _to_tsv(events, events_fname)
    raw, event_id = _handle_events_reading(events_fname, raw)
    # don't pass event_id to mne.events_from_annotatations; its values are strings
    assert event_id == {"A": "A", "B": "B"}
    assert raw.annotations.description.tolist() == ["A", "B"]


@pytest.mark.filterwarnings(warning_str["channel_unit_changed"])
@testing.requires_testing_data
def test_keep_essential_annotations(tmp_path):
    """Test that essential Annotations are not omitted during I/O roundtrip."""
    raw = _read_raw_fif(raw_fname)
    annotations = mne.Annotations(
        onset=[raw.times[0]], duration=[1], description=["BAD_ACQ_SKIP"]
    )
    raw.set_annotations(annotations)

    # Write data, remove events.tsv, then try to read again
    bids_path = BIDSPath(subject="01", task="task", datatype="meg", root=tmp_path)
    with pytest.warns(RuntimeWarning, match="Acquisition skips detected"):
        write_raw_bids(raw, bids_path, overwrite=True)

    bids_path.copy().update(suffix="events", extension=".tsv").fpath.unlink()
    raw_read = read_raw_bids(bids_path)

    assert len(raw_read.annotations) == len(raw.annotations) == 1
    assert raw_read.annotations[0]["description"] == raw.annotations[0]["description"]


@testing.requires_testing_data
def test_adding_essential_annotations_to_dict(tmp_path):
    """Test that essential Annotations are auto-added to the `event_id` dictionary."""
    raw = _read_raw_fif(raw_fname)
    annotations = mne.Annotations(
        onset=[raw.times[0]], duration=[1], description=["BAD_ACQ_SKIP"]
    )
    raw.set_annotations(annotations)
    events = mne.find_events(raw)
    event_id = sample_data_event_id.copy()
    obj_id = id(event_id)

    # see that no error is raised for missing event_id key for BAD_ACQ_SKIP
    bids_path = BIDSPath(subject="01", task="task", datatype="meg", root=tmp_path)
    with pytest.warns(RuntimeWarning, match="Acquisition skips detected"):
        write_raw_bids(raw, bids_path, overwrite=True, events=events, event_id=event_id)
    # make sure we didn't modify the user-passed dict
    assert event_id == sample_data_event_id
    assert obj_id == id(event_id)


@pytest.mark.filterwarnings(warning_str["channel_unit_changed"])
@testing.requires_testing_data
def test_handle_scans_reading(tmp_path):
    """Test reading data from a BIDS scans.tsv file."""
    raw = _read_raw_fif(raw_fname)
    suffix = "meg"

    # write copy of raw with line freq of 60
    # bids basename and fname
    bids_path = BIDSPath(
        subject="01",
        session="01",
        task="audiovisual",
        run="01",
        datatype=suffix,
        root=tmp_path,
    )
    bids_path = write_raw_bids(raw, bids_path, overwrite=True)
    raw_01 = read_raw_bids(bids_path)

    # find sidecar scans.tsv file and alter the
    # acquisition time to not have the optional microseconds
    scans_path = BIDSPath(
        subject=bids_path.subject,
        session=bids_path.session,
        root=tmp_path,
        suffix="scans",
        extension=".tsv",
    )
    scans_tsv = _from_tsv(scans_path)
    acq_time_str = scans_tsv["acq_time"][0]
    acq_time = datetime.strptime(acq_time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    acq_time = acq_time.replace(tzinfo=UTC)
    new_acq_time = acq_time_str.split(".")[0] + "Z"
    assert acq_time == raw_01.info["meas_date"]
    scans_tsv["acq_time"][0] = new_acq_time
    _to_tsv(scans_tsv, scans_path)

    # now re-load the data and it should be different
    # from the original date and the same as the newly altered date
    raw_02 = read_raw_bids(bids_path)
    new_acq_time = new_acq_time.replace("Z", ".0Z")
    new_acq_time = datetime.strptime(new_acq_time, "%Y-%m-%dT%H:%M:%S.%fZ")
    new_acq_time = new_acq_time.replace(tzinfo=UTC)
    assert raw_02.info["meas_date"] == new_acq_time
    assert new_acq_time != raw_01.info["meas_date"]

    # Test without optional zero-offset UTC time-zone indicator (i.e., without trailing
    # "Z")
    for has_microsecs in (True, False):
        new_acq_time_str = "2002-12-03T19:01:10"
        date_format = "%Y-%m-%dT%H:%M:%S"
        if has_microsecs:
            new_acq_time_str += ".0"
            date_format += ".%f"

        scans_tsv["acq_time"][0] = new_acq_time_str
        _to_tsv(scans_tsv, scans_path)

        # now re-load the data and it should be different
        # from the original date and the same as the newly altered date
        raw_03 = read_raw_bids(bids_path)
        new_acq_time = datetime.strptime(new_acq_time_str, date_format)
        assert raw_03.info["meas_date"] == new_acq_time.astimezone(UTC)

    # Regression for naive, pre-epoch acquisition times (Windows bug GH-1399)
    pre_epoch_str = "1950-06-15T13:45:30"
    scans_tsv["acq_time"][0] = pre_epoch_str
    _to_tsv(scans_tsv, scans_path)

    raw_pre_epoch = read_raw_bids(bids_path)
    pre_epoch_naive = datetime.strptime(pre_epoch_str, "%Y-%m-%dT%H:%M:%S")
    local_tz = datetime.now().astimezone().tzinfo or UTC
    expected_pre_epoch = pre_epoch_naive.replace(tzinfo=local_tz).astimezone(UTC)
    assert raw_pre_epoch.info["meas_date"] == expected_pre_epoch
    if raw_pre_epoch.annotations.orig_time is not None:
        assert raw_pre_epoch.annotations.orig_time == expected_pre_epoch


@pytest.mark.filterwarnings(warning_str["channel_unit_changed"])
def test_handle_scans_reading_brainvision(tmp_path):
    """Test stability of BrainVision's different file extensions."""
    test_scan_eeg = OrderedDict(
        [
            ("filename", [Path("eeg/sub-01_ses-eeg_task-rest_eeg.eeg")]),
            ("acq_time", ["2000-01-01T12:00:00.000000Z"]),
        ]
    )
    test_scan_vmrk = OrderedDict(
        [
            ("filename", [Path("eeg/sub-01_ses-eeg_task-rest_eeg.vmrk")]),
            ("acq_time", ["2000-01-01T12:00:00.000000Z"]),
        ]
    )
    test_scan_edf = OrderedDict(
        [
            ("filename", [Path("eeg/sub-01_ses-eeg_task-rest_eeg.edf")]),
            ("acq_time", ["2000-01-01T12:00:00.000000Z"]),
        ]
    )
    os.mkdir(tmp_path / "eeg")
    for test_scan in [test_scan_eeg, test_scan_vmrk, test_scan_edf]:
        _to_tsv(test_scan, tmp_path / test_scan["filename"][0])

    bids_path = BIDSPath(
        subject="01", session="eeg", task="rest", datatype="eeg", root=tiny_bids_root
    )

    raw = read_raw_bids(bids_path)

    for test_scan in [test_scan_eeg, test_scan_vmrk]:
        _handle_scans_reading(tmp_path / test_scan["filename"][0], raw, bids_path)

    with pytest.raises(ValueError, match="is not in list"):
        _handle_scans_reading(tmp_path / test_scan_edf["filename"][0], raw, bids_path)


@pytest.mark.filterwarnings(warning_str["channel_unit_changed"])
@testing.requires_testing_data
def test_handle_info_reading(tmp_path):
    """Test reading information from a BIDS sidecar JSON file."""
    # read in USA dataset, so it should find 50 Hz
    raw = _read_raw_fif(raw_fname)

    # write copy of raw with line freq of 60
    # bids basename and fname
    bids_path = BIDSPath(
        subject="01", session="01", task="audiovisual", run="01", root=tmp_path
    )
    suffix = "meg"
    bids_fname = bids_path.copy().update(suffix=suffix, extension=".fif")
    write_raw_bids(raw, bids_path, overwrite=True)

    # find sidecar JSON fname
    bids_fname.update(datatype=suffix)
    sidecar_fname = _find_matching_sidecar(bids_fname, suffix=suffix, extension=".json")
    sidecar_fname = Path(sidecar_fname)

    # assert that we get the same line frequency set
    raw = read_raw_bids(bids_path=bids_path)
    assert raw.info["line_freq"] == 60

    # setting line_freq to None should produce 'n/a' in the JSON sidecar
    raw.info["line_freq"] = None
    write_raw_bids(raw, bids_path, overwrite=True, format="FIF")
    raw = read_raw_bids(bids_path=bids_path)
    assert raw.info["line_freq"] is None

    sidecar_json = json.loads(sidecar_fname.read_text(encoding="utf-8"))
    assert sidecar_json["PowerLineFrequency"] == "n/a"

    # 2. if line frequency is not set in raw file, then ValueError
    del raw.info["line_freq"]
    with pytest.raises(ValueError, match="PowerLineFrequency .* required"):
        write_raw_bids(raw, bids_path, overwrite=True, format="FIF")

    # check whether there are "Extra points" in raw.info['dig'] if
    # DigitizedHeadPoints is set to True and not otherwise
    n_dig_points = 0
    for dig_point in raw.info["dig"]:
        if dig_point["kind"] == FIFF.FIFFV_POINT_EXTRA:
            n_dig_points += 1
    if sidecar_json["DigitizedHeadPoints"]:
        assert n_dig_points > 0
    else:
        assert n_dig_points == 0

    # check whether any of NAS/LPA/RPA are present in raw.info['dig']
    # DigitizedLandmark is set to True, and False otherwise
    landmark_present = False
    for dig_point in raw.info["dig"]:
        if dig_point["kind"] in [
            FIFF.FIFFV_POINT_LPA,
            FIFF.FIFFV_POINT_RPA,
            FIFF.FIFFV_POINT_NASION,
        ]:
            landmark_present = True
            break
    if landmark_present:
        assert sidecar_json["DigitizedLandmarks"] is True
    else:
        assert sidecar_json["DigitizedLandmarks"] is False

    # make a copy of the sidecar in "derivatives/"
    # to check that we make sure we always get the right sidecar
    # in addition, it should not break the sidecar reading
    # in `read_raw_bids`
    raw.info["line_freq"] = 60
    write_raw_bids(raw, bids_path, overwrite=True, format="FIF")
    deriv_dir = tmp_path / "derivatives"
    deriv_dir.mkdir()
    sidecar_copy = deriv_dir / op.basename(sidecar_fname)
    sidecar_json = json.loads(sidecar_fname.read_text(encoding="utf-8"))
    sidecar_json["PowerLineFrequency"] = 45
    _write_json(sidecar_copy, sidecar_json)
    raw = read_raw_bids(bids_path=bids_path)
    assert raw.info["line_freq"] == 60

    # 3. assert that we get an error when sidecar json doesn't match
    _update_sidecar(sidecar_fname, "PowerLineFrequency", 55)
    with pytest.warns(RuntimeWarning, match="Defaulting to .* sidecar JSON"):
        raw = read_raw_bids(bids_path=bids_path)
        assert raw.info["line_freq"] == 55


@pytest.mark.filterwarnings(warning_str["maxshield"])
@testing.requires_testing_data
def test_handle_chpi_reading(tmp_path):
    """Test reading of cHPI information."""
    raw = _read_raw_fif(raw_fname_chpi, allow_maxshield="yes")
    root = tmp_path / "chpi"
    root.mkdir()
    bids_path = BIDSPath(
        subject="01",
        session="01",
        task="audiovisual",
        run="01",
        root=root,
        datatype="meg",
    )
    bids_path = write_raw_bids(raw, bids_path)

    raw_read = read_raw_bids(bids_path)
    assert raw_read.info["hpi_subsystem"] is not None

    # cause conflicts between cHPI info in sidecar and raw data
    meg_json_path = bids_path.copy().update(suffix="meg", extension=".json")
    with open(meg_json_path, encoding="utf-8") as f:
        meg_json_data = json.load(f)

    # cHPI frequency mismatch
    meg_json_data_freq_mismatch = meg_json_data.copy()
    meg_json_data_freq_mismatch["HeadCoilFrequency"][0] = 123
    _write_json(meg_json_path, meg_json_data_freq_mismatch, overwrite=True)

    with (
        pytest.warns(RuntimeWarning, match="Defaulting to .* mne.Raw object"),
    ):
        raw_read = read_raw_bids(bids_path, extra_params=dict(allow_maxshield="yes"))

    # cHPI "off" according to sidecar, but present in the data
    meg_json_data_chpi_mismatch = meg_json_data.copy()
    meg_json_data_chpi_mismatch["ContinuousHeadLocalization"] = False
    _write_json(meg_json_path, meg_json_data_chpi_mismatch, overwrite=True)

    raw_read = read_raw_bids(bids_path)
    assert raw_read.info["hpi_subsystem"] is None
    assert raw_read.info["hpi_meas"] == []


@pytest.mark.filterwarnings(warning_str["nasion_not_found"])
@testing.requires_testing_data
def test_handle_eeg_coords_reading(tmp_path):
    """Test reading EEG coordinates from BIDS files."""
    bids_path = BIDSPath(
        subject=subject_id,
        session=session_id,
        run=run,
        acquisition=acq,
        task=task,
        root=tmp_path,
    )

    raw_fname = op.join(data_path, "EDF", "test_reduced.edf")
    raw = _read_raw_edf(raw_fname)

    # ensure we are writing 'eeg' data
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names}, on_unit_change="ignore")

    # set a `random` montage
    ch_names = raw.ch_names
    elec_locs = np.random.RandomState(0).randn(len(ch_names), 3)
    ch_pos = dict(zip(ch_names, elec_locs))

    # # create montage in 'unknown' coordinate frame
    # # and assert coordsystem/electrodes sidecar tsv don't exist
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="unknown")
    raw.set_montage(montage)
    ctx = nullcontext()
    with ctx:
        write_raw_bids(raw, bids_path, overwrite=True)

    bids_path.update(root=tmp_path)
    coordsystem_fname = _find_matching_sidecar(
        bids_path, suffix="coordsystem", extension=".json", on_error="warn"
    )
    electrodes_fname = _find_matching_sidecar(
        bids_path, suffix="electrodes", extension=".tsv", on_error="warn"
    )
    assert coordsystem_fname is not None
    assert electrodes_fname is not None

    # create montage in head frame and set should result in
    # an error if landmarks are not set
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    raw.set_montage(montage)
    ctx = nullcontext()
    with ctx:
        write_raw_bids(raw, bids_path, overwrite=True)

    write_raw_bids(raw, bids_path, overwrite=True)

    # obtain the sensor positions and assert ch_coords are same
    raw_test = read_raw_bids(bids_path, verbose=True)
    assert not object_diff(raw.info["chs"], raw_test.info["chs"])

    # modify coordinate frame to not-captrak
    coordsystem_fname = _find_matching_sidecar(
        bids_path, suffix="coordsystem", extension=".json"
    )
    _update_sidecar(coordsystem_fname, "EEGCoordinateSystem", "besa")
    with pytest.warns(
        RuntimeWarning, match="is not a BIDS-acceptable coordinate frame for EEG"
    ):
        raw_test = read_raw_bids(bids_path)
        assert raw_test.info["dig"] is None

    # Test EEGLAB and EEGLAB-HJ coordinate systems are accepted and
    # map to ctf_head, then get transformed to head via synthetic fiducials
    for eeglab_frame in ("EEGLAB", "EEGLAB-HJ"):
        _update_sidecar(coordsystem_fname, "EEGCoordinateSystem", eeglab_frame)
        with pytest.warns(RuntimeWarning, match="No fiducial points found"):
            raw_test = read_raw_bids(bids_path)
        assert raw_test.info["dig"] is not None
        montage = raw_test.get_montage()
        pos = montage.get_positions()
        # Synthetic fiducials enable ctf_head -> head transform
        assert pos["coord_frame"] == "head"

    # Test "n/a" coordinate units are handled by inferring from magnitudes
    # Reset to a known good coordinate system first
    _update_sidecar(coordsystem_fname, "EEGCoordinateSystem", "CTF")
    _update_sidecar(coordsystem_fname, "EEGCoordinateUnits", "n/a")
    with pytest.warns(RuntimeWarning) as record:
        raw_test = read_raw_bids(bids_path)
    messages = [str(w.message) for w in record]
    assert any('Coordinate unit is "n/a"' in m for m in messages)
    assert any("No fiducial points found" in m for m in messages)
    assert raw_test.info["dig"] is not None
    montage = raw_test.get_montage()
    pos = montage.get_positions()
    # CTF maps to ctf_head, then synthetic fiducials transform to head
    assert pos["coord_frame"] == "head"

    # Test that non-"n/a" invalid units still skip electrodes.tsv
    _update_sidecar(coordsystem_fname, "EEGCoordinateUnits", "km")
    with pytest.warns(
        RuntimeWarning, match="Coordinate unit is not an accepted BIDS unit"
    ):
        raw_test = read_raw_bids(bids_path)
    assert raw_test.info["dig"] is None


def test_read_eeg_missing_coordsystem_warns(tmp_path):
    """EEG reads should warn, not fail, if electrodes.tsv lacks coordsystem."""
    bids_root = tmp_path / "tiny_bids_missing_coordsystem"
    sh.copytree(tiny_bids_root, bids_root)

    subj_eeg_dir = bids_root / "sub-01" / "ses-eeg" / "eeg"
    coordsystem_files = [
        subj_eeg_dir / "sub-01_ses-eeg_coordsystem.json",
        subj_eeg_dir / "sub-01_ses-eeg_space-CapTrak_coordsystem.json",
    ]
    for coordsystem_file in coordsystem_files:
        coordsystem_file.unlink()
    (subj_eeg_dir / "sub-01_ses-eeg_space-CapTrak_electrodes.tsv").unlink()

    bids_path = BIDSPath(
        subject="01",
        session="eeg",
        task="rest",
        datatype="eeg",
        root=bids_root,
    )

    with pytest.warns(
        RuntimeWarning, match=r"Could not find coordsystem\.json for electrodes file:"
    ) as warning_record:
        raw = read_raw_bids(bids_path=bids_path)
    assert "coordsystem.json is REQUIRED whenever electrodes.tsv is present" in str(
        warning_record[0].message
    )
    assert "re-run the BIDS validator." in str(warning_record[0].message)
    assert raw.info["dig"] is None


def test_read_eeg_root_electrodes_no_coordsystem(tmp_path):
    """Root-level electrodes.tsv without coordsystem.json should not hard-fail."""
    bids_root = tmp_path / "tiny_bids_root_level_electrodes"
    sh.copytree(tiny_bids_root, bids_root)

    subj_eeg_dir = bids_root / "sub-01" / "ses-eeg" / "eeg"
    coordsystem_files = [
        subj_eeg_dir / "sub-01_ses-eeg_coordsystem.json",
        subj_eeg_dir / "sub-01_ses-eeg_space-CapTrak_coordsystem.json",
    ]
    for coordsystem_file in coordsystem_files:
        coordsystem_file.unlink()

    electrodes_src = subj_eeg_dir / "sub-01_ses-eeg_electrodes.tsv"
    electrodes_dst = bids_root / "electrodes.tsv"
    electrodes_src.replace(electrodes_dst)
    (subj_eeg_dir / "sub-01_ses-eeg_space-CapTrak_electrodes.tsv").unlink()

    bids_path = BIDSPath(
        subject="01",
        session="eeg",
        task="rest",
        datatype="eeg",
        root=bids_root,
    )
    electrodes_fname = _find_matching_sidecar(
        bids_path, suffix="electrodes", extension=".tsv", on_error="raise"
    )
    assert electrodes_fname == electrodes_dst

    with pytest.warns(
        RuntimeWarning, match=r"Could not find coordsystem\.json for electrodes file:"
    ) as warning_record:
        raw = read_raw_bids(bids_path=bids_path)
    assert "coordsystem.json is REQUIRED whenever electrodes.tsv is present" in str(
        warning_record[0].message
    )
    assert "re-run the BIDS validator." in str(warning_record[0].message)
    assert raw.info["dig"] is None


@pytest.mark.filterwarnings(warning_str["channel_unit_changed"])
@testing.requires_testing_data
def test_read_meg_missing_coordsystem_warns(tmp_path):
    """MEG reads should warn, not fail, if electrodes.tsv lacks coordsystem."""
    bids_path = _bids_path.copy().update(root=tmp_path, datatype="meg", suffix="meg")
    raw = _read_raw_fif(raw_fname)
    write_raw_bids(raw, bids_path, overwrite=True, verbose=False)

    # Remove the coordsystem.json created by write_raw_bids
    coordsystem_fname = _find_matching_sidecar(
        bids_path, suffix="coordsystem", extension=".json", on_error="ignore"
    )
    if coordsystem_fname is not None:
        os.remove(coordsystem_fname)

    # The MEG sample data does not produce electrodes.tsv by default.
    # Manually add one so we can test the warning when coordsystem.json
    # is missing but electrodes.tsv exists.
    electrodes_tsv = bids_path.copy().update(
        suffix="electrodes", extension=".tsv", task=None, run=None
    )
    Path(electrodes_tsv.fpath).write_text(
        "name\tx\ty\tz\nEEG001\t0.0\t0.0\t0.0\n"
    )

    with pytest.warns(
        RuntimeWarning, match=r"Could not find coordsystem\.json for electrodes file:"
    ) as warning_record:
        read_raw_bids(bids_path=bids_path, verbose=False)
    coord_warnings = [
        w
        for w in warning_record
        if "coordsystem.json is REQUIRED" in str(w.message)
    ]
    assert len(coord_warnings) == 1


@pytest.mark.parametrize("bids_path", [_bids_path, _bids_path_minimal])
@pytest.mark.filterwarnings(warning_str["nasion_not_found"])
@testing.requires_testing_data
def test_handle_ieeg_coords_reading(bids_path, tmp_path):
    """Test reading iEEG coordinates from BIDS files."""
    raw_fname = op.join(data_path, "EDF", "test_reduced.edf")
    bids_fname = bids_path.copy().update(
        datatype="ieeg", suffix="ieeg", extension=".edf", root=tmp_path
    )
    raw = _read_raw_edf(raw_fname)

    # ensure we are writing 'ecog'/'ieeg' data
    raw.set_channel_types({ch: "ecog" for ch in raw.ch_names}, on_unit_change="ignore")

    # coordinate frames in mne-python should all map correctly
    # set a `random` montage
    ch_names = raw.ch_names
    elec_locs = np.random.RandomState(0).randn(len(ch_names), 3)
    ch_pos = dict(zip(ch_names, elec_locs))
    coordinate_frames = ["mni_tal"]
    for coord_frame in coordinate_frames:
        # XXX: mne-bids doesn't support multiple electrodes.tsv files
        sh.rmtree(tmp_path)
        montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame=coord_frame)
        raw.set_montage(montage)
        write_raw_bids(raw, bids_fname, overwrite=True, verbose=False)
        # read in raw file w/ updated coordinate frame
        # and make sure all digpoints are correct coordinate frames
        raw_test = read_raw_bids(bids_path=bids_fname, verbose=False)
        coord_frame_int = MNE_STR_TO_FRAME[coord_frame]
        for digpoint in raw_test.info["dig"]:
            assert digpoint["coord_frame"] == coord_frame_int

    # start w/ new bids root
    sh.rmtree(tmp_path)
    write_raw_bids(raw, bids_fname, overwrite=True, verbose=False)

    # obtain the sensor positions and assert ch_coords are same
    raw_test = read_raw_bids(bids_path=bids_fname, verbose=False)
    orig_locs = raw.info["dig"][1]
    test_locs = raw_test.info["dig"][1]
    assert orig_locs == test_locs
    assert not object_diff(raw.info["chs"], raw_test.info["chs"])

    # read in the data and assert montage is the same
    # regardless of 'm', 'cm', 'mm', or 'pixel'
    scalings = {"m": 1, "cm": 100, "mm": 1000}
    bids_fname.update(root=tmp_path)
    coordsystem_fname = _find_matching_sidecar(
        bids_fname, suffix="coordsystem", extension=".json"
    )
    electrodes_fname = _find_matching_sidecar(
        bids_fname, suffix="electrodes", extension=".tsv"
    )
    orig_electrodes_dict = _from_tsv(electrodes_fname, [str, float, float, float, str])

    # not BIDS specified should not be read
    coord_unit = "km"
    scaling = 0.001
    _update_sidecar(coordsystem_fname, "iEEGCoordinateUnits", coord_unit)
    electrodes_dict = _from_tsv(electrodes_fname, [str, float, float, float, str])
    for axis in ["x", "y", "z"]:
        electrodes_dict[axis] = np.multiply(orig_electrodes_dict[axis], scaling)
    _to_tsv(electrodes_dict, electrodes_fname)
    with pytest.warns(
        RuntimeWarning, match="Coordinate unit is not an accepted BIDS unit"
    ):
        raw_test = read_raw_bids(bids_path=bids_fname, verbose=False)

    # correct BIDS units should scale to meters properly
    for coord_unit, scaling in scalings.items():
        # update coordinate SI units
        _update_sidecar(coordsystem_fname, "iEEGCoordinateUnits", coord_unit)
        electrodes_dict = _from_tsv(electrodes_fname, [str, float, float, float, str])
        for axis in ["x", "y", "z"]:
            electrodes_dict[axis] = np.multiply(orig_electrodes_dict[axis], scaling)
        _to_tsv(electrodes_dict, electrodes_fname)

        # read in raw file w/ updated montage
        raw_test = read_raw_bids(bids_path=bids_fname, verbose=False)

        # obtain the sensor positions and make sure they're the same
        assert_dig_allclose(raw.info, raw_test.info)

    # XXX: Improve by changing names to 'unknown' coordframe (needs mne PR)
    # check that coordinate systems other coordinate systems should be named
    # in the file and not the CoordinateSystem, which is reserved for keywords
    coordinate_frames = ["Other"]
    for coord_frame in coordinate_frames:
        # update coordinate units
        _update_sidecar(coordsystem_fname, "iEEGCoordinateSystem", coord_frame)
        # read in raw file w/ updated coordinate frame
        # and make sure all digpoints are MRI coordinate frame
        with (
            pytest.warns(RuntimeWarning, match="not an MNE-Python coordinate frame"),
        ):
            raw_test = read_raw_bids(bids_path=bids_fname, verbose=False)
            assert raw_test.info["dig"] is not None

    # check that standard template identifiers that are unsupported in
    # mne-python coordinate frames, still get read in, but produce a warning
    for coord_frame in BIDS_SHARED_COORDINATE_FRAMES:
        # update coordinate units
        _update_sidecar(coordsystem_fname, "iEEGCoordinateSystem", coord_frame)
        # read in raw file w/ updated coordinate frame
        # and make sure all digpoints are MRI coordinate frame
        if coord_frame in BIDS_TO_MNE_FRAMES:
            raw_test = read_raw_bids(bids_path=bids_fname, verbose=False)
        else:
            with (
                pytest.warns(
                    RuntimeWarning, match="not an MNE-Python coordinate frame"
                ),
            ):
                raw_test = read_raw_bids(bids_path=bids_fname, verbose=False)
        assert raw_test.info["dig"] is not None

    # ACPC should be read in as RAS for iEEG
    _update_sidecar(coordsystem_fname, "iEEGCoordinateSystem", "ACPC")
    raw_test = read_raw_bids(bids_path=bids_fname, verbose=False)
    coord_frame_int = MNE_STR_TO_FRAME["ras"]
    for digpoint in raw_test.info["dig"]:
        assert digpoint["coord_frame"] == coord_frame_int

    # ScanRAS should be read in as RAS for iEEG
    _update_sidecar(coordsystem_fname, "iEEGCoordinateSystem", "ScanRAS")
    raw_test = read_raw_bids(bids_path=bids_fname, verbose=False)
    coord_frame_int = MNE_STR_TO_FRAME["ras"]
    for digpoint in raw_test.info["dig"]:
        assert digpoint["coord_frame"] == coord_frame_int

    # if we delete the coordsystem.json file, an error will be raised
    os.remove(coordsystem_fname)
    with pytest.raises(
        RuntimeError,
        match="coordsystem.json is REQUIRED whenever electrodes.tsv is present",
    ):
        raw = read_raw_bids(bids_path=bids_fname, verbose=False)

    # test error message if electrodes is not a subset of Raw
    bids_path.update(root=tmp_path)
    write_raw_bids(raw, bids_path, overwrite=True)
    electrodes_dict = _from_tsv(electrodes_fname)
    # pop off 5 channels
    for key in electrodes_dict.keys():
        for i in range(5):
            electrodes_dict[key].pop()
    _to_tsv(electrodes_dict, electrodes_fname)
    # popping off channels should not result in an error
    # however, a warning will be raised through mne-python
    with (
        pytest.warns(RuntimeWarning, match="DigMontage is only a subset of info"),
    ):
        read_raw_bids(bids_path=bids_fname, verbose=False)

    # make sure montage is set if there are coordinates w/ 'n/a'
    raw.info["bads"] = []
    write_raw_bids(raw, bids_path, overwrite=True, verbose=False)
    electrodes_dict = _from_tsv(electrodes_fname)
    for axis in ["x", "y", "z"]:
        electrodes_dict[axis][0] = "n/a"
        electrodes_dict[axis][3] = "n/a"
    _to_tsv(electrodes_dict, electrodes_fname)

    # test if montage is correctly set via mne-bids
    # electrode coordinates should be nan
    # when coordinate is 'n/a'
    nan_chs = [electrodes_dict["name"][i] for i in [0, 3]]
    with (
        pytest.warns(RuntimeWarning, match="There are channels without locations"),
    ):
        raw = read_raw_bids(bids_path=bids_fname, verbose=False)
        for idx, ch in enumerate(raw.info["chs"]):
            if ch["ch_name"] in nan_chs:
                assert all(np.isnan(ch["loc"][:3]))
            else:
                assert not any(np.isnan(ch["loc"][:3]))
            assert ch["ch_name"] not in raw.info["bads"]


@pytest.mark.filterwarnings(warning_str["channel_unit_changed"])
@pytest.mark.parametrize("fname", ["testdata_ctf.ds", "catch-alp-good-f.ds"])
@testing.requires_testing_data
def test_get_head_mri_trans_ctf(fname, tmp_path):
    """Test getting a trans object from BIDS data in CTF."""
    nib = pytest.importorskip("nibabel")

    ctf_data_path = op.join(data_path, "CTF")
    raw_ctf_fname = op.join(ctf_data_path, fname)
    raw_ctf = _read_raw_ctf(raw_ctf_fname, clean_names=True)
    bids_path = _bids_path.copy().update(root=tmp_path, datatype="meg", suffix="meg")
    write_raw_bids(raw_ctf, bids_path, overwrite=False)

    # Take a fake trans
    trans = mne.read_trans(str(raw_fname).replace("_raw.fif", "-trans.fif"))

    # Get the T1 weighted MRI data file ... test write_anat with a nibabel
    # image instead of a file path
    t1w_mgh = op.join(data_path, "subjects", "sample", "mri", "T1.mgz")
    t1w_mgh = nib.load(t1w_mgh)

    t1w_bids_path = BIDSPath(
        subject=subject_id, session=session_id, acquisition=acq, root=tmp_path
    )
    landmarks = get_anat_landmarks(
        t1w_mgh,
        raw_ctf.info,
        trans,
        fs_subject="sample",
        fs_subjects_dir=op.join(data_path, "subjects"),
    )
    write_anat(t1w_mgh, bids_path=t1w_bids_path, landmarks=landmarks)

    # Try to get trans back through fitting points
    estimated_trans = get_head_mri_trans(
        bids_path=bids_path,
        extra_params=dict(clean_names=True),
        fs_subject="sample",
        fs_subjects_dir=op.join(data_path, "subjects"),
    )

    assert_almost_equal(trans["trans"], estimated_trans["trans"])


@pytest.mark.filterwarnings(warning_str["channel_unit_changed"])
@testing.requires_testing_data
def test_read_raw_bids_pathlike(tmp_path):
    """Test that read_raw_bids() can handle a Path-like bids_root."""
    bids_path = _bids_path.copy().update(root=tmp_path, datatype="meg")
    raw = _read_raw_fif(raw_fname, verbose=False)
    write_raw_bids(raw, bids_path, overwrite=True, verbose=False)
    raw = read_raw_bids(bids_path=bids_path)


@pytest.mark.filterwarnings(warning_str["channel_unit_changed"])
@testing.requires_testing_data
def test_read_raw_datatype(tmp_path):
    """Test that read_raw_bids() can infer the str_suffix if need be."""
    bids_path = _bids_path.copy().update(root=tmp_path, datatype="meg")
    raw = _read_raw_fif(raw_fname, verbose=False)
    write_raw_bids(raw, bids_path, overwrite=True, verbose=False)

    raw_1 = read_raw_bids(bids_path=bids_path)
    bids_path.update(datatype=None)
    raw_2 = read_raw_bids(bids_path=bids_path)
    raw_3 = read_raw_bids(bids_path=bids_path)

    raw_1.crop(0, 2).load_data()
    raw_2.crop(0, 2).load_data()
    raw_3.crop(0, 2).load_data()

    assert raw_1 == raw_2
    assert raw_1 == raw_3


@testing.requires_testing_data
def test_handle_channel_type_casing(tmp_path):
    """Test that non-uppercase entries in the `type` column are accepted."""
    bids_path = _bids_path.copy().update(root=tmp_path)
    raw = _read_raw_fif(raw_fname, verbose=False)

    write_raw_bids(raw, bids_path, overwrite=True, verbose=False)

    ch_path = bids_path.copy().update(
        root=tmp_path, datatype="meg", suffix="channels", extension=".tsv"
    )
    bids_channels_fname = ch_path.fpath

    # Convert all channel type entries to lowercase.
    channels_data = _from_tsv(bids_channels_fname)
    channels_data["type"] = [t.lower() for t in channels_data["type"]]
    _to_tsv(channels_data, bids_channels_fname)

    with pytest.warns(RuntimeWarning, match="lowercase spelling"):
        read_raw_bids(bids_path)


@testing.requires_testing_data
def test_handle_non_mne_channel_type(tmp_path):
    """Test that channel types not known to MNE will be read as 'misc'."""
    bids_path = _bids_path.copy().update(root=tmp_path)
    raw = _read_raw_fif(raw_fname, verbose=False)

    write_raw_bids(raw, bids_path, overwrite=True, verbose=False)

    channels_tsv_path = (
        bids_path.copy()
        .update(root=tmp_path, datatype="meg", suffix="channels", extension=".tsv")
        .fpath
    )

    channels_data = _from_tsv(channels_tsv_path)
    # Violates BIDS, but ensures we won't have an appropriate
    # BIDS -> MNE mapping.
    ch_idx = -1
    channels_data["type"][ch_idx] = "FOOBAR"
    _to_tsv(data=channels_data, fname=channels_tsv_path)

    with (
        pytest.warns(RuntimeWarning, match='will be set to "misc"'),
    ):
        raw = read_raw_bids(bids_path)

    # Should be a 'misc' channel.
    assert raw.get_channel_types([channels_data["name"][ch_idx]]) == ["misc"]


@pytest.mark.filterwarnings(warning_str["channel_unit_changed"])
@testing.requires_testing_data
def test_bads_reading(tmp_path):
    """Test reading bad channels."""
    bids_path = _bids_path.copy().update(root=tmp_path, datatype="meg")
    bads_raw = ["MEG 0112", "MEG 0113"]
    bads_sidecar = ["EEG 053", "MEG 2443"]

    # Produce conflicting information between raw and sidecar file.
    raw = _read_raw_fif(raw_fname, verbose=False)
    raw.info["bads"] = bads_sidecar
    write_raw_bids(raw, bids_path, verbose=False)

    raw = _read_raw(bids_path.copy().update(extension=".fif").fpath, preload=True)
    raw.info["bads"] = bads_raw
    raw.save(raw.filenames[0], overwrite=True)

    # Upon reading the data, only the sidecar info should be present.
    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    assert len(raw.info["bads"]) == len(bads_sidecar)
    assert set(raw.info["bads"]) == set(bads_sidecar)


@pytest.mark.filterwarnings(warning_str["channel_unit_changed"])
@testing.requires_testing_data
def test_write_read_fif_split_file(tmp_path, monkeypatch):
    """Test split files are read correctly."""
    # load raw test file, extend it to be larger than 2gb, and save it
    bids_root = tmp_path / "bids"
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()

    bids_path = _bids_path.copy().update(root=bids_root, datatype="meg")
    raw = _read_raw_fif(raw_fname, verbose=False)
    bids_path.update(acquisition=None)
    write_raw_bids(raw, bids_path, verbose=False)
    bids_path.update(acquisition="01")
    n_channels = len(raw.ch_names)
    n_times = int(2.5e6 / n_channels)  # enough to produce a 10MB split
    data = np.random.RandomState(0).randn(n_channels, n_times).astype(np.float32)
    raw = mne.io.RawArray(data, raw.info)
    big_fif_fname = Path(tmp_dir) / "test_raw.fif"

    split_size = "10MB"
    raw.save(big_fif_fname, split_size=split_size)
    raw = _read_raw_fif(big_fif_fname, verbose=False)

    with monkeypatch.context() as m:  # Force MNE-BIDS to split at 10MB
        m.setattr(mne_bids.write, "_FIFF_SPLIT_SIZE", split_size)
        write_raw_bids(raw, bids_path, verbose=False)

    # test whether split raw files were read correctly
    raw1 = read_raw_bids(bids_path=bids_path)
    assert "split-01" in str(bids_path.fpath)
    bids_path.update(split="01")
    raw2 = read_raw_bids(bids_path=bids_path)
    bids_path.update(split="02")
    raw3 = read_raw_bids(bids_path=bids_path)
    assert len(raw) == len(raw1)
    assert len(raw) == len(raw2)
    assert len(raw) > len(raw3)

    # check that split files both appear in scans.tsv
    scans_tsv = BIDSPath(
        subject=subject_id,
        session=session_id,
        suffix="scans",
        extension=".tsv",
        root=bids_root,
    )
    scan_data = _from_tsv(scans_tsv)
    scan_fnames = scan_data["filename"]
    scan_acqtime = scan_data["acq_time"]

    assert len(scan_fnames) == 3
    assert "split-01" in scan_fnames[0] and "split-02" in scan_fnames[1]
    # check that the acq_times in scans.tsv are the same
    assert scan_acqtime[0] == scan_acqtime[1]
    # check the recordings are in the correct order
    assert raw2.first_time < raw3.first_time

    # check whether non-matching acq_times are caught
    scan_data["acq_time"][0] = scan_acqtime[0].split(".")[0]
    _to_tsv(scan_data, scans_tsv)
    with pytest.raises(ValueError, match="Split files must have the same acq_time."):
        read_raw_bids(bids_path)

    # reset scans.tsv file for downstream tests
    scan_data["acq_time"][0] = scan_data["acq_time"][1]
    _to_tsv(scan_data, scans_tsv)


@pytest.mark.filterwarnings(warning_str["channel_unit_changed"])
@testing.requires_testing_data
def test_ignore_exclude_param(tmp_path):
    """Test that extra_params=dict(exclude=...) is being ignored."""
    bids_path = _bids_path.copy().update(root=tmp_path)
    ch_name = "EEG 001"
    raw = _read_raw_fif(raw_fname, verbose=False)
    write_raw_bids(raw, bids_path=bids_path, overwrite=True, verbose=False)

    raw = read_raw_bids(
        bids_path=bids_path, verbose=False, extra_params=dict(exclude=[ch_name])
    )
    assert ch_name in raw.ch_names


@testing.requires_testing_data
def test_read_raw_bids_respects_verbose(tmp_path, caplog):
    """Ensure ``verbose=False`` suppresses info-level logging."""
    bids_path = _bids_path.copy().update(root=tmp_path, datatype="meg")
    raw = _read_raw_fif(raw_fname, verbose=False)
    write_raw_bids(raw, bids_path=bids_path, overwrite=True, verbose=False)

    caplog.set_level("INFO", logger="mne")
    read_raw_bids(bids_path=bids_path, verbose=False)

    info_logs = [
        record
        for record in caplog.records
        if record.levelno <= logging.INFO and record.name.startswith("mne")
    ]
    assert not info_logs


@pytest.mark.filterwarnings(warning_str["channel_unit_changed"])
@testing.requires_testing_data
def test_channels_tsv_raw_mismatch(tmp_path):
    """Test mismatch between channels in channels.tsv and raw."""
    bids_path = _bids_path.copy().update(root=tmp_path, datatype="meg", task="rest")

    # Remove one channel from the raw data without updating channels.tsv
    raw = _read_raw_fif(raw_fname, verbose=False)
    write_raw_bids(raw, bids_path=bids_path, overwrite=True, verbose=False)

    raw_path = bids_path.copy().update(extension=".fif").fpath
    raw = _read_raw(raw_path, preload=True)
    raw.drop_channels(ch_names=raw.ch_names[-1])
    raw.load_data()
    raw.save(raw_path, overwrite=True)

    with (
        pytest.warns(
            RuntimeWarning,
            match="number of channels in the channels.tsv sidecar .* "
            "does not match the number of channels in the raw data",
        ),
        pytest.warns(RuntimeWarning, match="Cannot set channel type"),
    ):
        read_raw_bids(bids_path)

    # Remame a channel in the raw data without updating channels.tsv
    # (number of channels in channels.tsv and raw remains different)
    ch_name_orig = raw.ch_names[-1]
    ch_name_new = "MEGtest"
    raw.rename_channels({ch_name_orig: ch_name_new})
    raw.save(raw_path, overwrite=True)

    with (
        pytest.warns(
            RuntimeWarning,
            match=f"Cannot set channel type for the following channels, as they "
            f"are missing in the raw data: {ch_name_orig}",
        ),
        pytest.warns(RuntimeWarning, match="The number of channels in the channels"),
    ):
        read_raw_bids(bids_path)

    # Mark channel as bad in channels.tsv and remove it from the raw data
    raw = _read_raw_fif(raw_fname, verbose=False)
    ch_name_orig = raw.ch_names[-1]
    ch_name_new = "MEGtest"

    raw.info["bads"] = [ch_name_orig]
    write_raw_bids(raw, bids_path=bids_path, overwrite=True, verbose=False)

    raw.drop_channels(raw.ch_names[-2])
    raw.rename_channels({ch_name_orig: ch_name_new})
    raw.save(raw_path, overwrite=True)

    with (
        pytest.warns(
            RuntimeWarning,
            match=f'Cannot set "bad" status for the following channels, as '
            f"they are missing in the raw data: {ch_name_orig}",
        ),
        pytest.warns(RuntimeWarning, match="The number of channels in the channels"),
        pytest.warns(RuntimeWarning, match="Cannot set channel type"),
    ):
        read_raw_bids(bids_path)

    # Test mismatched channel ordering between channels.tsv and raw
    raw = _read_raw_fif(raw_fname, verbose=False)
    write_raw_bids(raw, bids_path=bids_path, overwrite=True, verbose=False)

    ch_names_orig = raw.ch_names.copy()
    ch_names_new = ch_names_orig.copy()
    ch_names_new[1], ch_names_new[0] = ch_names_new[0], ch_names_new[1]
    raw.reorder_channels(ch_names_new)
    raw.save(raw_path, overwrite=True)

    raw = read_raw_bids(bids_path, on_ch_mismatch="reorder")
    assert raw.ch_names == ch_names_orig


@testing.requires_testing_data
def test_file_not_found(tmp_path):
    """Check behavior if the requested file cannot be found."""
    # First a path with a filename extension.
    bp = BIDSPath(
        root=tmp_path,
        subject="foo",
        task="bar",
        datatype="eeg",
        suffix="eeg",
        extension=".fif",
    )
    bp.fpath.parent.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="File does not exist"):
        read_raw_bids(bids_path=bp)

    # Now without an extension
    bp.extension = None
    with pytest.raises(FileNotFoundError, match="File does not exist"):
        read_raw_bids(bids_path=bp)

    bp.update(extension=".fif")
    _read_raw_fif(raw_fname, verbose=False).save(bp.fpath)
    with (
        pytest.warns(RuntimeWarning, match=r"channels\.tsv"),
        pytest.warns(RuntimeWarning, match=r"Did not find any eeg\.json"),
        pytest.warns(RuntimeWarning, match=r"participants\.tsv file not found"),
    ):
        read_raw_bids(bp)  # smoke test


@pytest.mark.filterwarnings(warning_str["channel_unit_changed"])
def test_gsr_and_temp_reading():
    """Test GSR and temperature channels are handled correctly."""
    bids_path = BIDSPath(
        subject="01", session="eeg", task="rest", datatype="eeg", root=tiny_bids_root
    )
    raw = read_raw_bids(bids_path)
    assert raw.get_channel_types(["GSR"]) == ["gsr"]
    assert raw.get_channel_types(["Temperature"]) == ["temperature"]


def _setup_nirs_channel_mismatch(tmp_path):
    ch_order_snirf = ["S1_D1 760", "S1_D2 760", "S1_D1 850", "S1_D2 850"]
    ch_types = ["fnirs_cw_amplitude"] * len(ch_order_snirf)
    info = mne.create_info(ch_order_snirf, sfreq=10, ch_types=ch_types)
    data = np.arange(len(ch_order_snirf) * 10.0).reshape(len(ch_order_snirf), 10)
    raw = mne.io.RawArray(data, info)

    for i, ch_name in enumerate(raw.ch_names):
        loc = np.zeros(12)
        if "S1" in ch_name:
            loc[3:6] = np.array([0, 0, 0])
        if "D1" in ch_name:
            loc[6:9] = np.array([1, 0, 0])
        elif "D2" in ch_name:
            loc[6:9] = np.array([0, 1, 0])
        loc[9] = int(ch_name.split(" ")[1])
        loc[0:3] = (loc[3:6] + loc[6:9]) / 2
        raw.info["chs"][i]["loc"] = loc

    orig_name_to_loc = {
        name: raw.info["chs"][i]["loc"].copy() for i, name in enumerate(raw.ch_names)
    }
    orig_name_to_data = {
        name: raw.get_data(picks=i).copy() for i, name in enumerate(raw.ch_names)
    }

    ch_order_bids = ["S1_D1 760", "S1_D1 850", "S1_D2 760", "S1_D2 850"]
    ch_types_bids = ["NIRSCWAMPLITUDE"] * len(ch_order_bids)
    channels_dict = OrderedDict([("name", ch_order_bids), ("type", ch_types_bids)])
    channels_fname = tmp_path / "channels.tsv"
    _to_tsv(channels_dict, channels_fname)

    return (
        raw,
        ch_order_snirf,
        ch_order_bids,
        channels_fname,
        orig_name_to_loc,
        orig_name_to_data,
    )


def test_channel_mismatch_raise(tmp_path):
    """Raise error when ``on_ch_mismatch='raise'`` and names differ."""
    raw, _, _, channels_fname, _, _ = _setup_nirs_channel_mismatch(tmp_path)
    with pytest.raises(
        RuntimeError,
        match=("Channel mismatch between .*channels"),
    ):
        _handle_channels_reading(channels_fname, raw.copy(), on_ch_mismatch="raise")


def test_channel_mismatch_reorder(tmp_path):
    """Reorder channels to match ``channels.tsv`` ordering."""
    raw, _, ch_order_bids, channels_fname, orig_name_to_loc, orig_name_to_data = (
        _setup_nirs_channel_mismatch(tmp_path)
    )
    raw_out = _handle_channels_reading(channels_fname, raw, on_ch_mismatch="reorder")
    assert raw_out.ch_names == ch_order_bids
    for i, new_name in enumerate(raw_out.ch_names):
        np.testing.assert_allclose(
            raw_out.info["chs"][i]["loc"], orig_name_to_loc[new_name]
        )
        np.testing.assert_allclose(
            raw_out.get_data(picks=i), orig_name_to_data[new_name]
        )


def test_channel_mismatch_rename(tmp_path):
    """Rename channels to match ``channels.tsv`` names."""
    (
        raw,
        ch_order_snirf,
        ch_order_bids,
        channels_fname,
        orig_name_to_loc,
        orig_name_to_data,
    ) = _setup_nirs_channel_mismatch(tmp_path)
    raw_out_rename = _handle_channels_reading(
        channels_fname, raw.copy(), on_ch_mismatch="rename"
    )
    assert raw_out_rename.ch_names == ch_order_bids
    for i in range(len(ch_order_bids)):
        orig_name_at_i = ch_order_snirf[i]
        np.testing.assert_allclose(
            raw_out_rename.info["chs"][i]["loc"], orig_name_to_loc[orig_name_at_i]
        )
        np.testing.assert_allclose(
            raw_out_rename.get_data(picks=i), orig_name_to_data[orig_name_at_i]
        )


def test_channel_mismatch_invalid_option(tmp_path):
    """Invalid ``on_ch_mismatch`` value should raise ``ValueError``."""
    raw, _, _, channels_fname, _, _ = _setup_nirs_channel_mismatch(tmp_path)
    with pytest.raises(ValueError, match="on_ch_mismatch must be one of"):
        _handle_channels_reading(channels_fname, raw.copy(), on_ch_mismatch="invalid")


@pytest.mark.filterwarnings(warning_str["channel_unit_changed"])
def test_channel_units_from_tsv(tmp_path):
    """Test that channel units are correctly read from channels.tsv."""
    pytest.importorskip("edfio")

    # Create synthetic raw data with EEG and misc channels
    ch_names = ["EEG1", "EEG2", "MISC_RAD"]
    ch_types = ["eeg", "eeg", "misc"]
    info = mne.create_info(ch_names, sfreq=256, ch_types=ch_types)
    data = np.zeros((len(ch_names), 256))
    raw = mne.io.RawArray(data, info)
    raw.set_meas_date(datetime(2020, 1, 1, tzinfo=UTC))
    raw.info["line_freq"] = 60

    raw.set_annotations(mne.Annotations(onset=[0], duration=[1], description=["test"]))

    # Set the misc channel unit to radians before writing
    raw.info["chs"][2]["unit"] = FIFF.FIFF_UNIT_RAD

    # Write to BIDS as EDF
    bids_root = tmp_path / "bids"
    bids_path = _bids_path.copy().update(
        root=bids_root, datatype="eeg", suffix="eeg", extension=".edf"
    )
    with pytest.warns(RuntimeWarning, match="Converting data files to EDF format"):
        write_raw_bids(raw, bids_path, overwrite=True, allow_preload=True, format="EDF")

    # Check that channels.tsv contains "rad" for the misc channel
    channels_fname = _find_matching_sidecar(
        bids_path, suffix="channels", extension=".tsv"
    )
    channels_tsv = _from_tsv(channels_fname)
    ch_names_tsv = channels_tsv["name"]
    units_tsv = channels_tsv["units"]
    misc_idx = ch_names_tsv.index("MISC_RAD")
    assert units_tsv[misc_idx] == "rad", (
        f"Expected 'rad' in channels.tsv for MISC_RAD, got '{units_tsv[misc_idx]}'"
    )

    # Read back and verify units are set correctly
    raw_read = read_raw_bids(bids_path)

    # Verify the misc channel has radians unit after reading
    misc_ch_idx = raw_read.ch_names.index("MISC_RAD")
    assert raw_read.info["chs"][misc_ch_idx]["unit"] == FIFF.FIFF_UNIT_RAD


def test_events_file_to_annotation_kwargs(tmp_path):
    """Test that events file is read correctly."""
    bids_path = BIDSPath(
        subject="01", session="eeg", task="rest", datatype="eeg", root=tiny_bids_root
    )
    events_fname = _find_matching_sidecar(bids_path, suffix="events", extension=".tsv")

    # ---------------- plain read --------------------------------------------
    df = pd.read_csv(events_fname, sep="\t")
    ev_kwargs = events_file_to_annotation_kwargs(events_fname=events_fname)

    np.testing.assert_equal(ev_kwargs["onset"], df["onset"].values)
    np.testing.assert_equal(ev_kwargs["duration"], df["duration"].values)
    np.testing.assert_equal(ev_kwargs["description"], df["trial_type"].values)

    # ---------------- filtering out n/a values ------------------------------
    tmp_tsv_file = tmp_path / "events.tsv"
    dext = pd.concat(
        [df.copy().assign(onset=df.onset + i) for i in range(5)]
    ).reset_index(drop=True)

    dext = dext.assign(
        ix=range(len(dext)),
        value=dext.trial_type.map({"start_experiment": 1, "show_stimulus": 2}),
        duration=1.0,
    )

    # nan values for `_drop` must be string values, `_drop` is called on
    # `onset`, `value` and `trial_type`. `duration` n/a should end up as float 0
    for c in ["onset", "value", "trial_type", "duration"]:
        dext[c] = dext[c].astype(str)

    dext.loc[0, "onset"] = "n/a"
    dext.loc[1, "duration"] = "n/a"
    dext.loc[4, "trial_type"] = "n/a"
    dext.loc[4, "value"] = (
        "n/a"  # to check that filtering is also applied when we drop the `trial_type`
    )
    dext.to_csv(tmp_tsv_file, sep="\t", index=False)

    ev_kwargs_filtered = events_file_to_annotation_kwargs(events_fname=tmp_tsv_file)

    dext_f = dext[
        (dext["onset"] != "n/a")
        & (dext["trial_type"] != "n/a")
        & (dext["value"] != "n/a")
    ]

    assert (ev_kwargs_filtered["onset"] == dext_f["onset"].astype(float).values).all()
    assert (
        ev_kwargs_filtered["duration"]
        == dext_f["duration"].replace("n/a", "0.0").astype(float).values
    ).all()
    assert (ev_kwargs_filtered["description"] == dext_f["trial_type"].values).all()
    assert (
        ev_kwargs_filtered["duration"][0] == 0.0
    )  # now idx=0, as first row is filtered out

    # ---------------- default if missing trial_type  ------------------------
    dext.drop(columns="trial_type").to_csv(tmp_tsv_file, sep="\t", index=False)

    ev_kwargs_default = events_file_to_annotation_kwargs(events_fname=tmp_tsv_file)
    np.testing.assert_array_equal(
        ev_kwargs_default["onset"], dext_f["onset"].astype(float).values
    )
    np.testing.assert_array_equal(
        ev_kwargs_default["duration"],
        dext_f["duration"].replace("n/a", "0.0").astype(float).values,
    )
    np.testing.assert_array_equal(
        np.sort(np.unique(ev_kwargs_default["description"])),
        np.sort(dext_f["value"].unique()),
    )
