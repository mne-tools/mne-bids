"""Test the digitizations.

For each supported coordinate frame, implement a test.
"""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from pathlib import Path

import mne
import numpy as np
import pytest
from mne.datasets import testing
from numpy.testing import assert_almost_equal

import mne_bids
from mne_bids import BIDSPath, read_raw_bids, write_raw_bids
from mne_bids.config import (
    BIDS_STANDARD_TEMPLATE_COORDINATE_SYSTEMS,
    BIDS_TO_MNE_FRAMES,
    MNE_STR_TO_FRAME,
)
from mne_bids.dig import (
    _ensure_fiducials_ctf_head,
    _infer_coord_unit,
    _read_dig_bids,
    _write_dig_bids,
    convert_montage_to_mri,
    convert_montage_to_ras,
    template_to_head,
)

base_path = Path(mne.__file__).parent / "io"
subject_id = "01"
session_id = "01"
run = "01"
acq = "01"
run2 = "02"
task = "testing"

_bids_path = BIDSPath(
    subject=subject_id, session=session_id, run=run, acquisition=acq, task=task
)

data_path = testing.data_path(download=False)


def _load_raw():
    """Load the sample raw data."""
    raw_fname = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif"
    raw = mne.io.read_raw(raw_fname)
    raw.drop_channels(raw.info["bads"])
    raw.info["line_freq"] = 60
    return raw


@testing.requires_testing_data
def test_dig_io(tmp_path):
    """Test passing different coordinate frames give proper warnings."""
    bids_root = tmp_path / "bids1"
    raw = _load_raw()
    for datatype in ("eeg", "ieeg"):
        (bids_root / "sub-01" / "ses-01" / datatype).mkdir(exist_ok=True, parents=True)

    # test no coordinate frame in dig or in bids_path.space
    montage = raw.get_montage()
    montage.apply_trans(mne.transforms.Transform("head", "unknown"))
    for datatype in ("eeg", "ieeg"):
        bids_path = _bids_path.copy().update(
            root=bids_root, datatype=datatype, space=None
        )
        with pytest.warns(
            RuntimeWarning, match="Coordinate frame could not be inferred"
        ):
            _write_dig_bids(bids_path, raw, montage, acpc_aligned=True)

    # test coordinate frame-BIDSPath.space mismatch
    raw = _load_raw()
    montage = raw.get_montage()
    print(montage.get_positions()["coord_frame"])
    bids_path = _bids_path.copy().update(
        root=bids_root, datatype="eeg", space="fsaverage"
    )
    with pytest.raises(
        ValueError,
        match="Coordinates in the raw object "
        "or montage are in the CapTrak "
        "coordinate frame but "
        "BIDSPath.space is fsaverage",
    ):
        _write_dig_bids(bids_path, raw, montage)

    # test MEG space conflict fif (ElektaNeuromag) != CTF
    bids_path = _bids_path.copy().update(root=bids_root, datatype="meg", space="CTF")
    with pytest.raises(ValueError, match="conflicts"):
        write_raw_bids(raw, bids_path)


@testing.requires_testing_data
def test_dig_pixels(tmp_path):
    """Test dig stored correctly for the Pixels coordinate frame."""
    bids_root = tmp_path / "bids1"

    # test coordinates in pixels
    bids_path = _bids_path.copy().update(
        root=bids_root, datatype="ieeg", space="Pixels"
    )
    (bids_root / "sub-01" / "ses-01" / bids_path.datatype).mkdir(
        exist_ok=True, parents=True
    )
    raw = _load_raw()
    raw.pick(["eeg"])
    raw.del_proj()
    raw.set_channel_types({ch: "ecog" for ch in raw.ch_names})

    montage = raw.get_montage()
    # fake transform to pixel coordinates
    montage.apply_trans(mne.transforms.Transform("head", "unknown"))
    _write_dig_bids(bids_path, raw, montage)
    electrodes_path = bids_path.copy().update(
        task=None, run=None, suffix="electrodes", extension=".tsv"
    )
    coordsystem_path = bids_path.copy().update(
        task=None, run=None, suffix="coordsystem", extension=".json"
    )
    with pytest.warns(RuntimeWarning, match="not an MNE-Python coordinate frame"):
        _read_dig_bids(electrodes_path, coordsystem_path, bids_path.datatype, raw)
    montage2 = raw.get_montage()
    assert montage2.get_positions()["coord_frame"] == "unknown"
    assert_almost_equal(
        np.array(list(montage.get_positions()["ch_pos"].values())),
        np.array(list(montage2.get_positions()["ch_pos"].values())),
    )


@pytest.mark.filterwarnings("ignore:The unit for chann*.:RuntimeWarning:mne")
@testing.requires_testing_data
def test_dig_template(tmp_path):
    """Test that eeg and ieeg dig are stored properly."""
    bids_root = tmp_path / "bids1"
    for datatype in ("eeg", "ieeg"):
        (bids_root / "sub-01" / "ses-01" / datatype).mkdir(parents=True)

    raw = _load_raw()
    raw.pick(["eeg"])
    montage = raw.get_montage()
    pos = montage.get_positions()

    for datatype in ("eeg", "ieeg"):
        bids_path = _bids_path.copy().update(root=bids_root, datatype=datatype)
        for coord_frame in BIDS_STANDARD_TEMPLATE_COORDINATE_SYSTEMS:
            bids_path.update(space=coord_frame)
            raw.set_montage(None)
            _montage = montage.copy()
            mne_coord_frame = BIDS_TO_MNE_FRAMES.get(coord_frame, None)
            if mne_coord_frame is None:
                _montage.apply_trans(mne.transforms.Transform("head", "unknown"))
            else:
                _montage.apply_trans(mne.transforms.Transform("head", mne_coord_frame))
            _write_dig_bids(bids_path, raw, _montage, acpc_aligned=True)
            electrodes_path = bids_path.copy().update(
                task=None, run=None, suffix="electrodes", extension=".tsv"
            )
            coordsystem_path = bids_path.copy().update(
                task=None, run=None, suffix="coordsystem", extension=".json"
            )
            # _read_dig_bids updates raw inplace
            if mne_coord_frame is None:
                with pytest.warns(
                    RuntimeWarning, match="not an MNE-Python coordinate frame"
                ):
                    _read_dig_bids(electrodes_path, coordsystem_path, datatype, raw)
            else:
                if coord_frame == "MNI305":  # saved to fsaverage, same
                    electrodes_path.update(space="fsaverage")
                    coordsystem_path.update(space="fsaverage")
                _read_dig_bids(electrodes_path, coordsystem_path, datatype, raw)
            pos2 = raw.get_montage().get_positions()
            np.testing.assert_array_almost_equal(
                np.array(list(pos["ch_pos"].values())),
                np.array(list(pos2["ch_pos"].values())),
            )
            if mne_coord_frame is None:
                assert pos2["coord_frame"] == "unknown"
            else:
                assert pos2["coord_frame"] == mne_coord_frame

    # test MEG
    raw = _load_raw()
    for coord_frame in BIDS_STANDARD_TEMPLATE_COORDINATE_SYSTEMS:
        bids_path = _bids_path.copy().update(
            root=bids_root, datatype="meg", space=coord_frame
        )
        write_raw_bids(raw, bids_path)
        raw2 = read_raw_bids(bids_path)
        for ch, ch2 in zip(raw.info["chs"], raw2.info["chs"]):
            np.testing.assert_array_equal(ch["loc"], ch2["loc"])
            assert ch["coord_frame"] == ch2["coord_frame"]


def _set_montage_no_trans(raw, montage):
    """Set the montage without transforming to 'head'."""
    coord_frame = montage.get_positions()["coord_frame"]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=RuntimeWarning,
            message=".*nasion not found",
            module="mne",
        )
        raw.set_montage(montage, on_missing="ignore")
    for ch in raw.info["chs"]:
        ch["coord_frame"] = MNE_STR_TO_FRAME[coord_frame]
    for d in raw.info["dig"]:
        d["coord_frame"] = MNE_STR_TO_FRAME[coord_frame]


def _test_montage_trans(
    raw, montage, pos_test, space="fsaverage", coord_frame="auto", unit="auto"
):
    """Test if a montage is transformed correctly."""
    _set_montage_no_trans(raw, montage)
    trans = template_to_head(raw.info, space, coord_frame=coord_frame, unit=unit)[1]
    montage_test = raw.get_montage()
    montage_test.apply_trans(trans)
    assert_almost_equal(
        pos_test, np.array(list(montage_test.get_positions()["ch_pos"].values()))
    )


@testing.requires_testing_data
def test_template_to_head():
    """Test transforming a template montage to head."""
    # test no montage
    raw = _load_raw()
    raw.set_montage(None)
    with pytest.raises(RuntimeError, match="No montage found"):
        template_to_head(raw.info, "fsaverage", coord_frame="auto")

    # test no channels
    raw = _load_raw()
    montage_empty = mne.channels.make_dig_montage(hsp=[[0, 0, 0]])
    _set_montage_no_trans(raw, montage_empty)
    with pytest.raises(RuntimeError, match="No channel locations found in the montage"):
        template_to_head(raw.info, "fsaverage", coord_frame="auto")

    # test unexpected coordinate frame
    raw = _load_raw()
    with pytest.raises(RuntimeError, match="not expected for a template"):
        template_to_head(raw.info, "fsaverage", coord_frame="auto")

    # test all coordinate frames
    raw = _load_raw()
    raw.set_montage(None)
    raw.pick(["eeg"])
    raw.drop_channels(raw.ch_names[3:])
    montage = mne.channels.make_dig_montage(
        ch_pos={
            raw.ch_names[0]: [0, 0, 0],
            raw.ch_names[1]: [0, 0, 0.1],
            raw.ch_names[2]: [0, 0, 0.2],
        },
        coord_frame="unknown",
    )
    for space in BIDS_STANDARD_TEMPLATE_COORDINATE_SYSTEMS:
        for cf in ("mri", "mri_voxel", "ras"):
            _set_montage_no_trans(raw, montage)
            trans = template_to_head(raw.info, space, cf)[1]
            assert trans["from"] == MNE_STR_TO_FRAME["head"]
            assert trans["to"] == MNE_STR_TO_FRAME["mri"]
            montage_test = raw.get_montage()
            pos = montage_test.get_positions()
            assert pos["coord_frame"] == "head"
            assert pos["nasion"] is not None
            assert pos["lpa"] is not None
            assert pos["rpa"] is not None

    # test that we get the right transform
    _set_montage_no_trans(raw, montage)
    trans = template_to_head(raw.info, "fsaverage", "mri")[1]
    trans2 = mne.read_trans(
        Path(mne_bids.__file__).parent.parent
        / "mne_bids"
        / "data"
        / "space-fsaverage_trans.fif"
    )
    assert_almost_equal(trans["trans"], trans2["trans"])

    # test auto coordinate frame

    # test auto voxels
    montage_vox = mne.channels.make_dig_montage(
        ch_pos={
            raw.ch_names[0]: [2, 0, 10],
            raw.ch_names[1]: [0, 0, 5.5],
            raw.ch_names[2]: [0, 1, 3],
        },
        coord_frame="unknown",
    )
    pos_test = np.array(
        [[0.126, -0.118, 0.128], [0.128, -0.1225, 0.128], [0.128, -0.125, 0.127]]
    )
    _test_montage_trans(raw, montage_vox, pos_test, coord_frame="auto", unit="mm")

    # now negative values => scanner RAS
    montage_ras = mne.channels.make_dig_montage(
        ch_pos={
            raw.ch_names[0]: [-30.2, 20, -40],
            raw.ch_names[1]: [10, 30, 53.5],
            raw.ch_names[2]: [30, -21, 33],
        },
        coord_frame="unknown",
    )
    pos_test = np.array(
        [[-0.0302, 0.02, -0.04], [0.01, 0.03, 0.0535], [0.03, -0.021, 0.033]]
    )
    _set_montage_no_trans(raw, montage_ras)
    _test_montage_trans(raw, montage_ras, pos_test, coord_frame="auto", unit="mm")

    # test auto unit
    montage_mm = montage_ras.copy()
    _set_montage_no_trans(raw, montage_mm)
    _test_montage_trans(raw, montage_mm, pos_test, coord_frame="ras", unit="auto")

    montage_m = montage_ras.copy()
    for d in montage_m.dig:
        d["r"] = np.array(d["r"]) / 1000
    _test_montage_trans(raw, montage_m, pos_test, coord_frame="ras", unit="auto")


@testing.requires_testing_data
def test_convert_montage():
    """Test the montage RAS conversion."""
    raw = _load_raw()
    montage = raw.get_montage()
    trans = mne.read_trans(
        data_path / "MEG" / "sample" / "sample_audvis_trunc-trans.fif"
    )
    montage.apply_trans(trans)

    subjects_dir = data_path / "subjects"
    # test read
    with pytest.raises(RuntimeError, match="incorrectly formatted"):
        convert_montage_to_mri(montage, "foo", subjects_dir)

    # test write
    with pytest.raises(RuntimeError, match="incorrectly formatted"):
        convert_montage_to_ras(montage, "foo", subjects_dir)

    # test mri to ras
    convert_montage_to_ras(montage, "sample", subjects_dir)
    pos = montage.get_positions()
    assert pos["coord_frame"] == "ras"
    assert_almost_equal(pos["ch_pos"]["EEG 001"], [-0.0366405, 0.063066, 0.0676311])

    # test ras to mri
    convert_montage_to_mri(montage, "sample", subjects_dir)
    pos = montage.get_positions()
    assert pos["coord_frame"] == "mri"
    assert_almost_equal(pos["ch_pos"]["EEG 001"], [-0.0313669, 0.0540269, 0.0949191])


@testing.requires_testing_data
def test_electrodes_io(tmp_path):
    """Ensure only electrodes end up in *_electrodes.json."""
    raw = _load_raw()
    raw.pick(["eeg", "stim"])  # we don't need meg channels
    bids_root = tmp_path / "bids1"
    bids_path = _bids_path.copy().update(root=bids_root, datatype="eeg")
    write_raw_bids(raw=raw, bids_path=bids_path)

    electrodes_path = bids_path.copy().update(
        task=None, run=None, space="CapTrak", suffix="electrodes", extension=".tsv"
    )
    with open(electrodes_path, encoding="utf-8") as sidecar:
        n_entries = len(
            [line for line in sidecar if "name" not in line]
        )  # don't need the header
        # only eeg chs w/ electrode pos should be written to electrodes.tsv
        assert n_entries == len(raw.get_channel_types("eeg"))


@testing.requires_testing_data
def test_task_specific_electrodes_sidecar(tmp_path):
    """Test the optional task- entity in electrodes.tsv."""
    raw = _load_raw()
    raw.pick(["eeg"])
    raw_foo = raw.copy().load_data().pick(slice(None, len(raw.ch_names) // 2))
    raw_bar = raw.copy().load_data().pick(slice(len(raw.ch_names) // 2, None))

    bids_root1 = tmp_path / "bids1"
    bids_root2 = tmp_path / "bids2"

    bpath_kwargs = dict(
        root=bids_root1,
        subject="01",
        session="01",
        run="01",
        acquisition="01",
        task="foo",
    )

    write_kwargs = dict(
        allow_preload=True,
        electrodes_tsv_task=True,
        format="BrainVision",
    )

    bpath = mne_bids.BIDSPath(**bpath_kwargs)
    bpath_foo = mne_bids.write_raw_bids(raw_foo, bpath, **write_kwargs)

    bpath_kwargs.update(task="bar")
    bpath = mne_bids.BIDSPath(**bpath_kwargs)
    bpath_bar = mne_bids.write_raw_bids(raw_bar, bpath, **write_kwargs)

    elpath_foo = bpath_foo.find_matching_sidecar(suffix="electrodes", extension=".tsv")
    elpath_bar = bpath_bar.find_matching_sidecar(suffix="electrodes", extension=".tsv")

    assert (
        elpath_foo.name == "sub-01_ses-01_task-foo_acq-01_space-CapTrak_electrodes.tsv"
    )
    assert (
        elpath_bar.name == "sub-01_ses-01_task-bar_acq-01_space-CapTrak_electrodes.tsv"
    )

    coordpath_foo = bpath_foo.find_matching_sidecar(
        suffix="coordsystem", extension=".json"
    )
    coordpath_bar = bpath_bar.find_matching_sidecar(
        suffix="coordsystem", extension=".json"
    )

    assert coordpath_foo.name == "sub-01_ses-01_acq-01_space-CapTrak_coordsystem.json"
    assert coordpath_bar.name == "sub-01_ses-01_acq-01_space-CapTrak_coordsystem.json"

    # make sure we are reading the correct electrodes sidecar
    raw_foo_want_pos = raw_foo.get_montage().get_positions()["ch_pos"]["EEG 029"]
    raw_foo = mne_bids.read_raw_bids(bpath_foo)
    np.testing.assert_allclose(
        raw_foo.get_montage().get_positions()["ch_pos"]["EEG 029"], raw_foo_want_pos
    )

    # Now test for no task- entity in electrodes.tsv

    bpath_kwargs.update(root=bids_root2)
    write_kwargs.update(electrodes_tsv_task=False)

    bpath = mne_bids.BIDSPath(**bpath_kwargs)
    bpath_foo = mne_bids.write_raw_bids(raw_foo, bpath, **write_kwargs)

    elpath_foo = bpath_foo.find_matching_sidecar(suffix="electrodes", extension=".tsv")
    assert elpath_foo.name == "sub-01_ses-01_acq-01_space-CapTrak_electrodes.tsv"


def test_infer_coord_unit(tmp_path):
    """Test unit inference from electrode coordinate magnitudes."""
    # Test meters: typical EEG head coords have max ~0.1
    electrodes_m = tmp_path / "electrodes_m.tsv"
    electrodes_m.write_text(
        "name\tx\ty\tz\nFp1\t0.0949\t0.0307\t-0.0047\nFp2\t0.0949\t-0.0307\t-0.0047\n"
    )
    assert _infer_coord_unit(electrodes_m) == "m"

    # Test centimeters: coords ~1-10
    electrodes_cm = tmp_path / "electrodes_cm.tsv"
    electrodes_cm.write_text(
        "name\tx\ty\tz\nFp1\t9.49\t3.07\t-0.47\nFp2\t9.49\t-3.07\t-0.47\n"
    )
    assert _infer_coord_unit(electrodes_cm) == "cm"

    # Test millimeters: coords >= 100
    electrodes_mm = tmp_path / "electrodes_mm.tsv"
    electrodes_mm.write_text(
        "name\tx\ty\tz\nFp1\t104.9\t30.7\t-4.7\nFp2\t104.9\t-30.7\t-4.7\n"
    )
    assert _infer_coord_unit(electrodes_mm) == "mm"

    # Test with n/a values: should ignore them and still infer correctly
    electrodes_na = tmp_path / "electrodes_na.tsv"
    electrodes_na.write_text(
        "name\tx\ty\tz\nFp1\t0.0949\t0.0307\t-0.0047\nEMG\tn/a\tn/a\tn/a\n"
    )
    assert _infer_coord_unit(electrodes_na) == "m"

    # Test all n/a: should default to meters
    electrodes_all_na = tmp_path / "electrodes_allna.tsv"
    electrodes_all_na.write_text("name\tx\ty\tz\nEMG\tn/a\tn/a\tn/a\n")
    assert _infer_coord_unit(electrodes_all_na) == "m"

    # Test boundary: max_abs exactly 1.0 should be cm (not meters)
    electrodes_boundary_1 = tmp_path / "electrodes_b1.tsv"
    electrodes_boundary_1.write_text("name\tx\ty\tz\nFp1\t1.0\t0.0\t0.0\n")
    assert _infer_coord_unit(electrodes_boundary_1) == "cm"

    # Test boundary: max_abs exactly 100.0 should be mm (not cm)
    electrodes_boundary_100 = tmp_path / "electrodes_b100.tsv"
    electrodes_boundary_100.write_text("name\tx\ty\tz\nFp1\t100.0\t0.0\t0.0\n")
    assert _infer_coord_unit(electrodes_boundary_100) == "mm"


def test_ensure_fiducials_ctf_head():
    """Test synthesis of approximate fiducials for ctf_head montages."""
    # Create a ctf_head montage without fiducials
    ch_pos = {
        "Fp1": np.array([0.09, 0.03, 0.0]),
        "Fp2": np.array([0.09, -0.03, 0.0]),
        "Oz": np.array([-0.08, 0.0, 0.0]),
    }
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="ctf_head")
    pos = montage.get_positions()
    assert pos["nasion"] is None
    assert pos["lpa"] is None
    assert pos["rpa"] is None

    # Synthesize fiducials – should emit a warning
    with pytest.warns(RuntimeWarning, match="No fiducial points found"):
        _ensure_fiducials_ctf_head(montage)
    pos = montage.get_positions()
    assert pos["nasion"] is not None
    assert pos["lpa"] is not None
    assert pos["rpa"] is not None
    assert pos["coord_frame"] == "ctf_head"

    # In CTF/ALS: nasion along +X, LPA along +Y, RPA along -Y
    assert pos["nasion"][0] > 0  # +X
    assert pos["lpa"][1] > 0  # +Y
    assert pos["rpa"][1] < 0  # -Y

    # Test that existing fiducials are not overwritten
    montage_with_fids = mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        nasion=[0.1, 0, 0],
        lpa=[0, 0.08, 0],
        rpa=[0, -0.08, 0],
        coord_frame="ctf_head",
    )
    original_nasion = montage_with_fids.get_positions()["nasion"].copy()
    _ensure_fiducials_ctf_head(montage_with_fids)
    np.testing.assert_array_equal(
        montage_with_fids.get_positions()["nasion"], original_nasion
    )

    # Test all-NaN positions: should return without modification
    ch_pos_nan = {
        "Fp1": np.array([np.nan, np.nan, np.nan]),
        "Fp2": np.array([np.nan, np.nan, np.nan]),
    }
    montage_nan = mne.channels.make_dig_montage(
        ch_pos=ch_pos_nan, coord_frame="ctf_head"
    )
    _ensure_fiducials_ctf_head(montage_nan)
    assert montage_nan.get_positions()["nasion"] is None

    # Test non-ctf_head frame is ignored
    montage_head = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    _ensure_fiducials_ctf_head(montage_head)
    assert montage_head.get_positions()["nasion"] is None
