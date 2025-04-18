"""Testing copyfile functions."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

import datetime
import os.path as op
from pathlib import Path

import mne
import pytest
from mne.datasets import testing

from mne_bids import BIDSPath
from mne_bids.copyfiles import (
    _get_brainvision_encoding,
    _get_brainvision_paths,
    copyfile_brainvision,
    copyfile_bti,
    copyfile_edf,
    copyfile_eeglab,
    copyfile_kit,
)
from mne_bids.path import _parse_ext

testing_path = testing.data_path(download=False)
base_path = Path(mne.__file__).parent / "io"


@testing.requires_testing_data
def test_get_brainvision_encoding():
    """Test getting the file-encoding from a BrainVision header."""
    data_path = base_path / "brainvision" / "tests" / "data"
    raw_fname = data_path / "test.vhdr"

    with pytest.raises(UnicodeDecodeError):
        with open(raw_fname, encoding="ascii") as f:
            f.readlines()

    enc = _get_brainvision_encoding(raw_fname)
    with open(raw_fname, encoding=enc) as f:
        f.readlines()


def test_get_brainvision_paths(tmp_path):
    """Test getting the file links from a BrainVision header."""
    data_path = base_path / "brainvision" / "tests" / "data"
    raw_fname = data_path / "test.vhdr"

    with pytest.raises(ValueError):
        _get_brainvision_paths(data_path / "test.eeg")

    # Write some temporary test files
    with open(tmp_path / "test1.vhdr", "w") as f:
        f.write("DataFile=testing.eeg")

    with open(tmp_path / "test2.vhdr", "w") as f:
        f.write("MarkerFile=testing.vmrk")

    with pytest.raises(ValueError):
        _get_brainvision_paths(tmp_path / "test1.vhdr")

    with pytest.raises(ValueError):
        _get_brainvision_paths(tmp_path / "test2.vhdr")

    # This should work
    eeg_file_path, vmrk_file_path = _get_brainvision_paths(raw_fname)
    head, tail = op.split(eeg_file_path)
    assert tail == "test.eeg"
    head, tail = op.split(vmrk_file_path)
    assert tail == "test.vmrk"


@pytest.mark.filterwarnings(
    "ignore:.*Exception ignored.*:pytest.PytestUnraisableExceptionWarning"
)
def test_copyfile_brainvision(tmp_path):
    """Test the copying of BrainVision vhdr, vmrk and eeg files."""
    bids_root = tmp_path
    data_path = base_path / "brainvision" / "tests" / "data"
    raw_fname = data_path / "test.vhdr"
    new_name = bids_root / "tested_conversion.vhdr"

    # IO error testing
    with pytest.raises(ValueError, match="Need to move data with same"):
        copyfile_brainvision(raw_fname, new_name.with_suffix(".eeg"))

    # Try to copy the file
    copyfile_brainvision(raw_fname, new_name)

    # Have all been copied?
    for ext in [".vhdr", ".vmrk", ".eeg"]:
        assert (new_name.parent / f"tested_conversion{ext}").exists()

    # Try to read with MNE - if this works, the links are correct
    raw = mne.io.read_raw_brainvision(new_name)
    assert Path(raw.filenames[0]) == (new_name.parent / "tested_conversion.eeg")

    # Test with anonymization
    raw = mne.io.read_raw_brainvision(raw_fname)
    prev_date = raw.info["meas_date"]
    anonymize = {"daysback": 32459}
    copyfile_brainvision(raw_fname, new_name, anonymize)
    raw = mne.io.read_raw_brainvision(new_name)
    new_date = raw.info["meas_date"]
    assert new_date == (prev_date - datetime.timedelta(days=32459))


def test_copyfile_edf(tmp_path):
    """Test the anonymization of EDF/BDF files."""
    bids_root = tmp_path / "bids1"
    bids_root.mkdir()
    data_path = base_path / "edf" / "tests" / "data"

    # Test regular copying
    for ext in [".edf", ".bdf"]:
        raw_fname = data_path / f"test{ext}"
        new_name = bids_root / f"test_copy{ext}"
        copyfile_edf(raw_fname, new_name)

    # IO error testing
    with pytest.raises(ValueError, match="Need to move data with same"):
        raw_fname = data_path / "test.edf"
        new_name = bids_root / "test_copy.bdf"
        copyfile_edf(raw_fname, new_name)

    # Add some subject info to an EDF to test anonymization
    testfile = bids_root / "test_copy.edf"
    raw_date = mne.io.read_raw_edf(testfile).info["meas_date"]
    date = datetime.datetime.strftime(raw_date, "%d-%b-%Y").upper()
    test_id_info = "023 F 02-AUG-1951 Jane"
    test_rec_info = f"Startdate {date} ID-123 John BioSemi_ActiveTwo"
    with open(testfile, "r+b") as f:
        f.seek(8)
        f.write(bytes(test_id_info.ljust(80), "ascii"))
        f.write(bytes(test_rec_info.ljust(80), "ascii"))

    # Test date anonymization
    def _edf_get_real_date(fpath):
        with open(fpath, "rb") as f:
            f.seek(88)
            rec_info = f.read(80).decode("ascii").rstrip()
        startdate = rec_info.split(" ")[1]
        return datetime.datetime.strptime(startdate, "%d-%b-%Y")

    bids_root2 = tmp_path / "bids2"
    bids_root2.mkdir()
    infile = bids_root / "test_copy.edf"
    outfile = bids_root2 / "test_copy_anon.edf"
    anonymize = {"daysback": 33459, "keep_his": False}
    copyfile_edf(infile, outfile, anonymize)
    new_date = _edf_get_real_date(outfile)

    # new anonymized date should be the minimum in EDF spec
    # (i.e. 01-01-1985)
    assert new_date == datetime.datetime(year=1985, month=1, day=1)

    # Test full ID info anonymization
    anon_startdate = datetime.datetime.strftime(new_date, "%d-%b-%Y").upper()
    with open(outfile, "rb") as f:
        f.seek(8)
        id_info = f.read(80).decode("ascii").rstrip()
        rec_info = f.read(80).decode("ascii").rstrip()
    rec_info_tmp = "Startdate {0} X mne-bids_anonymize X"
    assert id_info == "0 X X X"
    assert rec_info == rec_info_tmp.format(anon_startdate)

    # Test partial ID info anonymization
    outfile2 = bids_root2 / "test_copy_anon_partial.edf"
    anonymize = {"daysback": 33459, "keep_his": True}
    copyfile_edf(infile, outfile2, anonymize)
    with open(outfile2, "rb") as f:
        f.seek(8)
        id_info = f.read(80).decode("ascii").rstrip()
        rec_info = f.read(80).decode("ascii").rstrip()
    rec = f"Startdate {anon_startdate} ID-123 John BioSemi_ActiveTwo"
    assert id_info == "023 F X X"
    assert rec_info == rec


def test_copyfile_edfbdf_uppercase(tmp_path):
    """Test the copying of EDF/BDF files with upper-case extension."""
    bids_root = tmp_path / "bids1"
    bids_root.mkdir()
    data_path = base_path / "edf" / "tests" / "data"

    # Test regular copying
    for ext in [".edf", ".bdf"]:
        raw_fname = data_path / f"test{ext}"
        new_name = bids_root / f"test_copy{ext.upper()}"

        with pytest.warns(RuntimeWarning, match="Upper-case extension"):
            copyfile_edf(raw_fname, new_name)
        assert Path(new_name).with_suffix(ext).exists()


@pytest.mark.parametrize(
    "fname", ("test_raw.set", "test_raw_chanloc.set", "test_raw_2021.set")
)
@testing.requires_testing_data
def test_copyfile_eeglab(tmp_path, fname):
    """Test the copying of EEGlab set and fdt files."""
    bids_root = tmp_path
    data_path = testing_path / "EEGLAB"
    raw_fname = data_path / fname
    new_name = bids_root / f"CONVERTED_{fname}.set"

    # IO error testing
    with pytest.raises(ValueError, match="Need to move data with same ext"):
        copyfile_eeglab(raw_fname, new_name.with_suffix(".wrong"))

    # Test copying and reading
    copyfile_eeglab(raw_fname, new_name)
    if fname == "test_raw_chanloc.set":  # combined set+fdt
        with pytest.warns(RuntimeWarning, match="The data contains 'boundary' events"):
            raw = mne.io.read_raw_eeglab(new_name)
            assert "Fp1" in raw.ch_names
    else:  # combined set+fdt and single set (new EEGLAB format)
        raw = mne.io.read_raw_eeglab(new_name, preload=True)
        assert "EEG 001" in raw.ch_names
    assert isinstance(raw, mne.io.BaseRaw)


def test_copyfile_kit(tmp_path):
    """Test copying and renaming KIT files to a new location."""
    output_path = str(tmp_path)
    data_path = base_path / "kit" / "tests" / "data"
    raw_fname = data_path / "test.sqd"
    hpi_fname = data_path / "test_mrk.sqd"
    electrode_fname = data_path / "test.elp"
    headshape_fname = data_path / "test.hsp"
    subject_id = "01"
    session_id = "01"
    run = "01"
    acq = "01"
    task = "testing"

    raw = mne.io.read_raw_kit(
        raw_fname, mrk=hpi_fname, elp=electrode_fname, hsp=headshape_fname
    )
    _, ext = _parse_ext(raw_fname)

    datatype = "meg"  # copyfile_kit makes the same assumption

    bids_path = BIDSPath(
        subject=subject_id, session=session_id, run=run, acquisition=acq, task=task
    )
    kit_bids_path = bids_path.copy().update(
        acquisition=None, datatype=datatype, root=output_path
    )
    bids_fname = (
        bids_path.copy()
        .update(datatype=datatype, suffix=datatype, extension=ext, root=output_path)
        .fpath
    )

    copyfile_kit(
        raw_fname, bids_fname, subject_id, session_id, task, run, raw._init_kwargs
    )
    assert bids_fname.exists()
    _, ext = _parse_ext(hpi_fname)
    if ext == ".sqd":
        kit_bids_path.update(suffix="markers", extension=".sqd")
        assert kit_bids_path.fpath.exists()
    elif ext == ".mrk":
        kit_bids_path.update(suffix="markers", extension=".mrk")
        assert kit_bids_path.fpath.exists()

    if electrode_fname.exists():
        task, run, key = None, None, "ELP"
        elp_ext = ".pos"
        elp_fname = BIDSPath(
            subject=subject_id,
            session=session_id,
            task=task,
            run=run,
            acquisition=key,
            suffix="headshape",
            extension=elp_ext,
            datatype="meg",
            root=output_path,
        )
        assert elp_fname.fpath.exists()

    if headshape_fname.exists():
        task, run, key = None, None, "HSP"
        hsp_ext = ".pos"
        hsp_fname = BIDSPath(
            subject=subject_id,
            session=session_id,
            task=task,
            run=run,
            acquisition=key,
            suffix="headshape",
            extension=hsp_ext,
            datatype="meg",
            root=output_path,
        )
        assert hsp_fname.fpath.exists()


@pytest.mark.parametrize("dataset", ("4Dsim", "erm_HFH"))
@testing.requires_testing_data
def test_copyfile_bti(tmp_path, dataset):
    """Test copying and renaming BTi files to a new location."""
    pdf_fname = testing_path / "BTi" / dataset / "c,rfDC"
    kwargs = dict(head_shape_fname=None) if dataset == "erm_HFH" else dict()
    raw = mne.io.read_raw_bti(pdf_fname, **kwargs, verbose=False)

    dest = tmp_path / dataset
    copyfile_bti(raw=raw, dest=dest)

    mne.io.read_raw_bti(dest / "c,rfDC", **kwargs, verbose=False)
