import json
import re
from warnings import warn

import mne
import numpy as np
from mne.preprocessing.eyetracking import set_channel_types_eyetrack
from mne.utils import _validate_type, logger


def _has_eyetracking(bids_path):
    directory = bids_path.directory
    phys_files = directory.glob("*_physio.tsv")
    try:
        phys_tsv = next(phys_files)
    except StopIteration:
        return False
    phys_json = phys_tsv.with_suffix(".json")
    phys_type = _get_physio_type(phys_json)
    return True if phys_type == "eyetrack" else False

def _get_physio_type(physio_json_fpath):
    """Return the type of data that is stored in a <match>_physio.tsv file.

    https://bids-specification.readthedocs.io/en/latest/glossary.html#physiotypegeneric-enums
    """  # noqa: E501
    fpath = physio_json_fpath  # shorter
    contents = json.loads(fpath.read_text())
    physio_type = contents.get("PhysioType", None)  # e.g. "eyetracking"

    if not physio_type:
        warn(
            "Expected a key labeled 'PhysioType', with a value such as 'eyetrack', but "
            f"none exists. Falling back to 'Generic':\n  Files: {fpath.name}.<tsv|json>"
        )
        physio_type = "Generic"
    _validate_type(physio_type, str, "physio_type")
    return physio_type


def read_raw_eyetracking_bids(bpath, *, ch_types: dict[str, str]):
    ch_info = {
        ch_name: {"ch_type": ch_type}
        for ch_name, ch_type in ch_types.items()
    }
    supported_units = {
        "eyegaze": ('au', 'px', 'deg', 'rad'),
        "pupil": ('au', 'mm', 'm'),
    }

    raw_path = bpath.fpath


    num_eyes = 1
    eye = re.search(r"_recording-(eye[12])", raw_path.name).group(1)
    if eye == "eye1":
        # Is there an eye2?
        eye2_name = raw_path.name.replace("recording-eye1", "recording-eye2")
        eye2_fpath = raw_path.parent / eye2_name
        if eye2_fpath.exists():
            num_eyes += 1
    logger.info(f"Reading data recorded from {num_eyes} eye(s)")

    eye1_json = raw_path.with_suffix(".json")
    eye1_dict = json.loads(eye1_json.read_text())

    try:
        eye1_cols = eye1_dict["Columns"]
    except KeyError:
        eye1_cols = eye1_dict["columns"]
    eye1_eye = eye1_dict["RecordedEye"]

    for ii, ch_name in enumerate(eye1_cols[1:]):
        unit = eye1_dict[ch_name]["Units"]
        ch_info[ch_name]["eye"] = eye1_eye
        if ii == 0:
            ch_info[ch_name]["axis"] = "x"
            if unit in ("au", "a.u"):
                ch_info[ch_name]["unit"] = "px"
            else:
                assert unit in supported_units
                ch_info[ch_name]["unit"] = eye1_dict[ch_name]["Units"]
        elif ii == 1:
            ch_info[ch_name]["axis"] = "y"
            if unit in ("au", "a.u"):
                ch_info[ch_name]["unit"] = "px"
            else:
                assert unit in supported_units
                ch_info[ch_name]["unit"] = eye1_dict[ch_name]["Units"]
        else:
            unit = eye1_dict[ch_name]["Units"]
            if unit == "a.u":
                unit = "au"
            ch_info[ch_name]["unit"] = unit
    # first columns is always 'time'
    n_cols = len(eye1_cols)

    eye1_array = np.loadtxt(raw_path, usecols=range(1, n_cols))

    if num_eyes == 2:
        eye2_json = eye2_fpath.with_suffix(".json")
        eye2_dict = json.loads(eye2_json.read_text())

        try:
            eye2_cols = eye2_dict["Columns"]
        except KeyError:
            eye2_cols = eye2_dict["columns"]
        eye2_eye = eye2_dict["RecordedEye"]
        for ii, ch_name in enumerate(eye2_cols[1:]):
            unit = eye2_dict[ch_name]["Units"]
            ch_info[ch_name]["eye"] = eye2_eye
            if ii == 0:
                ch_info[ch_name]["axis"] = "x"
                if unit in ("au", "a.u"):
                    ch_info[ch_name]["unit"] = "px"
                else:
                    assert unit in supported_units
                    ch_info[ch_name]["unit"] = eye2_dict[ch_name]["Units"]
            elif ii == 1:
                ch_info[ch_name]["axis"] = "y"
                if unit in ("au", "a.u"):
                    ch_info[ch_name]["unit"] = "px"
                else:
                    assert unit in supported_units
                    ch_info[ch_name]["unit"] = eye2_dict[ch_name]["Units"]
            else:
                unit =  eye2_dict[ch_name]["Units"]
                if unit == "a.u":
                    unit = "au"
                ch_info[ch_name]["unit"] = unit

        n_cols = len(eye2_cols)
        eye2_array = np.loadtxt(eye2_fpath, usecols=range(1, n_cols))

        data = np.concat([eye1_array, eye2_array], axis=1)
        ch_names = eye1_cols[1:] + eye2_cols[1:]
        types = [ch_info[name]["ch_type"] for name in ch_info]
    else:
        data = eye1_array
        ch_names = eye1_cols[1:]
        types = [ch_info[name]["ch_type"] for name in ch_info]

    sfreq = eye1_dict["SamplingFrequency"]
    info = mne.create_info(ch_names=ch_names, ch_types=types, sfreq=sfreq)
    raw = mne.io.RawArray(data.T, info)

    et_info = dict()
    for this_name, this_type in zip(raw.ch_names, raw.get_channel_types()):
        if this_type == "eyegaze":
            et_info[this_name] = (
                this_type,
                ch_info[this_name]["unit"],
                ch_info[this_name]["eye"],
                ch_info[this_name]["axis"]
                )
        elif this_type == "pupil":
            et_info[this_name] = (
                this_type,
                ch_info[this_name]["unit"],
                ch_info[this_name]["eye"],
                )
    set_channel_types_eyetrack(raw, mapping=et_info)
    return raw




