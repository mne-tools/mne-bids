"""Code to facilitate I/O of BIDS compliant eyetracking data (BEP 020)."""

import json
from pathlib import Path

import mne
import numpy as np
from mne._fiff.constants import FIFF
from mne.preprocessing.eyetracking import set_channel_types_eyetrack
from mne.utils import _validate_type, logger

from mne_bids.config import UNITS_BIDS_TO_FIFF_MAP
from mne_bids.physio._utils import _get_physio_type


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


def _get_eyetrack_ch_names(raw):
    """Check if the raw object contains eyetracking data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw object.

    Returns
    -------
    list
        A list with the names of the eyetracking channels, if any.
    """
    _validate_type(raw, mne.io.BaseRaw, item_name="raw")
    ch_types = raw.get_channel_types()
    eye_chs = [
        ch
        for ch, ch_type in zip(raw.ch_names, ch_types)
        if ch_type in ["eyegaze", "pupil"]
    ]
    return eye_chs


def _write_single_eye_physio(
    *, raw, bids_path, eye_chs, eye_recording_tag, recorded_eye, overwrite
):
    """Write TSV, JSON, and physioevents for a single eye.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw data.
    bids_path : mne_bids.BIDSPath
        The BIDSPath object.
    eye_chs : list of str
        Channel names corresponding to this eye.
    eye_recording_tag : str
        Recording entity value (e.g., "eye1" or "eye2").
    recorded_eye : str
        "left" or "right".
    overwrite : bool
        Whether to overwrite existing files.
    """
    times = raw.times
    data = raw.get_data(picks=eye_chs)

    # Write physio TSV
    fname_tsv = (
        bids_path.copy()
        .update(
            recording=eye_recording_tag,
            suffix="physio",
            extension=".tsv",
        )
        .fpath
    )
    _write_physio_tsv(times, data, fname_tsv, overwrite)

    # Build and write sidecar JSON
    json_dict = {
        "SamplingFrequency": raw.info["sfreq"],
        "StartTime": times[0],
        "Columns": ["time"] + eye_chs,
        "PhysioType": "eyetrack",
        "RecordedEye": recorded_eye,
        "SampleCoordinateSystem": "gaze-on-screen",
        "time": {
            "Description": "The timestamp of the data, in seconds.",
            "Units": "s",
        },
    }
    # Add per-channel descriptions when available (x, y, pupil).
    if len(eye_chs) >= 1:
        json_dict[eye_chs[0]] = {
            "Description": "The x-coordinate of the gaze on the screen in pixels.",
            "Units": "pixel",
        }
    if len(eye_chs) >= 2:
        json_dict[eye_chs[1]] = {
            "Description": "The y-coordinate of the gaze on the screen in pixels.",
            "Units": "pixel",
        }
    if len(eye_chs) >= 3:
        json_dict[eye_chs[2]] = {
            "Description": (
                "Pupil area of the recorded eye as calculated by the eye-tracker "
                "in arbitrary units"
            ),
            "Units": "arbitrary",
        }

    fname_json = (
        bids_path.copy()
        .update(
            recording=eye_recording_tag,
            suffix="physio",
            extension=".json",
        )
        .fpath
    )
    _write_physio_json(json_dict, fname_json, overwrite)

    # Write physioevents TSV
    fname_events = (
        bids_path.copy()
        .update(
            recording=eye_recording_tag,
            suffix="physioevents",
            extension=".tsv",
            check=False,  # physioevents is not an allowed suffix
        )
        .fpath
    )
    _write_eyetrack_events_tsv(raw=raw, fname_tsv=fname_events, overwrite=overwrite)


def _write_eyetrack_tsvs(raw, bids_path, overwrite, calibration=None):
    """Write eyetracking physio files (per-eye TSV, JSON, and physioevents)."""
    logger.info("Writing eyetracking data to physio.tsv files.")
    # Write the physio files to the modality that eyetracking was collected with.
    datatype = bids_path.datatype
    if datatype is None:
        raise ValueError("datatype must be specified in the BIDSPath object.")
    # Find the eyetracking channels
    info_array = np.array([raw.ch_names, raw.get_channel_types()]).T
    eyegaze_ch_idx = np.where(info_array[:, 1] == "eyegaze")[0]
    pupil_ch_idx = np.where(info_array[:, 1] == "pupil")[0]
    assert len(eyegaze_ch_idx)
    assert len(pupil_ch_idx)
    # What eyes were recorded.
    left_eye_chs = []
    right_eye_chs = []
    for idx in np.concatenate([eyegaze_ch_idx, pupil_ch_idx]):
        # index 3 the loc array specifies left/right eye
        which_eye = raw.info["chs"][idx]["loc"][3]
        if which_eye == -1:
            left_eye_chs.append(raw.ch_names[idx])
        elif which_eye == 1:
            right_eye_chs.append(raw.ch_names[idx])
        else:
            raise ValueError(
                "A raw object with eyetrack channels must specify the eye that each "
                "channel corresponds to in raw.info['chs'][channel_index]['loc'][3]. "
                "This value must be -1 for the left eye, or 1 for the right eye. "
                f"Got {which_eye}."
            )
    # If we have data for both eyes, left eye is eye1 and right eye is eye2
    if all([len(left_eye_chs) and len(right_eye_chs)]):
        eye1_chs = left_eye_chs
        eye2_chs = right_eye_chs
        recorded_eye_1 = "left"
        recorded_eye_2 = "right"
    # Otherwise, if we only have data for one eye, that eye is eye1
    elif len(left_eye_chs):
        eye1_chs = left_eye_chs
        eye2_chs = []
        recorded_eye_1 = "left"
    elif len(right_eye_chs):
        eye1_chs = right_eye_chs
        eye2_chs = []
        recorded_eye_1 = "right"
    # Write the *_physio.tsv/.json and *_physioevents.tsv files for each eye
    if eye1_chs:
        _write_single_eye_physio(
            raw=raw,
            bids_path=bids_path,
            eye_chs=eye1_chs,
            eye_recording_tag="eye1",
            recorded_eye=recorded_eye_1,
            overwrite=overwrite,
        )
    if eye2_chs:
        _write_single_eye_physio(
            raw=raw,
            bids_path=bids_path,
            eye_chs=eye2_chs,
            eye_recording_tag="eye2",
            recorded_eye=recorded_eye_2,
            overwrite=overwrite,
        )


def _write_eyetrack_events_tsv(*, raw, fname_tsv, overwrite):
    """Write a <match>_physioevents.tsv file."""
    from mne_bids.write import _events_json, _events_tsv

    raw = raw.copy()
    annotations = raw.annotations.copy()
    if "BAD_blink" in annotations.description:
        annotations.rename({"BAD_blink": "blink"})
    raw.set_annotations(annotations)
    eye_annot_indices = []
    # Get the names of eyetracking channels
    eye_ch_names = [
        ch_name
        for ch_name, ch_type in zip(raw.ch_names, raw.get_channel_types())
        if ch_type in ["eyegaze", "pupil"]
    ]
    # Get the indices of the annotations that contain eyetracking channels
    for annot_idx, this_annot in enumerate(annotations):
        if any([ch_name in this_annot["ch_names"] for ch_name in eye_ch_names]):
            eye_annot_indices.append(annot_idx)
    if len(eye_annot_indices) == 0:
        raise ValueError("No eyetracking annotations found.")
    # Get the descriptions of the eyetracking annotations
    eye_annotations = annotations[eye_annot_indices]
    descriptions = eye_annotations.description
    durations = eye_annotations.duration
    # Use mne.events_from_annotations to convert the annotations to events
    unique_descriptions = np.unique(descriptions)
    event_ids = {desc: ii for ii, desc in enumerate(unique_descriptions, start=1)}
    events, event_id = mne.events_from_annotations(raw, event_id=event_ids)
    # Let's use the _events_tsv function to write the file.
    assert len(durations) == len(events)
    _events_tsv(
        events=events,
        durations=durations,
        raw=raw,
        fname=fname_tsv,
        trial_type=event_id,
        event_metadata=None,
        include_column_names=False,
        overwrite=overwrite,
    )
    # Write the JSON file
    fname_json = fname_tsv.with_suffix(".json")
    _events_json(
        fname_json, extra_columns=None, has_trial_type=True, overwrite=overwrite
    )


def _write_physio_tsv(times, data, fname, overwrite):
    """Write a *_physio.tsv file.

    Parameters
    ----------
    time : np.ndarray
        The time.
    data : np.ndarray
        The data.
    fname : str
        The file name.
    overwrite : bool
        Whether to overwrite existing files.
    """
    # Check for overwrite
    if Path(fname).exists() and not overwrite:
        raise FileExistsError(
            f"{fname} already exists. Set overwrite=True to overwrite."
        )
    # Check the data
    if data.shape[1] != len(times):
        raise ValueError("Data and time must have the same length.")
    # put the times and data into a numpy array
    times = np.array(times)  # in seconds
    eye_data = np.array(data)
    data = np.vstack((times, eye_data)).T
    # Write the file
    np.savetxt(fname, data, delimiter="\t", fmt="%1.3f", encoding="utf-8")


def _write_physio_json(json_dict, fname, overwrite):
    """Write a *_physio.json file.

    Parameters
    ----------
    json_dict : dict
        The JSON dictionary.
    fname : str
        The file name.
    overwrite : bool
        Whether to overwrite existing files.
    """
    # Check for overwrite
    if Path(fname).exists() and not overwrite:
        raise FileExistsError(
            f"{fname} already exists. Set overwrite=True to overwrite."
        )
    # Write the file
    with open(fname, "w") as f:
        json.dump(json_dict, f, indent=4)


def read_raw_eyetracking_bids(bids_path, *, ch_types: dict[str, str]):
    """Read BIDS compliant eyetracking data from TSV sidecar files.

    bids_path : mne_bids.BIDSPath
        the BIDSPath instance that points to the ``<match>_recording-eye1_physio.tsv``
        eyetracking file. You must specify the following entities in the BIDSPath
        constructor: ``recording="eye1"``, ``suffix="physio"``, ``extension=".tsv"``.
        If the eyetracking data was recorded without another modality, you must also
        specify ``datatype="beh"``. Otherwise, if the eyetracking was collected
        alongside another modality such as ``eeg``, then you must specify
        ``datatype="eeg"``.
    ch_types : dict of str
        a dictionary whose keys correspond to eyetracking channel names, and whose
        values correspond to the MNE-Python compatible channel types for said channel,
        such as ``'eyegaze'``, ``'pupil'``, or ``'misc'``.
    """
    ch_info = {ch_name: {"ch_type": ch_type} for ch_name, ch_type in ch_types.items()}
    fiff_to_func = {
        FIFF.FIFF_UNIT_PX: "px",
        FIFF.FIFF_UNIT_NONE: "au",
        FIFF.FIFF_UNIT_M: "m",
        FIFF.FIFF_UNIT_RAD: "rad",
    }

    raw_path = bids_path.fpath

    num_eyes = 1
    eye = bids_path.recording
    if not eye:
        raise RuntimeError(
            "To read eyetracking data, you must specify a recording entity in the "
            "BIDSPath constructor, e.g. recording='eye1'."
        )
    if eye == "eye1":
        # Is there an eye2?
        eye2_name = raw_path.name.replace("recording-eye1", "recording-eye2")
        eye2_fpath = raw_path.parent / eye2_name
        if eye2_fpath.exists():
            num_eyes += 1
    logger.info(f"Reading data recorded from {num_eyes} eye(s)")

    eye1_json = raw_path.with_suffix(".json")
    eye1_dict = json.loads(eye1_json.read_text())

    eye1_cols = eye1_dict["Columns"]
    eye1_eye = eye1_dict["RecordedEye"]

    for ii, ch_name in enumerate(eye1_cols[1:]):
        unit = eye1_dict[ch_name]["Units"]
        ch_info[ch_name]["eye"] = eye1_eye
        ch_info[ch_name]["unit"] = UNITS_BIDS_TO_FIFF_MAP[unit]
        if ii == 0:
            ch_info[ch_name]["axis"] = "x"
        elif ii == 1:
            ch_info[ch_name]["axis"] = "y"
        else:
            ch_info[ch_name]["axis"] = None
    # first columns is always 'time'
    n_cols = len(eye1_cols)

    eye1_array = np.loadtxt(raw_path, usecols=range(1, n_cols))

    if num_eyes == 2:
        eye2_json = eye2_fpath.with_suffix(".json")
        eye2_dict = json.loads(eye2_json.read_text())

        eye2_cols = eye2_dict["Columns"]
        eye2_eye = eye2_dict["RecordedEye"]
        for ii, ch_name in enumerate(eye2_cols[1:]):
            unit = eye2_dict[ch_name]["Units"]
            ch_info[ch_name]["eye"] = eye2_eye
            ch_info[ch_name]["unit"] = UNITS_BIDS_TO_FIFF_MAP[unit]
            if ii == 0:
                ch_info[ch_name]["axis"] = "x"
            elif ii == 1:
                ch_info[ch_name]["axis"] = "y"
            else:
                ch_info[ch_name]["axis"] = None

        n_cols = len(eye2_cols)
        eye2_array = np.loadtxt(eye2_fpath, usecols=range(1, n_cols))

        data = np.concat([eye1_array, eye2_array], axis=1)
        ch_names = eye1_cols[1:] + eye2_cols[1:]
        types = [ch_info[name]["ch_type"] for name in ch_names]
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
                fiff_to_func[ch_info[this_name]["unit"]],
                ch_info[this_name]["eye"],
                ch_info[this_name]["axis"],
            )
        elif this_type == "pupil":
            et_info[this_name] = (
                this_type,
                fiff_to_func[ch_info[this_name]["unit"]],
                ch_info[this_name]["eye"],
            )
    set_channel_types_eyetrack(raw, mapping=et_info)
    return raw
