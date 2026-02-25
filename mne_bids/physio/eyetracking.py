"""Code to facilitate I/O of BIDS compliant eyetracking data (BEP 020)."""

import json
from pathlib import Path

import mne
import numpy as np
from mne._fiff.constants import FIFF
from mne.preprocessing.eyetracking import Calibration, set_channel_types_eyetrack
from mne.utils import _validate_type, logger

from mne_bids.config import UNITS_BIDS_TO_FIFF_MAP, UNITS_FIFF_TO_BIDS_MAP
from mne_bids.path import BIDSPath
from mne_bids.physio._utils import _get_physio_type
from mne_bids.tsv_handler import _from_compressed_tsv
from mne_bids.utils import _write_json, _write_tsv

# Parameters accepted by MNE's Calibration class
BIDS_CALIBRATION_TO_MNE = {
    "AverageCalibrationError": "avg_error",
    "MaximalCalibrationError": "max_error",
    "CalibrationType": "model",
    "CalibrationPosition": "positions",
    "CalibrationDistance": "screen_distance",
    # FIXME: Add CalibrationUnit to MNE's Calibration constructor
    "CalibrationUnit": "unit",
}
MNE_CALIBRATION_TO_BIDS = {
    bids_key: mne_key for mne_key, bids_key in BIDS_CALIBRATION_TO_MNE.items()
}


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


def _get_eyetrack_annotation_inds(raw):
    """Get indices of annotations associated with eyetracking channels."""
    _validate_type(raw, mne.io.BaseRaw, item_name="raw")
    eye_ch_names = _get_eyetrack_ch_names(raw)
    if len(eye_ch_names) == 0:
        return np.array([], dtype=int)

    return np.array(
        [
            annot_idx
            for annot_idx, this_annot in enumerate(raw.annotations)
            if any(ch_name in eye_ch_names for ch_name in this_annot["ch_names"])
        ],
        dtype=int,
    )


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
    phys_bpath = bids_path.copy().update(
        recording=eye_recording_tag,
        suffix="physio",
        extension="tsv.gz",
    )
    fname_tsv = phys_bpath.fpath

    data, times = raw.get_data(picks=eye_chs, return_times=True)
    ch_types = raw.get_channel_types(picks=eye_chs)
    data_dict = {"time": times}

    # Build sidecar JSON template
    json_dict = {
        "SamplingFrequency": raw.info["sfreq"],
        "StartTime": times[0],
        "Columns": ["time"],
        "PhysioType": "eyetrack",
        "RecordedEye": recorded_eye,
        "SampleCoordinateSystem": "gaze-on-screen",
        "time": {
            "Description": "The timestamp of the data, in seconds.",
            "Units": "s",
        },
    }
    # Update sidecar JSON with channels specific info
    raw_ch_names_to_bids = {}
    for ch_i, (ch_name, ch_type) in enumerate(zip(eye_chs, ch_types)):
        ch_idx = raw.ch_names.index(ch_name)
        bids_ch_name = ch_name
        unit = UNITS_FIFF_TO_BIDS_MAP[raw.info["chs"][ch_idx]["unit"]]
        # FIXME: Assumes only 1 x-coordinate and 1 y-coordinate eyegaze ch per eye
        if ch_type == "eyegaze":
            axis_code = raw.info["chs"][ch_idx]["loc"][4]
            if axis_code == -1:
                bids_ch_name = "x_coordinate"
                description = "The x-coordinate of the gaze on the screen."
            elif axis_code == 1:
                bids_ch_name = "y_coordinate"
                description = "The y-coordinate of the gaze on the screen."
            else:
                raise ValueError(
                    "Eyegaze channels must set "
                    "raw.info['chs'][channel_index]['loc'][4] to -1 for x-coordinate "
                    f"or 1 for y-coordinate. Got {axis_code} for channel {ch_name}. "
                    "Please use  "
                    "`mne.preprocessing.eyetracking.set_channel_types_eyetrack` to "
                    "Set eyetrack channel info according to MNE expectations."
                )
        elif ch_type == "pupil":
            bids_ch_name = "pupil_size"
            description = "Pupil size of the recorded eye"
        else:
            description = "Additional Channel written by MNE-Python"

        raw_ch_names_to_bids[ch_name] = bids_ch_name
        if bids_ch_name in data_dict:
            raise ValueError(
                f"Trying to rename {ch_name} to a BIDS compliant eyetracking name of "
                f"{bids_ch_name}, but this will result in duplicate BIDS names. Is it "
                "possible that  you have more than 1 x-coordinate, y-coordinate, "
                "and/or pupil size channel(s) for a single eye? Here is the current "
                f"mapping of your channel names to BIDS names:\n {raw_ch_names_to_bids}"
            )

        data_dict[bids_ch_name] = data[ch_i]
        json_dict["Columns"].append(bids_ch_name)
        json_dict[bids_ch_name] = {
            "Description": description,
            "Units": unit,
        }
    _write_tsv(fname_tsv, data_dict, compress=True, overwrite=overwrite)

    fname_json = (
        bids_path.copy()
        .update(
            recording=eye_recording_tag,
            suffix="physio",
            extension=".json",
        )
        .fpath
    )
    _write_json(fname_json, json_dict, overwrite)

    # Write physioevents TSV
    fname_events = (
        bids_path.copy()
        .update(
            recording=eye_recording_tag,
            suffix="physioevents",
            extension=".tsv.gz",
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
    eye_annot_indices = _get_eyetrack_annotation_inds(raw)
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
        compress=True,
        overwrite=overwrite,
    )
    # Write the JSON file
    fname_json = fname_tsv.with_suffix(".json")
    _events_json(
        fname_json, extra_columns=None, has_trial_type=True, overwrite=overwrite
    )


def _json_safe(value):
    """Convert values to JSON-serializable equivalents when needed."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        # np.int32 etc
        return value.item()
    return value


def _calibration_to_sidecar_updates(calibrations):
    """Convert calibration object(s) for one eye to sidecar updates."""
    updates = {}
    updates["CalibrationCount"] = len(calibrations)
    # FIXME: BEP020 allows CalibrationCount (per session/run) yet only provides one set
    # of Calibration* fields per physio sidecar. For now, if more than 1 calibrations
    # were run, I guess it makes most sense to take the last calibration.
    cal = calibrations[-1].copy()

    for from_key, to_key in MNE_CALIBRATION_TO_BIDS.items():
        value = cal.get(from_key)
        if value is not None:
            updates[to_key] = _json_safe(value)
    return updates


def write_eyetrack_calibration(
    bids_path: BIDSPath,
    calibrations: Calibration | list[Calibration],
) -> list[Path]:
    """Write eyetracking calibration metadata into an existing ``*_physio.json`` sidecar.

    Parameters
    ----------
    bids_path : mne_bids.BIDSPath
        BIDSPath for the eyetracking recording. The BIDSPath should point to a modality
        directory (e.g. ``beh`` or ``eeg``) that contains ``<match>_physio.json``
        file(s). If the BIDSPath contains a ``recording`` entity (e.g. ``eye1``), it
        will be ignored (see the notes section).
    calibration : CalibrationObject | list of CalibrationObject
        Calibration instance(s) (e.g., an item returned by
        :func:`~mne.preprocessing.eyetracking.read_eyelink_calibration`). Each instance
        must expose an ``eye`` attribute with value ``"left"`` or ``"right"``

    Returns
    -------
    Updated sidecar filepaths : list of pathlib.Path
        a list of filepaths pointing to the ``<match>_physio.tsv`` files that were
        updated with calibration information.

    Notes
    -----
    This function routes calibration metadata to the correct per-eye physio sidecar(s):

    - Binocular recordings: left eye -> ``<match>_recording-eye1_physio.tsv``,
      right eye -> ``<match>_recording-eye2_physio.tsv``
    - Monocular recordings: whichever eye was recorded ->
      ``<match>_recording-eye1_physio.tsv``

    If more than one calibration was run on the participant, this function will write
    the last calibration in the sequence passed to the ``calibration`` parameter.

    See `The Eyetracking BIDS specification`_.
    """  # noqa: E501 FIXME: Can we use an alias to make the long line fit?
    _validate_type(bids_path, BIDSPath, item_name="bids_path")

    if isinstance(calibrations, mne.preprocessing.eyetracking.Calibration):
        calibrations = [calibrations]

    cals_by_eye = {"left": [], "right": []}
    for cal in calibrations:
        eye = cal["eye"]
        cals_by_eye[eye].append(cal)
    eyes_present = {eye for eye, cals in cals_by_eye.items() if len(cals)}
    if not eyes_present:
        raise ValueError("No calibration entries were provided.")

    # Determine monocular vs binocular mapping to the *_physio.tsv files
    if eyes_present == {"left", "right"}:
        eye_to_recording = {"left": "eye1", "right": "eye2"}
    else:
        only_eye = next(iter(eyes_present))
        eye_to_recording = {only_eye: "eye1"}

    # Construct base path eye1 and/or eye2 <match>_physio.tsv files
    base_path = bids_path.copy().update(suffix="physio", extension=".json")

    updated_sidecar_fpaths = []
    for eye, recording_tag in eye_to_recording.items():
        sidecar_fpath = base_path.copy().update(recording=recording_tag).fpath
        if not sidecar_fpath.exists():
            msg = (
                "Eyetracking sidecar not found at "
                f"{sidecar_fpath}. Write the BIDS dataset first using write_raw_bids."
            )
            raise FileNotFoundError(msg)

        updates = _calibration_to_sidecar_updates(cals_by_eye[eye])
        if updates:
            sidecar = json.loads(sidecar_fpath.read_text(encoding="utf-8-sig"))
            sidecar.update(updates)
            _write_json(sidecar_fpath, sidecar, overwrite=True)
            updated_sidecar_fpaths.append(sidecar_fpath)
    return updated_sidecar_fpaths


def read_eyetrack_calibration(bids_path: BIDSPath) -> list[dict]:
    """Read eyetracking calibration metadata from ``*_physio.json`` sidecars.

    Parameters
    ----------
    bids_path : mne_bids.BIDSPath
        BIDSPath for the eyetracking recording. If ``recording`` is provided, only the
        matching eye sidecar is inspected. Otherwise, ``eye1`` and ``eye2`` sidecars
        are checked if present.

    Returns
    -------
    calibrations : list of mne.preprocessing.eyetracking.Calibration
        Calibration metadata entries using MNE-style keys
        (e.g. ``avg_error``, ``max_error``, ``model``, ``positions``, ``eye``). If
        ``CalibrationCount`` is present in the sidecar, it is returned as
        ``calibration_count``.

    Notes
    -----
    .. Warning::
        BIDS does not provide fields to store the time of the calibration, nor the
        actual participant gaze positions to each calibration dot, or the offset between
        each dot and the participants gaze to it. However, MNE-Python
        :class:`~mne.preprocessing.eyetracking.Calibration` instances typically do store
        this information. Thus, when reading calibration info from a BIDS dataset, note
        that the ``'onset'``, ``'offsets'``, and ``'gaze'`` keys of
        :class:`~mne.preprocessing.eyetracking.Calibration` instance(s) will be set
        to ``np.nan``.
    """
    _validate_type(bids_path, BIDSPath, item_name="bids_path")
    base_path = bids_path.copy().update(suffix="physio", extension=".json")

    if base_path.recording is not None:
        candidate_sidecars = [base_path.fpath]
    else:
        candidate_sidecars = []
        for recording_tag in ("eye1", "eye2"):
            fpath = base_path.copy().update(recording=recording_tag).fpath
            if fpath.exists():
                candidate_sidecars.append(fpath)

    if not candidate_sidecars:
        raise FileNotFoundError(
            "No eyetracking physio sidecar JSON files found. "
            f"Tried base path: {base_path.fpath.parent}"
        )

    calibrations = []
    for sidecar_fpath in candidate_sidecars:
        sidecar = json.loads(sidecar_fpath.read_text(encoding="utf-8-sig"))
        calibration = {}
        for bids_key, mne_key in BIDS_CALIBRATION_TO_MNE.items():
            if bids_key in sidecar:
                value = sidecar[bids_key]
                if bids_key == "CalibrationPosition":
                    value = np.array(value)
                calibration[mne_key] = value
        if "RecordedEye" in sidecar:
            calibration["eye"] = sidecar["RecordedEye"]

        # And unfortunately BIDS doesnt have these fields..
        onset = np.nan
        gaze = np.full_like(calibration["positions"], np.nan)
        offsets = np.full_like(calibration["positions"], np.nan)
        if calibration:
            mne_cal = Calibration(
                onset=onset, gaze=gaze, offsets=offsets, **calibration
            )
            calibrations.append(mne_cal)

    if not calibrations:
        raise ValueError(f"No calibration metadata found in {candidate_sidecars}.")

    return calibrations


def read_raw_bids_eyetrack(bids_path):
    """Read BIDS compliant eyetracking data from TSV sidecar files.

    bids_path : mne_bids.BIDSPath
        the BIDSPath instance that points to the ``<match>_recording-eye1_physio.tsv``
        eyetracking file. You must specify the following entities in the BIDSPath
        constructor: ``recording="eye1"``, ``suffix="physio"``, ``extension=".tsv"``.
        If the eyetracking data was recorded without another modality, you must also
        specify ``datatype="beh"``. Otherwise, if the eyetracking was collected
        alongside another modality such as ``eeg``, then you must specify
        ``datatype="eeg"``.
    """
    ch_info = {}
    # Mapping from FIFF units to the str codes wanted by set_channel_types_eyetrack
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

    eye1_data_dict, eye1_ch_info = _read_one_eye_physio(
        raw_path,
    )
    ch_info.update(eye1_ch_info)
    eye1_array = np.array(list(eye1_data_dict.values()))

    json_sidecar_fpath = raw_path.with_suffix("").with_suffix(".json")
    # We are assuming that sfreq is consistent across eyes but I think that is safe..
    eye1_sfreq = json.loads(json_sidecar_fpath.read_text())["SamplingFrequency"]

    if num_eyes == 2:
        eye2_data_dict, eye2_ch_info = _read_one_eye_physio(
            eye2_fpath,
        )
        ch_info.update(eye2_ch_info)
        eye2_array = np.array(list(eye2_data_dict.values()))

        # timestamp is row 0. Exclude it
        data = np.concat([eye1_array[1:, :], eye2_array[1:, :]], axis=0)
    else:
        data = eye1_array[1:, :]  # timestamp is row 0. Exclude it

    ch_names = [ch_name for ch_name in ch_info]
    ch_types = [ch_info[ch_name]["ch_type"] for ch_name in ch_info]
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=eye1_sfreq)
    raw = mne.io.RawArray(data, info)

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


def _read_one_eye_physio(raw_tsv_fpath):
    """Read one eye's physio.tsv/.json and return channel metadata and samples."""

    def _infer_et_type(ch_name):
        """Eyetracking BIDS pre-specifies valid column names for x/y + pupil data."""
        return (
            "eyegaze"
            if ch_name in ["x_coordinate", "y_coordinate"]
            else "pupil"
            if ch_name == "pupil_size"
            else "misc"
        )

    json_fpath = raw_tsv_fpath.with_suffix("").with_suffix(".json")
    sidecar = json.loads(json_fpath.read_text())

    cols = sidecar["Columns"]
    recorded_eye = sidecar["RecordedEye"]

    ch_info = {}
    # The first column is 'time' so skip it
    for col_idx, ch_name in enumerate(cols[1:], start=1):
        unit_str = sidecar[ch_name]["Units"]
        ch_type = _infer_et_type(ch_name)

        ch_info[f"{ch_name}_{recorded_eye}"] = dict(
            ch_type=ch_type,
            unit=UNITS_BIDS_TO_FIFF_MAP[unit_str],
            eye=recorded_eye,
            axis=(
                "x"
                if ch_name == "x_coordinate"
                else "y"
                if ch_name == "y_coordinate"
                else None
            ),
        )

    data_dict = _from_compressed_tsv(raw_tsv_fpath)
    # append eye to ch_name
    for col_name in list(data_dict.keys()):
        if col_name != "time":
            data_dict[f"{col_name}_{recorded_eye}"] = data_dict.pop(col_name)
    return data_dict, ch_info
