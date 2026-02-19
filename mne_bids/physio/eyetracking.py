"""Code to facilitate I/O of BIDS compliant eyetracking data (BEP 020)."""

import json
from pathlib import Path

import mne
import numpy as np
from mne._fiff.constants import FIFF
from mne.preprocessing.eyetracking import Calibration, set_channel_types_eyetrack
from mne.utils import _validate_type, logger, warn

from mne_bids.config import UNITS_BIDS_TO_FIFF_MAP
from mne_bids.path import BIDSPath
from mne_bids.physio._utils import _get_physio_type
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
        extension=".tsv",
    )
    fname_tsv = phys_bpath.fpath

    data, times = raw.get_data(picks=eye_chs, return_times=True)
    data_dict = {}
    data_dict["time"] = times
    for ch_i, ch_name in enumerate(eye_chs):
        data_dict[ch_name] = data[ch_i]
    _write_tsv(fname_tsv, data_dict, include_column_names=False, overwrite=overwrite)

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
    _write_json(fname_json, json_dict, overwrite)

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
        include_column_names=False,
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


def read_raw_bids_eyetrack(bids_path, *, ch_types: None | dict[str, str]):
    """Read BIDS compliant eyetracking data from TSV sidecar files.

    bids_path : mne_bids.BIDSPath
        the BIDSPath instance that points to the ``<match>_recording-eye1_physio.tsv``
        eyetracking file. You must specify the following entities in the BIDSPath
        constructor: ``recording="eye1"``, ``suffix="physio"``, ``extension=".tsv"``.
        If the eyetracking data was recorded without another modality, you must also
        specify ``datatype="beh"``. Otherwise, if the eyetracking was collected
        alongside another modality such as ``eeg``, then you must specify
        ``datatype="eeg"``.
    ch_types : None | dict of str
        Either ``None``, or a dictionary whose keys correspond to eyetracking channel
        names, and whose values correspond to the MNE-Python compatible channel types
        for said channel, such as ``'eyegaze'``, ``'pupil'``, or ``'misc'``. If
        ``None``, then the data in the 2nd and 3rd columns of
        ``<match>_recording-{eye1,eye2}_physio.tsv`` will be set to ``eyegaze``, and
        data from all subsequent columns will be set to ``'misc'``.
    """
    if ch_types is None:
        ch_types = {}

    ch_info = {}
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

    eye1_cols, eye1_array, eye1_ch_info, unknown_types = _read_one_eye_physio(
        raw_path,
        ch_types,
    )
    ch_info.update(eye1_ch_info)
    json_sidecar_fpath = raw_path.with_suffix(".json")
    eye1_sfreq = json.loads(json_sidecar_fpath.read_text())["SamplingFrequency"]

    if num_eyes == 2:
        eye2_cols, eye2_array, eye2_ch_info, eye2_unknown_types = _read_one_eye_physio(
            eye2_fpath,
            ch_types,
        )
        ch_info.update(eye2_ch_info)
        unknown_types.extend(eye2_unknown_types)

        data = np.concat([eye1_array, eye2_array], axis=1)
        ch_names = eye1_cols[1:] + eye2_cols[1:]
        types = [ch_info[name]["ch_type"] for name in ch_names]
    else:
        data = eye1_array
        ch_names = eye1_cols[1:]
        types = [ch_info[name]["ch_type"] for name in ch_names]

    if unknown_types:
        warn(
            f"Assigning channel type 'misc' to {unknown_types}.\n"
            "If this is incorrect, pass the correct channel types to the "
            "eyetrack_ch_types parameter."
        )
    info = mne.create_info(ch_names=ch_names, ch_types=types, sfreq=eye1_sfreq)
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


def _read_one_eye_physio(raw_tsv_fpath, ch_types):
    """Read one eye's physio.tsv/.json and return channel metadata and samples."""

    def _infer_et_type(column_idx):
        return "eyegaze" if column_idx in [1, 2] else "misc"

    json_fpath = raw_tsv_fpath.with_suffix(".json")
    sidecar = json.loads(json_fpath.read_text())

    cols = sidecar["Columns"]
    recorded_eye = sidecar["RecordedEye"]

    ch_info = {}
    unknown_types = []
    for col_idx, ch_name in enumerate(cols[1:], start=1):
        unit_str = sidecar[ch_name]["Units"]
        ch_type = ch_types.get(ch_name, _infer_et_type(col_idx))
        if col_idx > 2 and ch_type == "misc":
            unknown_types.append(ch_name)

        ch_info[ch_name] = dict(
            ch_type=ch_type,
            unit=UNITS_BIDS_TO_FIFF_MAP[unit_str],
            eye=recorded_eye,
            axis=("x" if col_idx == 1 else "y" if col_idx == 2 else None),
        )

    n_cols = len(cols)
    data = np.loadtxt(raw_tsv_fpath, usecols=range(1, n_cols))
    return cols, data, ch_info, unknown_types
