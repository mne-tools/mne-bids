"""Code to facilitate I/O of BIDS compliant eyetracking data (BEP 020)."""

import json
from pathlib import Path

import mne
import numpy as np
from mne.preprocessing.eyetracking import Calibration
from mne.utils import _validate_type, logger, warn

from mne_bids.config import UNITS_FIFF_TO_BIDS_MAP
from mne_bids.path import BIDSPath
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

EYETRACK_CALIBRATION_TO_STIMULUS_PRESENTATION = (
    ("screen_distance", "ScreenDistance"),
    ("screen_origin", "ScreenOrigin"),
    ("screen_resolution", "ScreenResolution"),
    ("screen_size", "ScreenSize"),
)


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
            if any(
                ch_name in eye_ch_names for ch_name in this_annot.get("ch_names", [])
            )
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
    data_dict = {"timestamp": times}

    # Build sidecar JSON template
    json_dict = {
        "SamplingFrequency": raw.info["sfreq"],
        "StartTime": times[0],
        "Columns": ["timestamp"],
        "PhysioType": "eyetrack",
        "RecordedEye": recorded_eye,
        "SampleCoordinateSystem": "gaze-on-screen",
        "timestamp": {
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
    _write_json(fname_json, json_dict, overwrite=overwrite)

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
        warn(f"No eyetracking annotations found. {fname_tsv} will NOT be written.")
        return
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
    ev_dict = _events_tsv(
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
    metadata = {"Columns": list(ev_dict.keys()), "OnsetSource": "timestamp"}
    fname_json = fname_tsv.with_suffix("").with_suffix(".json")
    _events_json(
        fname_json, metadata=metadata, has_trial_type=True, overwrite=overwrite
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
    # BEP020 allows CalibrationCount (per session/run) yet only provides one set
    # of Calibration* fields per physio sidecar. For now, if more than 1 calibrations
    # and the user passes a squence of calibrations in, I guess it makes most sense to
    # take the last calibration collected.
    if (n_cals := len(calibrations)) > 1:
        most_recent = max(calibrations, key=lambda c: c["onset"])
        logger.info(
            f"{n_cals} calibrations were provided {most_recent['eye']} eye, writing "
            f" the calibration collected at {most_recent['onset']} seconds."
        )
        cal = most_recent.copy()
    else:
        cal = calibrations[-1]

    for from_key, to_key in MNE_CALIBRATION_TO_BIDS.items():
        value = cal.get(from_key)
        if value is not None:
            updates[to_key] = _json_safe(value)
    return updates


def write_eyetrack_calibration(
    bids_path: BIDSPath,
    calibrations: Calibration | list[Calibration],
) -> list[Path]:
    """Write eyetrack calibration metadata into an existing ``*_physio.json`` sidecar.

    Parameters
    ----------
    bids_path : mne_bids.BIDSPath
        BIDSPath for the eyetracking recording. The BIDSPath should point to a modality
        directory (e.g. ``beh`` or ``eeg``) that contains ``<match>_physio.json``
        file(s). If the BIDSPath contains a ``recording`` entity (e.g. ``eye1``), it
        will be ignored (see the notes section).
    calibrations : CalibrationObject | list of CalibrationObject
        Calibration instance(s) (e.g., an item returned by
        :func:`~mne.preprocessing.eyetracking.read_eyelink_calibration`). Each instance
        must expose an ``eye`` attribute with value ``"left"`` or ``"right"``.

    Returns
    -------
    Updated sidecar filepaths : list of pathlib.Path
        A list of filepaths pointing to the ``<match>_physio.tsv`` files that were
        updated with calibration information.

    Notes
    -----
    This function routes calibration metadata to the correct per-eye physio sidecar(s):

    - Binocular recordings: left eye -> ``<match>_recording-eye1_physio.tsv``,
      right eye -> ``<match>_recording-eye2_physio.tsv``
    - Monocular recordings: whichever eye was recorded ->
      ``<match>_recording-eye1_physio.tsv``

    If more than one calibration was run on the participant, this function will write
    the last calibration in the sequence passed to the ``calibrations`` parameter.

    See `The Eyetracking BIDS specification`_.
    """
    _validate_type(bids_path, BIDSPath, item_name="bids_path")

    if isinstance(calibrations, mne.preprocessing.eyetracking.Calibration):
        calibrations = [calibrations]

    cals_by_eye = {"left": [], "right": []}
    for cal in calibrations:
        eye = cal["eye"]
        if eye not in cals_by_eye.keys():
            raise ValueError(
                "Each mne.preprocessing.Calibration instance must contain either "
                f"'left' or 'right' in its 'eye' key. Got {eye} "
            )
        cals_by_eye[eye].append(cal)
    eyes_present = {eye for eye, cals in cals_by_eye.items() if len(cals)}

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


def _eyetrack_calibration_to_events_metadata(eyetrack_calibration):
    """Extract BIDS StimulusPresentation metadata from eyetracking calibration."""
    if eyetrack_calibration is None:
        raise ValueError(
            "Writing eyetracking data requires `eyetrack_calibration`. The "
            "calibration object must include screen_distance, screen_origin, "
            "screen_resolution, and screen_size."
        )
    if isinstance(eyetrack_calibration, mne.preprocessing.eyetracking.Calibration):
        calibrations = [eyetrack_calibration]
    else:
        _validate_type(
            eyetrack_calibration, (list, tuple), item_name="eyetrack_calibration"
        )
        calibrations = list(eyetrack_calibration)
    if len(calibrations) == 0:
        raise ValueError(
            "`eyetrack_calibration` must contain at least one calibration."
        )

    stimulus_presentation = {}
    missing = []
    for mne_key, bids_key in EYETRACK_CALIBRATION_TO_STIMULUS_PRESENTATION:
        values = []
        for calibration in calibrations:
            value = calibration.get(mne_key)
            if value is not None:
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                elif isinstance(value, tuple):
                    value = list(value)
                values.append(value)
        if not values:
            missing.append(mne_key)
            continue
        first_value = values[0]
        if any(value != first_value for value in values[1:]):
            raise ValueError(
                f"`eyetrack_calibration` contains inconsistent values for {mne_key!r}."
            )
        stimulus_presentation[bids_key] = first_value

    if missing:
        raise ValueError(
            "`eyetrack_calibration` is missing screen metadata required for "
            "eyetracking BIDS: " + ", ".join(missing)
        )
    return {"StimulusPresentation": stimulus_presentation}
