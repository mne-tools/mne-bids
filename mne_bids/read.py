"""Check whether a file format is supported by BIDS and then load it."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import re
from datetime import datetime, timedelta, timezone
from difflib import get_close_matches
from pathlib import Path

import mne
import numpy as np
from mne import events_from_annotations, io, pick_channels_regexp, read_events
from mne.coreg import fit_matched_points
from mne.transforms import apply_trans
from mne.utils import get_subjects_dir, logger

from mne_bids.config import (
    ALLOWED_DATATYPE_EXTENSIONS,
    ANNOTATIONS_TO_KEEP,
    _map_options,
    reader,
)
from mne_bids.dig import _read_dig_bids
from mne_bids.path import (
    BIDSPath,
    _find_matching_sidecar,
    _infer_datatype,
    _parse_ext,
    get_bids_path_from_fname,
)
from mne_bids.tsv_handler import _drop, _from_tsv
from mne_bids.utils import _get_ch_type_mapping, _import_nibabel, verbose, warn


def _read_raw(
    raw_path,
    electrode=None,
    hsp=None,
    hpi=None,
    allow_maxshield=False,
    config_path=None,
    **kwargs,
):
    """Read a raw file into MNE, making inferences based on extension."""
    _, ext = _parse_ext(raw_path)

    # KIT systems
    if ext in [".con", ".sqd"]:
        raw = io.read_raw_kit(
            raw_path, elp=electrode, hsp=hsp, mrk=hpi, preload=False, **kwargs
        )

    # BTi systems
    elif ext == ".pdf":
        raw = io.read_raw_bti(
            pdf_fname=raw_path,
            config_fname=config_path,
            head_shape_fname=hsp,
            preload=False,
            **kwargs,
        )

    elif ext == ".fif":
        raw = reader[ext](raw_path, allow_maxshield, **kwargs)

    elif ext in [".ds", ".vhdr", ".set", ".edf", ".bdf", ".EDF", ".snirf", ".cdt"]:
        raw_path = Path(raw_path)
        raw = reader[ext](raw_path, **kwargs)

    # MEF and NWB are allowed, but not yet implemented
    elif ext in [".mef", ".nwb"]:
        raise ValueError(
            f'Got "{ext}" as extension. This is an allowed '
            f"extension but there is no IO support for this "
            f"file format yet."
        )

    # No supported data found ...
    # ---------------------------
    else:
        raise ValueError(
            f"Raw file name extension must be one "
            f"of {ALLOWED_DATATYPE_EXTENSIONS}\n"
            f"Got {ext}"
        )
    return raw


def _read_events(events, event_id, raw, bids_path=None):
    """Retrieve events (for use in *_events.tsv) from FIFF/array & Annotations.

    Parameters
    ----------
    events : path-like | np.ndarray | None
        If a string, a path to an events file. If an array, an MNE events array
        (shape n_events, 3). If None, events will be generated from
        ``raw.annotations``.
    event_id : dict | None
        The event id dict used to create a 'trial_type' column in events.tsv,
        mapping a description key to an integer-valued event code.
    raw : mne.io.Raw
        The data as MNE-Python Raw object.
    bids_path : BIDSPath | None
        Can be used to determine if the data is a resting-state or empty-room
        recording, and will suppress a warning about missing events in this
        case.

    Returns
    -------
    all_events : np.ndarray, shape = (n_events, 3)
        The first column contains the event time in samples and the third
        column contains the event id. The second column is ignored for now but
        typically contains the value of the trigger channel either immediately
        before the event or immediately after.
    all_dur : np.ndarray, shape (n_events,)
        The event durations in seconds.
    all_desc : dict
        A dictionary with the keys corresponding to the event descriptions and
        the values to the event IDs.

    """
    # retrieve events
    if isinstance(events, np.ndarray):
        if events.ndim != 2:
            raise ValueError(f"Events must have two dimensions, found {events.ndim}")
        if events.shape[1] != 3:
            raise ValueError(
                "Events must have second dimension of length 3, "
                f"found {events.shape[1]}"
            )
        events = events
    elif events is None:
        events = np.empty(shape=(0, 3), dtype=int)
    else:
        events = read_events(events).astype(int)

    if raw.annotations:
        if event_id is None:
            logger.info(
                "The provided raw data contains annotations, but you did not "
                'pass an "event_id" mapping from annotation descriptions to '
                "event codes. We will generate arbitrary event codes. "
                'To specify custom event codes, please pass "event_id".'
            )
        else:
            special_annots = {"BAD_ACQ_SKIP"}
            desc_without_id = sorted(
                set(raw.annotations.description) - set(event_id.keys())
            )
            # auto-add entries to `event_id` for "special" annotation values
            # (but only if they're needed)
            if set(desc_without_id) & special_annots:
                for annot in special_annots:
                    # use a value guaranteed to not be in use
                    event_id = {annot: max(event_id.values()) + 90000} | event_id
                # remove the "special" annots from the list of problematic annots
                desc_without_id = sorted(set(desc_without_id) - special_annots)
            if desc_without_id:
                raise ValueError(
                    f"The provided raw data contains annotations, but "
                    f'"event_id" does not contain entries for all annotation '
                    f"descriptions. The following entries are missing: "
                    f"{', '.join(desc_without_id)}"
                )

    # If we have events, convert them to Annotations so they can be easily
    # merged with existing Annotations.
    if events.size > 0 and event_id is not None:
        ids_without_desc = set(events[:, 2]) - set(event_id.values())
        if ids_without_desc:
            raise ValueError(
                f"No description was specified for the following event(s): "
                f"{', '.join([str(x) for x in sorted(ids_without_desc)])}. "
                f"Please add them to the event_id dictionary, or drop them "
                f"from the events array."
            )

        # Append events to raw.annotations. All event onsets are relative to
        # measurement beginning.
        id_to_desc_map = dict(zip(event_id.values(), event_id.keys()))
        # We don't pass `first_samp`, as set_annotations() below will take
        # care of this shift automatically.
        new_annotations = mne.annotations_from_events(
            events=events,
            sfreq=raw.info["sfreq"],
            event_desc=id_to_desc_map,
            orig_time=raw.annotations.orig_time,
        )

        raw = raw.copy()  # Don't alter the original.
        annotations = raw.annotations.copy()

        # We use `+=` here because `Annotations.__iadd__()` does the right
        # thing and also performs a sanity check on `Annotations.orig_time`.
        annotations += new_annotations
        raw.set_annotations(annotations)
        del id_to_desc_map, annotations, new_annotations

    if events.size > 0 and event_id is None:
        new_annotations = mne.annotations_from_events(
            events=events,
            sfreq=raw.info["sfreq"],
            orig_time=raw.annotations.orig_time,
        )

        raw = raw.copy()  # Don't alter the original.
        annotations = raw.annotations.copy()

        # We use `+=` here because `Annotations.__iadd__()` does the right
        # thing and also performs a sanity check on `Annotations.orig_time`.
        annotations += new_annotations
        raw.set_annotations(annotations)
        del annotations, new_annotations

    # Now convert the Annotations to events.
    all_events, all_desc = events_from_annotations(
        raw,
        event_id=event_id,
        regexp=None,  # Include `BAD_` and `EDGE_` Annotations, too.
    )
    all_dur = raw.annotations.duration

    # Warn about missing events if not rest or empty-room data
    if (all_events.size == 0 and bids_path.task is not None) and (
        not bids_path.task.startswith("rest")
        and not (bids_path.subject == "emptyroom" and bids_path.task == "noise")
    ):
        warn(
            "No events found or provided. Please add annotations to the raw "
            "data, or provide the events and event_id parameters. For "
            "resting state data, BIDS recommends naming the task using "
            'labels beginning with "rest".'
        )

    return all_events, all_dur, all_desc


def _verbose_list_index(lst, val, *, allow_all=False):
    # try to "return lst.index(val)" for list of str, but be more
    # informative/verbose when it fails
    try:
        return lst.index(val)
    except ValueError as exc:
        # Use str cast here to deal with pathlib.Path instances
        extra = get_close_matches(str(val), [str(ll) for ll in lst])
        if allow_all and not extra:
            extra = lst
        extra = f". Did you mean one of {extra}?" if extra else ""
        raise ValueError(f"{exc}{extra}") from None


def _handle_participants_reading(participants_fname, raw, subject):
    participants_tsv = _from_tsv(participants_fname)
    subjects = participants_tsv["participant_id"]
    row_ind = _verbose_list_index(subjects, subject, allow_all=True)
    raw.info["subject_info"] = dict()  # start from scratch

    # set data from participants tsv into subject_info
    # TODO: Could potentially use "comment" someday to store other options e.g. in JSON
    # https://github.com/mne-tools/fiff-constants/blob/e27f68cbf74dbfc5193ad429cc77900a59475181/DictionaryTags.txt#L369
    allowed_keys = set(
        """
    id his_id last_name first_name middle_name birthday sex hand weight height
    """.strip().split()
    )
    bad_key_vals = list()
    for col_name, value in participants_tsv.items():
        orig_value = value = value[row_ind]
        if col_name in ("sex", "hand"):
            value = _map_options(what=col_name, key=value, fro="bids", to="mne")
            # We don't know how to translate to MNE, so skip.
            if value is None:
                if col_name == "sex":
                    info_str = "subject sex"
                else:
                    info_str = "subject handedness"
                bad_key_vals.append((col_name, orig_value, info_str))
        elif col_name in ("height", "weight"):
            try:
                value = float(value)
            except ValueError:
                value = None
        elif col_name == "age":
            if raw.info["meas_date"] is None:
                value = None
            elif value is not None:
                try:
                    value = float(value)
                except Exception:
                    value = None
                else:
                    value = (
                        raw.info["meas_date"]
                        - timedelta(days=int(np.ceil(365.25 * value)))
                    ).date()
        else:
            if value == "n/a":
                value = None

        # adjust keys to match MNE nomenclature
        key = col_name
        if col_name == "participant_id":
            key = "his_id"
        elif col_name == "age":
            key = "birthday"

        if key not in allowed_keys:
            bad_key_vals.append((col_name, orig_value, None))
            continue

        # add data into raw.Info
        if value is not None:
            assert key not in raw.info["subject_info"]
            raw.info["subject_info"][key] = value

    if bad_key_vals:
        warn_str = "Unable to map the following column(s) to to MNE:"
        for col_name, orig_value, info_str in bad_key_vals:
            warn_str += f"\n{col_name}"
            if info_str is not None:
                warn_str += f" ({info_str})"
            warn_str += f": {orig_value}"
        warn(warn_str)

    return raw


def _handle_scans_reading(scans_fname, raw, bids_path):
    """Read associated scans.tsv and set meas_date."""
    scans_tsv = _from_tsv(scans_fname)
    fname = bids_path.fpath.name

    if fname.endswith(".pdf"):
        # for BTi files, the scan is an entire directory
        fname = fname.split(".")[0]

    # get the row corresponding to the file
    # use string concatenation instead of os.path
    # to work nicely with windows
    data_fname = Path(bids_path.datatype) / fname
    fnames = scans_tsv["filename"]
    fnames = [Path(fname) for fname in fnames]
    if "acq_time" in scans_tsv:
        acq_times = scans_tsv["acq_time"]
    else:
        acq_times = ["n/a"] * len(fnames)

    # There are three possible extensions for BrainVision
    # First gather all the possible extensions
    acq_suffixes = set(fname.suffix for fname in fnames)
    # Add the filename extension for the bids folder
    acq_suffixes.add(Path(data_fname).suffix)

    if all(suffix in (".vhdr", ".eeg", ".vmrk") for suffix in acq_suffixes):
        ext = fnames[0].suffix
        data_fname = Path(data_fname).with_suffix(ext)
    row_ind = _verbose_list_index(fnames, data_fname)

    # check whether all split files have the same acq_time
    # and throw an error if they don't
    if "_split-" in fname:
        split_idx = fname.find("split-")
        pattern = re.compile(
            bids_path.datatype
            + "/"
            + bids_path.basename[:split_idx]
            + r"split-\d+_"
            + bids_path.datatype
            + bids_path.fpath.suffix
        )
        split_fnames = list(filter(lambda x: pattern.match(x.as_posix()), fnames))
        split_acq_times = []
        for split_f in split_fnames:
            split_acq_times.append(acq_times[_verbose_list_index(fnames, split_f)])
        if len(set(split_acq_times)) != 1:
            raise ValueError("Split files must have the same acq_time.")

    # extract the acquisition time from scans file
    acq_time = acq_times[row_ind]
    if acq_time != "n/a":
        # BIDS allows the time to be stored in UTC with a zero time-zone offset, as
        # indicated by a trailing "Z" in the datetime string. If the "Z" is missing, the
        # time is represented as "local" time. We have no way to know what the local
        # time zone is at the *acquisition* site; so we simply assume the same time zone
        # as the user's current system (this is what the spec demands anyway).
        acq_time_is_utc = acq_time.endswith("Z")

        # microseconds part in the acquisition time is optional; add it if missing
        if "." not in acq_time:
            if acq_time_is_utc:
                acq_time = acq_time.replace("Z", ".0Z")
            else:
                acq_time += ".0"

        date_format = "%Y-%m-%dT%H:%M:%S.%f"
        if acq_time_is_utc:
            date_format += "Z"

        acq_time = datetime.strptime(acq_time, date_format)

        if acq_time_is_utc:
            # Enforce setting timezone to UTC without additonal conversion
            acq_time = acq_time.replace(tzinfo=timezone.utc)
        else:
            # Convert time offset to UTC
            acq_time = acq_time.astimezone(timezone.utc)

        logger.debug(f"Loaded {scans_fname} scans file to set acq_time as {acq_time}.")
        # First set measurement date to None and then call call anonymize() to
        # remove any traces of the measurement date we wish
        # to replace – it might lurk out in more places than just
        # raw.info['meas_date'], e.g. in info['meas_id]['secs'] and in
        # info['file_id'], which are not affected by set_meas_date().
        # The combined use of set_meas_date(None) and anonymize() is suggested
        # by the MNE documentation, and in fact we cannot load e.g. OpenNeuro
        # ds003392 without this combination.
        raw.set_meas_date(None)
        raw.anonymize(daysback=None, keep_his=True)
        raw.set_meas_date(acq_time)

    return raw


def _handle_info_reading(sidecar_fname, raw):
    """Read associated sidecar JSON and populate raw.

    Handle PowerLineFrequency of recording.
    """
    with open(sidecar_fname, encoding="utf-8-sig") as fin:
        sidecar_json = json.load(fin)

    # read in the sidecar JSON's and raw object's line frequency
    json_linefreq = sidecar_json.get("PowerLineFrequency")
    raw_linefreq = raw.info["line_freq"]

    # If both are defined, warn if there is a conflict, else all is fine
    if (json_linefreq is not None) and (raw_linefreq is not None):
        if json_linefreq != raw_linefreq:
            msg = (
                f"Line frequency in sidecar JSON does not match the info "
                f"data structure of the mne.Raw object:\n"
                f"Sidecar JSON is -> {json_linefreq}\n"
                f"Raw is -> {raw_linefreq}\n\n"
            )

            if json_linefreq == "n/a":
                msg += "Defaulting to the info from mne.Raw object."
                raw.info["line_freq"] = raw_linefreq
            else:
                msg += "Defaulting to the info from sidecar JSON."
                raw.info["line_freq"] = json_linefreq

            warn(msg)

    # Else, try to use JSON, fall back on mne.Raw
    elif (json_linefreq is not None) and (json_linefreq != "n/a"):
        raw.info["line_freq"] = json_linefreq
    else:
        pass  # line freq is either defined or None in mne.Raw

    # get cHPI info
    chpi = sidecar_json.get("ContinuousHeadLocalization")
    if chpi is None:
        # no cHPI info in the sidecar – leave raw.info unchanged
        pass
    elif chpi is True:
        from mne.io.ctf import RawCTF
        from mne.io.kit.kit import RawKIT

        msg = (
            "Cannot verify that the cHPI frequencies from "
            "the MEG JSON sidecar file correspond to the raw data{}"
        )

        if isinstance(raw, RawCTF):
            # Pick channels corresponding to the cHPI positions
            hpi_picks = pick_channels_regexp(raw.info["ch_names"], "HLC00[123][123].*")
            if len(hpi_picks) != 9:
                raise ValueError(
                    f"Could not find all cHPI channels that we expected for "
                    f"CTF data. Expected: 9, found: {len(hpi_picks)}"
                )
            logger.info(msg.format(" for CTF files."))

        elif isinstance(raw, RawKIT):
            logger.info(msg.format(" for KIT files."))

        elif "HeadCoilFrequency" in sidecar_json:
            hpi_freqs_json = sidecar_json["HeadCoilFrequency"]
            try:
                hpi_freqs_raw, _, _ = mne.chpi.get_chpi_info(raw.info)
            except ValueError:
                logger.info(msg.format("."))
            else:
                # XXX: Set chpi info in mne.Raw to what is in the sidecar
                if not np.allclose(hpi_freqs_json, hpi_freqs_raw):
                    warn(
                        f"The cHPI coil frequencies in the sidecar file "
                        f"{sidecar_fname}:\n    {hpi_freqs_json}\n "
                        f"differ from what is stored in the raw data:\n"
                        f"    {hpi_freqs_raw}.\n"
                        f"Defaulting to the info from mne.Raw object."
                    )
        else:
            addmsg = (
                ".\n(Because no 'HeadCoilFrequency' data was found in the sidecar.)"
            )
            logger.info(msg.format(addmsg))

    else:
        if raw.info["hpi_subsystem"]:
            logger.info(
                "Dropping cHPI information stored in raw data, "
                "following specification in sidecar file"
            )
        with raw.info._unlock():
            raw.info["hpi_subsystem"] = None
            raw.info["hpi_meas"] = []

    return raw


def events_file_to_annotation_kwargs(events_fname: str | Path) -> dict:
    r"""
    Read the ``events.tsv`` file and extract onset, duration, and description.

    Parameters
    ----------
    events_fname : str
        The file path to the ``events.tsv`` file.

    Returns
    -------
    kwargs_dict : dict

        A dictionary containing the following keys:

        - 'onset' : np.ndarray
            The onset times of the events in seconds.
        - 'duration' : np.ndarray
            The durations of the events in seconds.
        - 'description' : np.ndarray
            The descriptions of the events.
        - 'event_id' : dict
            A dictionary mapping event descriptions to integer event IDs.
        - 'extras' : list of dict
            A list of dictionaries containing additional columns from the
            ``events.tsv`` file. Each dictionary corresponds to a row.
            This corresponds to the ``extras`` argument of class
            :class:`mne.Annotations`.

    Notes
    -----
    The function handles the following cases:

    - If the ``trial_type`` column is available, it uses it for event descriptions.
    - If the ``stim_type`` column is available, it uses it for backward compatibility.
    - If the ``value`` column is available, it uses it to create the ``event_id``.
    - If none of the above columns are available, it defaults to using 'n/a' for
      descriptions and 1 for event IDs.

    Examples
    --------
    >>> import pandas as pd
    >>> from pathlib import Path
    >>> import tempfile
    >>>
    >>> # Create a sample DataFrame
    >>> data = {
    ...     'onset': [0.1, 0.2, 0.3],
    ...     'duration': [0.1, 0.1, 0.1],
    ...     'trial_type': ['event1', 'event2', 'event1'],
    ...     'value': [1, 2, 1],
    ...     'sample': [10, 20, 30]
            'foo': ['a', 'b', 'c'],
    ... }
    >>> df = pd.DataFrame(data)
    >>>
    >>> # Write the DataFrame to a temporary file
    >>> temp_dir = tempfile.gettempdir()
    >>> events_file = Path(temp_dir) / 'events.tsv'
    >>> df.to_csv(events_file, sep='\t', index=False)
    >>>
    >>> # Read the events file using the function
    >>> events_dict = events_file_to_annotation_kwargs(events_file)
    >>> events_dict
    {'onset': array([0.1, 0.2, 0.3]),
    'duration': array([0.1, 0.1, 0.1]),
    'description': array(['event1', 'event2', 'event1'], dtype='<U6'),
    'event_id': {'event1': 1, 'event2': 2},
    'extras': [{'foo': 'a'}, {'foo': 'b'}, {'foo': 'c'}]}

    """
    logger.info(f"Reading events from {events_fname}.")
    events_dict = _from_tsv(events_fname)

    # drop events where onset is n/a; we can't annotate them and thus don't need entries
    # for them in event_id either
    events_dict = _drop(events_dict, "n/a", "onset")

    # Get event descriptions. Use `trial_type` column if available.
    if "trial_type" in events_dict:
        trial_type_col_name = "trial_type"
    # allow `stim_type` for backward-compat with old datasets.
    elif "stim_type" in events_dict:
        trial_type_col_name = "stim_type"
        warn(
            f'The events file, {events_fname}, contains a "stim_type" column. This '
            'column should be renamed to "trial_type" for BIDS compatibility.'
        )
    # If we lack proper event descriptions, perhaps we have at least an event value?
    elif "value" in events_dict:
        trial_type_col_name = "value"
    # Worst case: all events become `n/a` and all values become `1`
    else:
        trial_type_col_name = None
        descrs = np.full(len(events_dict["onset"]), "n/a")
        event_id = {descrs[0]: 1}

    if trial_type_col_name is not None:
        # Drop events unrelated to a trial type
        events_dict = _drop(events_dict, "n/a", trial_type_col_name)
        trial_types = events_dict[trial_type_col_name]
        # handle event values (if provided); ensure pairings are 1 value per description
        if "value" in events_dict:
            values = np.asarray(events_dict["value"], dtype=str)
            for trial_type in np.unique(trial_types):
                idx = np.where(trial_type == np.atleast_1d(trial_types))[0]
                matching_values = values[idx]
                if len(np.unique(matching_values)) > 1:
                    # Event type descriptors are ambiguous; create hierarchical event
                    # descriptors (to ensure trial_type -> integerID is 1:1)
                    logger.info(
                        f'The event "{trial_type}" refers to multiple event values.'
                        "Creating hierarchical event names."
                    )
                    for ii in idx:
                        # strip `/` from `n/a` before incorporating into trial type name
                        value = values[ii] if values[ii] != "n/a" else "na"
                        new_name = f"{trial_type}/{value}"
                        logger.info(f"    Renaming event: {trial_type} -> {new_name}")
                        trial_types[ii] = new_name
            # make a copy with rows dropped where `value` is `n/a` (only for making our
            # `event_id` dict; `value = n/a` doesn't prevent making annotations).
            culled = _drop(events_dict, "n/a", "value")
            # Often (but not always!) the `value` column was written by MNE-BIDS and
            # represents integer event IDs (as would be found in MNE-Python events
            # arrays / event_id dicts). But in case not, let's be defensive:
            culled_vals = culled["value"]
            try:
                culled_vals = np.asarray(culled_vals, dtype=float)
            except ValueError:  # contained strings or complex numbers
                pass
            else:
                try:
                    culled_vals = culled_vals.astype(int)
                except ValueError:  # numeric, but has some non-integer values
                    pass
            event_id = dict(zip(culled[trial_type_col_name], culled_vals))
        else:
            event_id = dict(zip(trial_types, np.arange(len(trial_types))))
        descrs = np.asarray(trial_types, dtype=str)

    # convert onsets & durations to floats ("n/a" onsets were already dropped)
    ons = np.asarray(events_dict["onset"], dtype=float)
    durs = np.array(
        [0 if du == "n/a" else du for du in events_dict["duration"]], dtype=float
    )

    extras = None
    extra_columns = list(
        set(events_dict)
        - {
            "onset",
            "duration",
            "value",
            "trial_type",
            "stim_type",
            "sample",
        }
    )
    if extra_columns:
        extras = [
            dict(zip(extra_columns, values))
            for values in zip(*[events_dict[col] for col in extra_columns])
        ]

    return {
        "onset": ons,
        "duration": durs,
        "description": descrs,
        "event_id": event_id,
        "extras": extras,
    }


def _handle_events_reading(events_fname, raw):
    """Read associated events.tsv and convert valid events to annotations on Raw."""
    annotations_info = events_file_to_annotation_kwargs(events_fname)
    event_id = annotations_info["event_id"]

    # Add events as Annotations, but keep essential Annotations present in raw file
    annot_from_raw = raw.annotations.copy()
    try:
        annot_from_events = mne.Annotations(
            onset=annotations_info["onset"],
            duration=annotations_info["duration"],
            description=annotations_info["description"],
            extras=annotations_info["extras"],
        )
    except TypeError:
        if (
            annotations_info["extras"] is not None
            and len(annotations_info["extras"]) > 0
        ):
            warn(
                "The version of MNE-Python you are using (<1.10) "
                "does not support the extras argument in mne.Annotations. "
                f"The extra column(s) {list(annotations_info['extras'][0].keys())} "
                "will be ignored."
            )
        annot_from_events = mne.Annotations(
            onset=annotations_info["onset"],
            duration=annotations_info["duration"],
            description=annotations_info["description"],
        )
    raw.set_annotations(annot_from_events)

    annot_idx_to_keep = [
        idx
        for idx, descr in enumerate(annot_from_raw.description)
        if descr in ANNOTATIONS_TO_KEEP
    ]
    annot_to_keep = annot_from_raw[annot_idx_to_keep]

    if len(annot_to_keep):
        raw.set_annotations(raw.annotations + annot_to_keep)

    return raw, event_id


def _get_bads_from_tsv_data(tsv_data):
    """Extract names of bads from data read from channels.tsv."""
    idx = []
    for ch_idx, status in enumerate(tsv_data["status"]):
        if status.lower() == "bad":
            idx.append(ch_idx)

    bads = [tsv_data["name"][i] for i in idx]
    return bads


def _handle_channels_reading(channels_fname, raw):
    """Read associated channels.tsv and populate raw.

    Updates status (bad) and types of channels.
    """
    logger.info(f"Reading channel info from {channels_fname}.")
    channels_dict = _from_tsv(channels_fname)
    ch_names_tsv = channels_dict["name"]

    # Now we can do some work.
    # The "type" column is mandatory in BIDS. We can use it to set channel
    # types in the raw data using a mapping between channel types
    channel_type_bids_mne_map = dict()

    # Get the best mapping we currently have from BIDS to MNE nomenclature
    bids_to_mne_ch_types = _get_ch_type_mapping(fro="bids", to="mne")
    ch_types_json = channels_dict["type"]
    for ch_name, ch_type in zip(ch_names_tsv, ch_types_json):
        # We don't map MEG channels for now, as there's no clear 1:1 mapping
        # from BIDS to MNE coil types.
        if ch_type.upper() in (
            "MEGGRADAXIAL",
            "MEGMAG",
            "MEGREFGRADAXIAL",
            "MEGGRADPLANAR",
            "MEGREFMAG",
            "MEGOTHER",
        ):
            continue

        # Try to map from BIDS nomenclature to MNE, leave channel type
        # untouched if we are uncertain
        updated_ch_type = bids_to_mne_ch_types.get(ch_type, None)

        if updated_ch_type is None:
            # XXX Try again with uppercase spelling – this should be removed
            # XXX once https://github.com/bids-standard/bids-validator/issues/1018
            # XXX has been resolved.
            # XXX x-ref https://github.com/mne-tools/mne-bids/issues/481
            updated_ch_type = bids_to_mne_ch_types.get(ch_type.upper(), None)
            if updated_ch_type is not None:
                msg = (
                    "The BIDS dataset contains channel types in lowercase "
                    "spelling. This violates the BIDS specification and "
                    "will raise an error in the future."
                )
                warn(msg)

        if updated_ch_type is None:
            # We don't have an appropriate mapping, so make it a "misc" channel
            channel_type_bids_mne_map[ch_name] = "misc"
            warn(
                f'No BIDS -> MNE mapping found for channel type "{ch_type}". '
                f'Type of channel "{ch_name}" will be set to "misc".'
            )
        else:
            # We found a mapping, so use it
            channel_type_bids_mne_map[ch_name] = updated_ch_type

    # Special handling for (synthesized) stimulus channel
    synthesized_stim_ch_name = "STI 014"
    if (
        synthesized_stim_ch_name in raw.ch_names
        and synthesized_stim_ch_name not in ch_names_tsv
    ):
        logger.info(
            f'The stimulus channel "{synthesized_stim_ch_name}" is present in '
            f"the raw data, but not included in channels.tsv. Removing the "
            f"channel."
        )
        raw.drop_channels([synthesized_stim_ch_name])

    # Rename channels in loaded Raw to match those read from the BIDS sidecar
    if len(ch_names_tsv) != len(raw.ch_names):
        warn(
            f"The number of channels in the channels.tsv sidecar file "
            f"({len(ch_names_tsv)}) does not match the number of channels "
            f"in the raw data file ({len(raw.ch_names)}). Will not try to "
            f"set channel names."
        )
    else:
        raw.rename_channels(dict(zip(raw.ch_names, ch_names_tsv)))

    # Set the channel types in the raw data according to channels.tsv
    channel_type_bids_mne_map_available_channels = {
        ch_name: ch_type
        for ch_name, ch_type in channel_type_bids_mne_map.items()
        if ch_name in raw.ch_names
    }
    ch_diff = set(channel_type_bids_mne_map.keys()) - set(
        channel_type_bids_mne_map_available_channels.keys()
    )
    if ch_diff:
        warn(
            f"Cannot set channel type for the following channels, as they "
            f"are missing in the raw data: {', '.join(sorted(ch_diff))}"
        )
    raw.set_channel_types(
        channel_type_bids_mne_map_available_channels, on_unit_change="ignore"
    )

    # Set bad channels based on _channels.tsv sidecar
    if "status" in channels_dict:
        bads_tsv = _get_bads_from_tsv_data(channels_dict)
        bads_avail = [ch_name for ch_name in bads_tsv if ch_name in raw.ch_names]

        ch_diff = set(bads_tsv) - set(bads_avail)
        if ch_diff:
            warn(
                f'Cannot set "bad" status for the following channels, as '
                f"they are missing in the raw data: "
                f"{', '.join(sorted(ch_diff))}"
            )

        raw.info["bads"] = bads_avail

    return raw


@verbose
def read_raw_bids(
    bids_path, extra_params=None, *, return_event_dict=False, verbose=None
):
    """Read BIDS compatible data.

    Will attempt to read associated events.tsv and channels.tsv files to
    populate the returned raw object with raw.annotations and raw.info['bads'].

    Parameters
    ----------
    bids_path : BIDSPath
        The file to read. The :class:`mne_bids.BIDSPath` instance passed here
        **must** have the ``.root`` attribute set. The ``.datatype`` attribute
        **may** be set. If ``.datatype`` is not set and only one data type
        (e.g., only EEG or MEG data) is present in the dataset, it will be
        selected automatically.

        .. note::
           If ``bids_path`` points to a symbolic link of a ``.fif`` file
           without a ``split`` entity, the link will be resolved before
           reading.

    extra_params : None | dict
        Extra parameters to be passed to MNE read_raw_* functions.
        Note that the ``exclude`` parameter, which is supported by some
        MNE-Python readers, is not supported; instead, you need to subset
        your channels **after** reading.
    return_event_dict : bool
        Whether to return a dictionary that maps annotation descriptions to integer
        event IDs, in addition to the :class:`~mne.io.Raw` object. If a ``value`` column
        is present in the ``*_events.tsv`` file, it will be used as the source of the
        integer event ID values (events with ``value="n/a"`` will be omitted).
    %(verbose)s

    Returns
    -------
    raw : mne.io.Raw
        The data as MNE-Python Raw object.
    event_id : dict
        A mapping from event descriptions to integer event IDs, suitable for,
        e.g., passing to :func:`mne.events_from_annotations`. Only returned if
        ``return_event_dict=True``.

    Raises
    ------
    RuntimeError
        If multiple recording data types are present in the dataset, but
        ``datatype=None``.

    RuntimeError
        If more than one data files exist for the specified recording.

    RuntimeError
        If no data file in a supported format can be located.

    ValueError
        If the specified ``datatype`` cannot be found in the dataset.

    """
    if not isinstance(bids_path, BIDSPath):
        raise RuntimeError(
            '"bids_path" must be a BIDSPath object. Please '
            "instantiate using mne_bids.BIDSPath()."
        )
    for required in ["root", "subject", "task"]:
        if not getattr(bids_path, required):
            raise RuntimeError(
                '"bids_path" must contain `root`, `subject`, and `task` '
                f"attributes but it's missing `{required}`."
            )

    bids_path = bids_path.copy()
    sub = bids_path.subject
    ses = bids_path.session
    bids_root = bids_path.root
    datatype = bids_path.datatype
    suffix = bids_path.suffix

    # check root available
    if bids_root is None:
        raise ValueError(
            'The root of the "bids_path" must be set. '
            'Please use `bids_path.update(root="<root>")` '
            "to set the root of the BIDS folder to read."
        )

    # infer the datatype and suffix if they are not present in the BIDSPath
    if datatype is None:
        datatype = _infer_datatype(root=bids_root, sub=sub, ses=ses)
        bids_path.update(datatype=datatype)
    if suffix is None:
        bids_path.update(suffix=datatype)

    if bids_path.fpath.suffix == ".pdf":
        bids_raw_folder = bids_path.directory / f"{bids_path.basename}"

        # try to find the processed data file ("pdf")
        # see: https://www.fieldtriptoolbox.org/getting_started/bti/
        bti_pdf_patterns = ["0", "c,rf*", "hc,rf*", "e,rf*"]
        pdf_list = []
        for pattern in bti_pdf_patterns:
            pdf_list += sorted(bids_raw_folder.glob(pattern))

        if len(pdf_list) == 0:
            raise RuntimeError(
                "Cannot find BTi 'processed data file' (pdf). Please open an issue on "
                "the mne-bids repository to discuss with the developers:\n\n"
                "https://github.com/mne-tools/mne-bids/issues/new/choose\n\n"
                f"No matches for following patterns:\n\n{bti_pdf_patterns}\n\n"
                f"In: {bids_raw_folder}"
            )
        elif len(pdf_list) > 1:  # pragma: no cover
            logger.warn(
                "Found more than one BTi 'processed data file' (pdf). "
                f"Picking:\n\n    {pdf_list[0]}\n\nout of the options:\n\n"
                f"{pdf_list}\n\n"
            )
        raw_path = pdf_list[0]
        config_path = bids_raw_folder / "config"
    else:
        raw_path = bids_path.fpath
        # Resolve for FIFF files
        if (
            raw_path.suffix == ".fif"
            and bids_path.split is None
            and raw_path.is_symlink()
        ):
            target_path = raw_path.resolve()
            logger.info(f"Resolving symbolic link: {raw_path} -> {target_path}")
            raw_path = target_path
        config_path = None

    # Special-handle EDF filenames: we accept upper- and lower-case extensions
    if raw_path.suffix.lower() == ".edf":
        for extension in (".edf", ".EDF"):
            candidate_path = raw_path.with_suffix(extension)
            if candidate_path.exists():
                raw_path = candidate_path
                break

    if not raw_path.exists():
        options = os.listdir(bids_path.directory)
        matches = get_close_matches(bids_path.basename, options)
        msg = f"File does not exist:\n{raw_path}"
        if matches:
            msg += (
                "\nDid you mean one of:\n"
                + "\n".join(matches)
                + "\ninstead of:\n"
                + bids_path.basename
            )
        raise FileNotFoundError(msg)
    if config_path is not None and not config_path.exists():
        raise FileNotFoundError(f"config directory not found: {config_path}")

    if extra_params is None:
        extra_params = dict()
    elif "exclude" in extra_params:
        del extra_params["exclude"]
        logger.info('"exclude" parameter is not supported by read_raw_bids')

    if raw_path.suffix == ".fif" and "allow_maxshield" not in extra_params:
        extra_params["allow_maxshield"] = True
    raw = _read_raw(
        raw_path,
        electrode=None,
        hsp=None,
        hpi=None,
        config_path=config_path,
        **extra_params,
    )

    # Try to find an associated events.tsv to get information about the
    # events in the recorded data
    if (
        bids_path.subject == "emptyroom" and bids_path.task == "noise"
    ) or bids_path.task.startswith("rest"):
        on_error = "ignore"
    else:
        on_error = "warn"

    events_fname = _find_matching_sidecar(
        bids_path, suffix="events", extension=".tsv", on_error=on_error
    )
    if events_fname is not None:
        raw, event_id = _handle_events_reading(events_fname, raw)

    # Try to find an associated channels.tsv to get information about the
    # status and type of present channels
    channels_fname = _find_matching_sidecar(
        bids_path, suffix="channels", extension=".tsv", on_error="warn"
    )
    if channels_fname is not None:
        raw = _handle_channels_reading(channels_fname, raw)

    # Try to find an associated electrodes.tsv and coordsystem.json
    # to get information about the status and type of present channels
    on_error = "warn" if suffix == "ieeg" else "ignore"
    electrodes_fname = _find_matching_sidecar(
        bids_path, suffix="electrodes", extension=".tsv", on_error=on_error
    )
    coordsystem_fname = _find_matching_sidecar(
        bids_path, suffix="coordsystem", extension=".json", on_error=on_error
    )
    if electrodes_fname is not None:
        if coordsystem_fname is None:
            raise RuntimeError(
                f"BIDS mandates that the coordsystem.json "
                f"should exist if electrodes.tsv does. "
                f"Please create coordsystem.json for"
                f"{bids_path.basename}"
            )
        if datatype in ["meg", "eeg", "ieeg"]:
            _read_dig_bids(
                electrodes_fname, coordsystem_fname, raw=raw, datatype=datatype
            )

    # Try to find an associated sidecar .json to get information about the
    # recording snapshot
    sidecar_fname = _find_matching_sidecar(
        bids_path, suffix=datatype, extension=".json", on_error="warn"
    )
    if sidecar_fname is not None:
        raw = _handle_info_reading(sidecar_fname, raw)

    # read in associated scans filename
    scans_fname = BIDSPath(
        subject=bids_path.subject,
        session=bids_path.session,
        suffix="scans",
        extension=".tsv",
        root=bids_path.root,
    ).fpath

    if scans_fname.exists():
        raw = _handle_scans_reading(scans_fname, raw, bids_path)

    # read in associated subject info from participants.tsv
    participants_tsv_path = bids_root / "participants.tsv"
    subject = f"sub-{bids_path.subject}"
    if participants_tsv_path.exists():
        raw = _handle_participants_reading(
            participants_fname=participants_tsv_path, raw=raw, subject=subject
        )
    else:
        warn(f"participants.tsv file not found for {raw_path}")
        raw.info["subject_info"] = dict()

    assert raw.annotations.orig_time == raw.info["meas_date"]
    if return_event_dict:
        return raw, event_id
    return raw


@verbose
def get_head_mri_trans(
    bids_path,
    extra_params=None,
    t1_bids_path=None,
    fs_subject=None,
    fs_subjects_dir=None,
    *,
    kind=None,
    verbose=None,
):
    """Produce transformation matrix from MEG and MRI landmark points.

    Will attempt to read the landmarks of Nasion, LPA, and RPA from the sidecar
    files of (i) the MEG and (ii) the T1-weighted MRI data. The two sets of
    points will then be used to calculate a transformation matrix from head
    coordinates to MRI coordinates.

    .. note:: The MEG and MRI data need **not** necessarily be stored in the
              same session or even in the same BIDS dataset. See the
              ``t1_bids_path`` parameter for details.

    Parameters
    ----------
    bids_path : BIDSPath
        The path of the electrophysiology recording. If ``datatype`` and
        ``suffix`` are not present, they will be set to ``'meg'``, and a
        warning will be raised.

        .. versionchanged:: 0.10
           A warning is raised it ``datatype`` or ``suffix`` are not set.
    extra_params : None | dict
        Extra parameters to be passed to :func:`mne.io.read_raw` when reading
        the MEG file.
    t1_bids_path : BIDSPath | None
        If ``None`` (default), will try to discover the T1-weighted MRI file
        based on the name and location of the MEG recording specified via the
        ``bids_path`` parameter. Alternatively, you explicitly specify which
        T1-weighted MRI scan to use for extraction of MRI landmarks. To do
        that, pass a :class:`mne_bids.BIDSPath` pointing to the scan.
        Use this parameter e.g. if the T1 scan was recorded during a different
        session than the MEG. It is even possible to point to a T1 image stored
        in an entirely different BIDS dataset than the MEG data.
    fs_subject : str
        The subject identifier used for FreeSurfer.

        .. versionchanged:: 0.10
           Does not default anymore to ``bids_path.subject`` if ``None``.
    fs_subjects_dir : path-like | None
        The FreeSurfer subjects directory. If ``None``, defaults to the
        ``SUBJECTS_DIR`` environment variable.

        .. versionadded:: 0.8
    kind : str | None
        The suffix of the anatomical landmark names in the JSON sidecar.
        A suffix might be present e.g. to distinguish landmarks between
        sessions. If provided, should not include a leading underscore ``_``.
        For example, if the landmark names in the JSON sidecar file are
        ``LPA_ses-1``, ``RPA_ses-1``, ``NAS_ses-1``, you should pass
        ``'ses-1'`` here.
        If ``None``, no suffix is appended, the landmarks named
        ``Nasion`` (or ``NAS``), ``LPA``, and ``RPA`` will be used.

        .. versionadded:: 0.10
    %(verbose)s

    Returns
    -------
    trans : mne.transforms.Transform
        The data transformation matrix from head to MRI coordinates.
    """
    nib = _import_nibabel("get a head to MRI transform")

    if not isinstance(bids_path, BIDSPath):
        raise RuntimeError(
            '"bids_path" must be a BIDSPath object. Please '
            "instantiate using mne_bids.BIDSPath()."
        )

    # check root available
    meg_bids_path = bids_path.copy()
    del bids_path
    if meg_bids_path.root is None:
        raise ValueError(
            'The root of the "bids_path" must be set. '
            'Please use `bids_path.update(root="<root>")` '
            "to set the root of the BIDS folder to read."
        )

    # if the bids_path is underspecified, only get info for MEG data
    if meg_bids_path.datatype is None:
        meg_bids_path.datatype = "meg"
        warn(
            'bids_path did not have a datatype set. Assuming "meg". This '
            "will raise an exception in the future.",
            module="mne_bids",
            category=DeprecationWarning,
        )
    if meg_bids_path.suffix is None:
        meg_bids_path.suffix = "meg"
        warn(
            'bids_path did not have a suffix set. Assuming "meg". This '
            "will raise an exception in the future.",
            module="mne_bids",
            category=DeprecationWarning,
        )

    # Get the sidecar file for MRI landmarks
    t1w_bids_path = (
        (meg_bids_path if t1_bids_path is None else t1_bids_path)
        .copy()
        .update(datatype="anat", suffix="T1w", task=None)
    )
    t1w_json_path = _find_matching_sidecar(
        bids_path=t1w_bids_path, extension=".json", on_error="ignore"
    )
    del t1_bids_path

    if t1w_json_path is not None:
        t1w_json_path = Path(t1w_json_path)

    if t1w_json_path is None or not t1w_json_path.exists():
        raise FileNotFoundError(
            f"Did not find T1w JSON sidecar file, tried location: {t1w_json_path}"
        )
    for extension in (".nii", ".nii.gz"):
        t1w_path_candidate = t1w_json_path.with_suffix(extension)
        if t1w_path_candidate.exists():
            t1w_bids_path = get_bids_path_from_fname(fname=t1w_path_candidate)
            break

    if not t1w_bids_path.fpath.exists():
        raise FileNotFoundError(
            f"Did not find T1w recording file, tried location: "
            f"{t1w_path_candidate.name.replace('.nii.gz', '')}[.nii, .nii.gz]"
        )

    # Get MRI landmarks from the JSON sidecar
    t1w_json = json.loads(t1w_json_path.read_text(encoding="utf-8"))
    mri_coords_dict = t1w_json.get("AnatomicalLandmarkCoordinates", dict())

    # landmarks array: rows: [LPA, NAS, RPA]; columns: [x, y, z]
    suffix = f"_{kind}" if kind is not None else ""
    mri_landmarks = np.full((3, 3), np.nan)
    for landmark_name, coords in mri_coords_dict.items():
        if landmark_name.upper() == ("LPA" + suffix).upper():
            mri_landmarks[0, :] = coords
        elif landmark_name.upper() == ("RPA" + suffix).upper():
            mri_landmarks[2, :] = coords
        elif (
            landmark_name.upper() == ("NAS" + suffix).upper()
            or landmark_name.lower() == ("nasion" + suffix).lower()
        ):
            mri_landmarks[1, :] = coords
        else:
            continue

    if np.isnan(mri_landmarks).any():
        raise RuntimeError(
            f"Could not extract fiducial points from T1w sidecar file: "
            f"{t1w_json_path}\n\n"
            f"The sidecar file SHOULD contain a key "
            f'"AnatomicalLandmarkCoordinates" pointing to an '
            f'object with the keys "LPA", "NAS", and "RPA". '
            f"Yet, the following structure was found:\n\n"
            f"{mri_coords_dict}"
        )

    # The MRI landmarks are in "voxels". We need to convert them to the
    # Neuromag RAS coordinate system in order to compare them with MEG
    # landmarks. See also: `mne_bids.write.write_anat`
    if fs_subject is None:
        warn(
            'Passing "fs_subject=None" has been deprecated and will raise '
            "an error in future versions. Please explicitly specify the "
            "FreeSurfer subject name.",
            DeprecationWarning,
        )
        fs_subject = f"sub-{meg_bids_path.subject}"

    fs_subjects_dir = get_subjects_dir(fs_subjects_dir, raise_error=False)
    fs_t1_path = fs_subjects_dir / fs_subject / "mri" / "T1.mgz"
    if not fs_t1_path.exists():
        raise ValueError(
            f"Could not find {fs_t1_path}. Consider running FreeSurfer's "
            f"'recon-all` for subject {fs_subject}."
        )
    fs_t1_mgh = nib.load(str(fs_t1_path))
    t1_nifti = nib.load(str(t1w_bids_path.fpath))

    # Convert to MGH format to access vox2ras method
    t1_mgh = nib.MGHImage(t1_nifti.dataobj, t1_nifti.affine)

    # convert to scanner RAS
    mri_landmarks = apply_trans(t1_mgh.header.get_vox2ras(), mri_landmarks)

    # convert to FreeSurfer T1 voxels (same scanner RAS as T1)
    mri_landmarks = apply_trans(fs_t1_mgh.header.get_ras2vox(), mri_landmarks)

    # now extract transformation matrix and put back to RAS coordinates of MRI
    vox2ras_tkr = fs_t1_mgh.header.get_vox2ras_tkr()
    mri_landmarks = apply_trans(vox2ras_tkr, mri_landmarks)
    mri_landmarks = mri_landmarks * 1e-3

    # Get MEG landmarks from the raw file
    _, ext = _parse_ext(meg_bids_path)
    if extra_params is None:
        extra_params = dict()
    if ext == ".fif":
        extra_params["allow_maxshield"] = "yes"

    raw = read_raw_bids(bids_path=meg_bids_path, extra_params=extra_params)

    if (
        raw.get_montage() is None
        or raw.get_montage().get_positions() is None
        or any(
            [
                raw.get_montage().get_positions()[fid_key] is None
                for fid_key in ("nasion", "lpa", "rpa")
            ]
        )
    ):
        raise RuntimeError(
            f"Could not extract fiducial points from ``raw`` file: "
            f"{meg_bids_path}\n\n"
            f"The ``raw`` file SHOULD contain digitization points "
            "for the nasion and left and right pre-auricular points "
            "but none were found"
        )
    pos = raw.get_montage().get_positions()
    meg_landmarks = np.asarray((pos["lpa"], pos["nasion"], pos["rpa"]))

    # Given the two sets of points, fit the transform
    trans_fitted = fit_matched_points(src_pts=meg_landmarks, tgt_pts=mri_landmarks)
    trans = mne.transforms.Transform(fro="head", to="mri", trans=trans_fitted)
    return trans
