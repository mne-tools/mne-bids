"""Utilities for reading 'generic' physiological BIDS data."""

import json
from pathlib import Path
from warnings import warn

import numpy as np
from mne.utils import _check_option, _validate_type


def _get_physio_type(physio_json_fpath):
    """Return the type of data that is stored in a <match>_physio.tsv file.

    https://bids-specification.readthedocs.io/en/latest/glossary.html#physiotypegeneric-enums
    """  # noqa: E501
    default = "generic"
    info = _read_json(physio_json_fpath)
    physio_type = info.get("PhysioType", None)  # e.g. "eyetrack"

    if not physio_type:
        warn(
            "Expected a key labeled 'PhysioType', with a value such as 'eyetrack', but "
            f"none exists. Falling back to {default}:\n  Files {physio_json_fpath.name}"
        )
        physio_type = default
    _validate_type(physio_type, str, "physio_type")
    return physio_type


def _read_json(physio_json_fpath):
    physio_json_fpath = Path(physio_json_fpath)
    encoding = "utf-8-sig"
    info = json.loads(physio_json_fpath.read_text(encoding=encoding))
    return info


def _read_physioevents(bids_path, ch_names="auto"):
    """Read a ``*_physioevents.tsv(.gz)`` sidecar and return annotation kwargs.

    Parameters
    ----------
    bids_path : mne_bids.BIDSPath
        Path that resolves to one physio recording (e.g. ``recording-eye1``).
    ch_names : tuple of str | list of str | None | "auto"
        Channel names to associate with each event annotation. If ``"auto"``,
        channel names are inferred from the corresponding ``*_physio.json``
        sidecar ``"Columns"`` key exactly as provided. If ``None``, no specific
        channel names will be associated with the annotations.

    Returns
    -------
    annot_kwargs : dict of ndarray
        Keyword arguments suitable for ``mne.Annotations(**annot_kwargs)``.
    """
    from mne_bids.read import _make_annotation_kwargs
    from mne_bids.tsv_handler import _from_tsv

    encoding = "utf-8-sig"

    _validate_type(ch_names, (list, tuple, str, type(None)), item_name="ch_names")

    # Make annotations channel-agnostic
    if ch_names is None:
        ch_names = tuple()
    # Pull names from TSV file. Assumes MNE ch_names are identical to these column names
    elif ch_names == "auto":
        physio_json_fpath = bids_path.find_matching_sidecar(
            suffix="physio", extension=".json"
        )
        physio_sidecar = json.loads(physio_json_fpath.read_text(encoding=encoding))
        ch_names = tuple(physio_sidecar.get("Columns", []))
    # MNE channel names were passed in from the call site
    else:
        ch_names = tuple(ch_names)

    events_fname = bids_path.find_matching_sidecar(
        suffix="physioevents",
        extension=".tsv.gz",
    )

    events = _from_tsv(events_fname)
    events.setdefault("duration", ["0"] * len(events["onset"]))  # optional column
    annot_kwargs, _ = _make_annotation_kwargs(events, events_fname)
    physio_sidecar = json.loads(
        bids_path.find_matching_sidecar(suffix="physio", extension=".json").read_text(
            encoding=encoding
        )
    )
    events_json = events_fname.with_suffix("").with_suffix(".json")
    source = json.loads(events_json.read_text(encoding=encoding))["OnsetSource"]
    sfreq = physio_sidecar["SamplingFrequency"]
    if source == "n/a":  # sample indices
        annot_kwargs["onset"] = annot_kwargs["onset"] / sfreq
    else:  # values of a column of the physio file, e.g. device timestamps
        timestamps = np.asarray(_from_tsv(bids_path.fpath, dtypes=float)[source])
        unit = physio_sidecar.get(source, {}).get("Units")
        if unit is None:  # without units, use the sample index of each onset
            annot_kwargs["onset"] = (
                np.searchsorted(timestamps, annot_kwargs["onset"]) / sfreq
            )
        else:
            scales = {"s": 1, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "ns": 1e-9}
            _check_option("OnsetSource units", unit, scales)
            annot_kwargs["onset"] = (annot_kwargs["onset"] - timestamps[0]) * scales[
                unit
            ]
    n_annots = len(annot_kwargs["onset"])
    if ch_names:
        # This is the best way I could work out to create a 1D array of tuples
        tuples = [ch_names] * n_annots
        arr = np.empty(len(tuples), dtype=object)
        arr[:] = tuples
        annot_kwargs["ch_names"] = arr

    return annot_kwargs
