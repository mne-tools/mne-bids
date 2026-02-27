import json
from pathlib import Path
from warnings import warn

import numpy as np
from mne.utils import _validate_type


def _get_physio_type(physio_json_fpath):
    """Return the type of data that is stored in a <match>_physio.tsv file.

    https://bids-specification.readthedocs.io/en/latest/glossary.html#physiotypegeneric-enums
    """  # noqa: E501
    default = "generic"
    info = _read_json(physio_json_fpath)
    physio_type = info.get("PhysioType", None)  # e.g. "eyetracking"

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
    from mne_bids.read import events_file_to_annotation_kwargs

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

    annot_kwargs = events_file_to_annotation_kwargs(events_fname)
    n_annots = len(annot_kwargs["onset"])
    if ch_names:
        # This is the best way I could work out to create a 1D array of tuples
        tuples = [ch_names] * n_annots
        arr = np.empty(len(tuples), dtype=object)
        arr[:] = tuples
        annot_kwargs["ch_names"] = arr

    return annot_kwargs
