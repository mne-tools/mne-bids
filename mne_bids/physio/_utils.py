import json
from warnings import warn

from mne.utils import _validate_type


def _get_physio_type(physio_json_fpath):
    """Return the type of data that is stored in a <match>_physio.tsv file.

    https://bids-specification.readthedocs.io/en/latest/glossary.html#physiotypegeneric-enums
    """  # noqa: E501
    default = "generic"
    fpath = physio_json_fpath  # shorter
    contents = json.loads(fpath.read_text())
    physio_type = contents.get("PhysioType", None)  # e.g. "eyetracking"

    if not physio_type:
        warn(
            "Expected a key labeled 'PhysioType', with a value such as 'eyetrack', but "
            f"none exists. Falling back to {default}:\n  Files {fpath.name}"
        )
        physio_type = default
    _validate_type(physio_type, str, "physio_type")
    return physio_type
