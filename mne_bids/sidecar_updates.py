"""Update BIDS directory structures and sidecar files meta data."""
# Authors: Adam Li <adam2392@gmail.com>
#          Austin Hurst <mynameisaustinhurst@gmail.com>
# License: BSD (3-clause)

import json
from collections import OrderedDict

from mne.utils import warn

from mne_bids.utils import _write_json


# TODO: add support for tsv files
def update_sidecars(bids_path, sidecar_template, verbose=True):
    """Update sidecar files using template JSON file.

    Will match all paths based on BIDS entities defined in the ``bids_path``
    argument and update metadata fields according to the ``sidecar_template``.
    If a field does not exist in the corresponding sidecar file, then that
    field will not be updated. For example, if ``InstitutionName`` is
    not defined in the sidecar json file, then trying to update
    ``InstitutionName`` to ``Martinos Center`` will not update
    the sidecar json file.

    Currently, only works for JSON files.

    Parameters
    ----------
    bids_path : BIDSPath
        The set of paths to update. The :class:`mne_bids.BIDSPath` instance
        passed here **must** have the ``.root`` attribute set. The
        ``.datatype`` attribute **may** be set. If ``.datatype`` is
        not set and only one data type (e.g., only EEG or MEG data)
        is present in the dataset, it will be
        selected automatically. All matching bids paths via
        ``bids_path.match()`` will be updated according to
        the ``sidecar_template``.
    sidecar_template : dict | str | pathlib.Path
        A dictionary, or JSON file that defines the
        sidecar fields and corresponding values to be updated to.
    verbose : bool
        The verbosity level.

    Examples
    --------
    >>> # update all files in BIDS dataset
    >>> bids_path = BIDSPath(root='./')
    >>> update_sidecars(bids_path, sidecar_template_fpath)
    """
    # get all matching json files
    bids_path = bids_path.copy()
    bids_path.update(extension='.json')
    sidecar_paths = bids_path.match(return_json=True)

    if isinstance(sidecar_template, dict):
        sidecar_tmp = sidecar_template
    else:
        # load sidecar template
        with open(sidecar_template, 'r') as tmp_f:
            sidecar_tmp = json.load(tmp_f,
                                    object_pairs_hook=OrderedDict
                                    )

    template_fields = set(sidecar_tmp.keys())

    if verbose:
        print(sidecar_tmp)
        print(f'Updating {sidecar_paths}...')

    # keep track of all used fields in the template
    used_fields = []

    # update all matching sidecar paths
    for sidecar_bids_path in sidecar_paths:
        fpath = sidecar_bids_path.fpath

        # load in sidecar filepath
        with open(fpath, 'r') as tmp_f:
            sidecar_json = json.load(
                tmp_f, object_pairs_hook=OrderedDict)

        # get the fields inside JSON
        default_fields = set(sidecar_json.keys())

        # XXX: how to we do this if fields are not in the sidecar file already?

        # get fields that are not in sidecar file
        # only_in_template = template_fields.difference(default_fields)
        # get fields that are in both the sidecar file and template
        in_both = template_fields.intersection(default_fields)

        # keep track of unused fields
        for key in in_both:
            used_fields[key] = 1

        # Use field order in template to sort default keys, if values are None
        field_order = list(sidecar_tmp.keys())

        # pop fields in the template if they are overwriting value to None
        for field in in_both:
            if sidecar_tmp[field] is None:
                sidecar_tmp.pop(field)

        # Update values in generate sidecar with non-None values in template
        sidecar_json.update(sidecar_tmp)

        # Sort updated sidecar according to sort order in template
        sorted_info = [(field, sidecar_json[field]) for field in field_order]
        sidecar_json = OrderedDict(sorted_info)

        _write_json(fpath, sidecar_json, overwrite=True, verbose=verbose)

    # warn user that there are some unused fields in the sidecar template
    # that need to first be instantiated in the file before it can update
    unused_fields = [key for key in sidecar_tmp.keys()
                     if key not in used_fields]
    warn('Leftover template fields were not used to update any '
         f'sidecar file {unused_fields}')


def _update_sidecar(sidecar_fname, key, val):
    """Update a sidecar JSON file with a given key/value pair.

    Parameters
    ----------
    sidecar_fname : str | os.PathLike
        Full name of the data file
    key : str
        The key in the sidecar JSON file. E.g. "PowerLineFrequency"
    val : str
        The corresponding value to change to in the sidecar JSON file.
    """
    with open(sidecar_fname, "r") as fin:
        sidecar_json = json.load(fin)
    sidecar_json[key] = val
    with open(sidecar_fname, "w") as fout:
        json.dump(sidecar_json, fout)
