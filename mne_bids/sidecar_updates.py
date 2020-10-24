"""Update BIDS directory structures and sidecar files meta data."""
# Authors: Adam Li <adam2392@gmail.com>
#          Austin Hurst <mynameisaustinhurst@gmail.com>
# License: BSD (3-clause)

import json
from collections import OrderedDict

from mne_bids.utils import _write_json


# TODO: add support for tsv files
def update_sidecars(bids_path, sidecar_template_fpath, verbose=True):
    """Update sidecar files using template JSON file.

    Currently, only works for JSON files.

    Parameters
    ----------
    bids_path : BIDSPath
    sidecar_template_fpath : str | pathlib.Path
    verbose : bool

    Examples
    --------
    >>> # update all files in BIDS dataset
    >>> bids_path = BIDSPath(root='./')
    >>> update_sidecars(bids_path, sidecar_template_fpath)
    """
    # get all matching json files
    bids_path.update(extension='.json')
    sidecar_paths = bids_path.match()

    # load sidecar template
    with open(sidecar_template_fpath, 'r') as tmp_f:
        sidecar_tmp = json.load(tmp_f,
                                object_pairs_hook=OrderedDict
                                )
    template_fields = set(sidecar_tmp.keys())

    if verbose:
        print(sidecar_tmp)
        print(f'Updating {sidecar_paths}...')

    for sidecar_bids_path in sidecar_paths:
        fpath = sidecar_bids_path.fpath

        # load in sidecar filepath
        with open(fpath, 'r') as tmp_f:
            sidecar_json = json.load(
                tmp_f, object_pairs_hook=OrderedDict)

        # get the fields inside JSON
        default_fields = set(sidecar_json.keys())

        # Use field order in template to sort default keys, if values are None
        not_in_template = default_fields.difference(template_fields)
        only_in_template = template_fields.difference(default_fields)
        in_both = template_fields.intersection(default_fields)
        for field in only_in_template:
            if sidecar_tmp[field] is None:
                sidecar_tmp.pop(field)
        field_order = list(not_in_template) + list(sidecar_tmp.keys())
        for field in in_both:
            if sidecar_tmp[field] is None:
                sidecar_tmp.pop(field)

        # Update values in generate sidecar with non-None values in template
        sidecar_json.update(sidecar_tmp)

        # Sort updated sidecar according to sort order in template
        sorted_info = [(field, sidecar_json[field]) for field in field_order]
        sidecar_json = OrderedDict(sorted_info)

        _write_json(fpath, sidecar_json, overwrite=True, verbose=verbose)
