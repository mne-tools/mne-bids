"""Update BIDS directory structures and sidecar files meta data."""
# Authors: Adam Li <adam2392@gmail.com>
#          Austin Hurst <mynameisaustinhurst@gmail.com>
#          mne-bids developers
# License: BSD (3-clause)

import json
from collections import OrderedDict

from mne.utils import logger

from mne_bids.utils import _write_json


# TODO: add support for tsv files
def update_sidecar_json(bids_path, entries, verbose=True):
    """Update sidecar files using a dictionary or JSON file.

    Will update metadata fields inside the path defined by
    ``bids_path.fpath`` according to the ``entries``. If a
    field does not exist in the corresponding sidecar file,
    then that field will be created according to the ``entries``.
    If a field does exist in the corresponding sidecar file,
    then that field will be updated according to the ``entries``.

    For example, if ``InstitutionName`` is not defined in
    the sidecar json file, then trying to update
    ``InstitutionName`` to ``Martinos Center`` will update
    the sidecar json file to have ``InstitutionName`` as
    ``Martinos Center``.

    Parameters
    ----------
    bids_path : BIDSPath
        The set of paths to update. The :class:`mne_bids.BIDSPath` instance
        passed here **must** have the ``.root`` attribute set. The
        ``.datatype`` attribute **may** be set. If ``.datatype`` is
        not set and only one data type (e.g., only EEG or MEG data)
        is present in the dataset, it will be
        selected automatically. This must uniquely identify
        an existing file path, else an error will be raised.
    entries : dict | str | pathlib.Path
        A dictionary, or JSON file that defines the
        sidecar fields and corresponding values to be updated to.
    verbose : bool
        The verbosity level.

    Notes
    -----
    This function can only update JSON files.

    Sidecar JSON files include files such as ``*_ieeg.json``,
    ``*_coordsystem.json``, ``*_scans.json``, etc.

    You should double check that your update dictionary is correct
    for the corresponding sidecar JSON file because it will perform
    a dictionary update of the sidecar fields according to
    the passed in dictionary overwriting any information that was
    previously there.

    Raises
    ------
    RuntimeError
        If the specified ``bids_path.fpath`` cannot be found
        in the dataset.

    RuntimeError
        If the ``bids_path.fpath`` does not have ``.json``
        extension.

    Examples
    --------
    >>> # update sidecar json file
    >>> bids_path = BIDSPath(root='./', subject='001', session='001',
                             task='test', run='01', suffix='ieeg',
                             extension='.json')
    >>> entries = {'PowerLineFrequency': 50}
    >>> update_sidecar_json(bids_path, entries)
    >>> # update sidecar coordsystem json file
    >>> bids_path = BIDSPath(root='./', subject='001', session='001',
                             suffix='coordsystem', extension='.json')
    >>> entries = {'iEEGCoordinateSystem,': 'Other'}
    >>> update_sidecar_json(bids_path, entries)
    """
    # get all matching json files
    bids_path = bids_path.copy()
    if bids_path.extension != '.json':
        raise RuntimeError('Only works for ".json" files. The '
                           'BIDSPath object passed in has '
                           f'{bids_path.extension} extension.')

    # get the file path
    fpath = bids_path.fpath
    if not fpath.exists():
        raise RuntimeError(f'Sidecar file does not '
                           f'exist for {fpath}.')

    # sidecar update either from file, or as dictionary
    if isinstance(entries, dict):
        sidecar_tmp = entries
    else:
        with open(entries, 'r') as tmp_f:
            sidecar_tmp = json.load(
                tmp_f, object_pairs_hook=OrderedDict)

    if verbose:
        logger.debug(sidecar_tmp)
        logger.debug(f'Updating {fpath}...')

    # load in sidecar filepath
    with open(fpath, 'r') as tmp_f:
        sidecar_json = json.load(
            tmp_f, object_pairs_hook=OrderedDict)

    # update sidecar JSON file with the fields passed in
    sidecar_json.update(**sidecar_tmp)

    # write back the sidecar JSON
    _write_json(fpath, sidecar_json, overwrite=True, verbose=verbose)


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
    with open(sidecar_fname, 'r', encoding='utf-8-sig') as fin:
        sidecar_json = json.load(fin)
    sidecar_json[key] = val
    with open(sidecar_fname, 'w', encoding='utf-8') as fout:
        json.dump(sidecar_json, fout)
