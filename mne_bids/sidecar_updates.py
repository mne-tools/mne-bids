"""Update BIDS directory structures and sidecar files meta data."""
# Authors: Adam Li <adam2392@gmail.com>
#          Austin Hurst <mynameisaustinhurst@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          mne-bids developers
#
# License: BSD-3-Clause

import json
from collections import OrderedDict

from mne.channels import DigMontage
from mne.utils import logger, _validate_type, verbose
from mne_bids import BIDSPath
from mne_bids.utils import _write_json


# TODO: add support for tsv files
@verbose
def update_sidecar_json(bids_path, entries, verbose=None):
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
    %(verbose)s

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
    Update a sidecar JSON file

    >>> from pathlib import Path
    >>> root = Path('./mne_bids/tests/data/tiny_bids').absolute()
    >>> bids_path = BIDSPath(subject='01', task='rest', session='eeg',
    ...                      suffix='eeg', extension='.json', root=root)
    >>> entries = {'PowerLineFrequency': 60}
    >>> update_sidecar_json(bids_path, entries, verbose=False)

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

    logger.debug(sidecar_tmp)
    logger.debug(f'Updating {fpath}...')

    # load in sidecar filepath
    with open(fpath, 'r') as tmp_f:
        sidecar_json = json.load(
            tmp_f, object_pairs_hook=OrderedDict)

    # update sidecar JSON file with the fields passed in
    sidecar_json.update(**sidecar_tmp)

    # write back the sidecar JSON
    _write_json(fpath, sidecar_json, overwrite=True)


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
    _write_json(sidecar_fname, sidecar_json, overwrite=True)


@verbose
def update_anat_landmarks(bids_path, landmarks, verbose=None):
    """Update the anatomical landmark coordinates of an MRI scan.

    This will change the ``AnatomicalLandmarkCoordinates`` entry in the
    respective JSON sidecar file, or create it if it doesn't exist.

    Parameters
    ----------
    bids_path : BIDSPath
        Path of the MR image.
    landmarks : mne.channels.DigMontage
        An :class:`mne.channels.DigMontage` instance with coordinates for the
        nasion and left and right pre-auricular points in MRI voxel
        coordinates.

        .. note:: :func:`mne_bids.get_anat_landmarks` provides a convenient and
                  reliable way to generate the landmark coordinates in the
                  required coordinate system.
    %(verbose)s

    Notes
    -----
    .. versionadded:: 0.8
    """
    _validate_type(item=bids_path, types=BIDSPath, item_name='bids_path')
    _validate_type(item=landmarks, types=DigMontage, item_name='landmarks')

    # Do some path verifications and fill in some gaps the users might have
    # left (datatype and extension)
    # XXX We could be more stringent (and less user-friendly) and insist on a
    # XXX full specification of all parts of the BIDSPath, thoughts?
    bids_path_mri = bids_path.copy()
    if bids_path_mri.datatype is None:
        bids_path_mri.datatype = 'anat'

    if bids_path_mri.datatype != 'anat':
        raise ValueError(
            f'Can only operate on "anat" MRI data, but the provided bids_path '
            f'points to: {bids_path_mri.datatype}')

    if bids_path_mri.suffix is None:
        raise ValueError('Please specify the "suffix" entity of the provided '
                         'bids_path.')
    elif bids_path_mri.suffix not in ('T1w', 'FLASH'):
        raise ValueError(
            f'Can only operate on "T1w" and "FLASH" images, but the bids_path '
            f'suffix indicates: {bids_path_mri.suffix}')

    valid_extensions = ('.nii', '.nii.gz')
    tried_paths = []
    file_exists = False
    if bids_path_mri.extension is None:
        # No extension was provided, start searching …
        for extension in valid_extensions:
            bids_path_mri.extension = extension
            tried_paths.append(bids_path_mri.fpath)

            if bids_path_mri.fpath.exists():
                file_exists = True
                break
    else:
        # An extension was provided
        tried_paths.append(bids_path_mri.fpath)
        if bids_path_mri.fpath.exists():
            file_exists = True

    if not file_exists:
        raise ValueError(
            f'Could not find an MRI scan. Please check the provided '
            f'bids_path. Tried the following filenames: '
            f'{", ".join([p.name for p in tried_paths])}')

    positions = landmarks.get_positions()
    coord_frame = positions['coord_frame']
    if coord_frame != 'mri_voxel':
        raise ValueError(
            f'The landmarks must be specified in MRI voxel coordinates, but '
            f'provided DigMontage is in "{coord_frame}"')

    # Extract the cardinal points
    name_to_coords_map = {
        'LPA': positions['lpa'],
        'NAS': positions['nasion'],
        'RPA': positions['rpa']
    }

    # Check if coordinates for any cardinal point are missing, and convert to
    # a list so we can easily store the data in JSON format
    missing_points = []
    for name, coords in name_to_coords_map.items():
        if coords is None:
            missing_points.append(name)
        else:
            name_to_coords_map[name] = list(coords)

    if missing_points:
        raise ValueError(
            f'The provided DigMontage did not contain all required cardinal '
            f'points (nasion and left and right pre-auricular points). The '
            f'following points are missing: '
            f'{", ".join(missing_points)}')

    mri_json = {
        'AnatomicalLandmarkCoordinates': name_to_coords_map
    }

    bids_path_json = bids_path.copy().update(extension='.json')
    if not bids_path_json.fpath.exists():  # Must exist before we can update it
        _write_json(bids_path_json.fpath, dict())

    update_sidecar_json(bids_path=bids_path_json, entries=mri_json)
