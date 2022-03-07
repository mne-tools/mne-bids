"""Update BIDS directory structures and sidecar files meta data."""
# Authors: Adam Li <adam2392@gmail.com>
#          Austin Hurst <mynameisaustinhurst@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          mne-bids developers
#
# License: BSD-3-Clause

import json
from collections import OrderedDict

import numpy as np

from mne.channels import DigMontage, make_dig_montage
from mne.utils import (
    logger, _validate_type, verbose, _check_on_missing, _on_missing
)
from mne.io import read_fiducials
from mne.io.constants import FIFF

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
def update_anat_landmarks(
    bids_path, landmarks, *, fs_subject=None, fs_subjects_dir=None,
    kind=None, on_missing='raise', verbose=None
):
    """Update the anatomical landmark coordinates of an MRI scan.

    This will change the ``AnatomicalLandmarkCoordinates`` entry in the
    respective JSON sidecar file, or create it if it doesn't exist.

    Parameters
    ----------
    bids_path : BIDSPath
        Path of the MR image.
    landmarks : mne.channels.DigMontage | path-like
        An :class:`mne.channels.DigMontage` instance with coordinates for the
        nasion and left and right pre-auricular points in MRI voxel
        coordinates. Alternatively, the path to a ``*-fiducials.fif`` file as
        produced by the MNE-Python coregistration GUI or via
        :func:`mne.io.write_fiducials`.

        .. note:: :func:`mne_bids.get_anat_landmarks` provides a convenient and
                  reliable way to generate the landmark coordinates in the
                  required coordinate system.

        .. note:: If ``path-like``, ``fs_subject`` and ``fs_subjects_dir``
                  must be provided as well.

        .. versionchanged:: 0.10
           Added support for ``path-like`` input.
    fs_subject : str | None
        The subject identifier used for FreeSurfer. Must be provided if
        ``landmarks`` is ``path-like``; otherwise, it will be ignored.
    fs_subjects_dir : path-like | None
        The FreeSurfer subjects directory. If ``None``, defaults to the
        ``SUBJECTS_DIR`` environment variable. Must be provided if
        ``landmarks`` is ``path-like``; otherwise, it will be ignored.
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
    on_missing : 'ignore' | 'warn' | 'raise'
        How to behave if the specified landmarks cannot be found in the MRI
        JSON sidecar file.

        .. versionadded:: 0.10
    %(verbose)s

    Notes
    -----
    .. versionadded:: 0.8
    """
    _validate_type(item=bids_path, types=BIDSPath, item_name='bids_path')
    _validate_type(
        item=landmarks, types=(DigMontage, 'path-like'), item_name='landmarks'
    )
    _check_on_missing(on_missing)

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

    if not isinstance(landmarks, DigMontage):  # it's pathlike
        if fs_subject is None:
            raise ValueError(
                'You must provide the "fs_subject" parameter when passing the '
                'path to fiducials'
            )
        landmarks = _get_landmarks_from_fiducials_file(
            bids_path=bids_path,
            fname=landmarks,
            fs_subject=fs_subject,
            fs_subjects_dir=fs_subjects_dir
        )

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
            # Funnily, np.float64 is JSON-serializabe, while np.float32 is not!
            # Thus, cast to float64 to avoid issues (which e.g. may arise when
            # fiducials were read from disk!)
            name_to_coords_map[name] = list(coords.astype('float64'))

    if missing_points:
        raise ValueError(
            f'The provided DigMontage did not contain all required cardinal '
            f'points (nasion and left and right pre-auricular points). The '
            f'following points are missing: '
            f'{", ".join(missing_points)}')

    bids_path_json = bids_path.copy().update(extension='.json')
    if not bids_path_json.fpath.exists():  # Must exist before we can update it
        _write_json(bids_path_json.fpath, dict())

    mri_json = json.loads(bids_path_json.fpath.read_text(encoding='utf-8'))
    if 'AnatomicalLandmarkCoordinates' not in mri_json:
        _on_missing(
            on_missing=on_missing,
            msg=f'No AnatomicalLandmarkCoordinates section found in '
                f'{bids_path_json.fpath.name}',
            error_klass=KeyError
        )
        mri_json['AnatomicalLandmarkCoordinates'] = dict()

    for name, coords in name_to_coords_map.items():
        if kind is not None:
            name = f'{name}_{kind}'

        if name not in mri_json['AnatomicalLandmarkCoordinates']:
            _on_missing(
                on_missing=on_missing,
                msg=f'Anatomical landmark not found in '
                    f'{bids_path_json.fpath.name}: {name}',
                error_klass=KeyError
            )

        mri_json['AnatomicalLandmarkCoordinates'][name] = coords

    update_sidecar_json(bids_path=bids_path_json, entries=mri_json)


def _get_landmarks_from_fiducials_file(*, bids_path, fname, fs_subject,
                                       fs_subjects_dir):
    """Get anatomical landmarks from fiducials file, in MRI voxel space."""
    # avoid dicrular imports
    from mne_bids.write import (
        _get_t1w_mgh, _mri_landmarks_to_mri_voxels, _get_fid_coords
    )

    digpoints, coord_frame = read_fiducials(fname)

    # All of this should be guaranteed, but better be safe than sorry!
    assert coord_frame == FIFF.FIFFV_COORD_MRI
    assert digpoints[0]['ident'] == FIFF.FIFFV_POINT_LPA
    assert digpoints[1]['ident'] == FIFF.FIFFV_POINT_NASION
    assert digpoints[2]['ident'] == FIFF.FIFFV_POINT_RPA

    montage_loaded = make_dig_montage(
        lpa=digpoints[0]['r'],
        nasion=digpoints[1]['r'],
        rpa=digpoints[2]['r'],
        coord_frame='mri'
    )
    landmark_coords_mri, _ = _get_fid_coords(dig_points=montage_loaded.dig)
    landmark_coords_mri = np.asarray(
        (landmark_coords_mri['lpa'],
         landmark_coords_mri['nasion'],
         landmark_coords_mri['rpa'])
    )

    t1w_mgh = _get_t1w_mgh(fs_subject, fs_subjects_dir)
    landmark_coords_voxels = _mri_landmarks_to_mri_voxels(
        mri_landmarks=landmark_coords_mri * 1000,  # in mm
        t1_mgh=t1w_mgh
    )
    montage_voxels = make_dig_montage(
        lpa=landmark_coords_voxels[0],
        nasion=landmark_coords_voxels[1],
        rpa=landmark_coords_voxels[2],
        coord_frame='mri_voxel'
    )

    return montage_voxels
