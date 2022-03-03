"""Read/write BIDS compatible electrode/coords structures from MNE."""
# Authors: Adam Li <adam2392@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD-3-Clause
import json
from collections import OrderedDict
from pathlib import Path

import mne
import numpy as np
from mne.io.constants import FIFF
from mne.transforms import _str_to_frame
from mne.utils import logger, warn

from mne_bids.config import (ALLOWED_SPACES, ALLOWED_SPACES_WRITE,
                             BIDS_COORDINATE_UNITS,
                             MNE_TO_BIDS_FRAMES, BIDS_TO_MNE_FRAMES,
                             MNE_FRAME_TO_STR, BIDS_COORD_FRAME_DESCRIPTIONS)
from mne_bids.tsv_handler import _from_tsv
from mne_bids.utils import (_extract_landmarks, _scale_coord_to_meters,
                            _write_json, _write_tsv)
from mne_bids.path import BIDSPath


def _handle_electrodes_reading(electrodes_fname, coord_frame,
                               coord_unit):
    """Read associated electrodes.tsv and populate raw.

    Handle xyz coordinates and coordinate frame of each channel.
    Assumes units of coordinates are in 'm'.
    """
    logger.info('Reading electrode '
                'coords from {}.'.format(electrodes_fname))
    electrodes_dict = _from_tsv(electrodes_fname)
    ch_names_tsv = electrodes_dict['name']

    def _float_or_nan(val):
        if val == "n/a":
            return np.nan
        else:
            return float(val)

    # convert coordinates to float and create list of tuples
    electrodes_dict['x'] = [_float_or_nan(x) for x in electrodes_dict['x']]
    electrodes_dict['y'] = [_float_or_nan(x) for x in electrodes_dict['y']]
    electrodes_dict['z'] = [_float_or_nan(x) for x in electrodes_dict['z']]
    ch_names_raw = [x for i, x in enumerate(ch_names_tsv)
                    if electrodes_dict['x'][i] != "n/a"]
    ch_locs = np.c_[electrodes_dict['x'],
                    electrodes_dict['y'],
                    electrodes_dict['z']]

    # convert coordinates to meters
    ch_locs = _scale_coord_to_meters(ch_locs, coord_unit)

    # create mne.DigMontage
    ch_pos = dict(zip(ch_names_raw, ch_locs))
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos,
                                            coord_frame=coord_frame)
    return montage


def _handle_coordsystem_reading(coordsystem_fpath, datatype):
    """Read associated coordsystem.json.

    Handle reading the coordinate frame and coordinate unit
    of each electrode.
    """
    # open coordinate system sidecar json
    with open(coordsystem_fpath, 'r', encoding='utf-8-sig') as fin:
        coordsystem_json = json.load(fin)

    if datatype == 'meg':
        coord_frame = coordsystem_json['MEGCoordinateSystem']
        coord_unit = coordsystem_json['MEGCoordinateUnits']
        coord_frame_desc = coordsystem_json.get('MEGCoordinateDescription',
                                                None)
    elif datatype == 'eeg':
        coord_frame = coordsystem_json['EEGCoordinateSystem']
        coord_unit = coordsystem_json['EEGCoordinateUnits']
        coord_frame_desc = coordsystem_json.get('EEGCoordinateDescription',
                                                None)
    elif datatype == 'ieeg':
        coord_frame = coordsystem_json['iEEGCoordinateSystem']
        coord_unit = coordsystem_json['iEEGCoordinateUnits']
        coord_frame_desc = coordsystem_json.get('iEEGCoordinateDescription',
                                                None)

    logger.info(f"Reading in coordinate system frame {coord_frame}: "
                f"{coord_frame_desc}.")

    return coord_frame, coord_unit


def _get_impedances(raw, names):
    """Get the impedance values in kOhm from raw.impedances."""
    if not hasattr(raw, 'impedances'):  # pragma: no cover
        return ['n/a'] * len(names)
    no_info = {'imp': np.nan, 'imp_unit': 'kOhm'}
    impedance_dicts = [raw.impedances.get(name, no_info) for name in names]
    # If we encounter a unit not defined in `scalings`, return NaN
    scalings = {'kOhm': 1, 'Ohm': 0.001}
    impedances = [
        imp_dict['imp'] * scalings.get(imp_dict['imp_unit'], np.nan)
        for imp_dict in impedance_dicts
    ]
    # replace np.nan with BIDS 'n/a' representation
    impedances = [i if not np.isnan(i) else "n/a" for i in impedances]
    return impedances


def _write_electrodes_tsv(raw, fname, datatype, overwrite=False):
    """Create an electrodes.tsv file and save it.

    Parameters
    ----------
    raw : mne.io.Raw
        The data as MNE-Python Raw object.
    fname : str
        Filename to save the electrodes.tsv to.
    datatype : str
        Type of the data recording. Can be ``meg``, ``eeg``,
        or ``ieeg``.
    overwrite : bool
        Defaults to False.
        Whether to overwrite the existing data in the file.
        If there is already data for the given `fname` and overwrite is False,
        an error will be raised.

    """
    # create list of channel coordinates and names
    x, y, z, names = list(), list(), list(), list()
    for ch in raw.info['chs']:
        if (
            np.isnan(ch['loc'][:3]).any() or
            np.allclose(ch['loc'][:3], 0)
        ):
            x.append('n/a')
            y.append('n/a')
            z.append('n/a')
        else:
            x.append(ch['loc'][0])
            y.append(ch['loc'][1])
            z.append(ch['loc'][2])
        names.append(ch['ch_name'])

    # create OrderedDict to write to tsv file
    if datatype == "ieeg":
        # XXX: size should be included in the future
        sizes = ['n/a'] * len(names)
        data = OrderedDict([('name', names),
                            ('x', x),
                            ('y', y),
                            ('z', z),
                            ('size', sizes),
                            ])
    elif datatype == 'eeg':
        data = OrderedDict([('name', names),
                            ('x', x),
                            ('y', y),
                            ('z', z),
                            ])
    else:  # pragma: no cover
        raise RuntimeError("datatype {} not supported.".format(datatype))

    # Add impedance values if available, currently only BrainVision:
    # https://github.com/mne-tools/mne-python/pull/7974
    if hasattr(raw, 'impedances'):
        data['impedance'] = _get_impedances(raw, names)

    # note that any coordsystem.json file shared within sessions
    # will be the same across all runs (currently). So
    # overwrite is set to True always
    # XXX: improve later when BIDS is updated
    # check that there already exists a coordsystem.json
    if Path(fname).exists() and not overwrite:
        electrodes_tsv = _from_tsv(fname)

        # cast values to str to make equality check work
        if any([list(map(str, vals1)) != list(vals2) for vals1, vals2 in
                zip(data.values(), electrodes_tsv.values())]):
            raise RuntimeError(
                f'Trying to write electrodes.tsv, but it already '
                f'exists at {fname} and the contents do not match. '
                f'You must differentiate this electrodes.tsv file '
                f'from the existing one, or set "overwrite" to True.')
    _write_tsv(fname, data, overwrite=True)


def _write_coordsystem_json(*, raw, unit, hpi_coord_system,
                            sensor_coord_system, fname, datatype,
                            overwrite=False):
    """Create a coordsystem.json file and save it.

    Parameters
    ----------
    raw : mne.io.Raw
        The data as MNE-Python Raw object.
    unit : str
        Units to be used in the coordsystem specification,
        as in BIDS_COORDINATE_UNITS.
    hpi_coord_system : str
        Name of the coordinate system for the head coils.
    sensor_coord_system : str | tuple of str
        Name of the coordinate system for the sensor positions.
        If a tuple of strings, should be in the form:
        ``(BIDS coordinate frame, MNE coordinate frame)``.
    fname : str
        Filename to save the coordsystem.json to.
    datatype : str
        Type of the data recording. Can be ``meg``, ``eeg``,
        or ``ieeg``.
    overwrite : bool
        Whether to overwrite the existing file.
        Defaults to False.

    """
    dig = raw.info['dig']
    if dig is None:
        dig = []

    coord_frame = set([dig[ii]['coord_frame'] for ii in range(len(dig))])
    if len(coord_frame) > 1:  # noqa E501
        raise ValueError('All HPI, electrodes, and fiducials must be in the '
                         'same coordinate frame. Found: "{}"'
                         .format(coord_frame))

    # get the coordinate frame description
    try:
        sensor_coord_system, sensor_coord_system_mne = sensor_coord_system
    except ValueError:
        sensor_coord_system_mne = "n/a"
    sensor_coord_system_descr = (BIDS_COORD_FRAME_DESCRIPTIONS
                                 .get(sensor_coord_system.lower(), "n/a"))
    if sensor_coord_system == 'Other':
        logger.info('Using the `Other` keyword for the CoordinateSystem '
                    'field. Please specify the CoordinateSystemDescription '
                    'field manually.')
        sensor_coord_system_descr = (BIDS_COORD_FRAME_DESCRIPTIONS
                                     .get(sensor_coord_system_mne.lower(),
                                          "n/a"))
    coords = _extract_landmarks(dig)
    # create the coordinate json data structure based on 'datatype'
    if datatype == 'meg':
        landmarks = dict(coords)
        hpi = {d['ident']: d for d in dig if d['kind'] == FIFF.FIFFV_POINT_HPI}
        if hpi:
            for ident in hpi.keys():
                coords['coil%d' % ident] = hpi[ident]['r'].tolist()

        fid_json = {
            'MEGCoordinateSystem': sensor_coord_system,
            'MEGCoordinateUnits': unit,  # XXX validate this
            'MEGCoordinateSystemDescription': sensor_coord_system_descr,
            'HeadCoilCoordinates': coords,
            'HeadCoilCoordinateSystem': hpi_coord_system,
            'HeadCoilCoordinateUnits': unit,  # XXX validate this
            'AnatomicalLandmarkCoordinates': landmarks,
            'AnatomicalLandmarkCoordinateSystem': sensor_coord_system,
            'AnatomicalLandmarkCoordinateUnits': unit
        }
    elif datatype == 'eeg':
        fid_json = {
            'EEGCoordinateSystem': sensor_coord_system,
            'EEGCoordinateUnits': unit,
            'EEGCoordinateSystemDescription': sensor_coord_system_descr,
            'AnatomicalLandmarkCoordinates': coords,
            'AnatomicalLandmarkCoordinateSystem': sensor_coord_system,
            'AnatomicalLandmarkCoordinateUnits': unit,
        }
    elif datatype == "ieeg":
        fid_json = {
            'iEEGCoordinateSystem': sensor_coord_system,
            'iEEGCoordinateSystemDescription': sensor_coord_system_descr,
            'iEEGCoordinateUnits': unit,  # m (MNE), mm, cm , or pixels
        }

    # note that any coordsystem.json file shared within sessions
    # will be the same across all runs (currently). So
    # overwrite is set to True always
    # XXX: improve later when BIDS is updated
    # check that there already exists a coordsystem.json
    if Path(fname).exists() and not overwrite:
        with open(fname, 'r', encoding='utf-8-sig') as fin:
            coordsystem_dict = json.load(fin)
        if fid_json != coordsystem_dict:
            raise RuntimeError(
                f'Trying to write coordsystem.json, but it already '
                f'exists at {fname} and the contents do not match. '
                f'You must differentiate this coordsystem.json file '
                f'from the existing one, or set "overwrite" to True.')
    _write_json(fname, fid_json, overwrite=True)


def _set_montage(raw, montage):
    """Set a montage for raw without transforming to head."""
    pos = montage.get_positions()
    ch_pos = pos['ch_pos']
    for ch in raw.info['chs']:
        if ch['ch_name'] in ch_pos:  # skip MEG, non-digitized
            ch['loc'][:3] = ch_pos[ch['ch_name']]
            ch['coord_frame'] = _str_to_frame[pos['coord_frame']]
    with raw.info._unlock():
        raw.info['dig'] = montage.dig


def _write_dig_bids(bids_path, raw, montage=None, acpc_aligned=False,
                    overwrite=False):
    """Write BIDS formatted DigMontage from Raw instance.

    Handles coordsystem.json and electrodes.tsv writing
    from DigMontage.

    Parameters
    ----------
    bids_path : BIDSPath
        Path in the BIDS dataset to save the ``electrodes.tsv``
        and ``coordsystem.json`` file for. ``datatype``
        attribute must be ``eeg``, or ``ieeg``. For ``meg``
        data, ``electrodes.tsv`` are not saved.
    raw : mne.io.Raw
        The data as MNE-Python Raw object.
    montage : mne.channels.DigMontage | None
        The montage to use rather than the one in ``raw`` if it
        must be transformed from the "head" coordinate frame.
    acpc_aligned : bool
        Whether "mri" space is aligned to ACPC.
    overwrite : bool
        Whether to overwrite the existing file.
        Defaults to False.

    """
    # write electrodes data for iEEG and EEG
    unit = "m"  # defaults to meters

    # set montage to raw for writing
    if montage is not None:
        _set_montage(raw, montage)

    # get coordinate frame from digMontage
    digpoint = raw.info['dig'][0]
    if any(digpoint['coord_frame'] != _digpoint['coord_frame']
           for _digpoint in raw.info['dig']):
        warn("Not all digpoints have the same coordinate frame. "
             "Skipping electrodes.tsv writing...")
        return

    # get the accepted mne-python coordinate frames
    coord_frame_int = int(digpoint['coord_frame'])
    mne_coord_frame = MNE_FRAME_TO_STR.get(coord_frame_int, None)
    coord_frame = MNE_TO_BIDS_FRAMES.get(mne_coord_frame, None)

    # If not in a template space, ieeg must be in ACPC or Pixels
    if bids_path.datatype == 'ieeg' and mne_coord_frame == 'mri':
        if not acpc_aligned:
            raise RuntimeError(
                '`acpc_aligned` is False, if your T1 is not aligned '
                'to ACPC and the coordinates are in fact in ACPC '
                'space there will be no way to relate the coordinates '
                'to the T1. If the T1 is ACPC-aligned, use '
                '`acpc_aligned=True`')
        coord_frame = 'ACPC'

    # If not in a template space, EEG is assigned to CapTrak
    if bids_path.datatype == 'eeg':
        # handle CapTrak coordinate frame
        coords = _extract_landmarks(raw.info['dig'])
        landmarks = set(['RPA', 'NAS', 'LPA']) == set(list(coords.keys()))

        # XXX: to be improved to allow rescaling if landmarks are present
        # mne-python automatically converts unknown coord frame to head
        if coord_frame_int == FIFF.FIFFV_COORD_HEAD and landmarks:
            mne_coord_frame = coord_frame = 'CapTrak'

    # fail on unrecognized or mismatched coordinate frames
    allowed = {cf.lower(): cf for cf in
               ALLOWED_SPACES_WRITE[bids_path.datatype]}
    if coord_frame is None:
        if bids_path.space is None:
            warn("Coordinate frame could not be inferred from the raw object "
                 "and the BIDSPath.space was none, skipping the writing of "
                 "channel positions")
            return
        # check that mne coordinate frame isn't something odd
        if mne_coord_frame not in ('mri', 'mri_voxel', 'mni_tal',
                                   'ras', 'fstal', 'unknown'):
            raise ValueError(f'Montage in {mne_coord_frame} inconsistent '
                             f'with BIDSPath.space {bids_path.space}')
        # must be allowed not ACPC or Captrak or a template now
        # ensure proper capitalization
        mne_coord_frame = coord_frame = allowed[bids_path.space.lower()]
    elif bids_path.space is not None:  # use raw object coordinate frame
        # ignore equivalent fsaverage and MNI305
        match = bids_path.space.lower() == coord_frame.lower()
        if bids_path.space.lower() == 'fsaverage' and coord_frame == 'MNI305':
            match = True
            mne_coord_frame = coord_frame = allowed[bids_path.space.lower()]
        if bids_path.space.lower() == 'mni305' and coord_frame == 'fsaverage':
            match = True
            mne_coord_frame = coord_frame = allowed[bids_path.space.lower()]
        if not match:
            raise ValueError('Coordinates in the montage are in the '
                             f'{coord_frame} coordinate frame but '
                             f'BIDSPath.space is {bids_path.space}')

    # create electrodes/coordsystem files using a subset of entities
    # that are specified for these files in the specification
    coord_file_entities = {
        'root': bids_path.root,
        'datatype': bids_path.datatype,
        'subject': bids_path.subject,
        'session': bids_path.session,
        'acquisition': bids_path.acquisition,
        'space': coord_frame
    }
    electrodes_path = BIDSPath(**coord_file_entities, suffix='electrodes',
                               extension='.tsv')
    coordsystem_path = BIDSPath(**coord_file_entities, suffix='coordsystem',
                                extension='.json')

    # write the data
    _write_electrodes_tsv(raw, electrodes_path, bids_path.datatype, overwrite)

    _write_coordsystem_json(raw=raw, unit=unit, hpi_coord_system='n/a',
                            sensor_coord_system=(coord_frame,
                                                 mne_coord_frame),
                            fname=coordsystem_path,
                            datatype=bids_path.datatype, overwrite=overwrite)


def _read_dig_bids(electrodes_fpath, coordsystem_fpath,
                   datatype, raw):
    """Read MNE-Python formatted DigMontage from BIDS files.

    Handles coordinatesystem.json and electrodes.tsv reading
    to DigMontage.

    Parameters
    ----------
    electrodes_fpath : str
        Filepath of the electrodes.tsv to read.
    coordsystem_fpath : str
        Filepath of the coordsystem.json to read.
    datatype : str
        Type of the data recording. Can be ``meg``, ``eeg``,
        or ``ieeg``.
    raw : mne.io.Raw
        The raw data as MNE-Python ``Raw`` object. Will set montage
        read in via ``raw.set_montage(montage)``. The montage is set
        in place.
    """
    # read in coordinate information
    bids_coord_frame, bids_coord_unit = _handle_coordsystem_reading(
        coordsystem_fpath, datatype)

    if bids_coord_frame not in ALLOWED_SPACES[datatype]:
        warn(f"{datatype} coordinate frame is not accepted "
             "BIDS keyword. The allowed keywords are: "
             "{}".format(ALLOWED_SPACES[datatype]))
        return

    if bids_coord_frame in BIDS_TO_MNE_FRAMES:
        coord_frame = BIDS_TO_MNE_FRAMES[bids_coord_frame]
    else:  # acceptable coordinate frame but not in MNE -> set to unknown
        if bids_coord_frame == 'Other':
            warn(f"Coordinate frame of {datatype} data is 'Other' "
                 "which will be set as 'unknown'")
        elif datatype == 'ieeg' and bids_coord_frame == 'Pixels':
            warn("Coordinate frame for iEEG data of pixels will be stored"
                 "as 'unknown' since it is not recognized by MNE.")
        else:
            warn("Setting coordinate frame to 'unknown' for "
                 f"{bids_coord_frame}, this template coordinate frame "
                 "is not implemented in MNE and so you will have to "
                 "keep track of the coordinate frame yourself")
        coord_frame = 'unknown'

    # check coordinate units
    if bids_coord_unit not in BIDS_COORDINATE_UNITS:
        warn(f"Coordinate unit is not an accepted BIDS unit for "
             f"{electrodes_fpath}. Please specify to be one of "
             f"{BIDS_COORDINATE_UNITS}. Skipping electrodes.tsv reading...")
        coord_frame = None

    # montage is interpretable only if coordinate frame was properly parsed
    if coord_frame is not None:
        # read in electrode coordinates as a DigMontage object
        montage = _handle_electrodes_reading(electrodes_fpath, coord_frame,
                                             bids_coord_unit)
    else:
        montage = None

    if montage is not None:
        # determine if there are problematic channels
        ch_pos = montage._get_ch_pos()
        nan_chs = []
        for ch_name, ch_coord in ch_pos.items():
            if any(np.isnan(ch_coord)) and ch_name not in raw.info['bads']:
                nan_chs.append(ch_name)
        if len(nan_chs) > 0:
            warn(f"There are channels without locations "
                 f"(n/a) that are not marked as bad: {nan_chs}")

    # set montage
    _set_montage(raw, montage)
