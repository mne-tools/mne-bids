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
from mne.utils import logger, warn

from mne_bids.config import (BIDS_IEEG_COORDINATE_FRAMES,
                             BIDS_MEG_COORDINATE_FRAMES,
                             BIDS_EEG_COORDINATE_FRAMES,
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

    summary_str = [(ch, coord) for idx, (ch, coord)
                   in enumerate(electrodes_dict.items())
                   if idx < 5]
    logger.info("The read in electrodes file is: \n", summary_str)

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
            # (Other, Pixels, ACPC)
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


def _write_dig_bids(bids_path, raw, montage=None, acpc_aligned=False,
                    overwrite=False):
    """Write BIDS formatted DigMontage from Raw instance.

    Handles coordinatesystem.json and electrodes.tsv writing
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

    if montage is None:
        montage = raw.get_montage()
    else:
        # prevent transformation back to "head", only should be used
        # in this specific circumstance
        if bids_path.datatype == 'ieeg':
            montage.remove_fiducials()
        raw.set_montage(montage)

    # get coordinate frame from digMontage
    digpoint = montage.dig[0]
    if any(digpoint['coord_frame'] != _digpoint['coord_frame']
           for _digpoint in montage.dig):
        warn("Not all digpoints have the same coordinate frame. "
             "Skipping electrodes.tsv writing...")
        return

    # get the accepted mne-python coordinate frames
    coord_frame_int = int(digpoint['coord_frame'])
    mne_coord_frame = MNE_FRAME_TO_STR.get(coord_frame_int, None)
    coord_frame = MNE_TO_BIDS_FRAMES.get(mne_coord_frame, None)

    if bids_path.datatype == 'ieeg' and mne_coord_frame == 'mri':
        if acpc_aligned:
            coord_frame = 'ACPC'
        else:
            raise RuntimeError(
                '`acpc_aligned` is False, if your T1 is not aligned '
                'to ACPC and the coordinates are in fact in ACPC '
                'space there will be no way to relate the coordinates '
                'to the T1. If the T1 is ACPC-aligned, use '
                '`acpc_aligned=True`')

    # create electrodes/coordsystem files using a subset of entities
    # that are specified for these files in the specification
    coord_file_entities = {
        'root': bids_path.root,
        'datatype': bids_path.datatype,
        'subject': bids_path.subject,
        'session': bids_path.session,
        'acquisition': bids_path.acquisition,
        'space': bids_path.space
    }
    datatype = bids_path.datatype
    electrodes_path = BIDSPath(**coord_file_entities, suffix='electrodes',
                               extension='.tsv')
    coordsystem_path = BIDSPath(**coord_file_entities, suffix='coordsystem',
                                extension='.json')

    logger.info(f'Writing electrodes file to... {electrodes_path}')
    logger.info(f'Writing coordsytem file to... {coordsystem_path}')

    if datatype == 'ieeg':
        if coord_frame is not None:
            # XXX: To improve when mne-python allows coord_frame='unknown'
            # coordinate frame is either
            coordsystem_path.update(space=coord_frame)
            electrodes_path.update(space=coord_frame)

            # Now write the data to the elec coords and the coordsystem
            _write_electrodes_tsv(raw, electrodes_path,
                                  datatype, overwrite)
            _write_coordsystem_json(raw=raw, unit=unit, hpi_coord_system='n/a',
                                    sensor_coord_system=(coord_frame,
                                                         mne_coord_frame),
                                    fname=coordsystem_path, datatype=datatype,
                                    overwrite=overwrite)
        else:
            # default coordinate frame to mri if not available
            warn("Coordinate frame of iEEG coords missing/unknown "
                 "for {}. Skipping reading "
                 "in of montage...".format(electrodes_path))
    elif datatype == 'eeg':
        # We only write EEG electrodes.tsv and coordsystem.json
        # if we have LPA, RPA, and NAS available to rescale to a known
        # coordinate system frame
        coords = _extract_landmarks(raw.info['dig'])
        landmarks = set(['RPA', 'NAS', 'LPA']) == set(list(coords.keys()))

        # XXX: to be improved to allow rescaling if landmarks are present
        # mne-python automatically converts unknown coord frame to head
        if coord_frame_int == FIFF.FIFFV_COORD_HEAD and landmarks:
            # Now write the data
            _write_electrodes_tsv(raw, electrodes_path, datatype,
                                  overwrite)
            _write_coordsystem_json(raw=raw, unit='m', hpi_coord_system='n/a',
                                    sensor_coord_system='CapTrak',
                                    fname=coordsystem_path, datatype=datatype,
                                    overwrite=overwrite)
        else:
            warn("Skipping EEG electrodes.tsv... "
                 "Setting montage not possible if anatomical "
                 "landmarks (NAS, LPA, RPA) are missing, "
                 "and coord_frame is not 'head'.")


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
        read in via ``raw.set_montage(montage)``.

    Returns
    -------
    montage : mne.channels.DigMontage
        The coordinate data as MNE-Python DigMontage object.
    """
    # read in coordinate information
    bids_coord_frame, bids_coord_unit = _handle_coordsystem_reading(
        coordsystem_fpath, datatype)

    if datatype == 'meg':
        if bids_coord_frame not in BIDS_MEG_COORDINATE_FRAMES:
            warn("MEG Coordinate frame is not accepted "
                 "BIDS keyword. The allowed keywords are: "
                 "{}".format(BIDS_MEG_COORDINATE_FRAMES))
            coord_frame = None
        elif bids_coord_frame == 'Other':
            warn("Coordinate frame of MEG data can't be determined "
                 "when 'other'. The currently accepted keywords are: "
                 "{}".format(BIDS_MEG_COORDINATE_FRAMES))
            coord_frame = None
        else:
            coord_frame = BIDS_TO_MNE_FRAMES.get(bids_coord_frame, None)
    elif datatype == 'ieeg':
        # iEEG datatype for BIDS only supports
        # acpc, pixels and then standard templates
        # iEEG datatype for mne-python only supports
        # mni_tal == fsaverage == MNI305
        if bids_coord_frame == 'Pixels':
            warn("Coordinate frame of iEEG data in pixels does not "
                 "get read in by mne-python. Skipping reading of "
                 "electrodes.tsv ...")
            coord_frame = None
        elif bids_coord_frame == 'ACPC':
            coord_frame = BIDS_TO_MNE_FRAMES.get(bids_coord_frame, None)
        elif bids_coord_frame == 'Other':
            # default coordinate frames to available ones in mne-python
            # noqa: see https://bids-specification.readthedocs.io/en/stable/99-appendices/08-coordinate-systems.html
            warn(f"Defaulting coordinate frame to unknown "
                 f"from coordinate system input {bids_coord_frame}")
            coord_frame = BIDS_TO_MNE_FRAMES.get(bids_coord_frame, None)
        else:
            coord_frame = BIDS_TO_MNE_FRAMES.get(bids_coord_frame, None)

            # XXX: if the coordinate frame is not recognized, then
            # coordinates are stored in a system that we cannot
            # recognize yet.
            if coord_frame is None:
                warn(f"iEEG Coordinate frame {bids_coord_frame} is not a "
                     f"readable BIDS keyword by mne-bids yet. The allowed "
                     f"keywords are: {BIDS_IEEG_COORDINATE_FRAMES}")
                coord_frame = 'unknown'
    elif datatype == 'eeg':
        # only accept captrak
        if bids_coord_frame not in BIDS_EEG_COORDINATE_FRAMES:
            warn("EEG Coordinate frame is not accepted "
                 "BIDS keyword. The allowed keywords are: "
                 "{}".format(BIDS_IEEG_COORDINATE_FRAMES))
            coord_frame = None
        else:
            coord_frame = BIDS_TO_MNE_FRAMES.get(bids_coord_frame, None)

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

    # add montage to Raw object
    # XXX: Starting with mne 0.24, this will raise a RuntimeWarning
    #      if channel types are included outside of
    #      (EEG/sEEG/ECoG/DBS/fNIRS). Probably needs a fix in the future.
    raw.set_montage(montage, on_missing='warn')

    return montage
