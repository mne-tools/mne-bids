"""Read/write BIDS compatible electrode/coords structures from MNE."""
# Authors: Adam Li <adam2392@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD-3-Clause
import json
from collections import OrderedDict
from pathlib import Path
import re
import warnings
from importlib_resources import files

import mne
import numpy as np
from mne.io.constants import FIFF
from mne.utils import logger, warn, _validate_type, _check_option
from mne.io.pick import _picks_to_idx

from mne_bids.config import (ALLOWED_SPACES, BIDS_COORDINATE_UNITS,
                             MNE_TO_BIDS_FRAMES, BIDS_TO_MNE_FRAMES,
                             MNE_FRAME_TO_STR, MNE_STR_TO_FRAME,
                             BIDS_COORD_FRAME_DESCRIPTIONS,
                             BIDS_STANDARD_TEMPLATE_COORDINATE_SYSTEMS)
from mne_bids.tsv_handler import _from_tsv
from mne_bids.utils import (_scale_coord_to_meters, _write_json, _write_tsv,
                            verbose)
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

    msg = f'Reading coordinate system frame {coord_frame}'
    if coord_frame_desc:
        msg += f': {coord_frame_desc}'

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


def _write_optodes_tsv(raw, fname, overwrite=False, verbose=True):
    """Create a optodes.tsv file and save it.

    Parameters
    ----------
    raw : instance of Raw
        The data as MNE-Python Raw object.
    fname : str | BIDSPath
        Filename to save the optodes.tsv to.
    overwrite : bool
        Whether to overwrite the existing file.
        Defaults to False.
    verbose : bool
        Set verbose output to True or False.
    """
    picks = _picks_to_idx(raw.info, 'fnirs', exclude=[], allow_empty=True)
    sources = np.zeros(picks.shape)
    detectors = np.zeros(picks.shape)
    for ii in picks:
        # NIRS channel names take a specific form in MNE-Python.
        # The channel names always reflect the source and detector
        # pair, followed by the wavelength frequency.
        # The following code extracts the source and detector
        # numbers from the channel name.
        ch1_name_info = re.match(r'S(\d+)_D(\d+) (\d+)',
                                 raw.info['chs'][ii]['ch_name'])
        sources[ii] = ch1_name_info.groups()[0]
        detectors[ii] = ch1_name_info.groups()[1]
    unique_sources = np.unique(sources)
    n_sources = len(unique_sources)
    unique_detectors = np.unique(detectors)
    names = np.concatenate((
        ["S" + str(s) for s in unique_sources.astype(int)],
        ["D" + str(d) for d in unique_detectors.astype(int)]))

    xs = np.zeros(names.shape)
    ys = np.zeros(names.shape)
    zs = np.zeros(names.shape)
    for i, source in enumerate(unique_sources):
        s_idx = np.where(sources == source)[0][0]
        xs[i] = raw.info["chs"][s_idx]["loc"][3]
        ys[i] = raw.info["chs"][s_idx]["loc"][4]
        zs[i] = raw.info["chs"][s_idx]["loc"][5]
    for i, detector in enumerate(unique_detectors):
        d_idx = np.where(detectors == detector)[0][0]
        xs[i + n_sources] = raw.info["chs"][d_idx]["loc"][6]
        ys[i + n_sources] = raw.info["chs"][d_idx]["loc"][7]
        zs[i + n_sources] = raw.info["chs"][d_idx]["loc"][8]

    ch_data = {
        'name': names,
        'type': np.concatenate(
            (np.full(len(unique_sources), 'source'),
             np.full(len(unique_detectors), 'detector'))
        ),
        'x': xs,
        'y': ys,
        'z': zs,
    }
    _write_tsv(fname, ch_data, overwrite, verbose)


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
    if raw.get_montage() is None:
        dig = list()
        coords = dict()
    else:
        montage = raw.get_montage()
        pos = montage.get_positions()
        dig = list() if montage.dig is None else montage.dig
        coords = dict(
            NAS=list() if pos['nasion'] is None else pos['nasion'].tolist(),
            LPA=list() if pos['lpa'] is None else pos['lpa'].tolist(),
            RPA=list() if pos['rpa'] is None else pos['rpa'].tolist())

    # get the coordinate frame description
    sensor_coord_system_descr = (BIDS_COORD_FRAME_DESCRIPTIONS
                                 .get(sensor_coord_system.lower(), "n/a"))

    # create the coordinate json data structure based on 'datatype'
    if datatype == 'meg':
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
            'AnatomicalLandmarkCoordinates': coords,
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
    elif datatype == "nirs":
        fid_json = {
            'NIRSCoordinateSystem': sensor_coord_system,
            'NIRSCoordinateSystemDescription': sensor_coord_system_descr,
            'NIRSCoordinateUnits': unit,
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

    if montage is None:
        montage = raw.get_montage()
    else:  # assign montage to raw but supress any coordinate transforms
        montage = montage.copy()  # don't modify original
        montage_coord_frame = montage.get_positions()['coord_frame']
        fids = [d for d in montage.dig  # save to add back
                if d['kind'] == FIFF.FIFFV_POINT_CARDINAL]
        montage.remove_fiducials()  # prevent coordinate transform
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', category=RuntimeWarning,
                                    message='.*nasion not found', module='mne')
            raw.set_montage(montage)
        for ch in raw.info['chs']:
            ch['coord_frame'] = MNE_STR_TO_FRAME[montage_coord_frame]
        for d in raw.info['dig']:
            d['coord_frame'] = MNE_STR_TO_FRAME[montage_coord_frame]
        with raw.info._unlock():  # add back fiducials
            raw.info['dig'] = fids + raw.info['dig']

    # get the accepted mne-python coordinate frames
    coord_frame_int = int(montage.dig[0]['coord_frame'])
    mne_coord_frame = MNE_FRAME_TO_STR.get(coord_frame_int, None)
    coord_frame = MNE_TO_BIDS_FRAMES.get(mne_coord_frame, None)

    if coord_frame == 'CapTrak' and bids_path.datatype in ('eeg', 'nirs'):
        pos = raw.get_montage().get_positions()
        if any([pos[fid_key] is None for fid_key in ('nasion', 'lpa', 'rpa')]):
            raise RuntimeError("'head' coordinate frame must contain nasion "
                               "and left and right pre-auricular point "
                               "landmarks")

    if bids_path.datatype == 'ieeg' and mne_coord_frame == 'mri':
        if not acpc_aligned:
            raise RuntimeError(
                '`acpc_aligned` is False, if your T1 is not aligned '
                'to ACPC and the coordinates are in fact in ACPC '
                'space there will be no way to relate the coordinates '
                'to the T1. If the T1 is ACPC-aligned, use '
                '`acpc_aligned=True`')
        coord_frame = 'ACPC'

    if bids_path.space is None:  # no space, use MNE coord frame
        if coord_frame is None:  # if no MNE coord frame, skip
            warn("Coordinate frame could not be inferred from the raw object "
                 "and the BIDSPath.space was none, skipping the writing of "
                 "channel positions")
            return
    else:  # either a space and an MNE coord frame or just a space
        if coord_frame is None:  # just a space, use that
            coord_frame = bids_path.space
        else:  # space and raw have coordinate frame, check match
            if bids_path.space != coord_frame and not (
                    coord_frame == 'fsaverage' and
                    bids_path.space == 'MNI305'):  # fsaverage == MNI305
                raise ValueError('Coordinates in the raw object or montage '
                                 f'are in the {coord_frame} coordinate '
                                 'frame but BIDSPath.space is '
                                 f'{bids_path.space}')

    # create electrodes/coordsystem files using a subset of entities
    # that are specified for these files in the specification
    coord_file_entities = {
        'root': bids_path.root,
        'datatype': bids_path.datatype,
        'subject': bids_path.subject,
        'session': bids_path.session,
        'acquisition': bids_path.acquisition,
        'space': None if bids_path.datatype == 'nirs' else coord_frame
    }
    channels_suffix = \
        'optodes' if bids_path.datatype == 'nirs' else 'electrodes'
    _channels_fun = _write_optodes_tsv if bids_path.datatype == 'nirs' else \
        _write_electrodes_tsv
    channels_path = BIDSPath(**coord_file_entities, suffix=channels_suffix,
                             extension='.tsv')
    coordsystem_path = BIDSPath(**coord_file_entities, suffix='coordsystem',
                                extension='.json')

    # Now write the data to the elec coords and the coordsystem
    _channels_fun(raw, channels_path, bids_path.datatype, overwrite)
    _write_coordsystem_json(raw=raw, unit=unit, hpi_coord_system='n/a',
                            sensor_coord_system=coord_frame,
                            fname=coordsystem_path,
                            datatype=bids_path.datatype,
                            overwrite=overwrite)


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
        The raw data as MNE-Python ``Raw`` object. The montage
        will be set in place.
    """
    bids_coord_frame, bids_coord_unit = _handle_coordsystem_reading(
        coordsystem_fpath, datatype)

    if bids_coord_frame not in ALLOWED_SPACES[datatype]:
        warn(f'"{bids_coord_frame}" is not a BIDS-acceptable coordinate frame '
             f'for {datatype.upper()}. The supported coordinate frames are: '
             '{}'.format(ALLOWED_SPACES[datatype]))
        coord_frame = None
    elif bids_coord_frame in BIDS_TO_MNE_FRAMES:
        coord_frame = BIDS_TO_MNE_FRAMES.get(bids_coord_frame, None)
    else:
        warn(f"{bids_coord_frame} is not an MNE-Python coordinate frame "
             f"for {datatype.upper()} data and so will be set to 'unknown'")
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

    # add montage to Raw object
    # XXX: Starting with mne 0.24, this will raise a RuntimeWarning
    #      if channel types are included outside of
    #      (EEG/sEEG/ECoG/DBS/fNIRS). Probably needs a fix in the future.
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', category=RuntimeWarning,
                                message='.*nasion not found', module='mne')
        raw.set_montage(montage, on_missing='warn')

    # put back in unknown for unknown coordinate frame
    if coord_frame == 'unknown':
        for ch in raw.info['chs']:
            ch['coord_frame'] = MNE_STR_TO_FRAME['unknown']
        for d in raw.info['dig']:
            d['coord_frame'] = MNE_STR_TO_FRAME['unknown']


# Remove once we depend on MNE-Python 1.0+
def _get_montage(info):
    if hasattr(info, 'get_montage'):
        return info.get_montage()
    # workaround
    return mne.io.RawArray(np.zeros((info['nchan'], 1)), info).get_montage()


@verbose
def template_to_head(info, space, coord_frame='auto', unit='auto',
                     verbose=None):
    """Transform a BIDS standard template montage to the head coordinate frame.

    Parameters
    ----------
    %(info_not_none)s The info is modified in place.
    space : str
        The name of the BIDS standard template. See
        https://bids-specification.readthedocs.io/en/stable/99-appendices/08-coordinate-systems.html#standard-template-identifiers
        for a list of acceptable spaces.
    coord_frame : 'mri' | 'mri_voxel' | 'ras'
        BIDS template coordinate systems do not specify a coordinate frame,
        so this must be determined by inspecting the documentation for the
        dataset or the ``electrodes.tsv`` file.  If ``'auto'``, the coordinate
        frame is assumed to be ``'mri_voxel'`` if the coordinates are strictly
        positive, and ``'ras'`` (``"scanner RAS"``) otherwise.

        .. warning::

            ``scanner RAS`` and ``surface RAS`` coordinates frames are similar
            so be very careful not to assume a BIDS dataset's coordinates are
            in one when they are actually in the other. The only way to tell
            for template coordinate systems, currently, is if it is specified
            in the dataset documentation.

    unit : 'm' | 'mm' | 'auto'
        The unit that was used in the coordinate system specification.
        If ``'auto'``, ``'m'`` will be inferred if the montage
        spans less than ``-1`` to ``1``, and ``'mm'`` otherwise. If the
        ``coord_frame`` is ``'mri_voxel'``, ``unit`` will be ignored.
    %(verbose)s

    Returns
    -------
    %(info_not_none)s The modified ``Info`` object.
    trans : mne.transforms.Transform
        The data transformation matrix from ``'head'`` to ``'mri'``
        coordinates.

    """
    _validate_type(info, mne.io.Info)
    _check_option('space', space, BIDS_STANDARD_TEMPLATE_COORDINATE_SYSTEMS)
    _check_option('coord_frame', coord_frame,
                  ('auto', 'mri', 'mri_voxel', 'ras'))
    _check_option('unit', unit, ('auto', 'm', 'mm'))
    # XXX: change to after 0.11 release
    # montage = info.get_montage()
    montage = _get_montage(info)
    if montage is None:
        raise RuntimeError('No montage found in the `raw` object')
    montage.remove_fiducials()  # we will add fiducials so remove any
    pos = montage.get_positions()
    if pos['coord_frame'] not in ('mni_tal', 'unknown'):
        raise RuntimeError(
            "Montage coordinate frame '{}' not expected for a template "
            "montage, should be 'unknown' or 'mni_tal'".format(
                pos['coord_frame']))
    locs = np.array(list(pos['ch_pos'].values()))
    locs = locs[~np.any(np.isnan(locs), axis=1)]  # only channels with loc
    if locs.size == 0:
        raise RuntimeError('No channel locations found in the montage')
    if unit == 'auto':
        unit = 'm' if abs(locs - locs.mean(axis=0)).max() < 1 else 'mm'
    if coord_frame == 'auto':
        coord_frame = 'mri_voxel' if locs.min() >= 0 else 'ras'
    # transform montage to head
    data_dir = files('mne_bids.data')
    # set to the right coordinate frame as specified by the user
    for d in montage.dig:  # ensure same coordinate frame
        d['coord_frame'] = MNE_STR_TO_FRAME[coord_frame]
    # do the transforms, first ras -> vox if needed
    if montage.get_positions()['coord_frame'] == 'ras':
        ras_vox_trans = mne.read_trans(
            data_dir / f'space-{space}_ras-vox_trans.fif')
        if unit == 'm':  # must be in mm here
            for d in montage.dig:
                d['r'] *= 1000
        montage.apply_trans(ras_vox_trans)
    if montage.get_positions()['coord_frame'] == 'mri_voxel':
        vox_mri_trans = mne.read_trans(
            data_dir / f'space-{space}_vox-mri_trans.fif')
        montage.apply_trans(vox_mri_trans)
    assert montage.get_positions()['coord_frame'] == 'mri'
    if not (unit == 'm' and coord_frame == 'mri'):  # if so, already in m
        for d in montage.dig:
            d['r'] /= 1000  # mm -> m
    # now add fiducials (in mri coordinates)
    fids = mne.io.read_fiducials(
        data_dir / f'space-{space}_fiducials.fif'
    )[0]
    montage.dig = fids + montage.dig  # add fiducials
    for fid in fids:  # ensure also in mri
        fid['coord_frame'] = MNE_STR_TO_FRAME['mri']
    info.set_montage(montage)  # transform to head
    # finally return montage
    return info, mne.read_trans(data_dir / f'space-{space}_trans.fif')
