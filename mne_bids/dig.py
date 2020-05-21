"""Read/write BIDS compatible electrode/coords structures from MNE."""
# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)
from collections import OrderedDict

import mne
import numpy as np
from mne.io.constants import FIFF
from mne.utils import _check_ch_locs, logger, warn

from mne_bids.tsv_handler import _from_tsv
from mne_bids.utils import (_scale_coord_to_meters)
from mne_bids.utils import (_write_json, _write_tsv, _extract_landmarks)


def _handle_electrodes_reading(electrodes_fname, coord_frame,
                               coord_unit, raw, verbose):
    """Read associated electrodes.tsv and populate raw.

    Handle xyz coordinates and coordinate frame of each channel.
    Assumes units of coordinates are in 'm'.
    """
    logger.info('Reading electrode '
                'coords from {}.'.format(electrodes_fname))
    electrodes_dict = _from_tsv(electrodes_fname)
    # First, make sure that ordering of names in channels.tsv matches the
    # ordering of names in the raw data. The "name" column is mandatory in BIDS
    ch_names_raw = list(raw.ch_names)
    ch_names_tsv = electrodes_dict['name']

    if ch_names_raw != ch_names_tsv:
        msg = ('Channels do not correspond between raw data and the '
               'channels.tsv file. For MNE-BIDS, the channel names in the '
               'tsv MUST be equal and in the same order as the channels in '
               'the raw data.\n\n'
               '{} channels in tsv file: "{}"\n\n --> {}\n\n'
               '{} channels in raw file: "{}"\n\n --> {}\n\n'
               .format(len(ch_names_tsv), electrodes_fname, ch_names_tsv,
                       len(ch_names_raw), raw.filenames, ch_names_raw)
               )

        # XXX: this could be due to MNE inserting a 'STI 014' channel as the
        # last channel: In that case, we can work. --> Can be removed soon,
        # because MNE will stop the synthesis of stim channels in the near
        # future
        if not (ch_names_raw[-1] == 'STI 014' and
                ch_names_raw[:-1] == ch_names_tsv):
            raise RuntimeError(msg)

    if verbose:
        summary_str = [(ch, coord) for idx, (ch, coord)
                       in enumerate(electrodes_dict.items())
                       if idx < 5]
        print("The read in electrodes file is: \n", summary_str)

    def _float_or_nan(val):
        if val == "n/a":
            return np.nan
        else:
            return float(val)

    # convert coordinates to float and create list of tuples
    electrodes_dict['x'] = [_float_or_nan(x) for x in electrodes_dict['x']]
    electrodes_dict['y'] = [_float_or_nan(x) for x in electrodes_dict['y']]
    electrodes_dict['z'] = [_float_or_nan(x) for x in electrodes_dict['z']]
    ch_names_raw = [x for i, x in enumerate(ch_names_raw)
                    if electrodes_dict['x'][i] != "n/a"]
    ch_locs = np.c_[electrodes_dict['x'],
                    electrodes_dict['y'],
                    electrodes_dict['z']]

    # determine if there are problematic channels
    nan_chs = []
    for ch_name, ch_coord in zip(ch_names_raw, ch_locs):
        if any(np.isnan(ch_coord)) and ch_name not in raw.info['bads']:
            nan_chs.append(ch_name)
    if len(nan_chs) > 0:
        warn("There are channels without locations "
             "(n/a) that are not marked as bad: {}".format(nan_chs))

    # convert coordinates to meters
    ch_locs = _scale_coord_to_meters(ch_locs, coord_unit)

    # create mne.DigMontage
    ch_pos = dict(zip(ch_names_raw, ch_locs))
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos,
                                            coord_frame=coord_frame)
    raw.set_montage(montage)
    return raw


def _electrodes_tsv(raw, fname, kind, overwrite=False, verbose=True):
    """Create an electrodes.tsv file and save it.

    Parameters
    ----------
    raw : instance of Raw
        The data as MNE-Python Raw object.
    fname : str
        Filename to save the electrodes.tsv to.
    kind : str
        Type of the data as in ALLOWED_KINDS. For iEEG, requires size.
    overwrite : bool
        Defaults to False.
        Whether to overwrite the existing data in the file.
        If there is already data for the given `fname` and overwrite is False,
        an error will be raised.
    verbose : bool
        Set verbose output to true or false.
    """
    # create list of channel coordinates and names
    x, y, z, names = list(), list(), list(), list()
    for ch in raw.info['chs']:
        if _check_ch_locs([ch]):
            x.append(ch['loc'][0])
            y.append(ch['loc'][1])
            z.append(ch['loc'][2])
        else:
            x.append('n/a')
            y.append('n/a')
            z.append('n/a')
        names.append(ch['ch_name'])

    # create OrderedDict to write to tsv file
    if kind == "ieeg":
        sizes = ['n/a'] * len(names)
        data = OrderedDict([('name', names),
                            ('x', x),
                            ('y', y),
                            ('z', z),
                            ('size', sizes),
                            ])
    elif kind == 'eeg':
        data = OrderedDict([('name', names),
                            ('x', x),
                            ('y', y),
                            ('z', z),
                            ])
    else:  # pragma: no cover
        raise RuntimeError("kind {} not supported.".format(kind))

    _write_tsv(fname, data, overwrite=overwrite, verbose=verbose)


def _coordsystem_json(raw, unit, orient, coordsystem_name, fname,
                      kind, overwrite=False, verbose=True):
    """Create a coordsystem.json file and save it.

    Parameters
    ----------
    raw : instance of Raw
        The data as MNE-Python Raw object.
    unit : str
        Units to be used in the coordsystem specification,
        as in BIDS_COORDINATE_UNITS.
    orient : str
        Used to define the coordinate system for the head coils.
    coordsystem_name : str
        Name of the coordinate system for the sensor positions.
    fname : str
        Filename to save the coordsystem.json to.
    kind : str
        Type of the data as in ALLOWED_KINDS.
    overwrite : bool
        Whether to overwrite the existing file.
        Defaults to False.
    verbose : bool
        Set verbose output to true or false.

    """
    dig = raw.info['dig']
    coords = _extract_landmarks(dig)
    hpi = {d['ident']: d for d in dig if d['kind'] == FIFF.FIFFV_POINT_HPI}
    if hpi:
        for ident in hpi.keys():
            coords['coil%d' % ident] = hpi[ident]['r'].tolist()

    coord_frame = set([dig[ii]['coord_frame'] for ii in range(len(dig))])
    if len(coord_frame) > 1:  # noqa E501
        raise ValueError('All HPI, electrodes, and fiducials must be in the '
                         'same coordinate frame. Found: "{}"'
                         .format(coord_frame))

    if kind == 'meg':
        hpi = {d['ident']: d for d in dig if d['kind'] == FIFF.FIFFV_POINT_HPI}
        if hpi:
            for ident in hpi.keys():
                coords['coil%d' % ident] = hpi[ident]['r'].tolist()

        fid_json = {'MEGCoordinateSystem': coordsystem_name,
                    'MEGCoordinateUnits': unit,  # XXX validate this
                    'HeadCoilCoordinates': coords,
                    'HeadCoilCoordinateSystem': orient,
                    'HeadCoilCoordinateUnits': unit  # XXX validate this
                    }
    elif kind == 'eeg':
        fid_json = {'EEGCoordinateSystem': coordsystem_name,
                    'EEGCoordinateUnits': unit,
                    'AnatomicalLandmarkCoordinates': coords,
                    'AnatomicalLandmarkCoordinateSystem': coordsystem_name,
                    'AnatomicalLandmarkCoordinateUnits': unit,
                    }
    elif kind == "ieeg":
        fid_json = {
            'iEEGCoordinateSystem': coordsystem_name,  # (Other, Pixels, ACPC)
            'iEEGCoordinateUnits': unit,  # m (MNE), mm, cm , or pixels
        }

    _write_json(fname, fid_json, overwrite, verbose)
