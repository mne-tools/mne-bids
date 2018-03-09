# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

import os
import os.path as op
import shutil as sh
import pandas as pd
import json
from collections import defaultdict, OrderedDict

import numpy as np
from mne import read_events, find_events
from mne.io.constants import FIFF
from mne.io.pick import channel_type

from datetime import datetime

from .utils import filename_bids, create_folders
from .io import _parse_ext, _read_raw

ALLOWED_KINDS = ['meg', 'ieeg']
orientation = {'.sqd': 'ALS', '.con': 'ALS', '.fif': 'RAS', '.gz': 'RAS',
               '.pdf': 'ALS', '.ds': 'ALS'}

units = {'.sqd': 'm', '.con': 'm', '.fif': 'm', '.gz': 'm', '.pdf': 'm',
         '.ds': 'cm'}

manufacturers = {'.sqd': 'KIT/Yokogawa', '.con': 'KIT/Yokogawa',
                 '.fif': 'Elekta', '.gz': 'Elekta', '.pdf': '4D Magnes',
                 '.ds': 'CTF'}


def _channels_tsv(raw, fname, verbose):
    """Create channel tsv."""
    map_chs = defaultdict(lambda: 'OTHER')
    map_chs = dict(grad='MEGGRAD', mag='MEGMAG', stim='TRIG', eeg='EEG',
                   ecog='ECOG', seeg='SEEG', eog='EOG', ecg='ECG', misc='MISC',
                   ref_meg='REFMEG')
    map_desc = defaultdict(lambda: 'Other type of channel')
    map_desc.update(grad='Gradiometer', mag='Magnetometer',
                    stim='Trigger',
                    eeg='ElectroEncephaloGram',
                    ecog='Electrocorticography',
                    seeg='StereoEEG',
                    ecg='ElectroCardioGram',
                    eog='ElectrOculoGram', misc='Miscellaneous',
                    ref_meg='Reference channel')

    status, ch_type, description = list(), list(), list()
    for idx, ch in enumerate(raw.info['ch_names']):
        status.append('bad' if ch in raw.info['bads'] else 'good')
        ch_type.append(map_chs[channel_type(raw.info, idx)])
        description.append(map_desc[channel_type(raw.info, idx)])
    low_cutoff, high_cutoff = (raw.info['highpass'], raw.info['lowpass'])
    n_channels = raw.info['nchan']
    sfreq = raw.info['sfreq']
    df = pd.DataFrame({'name': raw.info['ch_names'], 'type': ch_type,
                       'description': description,
                       'sampling_frequency': ['%.2f' % sfreq] * n_channels,
                       'low_cutoff': ['%.2f' % low_cutoff] * n_channels,
                       'high_cutoff': ['%.2f' % high_cutoff] * n_channels,
                       'status': status})
    df = df[['name', 'type', 'description', 'sampling_frequency', 'low_cutoff',
             'high_cutoff', 'status']]
    df.to_csv(fname, sep='\t', index=False)

    if verbose:
        print(os.linesep + "Writing '%s'..." % fname + os.linesep)
        print(df.head())

    return fname


def _events_tsv(events, raw, fname, event_id, verbose):
    """Create tsv file for events."""

    first_samp = raw.first_samp
    sfreq = raw.info['sfreq']
    events[:, 0] -= first_samp

    df = pd.DataFrame(np.c_[events[:, 0], np.zeros(events.shape[0]),
                            events[:, 2]],
                      columns=['onset', 'duration', 'condition'])
    if event_id:
        event_id_map = {v: k for k, v in event_id.items()}
        df.condition = df.condition.map(event_id_map)
    df.onset /= sfreq
    df = df.fillna('n/a')
    df.to_csv(fname, sep='\t', index=False)
    if verbose:
        print(os.linesep + "Writing '%s'..." % fname + os.linesep)
        print(df.head())

    return fname


def _scans_tsv(raw, raw_fname, fname, verbose):
    """Create tsv file for scans."""

    meas_date = raw.info['meas_date']
    if isinstance(meas_date, (np.ndarray, list)):
        meas_date = meas_date[0]

    if meas_date is None:
        acq_time = 'n/a'
    else:
        acq_time = datetime.fromtimestamp(
            meas_date).strftime('%Y-%m-%dT%H:%M:%S')

    df = pd.DataFrame({'filename': ['%s' % raw_fname],
                       'acq_time': [acq_time]})

    df.to_csv(fname, sep='\t', index=False)

    if verbose:
        print(os.linesep + "Writing '%s'..." % fname + os.linesep)
        print(df.head())

    return fname


def _coordsystem_json(raw, unit, orient, manufacturer, fname, verbose):
    dig = raw.info['dig']
    coords = dict()
    fids = {d['ident']: d for d in dig if d['kind'] ==
            FIFF.FIFFV_POINT_CARDINAL}
    if fids:
        if FIFF.FIFFV_POINT_NASION in fids:
            coords['NAS'] = fids[FIFF.FIFFV_POINT_NASION]['r'].tolist()
        if FIFF.FIFFV_POINT_LPA in fids:
            coords['LPA'] = fids[FIFF.FIFFV_POINT_LPA]['r'].tolist()
        if FIFF.FIFFV_POINT_RPA in fids:
            coords['RPA'] = fids[FIFF.FIFFV_POINT_RPA]['r'].tolist()

    hpi = {d['ident']: d for d in dig if d['kind'] == FIFF.FIFFV_POINT_HPI}
    if hpi:
        for ident in hpi.keys():
            coords['coil%d' % ident] = hpi[ident]['r'].tolist()

    coord_frame = set([dig[ii]['coord_frame'] for ii in range(len(dig))])
    if len(coord_frame) > 1:
        err = 'All HPI and Fiducials must be in the same coordinate frame.'
        raise ValueError(err)

    fid_json = {'MEGCoordinateSystem': manufacturer,
                'MEGCoordinateUnits': unit,  # XXX validate this
                'HeadCoilCoordinates': coords,
                'HeadCoilCoordinateSystem': orient,
                'HeadCoilCoordinateUnits': unit  # XXX validate this
                }
    json_output = json.dumps(fid_json, indent=4, sort_keys=True)
    with open(fname, 'w') as fid:
        fid.write(json_output)

    if verbose:
        print(os.linesep + "Writing '%s'..." % fname + os.linesep)
        print(json_output)

    return fname


def _channel_json(raw, task, manufacturer, fname, verbose):

    sfreq = raw.info['sfreq']

    n_megchan = len([ch for ch in raw.info['chs']
                     if ch['kind'] == FIFF.FIFFV_MEG_CH])
    n_megrefchan = len([ch for ch in raw.info['chs']
                        if ch['kind'] == FIFF.FIFFV_REF_MEG_CH])
    n_eegchan = len([ch for ch in raw.info['chs']
                     if ch['kind'] == FIFF.FIFFV_EEG_CH])
    n_ecogchan = len([ch for ch in raw.info['chs']
                     if ch['kind'] == FIFF.FIFFV_ECOG_CH])
    n_seegchan = len([ch for ch in raw.info['chs']
                     if ch['kind'] == FIFF.FIFFV_SEEG_CH])
    n_eogchan = len([ch for ch in raw.info['chs']
                     if ch['kind'] == FIFF.FIFFV_EOG_CH])
    n_ecgchan = len([ch for ch in raw.info['chs']
                     if ch['kind'] == FIFF.FIFFV_ECG_CH])
    n_emgchan = len([ch for ch in raw.info['chs']
                     if ch['kind'] == FIFF.FIFFV_EMG_CH])
    n_miscchan = len([ch for ch in raw.info['chs']
                     if ch['kind'] == FIFF.FIFFV_EOG_CH])
    n_stimchan = len([ch for ch in raw.info['chs']
                     if ch['kind'] == FIFF.FIFFV_STIM_CH])

    chs_json = OrderedDict([
                ('TaskName', task),
                ('SamplingFrequency', sfreq),
                ("PowerLineFrequency": 42),
                ("DewarPosition": "XXX"),
                ("DigitizedLandmarks": False),
                ("DigitizedHeadPoints": False),
                ("SoftwareFilters": "none"),
                ('Manufacturer', manufacturer),
                ('MEGChannelCount', n_megchan),
                ('MEGREFChannelCount', n_megrefchan),
                ('EEGChannelCount', n_eegchan),
                ('iEEGSurfChannelCount', n_ecogchan),
                ('iEEGDepthChannelCount', n_seegchan),
                ('EOGChannelCount', n_eogchan),
                ('ECGChannelCount', n_ecgchan),
                ('EMGChannelCount', n_emgchan),
                ('MiscChannelCount', n_miscchan),
                ('TriggerChannelCount', n_stimchan)]
    )
    json_output = json.dumps(chs_json, indent=4)
    with open(fname, 'w') as fid:
        fid.write(json_output)
        fid.write("\n")

    if verbose:
        print(os.linesep + "Writing '%s'..." % fname + os.linesep)
        print(json_output)

    return fname

def _dataset_description_json(output_path, verbose):
    """Create json for dataset description."""
    fname = op.join(output_path, 'dataset_description.json')
    dataset_description_json = {
        "Name": " ",
        "BIDSVersion": "1.0.2 (draft)"
    }
    json_output = json.dumps(dataset_description_json)
    with open(fname, 'w') as fid:
        fid.write(json_output)

    if verbose:
        print(os.linesep + "Writing '%s'..." % fname + os.linesep)
        print(json_output)

def raw_to_bids(subject_id, task, raw_fname, output_path, session_id=None, run=None,
                kind=None, events_data=None, event_id=None, hpi=None, electrode=None,
                hsp=None, config=None, overwrite=True, verbose=True):
    """Walk over a folder of files and create bids compatible folder.

    Parameters
    ----------
    subject_id : str
        The subject name in BIDS compatible format (01, 02, etc.)
    task : str
        The task name.
    raw_fname : str
        The path to the raw data file.
    output_path : str
        The path of the BIDS compatible folder
    session_id : str | None
        The session name in BIDS compatible format.
    run : str | None
        The run number in BIDS compatible format.
    kind : str, one of ('meg', 'ieeg')
        The kind of data being converted.
    events_data : str | array | None
        The events file. If a string, a path to the events file. If an array,
        the MNE events array (shape n_events, 3). If None, events will be
        inferred from the stim channel using `find_events`.
    event_id : dict
        The event id dict
    hpi : None | str | list of str
        Marker points representing the location of the marker coils with
        respect to the MEG Sensors, or path to a marker file.
        If list, all of the markers will be averaged together.
    electrode : None | str
        Digitizer points representing the location of the fiducials and the
        marker coils with respect to the digitized head shape, or path to a
        file containing these points.
    hsp : None | str | array, shape = (n_points, 3)
        Digitizer head shape points, or path to head shape file. If more than
        10`000 points are in the head shape, they are automatically decimated.
    config : str | None
        A path to the configuration file to use if the data is from a BTi
        system.
    overwrite : bool
        If the file already exists, whether to overwrite it.
    verbose : bool
        If verbose is True, this will print a snippet of the sidecar files. If
        False, no content will be printed.
    """
    fname, ext = _parse_ext(raw_fname)
    data_path = create_folders(subject=subject_id, session=session_id,
                               kind=kind, root=output_path)
    if session_id is None:
        ses_path = data_path
    else:
        ses_path = create_folders(subject=subject_id, session=session_id,
                                  root=output_path)

    # create filenames
    scans_fname = op.join(ses_path, filename_bids(subject=subject_id, session=session_id, suffix='scans.tsv'))
    fid_fname = op.join(ses_path, filename_bids(subject=subject_id, session=session_id, suffix='fid.json'))
    data_meta_fname = op.join(ses_path, filename_bids(subject=subject_id, session=session_id, suffix='%s.json' % kind))

    channels_fname = op.join(data_path, filename_bids(subject=subject_id, task=task, run=run, suffix='channel.tsv'))
    events_tsv_fname = op.join(data_path, filename_bids(subject=subject_id, task=task, run=run, suffix='events.tsv'))
    raw_fname_bids = op.join(data_path, filename_bids(subject=subject_id, task=task, run=run, suffix='%s%s' % (kind, ext)))

    # Read in Raw object and extract metadata from Raw object if needed
    if kind == 'meg':
        orient = orientation[ext]
        unit = units[ext]
        manufacturer = manufacturers[ext]
    else:
        orient = None
        unit = None
        manufacturer = None

    raw = _read_raw(raw_fname, electrode=electrode, hsp=hsp, hpi=hpi,
                    config=config, verbose=verbose)

    # save stuff
    if kind == 'meg':
        _scans_tsv(raw, raw_fname_bids, scans_fname, verbose)
        _fid_json(raw, unit, orient, manufacturer, fid_fname, verbose)

    _dataset_description_json(output_path, verbose)
    _coordsystem_json(raw, unit, orient, manufacturer, fid_fname, verbose)
    _channel_json(raw, task, manufacturer, data_meta_fname, verbose)
    _channels_tsv(raw, channels_fname, verbose)

    events = _read_events(events_data, raw)
    if len(events) > 0:
        _events_tsv(events, raw, events_tsv_fname, event_id, verbose)

    # for FIF, we need to re-save the file to fix the file pointer
    # for files with multiple parts
    if ext in ['.fif', '.gz']:
        raw.save(raw_fname_bids, overwrite=overwrite)
    elif ext == '.ds':
        if os.path.exists(raw_fname_bids):
            if overwrite:
                sh.rmtree(raw_fname_bids)
                sh.copytree(raw_fname, raw_fname_bids)
            else:
                raise ValueError('"%s" already exists. Please set overwrite to'
                                 ' True.' % raw_fname_bids)
    else:
        if os.path.exists(raw_fname_bids):
            if overwrite:
                os.remove(raw_fname_bids)
                sh.copyfile(raw_fname, raw_fname_bids)
            else:
                raise ValueError('"%s" already exists. Please set overwrite to'
                                 ' True.' % raw_fname_bids)

    return output_path


def _read_events(events_data, raw):
    """Read in events data."""
    if isinstance(events_data, str):
        events = read_events(events_data).astype(int)
    elif isinstance(events_data, np.ndarray):
        if events_data.ndim != 2:
            raise ValueError('Events must have two dimensions, found {}'.format(events.ndim))
        if events_data.shape[1] != 3:
            raise ValueError('Events must have second dimension of length 3, found {}'.format(events.shape[1]))
        events = events_data
    else:
        events = find_events(raw, min_duration=0.001)
    return events
