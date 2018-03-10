# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

import errno
import os
import os.path as op
import shutil as sh
import pandas as pd
import json
from collections import defaultdict

import numpy as np
from mne import read_events, find_events, io
from mne.io.constants import FIFF
from mne.io.pick import channel_type

from datetime import datetime


orientation = {'.sqd': 'ALS', '.con': 'ALS', '.fif': 'RAS', '.gz': 'RAS',
               '.pdf': 'ALS', '.ds': 'ALS'}

units = {'.sqd': 'm', '.con': 'm', '.fif': 'm', '.gz': 'm', '.pdf': 'm',
         '.ds': 'cm'}

manufacturers = {'.sqd': 'KIT/Yokogawa', '.con': 'KIT/Yokogawa',
                 '.fif': 'Elekta', '.gz': 'Elekta', '.pdf': '4D Magnes',
                 '.ds': 'CTF'}


def _mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def _channels_tsv(raw, fname, verbose):
    """Create channel tsv."""

    map_chs = defaultdict(lambda: 'OTHER')
    map_chs.update(grad='MEGGRAD', mag='MEGMAG', stim='TRIG', eeg='EEG',
                   eog='EOG', ecg='ECG', misc='MISC', ref_meg='REFMEG')
    map_desc = defaultdict(lambda: 'Other type of channel')
    map_desc.update(grad='Gradiometer', mag='Magnetometer',
                    stim='Trigger',
                    eeg='ElectroEncephaloGram',
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

    df = pd.DataFrame(np.c_[events[:, [0, 2]], np.zeros(events.shape[0])],
                      columns=['onset', 'duration', 'condition'])
    if event_id:
        event_id_map = {v: k for k, v in event_id.items()}
        df.condition = df.Condition.map(event_id_map)
    df.onset /= sfreq

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


def _meg_json(raw, task, manufacturer, fname, verbose):

    sfreq = raw.info['sfreq']

    n_megchan = len([ch for ch in raw.info['chs']
                     if ch['kind'] == FIFF.FIFFV_MEG_CH])
    n_megrefchan = len([ch for ch in raw.info['chs']
                        if ch['kind'] == FIFF.FIFFV_REF_MEG_CH])
    n_eegchan = len([ch for ch in raw.info['chs']
                     if ch['kind'] == FIFF.FIFFV_EEG_CH])
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

    meg_json = {'TaskName': task,
                'SamplingFrequency': sfreq,
                "PowerLineFrequency": 42,
                "DewarPosition": "XXX",
                "DigitizedLandmarks": "XXX",
                "DigitizedHeadPoints": "XXX",
                "SoftwareFilters": "XXX",
                'Manufacturer': manufacturer,
                'MEGChannelCount': n_megchan,
                'MEGREFChannelCount': n_megrefchan,
                'EEGChannelCount': n_eegchan,
                'EOGChannelCount': n_eogchan,
                'ECGChannelCount': n_ecgchan,
                'EMGChannelCount': n_emgchan,
                'MiscChannelCount': n_miscchan,
                'TriggerChannelCount': n_stimchan,
                }
    json_output = json.dumps(meg_json, indent=4, sort_keys=True)
    with open(fname, 'w') as fid:
        fid.write(json_output)

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


def raw_to_bids(subject_id, session_id, run, task, raw_fname, output_path,
                events_fname=None, event_id=None, hpi=None, electrode=None,
                hsp=None, config=None, overwrite=True, verbose=True):
    """Walk over a folder of files and create bids compatible folder.

    Parameters
    ----------
    subject_id : str
        The subject name in BIDS compatible format (01, 02, etc.)
    session_id : str
        The session name in BIDS compatible format.
    run : str
        The run number in BIDS compatible format.
    task : str
        The task name.
    raw_fname : str
        The path to the raw MEG file.
    output_path : str
        The path of the BIDS compatible folder
    events_fname : str
        The path to the events file.
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
    overwrite : bool
        If the file already exists, whether to overwrite it.
    verbose : bool
        If verbose is True, this will print a snippet of the sidecar files. If
        False, no content will be printed.
    """

    fname, ext = os.path.splitext(raw_fname)
    # BTi data is the only file format that does have a file extension
    if ext == '':
        ext = '.pdf'
    ses_path = op.join(output_path, 'sub-%s' % subject_id,
                       'ses-%s' % session_id)
    meg_path = op.join(ses_path, 'meg')
    if not op.exists(output_path):
        _mkdir_p(output_path)
    if not op.exists(meg_path):
        _mkdir_p(meg_path)

    # create the fnames
    channels_fname = op.join(meg_path, 'sub-%s_ses-%s_task-%s_run-%s'
                             '_channels.tsv'
                             % (subject_id, session_id, task, run))
    events_tsv_fname = op.join(meg_path, 'sub-%s_ses-%s_task-%s_run-%s'
                               '_events.tsv'
                               % (subject_id, session_id, task, run))
    scans_fname = op.join(ses_path,
                          'sub-%s_ses-%s_scans.tsv' % (subject_id, session_id))
    fid_fname = op.join(meg_path,
                        'sub-%s_ses-%s_coordsystem.json'
                        % (subject_id, session_id))
    meg_fname = op.join(meg_path,
                        'sub-%s_ses-%s_meg.json' % (subject_id, session_id))
    raw_fname_bids = op.join(meg_path, 'sub-%s_task-%s_run-%s_meg%s'
                             % (subject_id, task, run, ext))

    orient = orientation[ext]
    unit = units[ext]
    manufacturer = manufacturers[ext]

    # KIT systems
    if ext in ['.con', '.sqd']:
        raw = io.read_raw_kit(raw_fname, elp=electrode, hsp=hsp,
                              mrk=hpi, preload=False)

    # Neuromag or converted-to-fif systems
    elif ext in ['.fif', '.gz']:
        raw = io.read_raw_fif(raw_fname, preload=False)

    # BTi systems
    elif ext == '.pdf':
        if os.path.isfile(raw_fname):
            raw = io.read_raw_bti(raw_fname, config_fname=config,
                                  head_shape_fname=hsp,
                                  preload=False, verbose=verbose)

    # CTF systems
    elif ext == '.ds':
        raw = io.read_raw_ctf(raw_fname)

    # save stuff
    _dataset_description_json(output_path, verbose)
    _scans_tsv(raw, raw_fname_bids, scans_fname, verbose)
    _coordsystem_json(raw, unit, orient, manufacturer, fid_fname, verbose)
    _meg_json(raw, task, manufacturer, meg_fname, verbose)
    _channels_tsv(raw, channels_fname, verbose)
    if events_fname:
        events = read_events(events_fname).astype(int)
    else:
        events = find_events(raw, min_duration=0.001)

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
