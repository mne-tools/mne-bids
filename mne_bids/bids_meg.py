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

from mne import events, io
from mne.io.constants import FIFF
from mne.io.pick import channel_type

from datetime import datetime


def _mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def _channel_tsv(raw, fname):
    """Create channel tsv."""

    map_chs = dict(grad='MEGGRAD', mag='MEGMAG', stim='TRIG', eeg='EEG',
                   eog='EOG', ecg='ECG', misc='MISC')
    map_desc = dict(grad='Gradiometer', mag='Magnetometer',
                    stim='Trigger',
                    eeg='ElectroEncephaloGram',
                    ecg='ElectroCardioGram',
                    eog='ElectrOculoGram', misc='Miscellaneous')

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


def _events_tsv(raw, events, fname, event_id):
    """Create tsv file for events."""

    ### may change
    raw = io.read_raw_fif(fnames['raw'])
    if 'events' in fnames.keys():
        events = read_events(fnames['events']).astype(int)
    else:
        events = find_events(raw, min_duration=0.001)
    ###

    events[:, 0] -= raw.first_samp

    event_id_map = {v: k for k, v in event_id.items()}

    df = pd.DataFrame(events[:, [0, 2]],
                      columns=['Onset', 'Condition'])
    df.Condition = df.Condition.map(event_id_map)
    df.Onset /= raw.info['sfreq']

    df.to_csv(fname, sep='\t', index=False)


def _scans_tsv(raw, raw_fname, fname):
    """Create tsv file for scans."""

    acq_time = datetime.fromtimestamp(raw.info['meas_date'][0]
                                      ).strftime('%Y-%m-%dT%H:%M:%S')

    df = pd.DataFrame({'filename': ['meg/%s' % raw_fname],
                       'acq_time': [acq_time]})

    print(df.head())

    df.to_csv(fname, sep='\t', index=False)


def _fid_json(raw):
    dig = raw.info['dig']
    coords = list()
    fids = {d['ident']: d for d in dig if d['kind'] ==
            FIFF.FIFFV_POINT_CARDINAL}
    if fids:
        if FIFF.FIFFV_POINT_NASION in fids:
            coords.append({'NAS': fids[FIFF.FIFFV_POINT_NASION]['r']})
        if FIFF.FIFFV_POINT_LPA in fids:
            coords.append({'LPA': fids[FIFF.FIFFV_POINT_LPA]['r']})
        if FIFF.FIFFV_POINT_RPA in fids:
            coords.append({'RPA': fids[FIFF.FIFFV_POINT_RPA]['r']})

    hpi = [d['ident']: d for d in dig if d['kind'] == FIFF.FIFFV_POINT_HPI}
    if hpi:
        coords.extend([{'coil%d' %ii: hpi[ii]['r']} for ii in range(len(hpi))])

    coord_frame = set([dig[ii]['coord_frame'] for ii in range(len(dig))])
    if len(coord_frame) > 1:
        err = 'All HPI and Fiducials must be in the same coordinate frame.'
        raise ValueError(err)

    {'MEGCoordinateSystem': 'a string of the manufacturer type'
     'MEGCoordinateUnits'
     'CoilCoordinates': coords
     'CoilCoordinateSystem':
     'CoilCoordinateUnits': 'm'
     }

     ## TO DO lookup table for systems based on info


def raw_to_bids(subject, run, task, input_fname, hpi=None, electrode=None, hsp=None,
                config=None, events=None, output_path, overwrite=True):
    """Walk over a folder of files and create bids compatible folder.

    Parameters
    ----------
    subject : str
        The subject name in BIDS compatible format (01, 02, etc.)
    run : str
        The run number in BIDS compatible format.
    task : str
        The task name.
    input_fname : str
        The path to the raw MEG file.
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

    event_id : dict
        The event id dict
    output_path : str
        The path of the BIDS compatible folder
    fnames : dict
        Dictionary of filenames. Valid keys are 'events' and 'raw'.
    overwrite : bool
        If the file already exists, whether to overwrite it.
    """

    ses_path = op.join(output_path, 'sub-%s' % subject_id, 'ses-01')
    meg_path = op.join(ses_path, 'meg')
    if not op.exists(output_path):
        _mkdir_p(output_path)
        if not op.exists(meg_path):
            _mkdir_p(meg_path)

    events = read_events(events).astype(int)

    fname, ext = os.path.splitext(input_fname)

    # KIT systems
    if ext in ['.con', '.sqd']:
         raw = io.read_raw_kit(input_fname, preload=False)

    # Neuromag or converted-to-fif systems
    elif ext in ['.fif', '.gz']:
        raw = io.read_raw_fif(input_fname, preload=False)

     # BTi systems
    elif ext == '.pdf':
        if os.path.isfile(input_fname):
            raw = io.read_raw_bti(input_fname, preload=preload, verbose=verbose,
                                   **kwargs)

    # CTF systems
    elif ext == '':
        pass

    # save stuff
    channels_fname = op.join(meg_path, 'sub-%s_task-%s_run-%s_channel.tsv'
                             % (subject_id, task, run))
    _channel_tsv(raw, channels_fname)

    events_fname = op.join(meg_path, 'sub-%s_task-%s_run-%s_events.tsv'
                           % (subject_id, task, run))
    _events_tsv(raw, events, events_fname, event_id)

    _scans_tsv(raw, fnames['raw'],
               op.join(ses_path, 'sub-%s_ses-01_scans.tsv' % subject_id))

    raw_fname = op.join(meg_path,
                        'sub-%s_task-%s_run-%s_meg%s'
                        % (subject, task, run, ext))

    # for FIF, we need to re-save the file to fix the file pointer
    # for files with multiple parts
    if ext in ['.fif', '.gz']:
        raw.save(raw_fname, overwrite=overwrite)
    else:
        sh.copyfile(input_fname, raw_fname)

    return output_path
