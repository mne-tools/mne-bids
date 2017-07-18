# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import errno
import os
import os.path as op
import pandas as pd

import mne
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

    map_chs = dict(grad='GRAD', mag='MAG', stim='TRIG', eeg='EEG',
                   eog='EOG', misc='MISC')
    map_desc = dict(grad='Gradiometer', mag='Magnetometer',
                    stim='Trigger',
                    eeg='ElectroEncephaloGram',
                    ecg='ElectroCardioGram',
                    eog='ElectrOculogram', misc='Miscellaneous')

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


def _events_tsv(raw, events, fname):
    """Create tsv file for events."""

    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
                'Visual/Right': 4, 'Smiley': 5, 'Button': 32}
    events[:, 0] -= raw.first_samp

    event_id_map = {v: k for k, v in event_id.items()}

    df = pd.DataFrame(events[:, [0, 2]] / raw.info['sfreq'],
                      columns=['Onset', 'Condition'])
    df.Condition = df.Condition.map(event_id_map)

    df.to_csv(fname, sep='\t', index=False)


def _scans_tsv(raw, fname):
    """Create tsv file for scans."""

    acq_time = datetime.fromtimestamp(raw.info['meas_date'][0]
                                      ).strftime('%Y-%m-%dT%H:%M:%S')

    df = pd.DataFrame({'filename': ['meg/%s' % raw.filenames[0]],
                       'acq_time': [acq_time]})

    print(df.head())

    df.to_csv(fname, sep='\t', index=False)


def folder_to_bids(input_path, output_path, fnames, subject, run, task,
                   overwrite=True):
    """Walk over a folder of files and create bids compatible folder.

    Parameters
    ----------
    input_path : str
        The path to the folder from which to read files
    output_path : str
        The path of the BIDS compatible folder
    fnames : dict
        Dictionary of filenames. Valid keys are 'events' and 'raw'.
    subject : str
        The subject name in BIDS compatible format (01, 02, etc.)
    run : str
        The run number in BIDS compatible format.
    task : str
        The task name.
    overwrite : bool
        If the file already exists, whether to overwrite it.
    """

    meg_path = op.join(output_path, 'sub-%s' % subject, 'MEG')
    if not op.exists(output_path):
        os.mkdir(output_path)
        if not op.exists(meg_path):
            _mkdir_p(meg_path)

    for key in fnames:
        fnames[key] = op.join(input_path, fnames[key])

    events = mne.read_events(fnames['events']).astype(int)
    raw = mne.io.read_raw_fif(fnames['raw'])

    # save stuff
    channels_fname = op.join(meg_path, 'sub-%s_task-%s_run-%s_channel.tsv'
                             % (subject, task, run))
    _channel_tsv(raw, channels_fname)

    events_fname = op.join(meg_path, 'sub-%s_task-%s_run-%s_events.tsv'
                           % (subject, task, run))
    _events_tsv(raw, events, events_fname)

    _scans_tsv(raw, op.join(meg_path, 'sub-%s_scans.tsv' % subject))

    raw_fname = op.join(meg_path,
                        'sub-%s_task-%s_run-%s_meg.fif' % (subject, task, run))
    raw.save(raw_fname, overwrite=overwrite)
