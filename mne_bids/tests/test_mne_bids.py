"""Test the MNE BIDS converter.

For each supported file format, implement a test.
"""
# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Teon L Brooks <teon.brooks@gmail.com>
#          Chris Holdgraf <choldgraf@berkeley.edu>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Matt Sanderson <matt.sanderson@mq.edu.au>
#
# License: BSD (3-clause)

import os
import os.path as op
from errno import EEXIST
import pytest

import pandas as pd
import mne
from mne.datasets import testing
from mne.utils import _TempDir, run_subprocess
from mne.io.constants import FIFF

from mne_bids import raw_to_bids, make_bids_filename, make_bids_folders
from mne_bids.mne_bids import (_channels_tsv, _coordsystem_json, _events_tsv,
                               _participants_tsv, _sidecar_json, _scans_tsv)
from mne_bids.utils import _read_events

base_path = op.join(op.dirname(mne.__file__), 'io')
subject_id = '01'
subject_id2 = '02'
session_id = '01'
run = '01'
acq = '01'
run2 = '02'
task = 'testing'

# for windows, shell = True is needed
# to call npm, bids-validator etc.
#     see: https://stackoverflow.com/questions/
#          28891053/run-npm-commands-using-python-subprocess
shell = False
if os.name == 'nt':
    shell = True


# MEG Tests
# ---------
def test_fif():
    """Test functionality of the raw_to_bids conversion for Neuromag data."""
    output_path = _TempDir()
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')

    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
                'Visual/Right': 4, 'Smiley': 5, 'Button': 32}
    events_fname = op.join(data_path, 'MEG', 'sample',
                           'sample_audvis_trunc_raw-eve.fif')

    raw_to_bids(subject_id=subject_id, session_id=session_id, run=run,
                acquisition=acq, task=task, raw_file=raw_fname,
                events_data=events_fname, output_path=output_path,
                event_id=event_id, write_mode='append')

    # give the raw object some fake participant data
    raw = mne.io.read_raw_fif(raw_fname)
    raw.anonymize()
    raw.info['subject_info'] = {'his_id': subject_id2,
                                'birthday': (1994, 1, 26), 'sex': 1}
    data_path2 = _TempDir()
    raw_fname2 = op.join(data_path2, 'sample_audvis_raw.fif')
    raw.save(raw_fname2)
    raw_to_bids(subject_id=subject_id2, run=run, task=task, acquisition=acq,
                session_id=session_id, raw_file=raw_fname2,
                events_data=events_fname, output_path=output_path,
                event_id=event_id, write_mode='append')
    cmd = ['bids-validator', output_path]
    run_subprocess(cmd, shell=shell)

    assert op.exists(op.join(output_path, 'participants.tsv'))


def test_kit():
    """Test functionality of the raw_to_bids conversion for KIT data."""
    output_path = _TempDir()
    data_path = op.join(base_path, 'kit', 'tests', 'data')
    raw_fname = op.join(data_path, 'test.sqd')
    events_fname = op.join(data_path, 'test-eve.txt')
    hpi_fname = op.join(data_path, 'test_mrk.sqd')
    electrode_fname = op.join(data_path, 'test_elp.txt')
    headshape_fname = op.join(data_path, 'test_hsp.txt')
    event_id = dict(cond=1)

    raw_to_bids(subject_id=subject_id, session_id=session_id, run=run,
                task=task, acquisition=acq, raw_file=raw_fname,
                events_data=events_fname, event_id=event_id, hpi=hpi_fname,
                electrode=electrode_fname, hsp=headshape_fname,
                output_path=output_path, write_mode='append')
    cmd = ['bids-validator', output_path]
    run_subprocess(cmd, shell=shell)
    assert op.exists(op.join(output_path, 'participants.tsv'))

    # ensure the channels file has no STI 014 channel:
    channels_tsv = make_bids_filename(
        subject=subject_id, session=session_id, task=task, run=run,
        suffix='channels.tsv', acquisition=acq,
        prefix=op.join(output_path, 'sub-01/ses-01/meg'))
    df = pd.read_csv(channels_tsv, sep='\t')
    assert not ('STI 014' in df['name'].values)

    # ensure the marker file is produced in the right place
    raw_folder = make_bids_filename(
        subject=subject_id, session=session_id, task=task, run=run,
        acquisition=acq, suffix='%s' % 'meg')
    marker_fname = make_bids_filename(
        subject=subject_id, session=session_id, task=task, run=run,
        acquisition=acq, suffix='markers.sqd',
        prefix=os.path.join(output_path, 'sub-01/ses-01/meg', raw_folder))
    assert op.exists(marker_fname)

    # check for error if there are multiple marker coils specified
    with pytest.raises(ValueError):
        raw_to_bids(subject_id=subject_id, session_id=session_id, run=run,
                    task=task, acquisition=acq, raw_file=raw_fname,
                    events_data=events_fname, event_id=event_id,
                    hpi=[hpi_fname, hpi_fname], electrode=electrode_fname,
                    hsp=headshape_fname, output_path=output_path,
                    overwrite=True)


def test_ctf():
    """Test functionality of the raw_to_bids conversion for CTF data."""
    output_path = _TempDir()
    data_path = op.join(testing.data_path(download=False), 'CTF')
    raw_fname = op.join(data_path, 'testdata_ctf.ds')

    raw_to_bids(subject_id=subject_id, session_id=session_id, run=run,
                task=task, acquisition=acq, raw_file=raw_fname,
                output_path=output_path, write_mode='append')
    cmd = ['bids-validator', output_path]
    run_subprocess(cmd, shell=shell)

    # test to check that the 'error' write_mode parameter works correctly
    with pytest.raises(OSError, match="already exists"):
        raw_to_bids(subject_id=subject_id, session_id=session_id, run=run,
                    task=task, acquisition=acq, raw_file=raw_fname,
                    output_path=output_path, write_mode='error')

    assert op.exists(op.join(output_path, 'participants.tsv'))


def test_bti():
    """Test functionality of the raw_to_bids conversion for BTi data."""
    output_path = _TempDir()
    data_path = op.join(base_path, 'bti', 'tests', 'data')
    raw_fname = op.join(data_path, 'test_pdf_linux')
    config_fname = op.join(data_path, 'test_config_linux')
    headshape_fname = op.join(data_path, 'test_hs_linux')

    raw_to_bids(subject_id=subject_id, session_id=session_id, run=run,
                task=task, acquisition=acq, raw_file=raw_fname,
                config=config_fname, hsp=headshape_fname,
                output_path=output_path, verbose=True, write_mode='append')

    assert op.exists(op.join(output_path, 'participants.tsv'))

    cmd = ['bids-validator', output_path]
    run_subprocess(cmd, shell=shell)


# EEG Tests
# ---------
def test_vhdr():
    """Test raw_to_bids conversion for BrainVision data."""
    output_path = _TempDir()
    data_path = op.join(base_path, 'brainvision', 'tests', 'data')
    raw_fname = op.join(data_path, 'test.vhdr')

    raw_to_bids(subject_id=subject_id, session_id=session_id, run=run,
                task=task, acquisition=acq, raw_file=raw_fname,
                output_path=output_path, write_mode='append', kind='eeg')

    cmd = ['bids-validator', '--bep006', output_path]
    run_subprocess(cmd, shell=shell)

    # check the 'overwrite' write_mode command works correctly
    raw_to_bids(subject_id=subject_id, session_id=session_id, run=run2,
                task=task, acquisition=acq, raw_file=raw_fname,
                output_path=output_path, write_mode='overwrite', kind='eeg')
    # we will need to go over the tree to ensure there are no files with
    # `run-01` in their file names:
    for _, _, files in os.walk(output_path):
        for f in files:
            if 'run-%s' % run in f:
                raise OSError(EEXIST)

    # check that the scans list contains one scans
    scans_tsv = make_bids_filename(
        subject=subject_id, session=session_id, suffix='scans.tsv',
        prefix=op.join(output_path, 'sub-01/ses-01'))
    if op.exists(scans_tsv):
        df = pd.read_csv(scans_tsv, sep='\t')
        assert df.shape[0] == 1

    # finally, create another bids folder with the overwrite command and check
    # no files are in the folder
    data_path = make_bids_folders(subject=subject_id, session=session_id,
                                  kind='eeg', root=output_path,
                                  overwrite=True)
    assert len([f for f in os.listdir(data_path) if op.isfile(f)]) == 0


def test_edf():
    """Test raw_to_bids conversion for European Data Format data."""
    output_path = _TempDir()
    data_path = op.join(testing.data_path(), 'EDF')
    raw_fname = op.join(data_path, 'test_reduced.edf')

    raw = mne.io.read_raw_edf(raw_fname, preload=True)
    raw.rename_channels({raw.info['ch_names'][0]: 'EOG'})
    raw.info['chs'][0]['coil_type'] = FIFF.FIFFV_COIL_EEG_BIPOLAR
    raw.rename_channels({raw.info['ch_names'][1]: 'EMG'})
    raw.set_channel_types({'EMG': 'emg'})

    raw_to_bids(subject_id=subject_id, session_id=session_id, run=run,
                task=task, acquisition=acq, raw_file=raw,
                output_path=output_path, write_mode='append', kind='eeg')
    raw_to_bids(subject_id=subject_id, session_id=session_id, run=run2,
                task=task, acquisition=acq, raw_file=raw,
                output_path=output_path, write_mode='append', kind='eeg')

    cmd = ['bids-validator', '--bep006', output_path]
    run_subprocess(cmd, shell=shell)

    # ensure there is an EMG channel in the channels.tsv:
    channels_tsv = make_bids_filename(
        subject=subject_id, session=session_id, task=task, run=run,
        suffix='channels.tsv', acquisition=acq,
        prefix=op.join(output_path, 'sub-01/ses-01/eeg'))
    df = pd.read_csv(channels_tsv, sep='\t')
    assert 'ElectroMyoGram' in df['description'].values

    # check that the scans list contains two scans
    scans_tsv = make_bids_filename(
        subject=subject_id, session=session_id, suffix='scans.tsv',
        prefix=op.join(output_path, 'sub-01/ses-01'))
    if op.exists(scans_tsv):
        df = pd.read_csv(scans_tsv, sep='\t')
        assert df.shape[0] == 2


def test_bdf():
    """Test raw_to_bids conversion for Biosemi data."""
    output_path = _TempDir()
    data_path = op.join(base_path, 'edf', 'tests', 'data')
    raw_fname = op.join(data_path, 'test.bdf')

    raw_to_bids(subject_id=subject_id, session_id=session_id, run=run,
                task=task, acquisition=acq, raw_file=raw_fname,
                output_path=output_path, write_mode='append', kind='eeg')

    cmd = ['bids-validator', '--bep006', output_path]
    run_subprocess(cmd, shell=shell)


def test_set():
    """Test raw_to_bids conversion for EEGLAB data."""
    # standalone .set file
    output_path = _TempDir()
    data_path = op.join(testing.data_path(), 'EEGLAB')
    raw_fname = op.join(data_path, 'test_raw_onefile.set')

    raw_to_bids(subject_id=subject_id, session_id=session_id, run=run,
                task=task, acquisition=acq, raw_file=raw_fname,
                output_path=output_path, write_mode='append', kind='eeg')

    cmd = ['bids-validator', '--bep006', output_path]
    run_subprocess(cmd, shell=shell)

    with pytest.raises(OSError, match="already exists"):
        raw_to_bids(subject_id=subject_id, session_id=session_id, run=run,
                    task=task, acquisition=acq, raw_file=raw_fname,
                    output_path=output_path, write_mode='error', kind='eeg')

    # .set with associated .fdt
    output_path = _TempDir()
    data_path = op.join(testing.data_path(), 'EEGLAB')
    raw_fname = op.join(data_path, 'test_raw.set')

    raw_to_bids(subject_id=subject_id, session_id=session_id, run=run,
                task=task, acquisition=acq, raw_file=raw_fname,
                output_path=output_path, write_mode='append', kind='eeg')

    cmd = ['bids-validator', '--bep006', output_path]
    run_subprocess(cmd, shell=shell)


def test_cnt():
    """Test raw_to_bids conversion for Neuroscan data."""
    output_path = _TempDir()
    data_path = op.join(testing.data_path(), 'CNT')
    raw_fname = op.join(data_path, 'scan41_short.cnt')

    raw_to_bids(subject_id=subject_id, session_id=session_id, run=run,
                task=task, acquisition=acq, raw_file=raw_fname,
                output_path=output_path, write_mode='append', kind='eeg')

    cmd = ['bids-validator', '--bep006', output_path]
    run_subprocess(cmd, shell=shell)


def test_json_tsv():
    """ Test each of the .tsv and .json generating functions in error mode."""
    output_path = _TempDir()
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')

    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
                'Visual/Right': 4, 'Smiley': 5, 'Button': 32}
    events_fname = op.join(data_path, 'MEG', 'sample',
                           'sample_audvis_trunc_raw-eve.fif')

    raw = mne.io.read_raw_fif(raw_fname)

    raw_to_bids(subject_id=subject_id, session_id=session_id, run=run,
                task=task, acquisition=acq, raw_file=raw,
                events_data=events_fname, output_path=output_path,
                event_id=event_id, write_mode='append')

    ses_path = make_bids_folders(subject=subject_id, session=session_id,
                                 root=output_path, make_dir=False)

    data_path = op.join(ses_path, 'meg')

    participants_fname = make_bids_filename(prefix=output_path,
                                            suffix='participants.tsv')
    coordsystem_fname = make_bids_filename(
        subject=subject_id, session=session_id, acquisition=acq,
        suffix='coordsystem.json', prefix=data_path)
    data_meta_fname = make_bids_filename(
        subject=subject_id, session=session_id, task=task, run=run,
        acquisition=acq, suffix='%s.json' % 'meg', prefix=data_path)
    events_tsv_fname = make_bids_filename(
        subject=subject_id, session=session_id, task=task, run=run,
        acquisition=acq, suffix='events.tsv', prefix=data_path)
    channels_fname = make_bids_filename(
        subject=subject_id, session=session_id, task=task, run=run,
        acquisition=acq, suffix='channels.tsv', prefix=data_path)
    scans_fname = make_bids_filename(
        subject=subject_id, session=session_id, suffix='scans.tsv',
        prefix=ses_path)
    raw_file_bids = make_bids_filename(
        subject=subject_id, session=session_id, task=task, run=run,
        acquisition=acq, suffix='%s%s' % ('meg', '.fif'))

    with pytest.raises(OSError, match="already exists"):
        _channels_tsv(raw, channels_fname, write_mode='error', verbose=False)
    with pytest.raises(OSError, match="already exists"):
        _sidecar_json(raw, task, 'Elekta', data_meta_fname, 'meg',
                      write_mode='error', verbose=False)
    with pytest.raises(OSError, match="already exists"):
        _participants_tsv(raw, subject_id, "n/a", participants_fname, 'error',
                          False)
    with pytest.raises(OSError, match="already exists"):
        _coordsystem_json(raw, "n/a", "n/a", 'Elekta', coordsystem_fname,
                          'error', False)
    with pytest.raises(OSError, match="already exists"):
        events = _read_events(events_fname, raw)
        _events_tsv(events, raw, events_tsv_fname, event_id, 'error', False)
    with pytest.raises(OSError, match="already exists"):
        _scans_tsv(raw, op.join('meg', raw_file_bids), scans_fname,
                   'error', False)
