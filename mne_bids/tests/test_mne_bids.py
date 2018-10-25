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
import pytest

import pandas as pd

import mne
from mne.datasets import testing
from mne.utils import _TempDir, run_subprocess
from mne.io.constants import FIFF

from mne_bids import make_bids_basename, make_bids_folders, write_raw_bids

base_path = op.join(op.dirname(mne.__file__), 'io')
subject_id = '01'
subject_id2 = '02'
session_id = '01'
run = '01'
acq = '01'
run2 = '02'
task = 'testing'

bids_basename = make_bids_basename(
    subject=subject_id, session=session_id, run=run, acquisition=acq,
    task=task)

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
    """Test functionality of the write_raw_bids conversion for fif."""
    output_path = _TempDir()
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')

    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
                'Visual/Right': 4, 'Smiley': 5, 'Button': 32}
    events_fname = op.join(data_path, 'MEG', 'sample',
                           'sample_audvis_trunc_raw-eve.fif')

    raw = mne.io.read_raw_fif(raw_fname)
    write_raw_bids(raw, bids_basename, output_path, events_data=events_fname,
                   event_id=event_id, overwrite=False)

    # give the raw object some fake participant data
    raw = mne.io.read_raw_fif(raw_fname)
    raw.anonymize()
    raw.info['subject_info'] = {'his_id': subject_id2,
                                'birthday': (1994, 1, 26), 'sex': 1}
    data_path2 = _TempDir()
    raw_fname2 = op.join(data_path2, 'sample_audvis_raw.fif')
    raw.save(raw_fname2)

    bids_fname = bids_basename.replace(subject_id, subject_id2)
    write_raw_bids(raw, bids_fname, output_path, events_data=events_fname,
                   event_id=event_id, overwrite=False)
    # check that the overwrite parameters work correctly for the participant
    # data
    # change the gender but don't force overwrite.
    raw.info['subject_info'] = {'his_id': subject_id2,
                                'birthday': (1994, 1, 26), 'sex': 2}
    with pytest.raises(OSError, match="already exists"):
        write_raw_bids(raw, bids_fname, output_path,
                       events_data=events_fname, event_id=event_id,
                       overwrite=False)
    # now force the overwrite
    write_raw_bids(raw, bids_fname, output_path, events_data=events_fname,
                   event_id=event_id, overwrite=True)

    with pytest.raises(ValueError, match='raw_file must be'):
        write_raw_bids('blah', bids_basename, output_path)

    bids_fname = 'sub-01_ses-01_xyz-01_run-01'
    with pytest.raises(KeyError, match='Unexpected entity'):
        write_raw_bids(raw, bids_fname, output_path)

    bids_fname = 'sub-01_run-01_task-auditory'
    with pytest.raises(ValueError, match='ordered correctly'):
        write_raw_bids(raw, bids_fname, output_path, overwrite=True)

    bids_fname = 'sub-01_task-auditory_run-01'
    with pytest.raises(ValueError, match='convert filetype'):
        write_raw_bids(raw, bids_fname, output_path, overwrite=True)

    del raw._filenames
    with pytest.raises(ValueError, match='raw.filenames is missing'):
        write_raw_bids(raw, bids_fname, output_path)

    cmd = ['bids-validator', output_path]
    run_subprocess(cmd, shell=shell)

    assert op.exists(op.join(output_path, 'participants.tsv'))


def test_kit():
    """Test functionality of the write_raw_bids conversion for KIT data."""
    output_path = _TempDir()
    data_path = op.join(base_path, 'kit', 'tests', 'data')
    raw_fname = op.join(data_path, 'test.sqd')
    events_fname = op.join(data_path, 'test-eve.txt')
    hpi_fname = op.join(data_path, 'test_mrk.sqd')
    electrode_fname = op.join(data_path, 'test_elp.txt')
    headshape_fname = op.join(data_path, 'test_hsp.txt')
    event_id = dict(cond=1)

    bids_fname = bids_basename + '_meg.sqd'
    raw = mne.io.read_raw_kit(
        raw_fname, hpi=hpi_fname, electrode=electrode_fname,
        hsp=headshape_fname)
    write_raw_bids(raw, bids_fname, output_path, events_data=events_fname,
                   event_id=event_id, hpi=hpi_fname, overwrite=False)
    cmd = ['bids-validator', output_path]
    run_subprocess(cmd, shell=shell)
    assert op.exists(op.join(output_path, 'participants.tsv'))

    # ensure the channels file has no STI 014 channel:
    channels_tsv = make_bids_basename(
        subject=subject_id, session=session_id, task=task, run=run,
        suffix='channels.tsv', acquisition=acq,
        prefix=op.join(output_path, 'sub-01/ses-01/meg'))
    df = pd.read_csv(channels_tsv, sep='\t')
    assert not ('STI 014' in df['name'].values)

    # ensure the marker file is produced in the right place
    raw_folder = make_bids_basename(
        subject=subject_id, session=session_id, task=task, run=run,
        acquisition=acq, suffix='%s' % 'meg')
    marker_fname = make_bids_basename(
        subject=subject_id, session=session_id, task=task, run=run,
        acquisition=acq, suffix='markers.sqd',
        prefix=os.path.join(output_path, 'sub-01/ses-01/meg', raw_folder))
    assert op.exists(marker_fname)


def test_ctf():
    """Test functionality of the write_raw_bids conversion for CTF data."""
    output_path = _TempDir()
    data_path = op.join(testing.data_path(download=False), 'CTF')
    raw_fname = op.join(data_path, 'testdata_ctf.ds')

    bids_fname = bids_basename + '_meg.ds'
    raw = mne.io.read_raw_ctf(raw_fname)
    write_raw_bids(raw, bids_fname, output_path=output_path)
    cmd = ['bids-validator', output_path]
    run_subprocess(cmd, shell=shell)

    # test to check that running again with overwrite == False raises an error
    with pytest.raises(OSError, match="already exists"):
        write_raw_bids(raw, bids_fname, output_path=output_path)

    assert op.exists(op.join(output_path, 'participants.tsv'))


def test_bti():
    """Test functionality of the write_raw_bids conversion for BTi data."""
    output_path = _TempDir()
    data_path = op.join(base_path, 'bti', 'tests', 'data')
    raw_fname = op.join(data_path, 'test_pdf_linux')
    config_fname = op.join(data_path, 'test_config_linux')
    headshape_fname = op.join(data_path, 'test_hs_linux')

    bids_fname = bids_basename + '_meg.pdf'
    raw = mne.io.read_raw_bti(raw_fname, config=config_fname,
                              hsp=headshape_fname)
    write_raw_bids(raw, bids_fname, output_path, verbose=True)

    assert op.exists(op.join(output_path, 'participants.tsv'))

    cmd = ['bids-validator', output_path]
    run_subprocess(cmd, shell=shell)


# EEG Tests
# ---------
def test_vhdr():
    """Test write_raw_bids conversion for BrainVision data."""
    output_path = _TempDir()
    data_path = op.join(base_path, 'brainvision', 'tests', 'data')
    raw_fname = op.join(data_path, 'test.vhdr')

    bids_fname = bids_basename + '_eeg.vhdr'
    raw = mne.io.read_raw_brainvision(raw_fname)
    write_raw_bids(raw, bids_fname, output_path, overwrite=False)

    cmd = ['bids-validator', '--bep006', output_path]
    run_subprocess(cmd, shell=shell)

    # create another bids folder with the overwrite command and check
    # no files are in the folder
    data_path = make_bids_folders(subject=subject_id, session=session_id,
                                  kind='eeg', root=output_path,
                                  overwrite=True)
    assert len([f for f in os.listdir(data_path) if op.isfile(f)]) == 0


def test_edf():
    """Test write_raw_bids conversion for European Data Format data."""
    output_path = _TempDir()
    data_path = op.join(testing.data_path(), 'EDF')
    raw_fname = op.join(data_path, 'test_reduced.edf')

    raw = mne.io.read_raw_edf(raw_fname, preload=True)
    raw.rename_channels({raw.info['ch_names'][0]: 'EOG'})
    raw.info['chs'][0]['coil_type'] = FIFF.FIFFV_COIL_EEG_BIPOLAR
    raw.rename_channels({raw.info['ch_names'][1]: 'EMG'})
    raw.set_channel_types({'EMG': 'emg'})

    bids_fname = bids_basename + '_eeg.edf'
    write_raw_bids(raw, bids_fname, output_path)
    bids_fname = bids_fname.replace('run-01', 'run-%s' % run2)
    write_raw_bids(raw, bids_fname, output_path, overwrite=True)

    cmd = ['bids-validator', '--bep006', output_path]
    run_subprocess(cmd, shell=shell)

    # ensure there is an EMG channel in the channels.tsv:
    channels_tsv = make_bids_basename(
        subject=subject_id, session=session_id, task=task, run=run,
        suffix='channels.tsv', acquisition=acq,
        prefix=op.join(output_path, 'sub-01/ses-01/eeg'))
    df = pd.read_csv(channels_tsv, sep='\t')
    assert 'ElectroMyoGram' in df['description'].values

    # check that the scans list contains two scans
    scans_tsv = make_bids_basename(
        subject=subject_id, session=session_id, suffix='scans.tsv',
        prefix=op.join(output_path, 'sub-01/ses-01'))
    df = pd.read_csv(scans_tsv, sep='\t')
    assert df.shape[0] == 2


def test_bdf():
    """Test write_raw_bids conversion for Biosemi data."""
    output_path = _TempDir()
    data_path = op.join(base_path, 'edf', 'tests', 'data')
    raw_fname = op.join(data_path, 'test.bdf')

    bids_fname = bids_basename + '_eeg.bdf'
    raw = mne.io.read_raw_edf(raw_fname)
    write_raw_bids(raw, bids_fname, output_path, overwrite=False)

    cmd = ['bids-validator', '--bep006', output_path]
    run_subprocess(cmd, shell=shell)


def test_set():
    """Test write_raw_bids conversion for EEGLAB data."""
    # standalone .set file
    output_path = _TempDir()
    data_path = op.join(testing.data_path(), 'EEGLAB')
    raw_fname = op.join(data_path, 'test_raw_onefile.set')

    bids_fname = bids_basename + '_eeg.set'
    raw = mne.io.read_raw_eeglab(raw_fname)
    # XXX: remove hack below once mne v0.17 is released
    raw._filenames = [raw_fname]
    write_raw_bids(raw, bids_fname, output_path, overwrite=False)

    cmd = ['bids-validator', '--bep006', output_path]
    run_subprocess(cmd, shell=shell)

    with pytest.raises(OSError, match="already exists"):
        write_raw_bids(raw, bids_fname, output_path=output_path,
                       overwrite=False)

    # .set with associated .fdt
    output_path = _TempDir()
    data_path = op.join(testing.data_path(), 'EEGLAB')
    raw_fname = op.join(data_path, 'test_raw.set')

    raw = mne.io.read_raw_eeglab(raw_fname)
    write_raw_bids(raw, bids_fname, output_path, overwrite=False)

    cmd = ['bids-validator', '--bep006', output_path]
    run_subprocess(cmd, shell=shell)


def test_cnt():
    """Test write_raw_bids conversion for Neuroscan data."""
    output_path = _TempDir()
    data_path = op.join(testing.data_path(), 'CNT')
    raw_fname = op.join(data_path, 'scan41_short.cnt')

    bids_fname = bids_basename + '_eeg.cnt'
    raw = mne.io.read_raw_cnt(raw_fname)
    write_raw_bids(raw, bids_fname, output_path)

    cmd = ['bids-validator', '--bep006', output_path]
    run_subprocess(cmd, shell=shell)
