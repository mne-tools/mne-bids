"""Test the MNE BIDS converter.

For each supported file format, implement a test.
"""
# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Teon L Brooks <teon.brooks@gmail.com>
#          Chris Holdgraf <choldgraf@berkeley.edu>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)

import os
import os.path as op
import pytest
import subprocess

import pandas as pd
import mne
from mne.datasets import testing
from mne.utils import _TempDir, run_subprocess
from mne.io.constants import FIFF

from mne_bids import raw_to_bids, make_bids_filename

base_path = op.join(op.dirname(mne.__file__), 'io')
subject_id = '01'
subject_id2 = '02'
session_id = '01'
run = '01'
acq = '01'
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
                event_id=event_id, overwrite=True)

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
                event_id=event_id, overwrite=True)
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
                output_path=output_path, overwrite=True)
    cmd = ['bids-validator', output_path]
    run_subprocess(cmd, shell=shell)
    assert op.exists(op.join(output_path, 'participants.tsv'))

    # ensure the channels file has no STI 014 channel:
    channels_tsv = make_bids_filename(
        subject=subject_id, session=session_id, task=task, run=run,
        suffix='channels.tsv', acquisition=acq,
        prefix=op.join(output_path, 'sub-01/ses-01/meg'))
    if op.exists(channels_tsv):
        df = pd.read_csv(channels_tsv, sep='\t')
        assert not ('STI 014' in df['name'].values)


def test_ctf():
    """Test functionality of the raw_to_bids conversion for CTF data."""
    output_path = _TempDir()
    data_path = op.join(testing.data_path(download=False), 'CTF')
    raw_fname = op.join(data_path, 'testdata_ctf.ds')

    raw_to_bids(subject_id=subject_id, session_id=session_id, run=run,
                task=task, acquisition=acq, raw_file=raw_fname,
                output_path=output_path, overwrite=True)
    cmd = ['bids-validator', output_path]
    run_subprocess(cmd, shell=shell)
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
                output_path=output_path, verbose=True, overwrite=True)

    assert op.exists(op.join(output_path, 'participants.tsv'))

    # FIXME: see these issues for reference:
    # https://github.com/mne-tools/mne-bids/pull/84
    # https://github.com/INCF/bids-validator/issues/553
    with pytest.raises(subprocess.CalledProcessError):
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
                output_path=output_path, overwrite=True, kind='eeg')

    cmd = ['bids-validator', '--bep006', output_path]
    run_subprocess(cmd, shell=shell)


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
                output_path=output_path, overwrite=True, kind='eeg')

    cmd = ['bids-validator', '--bep006', output_path]
    run_subprocess(cmd, shell=shell)

    # ensure there is an EMG channel in the channels.tsv:
    channels_tsv = make_bids_filename(
        subject=subject_id, session=session_id, task=task, run=run,
        suffix='channels.tsv', acquisition=acq,
        prefix=op.join(output_path, 'sub-01/ses-01/eeg'))
    if op.exists(channels_tsv):
        df = pd.read_csv(channels_tsv, sep='\t')
        assert 'ElectroMyoGram' in df['description'].values


def test_bdf():
    """Test raw_to_bids conversion for Biosemi data."""
    output_path = _TempDir()
    data_path = op.join(base_path, 'edf', 'tests', 'data')
    raw_fname = op.join(data_path, 'test.bdf')

    raw_to_bids(subject_id=subject_id, session_id=session_id, run=run,
                task=task, acquisition=acq, raw_file=raw_fname,
                output_path=output_path, overwrite=True, kind='eeg')

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
                output_path=output_path, overwrite=True, kind='eeg')

    cmd = ['bids-validator', '--bep006', output_path]
    run_subprocess(cmd, shell=shell)

    # .set with associated .fdt
    output_path = _TempDir()
    data_path = op.join(testing.data_path(), 'EEGLAB')
    raw_fname = op.join(data_path, 'test_raw.set')

    raw_to_bids(subject_id=subject_id, session_id=session_id, run=run,
                task=task, raw_file=raw_fname, output_path=output_path,
                overwrite=True, kind='eeg')

    cmd = ['bids-validator', '--bep006', output_path]
    run_subprocess(cmd, shell=shell)


def test_cnt():
    """Test raw_to_bids conversion for Neuroscan data."""
    output_path = _TempDir()
    data_path = op.join(testing.data_path(), 'CNT')
    raw_fname = op.join(data_path, 'scan41_short.cnt')

    raw_to_bids(subject_id=subject_id, session_id=session_id, run=run,
                task=task, acquisition=acq, raw_file=raw_fname,
                output_path=output_path, overwrite=True, kind='eeg')

    cmd = ['bids-validator', '--bep006', output_path]
    run_subprocess(cmd, shell=shell)
