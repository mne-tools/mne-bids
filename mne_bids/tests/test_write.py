# -*- coding: utf-8 -*-
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
from glob import glob

from datetime import datetime
from numpy.testing import assert_array_equal

import mne
from mne.datasets import testing
from mne.utils import _TempDir, run_subprocess, check_version
from mne.io.constants import FIFF

from mne_bids import (write_raw_bids, read_raw_bids, make_bids_basename,
                      make_bids_folders)
from mne_bids.tsv_handler import _from_tsv

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

    # check if write_raw_bids works when there is no stim channel
    raw.set_channel_types({raw.ch_names[i]: 'misc'
                           for i in
                           mne.pick_types(raw.info, stim=True, meg=False)})
    output_path = _TempDir()
    write_raw_bids(raw, bids_basename, output_path, overwrite=False)

    cmd = ['bids-validator', output_path]
    run_subprocess(cmd, shell=shell)

    # write the same data but pretend it is empty room data:
    er_date = datetime.fromtimestamp(
        raw.info['meas_date'][0]).strftime('%Y%m%d')
    er_bids_basename = 'sub-emptyroom_ses-{0}_task-noise'.format(str(er_date))
    write_raw_bids(raw, er_bids_basename, output_path, overwrite=False)
    assert op.exists(op.join(
        output_path, 'sub-emptyroom', 'ses-{0}'.format(er_date), 'meg',
        'sub-emptyroom_ses-{0}_task-noise_meg.json'.format(er_date)))
    # test that an incorrect date raises an error.
    er_bids_basename_bad = 'sub-emptyroom_ses-19000101_task-noise'
    with pytest.raises(ValueError, match='Date provided'):
        write_raw_bids(raw, er_bids_basename_bad, output_path, overwrite=False)

    # give the raw object some fake participant data
    raw = mne.io.read_raw_fif(raw_fname)
    raw.anonymize()
    raw.info['subject_info'] = {'his_id': subject_id2,
                                'birthday': (1994, 1, 26), 'sex': 1}
    data_path2 = _TempDir()
    raw_fname2 = op.join(data_path2, 'sample_audvis_raw.fif')
    raw.save(raw_fname2)

    bids_basename2 = bids_basename.replace(subject_id, subject_id2)
    raw = mne.io.read_raw_fif(raw_fname2)
    bids_output_path = write_raw_bids(raw, bids_basename2, output_path,
                                      events_data=events_fname,
                                      event_id=event_id, overwrite=False)

    # check that the overwrite parameters work correctly for the participant
    # data
    # change the gender but don't force overwrite.
    raw.info['subject_info'] = {'his_id': subject_id2,
                                'birthday': (1994, 1, 26), 'sex': 2}
    with pytest.raises(FileExistsError, match="already exists"):  # noqa: F821
        write_raw_bids(raw, bids_basename2, output_path,
                       events_data=events_fname, event_id=event_id,
                       overwrite=False)
    # now force the overwrite
    write_raw_bids(raw, bids_basename2, output_path, events_data=events_fname,
                   event_id=event_id, overwrite=True)
    raw, events, _ = read_raw_bids(bids_basename2 + '_meg.fif', output_path)
    events2 = mne.read_events(events_fname)
    events2 = events2[events2[:, 2] != 0]
    assert_array_equal(events2[:, 0], events[:, 0])

    with pytest.raises(ValueError, match='raw_file must be'):
        write_raw_bids('blah', bids_basename, output_path)

    bids_basename2 = 'sub-01_ses-01_xyz-01_run-01'
    with pytest.raises(KeyError, match='Unexpected entity'):
        write_raw_bids(raw, bids_basename2, output_path)

    bids_basename2 = 'sub-01_run-01_task-auditory'
    with pytest.raises(ValueError, match='ordered correctly'):
        write_raw_bids(raw, bids_basename2, output_path, overwrite=True)

    del raw._filenames
    with pytest.raises(ValueError, match='raw.filenames is missing'):
        write_raw_bids(raw, bids_basename2, output_path)

    cmd = ['bids-validator', output_path]
    run_subprocess(cmd, shell=shell)

    assert op.exists(op.join(output_path, 'participants.tsv'))

    # asserting that single fif files do not include the part key
    files = glob(op.join(bids_output_path, 'sub-' + subject_id2,
                         'ses-' + subject_id2, 'meg', '*.fif'))
    for ii, FILE in enumerate(files):
        assert 'part' not in FILE
    assert ii < 1

    # check that split files have part key
    raw = mne.io.read_raw_fif(raw_fname)
    data_path3 = _TempDir()
    raw_fname3 = op.join(data_path3, 'sample_audvis_raw.fif')
    raw.save(raw_fname3, buffer_size_sec=1.0, split_size='10MB',
             split_naming='neuromag', overwrite=True)
    raw = mne.io.read_raw_fif(raw_fname3)
    subject_id3 = '03'
    bids_basename3 = bids_basename.replace(subject_id, subject_id3)
    bids_output_path = write_raw_bids(raw, bids_basename3, output_path,
                                      overwrite=False)
    files = glob(op.join(bids_output_path, 'sub-' + subject_id3,
                         'ses-' + subject_id3, 'meg', '*.fif'))
    for FILE in files:
        assert 'part' in FILE


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

    raw = mne.io.read_raw_kit(
        raw_fname, mrk=hpi_fname, elp=electrode_fname,
        hsp=headshape_fname)
    write_raw_bids(raw, bids_basename, output_path, events_data=events_fname,
                   event_id=event_id, overwrite=False)
    cmd = ['bids-validator', output_path]
    run_subprocess(cmd, shell=shell)
    assert op.exists(op.join(output_path, 'participants.tsv'))

    raw, events, _ = read_raw_bids(bids_basename + '_meg.sqd',
                                   output_path)

    # ensure the channels file has no STI 014 channel:
    channels_tsv = make_bids_basename(
        subject=subject_id, session=session_id, task=task, run=run,
        suffix='channels.tsv', acquisition=acq,
        prefix=op.join(output_path, 'sub-01', 'ses-01', 'meg'))
    data = _from_tsv(channels_tsv)
    assert 'STI 014' not in data['name']

    # ensure the marker file is produced in the right place
    marker_fname = make_bids_basename(
        subject=subject_id, session=session_id, task=task, run=run,
        acquisition=acq, suffix='markers.sqd',
        prefix=op.join(output_path, 'sub-01', 'ses-01', 'meg'))
    assert op.exists(marker_fname)


def test_ctf():
    """Test functionality of the write_raw_bids conversion for CTF data."""
    output_path = _TempDir()
    data_path = op.join(testing.data_path(download=False), 'CTF')
    raw_fname = op.join(data_path, 'testdata_ctf.ds')

    raw = mne.io.read_raw_ctf(raw_fname)
    write_raw_bids(raw, bids_basename, output_path=output_path)

    cmd = ['bids-validator', output_path]
    run_subprocess(cmd, shell=shell)

    with pytest.raises(ValueError, match="not found"):
        read_raw_bids(bids_basename + '_meg.ds', output_path)

    raw = read_raw_bids(bids_basename + '_meg.ds', output_path,
                        return_events=False)

    # test to check that running again with overwrite == False raises an error
    with pytest.raises(FileExistsError, match="already exists"):  # noqa: F821
        write_raw_bids(raw, bids_basename, output_path=output_path)

    assert op.exists(op.join(output_path, 'participants.tsv'))


def test_bti():
    """Test functionality of the write_raw_bids conversion for BTi data."""
    output_path = _TempDir()
    data_path = op.join(base_path, 'bti', 'tests', 'data')
    raw_fname = op.join(data_path, 'test_pdf_linux')
    config_fname = op.join(data_path, 'test_config_linux')
    headshape_fname = op.join(data_path, 'test_hs_linux')

    raw = mne.io.read_raw_bti(raw_fname, config_fname=config_fname,
                              head_shape_fname=headshape_fname)

    write_raw_bids(raw, bids_basename, output_path, verbose=True)

    assert op.exists(op.join(output_path, 'participants.tsv'))

    cmd = ['bids-validator', output_path]
    run_subprocess(cmd, shell=shell)

    raw = read_raw_bids(bids_basename + '_meg',
                        output_path, return_events=False)


def test_vhdr():
    """Test write_raw_bids conversion for BrainVision data."""
    output_path = _TempDir()
    data_path = op.join(base_path, 'brainvision', 'tests', 'data')
    raw_fname = op.join(data_path, 'test.vhdr')

    raw = mne.io.read_raw_brainvision(raw_fname)
    write_raw_bids(raw, bids_basename, output_path, overwrite=False)

    cmd = ['bids-validator', output_path]
    run_subprocess(cmd, shell=shell)

    read_raw_bids(bids_basename + '_eeg.vhdr',
                  output_path, return_events=False)

    # Test that correct channel units are written
    channels_tsv_name = op.join(output_path, 'sub-' + subject_id,
                                'ses-' + session_id, 'eeg',
                                bids_basename + '_channels.tsv')
    data = _from_tsv(channels_tsv_name)
    assert data['units'][data['name'].index('FP1')] == 'µV'
    assert data['units'][data['name'].index('CP5')] == 'n/a'

    # check events.tsv is written
    events_tsv_fname = channels_tsv_name.replace('channels', 'events')
    assert op.exists(events_tsv_fname)

    # create another bids folder with the overwrite command and check
    # no files are in the folder
    data_path = make_bids_folders(subject=subject_id, session=session_id,
                                  kind='eeg', output_path=output_path,
                                  overwrite=True)
    assert len([f for f in os.listdir(data_path) if op.isfile(f)]) == 0

    # Also cover iEEG
    # We use the same data and pretend that eeg channels are ecog
    raw.set_channel_types({raw.ch_names[i]: 'ecog'
                           for i in mne.pick_types(raw.info, eeg=True)})
    output_path = _TempDir()
    write_raw_bids(raw, bids_basename, output_path, overwrite=False)

    cmd = ['bids-validator', output_path]
    run_subprocess(cmd, shell=shell)


def test_edf():
    """Test write_raw_bids conversion for European Data Format data."""
    output_path = _TempDir()
    data_path = op.join(testing.data_path(), 'EDF')
    raw_fname = op.join(data_path, 'test_reduced.edf')

    raw = mne.io.read_raw_edf(raw_fname, preload=True)
    # XXX: hack that should be fixed later. Annotation reading is
    # broken for this file with preload=False and read_annotations_edf
    raw.preload = False

    raw.rename_channels({raw.info['ch_names'][0]: 'EOG'})
    raw.info['chs'][0]['coil_type'] = FIFF.FIFFV_COIL_EEG_BIPOLAR
    raw.rename_channels({raw.info['ch_names'][1]: 'EMG'})
    raw.set_channel_types({'EMG': 'emg'})

    write_raw_bids(raw, bids_basename, output_path)
    read_raw_bids(bids_basename + '_eeg.edf',
                  output_path, return_events=False)

    bids_fname = bids_basename.replace('run-01', 'run-%s' % run2)
    write_raw_bids(raw, bids_fname, output_path, overwrite=True)

    cmd = ['bids-validator', output_path]
    run_subprocess(cmd, shell=shell)

    # ensure there is an EMG channel in the channels.tsv:
    channels_tsv = make_bids_basename(
        subject=subject_id, session=session_id, task=task, run=run,
        suffix='channels.tsv', acquisition=acq,
        prefix=op.join(output_path, 'sub-01', 'ses-01', 'eeg'))
    data = _from_tsv(channels_tsv)
    assert 'ElectroMyoGram' in data['description']

    # check that the scans list contains two scans
    scans_tsv = make_bids_basename(
        subject=subject_id, session=session_id, suffix='scans.tsv',
        prefix=op.join(output_path, 'sub-01', 'ses-01'))
    data = _from_tsv(scans_tsv)
    assert len(list(data.values())[0]) == 2

    # Also cover iEEG
    # We use the same data and pretend that eeg channels are ecog
    raw.set_channel_types({raw.ch_names[i]: 'ecog'
                           for i in mne.pick_types(raw.info, eeg=True)})
    output_path = _TempDir()
    write_raw_bids(raw, bids_basename, output_path)

    cmd = ['bids-validator', output_path]
    run_subprocess(cmd, shell=shell)


def test_bdf():
    """Test write_raw_bids conversion for Biosemi data."""
    output_path = _TempDir()
    data_path = op.join(base_path, 'edf', 'tests', 'data')
    raw_fname = op.join(data_path, 'test.bdf')

    raw = mne.io.read_raw_edf(raw_fname)
    write_raw_bids(raw, bids_basename, output_path, overwrite=False)

    cmd = ['bids-validator', output_path]
    run_subprocess(cmd, shell=shell)

    read_raw_bids(bids_basename + '_eeg.bdf',
                  output_path, return_events=False)

    raw.crop(0, raw.times[-2])
    with pytest.raises(AssertionError, match='cropped'):
        write_raw_bids(raw, bids_basename, output_path)


def test_set():
    """Test write_raw_bids conversion for EEGLAB data."""
    # standalone .set file
    output_path = _TempDir()
    data_path = op.join(testing.data_path(), 'EEGLAB')

    # .set with associated .fdt
    output_path = _TempDir()
    data_path = op.join(testing.data_path(), 'EEGLAB')
    raw_fname = op.join(data_path, 'test_raw.set')

    raw = mne.io.read_raw_eeglab(raw_fname)

    # embedded - test mne-version assertion
    tmp_version = mne.__version__
    mne.__version__ = '0.16'
    with pytest.raises(ValueError, match='Your version of MNE is too old.'):
        write_raw_bids(raw, bids_basename, output_path)
    mne.__version__ = tmp_version

    # proceed with the actual test for EEGLAB data
    write_raw_bids(raw, bids_basename, output_path, overwrite=False)
    read_raw_bids(bids_basename + '_eeg.set',
                  output_path, return_events=False)

    with pytest.raises(FileExistsError, match="already exists"):  # noqa: F821
        write_raw_bids(raw, bids_basename, output_path=output_path,
                       overwrite=False)

    cmd = ['bids-validator', output_path]
    run_subprocess(cmd, shell=shell)

    # check events.tsv is written
    # XXX: only from 0.18 onwards because events_from_annotations
    # is broken for earlier versions
    events_tsv_fname = op.join(output_path, 'sub-' + subject_id,
                               'ses-' + session_id, 'eeg',
                               bids_basename + '_events.tsv')
    if check_version('mne', '0.18dev0'):
        assert op.exists(events_tsv_fname)

    # Also cover iEEG
    # We use the same data and pretend that eeg channels are ecog
    raw.set_channel_types({raw.ch_names[i]: 'ecog'
                           for i in mne.pick_types(raw.info, eeg=True)})
    output_path = _TempDir()
    write_raw_bids(raw, bids_basename, output_path)

    cmd = ['bids-validator', output_path]
    run_subprocess(cmd, shell=shell)
