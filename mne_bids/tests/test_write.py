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
from datetime import datetime, timezone
import platform
import shutil as sh
import json
from distutils.version import LooseVersion

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import mne
from mne.datasets import testing
from mne.utils import (_TempDir, run_subprocess, check_version,
                       requires_nibabel, requires_version)
from mne.io.constants import FIFF
from mne.io.kit.kit import get_kit_info

from mne_bids import (write_raw_bids, read_raw_bids, make_bids_basename,
                      make_bids_folders, write_anat,
                      get_anonymization_daysback)
from mne_bids.write import _stamp_to_dt, _get_anonymization_daysback
from mne_bids.tsv_handler import _from_tsv, _to_tsv
from mne_bids.utils import _find_matching_sidecar
from mne_bids.pick import coil_type

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
bids_basename_minimal = make_bids_basename(subject=subject_id, task=task)


# WINDOWS issues:
# the bids-validator development version does not work properly on Windows as
# of 2019-06-25 --> https://github.com/bids-standard/bids-validator/issues/790
# As a workaround, we try to get the path to the executable from an environment
# variable VALIDATOR_EXECUTABLE ... if this is not possible we assume to be
# using the stable bids-validator and make a direct call of bids-validator
# also: for windows, shell = True is needed to call npm, bids-validator etc.
# see: https://stackoverflow.com/q/28891053/5201771
@pytest.fixture(scope="session")
def _bids_validate():
    """Fixture to run BIDS validator."""
    shell = False
    bids_validator_exe = ['bids-validator', '--config.error=41',
                          '--config.error=41']
    if platform.system() == 'Windows':
        shell = True
        exe = os.getenv('VALIDATOR_EXECUTABLE', 'n/a')
        if 'VALIDATOR_EXECUTABLE' != 'n/a':
            bids_validator_exe = ['node', exe]

    def _validate(output_path):
        cmd = bids_validator_exe + [output_path]
        run_subprocess(cmd, shell=shell)

    return _validate


@requires_version('pybv', '0.2.0')
def _test_anonymize(raw, bids_basename, events_fname=None, event_id=None):
    output_path = _TempDir()
    write_raw_bids(raw, bids_basename, output_path,
                   events_data=events_fname,
                   event_id=event_id, anonymize=dict(daysback=33000),
                   overwrite=False)
    scans_tsv = make_bids_basename(
        subject=subject_id, session=session_id, suffix='scans.tsv',
        prefix=op.join(output_path, 'sub-01', 'ses-01'))
    data = _from_tsv(scans_tsv)
    if data['acq_time'] is not None and data['acq_time'][0] != 'n/a':
        assert datetime.strptime(data['acq_time'][0],
                                 '%Y-%m-%dT%H:%M:%S').year < 1925
    return output_path


def test_stamp_to_dt():
    """Test conversions of meas_date to datetime objects."""
    meas_date = (1346981585, 835782)
    meas_datetime = _stamp_to_dt(meas_date)
    assert(meas_datetime == datetime(2012, 9, 7, 1, 33, 5, 835782,
                                     tzinfo=timezone.utc))
    meas_date = (1346981585,)
    meas_datetime = _stamp_to_dt(meas_date)
    assert(meas_datetime == datetime(2012, 9, 7, 1, 33, 5, 0,
                                     tzinfo=timezone.utc))


def test_get_anonymization_daysback():
    """Test daysback querying for anonymization."""
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    raw = mne.io.read_raw_fif(raw_fname)
    daysback_min, daysback_max = _get_anonymization_daysback(raw)
    # max_val off by 1 on Windows for some reason
    assert abs(daysback_min - 28461) < 2 and abs(daysback_max - 36880) < 2
    raw2 = raw.copy()
    raw2.info['meas_date'] = (np.int32(1158942080), np.int32(720100))
    raw3 = raw.copy()
    raw3.info['meas_date'] = (np.int32(914992080), np.int32(720100))
    daysback_min, daysback_max = get_anonymization_daysback([raw, raw2, raw3])
    assert abs(daysback_min - 29850) < 2 and abs(daysback_max - 35446) < 2
    raw4 = raw.copy()
    raw4.info['meas_date'] = (np.int32(4992080), np.int32(720100))
    raw5 = raw.copy()
    raw5.info['meas_date'] = None
    daysback_min2, daysback_max2 = get_anonymization_daysback([raw, raw2,
                                                               raw3, raw5])
    assert daysback_min2 == daysback_min and daysback_max2 == daysback_max
    with pytest.raises(ValueError, match='The dataset spans more time'):
        daysback_min, daysback_max = \
            get_anonymization_daysback([raw, raw2, raw4])


@requires_version('pybv', '0.2.0')
def test_fif(_bids_validate):
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

    # Read the file back in to check that the data has come through cleanly.
    # Events and bad channel information was read through JSON sidecar files.
    with pytest.raises(TypeError, match="unexpected keyword argument 'foo'"):
        read_raw_bids(bids_basename + '_meg.fif', output_path,
                      extra_params=dict(foo='bar'))

    raw2 = read_raw_bids(bids_basename + '_meg.fif', output_path,
                         extra_params=dict(allow_maxshield=True))
    assert set(raw.info['bads']) == set(raw2.info['bads'])
    events, _ = mne.events_from_annotations(raw2)
    events2 = mne.read_events(events_fname)
    events2 = events2[events2[:, 2] != 0]
    assert_array_equal(events2[:, 0], events[:, 0])

    # check if write_raw_bids works when there is no stim channel
    raw.set_channel_types({raw.ch_names[i]: 'misc'
                           for i in
                           mne.pick_types(raw.info, stim=True, meg=False)})
    output_path = _TempDir()
    with pytest.warns(UserWarning, match='No events found or provided.'):
        write_raw_bids(raw, bids_basename, output_path, overwrite=False)

    _bids_validate(output_path)

    # try with eeg data only (conversion to bv)
    output_path = _TempDir()
    raw = mne.io.read_raw_fif(raw_fname)
    raw.load_data()
    raw2 = raw.pick_types(meg=False, eeg=True, stim=True, eog=True, ecg=True)
    raw2.save(op.join(output_path, 'test-raw.fif'), overwrite=True)
    raw2 = mne.io.Raw(op.join(output_path, 'test-raw.fif'), preload=False)
    events = mne.find_events(raw2)
    event_id = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
                'visual/right': 4, 'smiley': 5, 'button': 32}
    epochs = mne.Epochs(raw2, events, event_id=event_id, tmin=-0.2, tmax=0.5,
                        preload=True)
    with pytest.warns(UserWarning,
                      match='Converting data files to BrainVision format'):
        write_raw_bids(raw2, bids_basename, output_path,
                       events_data=events_fname, event_id=event_id,
                       verbose=True, overwrite=False)
    bids_dir = op.join(output_path, 'sub-%s' % subject_id,
                       'ses-%s' % session_id, 'eeg')
    for sidecar in ['channels.tsv', 'eeg.eeg', 'eeg.json', 'eeg.vhdr',
                    'eeg.vmrk', 'events.tsv']:
        assert op.isfile(op.join(bids_dir, bids_basename + '_' + sidecar))

    raw2 = read_raw_bids(bids_basename + '_eeg.vhdr', output_path)
    os.remove(op.join(output_path, 'test-raw.fif'))

    events2 = mne.find_events(raw2)
    epochs2 = mne.Epochs(raw2, events2, event_id=event_id, tmin=-0.2, tmax=0.5,
                         preload=True)
    assert_array_almost_equal(raw.get_data(), raw2.get_data())
    assert_array_almost_equal(epochs.get_data(), epochs2.get_data(), decimal=4)
    _bids_validate(output_path)

    # write the same data but pretend it is empty room data:
    raw = mne.io.read_raw_fif(raw_fname)
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

    # give the raw object some fake participant data (potentially overwriting)
    raw = mne.io.read_raw_fif(raw_fname)
    raw.info['subject_info'] = {'his_id': subject_id2,
                                'birthday': (1993, 1, 26), 'sex': 1}
    write_raw_bids(raw, bids_basename, output_path, events_data=events_fname,
                   event_id=event_id, overwrite=True)
    # assert age of participant is correct
    participants_tsv = op.join(output_path, 'participants.tsv')
    data = _from_tsv(participants_tsv)
    assert data['age'][data['participant_id'].index('sub-01')] == '9'

    # try and write preloaded data
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    with pytest.raises(ValueError, match='preloaded'):
        write_raw_bids(raw, bids_basename, output_path,
                       events_data=events_fname, event_id=event_id,
                       overwrite=False)

    # test anonymize
    raw = mne.io.read_raw_fif(raw_fname)
    raw.anonymize()

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

    _bids_validate(output_path)

    assert op.exists(op.join(output_path, 'participants.tsv'))

    # asserting that single fif files do not include the part key
    files = glob(op.join(bids_output_path, 'sub-' + subject_id2,
                         'ses-' + subject_id2, 'meg', '*.fif'))
    for ii, FILE in enumerate(files):
        assert 'part' not in FILE
    assert ii < 1

    # test keyword mne-bids anonymize
    raw = mne.io.read_raw_fif(raw_fname)
    with pytest.raises(ValueError, match='`daysback` argument required'):
        write_raw_bids(raw, bids_basename, output_path,
                       events_data=events_fname,
                       event_id=event_id,
                       anonymize=dict(),
                       overwrite=True)

    output_path = _TempDir()
    raw = mne.io.read_raw_fif(raw_fname)
    with pytest.warns(UserWarning, match='daysback` is too small'):
        write_raw_bids(raw, bids_basename, output_path,
                       events_data=events_fname,
                       event_id=event_id,
                       anonymize=dict(daysback=400),
                       overwrite=False)

    output_path = _TempDir()
    raw = mne.io.read_raw_fif(raw_fname)
    with pytest.raises(ValueError, match='`daysback` exceeds maximum value'):
        write_raw_bids(raw, bids_basename, output_path,
                       events_data=events_fname,
                       event_id=event_id,
                       anonymize=dict(daysback=40000),
                       overwrite=False)

    output_path = _TempDir()
    raw = mne.io.read_raw_fif(raw_fname)
    write_raw_bids(raw, bids_basename, output_path,
                   events_data=events_fname,
                   event_id=event_id,
                   anonymize=dict(daysback=30000, keep_his=True),
                   overwrite=False)
    scans_tsv = make_bids_basename(
        subject=subject_id, session=session_id, suffix='scans.tsv',
        prefix=op.join(output_path, 'sub-01', 'ses-01'))
    data = _from_tsv(scans_tsv)
    assert datetime.strptime(data['acq_time'][0],
                             '%Y-%m-%dT%H:%M:%S').year < 1925
    _bids_validate(output_path)

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

    # test unknown extention
    raw = mne.io.read_raw_fif(raw_fname)
    raw._filenames = (raw.filenames[0].replace('.fif', '.foo'),)
    with pytest.raises(ValueError, match='Unrecognized file format'):
        write_raw_bids(raw, bids_basename, output_path)


def test_kit(_bids_validate):
    """Test functionality of the write_raw_bids conversion for KIT data."""
    output_path = _TempDir()
    data_path = op.join(base_path, 'kit', 'tests', 'data')
    raw_fname = op.join(data_path, 'test.sqd')
    events_fname = op.join(data_path, 'test-eve.txt')
    hpi_fname = op.join(data_path, 'test_mrk.sqd')
    hpi_pre_fname = op.join(data_path, 'test_mrk_pre.sqd')
    hpi_post_fname = op.join(data_path, 'test_mrk_post.sqd')
    electrode_fname = op.join(data_path, 'test_elp.txt')
    headshape_fname = op.join(data_path, 'test_hsp.txt')
    event_id = dict(cond=1)

    kit_bids_basename = bids_basename.replace('_acq-01', '')

    raw = mne.io.read_raw_kit(
        raw_fname, mrk=hpi_fname, elp=electrode_fname,
        hsp=headshape_fname)
    write_raw_bids(raw, kit_bids_basename, output_path,
                   events_data=events_fname,
                   event_id=event_id, overwrite=False)

    _bids_validate(output_path)
    assert op.exists(op.join(output_path, 'participants.tsv'))

    read_raw_bids(kit_bids_basename + '_meg.sqd', output_path)

    # test anonymize
    output_path = _test_anonymize(raw, kit_bids_basename,
                                  events_fname, event_id)
    _bids_validate(output_path)

    # ensure the channels file has no STI 014 channel:
    channels_tsv = make_bids_basename(
        subject=subject_id, session=session_id, task=task, run=run,
        suffix='channels.tsv',
        prefix=op.join(output_path, 'sub-01', 'ses-01', 'meg'))
    data = _from_tsv(channels_tsv)
    assert 'STI 014' not in data['name']

    # ensure the marker file is produced in the right place
    marker_fname = make_bids_basename(
        subject=subject_id, session=session_id, task=task, run=run,
        suffix='markers.sqd',
        prefix=op.join(output_path, 'sub-01', 'ses-01', 'meg'))
    assert op.exists(marker_fname)

    # test attempts at writing invalid event data
    event_data = np.loadtxt(events_fname)
    # make the data the wrong number of dimensions
    event_data_3d = np.atleast_3d(event_data)
    other_output_path = _TempDir()
    with pytest.raises(ValueError, match='two dimensions'):
        write_raw_bids(raw, bids_basename, other_output_path,
                       events_data=event_data_3d, event_id=event_id,
                       overwrite=True)
    # remove 3rd column
    event_data = event_data[:, :2]
    with pytest.raises(ValueError, match='second dimension'):
        write_raw_bids(raw, bids_basename, other_output_path,
                       events_data=event_data, event_id=event_id,
                       overwrite=True)
    # test correct naming of marker files
    raw = mne.io.read_raw_kit(
        raw_fname, mrk=[hpi_pre_fname, hpi_post_fname], elp=electrode_fname,
        hsp=headshape_fname)
    write_raw_bids(raw,
                   kit_bids_basename.replace('sub-01', 'sub-%s' % subject_id2),
                   output_path, events_data=events_fname, event_id=event_id,
                   overwrite=False)

    _bids_validate(output_path)
    # ensure the marker files are renamed correctly
    marker_fname = make_bids_basename(
        subject=subject_id2, session=session_id, task=task, run=run,
        suffix='markers.sqd', acquisition='pre',
        prefix=os.path.join(output_path, 'sub-02', 'ses-01', 'meg'))
    info = get_kit_info(marker_fname, False)[0]
    assert info['meas_date'] == get_kit_info(hpi_pre_fname,
                                             False)[0]['meas_date']
    marker_fname = marker_fname.replace('acq-pre', 'acq-post')
    info = get_kit_info(marker_fname, False)[0]
    assert info['meas_date'] == get_kit_info(hpi_post_fname,
                                             False)[0]['meas_date']

    # check that providing markers in the wrong order raises an error
    raw = mne.io.read_raw_kit(
        raw_fname, mrk=[hpi_post_fname, hpi_pre_fname], elp=electrode_fname,
        hsp=headshape_fname)
    with pytest.raises(ValueError, match='Markers'):
        write_raw_bids(
            raw,
            kit_bids_basename.replace('sub-01', 'sub-%s' % subject_id2),
            output_path, events_data=events_fname, event_id=event_id,
            overwrite=True)


def test_ctf(_bids_validate):
    """Test functionality of the write_raw_bids conversion for CTF data."""
    output_path = _TempDir()
    data_path = op.join(testing.data_path(download=False), 'CTF')
    raw_fname = op.join(data_path, 'testdata_ctf.ds')

    raw = mne.io.read_raw_ctf(raw_fname)
    with pytest.warns(UserWarning, match='No line frequency'):
        write_raw_bids(raw, bids_basename, output_path=output_path)

    _bids_validate(output_path)
    with pytest.warns(UserWarning, match='Did not find any events'):
        raw = read_raw_bids(bids_basename + '_meg.ds', output_path,
                            extra_params=dict(clean_names=False))

    # test to check that running again with overwrite == False raises an error
    with pytest.raises(FileExistsError, match="already exists"):  # noqa: F821
        write_raw_bids(raw, bids_basename, output_path=output_path)

    assert op.exists(op.join(output_path, 'participants.tsv'))

    # test anonymize
    raw = mne.io.read_raw_ctf(raw_fname)
    with pytest.warns(UserWarning,
                      match='Converting to FIF for anonymization'):
        output_path = _test_anonymize(raw, bids_basename)
    _bids_validate(output_path)

    # XXX: change the next two lines once raw.set_meas_date() is
    # available.
    raw.info['meas_date'] = None
    raw.anonymize()
    with pytest.raises(ValueError, match='All measurement dates are None'):
        get_anonymization_daysback(raw)


def test_bti(_bids_validate):
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
    _bids_validate(output_path)

    raw = read_raw_bids(bids_basename + '_meg', output_path)

    with pytest.raises(TypeError, match="unexpected keyword argument 'foo'"):
        read_raw_bids(bids_basename + '_meg', output_path,
                      extra_params=dict(foo='bar'))

    # test anonymize
    raw = mne.io.read_raw_bti(raw_fname, config_fname=config_fname,
                              head_shape_fname=headshape_fname)
    with pytest.warns(UserWarning,
                      match='Converting to FIF for anonymization'):
        output_path = _test_anonymize(raw, bids_basename)
    _bids_validate(output_path)


# XXX: vhdr test currently passes only on MNE master. Skip until next release.
# see: https://github.com/mne-tools/mne-python/pull/6558
@pytest.mark.skipif(LooseVersion(mne.__version__) < LooseVersion('0.19'),
                    reason="requires mne 0.19.dev0 or higher")
def test_vhdr(_bids_validate):
    """Test write_raw_bids conversion for BrainVision data."""
    output_path = _TempDir()
    data_path = op.join(base_path, 'brainvision', 'tests', 'data')
    raw_fname = op.join(data_path, 'test.vhdr')

    raw = mne.io.read_raw_brainvision(raw_fname)

    # inject a bad channel
    assert not raw.info['bads']
    injected_bad = ['FP1']
    raw.info['bads'] = injected_bad

    # write with injected bad channels
    write_raw_bids(raw, bids_basename_minimal, output_path, overwrite=False)
    _bids_validate(output_path)

    # read and also get the bad channels
    raw = read_raw_bids(bids_basename_minimal + '_eeg.vhdr', output_path)

    with pytest.raises(TypeError, match="unexpected keyword argument 'foo'"):
        read_raw_bids(bids_basename_minimal + '_eeg.vhdr', output_path,
                      extra_params=dict(foo='bar'))

    # Check that injected bad channel shows up in raw after reading
    np.testing.assert_array_equal(np.asarray(raw.info['bads']),
                                  np.asarray(injected_bad))

    # Test that correct channel units are written ... and that bad channel
    # is in channels.tsv
    channels_tsv_name = op.join(output_path, 'sub-{}'.format(subject_id),
                                'eeg', bids_basename_minimal + '_channels.tsv')
    data = _from_tsv(channels_tsv_name)
    assert data['units'][data['name'].index('FP1')] == 'µV'
    assert data['units'][data['name'].index('CP5')] == 'n/a'
    assert data['status'][data['name'].index(injected_bad[0])] == 'bad'

    # check events.tsv is written
    events_tsv_fname = channels_tsv_name.replace('channels', 'events')
    assert op.exists(events_tsv_fname)

    # create another bids folder with the overwrite command and check
    # no files are in the folder
    data_path = make_bids_folders(subject=subject_id, kind='eeg',
                                  output_path=output_path, overwrite=True)
    assert len([f for f in os.listdir(data_path) if op.isfile(f)]) == 0

    # test anonymize and convert
    raw = mne.io.read_raw_brainvision(raw_fname)
    _test_anonymize(raw, bids_basename)

    # Also cover iEEG
    # We use the same data and pretend that eeg channels are ecog
    raw = mne.io.read_raw_brainvision(raw_fname)
    raw.set_channel_types({raw.ch_names[i]: 'ecog'
                           for i in mne.pick_types(raw.info, eeg=True)})
    output_path = _TempDir()
    write_raw_bids(raw, bids_basename, output_path, overwrite=False)
    _bids_validate(output_path)


def test_edf(_bids_validate):
    """Test write_raw_bids conversion for European Data Format data."""
    output_path = _TempDir()
    data_path = op.join(testing.data_path(), 'EDF')
    raw_fname = op.join(data_path, 'test_reduced.edf')

    raw = mne.io.read_raw_edf(raw_fname)

    raw.rename_channels({raw.info['ch_names'][0]: 'EOG'})
    raw.info['chs'][0]['coil_type'] = FIFF.FIFFV_COIL_EEG_BIPOLAR
    raw.rename_channels({raw.info['ch_names'][1]: 'EMG'})
    raw.set_channel_types({'EMG': 'emg'})

    write_raw_bids(raw, bids_basename, output_path)

    # Reading the file back should raise an error, because we renamed channels
    # in `raw` and used that information to write a channels.tsv. Yet, we
    # saved the unchanged `raw` in the BIDS folder, so channels in the TSV and
    # in raw clash
    with pytest.raises(RuntimeError, match='Channels do not correspond'):
        read_raw_bids(bids_basename + '_eeg.edf', output_path)

    with pytest.raises(TypeError, match="unexpected keyword argument 'foo'"):
        read_raw_bids(bids_basename + '_eeg.edf', output_path,
                      extra_params=dict(foo='bar'))

    bids_fname = bids_basename.replace('run-01', 'run-%s' % run2)
    write_raw_bids(raw, bids_fname, output_path, overwrite=True)
    _bids_validate(output_path)

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
    _bids_validate(output_path)

    # test anonymize and convert
    raw = mne.io.read_raw_edf(raw_fname)
    output_path = _test_anonymize(raw, bids_basename)
    _bids_validate(output_path)


def test_bdf(_bids_validate):
    """Test write_raw_bids conversion for Biosemi data."""
    output_path = _TempDir()
    data_path = op.join(base_path, 'edf', 'tests', 'data')
    raw_fname = op.join(data_path, 'test.bdf')

    raw = mne.io.read_raw_bdf(raw_fname)
    with pytest.warns(UserWarning, match='No line frequency found'):
        write_raw_bids(raw, bids_basename, output_path, overwrite=False)
    _bids_validate(output_path)

    # Test also the reading of channel types from channels.tsv
    # the first channel in the raw data is not MISC right now
    test_ch_idx = 0
    assert coil_type(raw.info, test_ch_idx) != 'misc'

    # we will change the channel type to MISC and overwrite the channels file
    bids_fname = bids_basename + '_eeg.bdf'
    channels_fname = _find_matching_sidecar(bids_fname, output_path,
                                            'channels.tsv')
    channels_dict = _from_tsv(channels_fname)
    channels_dict['type'][test_ch_idx] = 'MISC'
    _to_tsv(channels_dict, channels_fname)

    # Now read the raw data back from BIDS, with the tampered TSV, to show
    # that the channels.tsv truly influences how read_raw_bids sets ch_types
    # in the raw data object
    raw = read_raw_bids(bids_fname, output_path)
    assert coil_type(raw.info, test_ch_idx) == 'misc'

    with pytest.raises(TypeError, match="unexpected keyword argument 'foo'"):
        read_raw_bids(bids_fname, output_path, extra_params=dict(foo='bar'))

    # Test cropped assertion error
    raw = mne.io.read_raw_bdf(raw_fname)
    raw.crop(0, raw.times[-2])
    with pytest.raises(AssertionError, match='cropped'):
        write_raw_bids(raw, bids_basename, output_path)

    # test anonymize and convert
    raw = mne.io.read_raw_bdf(raw_fname)
    output_path = _test_anonymize(raw, bids_basename)
    _bids_validate(output_path)


def test_set(_bids_validate):
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
    read_raw_bids(bids_basename + '_eeg.set', output_path)

    with pytest.raises(TypeError, match="unexpected keyword argument 'foo'"):
        read_raw_bids(bids_basename + '_eeg.set', output_path,
                      extra_params=dict(foo='bar'))

    with pytest.raises(FileExistsError, match="already exists"):  # noqa: F821
        write_raw_bids(raw, bids_basename, output_path=output_path,
                       overwrite=False)
    _bids_validate(output_path)

    # check events.tsv is written
    # XXX: only from 0.18 onwards because events_from_annotations
    # is broken for earlier versions
    events_tsv_fname = op.join(output_path, 'sub-' + subject_id,
                               'ses-' + session_id, 'eeg',
                               bids_basename + '_events.tsv')
    if check_version('mne', '0.18'):
        assert op.exists(events_tsv_fname)

    # Also cover iEEG
    # We use the same data and pretend that eeg channels are ecog
    raw.set_channel_types({raw.ch_names[i]: 'ecog'
                           for i in mne.pick_types(raw.info, eeg=True)})
    output_path = _TempDir()
    write_raw_bids(raw, bids_basename, output_path)
    _bids_validate(output_path)

    # test anonymize and convert
    output_path = _test_anonymize(raw, bids_basename)
    _bids_validate(output_path)


@requires_nibabel()
def test_write_anat(_bids_validate):
    """Test writing anatomical data."""
    # Get the MNE testing sample data
    import nibabel as nib
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

    # Write some MRI data and supply a `trans`
    trans_fname = raw_fname.replace('_raw.fif', '-trans.fif')
    trans = mne.read_trans(trans_fname)

    # Get the T1 weighted MRI data file
    # Needs to be converted to Nifti because we only have mgh in our test base
    t1w_mgh = op.join(data_path, 'subjects', 'sample', 'mri', 'T1.mgz')

    anat_dir = write_anat(output_path, subject_id, t1w_mgh, session_id, acq,
                          raw=raw, trans=trans, deface=True, verbose=True,
                          overwrite=True)
    _bids_validate(output_path)

    # Validate that files are as expected
    t1w_json_path = op.join(anat_dir, 'sub-01_ses-01_acq-01_T1w.json')
    assert op.exists(t1w_json_path)
    assert op.exists(op.join(anat_dir, 'sub-01_ses-01_acq-01_T1w.nii.gz'))
    with open(t1w_json_path, 'r') as f:
        t1w_json = json.load(f)
    print(t1w_json)
    # We only should have AnatomicalLandmarkCoordinates as key
    np.testing.assert_array_equal(list(t1w_json.keys()),
                                  ['AnatomicalLandmarkCoordinates'])
    # And within AnatomicalLandmarkCoordinates only LPA, NAS, RPA in that order
    anat_dict = t1w_json['AnatomicalLandmarkCoordinates']
    point_list = ['LPA', 'NAS', 'RPA']
    np.testing.assert_array_equal(list(anat_dict.keys()),
                                  point_list)
    # test the actual values of the voxels (no floating points)
    for i, point in enumerate([(66, 51, 46), (41, 32, 74), (17, 53, 47)]):
        coords = anat_dict[point_list[i]]
        np.testing.assert_array_equal(np.asarray(coords, dtype=int),
                                      point)

    # BONUS: test also that we can find the matching sidecar
        side_fname = _find_matching_sidecar('sub-01_ses-01_acq-01_T1w.nii.gz',
                                            output_path, 'T1w.json')
        assert op.split(side_fname)[-1] == 'sub-01_ses-01_acq-01_T1w.json'

    # Now try some anat writing that will fail
    # We already have some MRI data there
    with pytest.raises(IOError, match='`overwrite` is set to False'):
        write_anat(output_path, subject_id, t1w_mgh, session_id, acq,
                   raw=raw, trans=trans, verbose=True, deface=False,
                   overwrite=False)

    # pass some invalid type as T1 MRI
    with pytest.raises(ValueError, match='must be a path to a T1 weighted'):
        write_anat(output_path, subject_id, 9999999999999, session_id, raw=raw,
                   trans=trans, verbose=True, deface=False, overwrite=True)

    # Return without writing sidecar
    sh.rmtree(anat_dir)
    write_anat(output_path, subject_id, t1w_mgh, session_id)
    # Assert that we truly cannot find a sidecar
    with pytest.raises(RuntimeError, match='Did not find any'):
        _find_matching_sidecar('sub-01_ses-01_acq-01_T1w.nii.gz',
                               output_path, 'T1w.json')

    # trans has a wrong type
    wrong_type = 1
    match = 'transform type {} not known, must be'.format(type(wrong_type))
    with pytest.raises(ValueError, match=match):
        write_anat(output_path, subject_id, t1w_mgh, session_id, raw=raw,
                   trans=wrong_type, verbose=True, deface=False,
                   overwrite=True)

    # trans is a str, but file does not exist
    wrong_fname = 'not_a_trans'
    match = 'trans file "{}" not found'.format(wrong_fname)
    with pytest.raises(IOError, match=match):
        write_anat(output_path, subject_id, t1w_mgh, session_id, raw=raw,
                   trans=wrong_fname, verbose=True, overwrite=True)

    # However, reading trans if it is a string pointing to trans is fine
    write_anat(output_path, subject_id, t1w_mgh, session_id, raw=raw,
               trans=trans_fname, verbose=True, deface=False,
               overwrite=True)

    # Writing without a session does NOT yield "ses-None" anywhere
    anat_dir2 = write_anat(output_path, subject_id, t1w_mgh, None)
    assert 'ses-None' not in anat_dir2
    assert op.exists(op.join(anat_dir2, 'sub-01_T1w.nii.gz'))

    # specify trans but not raw
    with pytest.raises(ValueError, match='must be specified if `trans`'):
        write_anat(output_path, subject_id, t1w_mgh, session_id, raw=None,
                   trans=trans, verbose=True, deface=False, overwrite=True)

    # test deface
    anat_dir = write_anat(output_path, subject_id, t1w_mgh,
                          session_id, raw=raw, trans=trans_fname,
                          verbose=True, deface=True, overwrite=True)
    t1w = nib.load(op.join(anat_dir, 'sub-01_ses-01_T1w.nii.gz'))
    vox_sum = t1w.get_data().sum()

    anat_dir2 = write_anat(output_path, subject_id, t1w_mgh,
                           session_id, raw=raw, trans=trans_fname,
                           verbose=True, deface=dict(inset=25.),
                           overwrite=True)
    t1w2 = nib.load(op.join(anat_dir2, 'sub-01_ses-01_T1w.nii.gz'))
    vox_sum2 = t1w2.get_data().sum()

    assert vox_sum > vox_sum2

    anat_dir3 = write_anat(output_path, subject_id, t1w_mgh,
                           session_id, raw=raw, trans=trans_fname,
                           verbose=True, deface=dict(theta=25),
                           overwrite=True)
    t1w3 = nib.load(op.join(anat_dir3, 'sub-01_ses-01_T1w.nii.gz'))
    vox_sum3 = t1w3.get_data().sum()

    assert vox_sum > vox_sum3

    with pytest.raises(ValueError,
                       match='The raw object, trans and raw or the landmarks'):
        write_anat(output_path, subject_id, t1w_mgh, session_id, raw=raw,
                   trans=None, verbose=True, deface=True,
                   overwrite=True)

    with pytest.raises(ValueError, match='inset must be numeric'):
        write_anat(output_path, subject_id, t1w_mgh, session_id, raw=raw,
                   trans=trans, verbose=True, deface=dict(inset='small'),
                   overwrite=True)

    with pytest.raises(ValueError, match='inset should be positive'):
        write_anat(output_path, subject_id, t1w_mgh, session_id, raw=raw,
                   trans=trans, verbose=True, deface=dict(inset=-2.),
                   overwrite=True)

    with pytest.raises(ValueError, match='theta must be numeric'):
        write_anat(output_path, subject_id, t1w_mgh, session_id, raw=raw,
                   trans=trans, verbose=True, deface=dict(theta='big'),
                   overwrite=True)

    with pytest.raises(ValueError,
                       match='theta should be between 0 and 90 degrees'):
        write_anat(output_path, subject_id, t1w_mgh, session_id, raw=raw,
                   trans=trans, verbose=True, deface=dict(theta=100),
                   overwrite=True)

    # Write some MRI data and supply `landmarks`
    mri_voxel_landmarks = mne.channels.make_dig_montage(
        lpa=[66.08580, 51.33362, 46.52982],
        nasion=[41.87363, 32.24694, 74.55314],
        rpa=[17.23812, 53.08294, 47.01789],
        coord_frame='mri_voxel')

    mri_landmarks = mne.channels.make_dig_montage(
        lpa=[-0.07629625, -0.00062556, -0.00776012],
        nasion=[0.00267222, 0.09362256, 0.03224791],
        rpa=[0.07635873, -0.00258065, -0.01212903],
        coord_frame='mri')

    meg_landmarks = mne.channels.make_dig_montage(
        lpa=[-7.13766068e-02, 0.00000000e+00, 5.12227416e-09],
        nasion=[3.72529030e-09, 1.02605611e-01, 4.19095159e-09],
        rpa=[7.52676800e-02, 0.00000000e+00, 5.58793545e-09],
        coord_frame='head')

    # test mri voxel landmarks
    anat_dir = write_anat(output_path, subject_id, t1w_mgh, session_id, acq,
                          deface=True, landmarks=mri_voxel_landmarks,
                          verbose=True, overwrite=True)
    _bids_validate(output_path)

    t1w1 = nib.load(op.join(anat_dir, 'sub-01_ses-01_acq-01_T1w.nii.gz'))
    vox1 = t1w1.get_data()

    # test mri landmarks
    anat_dir = write_anat(output_path, subject_id, t1w_mgh, session_id,
                          acq, deface=True,
                          landmarks=mri_landmarks, verbose=True,
                          overwrite=True)
    _bids_validate(output_path)

    t1w2 = nib.load(op.join(anat_dir, 'sub-01_ses-01_acq-01_T1w.nii.gz'))
    vox2 = t1w2.get_data()

    # because of significant rounding errors the voxels are fairly different
    # but the deface works in all three cases and was checked
    assert abs(vox1 - vox2).sum() / abs(vox1).sum() < 0.2

    # crash for raw also
    with pytest.raises(ValueError, match='Please use either `landmarks`'):
        anat_dir = write_anat(output_path, subject_id, t1w_mgh, session_id,
                              acq, raw=raw, trans=trans, deface=True,
                              landmarks=mri_landmarks, verbose=True,
                              overwrite=True)

    # crash for trans also
    with pytest.raises(ValueError, match='`trans` was provided'):
        anat_dir = write_anat(output_path, subject_id, t1w_mgh, session_id,
                              acq, trans=trans, deface=True,
                              landmarks=mri_landmarks, verbose=True,
                              overwrite=True)

    # test meg landmarks
    tmp_dir = _TempDir()
    meg_landmarks.save(op.join(tmp_dir, 'meg_landmarks.fif'))
    anat_dir = write_anat(output_path, subject_id, t1w_mgh, session_id,
                          acq, deface=True, trans=trans,
                          landmarks=op.join(tmp_dir, 'meg_landmarks.fif'),
                          verbose=True, overwrite=True)
    _bids_validate(output_path)

    t1w3 = nib.load(op.join(anat_dir, 'sub-01_ses-01_acq-01_T1w.nii.gz'))
    vox3 = t1w3.get_data()

    assert abs(vox1 - vox3).sum() / abs(vox1).sum() < 0.2

    # test raise error on meg_landmarks with no trans
    with pytest.raises(ValueError, match='Head space landmarks provided'):
        anat_dir = write_anat(output_path, subject_id, t1w_mgh, session_id,
                              acq, deface=True, landmarks=meg_landmarks,
                              verbose=True, overwrite=True)

    # test unsupported (any coord_frame other than head and mri) coord_frame
    fail_landmarks = meg_landmarks.copy()
    fail_landmarks.dig[0]['coord_frame'] = 3
    fail_landmarks.dig[1]['coord_frame'] = 3
    fail_landmarks.dig[2]['coord_frame'] = 3

    with pytest.raises(ValueError, match='Coordinate frame not recognized'):
        anat_dir = write_anat(output_path, subject_id, t1w_mgh, session_id,
                              acq, deface=True, landmarks=fail_landmarks,
                              verbose=True, overwrite=True)
