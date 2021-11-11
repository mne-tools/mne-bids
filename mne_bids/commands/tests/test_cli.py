"""Test command line."""
# Authors: Teon L Brooks <teon.brooks@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD-3-Clause
import os.path as op
from pathlib import Path
from functools import partial

import pytest

# This is here to handle mne-python <0.20
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings(action='ignore',
                            message="can't resolve package",
                            category=ImportWarning)
    import mne

from mne.datasets import testing
from mne.utils import ArgvSetter, requires_pandas, requires_version
from mne.utils._testing import requires_module

from mne_bids.commands import (mne_bids_raw_to_bids,
                               mne_bids_cp,
                               mne_bids_mark_channels,
                               mne_bids_calibration_to_bids,
                               mne_bids_crosstalk_to_bids,
                               mne_bids_count_events,
                               mne_bids_inspect)

from mne_bids import BIDSPath, read_raw_bids, write_raw_bids

requires_matplotlib = partial(requires_module, name='matplotlib',
                              call='import matplotlib')


base_path = op.join(op.dirname(mne.__file__), 'io')
subject_id = '01'
task = 'testing'
datatype = 'meg'


def check_usage(module, force_help=False):
    """Ensure we print usage."""
    args = ('--help',) if force_help else ()
    with ArgvSetter(args) as out:
        try:
            module.run()
        except SystemExit:
            pass
        assert 'Usage: ' in out.stdout.getvalue()


def test_raw_to_bids(tmp_path):
    """Test mne_bids raw_to_bids."""
    output_path = str(tmp_path)
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    # Check that help is printed
    check_usage(mne_bids_raw_to_bids)

    # Should work
    with ArgvSetter(('--subject_id', subject_id, '--task', task, '--raw',
                     raw_fname, '--bids_root', output_path,
                     '--line_freq', 60)):
        mne_bids_raw_to_bids.run()

    # Test line_freq == 'None'
    with ArgvSetter(('--subject_id', subject_id, '--task', task, '--raw',
                     raw_fname, '--bids_root', output_path,
                     '--line_freq', 'None', '--overwrite', 1)):
        mne_bids_raw_to_bids.run()

    # Test EDF files as well
    edf_data_path = op.join(base_path, 'edf', 'tests', 'data')
    edf_fname = op.join(edf_data_path, 'test.edf')
    with ArgvSetter(('--subject_id', subject_id, '--task', task, '--raw',
                     edf_fname, '--bids_root', output_path,
                     '--line_freq', 60)):
        mne_bids_raw_to_bids.run()

    # Too few input args
    with pytest.raises(SystemExit):
        with ArgvSetter(('--subject_id', subject_id)):
            mne_bids_cp.run()


def test_cp(tmp_path):
    """Test mne_bids cp."""
    output_path = str(tmp_path)
    data_path = op.join(base_path, 'brainvision', 'tests', 'data')
    raw_fname = op.join(data_path, 'test.vhdr')
    outname = op.join(output_path, 'test2.vhdr')

    # Check that help is printed
    check_usage(mne_bids_cp)

    # Should work
    with ArgvSetter(('--input', raw_fname, '--output', outname)):
        mne_bids_cp.run()

    # Too few input args
    with pytest.raises(SystemExit):
        with ArgvSetter(('--input', raw_fname)):
            mne_bids_cp.run()


def test_mark_bad_chanels_single_file(tmp_path):
    """Test mne_bids mark_channels."""

    # Check that help is printed
    check_usage(mne_bids_mark_channels)

    # Create test dataset.
    output_path = str(tmp_path)
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    old_bads = mne.io.read_raw_fif(raw_fname).info['bads']
    bids_path = BIDSPath(subject=subject_id, task=task, root=output_path,
                         datatype=datatype)

    with ArgvSetter(('--subject_id', subject_id, '--task', task,
                     '--raw', raw_fname, '--bids_root', output_path,
                     '--line_freq', 60)):
        mne_bids_raw_to_bids.run()

    # Update the dataset.
    ch_names = ['MEG 0112', 'MEG 0131']
    descriptions = ['Really bad!', 'Even worse.']

    args = ['--subject_id', subject_id, '--task', task,
            '--bids_root', output_path, '--type', datatype]
    for ch_name, description in zip(ch_names, descriptions):
        args.extend(['--ch_name', ch_name])
        args.extend(['--description', description])

    args = tuple(args)
    with ArgvSetter(args):
        mne_bids_mark_channels.run()

    # Check the data was properly written
    with pytest.warns(RuntimeWarning, match='The unit for chann*'):
        raw = read_raw_bids(bids_path=bids_path, verbose=False)
    assert set(old_bads + ch_names) == set(raw.info['bads'])

    # Test resetting bad channels.
    args = ('--subject_id', subject_id, '--task', task,
            '--bids_root', output_path, '--type', datatype,
            '--status', 'good', '--ch_name', '')
    with ArgvSetter(args):
        mne_bids_mark_channels.run()
    print('Finished running the reset...')

    # Check the data was properly written
    with pytest.warns(RuntimeWarning, match='The unit for chann*'):
        raw = read_raw_bids(bids_path=bids_path)
    assert raw.info['bads'] == []


def test_mark_bad_chanels_multiple_files(tmp_path):
    """Test mne_bids mark_channels."""

    # Check that help is printed
    check_usage(mne_bids_mark_channels)

    # Create test dataset.
    output_path = str(tmp_path)
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    old_bads = mne.io.read_raw_fif(raw_fname).info['bads']
    bids_path = BIDSPath(task=task, root=output_path, datatype=datatype)

    subjects = ['01', '02', '03']
    for subject in subjects:
        with ArgvSetter(('--subject_id', subject, '--task', task,
                         '--raw', raw_fname, '--bids_root', output_path,
                         '--line_freq', 60)):
            mne_bids_raw_to_bids.run()

    # Update the dataset.
    ch_names = ['MEG 0112', 'MEG 0131']
    descriptions = ['Really bad!', 'Even worse.']

    args = ['--task', task, '--bids_root', output_path, '--type', datatype]
    for ch_name, description in zip(ch_names, descriptions):
        args.extend(['--ch_name', ch_name])
        args.extend(['--description', description])

    args = tuple(args)
    with ArgvSetter(args):
        mne_bids_mark_channels.run()

    # Check the data was properly written
    for subject in subjects:
        with pytest.warns(RuntimeWarning, match='The unit for chann*'):
            raw = read_raw_bids(bids_path=bids_path.copy()
                                .update(subject=subject))
        assert set(old_bads + ch_names) == set(raw.info['bads'])


def test_calibration_to_bids(tmp_path):
    """Test mne_bids calibration_to_bids."""

    # Check that help is printed
    check_usage(mne_bids_calibration_to_bids)

    output_path = str(tmp_path)
    data_path = Path(testing.data_path())
    fine_cal_fname = data_path / 'SSS' / 'sss_cal_mgh.dat'
    bids_path = BIDSPath(subject=subject_id, root=output_path)

    # Write fine-calibration file and check that it was actually written.
    args = ('--file', fine_cal_fname, '--subject', subject_id,
            '--bids_root', output_path)
    with ArgvSetter(args):
        mne_bids_calibration_to_bids.run()

    assert bids_path.meg_calibration_fpath.exists()


def test_crosstalk_to_bids(tmp_path):
    """Test mne_bids crosstalk_to_bids."""

    # Check that help is printed
    check_usage(mne_bids_crosstalk_to_bids)

    output_path = str(tmp_path)
    data_path = Path(testing.data_path())
    crosstalk_fname = data_path / 'SSS' / 'ct_sparse.fif'
    bids_path = BIDSPath(subject=subject_id, root=output_path)

    # Write fine-calibration file and check that it was actually written.
    # Write fine-calibration file and check that it was actually written.
    args = ('--file', crosstalk_fname, '--subject', subject_id,
            '--bids_root', output_path)
    with ArgvSetter(args):
        mne_bids_crosstalk_to_bids.run()
    assert bids_path.meg_crosstalk_fpath.exists()


@requires_pandas
def test_count_events(tmp_path):
    """Test mne_bids count_events."""

    # Check that help is printed
    check_usage(mne_bids_count_events)

    # Create test dataset.
    output_path = str(tmp_path)
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')

    raw = mne.io.read_raw(raw_fname)
    raw.info['line_freq'] = 60.
    events = mne.find_events(raw)
    event_id = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
                'visual/right': 4, 'face': 5, 'button': 32}

    bids_path = BIDSPath(subject='01', root=output_path, task='foo')
    write_raw_bids(raw, bids_path, events, event_id, overwrite=True,
                   verbose=False)

    with ArgvSetter(('--bids_root', output_path)):
        mne_bids_count_events.run()

    with ArgvSetter(('--bids_root', output_path, '--describe')):
        mne_bids_count_events.run()

    with ArgvSetter(('--bids_root', output_path, '--silent')):
        mne_bids_count_events.run()

    with ArgvSetter(('--bids_root', output_path,
                     '--output', str(Path(output_path) / 'counts.csv'))):
        mne_bids_count_events.run()

    with ArgvSetter(
        ('--bids_root', output_path,
         '--output', str(Path(output_path) / 'counts.csv'),
         '--overwrite')
    ):
        mne_bids_count_events.run()


@requires_matplotlib
@requires_version('mne', '0.22')
def test_inspect(tmp_path):
    """Test mne_bids inspect."""

    # Check that help is printed
    check_usage(mne_bids_inspect)

    # Create test dataset.
    bids_root = str(tmp_path)
    data_path = testing.data_path()
    subject = '01'
    task = 'test'
    datatype = 'meg'
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')

    raw = mne.io.read_raw(raw_fname)
    raw.info['line_freq'] = 60.

    bids_path = BIDSPath(subject=subject, task=task, datatype=datatype,
                         root=bids_root)
    write_raw_bids(raw, bids_path, overwrite=True, verbose=False)

    import matplotlib
    matplotlib.use('agg')

    h_freqs = (30.0, 30, '30')
    for h_freq in h_freqs:
        args = ('--bids_root', bids_root, '--h_freq', h_freq,
                '--find_flat', 0)
        with ArgvSetter(args):
            mne_bids_inspect.run()
