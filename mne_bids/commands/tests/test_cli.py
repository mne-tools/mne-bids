"""Test command line."""
# Authors: Teon L Brooks <teon.brooks@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
from os import path as op

import pytest

# This is here to handle mne-python <0.20
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings(action='ignore',
                            message="can't resolve package",
                            category=ImportWarning)
    import mne

from mne.datasets import testing
from mne.utils import run_tests_if_main, ArgvSetter

from mne_bids.commands import (mne_bids_raw_to_bids, mne_bids_cp,
                               mne_bids_mark_bad_channels)


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


def test_raw_to_bids(tmpdir):
    """Test mne_bids raw_to_bids."""
    output_path = str(tmpdir)
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


def test_cp(tmpdir):
    """Test mne_bids cp."""
    output_path = str(tmpdir)
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


def test_mark_bad_chanels(tmpdir):
    """Test mne_bids mark_bad_channels."""

    # Check that help is printed
    check_usage(mne_bids_mark_bad_channels)

    # Create test dataset.
    output_path = str(tmpdir)
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')

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
        with pytest.warns(RuntimeWarning, match='The unit for chann*'):
            mne_bids_mark_bad_channels.run()

    # Test resettig bad channels.
    args = ('--subject_id', subject_id, '--task', task,
            '--bids_root', output_path, '--type', datatype,
            '--ch_name', '', '--overwrite')
    with ArgvSetter(args):
        mne_bids_mark_bad_channels.run()


run_tests_if_main()
