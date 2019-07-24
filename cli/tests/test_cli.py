"""Test command line."""
# Authors: Teon L Brooks <teon.brooks@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
from os import path as op

import pytest
from mne.datasets import testing
from mne.utils import _TempDir, run_tests_if_main, ArgvSetter

from mne_bids.datasets import fetch_brainvision_testing_data

from cli import mne_bids_raw_to_bids, mne_bids_cp


subject_id = '01'
task = 'testing'


def check_usage(module, force_help=False):
    """Ensure we print usage."""
    args = ('--help',) if force_help else ()
    with ArgvSetter(args) as out:
        try:
            module.run()
        except SystemExit:
            pass
        assert 'Usage: ' in out.stdout.getvalue()


def test_raw_to_bids():
    """Test mne_bids raw_to_bids."""
    output_path = _TempDir()
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')
    # Check that help is printed
    check_usage(mne_bids_raw_to_bids)

    # Should work
    with ArgvSetter(('--subject_id', subject_id, '--task', task, '--raw',
                     raw_fname, '--output_path', output_path)):
        mne_bids_raw_to_bids.run()

    # Too few input args
    with pytest.raises(SystemExit):
        with ArgvSetter(('--subject_id', subject_id)):
            mne_bids_cp.run()


def test_cp():
    """Test mne_bids cp."""
    output_path = _TempDir()
    data_path = fetch_brainvision_testing_data()
    raw_fname = op.join(data_path, 'test.vhdr')
    outname = op.join(output_path, 'test2.vhdr')

    # Check that help is printed
    check_usage(mne_bids_cp)

    # should work
    with ArgvSetter(('--input', raw_fname, '--output', outname)):
        mne_bids_cp.run()

    # too few inputs
    with pytest.raises(SystemExit):
        with ArgvSetter(('--input', raw_fname)):
            mne_bids_cp.run()


run_tests_if_main()
