# -*- coding: utf-8 -*-
from os import path as op

from mne.datasets import testing
from mne.utils import _TempDir, run_tests_if_main, ArgvSetter
from mne_bids.commands import mne_bids_raw_to_bids


subject_id = '01'
task = 'testing'
kind = 'meg'


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

    # check_usage(mne_bids_raw_to_bids)
    with ArgvSetter(('--subject_id', subject_id, '--task', task, '--raw_file',
                     raw_fname, '--kind', kind, '--output_path', output_path)):
        mne_bids_raw_to_bids.run()


run_tests_if_main()
