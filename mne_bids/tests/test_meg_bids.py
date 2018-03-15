""" Testing MNE BIDS converter

There are four supported MEG systems in MNE.
We should test their functionality with their sample data.
"""
# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Teon L Brooks <teon.brooks@gmail.com>
# License: BSD (3-clause)

import os.path as op

import mne
from mne.datasets import testing
from mne.utils import _TempDir, run_subprocess

from mne_bids import raw_to_bids
from warnings import simplefilter
simplefilter('ignore', DeprecationWarning)

base_path = op.join(op.dirname(mne.__file__), 'io')
subject_id = '01'
session_id = '01'
run = '01'
task = 'testing'


def test_fif():
    """
    Testing the functionality of the raw_to_bids conversion for
    Neuromag data.
    """

    output_path = _TempDir()
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')

    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
                'Visual/Right': 4, 'Smiley': 5, 'Button': 32}
    events_fname = op.join(data_path, 'MEG', 'sample',
                           'sample_audvis_trunc_raw-eve.fif')

    raw_to_bids(subject_id=subject_id, session_id=session_id, run=run,
                task=task, raw_file=raw_fname, events_data=events_fname,
                output_path=output_path, event_id=event_id,
                overwrite=True)

    # let's do some modifications to meas_date
    raw = mne.io.read_raw_fif(raw_fname)
    raw.anonymize()
    data_path2 = _TempDir()
    raw_fname2 = op.join(data_path2, 'sample_audvis_raw.fif')
    raw.save(raw_fname2)

    raw_to_bids(subject_id=subject_id, run=run, task=task,
                session_id=session_id, raw_file=raw_fname2,
                events_data=events_fname, output_path=output_path,
                event_id=event_id, overwrite=True)

    cmd = ['bids-validator', output_path]
    run_subprocess(cmd)


def test_kit():
    """
    Testing the functionality of the raw_to_bids conversion for KIT data.
    """
    output_path = _TempDir()
    data_path = op.join(base_path, 'kit', 'tests', 'data')
    raw_fname = op.join(data_path, 'test.sqd')
    events_fname = op.join(data_path, 'test-eve.txt')
    hpi_fname = op.join(data_path, 'test_mrk.sqd')
    electrode_fname = op.join(data_path, 'test_elp.txt')
    headshape_fname = op.join(data_path, 'test_hsp.txt')
    event_id = dict(cond=1)

    raw_to_bids(subject_id=subject_id, session_id=session_id, run=run,
                task=task, raw_file=raw_fname, events_data=events_fname,
                event_id=event_id, hpi=hpi_fname, electrode=electrode_fname,
                hsp=headshape_fname, output_path=output_path,
                overwrite=True)
    cmd = ['bids-validator', output_path]
    run_subprocess(cmd)


def test_ctf():
    """
    Testing the functionality of the raw_to_bids conversion for CTF data.
    """
    output_path = _TempDir()
    data_path = op.join(testing.data_path(download=False), 'CTF')
    raw_fname = op.join(data_path, 'testdata_ctf.ds')

    raw_to_bids(subject_id=subject_id, session_id=session_id, run=run,
                task=task, raw_file=raw_fname, output_path=output_path,
                overwrite=True)
    cmd = ['bids-validator', output_path]
    run_subprocess(cmd)


def test_bti():
    """
    Testing the functionality of the raw_to_bids conversion for BTi data.
    """
    output_path = _TempDir()
    data_path = op.join(base_path, 'bti', 'tests', 'data')
    raw_fname = op.join(data_path, 'test_pdf_linux')
    config_fname = op.join(data_path, 'test_config_linux')
    headshape_fname = op.join(data_path, 'test_hs_linux')

    raw_to_bids(subject_id=subject_id, session_id=session_id, run=run,
                task=task, raw_file=raw_fname, config=config_fname,
                hsp=headshape_fname, output_path=output_path,
                verbose=True, overwrite=True)

    cmd = ['bids-validator', output_path]
    run_subprocess(cmd)
