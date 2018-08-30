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
from warnings import simplefilter, catch_warnings

import mne
from mne.datasets import testing
from mne.utils import _TempDir, run_subprocess

from mne_bids import raw_to_bids

simplefilter('ignore', DeprecationWarning)

base_path = op.join(op.dirname(mne.__file__), 'io')
subject_id = '01'
subject_id2 = '02'
session_id = '01'
run = '01'
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
                task=task, raw_file=raw_fname, events_data=events_fname,
                output_path=output_path, event_id=event_id, overwrite=True)

    # give the raw object some fake participant data
    raw = mne.io.read_raw_fif(raw_fname)
    #raw.anonymize()
    raw.info['subject_info'] = {'his_id': subject_id,
                                'birthday': (1994, 1, 26), 'sex': 1}
    data_path2 = _TempDir()
    raw_fname2 = op.join(data_path2, 'sample_audvis_raw.fif')
    raw.save(raw_fname2)

    with catch_warnings(record=True) as w:
        raw_to_bids(subject_id=subject_id2, run=run, task=task,
                    session_id=session_id, raw_file=raw_fname2,
                    events_data=events_fname, output_path=output_path,
                    event_id=event_id, overwrite=True)
        cmd = ['bids-validator', output_path]
        run_subprocess(cmd, shell=shell)

        participant_warning = [pw for pw in w
                               if 'participant' in str(pw.message)]
        if len(participant_warning) == 0:
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

    with catch_warnings(record=True) as w:
        raw_to_bids(subject_id=subject_id, session_id=session_id, run=run,
                    task=task, raw_file=raw_fname, events_data=events_fname,
                    event_id=event_id, hpi=hpi_fname,
                    electrode=electrode_fname, hsp=headshape_fname,
                    output_path=output_path, overwrite=True)
        cmd = ['bids-validator', output_path]
        run_subprocess(cmd, shell=shell)

        participant_warning = [pw for pw in w
                               if 'participant' in str(pw.message)]
        if len(participant_warning) == 0:
            assert op.exists(op.join(output_path, 'participants.tsv'))


def test_ctf():
    """Test functionality of the raw_to_bids conversion for CTF data."""
    output_path = _TempDir()
    data_path = op.join(testing.data_path(download=False), 'CTF')
    raw_fname = op.join(data_path, 'testdata_ctf.ds')

    with catch_warnings(record=True) as w:
        raw_to_bids(subject_id=subject_id, session_id=session_id, run=run,
                    task=task, raw_file=raw_fname, output_path=output_path,
                    overwrite=True)
        cmd = ['bids-validator', output_path]
        run_subprocess(cmd, shell=shell)

        participant_warning = [pw for pw in w
                               if 'participant' in str(pw.message)]
        if len(participant_warning) == 0:
            assert op.exists(op.join(output_path, 'participants.tsv'))


def test_bti():
    """Test functionality of the raw_to_bids conversion for BTi data."""
    output_path = _TempDir()
    data_path = op.join(base_path, 'bti', 'tests', 'data')
    raw_fname = op.join(data_path, 'test_pdf_linux')
    config_fname = op.join(data_path, 'test_config_linux')
    headshape_fname = op.join(data_path, 'test_hs_linux')

    with catch_warnings(record=True) as w:
        raw_to_bids(subject_id=subject_id, session_id=session_id, run=run,
                    task=task, raw_file=raw_fname, config=config_fname,
                    hsp=headshape_fname, output_path=output_path,
                    verbose=True, overwrite=True)
        cmd = ['bids-validator', output_path]
        run_subprocess(cmd, shell=shell)

        participant_warning = [pw for pw in w
                               if 'participant' in str(pw.message)]
        if len(participant_warning) == 0:
            assert op.exists(op.join(output_path, 'participants.tsv'))
