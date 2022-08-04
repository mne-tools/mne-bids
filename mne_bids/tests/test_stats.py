"""Testing stats reporting."""
# Authors: Alex Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause


from pathlib import Path
import itertools

import pytest
import numpy as np

import mne
from mne.utils import requires_pandas
from mne.datasets import testing

from mne_bids import BIDSPath, write_raw_bids
from mne_bids.stats import count_events
from mne_bids.read import _from_tsv
from mne_bids.write import _write_tsv


def _make_dataset(root, subjects, tasks=(None,), runs=(None,),
                  sessions=(None,)):
    data_path = testing.data_path()
    raw_fname = \
        Path(data_path) / 'MEG' / 'sample' / 'sample_audvis_trunc_raw.fif'
    raw = mne.io.read_raw(raw_fname)
    raw.info['line_freq'] = 60.
    events = mne.find_events(raw)
    event_id = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
                'visual/right': 4, 'face': 5, 'button': 32}

    for subject, session, task, run in \
            itertools.product(subjects, sessions, tasks, runs):
        bids_path = BIDSPath(
            subject=subject, session=session, run=run, task=task, root=root,
        )
        write_raw_bids(raw, bids_path, events, event_id, overwrite=True,
                       verbose=False)

    return root, events, event_id


def _check_counts(counts, events, event_id, subjects,
                  tasks=(None,), runs=(None,), sessions=(None,)):
    if (sessions[0] is None) and (runs[0] is None):
        assert np.all(counts.index == subjects)
    else:
        assert np.all(counts.index.levels[0] == subjects)

    if (sessions[0] is not None) and (runs[0] is not None):
        assert np.all(counts.index.levels[1] == sessions)
        assert np.all(counts.index.levels[2] == runs)
    elif sessions[0] is not None:
        assert np.all(counts.index.levels[1] == sessions)
    elif runs[0] is not None:
        assert np.all(counts.index.levels[1] == runs)

    assert np.all(counts.columns.levels[0] == tasks)
    assert sorted(counts.columns.levels[1]) == sorted(event_id.keys())
    for k, v in event_id.items():
        key = (subjects[0],)
        if sessions[0] is not None:
            key += (sessions[0],)
        if runs[0] is not None:
            key += (runs[0],)
        key = key if len(key) > 1 else key[0]

        assert (
            counts.at[key, (tasks[0], k)] ==
            (events[:, 2] == v).sum()
        )


@pytest.mark.parametrize(
    ('subjects', 'tasks', 'runs', 'sessions'),
    [
        (['01'], ['task1'], ['01'], ['01']),
        (['01', '02'], ['task1'], ['01'], ['01']),
        (['01', '02'], ['task1', 'task2'], ['01'], ['01']),
        (['01'], ['task1', 'task2'], [None], ['01']),
        (['01'], ['task1', 'task2'], ['01'], [None]),
        (['01'], ['task1', 'task2'], [None], [None]),
    ]
)
@requires_pandas
def test_count_events(tmp_path, subjects, tasks, runs, sessions):
    """Test the event counts."""
    root, events, event_id = _make_dataset(tmp_path, subjects, tasks, runs,
                                           sessions)

    counts = count_events(root)

    _check_counts(counts, events, event_id, subjects, tasks, runs, sessions)


@requires_pandas
def test_count_events_bids_path(tmp_path):
    """Test the event counts passing a BIDSPath."""
    root, events, event_id = \
        _make_dataset(tmp_path, subjects=['01', '02'], tasks=['task1'])

    with pytest.raises(ValueError, match='datatype .*anat.* is not supported'):
        bids_path = BIDSPath(root=root, subject='01', datatype='anat')
        count_events(bids_path)

    bids_path = BIDSPath(root=root, subject='01', datatype='meg')
    counts = count_events(bids_path)

    _check_counts(counts, events, event_id, subjects=['01'], tasks=['task1'])


@requires_pandas
def test_count_no_events_file(tmp_path):
    """Test count_events with no event present."""
    data_path = testing.data_path()
    raw_fname = \
        Path(data_path) / 'MEG' / 'sample' / 'sample_audvis_trunc_raw.fif'
    raw = mne.io.read_raw(raw_fname)
    raw.info['line_freq'] = 60.
    root = str(tmp_path)

    bids_path = BIDSPath(
        subject='01', task='task1', root=root,
    )
    write_raw_bids(raw, bids_path, overwrite=True, verbose=False)

    with pytest.raises(ValueError, match='No events files found.'):
        count_events(root)


@requires_pandas
def test_count_no_events_column(tmp_path):
    """Test case where events.tsv doesn't contain [stim,trial]_type column."""
    subject, task, run, session, datatype = '01', 'task1', '01', '01', 'meg'
    root, events, event_id = _make_dataset(tmp_path, [subject], [task], [run],
                                           [session])

    # Delete the `stim_type` column.
    events_tsv_fpath = BIDSPath(root=root, subject=subject, task=task, run=run,
                                session=session, datatype=datatype,
                                suffix='events', extension='.tsv').fpath
    events_tsv = _from_tsv(events_tsv_fpath)
    events_tsv['stim_type'] = events_tsv['trial_type']
    del events_tsv['trial_type']
    _write_tsv(fname=events_tsv_fpath, dictionary=events_tsv, overwrite=True)

    counts = count_events(root)
    _check_counts(counts, events, event_id, [subject], [task], [run],
                  [session])
