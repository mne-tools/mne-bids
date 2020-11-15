# Authors: Alex Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)


from pathlib import Path
import itertools

import pytest
import numpy as np

import mne
from mne.utils import requires_pandas
from mne.datasets import testing
from mne.utils import _TempDir

from mne_bids import BIDSPath, write_raw_bids
from mne_bids.stats import count_events


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
def test_count_events(subjects, tasks, runs, sessions):
    """Test the event counts."""
    data_path = testing.data_path()
    raw_fname = \
        Path(data_path) / 'MEG' / 'sample' / 'sample_audvis_trunc_raw.fif'
    raw = mne.io.read_raw(raw_fname)
    raw.info['line_freq'] = 60.
    root = _TempDir()
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

    counts = count_events(root)

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
