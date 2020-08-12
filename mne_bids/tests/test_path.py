"""Test for the MNE BIDS path functions."""
# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)
import os
import os.path as op
# This is here to handle mne-python <0.20
import warnings
from pathlib import Path
from shutil import copyfile

import pytest

with warnings.catch_warnings():
    warnings.filterwarnings(action='ignore',
                            message="can't resolve package",
                            category=ImportWarning)
    import mne

from mne.datasets import testing
from mne.utils import _TempDir

from mne_bids import (get_kinds, get_entity_vals, print_dir_tree,
                      make_bids_folders, make_bids_basename,
                      write_raw_bids)
from mne_bids.path import (_parse_ext, parse_bids_filename,
                           _find_best_candidates, _find_matching_sidecar)

subject_id = '01'
session_id = '01'
run = '01'
acq = None
task = 'testing'

bids_basename = make_bids_basename(
    subject=subject_id, session=session_id, run=run, acquisition=acq,
    task=task)


@pytest.fixture(scope='session')
def return_bids_test_dir(tmpdir_factory):
    """Return path to a written test BIDS dir."""
    bids_root = str(tmpdir_factory.mktemp('mnebids_utils_test_bids_ds'))
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')

    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
                'Visual/Right': 4, 'Smiley': 5, 'Button': 32}
    events_fname = op.join(data_path, 'MEG', 'sample',
                           'sample_audvis_trunc_raw-eve.fif')

    raw = mne.io.read_raw_fif(raw_fname)
    # Write multiple runs for test_purposes
    for run_idx in [run, '02']:
        name = bids_basename.copy().update(run=run_idx)
        with pytest.warns(RuntimeWarning, match='No line frequency'):
            write_raw_bids(raw, name, bids_root,
                           events_data=events_fname, event_id=event_id,
                           overwrite=True)
    return bids_root


def test_get_keys(return_bids_test_dir):
    """Test getting the datatypes (=kinds) of a dir."""
    kinds = get_kinds(return_bids_test_dir)
    assert kinds == ['meg']


@pytest.mark.parametrize('entity, expected_vals, kwargs',
                         [('bogus', None, None),
                          ('subject', [subject_id], None),
                          ('session', [session_id], None),
                          ('run', [run, '02'], None),
                          ('acquisition', [], None),
                          ('task', [task], None),
                          ('subject', [], dict(ignore_subjects=[subject_id])),
                          ('subject', [], dict(ignore_subjects=subject_id)),
                          ('session', [], dict(ignore_sessions=[session_id])),
                          ('session', [], dict(ignore_sessions=session_id)),
                          ('run', [run], dict(ignore_runs=['02'])),
                          ('run', [run], dict(ignore_runs='02')),
                          ('task', [], dict(ignore_tasks=[task])),
                          ('task', [], dict(ignore_tasks=task)),
                          ('run', [run, '02'], dict(ignore_runs=['bogus']))])
def test_get_entity_vals(entity, expected_vals, kwargs, return_bids_test_dir):
    """Test getting a list of entities."""
    bids_root = return_bids_test_dir
    if kwargs is None:
        kwargs = dict()

    if entity == 'bogus':
        with pytest.raises(ValueError, match='`key` must be one of'):
            get_entity_vals(bids_root=bids_root, entity_key=entity, **kwargs)
    else:
        vals = get_entity_vals(bids_root=bids_root, entity_key=entity,
                               **kwargs)
        assert vals == expected_vals


def test_print_dir_tree(capsys):
    """Test printing a dir tree."""
    with pytest.raises(ValueError, match='Directory does not exist'):
        print_dir_tree('i_dont_exist')

    # We check the testing directory
    test_dir = op.dirname(__file__)
    with pytest.raises(ValueError, match='must be a positive integer'):
        print_dir_tree(test_dir, max_depth=-1)
    with pytest.raises(ValueError, match='must be a positive integer'):
        print_dir_tree(test_dir, max_depth='bad')

    # Do not limit depth
    print_dir_tree(test_dir)
    captured = capsys.readouterr()
    assert '|--- test_utils.py' in captured.out.split('\n')
    assert '|--- __pycache__{}'.format(os.sep) in captured.out.split('\n')
    assert '.pyc' in captured.out

    # Now limit depth ... we should not descend into pycache
    print_dir_tree(test_dir, max_depth=1)
    captured = capsys.readouterr()
    assert '|--- test_utils.py' in captured.out.split('\n')
    assert '|--- __pycache__{}'.format(os.sep) in captured.out.split('\n')
    assert '.pyc' not in captured.out

    # Limit depth even more
    print_dir_tree(test_dir, max_depth=0)
    captured = capsys.readouterr()
    assert captured.out == '|tests{}\n'.format(os.sep)


def test_make_folders():
    """Test that folders are created and named properly."""
    # Make sure folders are created properly
    bids_root = _TempDir()
    make_bids_folders(subject='hi', session='foo', kind='ba',
                      bids_root=bids_root)
    assert op.isdir(op.join(bids_root, 'sub-hi', 'ses-foo', 'ba'))

    # If we remove a kwarg the folder shouldn't be created
    bids_root = _TempDir()
    make_bids_folders(subject='hi', kind='ba', bids_root=bids_root)
    assert op.isdir(op.join(bids_root, 'sub-hi', 'ba'))

    # check overwriting of folders
    make_bids_folders(subject='hi', kind='ba', bids_root=bids_root,
                      overwrite=True, verbose=True)

    # Check if a pathlib.Path bids_root works.
    bids_root = Path(_TempDir())
    make_bids_folders(subject='hi', session='foo', kind='ba',
                      bids_root=bids_root)
    assert op.isdir(op.join(bids_root, 'sub-hi', 'ses-foo', 'ba'))

    # Check if bids_root=None creates folders in the current working directory
    bids_root = _TempDir()
    curr_dir = os.getcwd()
    os.chdir(bids_root)
    make_bids_folders(subject='hi', session='foo', kind='ba',
                      bids_root=None)
    assert op.isdir(op.join(os.getcwd(), 'sub-hi', 'ses-foo', 'ba'))
    os.chdir(curr_dir)


def test_parse_ext():
    """Test the file extension extraction."""
    f = 'sub-05_task-matchingpennies.vhdr'
    fname, ext = _parse_ext(f)
    assert fname == 'sub-05_task-matchingpennies'
    assert ext == '.vhdr'

    # Test for case where no extension: assume BTI format
    f = 'sub-01_task-rest'
    fname, ext = _parse_ext(f)
    assert fname == f
    assert ext == '.pdf'

    # Get a .nii.gz file
    f = 'sub-01_task-rest.nii.gz'
    fname, ext = _parse_ext(f)
    assert fname == 'sub-01_task-rest'
    assert ext == '.nii.gz'


@pytest.mark.parametrize('fname', [
    'sub-01_ses-02_task-test_run-3_split-01_meg.fif',
    'sub-01_ses-02_task-test_run-3_split-01.fif',
    'sub-01_ses-02_task-test_run-3_split-01',
    ('/bids_root/sub-01/ses-02/meg/' +
     'sub-01_ses-02_task-test_run-3_split-01_meg.fif'),
])
def test_parse_bids_filename(fname):
    """Test parsing entities from a bids filename."""
    params = parse_bids_filename(fname)
    assert params['sub'] == '01'
    assert params['ses'] == '02'
    assert params['run'] == '3'
    assert params['task'] == 'test'
    assert params['split'] == '01'
    assert list(params.keys()) == ['sub', 'ses', 'task', 'acq', 'run', 'proc',
                                   'space', 'rec', 'split', 'kind']


@pytest.mark.parametrize('candidate_list, best_candidates', [
    # Only one candidate
    (['sub-01_ses-02'], ['sub-01_ses-02']),

    # Two candidates, but the second matches on more entities
    (['sub-01', 'sub-01_ses-02'], ['sub-01_ses-02']),

    # No candidates match
    (['sub-02_ses-02', 'sub-01_ses-01'], []),

    # First candidate is disqualified (session doesn't match)
    (['sub-01_ses-01', 'sub-01_ses-02'], ['sub-01_ses-02']),

    # Multiple equally good candidates
    (['sub-01_run-01', 'sub-01_run-02'], ['sub-01_run-01', 'sub-01_run-02']),
])
def test_find_best_candidates(candidate_list, best_candidates):
    """Test matching of candidate sidecar files."""
    params = dict(sub='01', ses='02', acq=None)
    assert _find_best_candidates(params, candidate_list) == best_candidates


def test_find_matching_sidecar(return_bids_test_dir):
    """Test finding a sidecar file from a BIDS dir."""
    bids_root = return_bids_test_dir

    bids_fpath = bids_basename.copy().update(bids_root=bids_root)
    # Now find a sidecar
    sidecar_fname = _find_matching_sidecar(bids_fpath,
                                           'coordsystem.json')
    expected_file = op.join('sub-01', 'ses-01', 'meg',
                            'sub-01_ses-01_coordsystem.json')
    assert sidecar_fname.endswith(expected_file)

    # Find multiple sidecars, tied in score, triggering an error
    with pytest.raises(RuntimeError, match='Expected to find a single'):
        open(sidecar_fname.replace('coordsystem.json',
                                   '2coordsystem.json'), 'w').close()
        _find_matching_sidecar(bids_fpath, 'coordsystem.json')

    # Find nothing but receive None, because we set `allow_fail` to True
    with pytest.warns(RuntimeWarning, match='Did not find any'):
        _find_matching_sidecar(bids_fpath, 'foo.bogus', True)


def test_bids_path(return_bids_test_dir):
    """Test usage of BIDSPath object."""
    bids_root = return_bids_test_dir
    bids_basename = make_bids_basename(
        subject=subject_id, session=session_id, run=run,
        task=task)

    # test bids path without bids_root, kind, extension
    # basename and fpath should be the same
    expected_basename = \
        f'sub-{subject_id}_ses-{session_id}_task-{task}_run-{run}'
    assert (bids_basename.basename == expected_basename)
    assert (bids_basename.fpath == expected_basename)

    # without bids_root and with kind/extension
    # basename and fpath should be the same
    bids_basename.update(kind='ieeg', extension='vhdr')
    expected_basename2 = expected_basename + '_ieeg.vhdr'
    assert (bids_basename.basename == expected_basename2)
    bids_basename.update(extension='.vhdr')
    assert (bids_basename.basename == expected_basename2)

    with pytest.warns(RuntimeWarning,
                      match='No bids root was passed in'):
        assert (bids_basename.fpath == expected_basename2)

    # with bids_root, but without kind/extension
    # basename should work, but fpath should not.
    bids_basename.update(bids_root=bids_root, kind=None, extension=None)
    assert bids_basename.basename == expected_basename

    # get_bids_fname should fire error when no ``kind`` is passed
    with pytest.raises(RuntimeError, match='No kind was provided'):
        bids_fpath = bids_basename.fpath

    # should find the correct filename if kind was passed
    bids_basename.update(kind='meg')
    bids_fpath = bids_basename.fpath
    assert op.basename(bids_fpath) == \
           bids_basename.update(extension='.fif').basename

    # temporarily create another copy of the dataset
    # with another extension and it will cause the
    # inference of the correct fpath to be incorrect
    bids_basename.update(extension=None)
    wrong_fpath = op.join(bids_root, f'sub-{subject_id}',
                          f'ses-{session_id}', 'meg',
                          bids_basename.basename + '.sqd')
    copyfile(bids_fpath, wrong_fpath)
    with pytest.raises(RuntimeError, match='Found more than one '
                                           'matching data file'):
        bids_fpath = bids_basename.fpath
        os.remove(wrong_fpath)

    # confirm BIDSPath assigns properties correctly
    bids_basename = make_bids_basename(subject=subject_id, session=session_id)
    assert bids_basename.subject == subject_id
    assert bids_basename.session == session_id
    assert 'subject' in bids_basename.entities
    assert 'session' in bids_basename.entities
    assert all(bids_basename.entities.get(entity) is None
               for entity in ['task', 'run', 'recording', 'acquisition',
                              'space', 'processing',
                              'bids_root', 'kind', 'extension'])

    # test updating functionality
    bids_basename.update(acquisition='03', run='2', session='02',
                         task=None)
    assert bids_basename.subject == subject_id
    assert bids_basename.session == '02'
    assert bids_basename.acquisition == '03'
    assert bids_basename.run == '2'
    assert bids_basename.task is None

    new_bids_basename = bids_basename.copy().update(task='02',
                                                    acquisition=None)
    assert new_bids_basename.task == '02'
    assert new_bids_basename.acquisition is None

    # equality of bids basename
    assert new_bids_basename != bids_basename
    assert new_bids_basename == bids_basename.copy().update(task='02',
                                                            acquisition=None)

    # error check on kwargs of update
    with pytest.raises(ValueError, match='Key must be one of*'):
        bids_basename.update(sub=subject_id, session=session_id)

    # error check on the passed in entity containing a magic char
    with pytest.raises(ValueError, match='Unallowed*'):
        bids_basename.update(subject=subject_id + '-')

    # error check on kind in update
    kind = 'meeg'
    with pytest.raises(ValueError, match=f'Kind {kind} is not'):
        bids_basename.update(kind=kind)

    # error check on extension in update
    ext = '.mat'
    with pytest.raises(ValueError, match=f'Extension {ext} is not'):
        bids_basename.update(extension=ext)

    # test repr
    bids_path = make_bids_basename(subject='01', session='02',
                                   task='03', suffix='ieeg.edf')
    assert repr(bids_path) == ('BIDSPath(\n'
                               'bids_root: None\n'
                               'basename: sub-01_ses-02_task-03_ieeg.edf)')
