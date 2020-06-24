"""Testing utilities for the MNE BIDS converter."""
# Authors: Chris Holdgraf <choldgraf@berkeley.edu>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
import os
import os.path as op
import pytest
from datetime import datetime
from pathlib import Path
import platform

from numpy.random import random

# This is here to handle mne-python <0.20
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings(action='ignore',
                            message="can't resolve package",
                            category=ImportWarning)
    import mne

from mne.utils import _TempDir, run_subprocess
from mne.datasets import testing

from mne_bids import make_bids_folders, make_bids_basename, write_raw_bids
from mne_bids.utils import (_check_types, print_dir_tree, _age_on_date,
                            _infer_eeg_placement_scheme, _handle_kind,
                            _find_matching_sidecar, _parse_ext,
                            _get_ch_type_mapping, _parse_bids_filename,
                            _find_best_candidates, get_entity_vals,
                            _path_to_str, get_kinds, delete_scan)
from mne_bids.tsv_handler import _from_tsv

base_path = op.join(op.dirname(mne.__file__), 'io')
subject_id = '01'
session_id = '01'
run = '01'
acq = None
task = 'testing'

bids_basename = make_bids_basename(
    subject=subject_id, session=session_id, run=run, acquisition=acq,
    task=task)


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
    vadlidator_args = ['--config.error=41']
    exe = os.getenv('VALIDATOR_EXECUTABLE', 'bids-validator')

    if platform.system() == 'Windows':
        shell = True
    else:
        shell = False

    bids_validator_exe = [exe, *vadlidator_args]

    def _validate(bids_root):
        cmd = [*bids_validator_exe, bids_root]
        run_subprocess(cmd, shell=shell)

    return _validate


@pytest.fixture(scope='function')
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
                          ('sub', [subject_id], None),
                          ('ses', [session_id], None),
                          ('run', [run, '02'], None),
                          ('acq', [], None),
                          ('task', [task], None),
                          ('sub', [], dict(ignore_sub=[subject_id])),
                          ('sub', [], dict(ignore_sub=subject_id)),
                          ('ses', [], dict(ignore_ses=[session_id])),
                          ('ses', [], dict(ignore_ses=session_id)),
                          ('run', [run], dict(ignore_run=['02'])),
                          ('run', [run], dict(ignore_run='02')),
                          ('task', [], dict(ignore_task=[task])),
                          ('task', [], dict(ignore_task=task)),
                          ('run', [run, '02'], dict(ignore_run=['bogus']))])
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


def test_get_ch_type_mapping():
    """Test getting a correct channel mapping."""
    with pytest.raises(ValueError, match='specified from "bogus" to "mne"'):
        _get_ch_type_mapping(fro='bogus', to='mne')


def test_handle_kind():
    """Test the automatic extraction of kind from the data."""
    # Create a dummy raw
    n_channels = 1
    sampling_rate = 100
    data = random((n_channels, sampling_rate))
    channel_types = ['grad', 'eeg', 'ecog']
    expected_kinds = ['meg', 'eeg', 'ieeg']
    # do it once for each type ... and once for "no type"
    for chtype, kind in zip(channel_types, expected_kinds):
        info = mne.create_info(n_channels, sampling_rate, ch_types=[chtype])
        raw = mne.io.RawArray(data, info)
        assert _handle_kind(raw) == kind

    # if the situation is ambiguous (EEG and iEEG channels both), raise error
    with pytest.raises(ValueError, match='Both EEG and iEEG channels found'):
        info = mne.create_info(2, sampling_rate,
                               ch_types=['eeg', 'ecog'])
        raw = mne.io.RawArray(random((2, sampling_rate)), info)
        _handle_kind(raw)

    # if we cannot find a proper channel type, we raise an error
    with pytest.raises(ValueError, match='Neither MEG/EEG/iEEG channels'):
        info = mne.create_info(n_channels, sampling_rate, ch_types=['misc'])
        raw = mne.io.RawArray(data, info)
        _handle_kind(raw)


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


def test_make_filenames():
    """Test that we create filenames according to the BIDS spec."""
    # All keys work
    prefix_data = dict(subject='one', session='two', task='three',
                       acquisition='four', run='five', processing='six',
                       recording='seven', suffix='suffix.csv')
    assert str(make_bids_basename(**prefix_data)) == 'sub-one_ses-two_task-three_acq-four_run-five_proc-six_rec-seven_suffix.csv'  # noqa

    # subsets of keys works
    assert make_bids_basename(subject='one', task='three', run=4) == 'sub-one_task-three_run-04'  # noqa
    assert make_bids_basename(subject='one', task='three', suffix='hi.csv') == 'sub-one_task-three_hi.csv'  # noqa

    with pytest.raises(ValueError):
        make_bids_basename(subject='one-two', suffix='there.csv')

    with pytest.raises(ValueError, match='At least one'):
        make_bids_basename()

    # emptyroom checks
    with pytest.raises(ValueError, match='empty-room session should be a '
                                         'string of format YYYYMMDD'):
        make_bids_basename(subject='emptyroom', session='12345', task='noise')
    with pytest.raises(ValueError, match='task must be'):
        make_bids_basename(subject='emptyroom', session='20131201',
                           task='blah')


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


def test_check_types():
    """Test the check whether vars are str or None."""
    assert _check_types(['foo', 'bar', None]) is None
    with pytest.raises(ValueError):
        _check_types([None, 1, 3.14, 'meg', [1, 2]])


def test_path_to_str():
    """Test that _path_to_str returns a string."""
    path_str = 'foo'
    assert _path_to_str(path_str) == path_str
    assert _path_to_str(Path(path_str)) == path_str

    with pytest.raises(ValueError):
        _path_to_str(1)


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
    params = _parse_bids_filename(fname, verbose=False)
    assert params['sub'] == '01'
    assert params['ses'] == '02'
    assert params['run'] == '3'
    assert params['task'] == 'test'
    assert params['split'] == '01'
    assert list(params.keys()) == ['sub', 'ses', 'task', 'acq', 'run', 'proc',
                                   'space', 'rec', 'split', 'kind']


def test_age_on_date():
    """Test whether the age is determined correctly."""
    bday = datetime(1994, 1, 26)
    exp1 = datetime(2018, 1, 25)
    exp2 = datetime(2018, 1, 26)
    exp3 = datetime(2018, 1, 27)
    exp4 = datetime(1990, 1, 1)
    assert _age_on_date(bday, exp1) == 23
    assert _age_on_date(bday, exp2) == 24
    assert _age_on_date(bday, exp3) == 24
    with pytest.raises(ValueError):
        _age_on_date(bday, exp4)


def test_infer_eeg_placement_scheme():
    """Test inferring a correct EEG placement scheme."""
    # no eeg channels case (e.g., MEG data)
    data_path = op.join(base_path, 'bti', 'tests', 'data')
    raw_fname = op.join(data_path, 'test_pdf_linux')
    config_fname = op.join(data_path, 'test_config_linux')
    headshape_fname = op.join(data_path, 'test_hs_linux')
    raw = mne.io.read_raw_bti(raw_fname, config_fname, headshape_fname)
    placement_scheme = _infer_eeg_placement_scheme(raw)
    assert placement_scheme == 'n/a'

    # 1020 case
    data_path = op.join(base_path, 'brainvision', 'tests', 'data')
    raw_fname = op.join(data_path, 'test.vhdr')
    raw = mne.io.read_raw_brainvision(raw_fname)
    placement_scheme = _infer_eeg_placement_scheme(raw)
    assert placement_scheme == 'based on the extended 10/20 system'

    # Unknown case, use raw from 1020 case but rename a channel
    raw.rename_channels({'P3': 'foo'})
    placement_scheme = _infer_eeg_placement_scheme(raw)
    assert placement_scheme == 'n/a'


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

    # Now find a sidecar
    sidecar_fname = _find_matching_sidecar(bids_basename, bids_root,
                                           'coordsystem.json')
    expected_file = op.join('sub-01', 'ses-01', 'meg',
                            'sub-01_ses-01_coordsystem.json')
    assert sidecar_fname.endswith(expected_file)

    # Find multiple sidecars, tied in score, triggering an error
    with pytest.raises(RuntimeError, match='Expected to find a single'):
        open(sidecar_fname.replace('coordsystem.json',
                                   '2coordsystem.json'), 'w').close()
        _find_matching_sidecar(bids_basename, bids_root, 'coordsystem.json')

    # Find nothing but receive None, because we set `allow_fail` to True
    with pytest.warns(RuntimeWarning, match='Did not find any'):
        _find_matching_sidecar(bids_basename, bids_root, 'foo.bogus', True)


def test_bids_path(return_bids_test_dir):
    """Test usage of BIDSPath object."""
    bids_root = return_bids_test_dir

    bids_basename = make_bids_basename(
        subject=subject_id, session=session_id, run=run, acquisition=acq,
        task=task)

    # get_bids_fname should fire warning
    with pytest.raises(ValueError, match='No filename extension was provided'):
        bids_fname = bids_basename.get_bids_fname()

    # should find the correct filename if bids_root was passed
    bids_fname = bids_basename.get_bids_fname(bids_root=bids_root)
    assert bids_fname == bids_basename.update(suffix='meg.fif')

    # confirm BIDSPath assigns properties correctly
    bids_basename = make_bids_basename(subject=subject_id,
                                       session=session_id)
    assert bids_basename.subject == subject_id
    assert bids_basename.session == session_id
    assert 'subject' in bids_basename.entities
    assert 'session' in bids_basename.entities
    print(bids_basename.entities)
    assert all(bids_basename.entities.get(entity) is None
               for entity in ['task', 'run', 'recording', 'acquisition',
                              'space', 'processing',
                              'prefix', 'suffix'])

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

    # error check
    with pytest.raises(ValueError, match='Key must be one of*'):
        bids_basename.update(sub=subject_id, session=session_id)

    # test repr
    bids_path = make_bids_basename(subject='01', session='02',
                                   task='03', suffix='ieeg.edf')
    assert repr(bids_path) == 'BIDSPath(sub-01_ses-02_task-03_ieeg.edf)'


def test_delete_scans(return_bids_test_dir, _bids_validate):
    """Test update scans in a dir."""
    # deleting scan should conform to bids-validator
    delete_scan(bids_basename, bids_root=return_bids_test_dir)
    _bids_validate(return_bids_test_dir)

    ses_path = op.join(return_bids_test_dir,
                       f'sub-{subject_id}',
                       f'ses-{session_id}')
    scans_fpath = make_bids_basename(
        subject=subject_id, session=session_id,
        suffix='scans.tsv', prefix=ses_path)

    # the scan for bids_basename should be gone inside
    # scans.tsv and inside the actual session path
    scans_tsv = _from_tsv(scans_fpath)
    fnames = scans_tsv['filename']
    assert all([str(bids_basename) not in fname for fname in fnames])

    found_scans_fpaths = list(Path(ses_path).rglob(f'*{bids_basename}*'))
    assert found_scans_fpaths == []
