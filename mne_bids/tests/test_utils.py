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

from numpy.random import random
import mne
from mne.utils import _TempDir
from mne.datasets import testing

from mne_bids import make_bids_folders, make_bids_basename, write_raw_bids
from mne_bids.utils import (_check_types, print_dir_tree, _age_on_date,
                            _infer_eeg_placement_scheme, _handle_kind,
                            _find_matching_sidecar, _parse_ext,
                            _get_ch_type_mapping, _parse_bids_filename,
                            _find_best_candidates, get_entity_vals,
                            get_kinds)


base_path = op.join(op.dirname(mne.__file__), 'io')
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
    output_path = str(tmpdir_factory.mktemp('mnebids_utils_test_bids_ds'))
    data_path = testing.data_path()
    raw_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc_raw.fif')

    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2, 'Visual/Left': 3,
                'Visual/Right': 4, 'Smiley': 5, 'Button': 32}
    events_fname = op.join(data_path, 'MEG', 'sample',
                           'sample_audvis_trunc_raw-eve.fif')

    raw = mne.io.read_raw_fif(raw_fname)
    # Write multiple runs for test_purposes
    bids_basename2 = bids_basename.replace('run-{}'.format(run), 'run-02')
    for name in [bids_basename,
                 bids_basename2,
                 ]:
        with pytest.warns(UserWarning, match='No line frequency'):
            write_raw_bids(raw, name, output_path,
                           events_data=events_fname, event_id=event_id,
                           overwrite=True)

    return output_path


def test_get_keys(return_bids_test_dir):
    """Test getting the datatypes (=kinds) of a dir."""
    kinds = get_kinds(return_bids_test_dir)
    assert kinds == ['meg']


def test_get_entity_vals(return_bids_test_dir):
    """Test getting a list of entities."""
    bids_root = return_bids_test_dir
    with pytest.raises(ValueError, match='`key` must be one of'):
        get_entity_vals(bids_root, entity_key='bogus')

    assert get_entity_vals(bids_root, 'sub') == [subject_id]
    assert get_entity_vals(bids_root, 'ses') == [session_id]
    assert get_entity_vals(bids_root, 'run') == [run, '02']
    assert get_entity_vals(bids_root, 'acq') == []
    assert get_entity_vals(bids_root, 'task') == [task]


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
    assert make_bids_basename(**prefix_data) == 'sub-one_ses-two_task-three_acq-four_run-five_proc-six_recording-seven_suffix.csv'  # noqa

    # subsets of keys works
    assert make_bids_basename(subject='one', task='three', run=4) == 'sub-one_task-three_run-04'  # noqa
    assert make_bids_basename(subject='one', task='three', suffix='hi.csv') == 'sub-one_task-three_hi.csv'  # noqa

    with pytest.raises(ValueError):
        make_bids_basename(subject='one-two', suffix='there.csv')

    with pytest.raises(ValueError, match='At least one'):
        make_bids_basename()


def test_make_folders():
    """Test that folders are created and named properly."""
    # Make sure folders are created properly
    output_path = _TempDir()
    make_bids_folders(subject='hi', session='foo', kind='ba',
                      output_path=output_path)
    assert op.isdir(op.join(output_path, 'sub-hi', 'ses-foo', 'ba'))
    # If we remove a kwarg the folder shouldn't be created
    output_path = _TempDir()
    make_bids_folders(subject='hi', kind='ba', output_path=output_path)
    assert op.isdir(op.join(output_path, 'sub-hi', 'ba'))
    # check overwriting of folders
    make_bids_folders(subject='hi', kind='ba', output_path=output_path,
                      overwrite=True, verbose=True)


def test_check_types():
    """Test the check whether vars are str or None."""
    assert _check_types(['foo', 'bar', None]) is None
    with pytest.raises(ValueError):
        _check_types([None, 1, 3.14, 'meg', [1, 2]])


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
    'sub-01_ses-02_task-test_run-3_meg.fif',
    'sub-01_ses-02_task-test_run-3.fif',
    'sub-01_ses-02_task-test_run-3',
    '/bids_root/sub-01/ses-02/meg/sub-01_ses-02_task-test_run-3_meg.fif',
])
def test_parse_bids_filename(fname):
    """Test parsing entities from a bids filename."""
    params = _parse_bids_filename(fname, verbose=False)
    assert params['sub'] == '01'
    assert params['ses'] == '02'
    assert params['run'] == '3'
    assert params['task'] == 'test'
    assert list(params.keys()) == ['sub', 'ses', 'task', 'acq', 'run', 'proc',
                                   'space', 'recording', 'kind']


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
    output_path = return_bids_test_dir

    # Now find a sidecar
    sidecar_fname = _find_matching_sidecar(bids_basename, output_path,
                                           'coordsystem.json')
    expected_file = op.join('sub-01', 'ses-01', 'meg',
                            'sub-01_ses-01_coordsystem.json')
    assert sidecar_fname.endswith(expected_file)

    # Find multiple sidecars, tied in score, triggering an error
    with pytest.raises(RuntimeError, match='Expected to find a single'):
        open(sidecar_fname.replace('coordsystem.json',
                                   '2coordsystem.json'), 'w').close()
        _find_matching_sidecar(bids_basename, output_path, 'coordsystem.json')

    # Find nothing but receive None, because we set `allow_fail` to True
    with pytest.warns(UserWarning, match='Did not find any'):
        _find_matching_sidecar(bids_basename, output_path, 'foo.bogus', True)
