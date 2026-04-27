"""Test for the MNE BIDSPath functions."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

import functools
import glob
import os
import os.path as op
import pickle
import shutil as sh
import timeit
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import mne
import pytest
from mne.datasets import testing
from mne.io import anonymize_info, read_raw_fif
from test_read import _read_raw_fif, warning_str

import mne_bids
import mne_bids.path
from mne_bids import (
    BIDSPath,
    get_datatypes,
    get_entity_vals,
    print_dir_tree,
    read_raw_bids,
    write_meg_calibration,
    write_meg_crosstalk,
    write_raw_bids,
)
from mne_bids.config import ALLOWED_PATH_ENTITIES_SHORT
from mne_bids.path import (
    _DATATYPE_LIST,
    _filter_fnames,
    _find_best_candidates,
    _parse_ext,
    _return_root_paths,
    _suggest_fix_for_segment,
    find_matching_paths,
    get_bids_path_from_fname,
    get_entities_from_fname,
    search_folder_for_text,
)

subject_id = "01"
session_id = "01"
run = "01"
acq = None
task = "testing"

data_path = testing.data_path(download=False)
_bids_path = BIDSPath(
    subject=subject_id, session=session_id, run=run, acquisition=acq, task=task
)


def test_datatypes_alphabetical():
    """Make sure we keep this list alphabetized."""
    assert _DATATYPE_LIST == tuple(sorted(_DATATYPE_LIST))


@pytest.fixture(
    params=[pytest.param("testing", marks=mne.datasets.testing._pytest_mark())],
)
def bids_root(tmp_path):
    """Return path to a written test BIDS dir."""
    bids_root = tmp_path / "mb_ds"
    bids_root.mkdir()
    raw_fname = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif"

    event_id = {
        "Auditory/Left": 1,
        "Auditory/Right": 2,
        "Visual/Left": 3,
        "Visual/Right": 4,
        "Smiley": 5,
        "Button": 32,
    }
    events_fname = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw-eve.fif"
    cal_fname = data_path / "SSS" / "sss_cal_mgh.dat"
    crosstalk_fname = data_path / "SSS" / "ct_sparse.fif"

    raw = mne.io.read_raw_fif(raw_fname)
    raw.info["line_freq"] = 60

    # Drop unknown events.
    events = mne.read_events(events_fname)
    events = events[events[:, 2] != 0]

    bids_path = _bids_path.copy().update(root=bids_root)
    # Write multiple runs for test_purposes
    for run_idx in [run, "02"]:
        name = bids_path.copy().update(run=run_idx)
        write_raw_bids(raw, name, events=events, event_id=event_id, overwrite=True)

    write_meg_calibration(cal_fname, bids_path=bids_path)
    write_meg_crosstalk(crosstalk_fname, bids_path=bids_path)
    return bids_root


def test_get_datatypes(bids_root_dense, bids_root, path_counter):
    """Test getting the datatypes (=modalities) of a dir."""
    modalities = mne_bids.get_datatypes(bids_root)
    assert modalities == ["meg"]
    assert path_counter.count == 2
    modalities = mne_bids.get_datatypes(bids_root_dense.root)
    assert modalities == ["eeg", "meg"]
    assert path_counter.count == 108


_ROOT_NUM = 9
_SUB_NUM = 22


@pytest.mark.parametrize(
    "entity, expected_vals, kwargs, count",
    [
        pytest.param("bogus", None, None, 0, id="bogus"),
        pytest.param("subject", [subject_id], None, _ROOT_NUM, id="subject"),
        pytest.param("session", [session_id], None, _ROOT_NUM, id="session"),
        pytest.param(
            "session",
            [],
            dict(
                ignore_tasks="testing",
                ignore_acquisitions=("calibration", "crosstalk"),
                ignore_suffixes=("scans", "coordsystem"),
            ),
            _ROOT_NUM,
            id="session_ignore_kinds",
        ),
        pytest.param("run", [run, "02"], None, _SUB_NUM, id="run"),
        pytest.param(
            "acquisition",
            ["calibration", "crosstalk"],
            None,
            _SUB_NUM,
            id="acquisition",
        ),
        pytest.param("task", [task], None, _SUB_NUM, id="task"),
        pytest.param(
            "subject",
            [],
            dict(ignore_subjects=[subject_id]),
            _ROOT_NUM,
            id="subject_ignore",
        ),
        pytest.param(
            "subject",
            [],
            dict(ignore_subjects=subject_id),
            _ROOT_NUM,
            id="subject_ignore_single",
        ),
        pytest.param(
            "session",
            [],
            dict(ignore_sessions=[session_id]),
            _ROOT_NUM,
            id="session_ignore_sessions",
        ),
        pytest.param(
            "session",
            [],
            dict(ignore_sessions=session_id),
            _ROOT_NUM,
            id="session_ignore_single",
        ),
        pytest.param("run", [run], dict(ignore_runs=["02"]), _SUB_NUM, id="run_ignore"),
        pytest.param(
            "run", [run], dict(ignore_runs="02"), _SUB_NUM, id="run_ignore_single"
        ),
        pytest.param("task", [], dict(ignore_tasks=[task]), _SUB_NUM, id="task_ignore"),
        pytest.param(
            "task", [], dict(ignore_tasks=task), _SUB_NUM, id="task_ignore_single"
        ),
        pytest.param(
            "run",
            [run, "02"],
            dict(ignore_runs=["bogus"]),
            _SUB_NUM,
            id="run_ignore_bogus",
        ),
        pytest.param(
            "run", [], dict(ignore_datatypes=["meg"]), _SUB_NUM, id="ignore_datatypes"
        ),
    ],
)
def test_get_entity_vals(entity, expected_vals, kwargs, count, bids_root, path_counter):
    """Test getting a list of entities."""
    # Add some derivative data that should be ignored by get_entity_vals()
    deriv_path = Path(bids_root) / "derivatives"
    deriv_meg_dir = deriv_path / "pipeline" / "sub-deriv" / "ses-deriv" / "meg"
    deriv_meg_dir.mkdir(parents=True)
    (deriv_meg_dir / "sub-deriv_ses-deriv_task-deriv_meg.fif").touch()
    (deriv_meg_dir / "sub-deriv_ses-deriv_task-deriv_meg.json").touch()

    if kwargs is None:
        kwargs = dict()

    if entity == "bogus":
        with pytest.raises(ValueError, match="`key` must be one of"):
            get_entity_vals(root=bids_root, entity_key=entity, **kwargs)
        return
    assert path_counter.count == 0
    vals = get_entity_vals(root=bids_root, entity_key=entity, **kwargs)
    assert path_counter.count == count
    assert vals == expected_vals

    # test using ``with_key`` kwarg
    entities = get_entity_vals(
        root=bids_root, entity_key=entity, with_key=True, **kwargs
    )
    entity_long_to_short = {
        val: key for key, val in ALLOWED_PATH_ENTITIES_SHORT.items()
    }
    assert entities == [
        f"{entity_long_to_short[entity]}-{val}" for val in expected_vals
    ]

    # Test without ignoring the derivatives dir
    entities = get_entity_vals(
        root=bids_root, entity_key=entity, **kwargs, ignore_dirs=None
    )
    if entity not in ("acquisition", "run"):
        assert "deriv" in entities


@testing.requires_testing_data
def test_get_entity_vals_ignore_hidden(bids_root):
    """Test hidden directories are skipped by default and included when opted in."""
    hidden_meg_dir = Path(bids_root) / ".hidden_data" / "sub-hid" / "ses-hid" / "meg"
    hidden_meg_dir.mkdir(parents=True)
    (hidden_meg_dir / "sub-hid_ses-hid_task-hid_meg.fif").touch()

    # Default (ignore_hidden=True): hidden dir must not contribute any values
    for entity_key in ("subject", "session", "task"):
        vals = get_entity_vals(root=bids_root, entity_key=entity_key)
        assert "hid" not in vals, (
            f"entity '{entity_key}' leaked from hidden dir: {vals}"
        )

    # ignore_hidden=False + ignore_dirs=None: hidden dir must be included
    for entity_key in ("subject", "session", "task"):
        vals = get_entity_vals(
            root=bids_root,
            entity_key=entity_key,
            ignore_hidden=False,
            ignore_dirs=None,
        )
        assert "hid" in vals, (
            f"entity '{entity_key}' not found in hidden dir when "
            f"ignore_hidden=False: {vals}"
        )

    # include_match branch: hidden dir must also be excluded by default
    vals_include = get_entity_vals(
        root=bids_root, entity_key="subject", include_match="**/"
    )
    assert "hid" not in vals_include

    # include_match branch: hidden dir included when ignore_hidden=False
    vals_include_hidden = get_entity_vals(
        root=bids_root,
        entity_key="subject",
        include_match="**/",
        ignore_hidden=False,
        ignore_dirs=None,
    )
    assert "hid" in vals_include_hidden


@dataclass
class _PathAccessCounter:
    calls: list[str]
    files: list[str]
    count: int = 0


@pytest.fixture
def path_counter(monkeypatch):
    """Count number of files traversed using iglob."""
    path_counter = _PathAccessCounter(calls=list(), files=list())
    orig_iglob = glob.iglob
    orig_walk = os.walk

    # Internal MBP functions should set .count=0 and .files=[] and append themselves
    # to .calls. Builtin Python functions should increment .count and extend .files.

    def _return_root_paths_count(*args, **kwargs):
        path_counter.calls.append("_return_root_paths")
        path_counter.count = 0
        out = _return_root_paths(*args, **kwargs)
        path_counter.files.extend(out)
        return out

    def get_entity_vals_count(*args, **kwargs):
        path_counter.calls.append("get_entity_vals")
        path_counter.count = 0
        out = get_entity_vals(*args, **kwargs)
        path_counter.files.extend(out)
        return out

    def get_datatypes_count(*args, **kwargs):
        path_counter.calls.append("get_datatypes")
        path_counter.count = 0
        out = get_datatypes(*args, **kwargs)
        path_counter.files.extend(out)
        return out

    orig_find = mne_bids.path._find_matching_sidecar

    def _find_matching_sidecar_count(*args, **kwargs):
        path_counter.calls.append("_find_matching_sidecar")
        path_counter.count = 0
        out = orig_find(*args, **kwargs)
        func = getattr(
            path_counter.files, "extend" if isinstance(out, list) else "append"
        )
        func(out)
        path_counter.files.append(out)
        return out

    def _path_glob_iglob(root, pattern):
        # Reroute Path.glob through iglob to count accesses
        return [root / f for f in glob.iglob(pattern, root_dir=root, recursive=True)]

    def _path_rglob_iglob(root, pattern):
        # Reroute Path.rglob through iglob to count accesses
        return [
            root / f for f in glob.iglob(f"**/{pattern}", root_dir=root, recursive=True)
        ]

    def _iglob_count(*args, **kwargs):
        for fn in orig_iglob(*args, **kwargs):
            path_counter.count += 1
            path_counter.files.append(fn)
            yield fn

    def _os_walk_count(top, **kwargs):
        for dirpath, dirs, files in orig_walk(top, **kwargs):
            path_counter.count += len(files) + len(dirs)
            out = dirpath, dirs, files
            path_counter.files.extend(
                os.path.join(os.path.relpath(dirpath, top), d) for d in dirs
            )
            path_counter.files.extend(
                os.path.join(os.path.relpath(dirpath, top), f) for f in files
            )
            yield out

    monkeypatch.setattr(glob, "iglob", _iglob_count)
    monkeypatch.setattr(mne_bids.path, "_path_glob", _path_glob_iglob)
    monkeypatch.setattr(mne_bids.path, "_path_rglob", _path_rglob_iglob)
    monkeypatch.setattr(mne_bids.path, "_return_root_paths", _return_root_paths_count)
    monkeypatch.setattr(mne_bids, "get_entity_vals", get_entity_vals_count)
    monkeypatch.setattr(mne_bids, "get_datatypes", get_datatypes_count)
    monkeypatch.setattr(
        mne_bids.path, "_find_matching_sidecar", _find_matching_sidecar_count
    )
    monkeypatch.setattr(os, "walk", _os_walk_count)
    yield path_counter


@dataclass
class _BIDSDirDense:
    root: BIDSPath
    bids_subdirectories: list[str]
    n_subjects = 9
    n_sessions = 4
    n_derivatives = 17


@pytest.fixture
def bids_root_dense(tmp_path):
    """Create a dense BIDS directory tree."""
    tmp_bids_root = tmp_path / "mb_dense_ds"
    tmp_bids_root.mkdir()
    out = _BIDSDirDense(root=tmp_bids_root, bids_subdirectories=[])

    derivatives = [
        Path("derivatives", "derivatives" + str(i)) for i in range(out.n_derivatives)
    ]

    out.bids_subdirectories = ["", "sourcedata", *derivatives]

    # Create a BIDS compliant directory tree with high number of branches
    for root_file in (
        "dataset_description.json",
        "participants.tsv",
        "participants.csv",
        "README",
        "coordsystem.json",
        "scans.tsv",
    ):
        (tmp_bids_root / root_file).touch()
    for i in range(1, out.n_subjects + 1):
        sub_root = tmp_bids_root / f"sub-{i}"
        sub_root.mkdir()
        (sub_root / f"sub-{i}_electrodes.tsv").touch()
        for j in range(1, out.n_sessions + 1):
            for subdir in out.bids_subdirectories:
                for datatype in ["eeg", "meg"]:
                    bids_subdir = BIDSPath(
                        subject=str(i),
                        session=str(j),
                        datatype=datatype,
                        task="audvis",
                        root=str(tmp_bids_root / subdir),
                    )
                    bids_subdir.mkdir(exist_ok=True)
                    bsd_path = Path(bids_subdir.directory)
                    (bsd_path / "participants.tsv").touch()
                    (bsd_path / "participants.csv").touch()
                    (bsd_path / "README").touch()

                    # os.makedirs(bids_subdir.directory, exist_ok=True)
                    for ext_suf in ("events.tsv", "events.csv"):
                        (bsd_path / f"{bids_subdir.basename}_{ext_suf}").touch()

                    if datatype == "meg":
                        ctf_path = bsd_path / f"{bids_subdir.basename}_meg.ds"
                        ctf_path.mkdir(exist_ok=True)
                        (ctf_path / f"{bids_subdir.basename}.meg4").touch()
                        (ctf_path / f"{bids_subdir.basename}.hc").touch()
                        (ctf_path / "hz.ds").mkdir(exist_ok=True)
                        (ctf_path / "hz.ds" / "hz.meg4").touch()
                        (ctf_path / "hz.ds" / "hz.hc").touch()
                    else:
                        assert datatype == "eeg"
                        assert str(bsd_path).endswith("eeg")
                        for ext_suf in ("eeg.edf", "channels.tsv", "eeg.json"):
                            (bsd_path / f"{bids_subdir.basename}_{ext_suf}").touch()

    return out


@pytest.mark.slow  # ~5s
def test_path_benchmark(bids_root_dense, monkeypatch, path_counter):
    """Benchmark exploring bids tree."""
    tmp_bids_root = bids_root_dense.root
    n_subjects = bids_root_dense.n_subjects
    n_sessions = bids_root_dense.n_sessions
    # This benchmark is to verify the speed-up in function call get_entity_vals with
    # `include_match=sub-*/` in face of a bids tree hosting derivatives and sourcedata.
    fnames = mne_bids.path._return_root_paths(tmp_bids_root)
    assert len(fnames) == len(set(fnames))
    assert len(fnames) == 10956
    max_count = 11642
    assert path_counter.count == max_count

    # apply nosub on find_matching_matchs with root level bids directory should
    # yield a performance boost of order of length from bids_subdirectories.
    setup = "import mne_bids\ntmp_bids_root=r'" + str(tmp_bids_root) + "'"
    timed_all = timeit.timeit(
        "mne_bids.find_matching_paths(tmp_bids_root)", setup=setup, number=1
    )
    assert path_counter.count == max_count
    timed_ignored_nosub = timeit.timeit(
        "mne_bids.find_matching_paths(tmp_bids_root, ignore_nosub=True)",
        setup=setup,
        number=1,
    )
    assert path_counter.count == 621

    # while this should be of same order, lets give it some space by a factor of 3
    target = 3 * timed_all / len(bids_root_dense.bids_subdirectories)
    assert timed_ignored_nosub < target

    # these should be equivalent
    out_1 = mne_bids.get_entity_vals(tmp_bids_root, "session")
    assert path_counter.count == 638
    out_2 = mne_bids.get_entity_vals(tmp_bids_root, "session", include_match="**/")
    assert path_counter.count == 41328
    assert out_1 == out_2
    out_3 = mne_bids.get_entity_vals(tmp_bids_root, "session", include_match="sub-*/")
    assert path_counter.count == 360
    assert out_2 == out_3  # all are sub-* vals
    out_4 = mne_bids.get_entity_vals(tmp_bids_root, "session", include_match="none/")
    assert path_counter.count == 0
    assert out_4 == []

    # BIDSPath.match optimizations as well
    path = BIDSPath(
        datatype="meg",
        extension=".ds",
        root=tmp_bids_root,
    )
    paths = path.match()
    # TODO: We *should* have n_subjects * n_sessions of these, but we get way more:
    assert len(paths) == 684
    # This is because we have duplicate files here with BIDS entities, like:
    # bids_root/derivatives/derivatives0/sub-1/ses-1/meg/sub-1_ses-1_task-audvis_meg.ds
    # bids_root/derivatives/derivatives1/sub-1/ses-1/meg/sub-1_ses-1_task-audvis_meg.ds
    # And _fnames_to_bidspaths converts these to the same names!
    assert paths[0] == paths[n_subjects * n_sessions]
    assert path_counter.count == 3420
    path.subject = "1"  # add subject
    paths = path.match()
    assert len(paths) == n_sessions
    assert path_counter.count == 20
    path.session = "2"  # add session
    paths = path.match()
    assert len(paths) == 1, paths
    assert path_counter.count == 5


def _scan_targeted_meg(root, entities=None):
    return _return_root_paths(
        root,
        datatype="meg",
        ignore_json=True,
        ignore_nosub=False,
        entities=entities,
    )


def test_entity_targeted_scan_speed(tmp_path_factory):
    """Ensure entity-aware root scan is significantly faster."""
    bids_root = Path(tmp_path_factory.mktemp("mnebids_entity_scan"))

    n_subjects = 60
    n_sessions = 4
    n_runs = 12

    target_entities = {"subject": "01", "session": "02"}
    target_sub = f"sub-{target_entities['subject']}"
    target_ses = f"ses-{target_entities['session']}"

    for subj_idx in range(1, n_subjects + 1):
        sub_label = f"{subj_idx:02d}"
        for ses_idx in range(1, n_sessions + 1):
            ses_label = f"{ses_idx:02d}"
            meg_dir = bids_root / f"sub-{sub_label}" / f"ses-{ses_label}" / "meg"
            meg_dir.mkdir(parents=True, exist_ok=True)
            for run_idx in range(1, n_runs + 1):
                fname = (
                    f"sub-{sub_label}_ses-{ses_label}_task-"
                    f"task_run-{run_idx:02d}_meg.fif"
                )
                (meg_dir / fname).touch()

    timer = timeit.default_timer
    # Warm-up to mitigate cold-cache effects.
    _scan_targeted_meg(bids_root)
    baseline_durations = []
    for _ in range(5):
        start = timer()
        _scan_targeted_meg(bids_root)
        baseline_durations.append(timer() - start)
    baseline_durations.sort()
    baseline_mean = sum(baseline_durations[1:-1]) / 3

    optimized_durations = []
    for _ in range(5):
        start = timer()
        _scan_targeted_meg(bids_root, entities=target_entities)
        optimized_durations.append(timer() - start)
    optimized_durations.sort()
    optimized_mean = sum(optimized_durations[1:-1]) / 3

    optimized_paths = _scan_targeted_meg(bids_root, entities=target_entities)
    expected_len = n_runs
    assert all(
        target_sub in path.as_posix() and target_ses in path.as_posix()
        for path in optimized_paths
    )
    assert len(optimized_paths) == expected_len
    # Require a substantial speed-up to guard against regressions.
    assert optimized_mean < baseline_mean * 0.5


def test_search_folder_for_text(capsys):
    """Test finding entries."""
    with pytest.raises(ValueError, match="is not a directory"):
        search_folder_for_text("foo", "i_dont_exist")

    # We check the testing directory
    test_dir = op.dirname(__file__)
    search_folder_for_text("n/a", test_dir)
    captured = capsys.readouterr()
    assert "sub-01_ses-eeg_task-rest_eeg.json" in captured.out
    assert (
        "    1    name      type      units     low_cutof high_cuto descripti sampling_ status    status_de\n"  # noqa: E501
        "    2    Fp1       EEG       µV        0.0159154 1000.0    ElectroEn 5000.0    good      n/a"  # noqa: E501
    ) in captured.out
    # test if pathlib.Path object
    search_folder_for_text("n/a", Path(test_dir))

    # test returning a string and without line numbers
    out = search_folder_for_text("n/a", test_dir, line_numbers=False, return_str=True)
    assert "sub-01_ses-eeg_task-rest_eeg.json" in out
    assert (
        "    name      type      units     low_cutof high_cuto descripti sampling_ status    status_de\n"  # noqa: E501
        "    Fp1       EEG       µV        0.0159154 1000.0    ElectroEn 5000.0    good      n/a"  # noqa: E501
    ) in out


def test_print_dir_tree(capsys):
    """Test printing a dir tree."""
    with pytest.raises(FileNotFoundError, match="Folder does not exist"):
        print_dir_tree("i_dont_exist")

    # We check the testing directory
    test_dir = op.dirname(__file__)
    with pytest.raises(ValueError, match="must be a positive integer"):
        print_dir_tree(test_dir, max_depth=-1)
    with pytest.raises(TypeError, match="must be a positive integer"):
        print_dir_tree(test_dir, max_depth="bad")

    # Do not limit depth
    print_dir_tree(test_dir)
    captured = capsys.readouterr()
    assert "|--- test_utils.py" in captured.out.split("\n")
    assert f"|--- __pycache__{os.sep}" in captured.out.split("\n")
    assert ".pyc" in captured.out

    # Now limit depth ... we should not descend into pycache
    print_dir_tree(test_dir, max_depth=1)
    captured = capsys.readouterr()
    assert "|--- test_utils.py" in captured.out.split("\n")
    assert f"|--- __pycache__{os.sep}" in captured.out.split("\n")
    assert ".pyc" not in captured.out

    # Limit depth even more
    print_dir_tree(test_dir, max_depth=0)
    captured = capsys.readouterr()
    assert captured.out == f"|tests{os.sep}\n"

    # test if pathlib.Path object
    print_dir_tree(Path(test_dir))

    # test returning a string
    out = print_dir_tree(test_dir, return_str=True, max_depth=1)
    assert isinstance(out, str)
    assert "|--- test_utils.py" in out.split("\n")
    assert f"|--- __pycache__{os.sep}" in out.split("\n")
    assert ".pyc" not in out


def test_make_folders(tmp_path):
    """Test that folders are created and named properly."""
    # Make sure folders are created properly
    bids_path = BIDSPath(
        subject="01", session="foo", datatype="eeg", root=str(tmp_path)
    )
    bids_path.mkdir().directory
    assert op.isdir(tmp_path / "sub-01" / "ses-foo" / "eeg")

    # If we remove a kwarg the folder shouldn't be created
    bids_path = BIDSPath(subject="02", datatype="eeg", root=tmp_path)
    bids_path.mkdir().directory
    assert op.isdir(tmp_path / "sub-02" / "eeg")

    # Check if a pathlib.Path bids_root works.
    bids_path = BIDSPath(subject="03", session="foo", datatype="eeg", root=tmp_path)
    bids_path.mkdir().directory
    assert op.isdir(tmp_path / "sub-03" / "ses-foo" / "eeg")

    # Check if bids_root=None creates folders in the current working directory
    bids_root = tmp_path / "tmp"
    bids_root.mkdir()
    curr_dir = os.getcwd()
    os.chdir(bids_root)
    bids_path = BIDSPath(subject="04", session="foo", datatype="eeg")
    bids_path.mkdir().directory
    assert op.isdir(op.join(os.getcwd(), "sub-04", "ses-foo", "eeg"))
    os.chdir(curr_dir)


def test_rm(bids_root, capsys, tmp_path, path_counter, fast_sidecar):
    """Test BIDSPath's rm method to remove files."""
    # for some reason, mne's logger can't be captured by caplog....

    # without providing all the entities, ambiguous when trying to use fpath
    bids_path = BIDSPath(
        subject=subject_id,
        session=session_id,
        run="01",
        acquisition=acq,
        task=task,
        root=bids_root,
    )

    # Delete one run:
    deleted_paths = bids_path.match(ignore_json=False)
    assert path_counter.calls == ["_return_root_paths"]
    want_path = (
        bids_path.directory
        / f"sub-{bids_path.subject}_ses-{bids_path.session}_scans.tsv"
    )
    updated_paths = [
        bids_path.copy()
        .update(datatype=None)
        .find_matching_sidecar(
            suffix="scans",
            extension=".tsv",
            on_error="raise",
        )
    ]
    assert updated_paths[0] == want_path
    assert path_counter.calls == ["_return_root_paths", "_find_matching_sidecar"]
    want_count = 0 if fast_sidecar else 1
    assert path_counter.count == want_count
    expected = ["Executing the following operations:", "Delete:", "Update:", ""]
    expected += [str(p) for p in deleted_paths + updated_paths]
    assert len(path_counter.calls) == 2
    bids_path.rm(safe_remove=False, verbose="INFO")
    assert len(path_counter.calls) == 8
    captured = capsys.readouterr().out
    assert set(captured.splitlines()) == set(expected)

    # delete the last run of a subject:
    bids_path = BIDSPath(
        subject=subject_id,
        session=session_id,
        root=bids_root,
    )
    deleted_paths = bids_path.match(ignore_json=False)
    deleted_paths += [
        BIDSPath(
            root=bids_path.root,
            subject=bids_path.subject,
        ).directory
    ]
    assert len(path_counter.calls) == 9
    updated_paths = [
        bids_path.copy()
        .update(datatype=None)
        .find_matching_sidecar(
            suffix="scans",
            extension=".tsv",
            on_error="raise",
        ),
        bids_path.root / "participants.tsv",
    ]
    assert len(path_counter.calls) == 10
    assert path_counter.count == want_count
    expected = ["Executing the following operations:", "Delete:", "Update:", ""]
    expected += [str(p) for p in deleted_paths + updated_paths]
    bids_path.rm(safe_remove=False, verbose="INFO")
    captured2 = capsys.readouterr().out
    assert set(captured2.splitlines()) == set(expected)
    print("\n".join(captured))
    print("\n".join(captured2))


def test_parse_ext():
    """Test the file extension extraction."""
    f = "sub-05_task-matchingpennies.vhdr"
    fname, ext = _parse_ext(f)
    assert fname == "sub-05_task-matchingpennies"
    assert ext == ".vhdr"

    # Test for case where no extension: assume BTi format
    f = "sub-01_task-rest"
    fname, ext = _parse_ext(f)
    assert fname == f
    assert ext == ".pdf"

    # Get a .nii.gz file
    f = "sub-01_task-rest.nii.gz"
    fname, ext = _parse_ext(f)
    assert fname == "sub-01_task-rest"
    assert ext == ".nii.gz"


@pytest.mark.parametrize(
    "fname",
    [
        "sub-01_ses-02_task-test_run-3_split-1_meg.fif",
        "sub-01_ses-02_task-test_run-3_split-1",
        "/bids_root/sub-01/ses-02/meg/sub-01_ses-02_task-test_run-3_split-1_meg.fif",
        "sub-01/ses-02/meg/sub-01_ses-02_task-test_run-3_split-1_meg.fif",
    ],
)
def test_get_bids_path_from_fname(fname):
    """Test get_bids_path_from_fname()."""
    bids_path = get_bids_path_from_fname(fname)
    assert bids_path.basename == Path(fname).name

    if "/bids_root/" in fname:
        assert Path(bids_path.root) == Path("/bids_root")
    else:
        if "meg" in fname:
            # directory should match
            assert Path(bids_path.directory) == Path("sub-01/ses-02/meg")

        # root should be default '.'
        assert str(bids_path.root) == "."


@pytest.mark.parametrize(
    "fname",
    [
        "sub-01_ses-02_task-test_run-3_split-1_desc-filtered_meg.fif",
        "sub-01_ses-02_task-test_run-3_split-1_desc-filtered.fif",
        "sub-01_ses-02_task-test_run-3_split-1_desc-filtered",
        (
            "/bids_root/sub-01/ses-02/meg/"
            + "sub-01_ses-02_task-test_run-3_split-1_desc-filtered_meg.fif"
        ),
    ],
)
def test_get_entities_from_fname(fname):
    """Test parsing entities from a bids filename."""
    params = get_entities_from_fname(fname)
    assert params["subject"] == "01"
    assert params["session"] == "02"
    assert params["run"] == "3"
    assert params["task"] == "test"
    assert params["description"] == "filtered"
    assert params["split"] == "1"
    assert list(params.keys()) == [
        "subject",
        "session",
        "task",
        "acquisition",
        "run",
        "processing",
        "space",
        "recording",
        "split",
        "description",
        "tracking_system",
    ]


@pytest.mark.parametrize(
    "fname",
    [
        "sub-01_ses-02_task-test_run-3_split-01_meg.fif",
        ("/bids_root/sub-01/ses-02/meg/sub-01_ses-02_task-test_run-3_split-01_meg.fif"),
        "sub-01_ses-02_task-test_run-3_split-01_foo-tfr_meg.fif",
    ],
)
def test_get_entities_from_fname_errors(fname):
    """Test parsing entities from bids filename.

    Extends utility for not supported BIDS entities, such
    as 'foo'.
    """
    if "foo" in fname:
        with pytest.raises(KeyError, match="Unexpected entity"):
            params = get_entities_from_fname(fname, on_error="raise")
        with pytest.warns(RuntimeWarning, match="Unexpected entity"):
            params = get_entities_from_fname(fname, on_error="warn")
        params = get_entities_from_fname(fname, on_error="ignore")
    else:
        params = get_entities_from_fname(fname, on_error="raise")

    expected_keys = [
        "subject",
        "session",
        "task",
        "acquisition",
        "run",
        "processing",
        "space",
        "recording",
        "split",
        "description",
        "tracking_system",
    ]

    assert params["subject"] == "01"
    assert params["session"] == "02"
    assert params["run"] == "3"
    assert params["task"] == "test"
    assert params["split"] == "01"
    if "foo" in fname:
        assert params["foo"] == "tfr"
        expected_keys.append("foo")
    assert list(params.keys()) == expected_keys


@pytest.mark.parametrize(
    "fname, expected_warn_task, expected_fixed_task, expected_other_key,"
    " expected_other_val",
    [
        ("sub-01_task-ECONrun-1_eeg.set", "ECONrun", "ECON", "run", "1"),
        ("sub-01_task-restacq-full_eeg.set", "restacq", "rest", "acquisition", "full"),
        (
            "/bids_root/sub-01/eeg/sub-01_task-ECONrun-1_eeg.set",
            "ECONrun",
            "ECON",
            "run",
            "1",
        ),
    ],
)
def test_get_entities_from_fname_missing_separator(
    fname,
    expected_warn_task,
    expected_fixed_task,
    expected_other_key,
    expected_other_val,
):
    """Test missing separator handling across warn/autofix modes."""
    # on_error="raise" should raise with a suggested fix showing
    # the full corrected filename
    with pytest.raises(ValueError, match="Suggested fix.*\\.set"):
        get_entities_from_fname(fname, on_error="raise")
    # on_error="warn" should warn but not apply autofix
    with pytest.warns(RuntimeWarning, match="Suggested fix"):
        params_warn = get_entities_from_fname(fname, on_error="warn")
    assert params_warn["task"] == expected_warn_task
    assert params_warn[expected_other_key] is None
    # on_error="autofix" should warn and return autofixed entities
    with pytest.warns(RuntimeWarning, match="Suggested fix"):
        params_fix = get_entities_from_fname(fname, on_error="autofix")
    assert params_fix["task"] == expected_fixed_task
    assert params_fix[expected_other_key] == expected_other_val
    # on_error="ignore" should not raise or warn, and the merged segment
    # is parsed as-is by the regex (only the first key-value is captured)
    params = get_entities_from_fname(fname, on_error="ignore")
    assert params["task"] == expected_warn_task
    assert params[expected_other_key] is None


def test_get_entities_from_fname_missing_separator_three_entities():
    """Test detection and autofix when three entities are merged."""
    fname = "sub-01task-ECONrun-1_eeg.set"
    with pytest.raises(ValueError, match="Suggested fix"):
        get_entities_from_fname(fname, on_error="raise")
    with pytest.warns(RuntimeWarning, match="Suggested fix"):
        params = get_entities_from_fname(fname, on_error="autofix")
    assert params["subject"] == "01"
    assert params["task"] == "ECON"
    assert params["run"] == "1"


def test_get_entities_from_fname_missing_separator_no_autofix():
    """Test non-fixable segments still preserve best-effort parsing in warn mode."""
    # "rest" doesn't end with any known BIDS entity key, so autofix fails
    fname = "sub-01_task-rest-state_eeg.set"
    with pytest.raises(ValueError, match="multiple hyphens"):
        get_entities_from_fname(fname, on_error="raise")
    with pytest.warns(RuntimeWarning, match="multiple hyphens"):
        params = get_entities_from_fname(fname, on_error="warn")
    # Warn mode should still parse the key-value part it can extract.
    assert params["task"] == "rest"
    # But other valid entities are still parsed
    assert params["subject"] == "01"


def test_get_entities_from_fname_missing_separator_repeated_segment_suggestions():
    """Test suggested fixes are correct for repeated malformed segments."""
    fname = "sub-01_foo-ECONrun-1_foo-ECONrun-1_eeg.set"
    with pytest.warns(RuntimeWarning) as warnings:
        get_entities_from_fname(fname, on_error="autofix")

    messages = [str(w.message) for w in warnings]
    suggestion_msgs = [msg for msg in messages if "Suggested fix:" in msg]
    assert len(suggestion_msgs) == 2
    assert "foo-ECON_run-1_foo-ECONrun-1" in suggestion_msgs[0]
    assert "foo-ECON_run-1_foo-ECON_run-1" in suggestion_msgs[1]


@pytest.mark.parametrize(
    "fname",
    [
        "sub-01_ses-02_task-test_run-3_split-01_meg.fif",
        "sub-01_ses-02_task-test_run-3_bold.nii.gz",
        "sub-01_ses-02_task-test_channels.tsv.gz",
    ],
)
def test_get_entities_from_fname_no_false_positive(fname):
    """Test that valid filenames do not trigger missing separator warning."""
    # Should not raise or warn
    get_entities_from_fname(fname, on_error="raise")


@pytest.mark.parametrize(
    "segment, known_keys, expected",
    [
        # Line 1762: segments with fewer than 3 parts after split on "-"
        pytest.param("nohyphen", ["task", "run"], None, id="no_hyphen"),
        pytest.param("task-rest", ["task", "run"], None, id="single_hyphen"),
        pytest.param("", ["task", "run"], None, id="empty_string"),
        # Line 1772: ambiguous — two known keys match as suffix of middle part
        pytest.param(
            "k1-xcab-v",
            ["ab", "cab", "k1"],
            None,
            id="ambiguous_two_keys_match",
        ),
        # Line 1784: trailing hyphen produces empty last value
        pytest.param(
            "task-ECONrun-",
            list(ALLOWED_PATH_ENTITIES_SHORT.keys()),
            None,
            id="trailing_hyphen_empty_value",
        ),
        # No known key matches the middle part suffix
        pytest.param("task-FOOxyz-1", ["task", "run"], None, id="no_key_match"),
        # Positive: valid decomposition for regression
        pytest.param(
            "task-ECONrun-1",
            list(ALLOWED_PATH_ENTITIES_SHORT.keys()),
            ["task-ECON", "run-1"],
            id="valid_decomposition",
        ),
    ],
)
def test_suggest_fix_for_segment_edge_cases(segment, known_keys, expected):
    """Test _suggest_fix_for_segment handles edge cases correctly."""
    assert _suggest_fix_for_segment(segment, known_keys) == expected


@pytest.mark.parametrize(
    "fname",
    [
        pytest.param("sub-01_task-ECONrun-_eeg.set", id="trailing_hyphen_with_ext"),
        pytest.param(
            "sub-01_task-myexprun-_bold.nii.gz", id="trailing_hyphen_compound_ext"
        ),
    ],
)
def test_get_entities_from_fname_trailing_hyphen(fname):
    """Test that trailing-hyphen segments warn without suggesting a fix."""
    # on_error="raise" should raise about multiple hyphens, not suggest a fix
    with pytest.raises(ValueError, match="multiple hyphens") as exc_info:
        get_entities_from_fname(fname, on_error="raise")
    assert "Suggested fix" not in str(exc_info.value)

    # on_error="warn" should warn without suggesting a fix
    with pytest.warns(RuntimeWarning, match="multiple hyphens") as warnings:
        params = get_entities_from_fname(fname, on_error="warn")
    messages = [str(w.message) for w in warnings]
    assert all("Suggested fix" not in msg for msg in messages)
    # Should still parse what it can
    assert params["subject"] == "01"


@pytest.mark.parametrize(
    "candidate_list, best_candidates",
    [
        # Only one candidate
        (["sub-01_ses-02"], ["sub-01_ses-02"]),
        # Two candidates, but the second matches on more entities
        (["sub-01", "sub-01_ses-02"], ["sub-01_ses-02"]),
        # No candidates match
        (["sub-02_ses-02", "sub-01_ses-01"], []),
        # First candidate is disqualified (session doesn't match)
        (["sub-01_ses-01", "sub-01_ses-02"], ["sub-01_ses-02"]),
        # Multiple equally good candidates
        (["sub-01_run-1", "sub-01_run-2"], ["sub-01_run-1", "sub-01_run-2"]),
    ],
)
def test_find_best_candidates(candidate_list, best_candidates):
    """Test matching of candidate sidecar files."""
    params = dict(subject="01", session="02", acquisition=None)
    assert _find_best_candidates(params, candidate_list) == best_candidates


def test_find_matching_sidecar_basic(bids_root, tmp_path, path_counter, fast_sidecar):
    """Test finding a sidecar file from a BIDS dir."""
    bids_path = _bids_path.copy().update(root=bids_root)

    # Now find a sidecar
    expected_file = (
        bids_root / "sub-01" / "ses-01" / "meg" / "sub-01_ses-01_coordsystem.json"
    )
    sidecar_fname = bids_path.find_matching_sidecar(
        suffix="coordsystem", extension=".json"
    )
    assert sidecar_fname == expected_file
    assert len(path_counter.calls) == 1
    assert path_counter.count == 1

    # create a duplicate sidecar, which will be tied in match score, triggering an error
    dupe = Path(str(sidecar_fname).replace("coordsystem.json", "2coordsystem.json"))
    dupe.touch()
    with pytest.raises(RuntimeError, match="Expected to find a single"):
        print_dir_tree(bids_root)
        bids_path.find_matching_sidecar(suffix="coordsystem", extension=".json")
    dupe.unlink()  # clean up extra file

    # Find nothing and raise.
    with pytest.raises(RuntimeError, match="Did not find any"):
        bids_path.find_matching_sidecar(suffix="foo", extension=".bogus")

    # Find nothing and receive None and a warning.
    on_error = "warn"
    with pytest.warns(RuntimeWarning, match="Did not find any"):
        fname = bids_path.find_matching_sidecar(
            suffix="foo", extension=".bogus", on_error=on_error
        )
    assert fname is None

    # Find nothing and receive None.
    on_error = "ignore"
    fname = bids_path.find_matching_sidecar(
        suffix="foo", extension=".bogus", on_error=on_error
    )
    assert fname is None

    # Invalid on_error.
    on_error = "hello"
    with pytest.raises(ValueError, match="Acceptable values for on_error are"):
        bids_path.find_matching_sidecar(
            suffix="coordsystem", extension=".json", on_error=on_error
        )

    # Test behavior of suffix and extension params when suffix and extension
    # are also (not) present in the passed BIDSPath
    bids_path = BIDSPath(subject="test", task="task", datatype="eeg", root=tmp_path)
    bids_path.mkdir()

    for suffix, extension in zip(
        ["eeg", "eeg", "events", "events"], [".fif", ".json", ".tsv", ".json"]
    ):
        bids_path.suffix = suffix
        bids_path.extension = extension
        bids_path.fpath.touch()

    # suffix parameter should always override BIDSPath.suffix
    bids_path.extension = ".json"

    for bp_suffix in (None, "eeg"):
        bids_path.suffix = bp_suffix
        s = bids_path.find_matching_sidecar(suffix="events")
        assert Path(s).name == "sub-test_task-task_events.json"

    # extension parameter should always override BIDSPath.extension
    bids_path.suffix = "events"

    for bp_extension in (None, ".json"):
        bids_path.extension = bp_extension
        s = bids_path.find_matching_sidecar(extension=".tsv")
        assert Path(s).name == "sub-test_task-task_events.tsv"

    # If suffix and extension parameters are not passed, use BIDSPath
    # attributes
    bids_path.update(suffix="events", extension=".tsv")
    s = bids_path.find_matching_sidecar()
    assert s.name == "sub-test_task-task_events.tsv"

    # Degenerate case
    bad_path = BIDSPath(subject="foo")
    with pytest.raises(ValueError, match="The root"):
        bad_path.find_matching_sidecar()


@pytest.fixture(params=[True, False], ids=["fast", "slow"])
def fast_sidecar(request, monkeypatch):
    """Fixture to test both fast and slow sidecar finding."""
    fast = request.param
    if not fast:
        monkeypatch.setattr(
            mne_bids.path,
            "_find_matching_sidecar_shortcut",
            lambda *args, **kwargs: None,
        )
    return fast


@pytest.mark.parametrize("dataset", ("basic", "dense"))
def test_find_matching_sidecar_fast(
    bids_root, bids_root_dense, path_counter, fast_sidecar, dataset
):
    """Test that we can find sidecars with the fast shortcut method."""
    if dataset == "basic":
        path = BIDSPath(
            subject="01",
            session="01",
            run="01",
            task="testing",
            root=bids_root,
        )
        datatype = "meg"
        call_count = 2
    else:
        path = BIDSPath(
            subject="1",
            session="1",
            task="audvis",
            root=bids_root_dense.root,
        )
        datatype = "eeg"
        call_count = 1
    del bids_root, bids_root_dense
    path.update(datatype=datatype)
    assert path.fpath.is_file()

    for ii, (suffix, extension, slow_count) in enumerate(
        (
            ("events", ".tsv", call_count),
            ("channels", ".tsv", call_count),
            ("electrodes", ".tsv", 1),
            ("coordsystem", ".json", 1),
            (datatype, ".json", call_count),
            ("scans", ".tsv", 1),
        )
    ):
        if suffix == "electrodes" and datatype != "eeg":
            with pytest.raises(RuntimeError, match="Did not find any"):
                path.find_matching_sidecar(suffix=suffix, extension=extension)
            continue
        sidecar_path = path.find_matching_sidecar(suffix=suffix, extension=extension)
        assert sidecar_path.is_file(), suffix
        assert sidecar_path.name.endswith(f"{suffix}{extension}"), suffix
        assert len(path_counter.calls) == ii + 1, suffix
        if fast_sidecar:
            want_count = 1 if suffix in ("coordsystem", "electrodes") else 0
        else:
            want_count = slow_count
        assert path_counter.count == want_count, suffix


def test_find_matching_sidecar_at_root(tmp_path, path_counter, fast_sidecar):
    """Test finding a sidecar file located at dataset root level.

    BIDS inheritance principle allows sidecars at dataset root that apply
    to all subjects. See https://github.com/mne-tools/mne-bids/issues/1488
    """
    # Create a minimal BIDS structure with a root-level sidecar
    bids_root = tmp_path / "bids_root"
    bids_root.mkdir()

    # Create subject directory structure
    sub_dir = bids_root / "sub-01" / "eeg"
    sub_dir.mkdir(parents=True)

    # Create a data file placeholder
    data_file = sub_dir / "sub-01_task-rest_eeg.vhdr"
    data_file.touch()

    # Create a root-level coordsystem.json (no subject entity in filename)
    root_coordsystem = bids_root / "coordsystem.json"
    root_coordsystem.write_text('{"EEGCoordinateSystem": "CapTrak"}')

    # Now try to find the sidecar from a BIDSPath
    bids_path = BIDSPath(subject="01", task="rest", datatype="eeg", root=bids_root)
    sidecar = bids_path.find_matching_sidecar(suffix="coordsystem", extension=".json")
    assert sidecar == root_coordsystem
    assert len(path_counter.calls) == 1
    assert path_counter.count == 1

    # Test that subject-level sidecar takes precedence over root-level
    sub_coordsystem = sub_dir / "sub-01_coordsystem.json"
    sub_coordsystem.write_text('{"EEGCoordinateSystem": "Other"}')
    sidecar = bids_path.find_matching_sidecar(suffix="coordsystem", extension=".json")
    assert sidecar == sub_coordsystem
    assert len(path_counter.calls) == 2
    assert path_counter.count == 1

    # Clean up subject-level sidecar and test root-level again
    sub_coordsystem.unlink()
    sidecar = bids_path.find_matching_sidecar(suffix="coordsystem", extension=".json")
    assert sidecar == root_coordsystem
    assert len(path_counter.calls) == 3
    assert path_counter.count == 1

    # Test with task-specific root-level sidecar
    task_coordsystem = bids_root / "task-rest_coordsystem.json"
    task_coordsystem.write_text('{"EEGCoordinateSystem": "TaskSpecific"}')
    sidecar = bids_path.find_matching_sidecar(suffix="coordsystem", extension=".json")
    # task-rest_coordsystem.json should match better than coordsystem.json
    assert sidecar == task_coordsystem
    assert len(path_counter.calls) == 4
    assert path_counter.count == 2


def test_bids_path_inference(bids_root):
    """Test usage of BIDSPath object and fpath."""
    # without providing all the entities, ambiguous when trying
    # to use fpath
    bids_path = BIDSPath(
        subject=subject_id,
        session=session_id,
        acquisition=acq,
        task=task,
        root=bids_root,
    )
    with pytest.raises(RuntimeError, match="Found more than one"):
        bids_path.fpath

    # shouldn't error out when there is no uncertainty
    channels_fname = BIDSPath(
        subject=subject_id,
        session=session_id,
        run=run,
        acquisition=acq,
        task=task,
        root=bids_root,
        suffix="channels",
    )
    channels_fname.fpath

    # create an extra file under 'eeg'
    extra_file = (
        bids_root
        / f"sub-{subject_id}"
        / f"ses-{session_id}"
        / "eeg"
        / f"{channels_fname.basename}.tsv"
    )
    extra_file.parent.mkdir(exist_ok=True, parents=True)
    # Creates a new file and because of this new file, there is now
    # ambiguity
    with open(extra_file, "w", encoding="utf-8"):
        pass
    with pytest.raises(RuntimeError, match="Found data of more than one"):
        channels_fname.fpath

    # if you set datatype, now there is no ambiguity
    channels_fname.update(datatype="eeg")
    assert str(channels_fname.fpath) == str(extra_file)


def test_bids_path(bids_root):
    """Test usage of BIDSPath object."""
    bids_path = BIDSPath(
        subject=subject_id,
        session=session_id,
        run=run,
        acquisition=acq,
        task=task,
        root=bids_root,
        suffix="meg",
    )

    expected_parent_dir = op.join(
        bids_root, f"sub-{subject_id}", f"ses-{session_id}", "meg"
    )
    assert str(bids_path.fpath.parent) == expected_parent_dir

    # test BIDSPath without bids_root, suffix, extension
    # basename and fpath should be the same
    expected_basename = f"sub-{subject_id}_ses-{session_id}_task-{task}_run-{run}"
    assert op.basename(bids_path.fpath) == expected_basename + "_meg.fif"
    assert op.dirname(bids_path.fpath).startswith(str(bids_root))

    # when bids root is not passed in, passes relative path
    bids_path2 = bids_path.copy().update(datatype="meg", root=None)
    expected_relpath = op.join(
        f"sub-{subject_id}", f"ses-{session_id}", "meg", expected_basename + "_meg"
    )
    assert str(bids_path2.fpath) == expected_relpath

    # without bids_root and with suffix/extension
    # basename and fpath should be the same
    bids_path.update(suffix="ieeg", extension=".vhdr")
    expected_basename2 = expected_basename + "_ieeg.vhdr"
    assert bids_path.basename == expected_basename2
    bids_path.update(extension=".vhdr")
    assert bids_path.basename == expected_basename2

    # with bids_root, but without suffix/extension
    # basename should work, but fpath should not.
    bids_path.update(root=bids_root, suffix=None, extension=None)
    assert bids_path.basename == expected_basename

    # should find the correct filename if suffix was passed
    bids_path.update(suffix="meg", extension=".fif")
    bids_fpath = bids_path.fpath
    assert op.basename(bids_fpath) == bids_path.basename
    # Same test, but exploiting the fact that bids_fpath is a pathlib.Path
    assert bids_fpath.name == bids_path.basename

    # confirm BIDSPath assigns properties correctly
    bids_path = BIDSPath(subject=subject_id, session=session_id)
    assert bids_path.subject == subject_id
    assert bids_path.session == session_id
    assert "subject" in bids_path.entities
    assert "session" in bids_path.entities
    print(bids_path.entities)
    assert all(
        bids_path.entities.get(entity) is None
        for entity in [
            "task",
            "run",
            "recording",
            "acquisition",
            "space",
            "processing",
            "split",
            "root",
            "datatype",
            "suffix",
            "extension",
        ]
    )

    # test updating functionality
    bids_path.update(acquisition="03", run="2", session="02", task=None)
    assert bids_path.subject == subject_id
    assert bids_path.session == "02"
    assert bids_path.acquisition == "03"
    assert bids_path.run == "2"
    assert bids_path.task is None

    new_bids_path = bids_path.copy().update(task="02", acquisition=None)
    assert new_bids_path.task == "02"
    assert new_bids_path.acquisition is None

    # equality of bids basename
    assert new_bids_path != bids_path
    assert new_bids_path == bids_path.copy().update(task="02", acquisition=None)

    # error check on kwargs of update
    with pytest.raises(ValueError, match="Key must be one of*"):
        bids_path.update(sub=subject_id, session=session_id)

    # error check on the passed in entity containing a magic char
    with pytest.raises(ValueError, match="Unallowed*"):
        bids_path.update(subject=subject_id + "-")

    # error check on suffix in BIDSPath (deep check)
    suffix = "meeg"
    with pytest.raises(ValueError, match=f"Suffix {suffix} is not"):
        BIDSPath(subject=subject_id, session=session_id, suffix=suffix)

    # do error check suffix in update
    error_kind = "foobar"
    with pytest.raises(ValueError, match=f"Suffix {error_kind} is not"):
        bids_path.update(suffix=error_kind)

    # does not error check on suffix in BIDSPath (deep check)
    suffix = "meeg"
    bids_path = BIDSPath(
        subject=subject_id, session=session_id, suffix=suffix, check=False
    )

    # also inherits error check from instantiation
    # always error check entities though
    with pytest.raises(ValueError, match="Key must be one of"):
        bids_path.copy().update(blah="blah-entity")

    # error check datatype if check is turned back on
    with pytest.raises(ValueError, match="datatype .* is not valid"):
        bids_path.copy().update(check=True, datatype=error_kind)

    # does not error check on space if check=False ...
    BIDSPath(subject=subject_id, space="foo", suffix="eeg", check=False)

    # ... but raises an error with check=True
    match = r"space \(foo\) is not valid for datatype \(eeg\)"
    with pytest.raises(ValueError, match=match):
        BIDSPath(subject=subject_id, space="foo", suffix="eeg", datatype="eeg")

    # error check on space for datatypes that do not support space
    match = "space entity is not valid for datatype anat"
    with pytest.raises(ValueError, match=match):
        BIDSPath(subject=subject_id, space="foo", datatype="anat")

    # error check on space if datatype is None
    bids_path_tmpcopy = bids_path.copy().update(suffix="meeg")
    match = "You must define datatype if you want to use space"
    with pytest.raises(ValueError, match=match):
        bids_path_tmpcopy.update(space="CapTrak", check=True)

    # making a valid space update works
    bids_path_tmpcopy.update(suffix="eeg", datatype="eeg", space="CapTrak", check=True)

    # suffix won't be error checks if initial check was false
    bids_path.update(suffix=suffix)

    # error check on extension in BIDSPath (deep check)
    extension = ".mat"
    with pytest.raises(ValueError, match=f"Extension {extension} is not"):
        BIDSPath(subject=subject_id, session=session_id, extension=extension)

    # do not error check extension in update (not deep check)
    bids_path.update(extension=".foo")

    # test repr
    bids_path = BIDSPath(
        subject="01",
        session="02",
        task="03",
        suffix="ieeg",
        datatype="ieeg",
        extension=".edf",
    )
    assert repr(bids_path) == (
        "BIDSPath(\n"
        "root: None\n"
        "datatype: ieeg\n"
        "basename: sub-01_ses-02_task-03_ieeg.edf)"
    )

    # test update can change check
    bids_path.update(check=False)
    bids_path.update(extension=".mat")

    # test that split gets properly set
    bids_path.update(split=1)
    assert bids_path.basename == "sub-01_ses-02_task-03_split-1_ieeg.mat"

    # test home dir expansion
    bids_path = BIDSPath(root="~/foo")
    assert "~/foo" not in str(bids_path.root)
    # explicitly test update() method too
    bids_path.update(root="~/foo")
    assert "~/foo" not in str(bids_path.root)

    # Test property setters
    bids_path = BIDSPath(subject="01", task="noise", datatype="eeg")

    for entity in (
        "subject",
        "session",
        "task",
        "run",
        "acquisition",
        "processing",
        "recording",
        "space",
        "suffix",
        "extension",
        "datatype",
        "root",
        "split",
    ):
        if entity == "run":
            new_val = "01"
        elif entity == "space":
            new_val = "CapTrak"
        elif entity in ["suffix", "datatype"]:
            new_val = "eeg"
        elif entity == "extension":
            new_val = ".fif"
        elif entity == "root":
            new_val = Path("foo")
        elif entity == "split":
            new_val = "01"
        else:
            new_val = "foo"

        setattr(bids_path, entity, new_val)
        assert getattr(bids_path, entity) == new_val


def test_make_filenames():
    """Test that we create filenames according to the BIDS spec."""
    # All keys work
    prefix_data = dict(
        subject="one",
        session="two",
        task="three",
        acquisition="four",
        run=1,
        processing="six",
        recording="seven",
        suffix="ieeg",
        extension=".json",
        datatype="ieeg",
    )
    expected_str = (
        "sub-one_ses-two_task-three_acq-four_run-1_proc-six_recording-seven_ieeg.json"
    )
    assert BIDSPath(**prefix_data).basename == expected_str
    assert (
        BIDSPath(**prefix_data)
        == (Path("sub-one") / "ses-two" / "ieeg" / expected_str).as_posix()
    )

    # subsets of keys works
    assert (
        BIDSPath(subject="one", task="three", run=4).basename
        == "sub-one_task-three_run-4"
    )
    assert (
        BIDSPath(subject="one", task="three", suffix="meg", extension=".json").basename
        == "sub-one_task-three_meg.json"
    )

    with pytest.raises(ValueError):
        BIDSPath(subject="one-two", suffix="ieeg", extension=".edf")

    with pytest.raises(ValueError, match="At least one"):
        BIDSPath()

    # emptyroom check: invalid task
    with pytest.raises(ValueError, match="task must be"):
        BIDSPath(subject="emptyroom", session="20131201", task="blah", suffix="meg")

    # when the suffix is not 'meg', then it does not result in
    # an error
    BIDSPath(subject="emptyroom", session="20131201", task="blah")

    # test what would happen if you don't want to check
    prefix_data["extension"] = ".h5"
    with pytest.raises(ValueError, match="Extension .h5 is not allowed"):
        BIDSPath(**prefix_data)
    basename = BIDSPath(**prefix_data, check=False)
    assert (
        basename.basename
        == "sub-one_ses-two_task-three_acq-four_run-1_proc-six_recording-seven_ieeg.h5"
    )

    # what happens with scans.tsv file
    with pytest.raises(ValueError, match="scans.tsv file name can only contain"):
        BIDSPath(
            subject=subject_id,
            session=session_id,
            task=task,
            suffix="scans",
            extension=".tsv",
        )

    # We should be able to create a BIDSPath for a *_sessions.tsv file
    BIDSPath(subject=subject_id, suffix="sessions", extension=".tsv")


@pytest.mark.parametrize(
    "entities, expected_n_matches",
    [
        (dict(), 9),
        (dict(subject="01"), 2),
        (dict(task="audio"), 2),
        (dict(processing="sss"), 1),
        (dict(suffix="meg"), 4),
        (dict(acquisition="lowres"), 1),
        (dict(task="test", processing="ica", suffix="eeg"), 2),
        (dict(subject="5", task="test", processing="ica", suffix="eeg"), 1),
        (dict(subject=["01", "02"]), 3),  # test multiple input
    ],
)
def test_filter_fnames(entities, expected_n_matches):
    """Test filtering filenames based on BIDS entities works."""
    fnames = (
        "sub-01_task-audio_meg.fif",
        "sub-01_ses-05_task-audio_meg.fif",
        "sub-02_task-visual_eeg.vhdr",
        "sub-Foo_ses-bar_meg.fif",
        "sub-Bar_task-invasive_run-1_ieeg.fif",
        "sub-3_task-fun_proc-sss_meg.fif",
        "sub-3_task-fun_proc-sss_meg_channels.tsv",
        "sub-4_task-pain_acq-lowres_T1w.nii.gz",
        "sub-5_task-test_proc-ica_eeg.vhdr",
        "sub-6_task-test_proc-ica_eeg.vhdr",
    )

    output = _filter_fnames(fnames, **entities)
    assert len(output) == expected_n_matches


def test_match_basic(bids_root):
    """Test retrieval of matching basenames."""
    bids_path_01 = BIDSPath(root=bids_root)
    paths = bids_path_01.match()
    assert len(paths) == 9
    assert all("sub-01_ses-01" in p.basename for p in paths)
    assert all([p.root == bids_root for p in paths])

    bids_path_01 = BIDSPath(root=bids_root, run="01")
    paths = bids_path_01.match()
    assert len(paths) == 3
    assert paths[0].basename == "sub-01_ses-01_task-testing_run-01_channels.tsv"

    bids_path_01 = BIDSPath(root=bids_root, subject="unknown")
    paths = bids_path_01.match()
    assert len(paths) == 0

    bids_path_01 = _bids_path.copy().update(root=None)
    with pytest.raises(RuntimeError, match="Cannot match"):
        bids_path_01.match()

    bids_path_01.update(datatype="meg", root=bids_root)
    same_paths = bids_path_01.match()
    assert len(same_paths) == 3

    # Check handling of `extension`, part 1: no extension specified.
    bids_path_01 = BIDSPath(root=bids_root, run="01")
    paths = bids_path_01.match()
    assert [p.extension for p in paths] == [".tsv", ".tsv", ".fif"]

    # Check handling of `extension`, part 2: extension specified.
    bids_path_01 = BIDSPath(root=bids_root, run="01", extension=".fif", datatype="meg")
    paths = bids_path_01.match()
    assert len(paths) == 1
    assert paths[0].extension == ".fif"

    # Check handling of `extension` and `suffix`, part 1: no suffix
    bids_path_01 = BIDSPath(root=bids_root, run="01", extension=".tsv", datatype="meg")
    paths = bids_path_01.match()
    assert len(paths) == 2
    assert paths[0].extension == ".tsv"

    # Check handling of `extension` and `suffix`, part 1: suffix passed
    bids_path_01 = BIDSPath(
        root=bids_root, run="01", suffix="channels", extension=".tsv", datatype="meg"
    )
    paths = bids_path_01.match()
    assert len(paths) == 1
    assert paths[0].extension == ".tsv"
    assert paths[0].suffix == "channels"

    # Check handling of `datatype` when explicitly passed in
    print_dir_tree(bids_root)
    bids_path_01 = BIDSPath(
        root=bids_root, run="01", suffix="channels", extension=".tsv", datatype="meg"
    )
    paths = bids_path_01.match()
    assert len(paths) == 1
    assert paths[0].extension == ".tsv"
    assert paths[0].suffix == "channels"
    assert Path(paths[0]).parent.name == "meg"

    # Check handling of `datatype`, no datatype passed in
    # should be exactly the same if there is only one datatype
    # present in the dataset
    bids_path_01 = BIDSPath(
        root=bids_root, run="01", suffix="channels", extension=".tsv"
    )
    paths = bids_path_01.match()
    assert len(paths) == 1
    assert paths[0].extension == ".tsv"
    assert paths[0].suffix == "channels"
    assert Path(paths[0]).parent.name == "meg"

    # Test `check` parameter
    bids_path_01 = _bids_path.copy()
    bids_path_01.update(
        root=bids_root,
        session=None,
        task=None,
        run=None,
        suffix="foo",
        extension=".eeg",
        check=False,
    )
    bids_path_01.fpath.touch()

    assert bids_path_01.match(check=True) == []
    assert bids_path_01.match(check=False)[0].fpath.name == "sub-01_foo.eeg"
    bids_path_01.fpath.unlink()  # clean up created file


def test_match_advanced(tmp_path):
    """Test additional match functionality."""
    bids_root = tmp_path
    fnames = (
        "sub-01/nirs/sub-01_task-tapping_events.tsv",
        "sub-02/nirs/sub-02_task-tapping_events.tsv",
    )
    for fname in fnames:
        this_path = Path(bids_root / fname)
        this_path.parent.mkdir(parents=True, exist_ok=True)
        this_path.touch()
    path = BIDSPath(
        root=bids_root,
        datatype="nirs",
        suffix="events",
        extension=".tsv",
    )
    matches = path.match()
    assert len(matches) == len(fnames), path


def test_find_matching_paths(bids_root):
    """We test by yielding the same results as BIDSPath.match().

    BIDSPath.match() is extensively tested above.
    """
    # Check a few exemplary entities
    bids_path_01 = BIDSPath(root=bids_root)
    paths_match = bids_path_01.match(ignore_json=False)
    paths_find = find_matching_paths(bids_root)
    assert paths_match == paths_find

    # Datatype is important because handled differently
    bids_path_01 = BIDSPath(root=bids_root, datatype="meg")
    paths_match = bids_path_01.match(ignore_json=False)
    paths_find = find_matching_paths(bids_root, datatypes="meg")
    assert paths_match == paths_find

    bids_path_01 = BIDSPath(root=bids_root, run="02")
    paths_match = bids_path_01.match(ignore_json=False)
    paths_find = find_matching_paths(bids_root, runs="02")
    assert paths_match == paths_find

    # Check list of str as input
    bids_path_01 = BIDSPath(root=bids_root, extension=".tsv")
    bids_path_02 = BIDSPath(root=bids_root, extension=".json")
    paths_match1 = bids_path_01.match(ignore_json=False)
    paths_match2 = bids_path_02.match(ignore_json=False)
    paths_match = paths_match1 + paths_match2
    paths_match = sorted([str(f.fpath) for f in paths_match])
    paths_find = find_matching_paths(bids_root, extensions=[".tsv", ".json"])
    paths_find = sorted([str(f.fpath) for f in paths_find])
    assert paths_match == paths_find

    # Test ignore_json parameter
    bids_path_01 = BIDSPath(root=bids_root)
    paths_match = bids_path_01.match(ignore_json=True)
    paths_find = find_matching_paths(
        bids_root, extensions=[".tsv", ".fif", ".dat", ".eeg"]
    )
    assert paths_match == paths_find

    # Test `check` parameter
    bids_path_01 = _bids_path.copy()
    bids_path_01.update(
        root=bids_root,
        session=None,
        task=None,
        run=None,
        suffix="foo",
        extension=".eeg",
        check=False,
    )
    bids_path_01.fpath.touch()
    paths_match = bids_path_01.match(check=True)
    paths_find = find_matching_paths(
        bids_root,
        sessions=None,
        tasks=None,
        runs=None,
        suffixes="foo",
        extensions=".eeg",
        check=True,
    )
    assert paths_match == paths_find

    paths_match = bids_path_01.match(check=False)
    paths_find = find_matching_paths(
        bids_root,
        sessions=None,
        tasks=None,
        runs=None,
        suffixes="foo",
        extensions=".eeg",
        check=False,
    )
    assert paths_match == paths_find
    bids_path_01.fpath.unlink()  # clean up created file


def test_return_root_paths_entity_aware(tmp_path):
    """Test that `_return_root_paths` respects `entities['subject']`.

    Returns only paths under that subject when provided.

    This validates the entity-aware optimization added to reduce filesystem
    scanning when the subject (and optionally session) is known.
    """
    root = tmp_path / "bids"
    # Create two subjects each with meg and eeg directories and a file
    for subj in ("subjA", "subjB"):
        for dtype in ("meg", "eeg"):
            p = BIDSPath(subject=subj, session="sesA", datatype=dtype, root=root)
            p.mkdir(exist_ok=True)
            # create a predictable dummy data file so glob matches reliably
            ext = "fif" if dtype == "meg" else "edf"
            fname = f"sub-{subj}_ses-sesA_{dtype}.{ext}"
            (p.directory / fname).touch()

    # Full scan (no entities) should return files for both subjects
    # Pass root as string to match how glob.iglob expects root_dir
    root_str = str(root)
    # ensure directories exist
    assert (root / "sub-subjA").exists()
    all_paths = mne_bids.path._return_root_paths(
        root_str, datatype=("meg", "eeg"), ignore_json=True
    )
    assert len(all_paths) >= 2

    # Entity-aware scan for subjA should only return files under sub-subjA
    subj_paths = mne_bids.path._return_root_paths(
        root_str,
        datatype=("meg", "eeg"),
        ignore_json=True,
        entities={"subject": "subjA"},
    )
    assert len(subj_paths) < len(all_paths)
    assert all("sub-subjA" in str(p) for p in subj_paths)


@testing.requires_testing_data
@pytest.mark.filterwarnings(warning_str["meas_date_set_to_none"])
@pytest.mark.filterwarnings(warning_str["channel_unit_changed"])
def test_find_empty_room_basic(tmp_path):
    """Test reading of empty room data."""
    raw_fname = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif"
    bids_root = tmp_path / "bids"
    bids_root.mkdir()

    raw = _read_raw_fif(raw_fname)
    bids_path = BIDSPath(
        subject="01",
        session="01",
        task="audiovisual",
        run="01",
        root=bids_root,
        datatype="meg",
        suffix="meg",
    )
    with raw.info._unlock():
        raw.info["maxshield"] = True
    write_raw_bids(raw, bids_path, allow_preload=True, format="FIF")
    noroot = bids_path.copy().update(root=None)
    with pytest.raises(ValueError, match="The root of the"):
        noroot.find_empty_room()

    # No empty-room data present.
    er_basename = bids_path.find_empty_room()
    assert er_basename is None

    # Now create data resembling an empty-room recording.
    # The testing data has no "noise" recording, so save the actual data
    # as named as if it were noise. We first need to write the FIFF file
    # before reading it back in.
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    er_raw_fname = op.join(tmp_dir, "ernoise_raw.fif")
    raw.copy().crop(0, 10).save(er_raw_fname)
    er_raw = _read_raw_fif(er_raw_fname, allow_maxshield="yes").load_data()
    # set it as MaxShielded (ensure we don't warn about reading MS data in
    # find_empty_room)
    with er_raw.info._unlock():
        er_raw.info["maxshield"] = True

    er_date = er_raw.info["meas_date"].strftime("%Y%m%d")
    er_bids_path = BIDSPath(
        subject="emptyroom", task="noise", session=er_date, suffix="meg", root=bids_root
    )
    write_raw_bids(er_raw, er_bids_path, allow_preload=True, format="FIF")
    with pytest.raises(ValueError, match="Internal Active"):
        read_raw_fif(er_bids_path.fpath)

    recovered_er_bids_path = bids_path.find_empty_room()
    assert er_bids_path == recovered_er_bids_path

    # Test that when there is a noise task file in the subject directory it will take
    # precedence over the emptyroom directory file
    er_raw = _read_raw_fif(er_raw_fname, allow_maxshield="yes")
    er_noise_task_path = bids_path.copy().update(
        run=None,
        task="noise",
    )

    write_raw_bids(er_raw, er_noise_task_path, overwrite=True, verbose=False)
    recovered_er_bids_path = bids_path.find_empty_room()
    assert er_noise_task_path == recovered_er_bids_path
    er_noise_task_path.fpath.unlink()

    # When a split empty room file is present, the first split should be returned as
    # the matching empty room file
    split_er_bids_path = er_noise_task_path.copy().update(split="01", extension=".fif")
    split_er_bids_path.fpath.touch()
    split_er_bids_path2 = split_er_bids_path.copy().update(
        split="02", extension=".fif"
    )  # not used
    split_er_bids_path2.fpath.touch()
    recovered_er_bids_path = bids_path.find_empty_room()
    assert split_er_bids_path == recovered_er_bids_path
    split_er_bids_path.fpath.unlink()
    split_er_bids_path2.fpath.unlink()
    write_raw_bids(er_raw, er_noise_task_path, overwrite=True, verbose=False)

    # Check that when there are multiple matches that cannot be resolved via assigning
    # split=01 that the sub-emptyroom is the fallback
    dup_noise_task_path = er_noise_task_path.copy()
    dup_noise_task_path.update(run="100", split=None)
    write_raw_bids(er_raw, dup_noise_task_path, overwrite=True, verbose=False)
    write_raw_bids(er_raw, er_bids_path, overwrite=True, verbose=False)
    with pytest.warns(RuntimeWarning):
        recovered_er_bids_path = bids_path.find_empty_room()
    assert er_bids_path == recovered_er_bids_path
    er_noise_task_path.fpath.unlink()
    dup_noise_task_path.fpath.unlink()

    # assert that we get best emptyroom if there are multiple available
    sh.rmtree(op.join(bids_root, "sub-emptyroom"))
    dates = ["20021204", "20021201", "20021001"]
    for date in dates:
        er_bids_path.update(session=date)
        er_meas_date = datetime.strptime(date, "%Y%m%d")
        er_meas_date = er_meas_date.replace(tzinfo=UTC)
        er_raw.set_meas_date(er_meas_date)
        write_raw_bids(er_raw, er_bids_path, verbose=False)

    best_er_basename = bids_path.find_empty_room()
    assert best_er_basename.session == "20021204"

    with pytest.raises(ValueError, match='The root of the "bids_path" must be set'):
        bids_path.copy().update(root=None).find_empty_room()

    # assert that we get an error if meas_date is not available.
    extra_params = dict(allow_maxshield="yes")
    raw = read_raw_bids(bids_path=bids_path, extra_params=extra_params)
    raw.set_meas_date(None)
    anonymize_info(raw.info)
    write_raw_bids(raw, bids_path, overwrite=True, format="FIF")
    with pytest.raises(
        ValueError,
        match="The provided recording does not have a measurement date set",
    ):
        bids_path.find_empty_room()

    # test that the `AssociatedEmptyRoom` key in MEG sidecar is respected
    bids_root = tmp_path / "associated-empty-room"
    bids_root.mkdir()
    raw = _read_raw_fif(raw_fname)
    meas_date = datetime(year=2020, month=1, day=10, tzinfo=UTC)
    er_date = datetime(year=2010, month=1, day=1, tzinfo=UTC)
    raw.set_meas_date(meas_date)

    er_raw_matching_date = er_raw.copy().set_meas_date(meas_date)
    er_raw_associated = er_raw.copy().set_meas_date(er_date)

    # First write empty-room data
    # We write two empty-room recordings: one with a date matching exactly the
    # experimental measurement date, and one dated approx. 10 years earlier
    # We will want to enforce using the older recording via
    # `AssociatedEmptyRoom` (without AssociatedEmptyRoom, find_empty_room()
    # would return the recording with the matching date instead)
    er_matching_date_bids_path = BIDSPath(
        subject="emptyroom",
        session="20200110",
        task="noise",
        root=bids_root,
        datatype="meg",
        suffix="meg",
        extension=".fif",
    )
    write_raw_bids(er_raw_matching_date, bids_path=er_matching_date_bids_path)

    er_associated_bids_path = er_matching_date_bids_path.copy().update(
        session="20100101"
    )
    write_raw_bids(er_raw_associated, bids_path=er_associated_bids_path)

    # Now we write experimental data and associate it with the earlier
    # empty-room recording
    with pytest.raises(RuntimeError, match="Did not find any"):
        bids_path.find_matching_sidecar()
    bids_path = er_matching_date_bids_path.copy().update(
        subject="01", session=None, task="task"
    )
    write_raw_bids(raw, bids_path=bids_path, empty_room=er_associated_bids_path)
    assert bids_path.find_matching_sidecar().is_file()

    # Retrieve empty-room BIDSPath
    assert bids_path.find_empty_room() == er_associated_bids_path
    for use_sidecar_only in [True, False]:  # same result either way
        path = bids_path.find_empty_room(use_sidecar_only=use_sidecar_only)
        assert path == er_associated_bids_path
    candidates = bids_path.get_empty_room_candidates()
    assert len(candidates) == 2

    # Should only work for MEG
    with pytest.raises(ValueError, match="only supported for MEG"):
        bids_path.copy().update(datatype="eeg").find_empty_room()

    # Raises an error if the file is missing
    os.remove(er_associated_bids_path.fpath)
    with pytest.raises(FileNotFoundError, match="Empty-room BIDS .* not foun"):
        bids_path.find_empty_room(use_sidecar_only=True)

    # Don't create `AssociatedEmptyRoom` entry in sidecar – we should now
    # retrieve the empty-room recording closer in time
    write_raw_bids(raw, bids_path=bids_path, empty_room=None, overwrite=True)
    path = bids_path.find_empty_room()
    candidates = bids_path.get_empty_room_candidates()
    assert path == er_matching_date_bids_path
    assert er_matching_date_bids_path in candidates

    # If we enforce searching only via `AssociatedEmptyRoom`, we should get no
    # result
    assert bids_path.find_empty_room(use_sidecar_only=True) is None


@pytest.mark.filterwarnings(warning_str["channel_unit_changed"])
@testing.requires_testing_data
def test_find_emptyroom_ties(tmp_path):
    """Test that we receive a warning on a date tie."""
    raw_fname = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif"

    bids_root = str(tmp_path)
    bids_path = _bids_path.copy().update(root=bids_root, datatype="meg")
    session = "20010101"
    er_dir_path = BIDSPath(
        subject="emptyroom", session=session, datatype="meg", root=bids_root
    )
    er_dir = er_dir_path.mkdir().directory

    meas_date = datetime.strptime(session, "%Y%m%d").replace(tzinfo=UTC)

    raw = _read_raw_fif(raw_fname)

    er_raw_fname = data_path / "MEG" / "sample" / "ernoise_raw.fif"
    raw.copy().crop(0, 10).save(er_raw_fname, overwrite=True)
    er_raw = _read_raw_fif(er_raw_fname)
    raw.set_meas_date(meas_date)
    er_raw.set_meas_date(meas_date)

    write_raw_bids(raw, bids_path, overwrite=True)
    er_bids_path = BIDSPath(subject="emptyroom", session=session)
    er_basename_1 = er_bids_path.basename
    er_basename_2 = BIDSPath(
        subject="emptyroom", session=session, task="noise"
    ).basename
    er_raw.save(op.join(er_dir, f"{er_basename_1}_meg.fif"))
    er_raw.save(op.join(er_dir, f"{er_basename_2}_meg.fif"))

    with pytest.warns(RuntimeWarning, match="Found more than one"):
        bids_path.find_empty_room()


@pytest.mark.filterwarnings(warning_str["channel_unit_changed"])
@testing.requires_testing_data
def test_find_emptyroom_no_meas_date(tmp_path):
    """Test that we warn if measurement date can be read or inferred."""
    raw_fname = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif"

    bids_root = str(tmp_path)
    bids_path = _bids_path.copy().update(root=bids_root)
    er_session = "mysession"
    er_meas_date = None

    er_dir_path = BIDSPath(
        subject="emptyroom", session=er_session, datatype="meg", root=bids_root
    )
    er_dir = er_dir_path.mkdir().directory

    er_bids_path = BIDSPath(
        subject="emptyroom", session=er_session, task="noise", check=False
    )
    er_basename = er_bids_path.basename
    raw = _read_raw_fif(raw_fname)

    er_raw_fname = data_path / "MEG" / "sample" / "ernoise_raw.fif"
    raw.copy().crop(0, 10).save(er_raw_fname, overwrite=True)
    er_raw = _read_raw_fif(er_raw_fname)
    er_raw.set_meas_date(er_meas_date)
    er_raw.save(op.join(er_dir, f"{er_basename}_meg.fif"), overwrite=True)

    # Write raw file data using mne-bids, and remove participants.tsv
    # as it's incomplete (doesn't contain the emptyroom subject we wrote
    # manually using MNE's Raw.save() above)
    raw = _read_raw_fif(raw_fname)
    write_raw_bids(raw, bids_path, overwrite=True)
    os.remove(op.join(bids_root, "participants.tsv"))

    with (
        pytest.warns(RuntimeWarning, match="Could not retrieve .* date"),
        pytest.warns(RuntimeWarning, match="participants.tsv file not found"),
        pytest.warns(RuntimeWarning, match=r"Did not find any channels\.tsv"),
        pytest.warns(RuntimeWarning, match=r"Did not find any meg\.json"),
    ):
        bids_path.find_empty_room()


def test_bids_path_label_vs_index_entity():
    """Test entities that must be strings vs those that may be an int."""
    match = "subject must be an instance of None or str"
    with pytest.raises(TypeError, match=match):
        BIDSPath(subject=1)
    match = "root must be an instance of path-like or None"
    with pytest.raises(TypeError, match=match):
        BIDSPath(root=1, subject="01")
    BIDSPath(subject="01", run=1)  # ok as <index> entity
    BIDSPath(subject="01", split=1)  # ok as <index> entity


def test_meg_calibration_fpath(bids_root, tmp_path):
    """Test BIDSPath.meg_calibration_fpath."""
    # File exists, so BIDSPath.meg_calibration_fpath should return a non-None
    # value.
    bids_path_ = _bids_path.copy().update(subject="01", root=bids_root)
    assert bids_path_.meg_calibration_fpath is not None

    # subject not set.
    bids_path_ = _bids_path.copy().update(root=bids_root, subject=None)
    with pytest.raises(ValueError, match="root and subject must be set"):
        bids_path_.meg_calibration_fpath

    # root not set.
    bids_path_ = _bids_path.copy().update(subject="01", root=None)
    with pytest.raises(ValueError, match="root and subject must be set"):
        bids_path_.meg_calibration_fpath

    # datatype is not 'meg''.
    bids_path_ = _bids_path.copy().update(subject="01", root=bids_root, datatype="eeg")
    with pytest.raises(ValueError, match="Can only find .* for MEG"):
        bids_path_.meg_calibration_fpath

    # Move the fine-calibration file. BIDSPath.meg_calibration_fpath should then be None
    bids_path_ = _bids_path.copy().update(subject="01", root=bids_root)
    src = Path(bids_path_.meg_calibration_fpath)
    src.rename(tmp_path / src.name)
    assert bids_path_.meg_calibration_fpath is None
    # restore the file
    (tmp_path / src.name).rename(src)
    assert bids_path_.meg_calibration_fpath is not None


def test_meg_crosstalk_fpath(bids_root, tmp_path):
    """Test BIDSPath.meg_crosstalk_fpath."""
    # File exists, so BIDSPath.crosstalk_fpath should return a non-None
    # value.
    bids_path = _bids_path.copy().update(subject="01", root=bids_root)
    assert bids_path.meg_crosstalk_fpath is not None

    # subject not set.
    bids_path = _bids_path.copy().update(root=bids_root, subject=None)
    with pytest.raises(ValueError, match="root and subject must be set"):
        bids_path.meg_crosstalk_fpath

    # root not set.
    bids_path = _bids_path.copy().update(subject="01", root=None)
    with pytest.raises(ValueError, match="root and subject must be set"):
        bids_path.meg_crosstalk_fpath

    # datatype is not 'meg''.
    bids_path = _bids_path.copy().update(subject="01", root=bids_root, datatype="eeg")
    with pytest.raises(ValueError, match="Can only find .* for MEG"):
        bids_path.meg_crosstalk_fpath

    # Move the crosstalk file. BIDSPath.meg_crosstalk_fpath should then be None.
    bids_path = _bids_path.copy().update(subject="01", root=bids_root)
    src = Path(bids_path.meg_crosstalk_fpath)
    src.rename(tmp_path / src.name)
    assert bids_path.meg_crosstalk_fpath is None
    # restore the file
    (tmp_path / src.name).rename(src)
    assert bids_path.meg_crosstalk_fpath is not None


def test_datasetdescription_with_bidspath(bids_root):
    """Test a BIDSPath can generate a valid path to dataset_description.json."""
    with pytest.raises(ValueError, match="Unallowed"):
        bids_path = BIDSPath(
            root=bids_root, suffix="dataset_description", extension=".json"
        )

    # initialization should work
    bids_path = BIDSPath(
        root=bids_root,
        suffix="dataset_description",
        extension=".json",
        check=False,
    )
    assert (
        bids_path.fpath.as_posix()
        == Path(f"{bids_root}/dataset_description.json").as_posix()
    )

    # setting it via update should work
    bids_path = BIDSPath(root=bids_root, extension=".json", check=True)
    bids_path.update(suffix="dataset_description", check=False)
    assert (
        bids_path.fpath.as_posix()
        == Path(f"{bids_root}/dataset_description.json").as_posix()
    )


def test_update_check(bids_root):
    """Test check argument is passed BIDSPath properly."""
    bids_path = BIDSPath(root=bids_root, check=False)
    bids_path.update(datatype="eyetrack")
    assert bids_path.fpath.as_posix() == (bids_root / "eyetrack").as_posix()


def test_update_fail_check_no_change():
    """Test BIDSPath.check works in preventing invalid changes."""
    bids_path = BIDSPath(subject="test")
    try:
        bids_path.update(suffix="ave")
    except Exception:
        pass
    assert bids_path.suffix is None


def test_setting_entities():
    """Test setting entities via assignment."""
    bids_path = BIDSPath(subject="test", check=False)
    for entity_name in bids_path.entities:
        if entity_name in ["dataype", "suffix"]:
            continue

        if entity_name in ["run", "split"]:
            value = "1"
        else:
            value = "foo"

        setattr(bids_path, entity_name, value)
        assert getattr(bids_path, entity_name) == value

        setattr(bids_path, entity_name, None)
        assert getattr(bids_path, entity_name) is None


def test_dont_create_dirs_on_fpath_access(tmp_path):
    """Regression test: don't create directories when accessing .fpath."""
    bp = BIDSPath(subject="01", datatype="eeg", root=tmp_path)
    bp.fpath  # accessing .fpath is required for this regression test
    assert not (tmp_path / "sub-01").exists()


def test_fpath_common_prefix(tmp_path):
    """Tests that fpath does not match multiple files with the same prefix.

    This might happen if indices are not zero-paddded.
    """
    sub_dir = tmp_path / "sub-1" / "eeg"
    sub_dir.mkdir(exist_ok=True, parents=True)
    (sub_dir / "sub-1_run-1_raw.fif").touch()
    (sub_dir / "sub-1_run-2.edf").touch()
    # Other valid BIDS paths with the same basename prefix:
    (sub_dir / "sub-1_run-10_raw.fif").touch()
    (sub_dir / "sub-1_run-20.edf").touch()

    assert (
        BIDSPath(root=tmp_path, subject="1", run="1").fpath
        == sub_dir / "sub-1_run-1_raw.fif"
    )
    assert (
        BIDSPath(root=tmp_path, subject="1", run="2").fpath
        == sub_dir / "sub-1_run-2.edf"
    )


def test_hash():
    """Test that BIDSPath is hashable."""
    bp1 = BIDSPath(subject="01", datatype="eeg", root=Path("foo"))
    bp2 = BIDSPath(subject="01", datatype="eeg", root=Path("foo"))
    # test standard hash properties:
    assert hash(bp1) == hash(bp1)  # same object
    assert hash(bp1) == hash(bp2)  # different objects, same content
    bp_other = bp1.copy().update(extension=".fif")
    assert bp1.extension != bp_other.extension
    assert hash(bp1) != hash(bp_other)  # different content
    assert hash(bp1) != hash("not a BIDSPath")  # different type
    assert bp1 == bp2  # equality should be true for same content
    bp3 = pickle.loads(pickle.dumps(bp1))  # test pickling
    assert bp1 == bp3
    assert hash(bp1) == hash(bp3)
    # equality and pickling
    assert bp1.__dict__ != bp_other.__dict__  # different content
    assert bp1.fpath == bp_other.fpath  # okay, I guess
    assert bp1 == bp_other
    bp_other.datatype = "meg"
    assert bp1 != bp_other
    assert bp1 != "foo"  # other type
    # make sure everything in our entities plus properties is in __getstate__
    slim_dict = {k: v for k, v in bp1.__dict__.items() if not k.startswith("_")}
    slim_dict.update(bp1.entities)
    slim_dict.update(
        {k: getattr(bp1, k) for k in ("datatype", "extension", "suffix", "root")}
    )
    assert bp1.__getstate__() == slim_dict
    functools.lru_cache(lambda x, y: x)(bp1, "whatever")  # test cachable
    # TODO: One more oddity about our equality that we maybe want to fix someday
    bp3 = BIDSPath(subject="01", datatype="eeg", root=Path("foo"), extension=".fif")
    bp4 = BIDSPath(subject="01", datatype="eeg", root=Path("foo"), extension=".bdf")
    # equality is true even though extension differs because .fpath omits it here
    assert bp3 == bp4
    assert "." not in str(bp3.fpath)
