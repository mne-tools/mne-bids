import glob
import os
import re
import shutil as sh
from collections import OrderedDict
from copy import deepcopy
from os import path as op
from pathlib import Path

from mne.utils import warn, logger

from mne_bids.config import BIDS_PATH_ENTITIES, reader
from mne_bids.utils import (_check_key_val, _check_empty_room_basename, _check_types,
                            _path_to_str, param_regex, _ensure_tuple)


class BIDSPath(object):
    """Create a partial/full BIDS filepath from its component parts.

    BIDS filename prefixes have one or more pieces of metadata in them. They
    must follow a particular order, which is followed by this function. This
    will generate the *prefix* for a BIDS filename that can be used with many
    subsequent files, or you may also give a suffix that will then complete
    the file name.

    BIDSPath allows dynamic updating of its entities in place, and operates
    similar to `pathlib.Path`.

    Note that not all parameters are applicable to each kind of data. For
    example, electrode location TSV files do not need a "task" field.

    Parameters
    ----------
    subject : str | None
        The subject ID. Corresponds to "sub".
    session : str | None
        The session for a item. Corresponds to "ses".
    task : str | None
        The task for a item. Corresponds to "task".
    acquisition: str | None
        The acquisition parameters for the item. Corresponds to "acq".
    run : int | None
        The run number for this item. Corresponds to "run".
    processing : str | None
        The processing label for this item. Corresponds to "proc".
    recording : str | None
        The recording name for this item. Corresponds to "rec".
    space : str | None
        The coordinate space for an anatomical file. Corresponds to "space".
    prefix : str | None
        The prefix for the filename to be created. E.g., a path to the folder
        in which you wish to create a file with this name.
    suffix : str | None
        The suffix for the filename to be created. E.g., 'audio.wav'.

    Examples
    --------
    >>> bids_basename = make_bids_basename(subject='test', session='two', task='mytask', suffix='data.csv')
    >>> print(bids_basename)
    sub-test_ses-two_task-mytask_data.csv
    >>> bids_basename
    BIDSPath(sub-test_ses-two_task-mytask_data.csv)
    >>> # copy and update multiple entities at once
    >>> new_basename = bids_basename.copy().update(subject='test2', session='one')
    >>> print(new_basename)
    sub-test2_ses-one_task-mytask_data.csv
    """  # noqa

    def __init__(self, subject=None, session=None,
                 task=None, acquisition=None, run=None, processing=None,
                 recording=None, space=None, prefix=None, suffix=None):
        if all(ii is None for ii in [subject, session, task,
                                     acquisition, run, processing,
                                     recording, space, prefix, suffix]):
            raise ValueError("At least one parameter must be given.")

        self.update(subject=subject, session=session, task=task,
                    acquisition=acquisition, run=run, processing=processing,
                    recording=recording, space=space, prefix=prefix,
                    suffix=suffix)

    @property
    def entities(self):
        """Return dictionary of the BIDS entities."""
        return OrderedDict([
            ('subject', self.subject),
            ('session', self.session),
            ('task', self.task),
            ('acquisition', self.acquisition),
            ('run', self.run),
            ('processing', self.processing),
            ('recording', self.recording),
            ('space', self.space),
            ('prefix', self.prefix),
            ('suffix', self.suffix)
        ])

    def __str__(self):
        """Return the string representation of the path."""
        basename = []
        for key, val in self.entities.items():
            if key not in ('prefix', 'suffix') and \
                    val is not None:
                _check_key_val(key, val)
                # convert certain keys to shorthand
                if key == 'subject':
                    key = 'sub'
                if key == 'session':
                    key = 'ses'
                if key == 'acquisition':
                    key = 'acq'
                if key == 'processing':
                    key = 'proc'
                if key == 'recording':
                    key = 'rec'
                basename.append('%s-%s' % (key, val))

        if self.suffix is not None:
            basename.append(self.suffix)

        basename = '_'.join(basename)
        if self.prefix is not None:
            basename = op.join(self.prefix, basename)

        return basename

    def __repr__(self):
        """Representation in the style of `pathlib.Path`."""
        return f'{self.__class__.__name__}({str(self)})'

    def __fspath__(self):
        """Return the string representation for any fs functions."""
        return str(self)

    def __eq__(self, other):
        """Compare str representations."""
        return str(self) == str(other)

    def __ne__(self, other):
        """Compare str representations."""
        return str(self) != str(other)

    def copy(self):
        """Copy the instance.

        Returns
        -------
        bidspath : instance of BIDSPath
            The copied bidspath.
        """
        return deepcopy(self)

    def get_bids_fname(self, kind=None, bids_root=None, extension=None):
        """Get the BIDS filename, by inferring kind and extension.

        Parameters
        ----------
        kind : str, optional
            The kind of recording to read. If ``None`` and only one
            kind (e.g., only EEG or only MEG data) is present in the
            dataset, it will be selected automatically.
        bids_root : str | os.PathLike, optional
            Path to root of the BIDS folder
        extension : str, optional
            If ``None``, try to infer the filename extension by searching
            for the file on disk. If the file cannot be found, an error
            will be raised. To disable this automatic inference attempt,
            pass a string (like ``'.fif'`` or ``'.vhdr'``).
            If an empty string is passed, no extension
            will be added to the filename.

        Returns
        -------
        bids_fname : BIDSPath
            A BIDSPath with a full filename.
        """
        # Get the BIDS parameters (=entities)
        sub = self.subject
        ses = self.session

        if extension is None and bids_root is None:
            msg = ('No filename extension was provided, and it cannot be '
                   'automatically inferred because no bids_root was passed.')
            raise ValueError(msg)

        if extension is None:
            bids_fname = _get_bids_fname_from_filesystem(
                bids_basename=self, bids_root=bids_root, sub=sub, ses=ses,
                kind=kind)
            new_suffix = bids_fname.split("_")[-1]
            bids_fname = self.copy().update(suffix=new_suffix)
        else:
            bids_fname = self.copy().update(suffix='{kind}.{extension}')

        return bids_fname

    def update(self, **entities):
        """Update inplace BIDS entity key/value pairs in object.

        Parameters
        ----------
        entities : dict | kwarg
            Allowed BIDS path entities:
            'subject', 'session', 'task', 'acquisition',
            'processing', 'run', 'recording', 'space', 'suffix', 'prefix'

        Returns
        -------
        bidspath : instance of BIDSPath
            The current instance of BIDSPath.

        Examples
        --------
        If one creates a bids basename using
        :func:`mne_bids.make_bids_basename`:

        >>> bids_basename = make_bids_basename(subject='test', session='two', task='mytask', suffix='data.csv')
        >>> print(bids_basename)
        sub-test_ses-two_task-mytask_data.csv
        >>> # Then, one can update this `BIDSPath` object in place
        >>> bids_basename.update(acquisition='test', suffix='ieeg.vhdr', task=None)
        BIDSPath(sub-test_ses-two_acq-test_ieeg.vhdr)
        >>> print(bids_basename)
        sub-test_ses-two_acq-test_ieeg.vhdr
        """  # noqa
        run = entities.get('run')
        if run is not None and not isinstance(run, str):
            # Ensure that run is a string
            entities['run'] = '{:02}'.format(run)

        # error check entities
        for key, val in entities.items():
            # error check allowed BIDS entity keywords
            if key not in BIDS_PATH_ENTITIES and key not in [
                'on_invalid_er_session', 'on_invalid_er_task',
            ]:
                raise ValueError('Key must be one of {BIDS_PATH_ENTITIES}, '
                                 'got %s' % key)

            # set entity value
            setattr(self, key, val)

        self._check(with_emptyroom=False)
        return self

    def _check(self, with_emptyroom=True):
        # check the task/session of er basename
        str(self)  # run string representation to check validity of arguments
        if with_emptyroom and self.subject == 'emptyroom':
            _check_empty_room_basename(self)


def print_dir_tree(folder, max_depth=None):
    """Recursively print dir tree starting from `folder` up to `max_depth`."""
    if not op.exists(folder):
        raise ValueError('Directory does not exist: {}'.format(folder))
    msg = '`max_depth` must be a positive integer or None'
    if not isinstance(max_depth, (int, type(None))):
        raise ValueError(msg)
    if max_depth is None:
        max_depth = float('inf')
    if max_depth < 0:
        raise ValueError(msg)

    # Use max_depth same as the -L param in the unix `tree` command
    max_depth += 1

    # Base length of a tree branch, to normalize each tree's start to 0
    baselen = len(folder.split(os.sep)) - 1

    # Recursively walk through all directories
    for root, dirs, files in os.walk(folder):

        # Check how far we have walked
        branchlen = len(root.split(os.sep)) - baselen

        # Only print, if this is up to the depth we asked
        if branchlen <= max_depth:
            if branchlen <= 1:
                print('|{}'.format(op.basename(root) + os.sep))
            else:
                print('|{} {}'.format((branchlen - 1) * '---',
                                      op.basename(root) + os.sep))

            # Only print files if we are NOT yet up to max_depth or beyond
            if branchlen < max_depth:
                for file in files:
                    print('|{} {}'.format(branchlen * '---', file))


def make_bids_folders(subject, session=None, kind=None, bids_root=None,
                      make_dir=True, overwrite=False, verbose=False):
    """Create a BIDS folder hierarchy.

    This creates a hierarchy of folders *within* a BIDS dataset. You should
    plan to create these folders *inside* the bids_root folder of the dataset.

    Parameters
    ----------
    subject : str
        The subject ID. Corresponds to "sub".
    kind : str
        The kind of folder being created at the end of the hierarchy. E.g.,
        "anat", "func", etc.
    session : str | None
        The session for a item. Corresponds to "ses".
    bids_root : str | pathlib.Path | None
        The bids_root for the folders to be created. If None, folders will be
        created in the current working directory.
    make_dir : bool
        Whether to actually create the folders specified. If False, only a
        path will be generated but no folders will be created.
    overwrite : bool
        How to handle overwriting previously generated data.
        If overwrite == False then no existing folders will be removed, however
        if overwrite == True then any existing folders at the session level
        or lower will be removed, including any contained data.
    verbose : bool
        If verbose is True, print status updates
        as folders are created.

    Returns
    -------
    path : str
        The (relative) path to the folder that was created.

    Examples
    --------
    >>> make_bids_folders('sub_01', session='mysession', kind='meg', bids_root='/path/to/project', make_dir=False)  # noqa
    '/path/to/project/sub-sub_01/ses-mysession/meg'

    """  # noqa
    _check_types((subject, kind, session))
    if bids_root is not None:
        bids_root = _path_to_str(bids_root)

    if session is not None:
        _check_key_val('ses', session)

    path = ['sub-%s' % subject]
    if isinstance(session, str):
        path.append('ses-%s' % session)
    if isinstance(kind, str):
        path.append(kind)
    path = op.join(*path)
    if isinstance(bids_root, str):
        path = op.join(bids_root, path)

    if make_dir is True:
        _mkdir_p(path, overwrite=overwrite, verbose=verbose)
    return path


def _parse_ext(raw_fname, verbose=False):
    """Split a filename into its name and extension."""
    fname, ext = os.path.splitext(raw_fname)
    # BTi data is the only file format that does not have a file extension
    if ext == '' or 'c,rf' in fname:
        if verbose is True:
            print('Found no extension for raw file, assuming "BTi" format and '
                  'appending extension .pdf')
        ext = '.pdf'
    # If ending on .gz, check whether it is an .nii.gz file
    elif ext == '.gz' and raw_fname.endswith('.nii.gz'):
        ext = '.nii.gz'
        fname = fname[:-4]  # cut off the .nii
    return fname, ext


def _parse_bids_filename(fname, verbose):
    """Get dict from BIDS fname."""
    keys = ['sub', 'ses', 'task', 'acq', 'run', 'proc', 'run', 'space',
            'rec', 'split', 'kind']
    params = {key: None for key in keys}
    idx_key = 0
    for match in re.finditer(param_regex, op.basename(fname)):
        key, value = match.groups()
        if key not in keys:
            raise KeyError('Unexpected entity "%s" found in filename "%s"'
                           % (key, fname))
        if keys.index(key) < idx_key:
            raise ValueError('Entities in filename not ordered correctly.'
                             ' "%s" should have occurred earlier in the '
                             'filename "%s"' % (key, fname))
        idx_key = keys.index(key)
        params[key] = value
    return params


def _find_matching_sidecar(bids_fname, bids_root, suffix, allow_fail=False):
    """Try to find a sidecar file with a given suffix for a data file.

    Parameters
    ----------
    bids_fname : BIDSPath
        Full name of the data file
    bids_root : str | pathlib.Path
        Path to root of the BIDS folder
    suffix : str
        The suffix of the sidecar file to be found. E.g., "_coordsystem.json"
    allow_fail : bool
        If False, will raise RuntimeError if not exactly one matching sidecar
        was found. If True, will return None in that case. Defaults to False

    Returns
    -------
    sidecar_fname : str | None
        Path to the identified sidecar file, or None, if `allow_fail` is True
        and no sidecar_fname was found

    """
    # We only use subject and session as identifier, because all other
    # parameters are potentially not binding for metadata sidecar files
    search_str = f'sub-{bids_fname.subject}'
    if bids_fname.session is not None:
        search_str += f'_ses-{bids_fname.session}'

    # Find all potential sidecar files, doing a recursive glob
    # from bids_root/sub_id/
    search_str = op.join(bids_root, f'sub-{bids_fname.subject}',
                         '**', search_str + '*' + suffix)
    candidate_list = glob.glob(search_str, recursive=True)
    best_candidates = _find_best_candidates(bids_fname.entities,
                                            candidate_list)

    if len(best_candidates) == 1:
        # Success
        return best_candidates[0]

    # map suffix to make warning message readable
    if 'electrodes.tsv' in suffix:
        suffix = 'electrodes.tsv'
    if 'coordsystem.json' in suffix:
        suffix = 'coordsystem.json'

    # We failed. Construct a helpful error message.
    # If this was expected, simply return None, otherwise, raise an exception.
    msg = None
    if len(best_candidates) == 0:
        msg = ('Did not find any {} associated with {}.'
               .format(suffix, bids_fname))
    elif len(best_candidates) > 1:
        # More than one candidates were tied for best match
        msg = ('Expected to find a single {} file associated with '
               '{}, but found {}: "{}".'
               .format(suffix, bids_fname, len(candidate_list),
                       candidate_list))
    msg += '\n\nThe search_str was "{}"'.format(search_str)
    if allow_fail:
        warn(msg)
        return None
    else:
        raise RuntimeError(msg)


def make_bids_basename(subject=None, session=None, task=None,
                       acquisition=None, run=None, processing=None,
                       recording=None, space=None, prefix=None, suffix=None):
    """Create a partial/full BIDS basename from its component parts.

    BIDS filename prefixes have one or more pieces of metadata in them. They
    must follow a particular order, which is followed by this function. This
    will generate the *prefix* for a BIDS filename that can be used with many
    subsequent files, or you may also give a suffix that will then complete
    the file name.

    Note that all parameters are not applicable to each kind of data. For
    example, electrode location TSV files do not need a task field.

    Parameters
    ----------
    subject : str | None
        The subject ID. Corresponds to "sub".
    session : str | None
        The session identifier. Corresponds to "ses". Must be a date in
        format "YYYYMMDD" if subject is "emptyroom".
    task : str | None
        The task identifier. Corresponds to "task". Must be "noise" if
        subject is "emptyroom".
    acquisition: str | None
        The acquisition parameters. Corresponds to "acq".
    run : int | None
        The run number. Corresponds to "run".
    processing : str | None
        The processing label. Corresponds to "proc".
    recording : str | None
        The recording name. Corresponds to "rec".
    space : str | None
        The coordinate space for an anatomical file. Corresponds to "space".
    prefix : str | None
        The prefix for the filename to be created. E.g., a path to the folder
        in which you wish to create a file with this name.
    suffix : str | None
        The suffix for the filename to be created. E.g., 'audio.wav'.

    Returns
    -------
    basename : BIDSPath
        The BIDS basename you wish to create.

    Examples
    --------
    >>> print(make_bids_basename(subject='test', session='two', task='mytask', suffix='data.csv')) # noqa: E501
    sub-test_ses-two_task-mytask_data.csv
    """
    bids_path = BIDSPath(subject=subject, session=session, task=task,
                         acquisition=acquisition, run=run,
                         processing=processing, recording=recording,
                         space=space, prefix=prefix, suffix=suffix)
    bids_path._check()
    return bids_path


def get_kinds(bids_root):
    """Get list of data types ("kinds") present in a BIDS dataset.

    Parameters
    ----------
    bids_root : str | pathlib.Path
        Path to the root of the BIDS directory.

    Returns
    -------
    kinds : list of str
        List of the data types present in the BIDS dataset pointed to by
        `bids_root`.

    """
    # Take all possible kinds from "entity" table (Appendix in BIDS spec)
    kind_list = ('anat', 'func', 'dwi', 'fmap', 'beh', 'meg', 'eeg', 'ieeg')
    kinds = list()
    for root, dirs, files in os.walk(bids_root):
        for dir in dirs:
            if dir in kind_list and dir not in kinds:
                kinds.append(dir)

    return kinds


def get_entity_vals(bids_root, entity_key, *, ignore_sub='emptyroom',
                    ignore_task=None, ignore_ses=None, ignore_run=None,
                    ignore_acq=None):
    """Get list of values associated with an `entity_key` in a BIDS dataset.

    BIDS file names are organized by key-value pairs called "entities" [1]_.
    With this function, you can get all values for an entity indexed by its
    key.

    Parameters
    ----------
    bids_root : str | pathlib.Path
        Path to the root of the BIDS directory.
    entity_key : str
        The name of the entity key to search for. Can be one of
        ['sub', 'ses', 'task', 'run', 'acq'].
    ignore_sub : str | iterable | None
        Subject(s) to ignore. By default, entities from the ``emptyroom``
        mock-subject are not returned. If ``None``, include all subjects.
    ignore_task : str | iterable | None
        Task(s) to ignore. If ``None``, include all tasks.
    ignore_ses : str | iterable | None
        Session(s) to ignore. If ``None``, include all sessions.
    ignore_run : str | iterable | None
        Run(s) to ignore. If ``None``, include all runs.
    ignore_acq : str | iterable | None
        Acquisition(s) to ignore. If ``None``, include all acquisitions.

    Returns
    -------
    entity_vals : list of str
        List of the values associated with an `entity_key` in the BIDS dataset
        pointed to by `bids_root`.

    Examples
    --------
    >>> bids_root = os.path.expanduser('~/mne_data/eeg_matchingpennies')
    >>> entity_key = 'sub'
    >>> get_entity_vals(bids_root, entity_key)
    ['05', '06', '07', '08', '09', '10', '11']

    Notes
    -----
    This function will scan the entire ``bids_root``, except for a
    ``derivatives`` subfolder placed directly under ``bids_root``.

    References
    ----------
    .. [1] https://bids-specification.rtfd.io/en/latest/02-common-principles.html#file-name-structure  # noqa: E501

    """
    entities = ('sub', 'ses', 'task', 'run', 'acq')
    if entity_key not in entities:
        raise ValueError('`key` must be one of "{}". Got "{}"'
                         .format(entities, entity_key))

    ignore_sub = _ensure_tuple(ignore_sub)
    ignore_task = _ensure_tuple(ignore_task)
    ignore_ses = _ensure_tuple(ignore_ses)
    ignore_run = _ensure_tuple(ignore_run)
    ignore_acq = _ensure_tuple(ignore_acq)

    p = re.compile(r'{}-(.*?)_'.format(entity_key))
    value_list = list()
    for filename in Path(bids_root).rglob('*{}-*_*'.format(entity_key)):
        # Ignore `derivatives` folder.
        if str(filename).startswith(op.join(bids_root, 'derivatives')):
            continue

        if ignore_sub and any([filename.stem.startswith(f'sub-{s}_')
                               for s in ignore_sub]):
            continue
        if ignore_task and any([f'_task-{t}_' in filename.stem
                                for t in ignore_task]):
            continue
        if ignore_ses and any([f'_ses-{s}_' in filename.stem
                               for s in ignore_ses]):
            continue
        if ignore_run and any([f'_run-{r}_' in filename.stem
                               for r in ignore_run]):
            continue
        if ignore_acq and any([f'_acq-{a}_' in filename.stem
                               for a in ignore_acq]):
            continue

        match = p.search(filename.stem)
        value = match.group(1)
        if value not in value_list:
            value_list.append(value)
    return sorted(value_list)


def _mkdir_p(path, overwrite=False, verbose=False):
    """Create a directory, making parent directories as needed [1].

    References
    ----------
    .. [1] stackoverflow.com/questions/600268/mkdir-p-functionality-in-python

    """
    if overwrite and op.isdir(path):
        sh.rmtree(path)
        if verbose is True:
            print('Clearing path: %s' % path)

    os.makedirs(path, exist_ok=True)
    if verbose is True:
        print('Creating folder: %s' % path)


def _find_best_candidates(params, candidate_list):
    """Return the best candidate, based on the number of param matches.

    Assign each candidate a score, based on how many entities are shared with
    the ones supplied in the `params` parameter. The candidate with the highest
    score wins. Candidates with entities that conflict with the supplied
    `params` are disqualified.

    Parameters
    ----------
    params : dict
        The entities that the candidate should match.
    candidate_list : list of str
        A list of candidate filenames.

    Returns
    -------
    best_candidates : list of str
        A list of all the candidate filenames that are tied for first place.
        Hopefully, the list will have a length of one.
    """
    params = {key: value for key, value in params.items() if value is not None}

    best_candidates = []
    best_n_matches = 0
    for candidate in candidate_list:
        n_matches = 0
        candidate_disqualified = False
        candidate_params = _parse_bids_filename(candidate, verbose=False)
        for entity, value in params.items():
            if entity in candidate_params:
                if candidate_params[entity] is None:
                    continue
                elif candidate_params[entity] == value:
                    n_matches += 1
                else:
                    # Incompatible entity found, candidate is disqualified
                    candidate_disqualified = True
                    break
        if not candidate_disqualified:
            if n_matches > best_n_matches:
                best_n_matches = n_matches
                best_candidates = [candidate]
            elif n_matches == best_n_matches:
                best_candidates.append(candidate)

    return best_candidates


def _get_bids_fname_from_filesystem(*, bids_basename, bids_root, sub, ses,
                                    kind):
    if kind is None:
        kind = _infer_kind(bids_basename=bids_basename, bids_root=bids_root,
                           sub=sub, ses=ses)

    data_dir = make_bids_folders(subject=sub, session=ses, kind=kind,
                                 make_dir=False)

    bti_dir = op.join(bids_root, data_dir, f'{bids_basename}_{kind}')
    if op.isdir(bti_dir):
        logger.info(f'Assuming BTi data in {bti_dir}')
        bids_fname = f'{bti_dir}.pdf'
    else:
        # Find all matching files in all supported formats.
        valid_exts = list(reader.keys())
        matching_paths = glob.glob(op.join(bids_root, data_dir,
                                           f'{bids_basename}_{kind}.*'))
        matching_paths = [p for p in matching_paths
                          if _parse_ext(p)[1] in valid_exts]

        if not matching_paths:
            msg = ('Could not locate a data file of a supported format. This '
                   'is likely a problem with your BIDS dataset. Please run '
                   'the BIDS validator on your data.')
            raise RuntimeError(msg)

        # FIXME This will break e.g. with FIFF data split across multiple
        # FIXME files.
        if len(matching_paths) > 1:
            msg = ('Found more than one matching data file for the requested '
                   'recording. Cannot proceed due to the ambiguity. This is '
                   'likely a problem with your BIDS dataset. Please run the '
                   'BIDS validator on your data.')
            raise RuntimeError(msg)

        matching_path = matching_paths[0]
        bids_fname = op.basename(matching_path)

    return bids_fname


def _get_kinds_for_sub(*, bids_basename, bids_root, sub, ses=None):
    """Retrieve available data kinds for a specific subject and session."""
    subject_dir = op.join(bids_root, f'sub-{sub}')
    if ses is not None:
        subject_dir = op.join(subject_dir, f'ses-{ses}')

    # TODO We do this to ensure we don't accidentally pick up any "spurious"
    # TODO sub-directories. But is that really necessary with valid BIDS data?
    kinds_in_dataset = get_kinds(bids_root=bids_root)
    subdirs = [f.name for f in os.scandir(subject_dir) if f.is_dir()]
    available_kinds = [s for s in subdirs if s in kinds_in_dataset]
    return available_kinds


def _infer_kind(*, bids_basename, bids_root, sub, ses):
    # Check which kind is available for this particular
    # subject & session. If we get no or multiple hits, throw an error.

    kinds = _get_kinds_for_sub(bids_basename=bids_basename,
                               bids_root=bids_root, sub=sub, ses=ses)

    # We only want to handle electrophysiological data here.
    allowed_kinds = ['meg', 'eeg', 'ieeg']
    kinds = list(set(kinds) & set(allowed_kinds))
    if not kinds:
        raise ValueError('No electrophysiological data found.')
    elif len(kinds) >= 2:
        msg = (f'Found data of more than one recording modality. Please '
               f'pass the `kind` parameter to specify which data to load. '
               f'Found the following kinds: {kinds}')
        raise RuntimeError(msg)

    assert len(kinds) == 1
    return kinds[0]
