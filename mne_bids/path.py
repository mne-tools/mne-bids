"""BIDS compatible path functionality."""
# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)
import glob
import os
import re
from io import StringIO
import shutil as sh
from collections import OrderedDict
from copy import deepcopy
from os import path as op
from pathlib import Path
from datetime import datetime

import numpy as np
from mne.utils import warn, logger, _validate_type

from mne_bids.config import (
    ALLOWED_PATH_ENTITIES, ALLOWED_FILENAME_EXTENSIONS,
    ALLOWED_FILENAME_SUFFIX, ALLOWED_PATH_ENTITIES_SHORT,
    ALLOWED_DATATYPES, SUFFIX_TO_DATATYPE, ALLOWED_DATATYPE_EXTENSIONS,
    reader, ENTITY_VALUE_TYPE)
from mne_bids.utils import (_check_key_val, _check_empty_room_basename,
                            param_regex, _ensure_tuple)


def _get_matched_empty_room(bids_path):
    """Get matching empty-room file for an MEG recording."""
    # Check whether we have a BIDS root.
    bids_root = bids_path.root
    if bids_root is None:
        raise ValueError('The root of the "bids_path" must be set. '
                         'Please use `bids_path.update(root="<root>")` '
                         'to set the root of the BIDS folder to read.')

    from mne_bids import read_raw_bids  # avoid circular import.
    bids_path = bids_path.copy()

    datatype = 'meg'  # We're only concerned about MEG data here
    bids_fname = bids_path.update(suffix=datatype,
                                  root=bids_root).fpath
    _, ext = _parse_ext(bids_fname)
    extra_params = None
    if ext == '.fif':
        extra_params = dict(allow_maxshield=True)

    raw = read_raw_bids(bids_path=bids_path, extra_params=extra_params)
    if raw.info['meas_date'] is None:
        raise ValueError('The provided recording does not have a measurement '
                         'date set. Cannot get matching empty-room file.')

    ref_date = raw.info['meas_date']
    if not isinstance(ref_date, datetime):  # pragma: no cover
        # for MNE < v0.20
        ref_date = datetime.fromtimestamp(raw.info['meas_date'][0])

    emptyroom_dir = BIDSPath(root=bids_root, subject='emptyroom').directory

    if not emptyroom_dir.exists():
        return None

    # Find the empty-room recording sessions.
    emptyroom_session_dirs = [x for x in emptyroom_dir.iterdir()
                              if x.is_dir() and str(x.name).startswith('ses-')]
    if not emptyroom_session_dirs:  # No session sub-directories found
        emptyroom_session_dirs = [emptyroom_dir]

    # Now try to discover all recordings inside the session directories.

    allowed_extensions = list(reader.keys())
    # `.pdf` is just a "virtual" extension for BTi data (which is stored inside
    # a dedicated directory that doesn't have an extension)
    del allowed_extensions[allowed_extensions.index('.pdf')]

    candidate_er_fnames = []
    for session_dir in emptyroom_session_dirs:
        dir_contents = glob.glob(op.join(session_dir, datatype,
                                         f'sub-emptyroom_*_{datatype}*'))
        for item in dir_contents:
            item = Path(item)
            if ((item.suffix in allowed_extensions) or
                    (not item.suffix and item.is_dir())):  # Hopefully BTi?
                candidate_er_fnames.append(item.name)

    # Walk through recordings, trying to extract the recording date:
    # First, from the filename; and if that fails, from `info['meas_date']`.
    best_er_bids_path = None
    min_delta_t = np.inf
    date_tie = False

    failed_to_get_er_date_count = 0
    for er_fname in candidate_er_fnames:
        params = get_entities_from_fname(er_fname)
        er_meas_date = None
        params.pop('subject')  # er subject entity is different
        er_bids_path = BIDSPath(subject='emptyroom', **params, datatype='meg',
                                root=bids_root, check=False)

        # Try to extract date from filename.
        if params['session'] is not None:
            try:
                er_meas_date = datetime.strptime(params['session'], '%Y%m%d')
            except (ValueError, TypeError):
                # There is a session in the filename, but it doesn't encode a
                # valid date.
                pass

        if er_meas_date is None:  # No luck so far! Check info['meas_date']
            _, ext = _parse_ext(er_fname)
            extra_params = None
            if ext == '.fif':
                extra_params = dict(allow_maxshield=True)

            er_raw = read_raw_bids(bids_path=er_bids_path,
                                   extra_params=extra_params)

            er_meas_date = er_raw.info['meas_date']
            if er_meas_date is None:  # There's nothing we can do.
                failed_to_get_er_date_count += 1
                continue

        er_meas_date = er_meas_date.replace(tzinfo=ref_date.tzinfo)
        delta_t = er_meas_date - ref_date

        if abs(delta_t.total_seconds()) == min_delta_t:
            date_tie = True
        elif abs(delta_t.total_seconds()) < min_delta_t:
            min_delta_t = abs(delta_t.total_seconds())
            best_er_bids_path = er_bids_path
            date_tie = False

    if failed_to_get_er_date_count > 0:
        msg = (f'Could not retrieve the empty-room measurement date from '
               f'a total of {failed_to_get_er_date_count} recording(s).')
        warn(msg)

    if date_tie:
        msg = ('Found more than one matching empty-room measurement with the '
               'same recording date. Selecting the first match.')
        warn(msg)

    return best_er_bids_path


class BIDSPath(object):
    """A BIDS path object.

    BIDS filename prefixes have one or more pieces of metadata in them. They
    must follow a particular order, which is followed by this function. This
    will generate the *prefix* for a BIDS filename that can be used with many
    subsequent files, or you may also give a suffix that will then complete
    the file name.

    BIDSPath allows dynamic updating of its entities in place, and operates
    similar to `pathlib.Path`. In addition, it can query multiple paths
    with matching BIDS entities via the ``match`` method.

    Note that not all parameters are applicable to each suffix of data. For
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
    split : int | None
        The split of the continuous recording file for ``.fif`` data.
        Corresponds to "split".
    suffix : str | None
        The filename suffix. This is the entity after the
        last ``_`` before the extension. E.g., ``'channels'``.
        The following filename suffix's are accepted:
        'meg', 'markers', 'eeg', 'ieeg', 'T1w',
        'participants', 'scans', 'electrodes', 'coordsystem',
        'channels', 'events', 'headshape', 'digitizer',
        'behav', 'phsyio', 'stim'
    extension : str | None
        The extension of the filename. E.g., ``'.json'``.
    datatype : str
        The "data type" of folder being created at the end of the folder
        hierarchy. E.g., ``'anat'``, ``'func'``, ``'eeg'``, ``'meg'``,
        ``'ieeg'``, etc.
    root : str | pathlib.Path | None
        The root for the filename to be created. E.g., a path to the folder
        in which you wish to create a file with this name.
    check : bool
        If True enforces the entities to be valid according to the
        current BIDS standard. Defaults to True.

    Attributes
    ----------
    entities : dict
        The dictionary of the BIDS entities and their values:
        ``subject``, ``session``, ``task``, ``acquisition``,
        ``run``, ``processing``, ``space``, ``recording`` and ``suffix``.
    datatype : str | None
        The data type, i.e., one of ``'meg'``, ``'eeg'``, ``'ieeg'``,
        ``'anat'``.
    basename : str
        The basename of the file path. Similar to `os.path.basename(fpath)`.
    root : pathlib.Path
        The root of the BIDS path.
    directory : pathlib.Path
        The directory path.
    fpath : pathlib.Path
        The full file path.
    check : bool
        If ``True``, enforces the entities to be valid according to the
        BIDS specification. The check is performed on instantiation
        and any ``update`` function calls (and may be overridden in the
        latter).

    Examples
    --------
    >>> bids_path = BIDSPath(subject='test', session='two', task='mytask',
                                 suffix='ieeg', extension='.edf')
    >>> print(bids_path.basename)
    sub-test_ses-two_task-mytask_ieeg.edf
    >>> bids_path
    BIDSPath(root: None,
    basename: sub-test_ses-two_task-mytask_ieeg.edf)
    >>> # copy and update multiple entities at once
    >>> new_bids_path = bids_path.copy().update(subject='test2',
                                                   session='one')
    >>> print(new_bids_path.basename)
    sub-test2_ses-one_task-mytask_ieeg.edf
    >>> # printing the BIDSPath will show relative path when
    >>> # root is not set
    >>> print(new_bids_path)
    sub-test2/ses-one/ieeg/sub-test2_ses-one_task-mytask_ieeg.edf
    >>> new_bids_path.update(suffix='channels', extension='.tsv')
    >>> # setting suffix without an identifiable datatype will
    >>> # result in a wildcard at the datatype directory level
    >>> print(new_bids_path)
    sub-test2/ses-one/*/sub-test2_ses-one_task-mytask_channels.tsv
    >>> # set a root for the BIDS dataset
    >>> new_bids_path.update(root='/bids_dataset')
    >>> print(new_bids_path.root)
    /bids_dataset
    >>> print(new_bids_path.basename)
    sub-test2_ses-one_task-mytask_ieeg.edf
    >>> print(new_bids_path)
    /bids_dataset/sub-test2/ses-one/ieeg/sub-test2_ses-one_task-mytask_ieeg.edf
    >>> print(new_bids_path.directory)
    /bids_dataset/sub-test2/ses-one/ieeg/
    """

    def __init__(self, subject=None, session=None,
                 task=None, acquisition=None, run=None, processing=None,
                 recording=None, space=None, split=None, root=None,
                 suffix=None, extension=None, datatype=None, check=True):
        if all(ii is None for ii in [subject, session, task,
                                     acquisition, run, processing,
                                     recording, space, root, suffix,
                                     extension]):
            raise ValueError("At least one parameter must be given.")

        self.check = check

        self.update(subject=subject, session=session, task=task,
                    acquisition=acquisition, run=run, processing=processing,
                    recording=recording, space=space, split=split,
                    root=root, datatype=datatype,
                    suffix=suffix, extension=extension)

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
            ('space', self.space),
            ('recording', self.recording),
            ('split', self.split)
        ])

    @property
    def basename(self):
        """Path basename."""
        basename = []
        for key, val in self.entities.items():
            if val is not None and key != 'datatype':
                # convert certain keys to shorthand
                long_to_short_entity = {
                    val: key for key, val
                    in ALLOWED_PATH_ENTITIES_SHORT.items()
                }
                key = long_to_short_entity[key]
                basename.append(f'{key}-{val}')

        if self.suffix is not None:
            if self.extension is not None:
                basename.append(f'{self.suffix}{self.extension}')
            else:
                basename.append(self.suffix)

        basename = '_'.join(basename)
        return basename

    @property
    def directory(self):
        """Get the BIDS parent directory.

        If ``subject``, ``session`` and ``datatype`` are set, then they will be
        used to construct the directory location. For example, if
        ``subject='01'``, ``session='02'`` and ``datatype='ieeg'``, then the
        directory would be::

            <root>/sub-01/ses-02/ieeg

        Returns
        -------
        data_path : pathlib.Path
            The path of the BIDS directory.
        """
        # Create the data path based on the available entities:
        # root, subject, session, and datatype
        data_path = '' if self.root is None else self.root
        if self.subject is not None:
            data_path = op.join(data_path, f'sub-{self.subject}')
        if self.session is not None:
            data_path = op.join(data_path, f'ses-{self.session}')
        # datatype will allow 'meg', 'eeg', 'ieeg', 'anat'
        if self.datatype is not None:
            data_path = op.join(data_path, self.datatype)
        return Path(data_path)

    def __str__(self):
        """Return the string representation of the path."""
        return str(self.fpath)

    def __repr__(self):
        """Representation in the style of `pathlib.Path`."""
        return f'{self.__class__.__name__}(\n' \
               f'root: {self.root}\n' \
               f'datatype: {self.datatype}\n' \
               f'basename: {self.basename})'

    def __fspath__(self):
        """Return the string representation for any fs functions."""
        return str(self.fpath)

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

    def mkdir(self, exist_ok=True):
        """Create the directory structure of the BIDS path.

        Parameters
        ----------
        exist_ok : bool
            If ``False``, raise an exception if the directory already exists.
            Otherwise, do nothing (default).

        Returns
        -------
        self : instance of BIDSPath
            The BIDSPath object.
        """
        self.directory.mkdir(parents=True, exist_ok=exist_ok)
        return self

    @property
    def fpath(self):
        """Full filepath for this BIDS file.

        Getting the file path consists of the entities passed in
        and will get the relative (or full if ``root`` is passed)
        path.

        Returns
        -------
        bids_fpath : pathlib.Path
            Either the relative, or full path to the dataset.
        """
        # get the inner-most BIDS directory for this file path
        data_path = self.directory

        # account for MEG data that are directory-based
        # else, all other file paths attempt to match
        if self.suffix == 'meg' and self.extension == '.ds':
            bids_fpath = op.join(data_path, self.basename)
        elif self.suffix == 'meg' and self.extension == '.pdf':
            bids_fpath = op.join(data_path,
                                 op.splitext(self.basename)[0])
        else:
            # if suffix and/or extension is missing, and root is
            # not None, then BIDSPath will infer the dataset
            # else, return the relative path with the basename
            if (self.suffix is None or self.extension is None) and \
                    self.root is not None:
                # get matching BIDS paths inside the bids root
                matching_paths = \
                    _get_matching_bidspaths_from_filesystem(self)

                # FIXME This will break
                # FIXME e.g. with FIFF data split across multiple files.
                # if extension is not specified and no unique file path
                # return filepath of the actual dataset for MEG/EEG/iEEG data
                if self.suffix is None or self.suffix in ALLOWED_DATATYPES:
                    # now only use valid datatype extension
                    valid_exts = sum(ALLOWED_DATATYPE_EXTENSIONS.values(), [])
                    matching_paths = [p for p in matching_paths
                                      if _parse_ext(p)[1] in valid_exts]

                if (self.split is None and
                        (not matching_paths or
                         '_split-' in matching_paths[0])):
                    # try finding FIF split files (only first one)
                    this_self = self.copy().update(split='01')
                    matching_paths = \
                        _get_matching_bidspaths_from_filesystem(this_self)

                # found no matching paths
                if not matching_paths:
                    msg = (f'Could not locate a data file of a supported '
                           f'format. This is likely a problem with your '
                           f'BIDS dataset. Please run the BIDS validator '
                           f'on your data. (root={self.root}, '
                           f'basename={self.basename}). '
                           f'{matching_paths}')
                    warn(msg)

                    bids_fpath = op.join(data_path, self.basename)
                # if paths still cannot be resolved, then there is an error
                elif len(matching_paths) > 1:
                    msg = ('Found more than one matching data file for the '
                           'requested recording. Cannot proceed due to the '
                           'ambiguity. This is likely a problem with your '
                           'BIDS dataset. Please run the BIDS validator on '
                           'your data.')
                    raise RuntimeError(msg)
                else:
                    bids_fpath = matching_paths[0]

            else:
                bids_fpath = op.join(data_path, self.basename)

        bids_fpath = Path(bids_fpath)
        return bids_fpath

    def update(self, *, check=None, **kwargs):
        """Update inplace BIDS entity key/value pairs in object.

        ``run`` and ``split`` are auto-parsed to have two
        numbers when passed in. For example, if ``run=1``, then it will
        become ``run='01'``.

        Also performs error checks on various entities to
        adhere to the BIDS specification. Specifically:
        - ``suffix`` should be one of: ``anat``, ``eeg``, ``ieeg``, ``meg``
        - ``extension`` should be one of the accepted file
        extensions in the file path: ``.con``, ``.sqd``, ``.fif``,
        ``.pdf``, ``.ds``, ``.vhdr``, ``.edf``, ``.bdf``, ``.set``,
        ``.edf``, ``.set``, ``.mef``, ``.nwb``
        - ``suffix`` should be one acceptable file suffixes in: ``meg``,
        ``markers``, ``eeg``, ``ieeg``, ``T1w``,
        ``participants``, ``scans``, ``electrodes``, ``channels``,
        ``coordsystem``, ``events``, ``headshape``, ``digitizer``,
        ``behav``, ``phsyio``, ``stim``

        Parameters
        ----------
        check : None | bool
            If a boolean, controls whether to enforce the entities to be valid
            according to the BIDS specification. This will set the
            ``.check`` attribute accordingly. If ``None``, rely on the existing
            ``.check`` attribute instead, which is set upon ``BIDSPath``
            instantiation. Defaults to ``None``.
        **kwargs : dict
            It can contain updates for valid BIDS path entities:
            'subject', 'session', 'task', 'acquisition', 'processing', 'run',
            'recording', 'space', 'suffix'
            or updates for 'root' or 'datatype'.

        Returns
        -------
        bidspath : instance of BIDSPath
            The updated instance of BIDSPath.

        Examples
        --------
        If one creates a bids basename using
        :func:`mne_bids.BIDSPath`:

        >>> bids_path = BIDSPath(subject='test', session='two',
                                     task='mytask', suffix='channels',
                                     extension='.tsv')
        >>> print(bids_path.basename)
        sub-test_ses-two_task-mytask_channels.tsv
        >>> # Then, one can update this `BIDSPath` object in place
        >>> bids_path.update(acquisition='test', suffix='ieeg',
                             extension='.vhdr', task=None)
        >>> print(bids_path.basename)
        sub-test_ses-two_acq-test_ieeg.vhdr
        """
        for key, val in kwargs.items():
            if key == 'root':
                _validate_type(val, types=('path-like', None), item_name=key)
                continue

            if key == 'datatype':
                if val is not None and val not in ALLOWED_DATATYPES:
                    raise ValueError(f'datatype ({val}) is not valid. '
                                     f'Should be one of '
                                     f'{ALLOWED_DATATYPES}')
                else:
                    continue

            if key not in ENTITY_VALUE_TYPE:
                raise ValueError(f'Key must be one of '
                                 f'{ALLOWED_PATH_ENTITIES}, got {key}')

            if ENTITY_VALUE_TYPE[key] == 'label':
                _validate_type(val, types=(None, str),
                               item_name=key)
            else:
                assert ENTITY_VALUE_TYPE[key] == 'index'
                _validate_type(val, types=(int, str, None), item_name=key)
                if isinstance(val, str) and not val.isdigit():
                    raise ValueError(f'{key} is not an index (Got {val})')
                elif isinstance(val, int):
                    kwargs[key] = '{:02}'.format(val)

        # ensure extension starts with a '.'
        extension = kwargs.get('extension')
        if extension is not None:
            if not extension.startswith('.'):
                extension = f'.{extension}'
                kwargs['extension'] = extension

        # error check entities
        for key, val in kwargs.items():
            # check if there are any characters not allowed
            if val is not None and key != 'root':
                _check_key_val(key, val)

            # set entity value, ensuring `root` is a Path
            if key == 'root' and val is not None:
                val = Path(val)
            setattr(self, key, val)

        # infer datatype if suffix is uniquely the datatype
        if self.datatype is None and \
                self.suffix in SUFFIX_TO_DATATYPE:
            self.datatype = SUFFIX_TO_DATATYPE[self.suffix]

        # Update .check attribute and perform a check of the entities.
        if check is not None:
            self.check = check
        self._check()
        return self

    def match(self):
        """Get a list of all matching paths in the root directory.

        Performs a recursive search, starting in ``.root`` (if set), based on
        `BIDSPath.entities` object. Ignores ``.json`` files.

        Returns
        -------
        bids_paths : list of BIDSPath
            The matching paths.
        """
        if self.root is None:
            raise RuntimeError('Cannot match basenames if `root` '
                               'attribute is not set. Please set the'
                               'BIDS root directory path to `root` via '
                               'BIDSPath.update().')

        # allow searching by datatype
        # all other entities are filtered below
        if self.datatype is not None:
            search_str = f'*/{self.datatype}/*'
        else:
            search_str = '*.*'

        fnames = self.root.rglob(search_str)
        # Only keep files (not directories), and omit the JSON sidecars.
        fnames = [f.name for f in fnames
                  if f.is_file() and f.suffix != '.json']
        fnames = _filter_fnames(fnames, suffix=self.suffix,
                                extension=self.extension,
                                **self.entities)

        bids_paths = []
        for fname in fnames:
            # get all BIDS entities from the filename
            entities = get_entities_from_fname(fname)

            # extension is not an entity, so get it explicitly
            _, extension = _parse_ext(fname, verbose=False)

            # get datatype
            fpath = list(self.root.rglob(f'*{fname}*'))[0]
            datatype = _infer_datatype_from_path(fpath)

            bids_path = BIDSPath(root=self.root, datatype=datatype,
                                 extension=extension, **entities)
            bids_paths.append(bids_path)

        return bids_paths

    def _check(self):
        """Deep check or not of the instance."""
        self.basename  # run basename to check validity of arguments

        # perform error check on scans
        if (self.suffix == 'scans' and self.extension == '.tsv') \
                and _check_non_sub_ses_entity(self):
            raise ValueError('scans.tsv file name can only contain '
                             'subject and session entities. BIDSPath '
                             f'currently contains {self.entities}.')

        # perform deeper check if user has it turned on
        if self.check:
            if self.subject == 'emptyroom':
                _check_empty_room_basename(self)

            # ensure extension starts with a '.'
            extension = self.extension
            if extension is not None:
                # check validity of the extension
                if extension not in ALLOWED_FILENAME_EXTENSIONS:
                    raise ValueError(f'Extension {extension} is not '
                                     f'allowed. Use one of these extensions '
                                     f'{ALLOWED_FILENAME_EXTENSIONS}.')

            # error check suffix
            suffix = self.suffix
            if suffix is not None and \
                    suffix not in ALLOWED_FILENAME_SUFFIX:
                raise ValueError(f'Suffix {suffix} is not allowed. '
                                 f'Use one of these suffixes '
                                 f'{ALLOWED_FILENAME_SUFFIX}.')

    def find_empty_room(self):
        """Find the corresponding empty-room file of an MEG recording.

        This will only work if the ``.root`` attribute of the
        :class:`mne_bids.BIDSPath` instance has been set.

        Returns
        -------
        BIDSPath | None
            The path corresponding to the best-matching empty-room measurement.
            Returns None if none was found.

        """
        return _get_matched_empty_room(self)

    @property
    def meg_calibration_fpath(self):
        """Find the matching Elekta/Neuromag/MEGIN fine-calibration file.

        This requires that at least ``root`` and ``subject`` are set, and that
        ``datatype`` is either ``'meg'`` or ``None``.

        Returns
        -------
        path : pathlib.Path | None
            The path of the fine-calibration file, or ``None`` if it couldn't
            be found.
        """
        if self.root is None or self.subject is None:
            raise ValueError('root and subject must be set.')
        if self.datatype not in (None, 'meg'):
            raise ValueError('Can only find fine-calibration file for MEG '
                             'datasets.')

        path = BIDSPath(subject=self.subject, session=self.session,
                        acquisition='calibration', suffix='meg',
                        extension='.dat', datatype='meg', root=self.root).fpath
        if not path.exists():
            path = None

        return path

    @property
    def meg_crosstalk_fpath(self):
        """Find the matching Elekta/Neuromag/MEGIN crosstalk file.

        This requires that at least ``root`` and ``subject`` are set, and that
        ``datatype`` is either ``'meg'`` or ``None``.

        Returns
        -------
        path : pathlib.Path | None
            The path of the crosstalk file, or ``None`` if it couldn't be
            found.
        """
        if self.root is None or self.subject is None:
            raise ValueError('root and subject must be set.')
        if self.datatype not in (None, 'meg'):
            raise ValueError('Can only find crosstalk file for MEG datasets.')

        path = BIDSPath(subject=self.subject, session=self.session,
                        acquisition='crosstalk', suffix='meg',
                        extension='.fif', datatype='meg', root=self.root).fpath
        if not path.exists():
            path = None

        return path


def _get_matching_bidspaths_from_filesystem(bids_path):
    """Get matching file paths for a BIDS path.

    Assumes suffix and/or extension is not provided.
    """
    # extract relevant entities to find filepath
    sub, ses = bids_path.subject, bids_path.session
    datatype = bids_path.datatype
    basename, bids_root = bids_path.basename, bids_path.root

    if datatype is None:
        datatype = _infer_datatype(root=bids_root,
                                   sub=sub, ses=ses)

    data_dir = BIDSPath(subject=sub, session=ses, datatype=datatype,
                        root=bids_root).mkdir().directory

    # For BTI data, just return the directory with a '.pdf' extension
    # to facilitate reading in mne-bids
    bti_dir = op.join(data_dir, f'{basename}')
    if op.isdir(bti_dir):
        logger.info(f'Assuming BTi data in {bti_dir}')
        matching_paths = [f'{bti_dir}.pdf']
    # otherwise, search for valid file paths
    else:
        search_str = bids_root
        # parse down the BIDS directory structure
        if sub is not None:
            search_str = op.join(search_str, f'sub-{sub}')
        if ses is not None:
            search_str = op.join(search_str, f'ses-{ses}')
        if datatype is not None:
            search_str = op.join(search_str, datatype)
        else:
            search_str = op.join(search_str, '**')
        search_str = op.join(search_str, f'{basename}*')

        # Find all matching files in all supported formats.
        valid_exts = ALLOWED_FILENAME_EXTENSIONS
        matching_paths = glob.glob(search_str)
        matching_paths = [p for p in matching_paths
                          if _parse_ext(p)[1] in valid_exts]
    return matching_paths


def _check_non_sub_ses_entity(bids_path):
    """Check existence of non subject/session entities in BIDSPath."""
    if bids_path.task or bids_path.acquisition or \
            bids_path.run or bids_path.space or \
            bids_path.recording or bids_path.split or \
            bids_path.processing:
        return True
    return False


def print_dir_tree(folder, max_depth=None, return_str=False):
    """Recursively print a directory tree.

    Parameters
    ----------
    folder : str | pathlib.Path
        The folder for which to print the directory tree.
    max_depth : int
        The maximum depth into which to descend recursively for printing
        the directory tree.
    return_str : bool
        If ``True``, return the directory tree as a str instead of
        printing it.

    Returns
    -------
    str | None
        If `return_str` is ``True``, the directory tree is returned as a
        str. Else, ``None`` is returned and the directory tree is printed.
    """
    if not op.exists(folder):
        raise ValueError('Directory does not exist: {}'.format(folder))

    msg = '`max_depth` must be a positive integer or None'
    if not isinstance(max_depth, (int, type(None))):
        raise ValueError(msg)
    if max_depth is None:
        max_depth = float('inf')
    if max_depth < 0:
        raise ValueError(msg)

    if not isinstance(return_str, bool):
        raise ValueError('`return_str` must be either True or False.')
    outfile = None
    if return_str is True:
        outfile = StringIO()

    # Use max_depth same as the -L param in the unix `tree` command
    max_depth += 1

    # Base length of a tree branch, to normalize each tree's start to 0
    baselen = len(str(folder).split(os.sep)) - 1

    # Recursively walk through all directories
    for root, dirs, files in os.walk(folder, topdown=True):
        # Since we're using `topdown=True`, sorting `dirs` ensures that
        # `os.walk` will continue walking through directories in alphabetical
        # order. So although we're not actually using `dirs` anywhere below,
        # sorting it here is imperative to ensure the correct (alphabetical)
        # directory sort order in the output.
        dirs.sort()
        files.sort()

        # Check how far we have walked
        branchlen = len(root.split(os.sep)) - baselen

        # Only print if this is up to the depth we asked
        if branchlen <= max_depth:
            if branchlen <= 1:
                print('|{}'.format(op.basename(root) + os.sep), file=outfile)
            else:
                print('|{} {}'.format((branchlen - 1) * '---',
                                      op.basename(root) + os.sep),
                      file=outfile)

            # Only print files if we are NOT yet up to max_depth or beyond
            if branchlen < max_depth:
                for file in files:
                    print('|{} {}'.format(branchlen * '---', file),
                          file=outfile)

    if outfile is not None:
        return outfile.getvalue()


def _parse_ext(raw_fname, verbose=False):
    """Split a filename into its name and extension."""
    raw_fname = str(raw_fname)
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


def _infer_datatype_from_path(fname):
    # get the parent
    datatype = Path(fname).parent.name

    if any([datatype.startswith(entity) for entity in ['sub', 'ses']]):
        datatype = None

    if not datatype:
        datatype = None

    return datatype


def get_entities_from_fname(fname, on_error='raise'):
    """Retrieve a dictionary of BIDS entities from a filename.

    Entities not present in ``fname`` will be assigned the value of ``None``.

    Parameters
    ----------
    fname : BIDSPath | str
        The path to parse.
    on_error : 'raise' | 'warn' | 'ignore'
        If any unsupported labels in the filename are found and this is set
        to ``'raise'``, raise a ``RuntimeError``. If ``'warn'``,
        emit a warning and continue, and if ``'ignore'``,
        neither raise an exception nor a warning, and
        return all entities found. For example, currently MNE-BIDS does not
        support derivatives yet, but the ``desc`` entity label is used to
        differentiate different derivatives and will work with this function
        if ``on_error='ignore'``.

    Returns
    -------
    params : dict
        A dictionary with the keys corresponding to the BIDS entity names, and
        the values to the entity values encoded in the filename.

    Examples
    --------
    >>> fname = 'sub-01_ses-exp_run-02_meg.fif'
    >>> get_entities_from_fname(fname)
    {'subject': '01',
    'session': 'exp',
    'task': None,
    'acquisition': None,
    'run': '02',
    'processing': None,
    'space': None,
    'recording': None,
    'split': None,
    'suffix': 'meg'}
    """
    if on_error not in ('warn', 'raise', 'ignore'):
        raise ValueError(f'Acceptable values for on_error are: warn, raise, '
                         f'ignore, but got: {on_error}')

    fname = str(fname)  # to accept also BIDSPath or Path instances

    # filename keywords to the BIDS entity mapping
    entity_vals = list(ALLOWED_PATH_ENTITIES_SHORT.values())
    fname_vals = list(ALLOWED_PATH_ENTITIES_SHORT.keys())

    params = {key: None for key in entity_vals}
    idx_key = 0
    for match in re.finditer(param_regex, op.basename(fname)):
        key, value = match.groups()

        if on_error in ('raise', 'warn'):
            if key not in fname_vals:
                msg = (f'Unexpected entity "{key}" found in '
                       f'filename "{fname}"')
                if on_error == 'raise':
                    raise KeyError(msg)
                elif on_error == 'warn':
                    warn(msg)
                    continue
            if fname_vals.index(key) < idx_key:
                msg = (f'Entities in filename not ordered correctly.'
                       f' "{key}" should have occurred earlier in the '
                       f'filename "{fname}"')
                raise ValueError(msg)
            idx_key = fname_vals.index(key)

        key_short_hand = ALLOWED_PATH_ENTITIES_SHORT.get(key, key)
        params[key_short_hand] = value

    # parse suffix last
    last_entity = fname.split('-')[-1]
    if '_' in last_entity:
        suffix = last_entity.split('_')[-1]
        suffix, _ = _get_bids_suffix_and_ext(suffix)
        params['suffix'] = suffix

    return params


def _find_matching_sidecar(bids_path, suffix=None,
                           extension=None, on_error='raise'):
    """Try to find a sidecar file with a given suffix for a data file.

    Parameters
    ----------
    bids_path : BIDSPath
        Full name of the data file
    suffix : str | None
        The filename suffix. This is the entity after the last ``_``
        before the extension. E.g., ``'ieeg'``.
    extension : str | None
        The extension of the filename. E.g., ``'.json'``.
    on_error : 'raise' | 'warn' | 'ignore'
        If no matching sidecar file was found and this is set to ``'raise'``,
        raise a ``RuntimeError``. If ``'warn'``, emit a warning, and if
        ``'ignore'``, neither raise an exception nor a warning, and return
        ``None`` in both cases.

    Returns
    -------
    sidecar_fname : str | None
        Path to the identified sidecar file, or ``None`` if none could be found
        and ``on_error`` was set to ``'warn'`` or ``'ignore'``.

    """
    if on_error not in ('warn', 'raise', 'ignore'):
        raise ValueError(f'Acceptable values for on_error are: warn, raise, '
                         f'ignore, but got: {on_error}')

    bids_root = bids_path.root

    # search suffix is BIDS-suffix and extension
    search_suffix = ''
    if suffix is not None:
        search_suffix = search_suffix + suffix

        # do not search for suffix if suffix is explicitly passed
        bids_path = bids_path.copy()
        bids_path.check = False
        bids_path.update(suffix=None)

    if extension is not None:
        search_suffix = search_suffix + extension

        # do not search for extension if extension is explicitly passed
        bids_path = bids_path.copy()
        bids_path.check = False
        bids_path = bids_path.update(extension=None)

    # We only use subject and session as identifier, because all other
    # parameters are potentially not binding for metadata sidecar files
    search_str = f'sub-{bids_path.subject}'
    if bids_path.session is not None:
        search_str += f'_ses-{bids_path.session}'

    # Find all potential sidecar files, doing a recursive glob
    # from bids_root/sub_id/
    search_str = op.join(bids_root, f'sub-{bids_path.subject}',
                         '**', search_str + '*' + search_suffix)
    candidate_list = glob.glob(search_str, recursive=True)
    best_candidates = _find_best_candidates(bids_path.entities,
                                            candidate_list)
    if len(best_candidates) == 1:
        # Success
        return best_candidates[0]

    # We failed. Construct a helpful error message.
    # If this was expected, simply return None, otherwise, raise an exception.
    msg = None
    if len(best_candidates) == 0:
        msg = (f'Did not find any {search_suffix} '
               f'associated with {bids_path.basename}.')
    elif len(best_candidates) > 1:
        # More than one candidates were tied for best match
        msg = (f'Expected to find a single {suffix} file '
               f'associated with {bids_path.basename}, '
               f'but found {len(candidate_list)}: "{candidate_list}".')
    msg += '\n\nThe search_str was "{}"'.format(search_str)
    if on_error == 'raise':
        raise RuntimeError(msg)
    elif on_error == 'warn':
        warn(msg)

    return None


def _get_bids_suffix_and_ext(str_suffix):
    """Parse suffix for valid suffix and ext."""
    # no matter what the suffix is, suffix and extension are last
    suffix = str_suffix
    ext = None
    if '.' in str_suffix:
        # handle case of multiple '.' in extension
        split_str = str_suffix.split('.')
        suffix = split_str[0]
        ext = '.'.join(split_str[1:])
    return suffix, ext


def get_datatypes(root):
    """Get list of data types ("modalities") present in a BIDS dataset.

    Parameters
    ----------
    root : str | pathlib.Path
        Path to the root of the BIDS directory.

    Returns
    -------
    modalities : list of str
        List of the data types present in the BIDS dataset pointed to by
        `root`.

    """
    # Take all possible data types from "entity" table
    # (Appendix in BIDS spec)
    # https://bids-specification.readthedocs.io/en/stable/99-appendices/04-entity-table.html  # noqa
    datatype_list = ('anat', 'func', 'dwi', 'fmap', 'beh',
                     'meg', 'eeg', 'ieeg')
    datatypes = list()
    for root, dirs, files in os.walk(root):
        for dir in dirs:
            if dir in datatype_list and dir not in datatypes:
                datatypes.append(dir)

    return datatypes


def get_entity_vals(root, entity_key, *, ignore_subjects='emptyroom',
                    ignore_sessions=None, ignore_tasks=None, ignore_runs=None,
                    ignore_processings=None, ignore_spaces=None,
                    ignore_acquisitions=None, ignore_splits=None,
                    ignore_modalities=None, ignore_datatypes=None,
                    with_key=False):
    """Get list of values associated with an `entity_key` in a BIDS dataset.

    BIDS file names are organized by key-value pairs called "entities" [1]_.
    With this function, you can get all values for an entity indexed by its
    key.

    Parameters
    ----------
    root : str | pathlib.Path
        Path to the root of the BIDS directory.
    entity_key : str
        The name of the entity key to search for.
    ignore_subjects : str | iterable | None
        Subject(s) to ignore. By default, entities from the ``emptyroom``
        mock-subject are not returned. If ``None``, include all subjects.
    ignore_sessions : str | iterable | None
        Session(s) to ignore. If ``None``, include all sessions.
    ignore_tasks : str | iterable | None
        Task(s) to ignore. If ``None``, include all tasks.
    ignore_runs : str | iterable | None
        Run(s) to ignore. If ``None``, include all runs.
    ignore_processings : str | iterable | None
        Processing(s) to ignore. If ``None``, include all processings.
    ignore_spaces : str | iterable | None
        Space(s) to ignore. If ``None``, include all spaces.
    ignore_acquisitions : str | iterable | None
        Acquisition(s) to ignore. If ``None``, include all acquisitions.
    ignore_splits : str | iterable | None
        Split(s) to ignore. If ``None``, include all splits.
    ignore_modalities : str | iterable | None
        Modalities(s) to ignore. If ``None``, include all modalities.
    ignore_datatypes : str | iterable | None
        Datatype(s) to ignore. If ``None``, include all datatypes (i.e.
        ``anat``, ``ieeg``, ``eeg``, ``meg``, ``func``, etc.)
    with_key : bool
        If ``True``, returns the full entity with the key and the value. This
        will for example look like ``['sub-001', 'sub-002']``.
        If ``False`` (default), just returns the entity values. This
        will for example look like ``['001', '002']``.

    Returns
    -------
    entity_vals : list of str
        List of the values associated with an `entity_key` in the BIDS dataset
        pointed to by `root`.

    Examples
    --------
    >>> root = os.path.expanduser('~/mne_data/eeg_matchingpennies')
    >>> entity_key = 'sub'
    >>> get_entity_vals(root, entity_key)
    ['05', '06', '07', '08', '09', '10', '11']
    >>> get_entity_vals(root, entity_key, with_key=True)
    ['sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10', 'sub-11']

    Notes
    -----
    This function will scan the entire ``root``, except for a
    ``derivatives`` subfolder placed directly under ``root``.

    References
    ----------
    .. [1] https://bids-specification.rtfd.io/en/latest/02-common-principles.html#file-name-structure  # noqa: E501

    """
    entities = ('subject', 'task', 'session', 'run', 'processing', 'space',
                'acquisition', 'split', 'suffix')
    entities_abbr = ('sub', 'task', 'ses', 'run', 'proc', 'space', 'acq',
                     'split', 'suffix')
    entity_long_abbr_map = dict(zip(entities, entities_abbr))

    if entity_key not in entities:
        raise ValueError(f'`key` must be one of: {", ".join(entities)}. '
                         f'Got: {entity_key}')

    ignore_subjects = _ensure_tuple(ignore_subjects)
    ignore_sessions = _ensure_tuple(ignore_sessions)
    ignore_tasks = _ensure_tuple(ignore_tasks)
    ignore_runs = _ensure_tuple(ignore_runs)
    ignore_processings = _ensure_tuple(ignore_processings)
    ignore_spaces = _ensure_tuple(ignore_spaces)
    ignore_acquisitions = _ensure_tuple(ignore_acquisitions)
    ignore_splits = _ensure_tuple(ignore_splits)
    ignore_modalities = _ensure_tuple(ignore_modalities)

    p = re.compile(r'{}-(.*?)_'.format(entity_long_abbr_map[entity_key]))
    values = list()
    filenames = (Path(root)
                 .rglob(f'*{entity_long_abbr_map[entity_key]}-*_*'))
    for filename in filenames:
        # Ignore `derivatives` folder.
        if str(filename).startswith(op.join(root, 'derivatives')):
            continue

        if ignore_datatypes and filename.parent.name in ignore_datatypes:
            continue
        if ignore_subjects and any([filename.stem.startswith(f'sub-{s}_')
                                    for s in ignore_subjects]):
            continue
        if ignore_sessions and any([f'_ses-{s}_' in filename.stem
                                    for s in ignore_sessions]):
            continue
        if ignore_tasks and any([f'_task-{t}_' in filename.stem
                                 for t in ignore_tasks]):
            continue
        if ignore_runs and any([f'_run-{r}_' in filename.stem
                                for r in ignore_runs]):
            continue
        if ignore_processings and any([f'_proc-{p}_' in filename.stem
                                       for p in ignore_processings]):
            continue
        if ignore_spaces and any([f'_space-{s}_' in filename.stem
                                  for s in ignore_spaces]):
            continue
        if ignore_acquisitions and any([f'_acq-{a}_' in filename.stem
                                        for a in ignore_acquisitions]):
            continue
        if ignore_splits and any([f'_split-{s}_' in filename.stem
                                  for s in ignore_splits]):
            continue
        if ignore_modalities and any([f'_{k}' in filename.stem
                                      for k in ignore_modalities]):
            continue

        match = p.search(filename.stem)
        value = match.group(1)
        if with_key:
            value = f'{entity_long_abbr_map[entity_key]}-{value}'
        if value not in values:
            values.append(value)
    return sorted(values)


def _mkdir_p(path, overwrite=False, verbose=False):
    """Create a directory, making parent directories as needed [1].

    References
    ----------
    .. [1] stackoverflow.com/questions/600268/mkdir-p-functionality-in-python

    """
    if overwrite and op.isdir(path):
        sh.rmtree(path)
        if verbose is True:
            print(f'Clearing path: {path}')

    os.makedirs(path, exist_ok=True)
    if verbose is True:
        print(f'Creating folder: {path}')


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
        candidate_params = get_entities_from_fname(candidate)
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


def _get_datatypes_for_sub(*, root, sub, ses=None):
    """Retrieve data modalities for a specific subject and session."""
    subject_dir = op.join(root, f'sub-{sub}')
    if ses is not None:
        subject_dir = op.join(subject_dir, f'ses-{ses}')

    # TODO We do this to ensure we don't accidentally pick up any "spurious"
    # TODO sub-directories. But is that really necessary with valid BIDS data?
    modalities_in_dataset = get_datatypes(root=root)
    subdirs = [f.name for f in os.scandir(subject_dir) if f.is_dir()]
    available_modalities = [s for s in subdirs if s in modalities_in_dataset]
    return available_modalities


def _infer_datatype(*, root, sub, ses):
    # Check which suffix is available for this particular
    # subject & session. If we get no or multiple hits, throw an error.

    modalities = _get_datatypes_for_sub(root=root, sub=sub,
                                        ses=ses)

    # We only want to handle electrophysiological data here.
    allowed_recording_modalities = ['meg', 'eeg', 'ieeg']
    modalities = list(set(modalities) & set(allowed_recording_modalities))
    if not modalities:
        raise ValueError('No electrophysiological data found.')
    elif len(modalities) >= 2:
        msg = (f'Found data of more than one recording datatype. Please '
               f'pass the `suffix` parameter to specify which data to load. '
               f'Found the following modalitiess: {modalities}')
        raise RuntimeError(msg)

    assert len(modalities) == 1
    return modalities[0]


def _path_to_str(var):
    """Make sure var is a string or Path, return string representation."""
    if not isinstance(var, (Path, str)):
        raise ValueError(f"All path parameters must be either strings or "
                         f"pathlib.Path objects. Found type {type(var)}.")
    else:
        return str(var)


def _filter_fnames(fnames, *, subject=None, session=None, task=None,
                   acquisition=None, run=None, processing=None, recording=None,
                   space=None, split=None, suffix=None, extension=None):
    """Filter a list of BIDS filenames based on BIDS entity values.

    Parameters
    ----------
    fnames : iterable of pathlib.Path | iterable of str

    Returns
    -------
    list of pathlib.Path

    """
    sub_str = f'sub-{subject}' if subject else r'sub-([^_]+)'
    ses_str = f'_ses-{session}' if session else r'(|_ses-([^_]+))'
    task_str = f'_task-{task}' if task else r'(|_task-([^_]+))'
    acq_str = f'_acq-{acquisition}' if acquisition else r'(|_acq-([^_]+))'
    run_str = f'_run-{run}' if run else r'(|_run-([^_]+))'
    proc_str = f'_proc-{processing}' if processing else r'(|_proc-([^_]+))'
    rec_str = f'_rec-{recording}' if recording else r'(|_rec-([^_]+))'
    space_str = f'_space-{space}' if space else r'(|_space-([^_]+))'
    split_str = f'_split-{split}' if split else r'(|_split-([^_]+))'
    suffix_str = (f'_{suffix}' if suffix
                  else r'_(' + '|'.join(ALLOWED_FILENAME_SUFFIX) + ')')
    ext_str = extension if extension else r'.([^_]+)'

    regexp = (sub_str + ses_str + task_str + acq_str + run_str + proc_str +
              rec_str + space_str + split_str + suffix_str + ext_str)

    # Convert to str so we can apply the regexp ...
    fnames = [str(f) for f in fnames]

    # https://stackoverflow.com/a/51246151/1944216
    fnames_filtered = sorted(filter(re.compile(regexp).match, fnames))

    # ... and return Paths.
    fnames_filtered = [Path(f) for f in fnames_filtered]
    return fnames_filtered
