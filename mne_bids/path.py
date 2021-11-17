"""BIDS compatible path functionality."""
# Authors: Adam Li <adam2392@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD-3-Clause
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
import json
from typing import Optional

import numpy as np
from mne.utils import warn, logger, _validate_type, verbose, _check_fname

from mne_bids.config import (
    ALLOWED_PATH_ENTITIES, ALLOWED_FILENAME_EXTENSIONS,
    ALLOWED_FILENAME_SUFFIX, ALLOWED_PATH_ENTITIES_SHORT,
    ALLOWED_DATATYPES, SUFFIX_TO_DATATYPE, ALLOWED_DATATYPE_EXTENSIONS,
    ALLOWED_SPACES,
    reader, ENTITY_VALUE_TYPE)
from mne_bids.utils import (_check_key_val, _check_empty_room_basename,
                            param_regex, _ensure_tuple)


def _find_matched_empty_room(bids_path):
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
    raw = read_raw_bids(bids_path=bids_path)
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
        # get entities from filenamme
        er_bids_path = get_bids_path_from_fname(er_fname, check=False)
        er_bids_path.subject = 'emptyroom'  # er subject entity is different
        er_bids_path.root = bids_root
        er_meas_date = None

        # Try to extract date from filename.
        if er_bids_path.session is not None:
            try:
                er_meas_date = datetime.strptime(
                    er_bids_path.session, '%Y%m%d')
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
        The acquisition session. Corresponds to "ses".
    task : str | None
        The experimental task. Corresponds to "task".
    acquisition: str | None
        The acquisition parameters. Corresponds to "acq".
    run : int | None
        The run number. Corresponds to "run".
    processing : str | None
        The processing label. Corresponds to "proc".
    recording : str | None
        The recording name. Corresponds to "rec".
    space : str | None
        The coordinate space for anatomical and sensor location
        files (e.g., ``*_electrodes.tsv``, ``*_markers.mrk``).
        Corresponds to "space".
        Note that valid values for ``space`` must come from a list
        of BIDS keywords as described in the BIDS specification.
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
        'beh', 'physio', 'stim'
    extension : str | None
        The extension of the filename. E.g., ``'.json'``.
    datatype : str
        The BIDS data type, e.g., ``'anat'``, ``'func'``, ``'eeg'``, ``'meg'``,
        ``'ieeg'``.
    root : path-like | None
        The root directory of the BIDS dataset.
    check : bool
        If ``True``, enforces BIDS conformity. Defaults to ``True``.

    Attributes
    ----------
    entities : dict
        The dictionary of the BIDS entities and their values:
        ``subject``, ``session``, ``task``, ``acquisition``,
        ``run``, ``processing``, ``space``, ``recording``, ``split``,
        ``suffix``, and ``extension``.
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
        Whether to enforce BIDS conformity.

    Examples
    --------
    Generate a BIDSPath object and inspect it

    >>> bids_path = BIDSPath(subject='test', session='two', task='mytask',
    ...                      suffix='ieeg', extension='.edf')
    >>> print(bids_path.basename)
    sub-test_ses-two_task-mytask_ieeg.edf
    >>> bids_path
    BIDSPath(
    root: None
    datatype: ieeg
    basename: sub-test_ses-two_task-mytask_ieeg.edf)

    Copy and update multiple entities at once

    >>> new_bids_path = bids_path.copy().update(subject='test2',
    ...                                         session='one')
    >>> print(new_bids_path.basename)
    sub-test2_ses-one_task-mytask_ieeg.edf

    Printing a BIDSPath will show a relative path when `root` is not set

    >>> print(new_bids_path)
    sub-test2/ses-one/ieeg/sub-test2_ses-one_task-mytask_ieeg.edf

    Setting `suffix` without an identifiable datatype will make
    BIDSPath try to guess the datatype

    >>> new_bids_path = new_bids_path.update(suffix='channels',
    ...                                      extension='.tsv')
    >>> print(new_bids_path)
    sub-test2/ses-one/ieeg/sub-test2_ses-one_task-mytask_channels.tsv

    You can set a new root for the BIDS dataset. Let's see what the
    different properties look like for our object:

    >>> new_bids_path = new_bids_path.update(root='/bids_dataset')
    >>> print(new_bids_path.root.as_posix())
    /bids_dataset
    >>> print(new_bids_path.basename)
    sub-test2_ses-one_task-mytask_channels.tsv
    >>> print(new_bids_path)
    /bids_dataset/sub-test2/ses-one/ieeg/sub-test2_ses-one_task-mytask_channels.tsv
    >>> print(new_bids_path.directory.as_posix())
    /bids_dataset/sub-test2/ses-one/ieeg

    Notes
    -----
    BIDS entities are generally separated with a ``"_"`` character, while
    entity key/value pairs are separated with a ``"-"`` character.
    There are checks performed to make sure that there are no ``'-'``, ``'_'``,
    or ``'/'`` characters contained in any entity keys or values.

    To represent a filename such as ``dataset_description.json``,
    one can set ``check=False``, and pass ``suffix='dataset_description'``
    and ``extension='.json'``.

    ``BIDSPath`` can also be used to represent file and folder names of data
    types that are not yet supported through MNE-BIDS, but are recognized by
    BIDS. For example, one can set ``datatype`` to ``dwi`` or ``func`` and
    pass ``check=False`` to represent diffusion-weighted imaging and
    functional MRI paths.
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

    @property
    def subject(self) -> Optional[str]:
        """The subject ID."""
        return self._subject

    @subject.setter
    def subject(self, value):
        self.update(subject=value)

    @property
    def session(self) -> Optional[str]:
        """The acquisition session."""
        return self._session

    @session.setter
    def session(self, value):
        self.update(session=value)

    @property
    def task(self) -> Optional[str]:
        """The experimental task."""
        return self._task

    @task.setter
    def task(self, value):
        self.update(task=value)

    @property
    def run(self) -> Optional[str]:
        """The run number."""
        return self._run

    @run.setter
    def run(self, value):
        self.update(run=value)

    @property
    def acquisition(self) -> Optional[str]:
        """The acquisition parameters."""
        return self._acquisition

    @acquisition.setter
    def acquisition(self, value):
        self.update(acquisition=value)

    @property
    def processing(self) -> Optional[str]:
        """The processing label."""
        return self._processing

    @processing.setter
    def processing(self, value):
        self.update(processing=value)

    @property
    def recording(self) -> Optional[str]:
        """The recording name."""
        return self._recording

    @recording.setter
    def recording(self, value):
        self.update(recording=value)

    @property
    def space(self) -> Optional[str]:
        """The coordinate space for an anatomical or sensor position file."""
        return self._space

    @space.setter
    def space(self, value):
        self.update(space=value)

    @property
    def suffix(self) -> Optional[str]:
        """The filename suffix."""
        return self._suffix

    @suffix.setter
    def suffix(self, value):
        self.update(suffix=value)

    @property
    def root(self) -> Optional[Path]:
        """The root directory of the BIDS dataset."""
        return self._root

    @root.setter
    def root(self, value):
        self.update(root=value)

    @property
    def datatype(self) -> Optional[str]:
        """The BIDS data type, e.g. ``'anat'``, ``'meg'``, ``'eeg'``."""
        return self._datatype

    @datatype.setter
    def datatype(self, value):
        self.update(datatype=value)

    @property
    def split(self) -> Optional[str]:
        """The split of the continuous recording file for ``.fif`` data."""
        return self._split

    @split.setter
    def split(self, value):
        self.update(split=value)

    @property
    def extension(self) -> Optional[str]:
        """The extension of the filename, including a leading period."""
        return self._extension

    @extension.setter
    def extension(self, value):
        self.update(extension=value)

    def __str__(self):
        """Return the string representation of the path."""
        return str(self.fpath.as_posix())

    def __repr__(self):
        """Representation in the style of `pathlib.Path`."""
        root = self.root.as_posix() if self.root is not None else None

        return f'{self.__class__.__name__}(\n' \
               f'root: {root}\n' \
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
        bidspath : BIDSPath
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
        self : BIDSPath
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
                    if self.extension is None:
                        valid_exts = \
                            sum(ALLOWED_DATATYPE_EXTENSIONS.values(), [])
                    else:
                        valid_exts = [self.extension]
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
        - ``datatype`` should be one of: ``anat``, ``eeg``, ``ieeg``, ``meg``
        - ``extension`` should be one of the accepted file
        extensions in the file path: ``.con``, ``.sqd``, ``.fif``,
        ``.pdf``, ``.ds``, ``.vhdr``, ``.edf``, ``.bdf``, ``.set``,
        ``.edf``, ``.set``, ``.mef``, ``.nwb``
        - ``suffix`` should be one of the acceptable file suffixes in: ``meg``,
        ``markers``, ``eeg``, ``ieeg``, ``T1w``,
        ``participants``, ``scans``, ``electrodes``, ``channels``,
        ``coordsystem``, ``events``, ``headshape``, ``digitizer``,
        ``beh``, ``physio``, ``stim``
        - Depending on the modality of the data (EEG, MEG, iEEG),
        ``space`` should be a valid string according to Appendix VIII
        in the BIDS specification.

        Parameters
        ----------
        check : None | bool
            If a boolean, controls whether to enforce BIDS conformity. This
            will set the ``.check`` attribute accordingly. If ``None``, rely on
            the existing ``.check`` attribute instead, which is set upon
            `mne_bids.BIDSPath` instantiation. Defaults to ``None``.
        **kwargs : dict
            It can contain updates for valid BIDS path entities:
            'subject', 'session', 'task', 'acquisition', 'processing', 'run',
            'recording', 'space', 'suffix', 'split', 'extension',
            or updates for 'root' or 'datatype'.

        Returns
        -------
        bidspath : BIDSPath
            The updated instance of BIDSPath.

        Examples
        --------
        If one creates a bids basename using
        :func:`mne_bids.BIDSPath`:

        >>> bids_path = BIDSPath(subject='test', session='two',
        ...                      task='mytask', suffix='channels',
        ...                      extension='.tsv')
        >>> print(bids_path.basename)
        sub-test_ses-two_task-mytask_channels.tsv
        >>> # Then, one can update this `BIDSPath` object in place
        >>> bids_path = bids_path.update(acquisition='test', suffix='ieeg',
        ...                              extension='.vhdr', task=None)
        >>> print(bids_path.basename)
        sub-test_ses-two_acq-test_ieeg.vhdr
        """
        # Update .check attribute
        if check is not None:
            self.check = check

        for key, val in kwargs.items():
            if key == 'root':
                _validate_type(val, types=('path-like', None), item_name=key)
                continue

            if key == 'datatype':
                if val is not None and val not in ALLOWED_DATATYPES \
                        and self.check:
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
                kwargs['extension'] = f'.{extension}'

        # error check entities
        for key, val in kwargs.items():

            # check if there are any characters not allowed
            if val is not None and key != 'root':
                if key == 'suffix' and not self.check:
                    # suffix may skip a check if check=False to allow
                    # things like "dataset_description.json"
                    pass
                else:
                    _check_key_val(key, val)

            # set entity value, ensuring `root` is a Path
            if val is not None and key == 'root':
                val = Path(val).expanduser()
            setattr(self, f'_{key}', val)

        # infer datatype if suffix is uniquely the datatype
        if self.datatype is None and \
                self.suffix in SUFFIX_TO_DATATYPE:
            self._datatype = SUFFIX_TO_DATATYPE[self.suffix]

        # Perform a check of the entities.
        self._check()
        return self

    def match(self, check=False):
        """Get a list of all matching paths in the root directory.

        Performs a recursive search, starting in ``.root`` (if set), based on
        `BIDSPath.entities` object. Ignores ``.json`` files.

        Parameters
        ----------
        check : bool
            If ``True``, only returns paths that conform to BIDS. If ``False``
            (default), the ``.check`` attribute of the returned
            `mne_bids.BIDSPath` object will be set to ``True`` for paths that
            do conform to BIDS, and to ``False`` for those that don't.

        Returns
        -------
        bids_paths : list of mne_bids.BIDSPath
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

        paths = self.root.rglob(search_str)
        # Only keep files (not directories), and omit the JSON sidecars.
        paths = [p for p in paths
                 if p.is_file() and p.suffix != '.json']
        fnames = _filter_fnames(paths, suffix=self.suffix,
                                extension=self.extension,
                                **self.entities)

        bids_paths = []
        for fname in fnames:
            # Form the BIDSPath object.
            # To check whether the BIDSPath is conforming to BIDS if
            # check=True, we first instantiate without checking and then run
            # the check manually, allowing us to be more specific about the
            # exception to catch
            datatype = _infer_datatype_from_path(fname)
            bids_path = get_bids_path_from_fname(fname, check=False)
            bids_path.root = self.root
            bids_path.datatype = datatype
            bids_path.check = True

            try:
                bids_path._check()
            except ValueError:
                # path is not BIDS-compatible
                if check:  # skip!
                    continue
                else:
                    bids_path.check = False

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

            # labels from space entity must come from list (appendix VIII)
            space = self.space
            if space is not None:
                datatype = getattr(self, 'datatype', None)
                if datatype is None:
                    raise ValueError('You must define datatype if you want to '
                                     'use space in your BIDSPath.')

                allowed_spaces_for_dtype = ALLOWED_SPACES.get(datatype, None)
                if allowed_spaces_for_dtype is None:
                    raise ValueError(f'space entity is not valid for datatype '
                                     f'{self.datatype}')
                elif space not in allowed_spaces_for_dtype:
                    raise ValueError(f'space ({space}) is not valid for '
                                     f'datatype ({self.datatype}).\n'
                                     f'Should be one of '
                                     f'{allowed_spaces_for_dtype}')
                else:
                    pass

            # error check suffix
            suffix = self.suffix
            if suffix is not None and \
                    suffix not in ALLOWED_FILENAME_SUFFIX:
                raise ValueError(f'Suffix {suffix} is not allowed. '
                                 f'Use one of these suffixes '
                                 f'{ALLOWED_FILENAME_SUFFIX}.')

    @verbose
    def find_empty_room(self, use_sidecar_only=False, verbose=None):
        """Find the corresponding empty-room file of an MEG recording.

        This will only work if the ``.root`` attribute of the
        :class:`mne_bids.BIDSPath` instance has been set.

        Parameters
        ----------
        use_sidecar_only : bool
            Whether to only check the ``AssociatedEmptyRoom`` entry in the
            sidecar JSON file or not. If ``False``, first look for the entry,
            and if unsuccessful, try to find the best-matching empty-room
            recording in the dataset based on the measurement date.

        Returns
        -------
        BIDSPath | None
            The path corresponding to the best-matching empty-room measurement.
            Returns ``None`` if none was found.
        %(verbose)s
        """
        if self.datatype not in ('meg', None):
            raise ValueError('Empty-room data is only supported for MEG '
                             'datasets')

        if self.root is None:
            raise ValueError('The root of the "bids_path" must be set. '
                             'Please use `bids_path.update(root="<root>")` '
                             'to set the root of the BIDS folder to read.')

        sidecar_fname = _find_matching_sidecar(self, extension='.json')
        with open(sidecar_fname, 'r', encoding='utf-8') as f:
            sidecar_json = json.load(f)

        if 'AssociatedEmptyRoom' in sidecar_json:
            logger.info('Using "AssociatedEmptyRoom" entry from MEG sidecar '
                        'file to retrieve empty-room path.')
            emptytoom_path = sidecar_json['AssociatedEmptyRoom']
            er_bids_path = get_bids_path_from_fname(emptytoom_path)
            er_bids_path.root = self.root
            er_bids_path.datatype = 'meg'
        elif use_sidecar_only:
            logger.info(
                'The MEG sidecar file does not contain an '
                '"AssociatedEmptyRoom" entry. Aborting search for an '
                'empty-room recording, as you passed use_sidecar_only=True'
            )
            return None
        else:
            logger.info(
                'The MEG sidecar file does not contain an '
                '"AssociatedEmptyRoom" entry. Will try to find a matching '
                'empty-room recording based on the measurement date …'
            )
            er_bids_path = _find_matched_empty_room(self)

        if er_bids_path is not None:
            assert er_bids_path.fpath.exists()

        return er_bids_path

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


def _print_lines_with_entry(file, entry, folder, is_tsv, line_numbers,
                            outfile):
    """Print the lines that contain the entry.

    Parameters
    ----------
    file : str
        The text file to look though.
    entry : str
        The string to look in the text file for.
    folder : str
        The base folder for relative file path printing.
    is_tsv : bool
        If ``True``, things that format a tsv nice will be used.
    line_numbers : bool
        Whether to include line numbers in the printout.
    outfile : io.StringIO | None
        The argument to pass to `print` for `file`. If ``None``,
        prints to the console, else a string is printed to.
    """
    entry_lines = list()
    with open(file, 'r', encoding='utf-8-sig') as fid:
        if is_tsv:  # format tsv files nicely
            header = _truncate_tsv_line(fid.readline())
            if line_numbers:
                header = f'1    {header}'
            header = header.rstrip()
        for i, line in enumerate(fid):
            if entry in line:
                if is_tsv:
                    line = _truncate_tsv_line(line)
                if line_numbers:
                    line = str(i + 2) + (5 - len(str(i + 2))) * ' ' + line
                entry_lines.append(line.rstrip())
    if entry_lines:
        print(op.relpath(file, folder), file=outfile)
        if is_tsv:
            print(f'    {header}', file=outfile)
        if len(entry_lines) > 10:
            entry_lines = entry_lines[:10]
            entry_lines.append('...')
        for line in entry_lines:
            print(f'    {line}', file=outfile)


def _truncate_tsv_line(line, lim=10):
    """Truncate a line to the specified number of characters."""
    return ''.join([str(val) + (lim - len(val)) * ' ' if
                    len(val) < lim else f'{val[:lim - 1]} '
                    for val in line.split('\t')])


def search_folder_for_text(entry, folder, extensions=('.json', '.tsv'),
                           line_numbers=True, return_str=False):
    """Find any particular string entry in the text files of a folder.

    .. note:: This is a search function like `grep
              <https://man7.org/linux/man-pages/man1/fgrep.1.html>`_
              that is formatted nicely for BIDS datasets.

    Parameters
    ----------
    entry : str
        The string to search for
    folder : path-like
        The folder in which to search.
    extensions : list | tuple | str
        The extensions to search through. Default is ``json`` and
        ``tsv`` which are the BIDS sidecar file types.
    line_numbers : bool
        Whether to include line numbers.
    return_str : bool
        If ``True``, return the fields with "n/a" as a str instead of
        printing them.

    Returns
    -------
    str | None
        If `return_str` is ``True``, the fields are returned as a
        string. Else, ``None`` is returned and the fields are printed.
    """
    _validate_type(entry, str, 'entry')
    if not op.isdir(folder):
        raise ValueError('{folder} is not a directory')
    folder = Path(folder)  # ensure pathlib.Path

    extensions = (extensions,) if isinstance(extensions, str) else extensions
    _validate_type(extensions, (tuple, list))
    _validate_type(line_numbers, bool, 'line_numbers')
    _validate_type(return_str, bool, 'return_str')
    outfile = StringIO() if return_str else None

    for extension in extensions:
        for file in folder.rglob('*' + extension):
            _print_lines_with_entry(file, entry, folder, extension == '.tsv',
                                    line_numbers, outfile)

    if outfile is not None:
        return outfile.getvalue()


def _check_max_depth(max_depth):
    """Check that max depth is a proper input."""
    msg = '`max_depth` must be a positive integer or None'
    if not isinstance(max_depth, (int, type(None))):
        raise ValueError(msg)
    if max_depth is None:
        max_depth = float('inf')
    if max_depth < 0:
        raise ValueError(msg)
    # Use max_depth same as the -L param in the unix `tree` command
    max_depth += 1
    return max_depth


def print_dir_tree(folder, max_depth=None, return_str=False):
    """Recursively print a directory tree.

    Parameters
    ----------
    folder : path-like
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
        string. Else, ``None`` is returned and the directory tree is printed.
    """
    if not op.exists(folder):
        raise ValueError('Directory does not exist: {}'.format(folder))

    max_depth = _check_max_depth(max_depth)

    _validate_type(return_str, bool, 'return_str')
    outfile = StringIO() if return_str else None

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


def _parse_ext(raw_fname):
    """Split a filename into its name and extension."""
    raw_fname = str(raw_fname)
    fname, ext = os.path.splitext(raw_fname)
    # BTi data is the only file format that does not have a file extension
    if ext == '' or 'c,rf' in fname:
        logger.info('Found no extension for raw file, assuming "BTi" format '
                    'and appending extension .pdf')
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


@verbose
def get_bids_path_from_fname(fname, check=True, verbose=None):
    """Retrieve a BIDSPath object from a filename.

    Parameters
    ----------
    fname : path-like
        The path to parse a `BIDSPath` from.
    check : bool
        Whether to check if the generated `BIDSPath` complies with the BIDS
        specification, i.e., whether all included entities and the suffix are
        valid.
    %(verbose)s

    Returns
    -------
    bids_path : BIDSPath
        The BIDS path object.
    """
    fpath = Path(fname)
    fname = fpath.name

    entities = get_entities_from_fname(fname)

    # parse suffix and extension
    last_entity = fname.split('-')[-1]
    if '_' in last_entity:
        suffix = last_entity.split('_')[-1]
        suffix, extension = _get_bids_suffix_and_ext(suffix)
    else:
        suffix = None
        extension = Path(fname).suffix
        if extension == '':
            extension = None

    datatype = _infer_datatype_from_path(fpath)

    # find root and datatype if it exists
    if fpath.parent == '':
        root = None
    else:
        root_level = 0
        # determine root if it's there
        if entities['subject'] is not None:
            root_level += 1
        if entities['session'] is not None:
            root_level += 1
        if suffix != 'scans':
            root_level += 1

        if root_level:
            root = fpath.parent
            for _ in range(root_level):
                root = root.parent

    bids_path = BIDSPath(root=root, datatype=datatype, suffix=suffix,
                         extension=extension, **entities, check=check)
    if verbose:
        logger.info(f'From {fpath}, formed a BIDSPath: {bids_path}.')
    return bids_path


@verbose
def get_entities_from_fname(fname, on_error='raise', verbose=None):
    """Retrieve a dictionary of BIDS entities from a filename.

    Entities not present in ``fname`` will be assigned the value of ``None``.

    Parameters
    ----------
    fname : BIDSPath | path-like
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
    %(verbose)s

    Returns
    -------
    params : dict
        A dictionary with the keys corresponding to the BIDS entity names, and
        the values to the entity values encoded in the filename.

    Examples
    --------
    >>> fname = 'sub-01_ses-exp_run-02_meg.fif'
    >>> get_entities_from_fname(fname)
    {'subject': '01', \
'session': 'exp', \
'task': None, \
'acquisition': None, \
'run': '02', \
'processing': None, \
'space': None, \
'recording': None, \
'split': None}
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
    return params


def _find_matching_sidecar(bids_path, suffix=None,
                           extension=None, on_error='raise'):
    """Try to find a sidecar file with a given suffix for a data file.

    Parameters
    ----------
    bids_path : BIDSPath
        Full name of the data file.
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
    if suffix is None and bids_path.suffix is not None:
        search_suffix = bids_path.suffix
    elif suffix is not None:
        search_suffix = suffix

        # do not search for suffix if suffix is explicitly passed
        bids_path = bids_path.copy()
        bids_path.check = False
        bids_path.update(suffix=None)

    if extension is None and bids_path.extension is not None:
        search_suffix = search_suffix + bids_path.extension
    elif extension is not None:
        search_suffix = search_suffix + extension

        # do not search for extension if extension is explicitly passed
        bids_path = bids_path.copy()
        bids_path.check = False
        bids_path = bids_path.update(extension=None)

    # We only use subject and session as identifier, because all other
    # parameters are potentially not binding for metadata sidecar files
    search_str_filename = f'sub-{bids_path.subject}'
    if bids_path.session is not None:
        search_str_filename += f'_ses-{bids_path.session}'

    # Find all potential sidecar files, doing a recursive glob
    # from bids_root/sub-*, potentially taking into account the data type
    search_dir = Path(bids_root) / f'sub-{bids_path.subject}'
    # ** -> don't forget about potentially present session directories
    if bids_path.datatype is None:
        search_dir = search_dir / '**'
    else:
        search_dir = search_dir / '**' / bids_path.datatype

    search_str_complete = str(
        search_dir / f'{search_str_filename}*{search_suffix}'
    )

    candidate_list = glob.glob(search_str_complete, recursive=True)
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
    msg += f'\n\nThe search_str was "{search_str_complete}"'
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


@verbose
def get_datatypes(root, verbose=None):
    """Get list of data types ("modalities") present in a BIDS dataset.

    Parameters
    ----------
    root : path-like
        Path to the root of the BIDS directory.
    %(verbose)s

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


@verbose
def get_entity_vals(root, entity_key, *, ignore_subjects='emptyroom',
                    ignore_sessions=None, ignore_tasks=None, ignore_runs=None,
                    ignore_processings=None, ignore_spaces=None,
                    ignore_acquisitions=None, ignore_splits=None,
                    ignore_modalities=None, ignore_datatypes=None,
                    ignore_dirs=('derivatives', 'sourcedata'), with_key=False,
                    verbose=None):
    """Get list of values associated with an `entity_key` in a BIDS dataset.

    BIDS file names are organized by key-value pairs called "entities" [1]_.
    With this function, you can get all values for an entity indexed by its
    key.

    Parameters
    ----------
    root : path-like
        Path to the "root" directory from which to start traversing to gather
        BIDS entities from file- and folder names. This will commonly be the
        BIDS root, but it may also be a subdirectory inside of a BIDS dataset,
        e.g., the ``sub-X`` directory of a hypothetical subject ``X``.

        .. note:: This function searches the names of all files and directories
                  nested within ``root``. Depending on the size of your
                  dataset and storage system, searching the entire BIDS dataset
                  may take a **considerable** amount of time (seconds up to
                  several minutes). If you find yourself running into such
                  performance issues, consider limiting the search to only a
                  subdirectory in the dataset, e.g., to a single subject or
                  session only.

    entity_key : str
        The name of the entity key to search for.
    ignore_subjects : str | array-like of str | None
        Subject(s) to ignore. By default, entities from the ``emptyroom``
        mock-subject are not returned. If ``None``, include all subjects.
    ignore_sessions : str | array-like of str | None
        Session(s) to ignore. If ``None``, include all sessions.
    ignore_tasks : str | array-like of str | None
        Task(s) to ignore. If ``None``, include all tasks.
    ignore_runs : str | array-like of str | None
        Run(s) to ignore. If ``None``, include all runs.
    ignore_processings : str | array-like of str | None
        Processing(s) to ignore. If ``None``, include all processings.
    ignore_spaces : str | array-like of str | None
        Space(s) to ignore. If ``None``, include all spaces.
    ignore_acquisitions : str | array-like of str | None
        Acquisition(s) to ignore. If ``None``, include all acquisitions.
    ignore_splits : str | array-like of str | None
        Split(s) to ignore. If ``None``, include all splits.
    ignore_modalities : str | array-like of str | None
        Modalities(s) to ignore. If ``None``, include all modalities.
    ignore_datatypes : str | array-like of str | None
        Datatype(s) to ignore. If ``None``, include all datatypes (i.e.
        ``anat``, ``ieeg``, ``eeg``, ``meg``, ``func``, etc.)
    ignore_dirs : str | array-like of str | None
        Directories nested directly within ``root`` to ignore. If ``None``,
        include all directories in the search.

        .. versionadded:: 0.9
    with_key : bool
        If ``True``, returns the full entity with the key and the value. This
        will for example look like ``['sub-001', 'sub-002']``.
        If ``False`` (default), just returns the entity values. This
        will for example look like ``['001', '002']``.
    %(verbose)s

    Returns
    -------
    entity_vals : list of str
        List of the values associated with an `entity_key` in the BIDS dataset
        pointed to by `root`.

    Examples
    --------
    >>> root = Path('./mne_bids/tests/data/tiny_bids').absolute()
    >>> entity_key = 'subject'
    >>> get_entity_vals(root, entity_key)
    ['01']
    >>> get_entity_vals(root, entity_key, with_key=True)
    ['sub-01']

    Notes
    -----
    This function will scan the entire ``root``, except for a
    ``derivatives`` subfolder placed directly under ``root``.

    References
    ----------
    .. [1] https://bids-specification.rtfd.io/en/latest/02-common-principles.html#file-name-structure  # noqa: E501

    """
    root = _check_fname(
        fname=root,
        overwrite='read',
        must_exist=True,
        need_dir=True,
        name='Root directory'
    )
    root = Path(root).expanduser()

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

    ignore_dirs = _ensure_tuple(ignore_dirs)
    existing_ignore_dirs = [
        root / d for d in ignore_dirs
        if (root / d).exists() and (root / d).is_dir()
    ]
    ignore_dirs = _ensure_tuple(existing_ignore_dirs)

    p = re.compile(r'{}-(.*?)_'.format(entity_long_abbr_map[entity_key]))
    values = list()
    filenames = root.glob(f'**/*{entity_long_abbr_map[entity_key]}-*_*')

    for filename in filenames:
        # Skip ignored directories
        # XXX In Python 3.9, we can use Path.is_relative_to() here
        if any([
            str(filename).startswith(str(ignore_dir))
            for ignore_dir in ignore_dirs
        ]):
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


def _mkdir_p(path, overwrite=False):
    """Create a directory, making parent directories as needed [1].

    References
    ----------
    .. [1] stackoverflow.com/questions/600268/mkdir-p-functionality-in-python

    """
    if overwrite and op.isdir(path):
        sh.rmtree(path)
        logger.info(f'Clearing path: {path}')

    os.makedirs(path, exist_ok=True)
    if not op.isdir(path):
        logger.info(f'Creating folder: {path}')


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
    """Filter a list of BIDS filenames / paths based on BIDS entity values.

    Parameters
    ----------
    fnames : iterable of pathlib.Path | iterable of str

    Returns
    -------
    list of pathlib.Path

    """
    leading_path_str = r'.*\/?'  # nothing or something ending with a `/`
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

    regexp = (
        leading_path_str +
        sub_str + ses_str + task_str + acq_str + run_str + proc_str +
        rec_str + space_str + split_str + suffix_str + ext_str
    )

    # Convert to str so we can apply the regexp ...
    fnames = [str(f) for f in fnames]

    # https://stackoverflow.com/a/51246151/1944216
    fnames_filtered = sorted(filter(re.compile(regexp).match, fnames))

    # ... and return Paths.
    fnames_filtered = [Path(f) for f in fnames_filtered]
    return fnames_filtered
