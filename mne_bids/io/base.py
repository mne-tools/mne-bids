"""Basic IO functionality: reading and writing BIDS directories."""
# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#          Chris Holdgraf <choldgraf@berkeley.edu>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Matt Sanderson <matt.sanderson@mq.edu.au>
#
# License: BSD (3-clause)
import os
import os.path as op
import glob
import shutil as sh
from datetime import datetime
from collections import OrderedDict

import numpy as np
from numpy.testing import assert_array_equal
from mne.io import BaseRaw
from mne.utils import check_version


from mne_bids.io._utils import (_read_raw, _parse_ext, reader,
                                _parse_bids_filename, _handle_kind,
                                ORIENTATION, UNITS, MANUFACTURERS,
                                ALLOWED_EXTENSIONS,
                                _participants_tsv, _participants_json,
                                _scans_tsv, _coordsystem_json, _read_events,
                                _events_tsv, _sidecar_json, _channels_tsv,
                                _mkdir_p, _check_key_val, _check_types,
                                _write_json)
from mne_bids.config import BIDS_VERSION
from mne_bids.tsv_handler import _from_tsv, _drop
from mne.copyfiles import (copyfile_ctf, copyfile_brainvision, copyfile_eeglab,
                           copyfile_bti)


def read_raw_bids(bids_fname, bids_root, return_events=True,
                  verbose=True):
    """Read BIDS compatible data.

    Parameters
    ----------
    bids_fname : str
        Full name of the data file
    bids_root : str
        Path to root of the BIDS folder
    return_events: bool
        Whether to return events or not. Default is True.
    verbose : bool
        The verbosity level

    Returns
    -------
    raw : instance of Raw
        The data as MNE-Python Raw object.
    events : ndarray, shape = (n_events, 3)
        The first column contains the event time in samples and the third
        column contains the event id. The second column is ignored for now but
        typically contains the value of the trigger channel either immediately
        before the event or immediately after.
    event_id : dict
        Dictionary of events in the raw data mapping from the event value
        to an index.

    """
    from mne_bids.utils import _parse_bids_filename

    # Full path to data file is needed so that mne-bids knows
    # what is the modality -- meg, eeg, ieeg to read
    bids_basename = '_'.join(bids_fname.split('_')[:-1])
    kind = bids_fname.split('_')[-1].split('.')[0]
    _, ext = _parse_ext(bids_fname)

    params = _parse_bids_filename(bids_basename, verbose)
    kind_dir = op.join(bids_root, 'sub-%s' % params['sub'],
                       'ses-%s' % params['ses'], kind)

    config = None
    if ext in ('.fif', '.ds', '.vhdr', '.edf', '.bdf', '.set'):
        bids_fname = op.join(kind_dir,
                             bids_basename + '_%s%s' % (kind, ext))
    elif ext == '.sqd':
        data_path = op.join(kind_dir, bids_basename + '_%s' % kind)
        bids_fname = op.join(data_path,
                             bids_basename + '_%s%s' % (kind, ext))
    elif ext == '.pdf':
        bids_raw_folder = op.join(kind_dir, bids_basename + '_%s' % kind)
        bids_fname = glob.glob(op.join(bids_raw_folder, 'c,rf*'))[0]
        config = op.join(bids_raw_folder, 'config')
    raw = _read_raw(bids_fname, electrode=None, hsp=None, hpi=None,
                    config=config, montage=None, verbose=None)

    events_fname = op.join(kind_dir, bids_basename + '_events.tsv')

    if not return_events:
        return raw

    if not op.exists(events_fname):
        raise ValueError('return_events=True but _events.tsv file'
                         ' not found')

    events, event_id = None, dict()
    dtypes = [float, float, str, int, int]
    events_dict = _from_tsv(events_fname, dtypes=dtypes)
    events_dict = _drop(events_dict, 'n/a', 'trial_type')

    if 'trial_type' not in events_dict:
        raise ValueError('trial_type column is missing. Cannot'
                         ' return events')

    for idx, ev in enumerate(np.unique(events_dict['trial_type'])):
        event_id[ev] = idx

    events = np.zeros((len(events_dict['onset']), 3), dtype=int)
    events[:, 0] = np.array(events_dict['onset']) * raw.info['sfreq'] + \
        raw.first_samp
    events[:, 2] = np.array([event_id[ev] for ev
                             in events_dict['trial_type']])

    return raw, events, event_id


def write_raw_bids(raw, bids_basename, output_path, events_data=None,
                   event_id=None, overwrite=False, verbose=True):
    """Walk over a folder of files and create BIDS compatible folder.

    .. warning:: The original files are simply copied over. This function
                 cannot convert modify data files from one format to another.
                 Modification of the original data files is not allowed.

    Parameters
    ----------
    raw : instance of mne.io.Raw
        The raw data. It must be an instance of mne.Raw. The data should not be
        loaded on disk, i.e., raw.preload must be False.
    bids_basename : str
        The base filename of the BIDS compatible files. Typically, this can be
        generated using make_bids_basename.
        Example: `sub-01_ses-01_task-testing_acq-01_run-01`.
        This will write the following files in the correct subfolder of the
        output_path::

            sub-01_ses-01_task-testing_acq-01_run-01_meg.fif
            sub-01_ses-01_task-testing_acq-01_run-01_meg.json
            sub-01_ses-01_task-testing_acq-01_run-01_channels.tsv
            sub-01_ses-01_task-testing_acq-01_run-01_coordsystem.json

        and the following one if events_data is not None::

            sub-01_ses-01_task-testing_acq-01_run-01_events.tsv

        and add a line to the following files::

            participants.tsv
            scans.tsv

        Note that the modality 'meg' is automatically inferred from the raw
        object and extension '.fif' is copied from raw.filenames.
    output_path : str
        The path of the root of the BIDS compatible folder. The session and
        subject specific folders will be populated automatically by parsing
        bids_basename.
    events_data : str | array | None
        The events file. If a string, a path to the events file. If an array,
        the MNE events array (shape n_events, 3). If None, events will be
        inferred from the stim channel using `mne.find_events`.
    event_id : dict | None
        The event id dict used to create a 'trial_type' column in events.tsv
    overwrite : bool
        Whether to overwrite existing files or data in files.
        Defaults to False.
        If overwrite is True, any existing files with the same BIDS parameters
        will be overwritten with the exception of the `participants.tsv` and
        `scans.tsv` files. For these files, parts of pre-existing data that
        match the current data will be replaced.
        If overwrite is False, no existing data will be overwritten or
        replaced.
    verbose : bool
        If verbose is True, this will print a snippet of the sidecar files. If
        False, no content will be printed.

    Notes
    -----
    For the participants.tsv file, the raw.info['subjects_info'] should be
    updated and raw.info['meas_date'] should not be None to compute the age
    of the participant correctly.

    """
    if not check_version('mne', '0.17'):
        raise ValueError('Your version of MNE is too old. '
                         'Please update to 0.17 or newer.')

    if not isinstance(raw, BaseRaw):
        raise ValueError('raw_file must be an instance of BaseRaw, '
                         'got %s' % type(raw))

    if not hasattr(raw, 'filenames') or raw.filenames[0] is None:
        raise ValueError('raw.filenames is missing. Please set raw.filenames'
                         'as a list with the full path of original raw file.')

    if raw.preload is not False:
        raise ValueError('The data should not be preloaded.')

    raw = raw.copy()

    raw_fname = raw.filenames[0]
    if '.ds' in op.dirname(raw.filenames[0]):
        raw_fname = op.dirname(raw.filenames[0])
    # point to file containing header info for multifile systems
    raw_fname = raw_fname.replace('.eeg', '.vhdr')
    raw_fname = raw_fname.replace('.fdt', '.set')
    _, ext = _parse_ext(raw_fname, verbose=verbose)

    raw_orig = reader[ext](**raw._init_kwargs)
    assert_array_equal(raw.times, raw_orig.times,
                       "raw.times should not have changed since reading"
                       " in from the file. It may have been cropped.")

    params = _parse_bids_filename(bids_basename, verbose)
    subject_id, session_id = params['sub'], params['ses']
    acquisition, task, run = params['acq'], params['task'], params['run']
    kind = _handle_kind(raw)

    bids_fname = bids_basename + '_%s%s' % (kind, ext)

    # check whether the info provided indicates that the data is emptyroom
    # data
    emptyroom = False
    if subject_id == 'emptyroom' and task == 'noise':
        emptyroom = True
        # check the session date provided is consistent with the value in raw
        meas_date = raw.info.get('meas_date', None)
        if meas_date is not None:
            er_date = datetime.fromtimestamp(
                raw.info['meas_date'][0]).strftime('%Y%m%d')
            if er_date != session_id:
                raise ValueError("Date provided for session doesn't match "
                                 "session date.")

    data_path = make_bids_folders(subject=subject_id, session=session_id,
                                  kind=kind, output_path=output_path,
                                  overwrite=False, verbose=verbose)
    if session_id is None:
        ses_path = os.sep.join(data_path.split(os.sep)[:-1])
    else:
        ses_path = make_bids_folders(subject=subject_id, session=session_id,
                                     output_path=output_path, make_dir=False,
                                     overwrite=False, verbose=verbose)

    # create filenames
    scans_fname = make_bids_basename(
        subject=subject_id, session=session_id, suffix='scans.tsv',
        prefix=ses_path)
    participants_tsv_fname = make_bids_basename(prefix=output_path,
                                                suffix='participants.tsv')
    participants_json_fname = make_bids_basename(prefix=output_path,
                                                 suffix='participants.json')
    coordsystem_fname = make_bids_basename(
        subject=subject_id, session=session_id, acquisition=acquisition,
        suffix='coordsystem.json', prefix=data_path)
    sidecar_fname = make_bids_basename(
        subject=subject_id, session=session_id, task=task, run=run,
        acquisition=acquisition, suffix='%s.json' % kind, prefix=data_path)
    events_fname = make_bids_basename(
        subject=subject_id, session=session_id, task=task,
        acquisition=acquisition, run=run, suffix='events.tsv',
        prefix=data_path)
    channels_fname = make_bids_basename(
        subject=subject_id, session=session_id, task=task, run=run,
        acquisition=acquisition, suffix='channels.tsv', prefix=data_path)
    if ext not in ['.fif', '.ds', '.vhdr', '.edf', '.bdf', '.set']:
        bids_raw_folder = bids_fname.split('.')[0]
        bids_fname = op.join(bids_raw_folder, bids_fname)

    # Read in Raw object and extract metadata from Raw object if needed
    orient = ORIENTATION.get(ext, 'n/a')
    unit = UNITS.get(ext, 'n/a')
    manufacturer = MANUFACTURERS.get(ext, 'n/a')

    # save all meta data
    _participants_tsv(raw, subject_id, participants_tsv_fname, overwrite,
                      verbose)
    _participants_json(participants_json_fname, True, verbose)
    _scans_tsv(raw, op.join(kind, bids_fname), scans_fname, overwrite, verbose)

    # TODO: Implement coordystem.json and electrodes.tsv for EEG and  iEEG
    if kind == 'meg' and not emptyroom:
        _coordsystem_json(raw, unit, orient, manufacturer, coordsystem_fname,
                          overwrite, verbose)

    events, event_id = _read_events(events_data, event_id, raw, ext)
    if events is not None and len(events) > 0 and not emptyroom:
        _events_tsv(events, raw, events_fname, event_id, overwrite, verbose)

    make_dataset_description(output_path, name=" ", verbose=verbose)
    _sidecar_json(raw, task, manufacturer, sidecar_fname, kind, overwrite,
                  verbose)
    _channels_tsv(raw, channels_fname, overwrite, verbose)

    # set the raw file name to now be the absolute path to ensure the files
    # are placed in the right location
    bids_fname = op.join(data_path, bids_fname)
    if os.path.exists(bids_fname) and not overwrite:
        raise FileExistsError('"%s" already exists. Please set '  # noqa: F821
                              'overwrite to True.' % bids_fname)
    _mkdir_p(os.path.dirname(bids_fname))

    if verbose:
        print('Copying data files to %s' % bids_fname)

    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError('ext must be in %s, got %s'
                         % (''.join(ALLOWED_EXTENSIONS), ext))

    # Copy the imaging data files
    if ext in ['.fif']:
        n_rawfiles = len(raw.filenames)
        if n_rawfiles > 1:
            split_naming = 'bids'
            raw.save(bids_fname, split_naming=split_naming, overwrite=True)
        else:
            # This ensures that single FIF files do not have the part param
            raw.save(bids_fname, overwrite=True, split_naming='neuromag')

    # CTF data is saved and renamed in a directory
    elif ext == '.ds':
        copyfile_ctf(raw_fname, bids_fname)
    # BrainVision is multifile, copy over all of them and fix pointers
    elif ext == '.vhdr':
        copyfile_brainvision(raw_fname, bids_fname)
    # EEGLAB .set might be accompanied by a .fdt - find out and copy it too
    elif ext == '.set':
        copyfile_eeglab(raw_fname, bids_fname)
    elif ext == '.pdf':
        copyfile_bti(raw_orig, op.join(data_path, bids_raw_folder))
    else:
        sh.copyfile(raw_fname, bids_fname)
    # KIT data requires the marker file to be copied over too
    if 'mrk' in raw._init_kwargs:
        hpi = raw._init_kwargs['mrk']
        _, marker_ext = _parse_ext(hpi)
        marker_fname = make_bids_basename(
            subject=subject_id, session=session_id, task=task, run=run,
            acquisition=acquisition, suffix='markers%s' % marker_ext,
            prefix=op.join(data_path, bids_raw_folder))
        sh.copyfile(hpi, marker_fname)

    return output_path


def make_bids_basename(subject=None, session=None, task=None,
                       acquisition=None, run=None, processing=None,
                       recording=None, space=None, prefix=None, suffix=None):
    """Create a partial/full BIDS filename from its component parts.

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
        The recording name for this item. Corresponds to "recording".
    space : str | None
        The coordinate space for an anatomical file. Corresponds to "space".
    prefix : str | None
        The prefix for the filename to be created. E.g., a path to the folder
        in which you wish to create a file with this name.
    suffix : str | None
        The suffix of a file that begins with this prefix. E.g., 'audio.wav'.

    Returns
    -------
    filename : str
        The BIDS filename you wish to create.

    Examples
    --------
    >>> print(make_bids_basename(subject='test', session='two', task='mytask', suffix='data.csv')) # noqa
    sub-test_ses-two_task-mytask_data.csv

    """
    order = OrderedDict([('sub', subject),
                         ('ses', session),
                         ('task', task),
                         ('acq', acquisition),
                         ('run', run),
                         ('proc', processing),
                         ('space', space),
                         ('recording', recording)])
    if order['run'] is not None and not isinstance(order['run'], str):
        # Ensure that run is a string
        order['run'] = '{:02}'.format(order['run'])

    _check_types(order.values())

    if not any(isinstance(ii, str) for ii in order.keys()):
        raise ValueError("At least one parameter must be given.")

    filename = []
    for key, val in order.items():
        if val is not None:
            _check_key_val(key, val)
            filename.append('%s-%s' % (key, val))

    if isinstance(suffix, str):
        filename.append(suffix)

    filename = '_'.join(filename)
    if isinstance(prefix, str):
        filename = op.join(prefix, filename)
    return filename


def make_bids_folders(subject, session=None, kind=None, output_path=None,
                      make_dir=True, overwrite=False, verbose=False):
    """Create a BIDS folder hierarchy.

    This creates a hierarchy of folders *within* a BIDS dataset. You should
    plan to create these folders *inside* the output_path folder of the dataset.

    Parameters
    ----------
    subject : str
        The subject ID. Corresponds to "sub".
    kind : str
        The kind of folder being created at the end of the hierarchy. E.g.,
        "anat", "func", etc.
    session : str | None
        The session for a item. Corresponds to "ses".
    output_path : str | None
        The output_path for the folders to be created. If None, folders will be
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
    >>> print(make_bids_folders('sub_01', session='my_session',
                                kind='meg', output_path='path/to/project',
                                make_dir=False))  # noqa
    path/to/project/sub-sub_01/ses-my_session/meg

    """
    _check_types((subject, kind, session, output_path))
    if session is not None:
        _check_key_val('ses', session)

    path = ['sub-%s' % subject]
    if isinstance(session, str):
        path.append('ses-%s' % session)
    if isinstance(kind, str):
        path.append(kind)
    path = op.join(*path)
    if isinstance(output_path, str):
        path = op.join(output_path, path)

    if make_dir is True:
        _mkdir_p(path, overwrite=overwrite, verbose=verbose)
    return path


def make_dataset_description(path, name=None, data_license=None,
                             authors=None, acknowledgements=None,
                             how_to_acknowledge=None, funding=None,
                             references_and_links=None, doi=None,
                             verbose=False):
    """Create json for a dataset description.

    BIDS datasets may have one or more fields, this function allows you to
    specify which you wish to include in the description. See the BIDS
    documentation for information about what each field means.

    Parameters
    ----------
    path : str
        A path to a folder where the description will be created.
    name : str | None
        The name of this BIDS dataset.
    data_license : str | None
        The license under which this datset is published.
    authors : list | str | None
        List of individuals who contributed to the creation/curation of the
        dataset. Must be a list of strings or a single comma separated string
        like ['a', 'b', 'c'].
    acknowledgements : list | str | None
        Either a str acknowledging individuals who contributed to the
        creation/curation of this dataset OR a list of the individuals'
        names as str.
    how_to_acknowledge : list | str | None
        Either a str describing how to acknowledge this dataset OR a list of
        publications that should be cited.
    funding : list | str | None
        List of sources of funding (e.g., grant numbers). Must be a list of
        strings or a single comma separated string like ['a', 'b', 'c'].
    references_and_links : list | str | None
        List of references to publication that contain information on the
        dataset, or links.  Must be a list of strings or a single comma
        separated string like ['a', 'b', 'c'].
    doi : str | None
        The DOI for the dataset.

    Notes
    -----
    The required field BIDSVersion will be automatically filled by mne_bids.

    """
    # Put potential string input into list of strings
    if isinstance(authors, str):
        authors = authors.split(', ')
    if isinstance(funding, str):
        funding = funding.split(', ')
    if isinstance(references_and_links, str):
        references_and_links = references_and_links.split(', ')

    fname = op.join(path, 'dataset_description.json')
    description = OrderedDict([('Name', name),
                               ('BIDSVersion', BIDS_VERSION),
                               ('License', data_license),
                               ('Authors', authors),
                               ('Acknowledgements', acknowledgements),
                               ('HowToAcknowledge', how_to_acknowledge),
                               ('Funding', funding),
                               ('ReferencesAndLinks', references_and_links),
                               ('DatasetDOI', doi)])
    pop_keys = [key for key, val in description.items() if val is None]
    for key in pop_keys:
        description.pop(key)
    _write_json(fname, description, overwrite=True, verbose=verbose)
