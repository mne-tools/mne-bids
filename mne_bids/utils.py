"""Utility and helper functions for MNE-BIDS."""
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
import re
from collections import OrderedDict
import json
import shutil as sh

import numpy as np
from scipy.io import loadmat, savemat
from mne import read_events, find_events
from mne.channels import read_montage
from mne.io.pick import pick_types

from .config import BIDS_VERSION
from .io import _parse_ext


def print_dir_tree(folder):
    """Recursively print a directory tree starting from `folder`."""
    if not op.exists(folder):
        raise ValueError('Directory does not exist: {}'.format(folder))

    for root, dirs, files in os.walk(folder):
        path = root.split(os.sep)
        print('|%s %s' % ((len(path) - 1) * '---', op.basename(root)))
        for file in files:
            print('|%s %s' % (len(path) * '---', file))


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


def _parse_bids_filename(fname, verbose):
    """Get dict from BIDS fname."""
    keys = ['sub', 'ses', 'task', 'acq', 'run', 'proc', 'run', 'space',
            'recording']
    params = {key: None for key in keys}
    entities = fname.split('_')
    idx_key = 0
    for entity in entities:
        assert '-' in entity
        key, value = entity.split('-')
        if key not in keys:
            raise KeyError('Unexpected entity ''%s'' found in filename ''%s'''
                           % (entity, fname))
        if keys.index(key) < idx_key:
            raise ValueError('Entities in filename not ordered correctly.'
                             ' "%s" should have occured earlier in the '
                             'filename "%s"' % (key, fname))
        idx_key = keys.index(key)
        params[key] = value
    return params


def _handle_kind(raw):
    """Get kind."""
    if 'eeg' in raw and 'ecog' in raw:
        raise ValueError('Both EEG and iEEG channels found in data.'
                         'There is currently no specification on how'
                         'to handle this data. Please proceed manually.')
    elif 'meg' in raw:
        kind = 'meg'
    elif 'ecog' in raw:
        kind = 'ieeg'
    elif 'eeg' in raw:
        kind = 'eeg'
    else:
        raise ValueError('Neither MEG/EEG/iEEG channels found in data.'
                         'Please use raw.set_channel_types to set the '
                         'channel types in the data.')
    return kind


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
    _write_json(description, fname, overwrite=True, verbose=verbose)


def _age_on_date(bday, exp_date):
    """Calculate age from birthday and experiment date.

    Parameters
    ----------
    bday : instance of datetime.datetime
        The birthday of the participant.
    exp_date : instance of datetime.datetime
        The date the experiment was performed on.

    """
    if exp_date < bday:
        raise ValueError("The experimentation date must be after the birth "
                         "date")
    if exp_date.month > bday.month:
        return exp_date.year - bday.year
    elif exp_date.month == bday.month:
        if exp_date.day >= bday.day:
            return exp_date.year - bday.year
    return exp_date.year - bday.year - 1


def _check_types(variables):
    """Make sure all vars are str or None."""
    for var in variables:
        if not isinstance(var, (str, type(None))):
            raise ValueError("All values must be either None or strings. "
                             "Found type %s." % type(var))


def _write_json(dictionary, fname, overwrite=False, verbose=False):
    """Write JSON to a file."""
    if op.exists(fname) and not overwrite:
        raise FileExistsError('"%s" already exists. Please set '
                              'overwrite to True.' % fname)

    json_output = json.dumps(dictionary, indent=4)
    with open(fname, 'w') as fid:
        fid.write(json_output)
        fid.write('\n')

    if verbose is True:
        print(os.linesep + "Writing '%s'..." % fname + os.linesep)
        print(json_output)


def _write_tsv(fname, df, overwrite=False, verbose=False):
    """Write dataframe to a .tsv file."""
    if op.exists(fname) and not overwrite:
        raise FileExistsError('"%s" already exists. Please set '
                              'overwrite to True.' % fname)
    df.to_csv(fname, sep='\t', index=False, na_rep='n/a', encoding='utf-8')

    if verbose:
        print(os.linesep + "Writing '%s'..." % fname + os.linesep)
        print(df.head())


def _check_key_val(key, val):
    """Perform checks on a value to make sure it adheres to the spec."""
    if any(ii in val for ii in ['-', '_', '/']):
        raise ValueError("Unallowed `-`, `_`, or `/` found in key/value pair"
                         " %s: %s" % (key, val))
    return key, val


def _read_events(events_data, raw):
    """Read in events data.

    Parameters
    ----------
    events_data : str | array | None
        The events file. If a string, a path to the events file. If an array,
        the MNE events array (shape n_events, 3). If None, events will be
        inferred from the stim channel using `find_events`.
    raw : instance of Raw
        The data as MNE-Python Raw object.

    Returns
    -------
    events : array, shape = (n_events, 3)
        The first column contains the event time in samples and the third
        column contains the event id. The second column is ignored for now but
        typically contains the value of the trigger channel either immediately
        before the event or immediately after.

    """
    if isinstance(events_data, str):
        events = read_events(events_data).astype(int)
    elif isinstance(events_data, np.ndarray):
        if events_data.ndim != 2:
            raise ValueError('Events must have two dimensions, '
                             'found %s' % events_data.ndim)
        if events_data.shape[1] != 3:
            raise ValueError('Events must have second dimension of length 3, '
                             'found %s' % events_data.shape[1])
        events = events_data
    else:
        events = find_events(raw, min_duration=0.001, initial_event=True)
    return events


def _get_brainvision_paths(vhdr_path):
    """Get the .eeg and .vmrk file paths from a BrainVision header file.

    Parameters
    ----------
    vhdr_path : str
        path to the header file

    Returns
    -------
    paths : tuple
        paths to the .eeg file at index 0 and the .vmrk file
        at index 1 of the returned tuple

    """
    fname, ext = _parse_ext(vhdr_path)
    if ext != '.vhdr':
        raise ValueError('Expecting file ending in ".vhdr",'
                         ' but got {}'.format(ext))

    # Header file seems fine, read it
    with open(vhdr_path, 'r') as f:
        lines = f.readlines()

    # Try to find data file .eeg
    eeg_file_match = re.search(r'DataFile=(.*\.eeg)', ' '.join(lines))
    if not eeg_file_match:
        raise ValueError('Could not find a .eeg file link in'
                         ' {}'.format(vhdr_path))
    else:
        eeg_file = eeg_file_match.groups()[0]

    # Try to find marker file .vmrk
    vmrk_file_match = re.search(r'MarkerFile=(.*\.vmrk)', ' '.join(lines))
    if not vmrk_file_match:
        raise ValueError('Could not find a .vmrk file link in'
                         ' {}'.format(vhdr_path))
    else:
        vmrk_file = vmrk_file_match.groups()[0]

    # Make sure we are dealing with file names as is customary, not paths
    # Paths are problematic when copying the files to another system. Instead,
    # always use the file name and keep the file triplet in the same directory
    assert os.sep not in eeg_file
    assert os.sep not in vmrk_file

    # Assert the paths exist
    head, tail = op.split(vhdr_path)
    eeg_file_path = op.join(head, eeg_file)
    vmrk_file_path = op.join(head, vmrk_file)
    assert op.exists(eeg_file_path)
    assert op.exists(vmrk_file_path)

    # Return the paths
    return (eeg_file_path, vmrk_file_path)


def copyfile_brainvision(vhdr_src, vhdr_dest):
    """Copy a BrainVision file triplet to a new location and repair links.

    Parameters
    ----------
    vhdr_src, vhdr_dest: str
        The src path of the .vhdr file to be copied and the destination
        path. The .eeg and .vmrk files associated with the .vhdr file
        will be given names as in vhdr_dest with adjusted extensions.
        Internal file pointers will be fixed.

    """
    # Get extenstion of the brainvision file
    fname_src, ext_src = _parse_ext(vhdr_src)
    fname_dest, ext_dest = _parse_ext(vhdr_dest)
    if ext_src != ext_dest:
        raise ValueError('Need to move data with same extension'
                         ' but got "{}", "{}"'.format(ext_src, ext_dest))

    eeg_file_path, vmrk_file_path = _get_brainvision_paths(vhdr_src)

    # Copy data .eeg ... no links to repair
    sh.copyfile(eeg_file_path, fname_dest + '.eeg')

    # Write new header and marker files, fixing the file pointer links
    # For that, we need to replace an old "basename" with a new one
    # assuming that all .eeg, .vhdr, .vmrk share one basename
    __, basename_src = op.split(fname_src)
    assert basename_src + '.eeg' == op.split(eeg_file_path)[-1]
    assert basename_src + '.vmrk' == op.split(vmrk_file_path)[-1]
    __, basename_dest = op.split(fname_dest)
    search_lines = ['DataFile=' + basename_src + '.eeg',
                    'MarkerFile=' + basename_src + '.vmrk']

    with open(vhdr_src, 'r') as fin:
        with open(vhdr_dest, 'w') as fout:
            for line in fin.readlines():
                if line.strip() in search_lines:
                    line = line.replace(basename_src, basename_dest)
                fout.write(line)

    with open(vmrk_file_path, 'r') as fin:
        with open(fname_dest + '.vmrk', 'w') as fout:
            for line in fin.readlines():
                if line.strip() in search_lines:
                    line = line.replace(basename_src, basename_dest)
                fout.write(line)


def copyfile_eeglab(src, dest):
    """Copy a EEGLAB files to a new location and adjust pointer to '.fdt' file.

    Some EEGLAB .set files come with a .fdt binary file that contains the data.
    When moving a .set file, we need to check for an associated .fdt file and
    move it to an appropriate location as well as update an internal pointer
    within the .set file.

    Notes
    -----
    Work in progress. This function will abort upon the encounter of a .fdt
    file.

    """
    # Get extenstion of the EEGLAB file
    fname_src, ext_src = _parse_ext(src)
    fname_dest, ext_dest = _parse_ext(dest)
    if ext_src != ext_dest:
        raise ValueError('Need to move data with same extension'
                         ' but got {}, {}'.format(ext_src, ext_dest))

    # Extract matlab struct "EEG" from EEGLAB file
    mat = loadmat(src, squeeze_me=False, chars_as_strings=False,
                  mat_dtype=False, struct_as_record=True)
    if 'EEG' not in mat:
        raise ValueError('Could not find "EEG" field in {}'.format(src))
    eeg = mat['EEG']

    # If the data field is a string, it points to a .fdt file in src dir
    data = eeg[0][0]['data']
    if all([item in data[0, -4:] for item in '.fdt']):
        head, tail = op.split(src)
        fdt_pointer = ''.join(data.tolist()[0])
        fdt_path = op.join(head, fdt_pointer)
        fdt_name, fdt_ext = _parse_ext(fdt_path)
        if fdt_ext != '.fdt':
            raise IOError('Expected extension {} for linked data but found'
                          ' {}'.format('.fdt', fdt_ext))

        # Copy the fdt file and give it a new name
        sh.copyfile(fdt_path, fname_dest + '.fdt')

        # Now adjust the pointer in the set file
        head, tail = op.split(fname_dest + '.fdt')
        mat['EEG'][0][0]['data'] = tail
        savemat(dest, mat, appendmat=False)

    # If no .fdt file, simply copy the .set file, no modifications necessary
    else:
        sh.copyfile(src, dest)


def _infer_eeg_placement_scheme(raw):
    """Based on the channel names, try to infer an EEG placement scheme.

    Parameters
    ----------
    raw : instance of Raw
        The data as MNE-Python Raw object.

    Returns
    -------
    placement_scheme : str
        Description of the EEG placement scheme. Will be "n/a" for unsuccessful
        extraction.

    """
    placement_scheme = 'n/a'
    # Check if the raw data contains eeg data at all
    if 'eeg' not in raw:
        return placement_scheme

    # How many of the channels in raw are based on the extended 10/20 system
    raw.load_data()
    sel = pick_types(raw.info, meg=False, eeg=True)
    ch_names = [raw.ch_names[i] for i in sel]
    channel_names = [ch.lower() for ch in ch_names]
    montage1005 = read_montage(kind='standard_1005')
    montage1005_names = [ch.lower() for ch in montage1005.ch_names]

    if set(channel_names).issubset(set(montage1005_names)):
        placement_scheme = 'based on the extended 10/20 system'

    return placement_scheme
