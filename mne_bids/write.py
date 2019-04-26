"""Make BIDS compatible directory structures and infer meta data from MNE."""
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

import shutil as sh
from collections import defaultdict, OrderedDict

import numpy as np
from numpy.testing import assert_array_equal

from mne import Epochs
from mne.io.constants import FIFF
from mne.io.pick import channel_type
from mne.io import BaseRaw
from mne.channels.channels import _unit2human
from mne.utils import check_version

from datetime import datetime
from warnings import warn

from .pick import coil_type
from .utils import (_write_json, _write_tsv, _read_events, _mkdir_p,
                    _age_on_date, _infer_eeg_placement_scheme, _check_key_val,
                    _parse_bids_filename, _handle_kind, _check_types)
from .copyfiles import (copyfile_brainvision, copyfile_eeglab, copyfile_ctf,
                        copyfile_bti)

from .read import _parse_ext, reader
from .tsv_handler import _from_tsv, _combine, _drop, _contains_row

from .config import (ALLOWED_KINDS, ORIENTATION, UNITS, MANUFACTURERS,
                     IGNORED_CHANNELS, ALLOWED_EXTENSIONS, BIDS_VERSION)


def _channels_tsv(raw, fname, overwrite=False, verbose=True):
    """Create a channels.tsv file and save it.

    Parameters
    ----------
    raw : instance of Raw
        The data as MNE-Python Raw object.
    fname : str
        Filename to save the channels.tsv to.
    overwrite : bool
        Whether to overwrite the existing file.
        Defaults to False.
    verbose : bool
        Set verbose output to true or false.

    """
    map_chs = defaultdict(lambda: 'OTHER')
    map_chs.update(meggradaxial='MEGGRADAXIAL',
                   megrefgradaxial='MEGREFGRADAXIAL',
                   meggradplanar='MEGGRADPLANAR',
                   megmag='MEGMAG', megrefmag='MEGREFMAG',
                   eeg='EEG', misc='MISC', stim='TRIG', emg='EMG',
                   ecog='ECOG', seeg='SEEG', eog='EOG', ecg='ECG')
    map_desc = defaultdict(lambda: 'Other type of channel')
    map_desc.update(meggradaxial='Axial Gradiometer',
                    megrefgradaxial='Axial Gradiometer Reference',
                    meggradplanar='Planar Gradiometer',
                    megmag='Magnetometer',
                    megrefmag='Magnetometer Reference',
                    stim='Trigger', eeg='ElectroEncephaloGram',
                    ecog='Electrocorticography',
                    seeg='StereoEEG',
                    ecg='ElectroCardioGram',
                    eog='ElectroOculoGram',
                    emg='ElectroMyoGram',
                    misc='Miscellaneous')
    get_specific = ('mag', 'ref_meg', 'grad')

    # get the manufacturer from the file in the Raw object
    manufacturer = None

    _, ext = _parse_ext(raw.filenames[0], verbose=verbose)
    manufacturer = MANUFACTURERS[ext]

    ignored_channels = IGNORED_CHANNELS.get(manufacturer, list())

    status, ch_type, description = list(), list(), list()
    for idx, ch in enumerate(raw.info['ch_names']):
        status.append('bad' if ch in raw.info['bads'] else 'good')
        _channel_type = channel_type(raw.info, idx)
        if _channel_type in get_specific:
            _channel_type = coil_type(raw.info, idx, _channel_type)
        ch_type.append(map_chs[_channel_type])
        description.append(map_desc[_channel_type])
    low_cutoff, high_cutoff = (raw.info['highpass'], raw.info['lowpass'])
    if raw._orig_units:
        units = [raw._orig_units.get(ch, 'n/a') for ch in raw.ch_names]
    else:
        units = [_unit2human.get(ch_i['unit'], 'n/a')
                 for ch_i in raw.info['chs']]
        units = [u if u not in ['NA'] else 'n/a' for u in units]
    n_channels = raw.info['nchan']
    sfreq = raw.info['sfreq']

    ch_data = OrderedDict([
        ('name', raw.info['ch_names']),
        ('type', ch_type),
        ('units', units),
        ('low_cutoff', np.full((n_channels), low_cutoff)),
        ('high_cutoff', np.full((n_channels), high_cutoff)),
        ('description', description),
        ('sampling_frequency', np.full((n_channels), sfreq)),
        ('status', status)])
    ch_data = _drop(ch_data, ignored_channels, 'name')

    _write_tsv(fname, ch_data, overwrite, verbose)

    return fname


def _events_tsv(events, raw, fname, trial_type, overwrite=False,
                verbose=True):
    """Create an events.tsv file and save it.

    This function will write the mandatory 'onset', and 'duration' columns as
    well as the optional 'value' and 'sample'. The 'value'
    corresponds to the marker value as found in the TRIG channel of the
    recording. In addition, the 'trial_type' field can be written.

    Parameters
    ----------
    events : array, shape = (n_events, 3)
        The first column contains the event time in samples and the third
        column contains the event id. The second column is ignored for now but
        typically contains the value of the trigger channel either immediately
        before the event or immediately after.
    raw : instance of Raw
        The data as MNE-Python Raw object.
    fname : str
        Filename to save the events.tsv to.
    trial_type : dict | None
        Dictionary mapping a brief description key to an event id (value). For
        example {'Go': 1, 'No Go': 2}.
    overwrite : bool
        Whether to overwrite the existing file.
        Defaults to False.
    verbose : bool
        Set verbose output to true or false.

    Notes
    -----
    The function writes durations of zero for each event.

    """
    # Start by filling all data that we know into an ordered dictionary
    first_samp = raw.first_samp
    sfreq = raw.info['sfreq']
    events[:, 0] -= first_samp

    # Onset column needs to be specified in seconds
    data = OrderedDict([('onset', events[:, 0] / sfreq),
                        ('duration', np.zeros(events.shape[0])),
                        ('trial_type', None),
                        ('value', events[:, 2]),
                        ('sample', events[:, 0])])

    # Now check if trial_type is specified or should be removed
    if trial_type:
        trial_type_map = {v: k for k, v in trial_type.items()}
        data['trial_type'] = [trial_type_map.get(i, 'n/a') for
                              i in events[:, 2]]
    else:
        del data['trial_type']

    _write_tsv(fname, data, overwrite, verbose)

    return fname


def _participants_tsv(raw, subject_id, fname, overwrite=False,
                      verbose=True):
    """Create a participants.tsv file and save it.

    This will append any new participant data to the current list if it
    exists. Otherwise a new file will be created with the provided information.

    Parameters
    ----------
    raw : instance of Raw
        The data as MNE-Python Raw object.
    subject_id : str
        The subject name in BIDS compatible format ('01', '02', etc.)
    fname : str
        Filename to save the participants.tsv to.
    overwrite : bool
        Whether to overwrite the existing file.
        Defaults to False.
        If there is already data for the given `subject_id` and overwrite is
        False, an error will be raised.
    verbose : bool
        Set verbose output to true or false.

    """
    subject_id = 'sub-' + subject_id
    data = OrderedDict(participant_id=[subject_id])

    subject_age = "n/a"
    sex = "n/a"
    subject_info = raw.info['subject_info']
    if subject_info is not None:
        sexes = {0: 'n/a', 1: 'M', 2: 'F'}
        sex = sexes[subject_info.get('sex', 0)]

        # determine the age of the participant
        age = subject_info.get('birthday', None)
        meas_date = raw.info.get('meas_date', None)
        if isinstance(meas_date, (tuple, list, np.ndarray)):
            meas_date = meas_date[0]

        if meas_date is not None and age is not None:
            bday = datetime(age[0], age[1], age[2])
            meas_datetime = datetime.fromtimestamp(meas_date)
            subject_age = _age_on_date(bday, meas_datetime)
        else:
            subject_age = "n/a"

    data.update({'age': [subject_age], 'sex': [sex]})

    if os.path.exists(fname):
        orig_data = _from_tsv(fname)
        # whether the new data exists identically in the previous data
        exact_included = _contains_row(orig_data,
                                       {'participant_id': subject_id,
                                        'age': subject_age,
                                        'sex': sex})
        # whether the subject id is in the previous data
        sid_included = subject_id in orig_data['participant_id']
        # if the subject data provided is different to the currently existing
        # data and overwrite is not True raise an error
        if (sid_included and not exact_included) and not overwrite:
            raise FileExistsError('"%s" already exists in the participant '  # noqa: E501 F821
                                  'list. Please set overwrite to '
                                  'True.' % subject_id)
        # otherwise add the new data
        data = _combine(orig_data, data, 'participant_id')

    # overwrite is forced to True as all issues with overwrite == False have
    # been handled by this point
    _write_tsv(fname, data, True, verbose)

    return fname


def _participants_json(fname, overwrite=False, verbose=True):
    """Create participants.json for non-default columns in accompanying TSV.

    Parameters
    ----------
    fname : str
        Filename to save the scans.tsv to.
    overwrite : bool
        Defaults to False.
        Whether to overwrite the existing data in the file.
        If there is already data for the given `fname` and overwrite is False,
        an error will be raised.
    verbose : bool
        Set verbose output to true or false.

    """
    cols = OrderedDict()
    cols['participant_id'] = {'Description': 'Unique participant identifier'}
    cols['age'] = {'Description': 'Age of the participant at time of testing',
                   'Units': 'years'}
    cols['sex'] = {'Description': 'Biological sex of the participant',
                   'Levels': {'F': 'female', 'M': 'male'}}

    _write_json(fname, cols, overwrite, verbose)

    return fname


def _scans_tsv(raw, raw_fname, fname, overwrite=False, verbose=True):
    """Create a scans.tsv file and save it.

    Parameters
    ----------
    raw : instance of Raw
        The data as MNE-Python Raw object.
    raw_fname : str
        Relative path to the raw data file.
    fname : str
        Filename to save the scans.tsv to.
    overwrite : bool
        Defaults to False.
        Whether to overwrite the existing data in the file.
        If there is already data for the given `fname` and overwrite is False,
        an error will be raised.
    verbose : bool
        Set verbose output to true or false.

    """
    # get measurement date from the data info
    meas_date = raw.info['meas_date']
    if isinstance(meas_date, (tuple, list, np.ndarray)):
        meas_date = meas_date[0]
        acq_time = datetime.fromtimestamp(
            meas_date).strftime('%Y-%m-%dT%H:%M:%S')
    else:
        acq_time = 'n/a'

    data = OrderedDict([('filename', ['%s' % raw_fname.replace(os.sep, '/')]),
                       ('acq_time', [acq_time])])

    if os.path.exists(fname):
        orig_data = _from_tsv(fname)
        # if the file name is already in the file raise an error
        if raw_fname in orig_data['filename'] and not overwrite:
            raise FileExistsError('"%s" already exists in the scans list. '  # noqa: E501 F821
                                  'Please set overwrite to True.' % raw_fname)
        # otherwise add the new data
        data = _combine(orig_data, data, 'filename')

    # overwrite is forced to True as all issues with overwrite == False have
    # been handled by this point
    _write_tsv(fname, data, True, verbose)

    return fname


def _coordsystem_json(raw, unit, orient, manufacturer, fname,
                      overwrite=False, verbose=True):
    """Create a coordsystem.json file and save it.

    Parameters
    ----------
    raw : instance of Raw
        The data as MNE-Python Raw object.
    unit : str
        Units to be used in the coordsystem specification.
    orient : str
        Used to define the coordinate system for the head coils.
    manufacturer : str
        Used to define the coordinate system for the MEG sensors.
    fname : str
        Filename to save the coordsystem.json to.
    overwrite : bool
        Whether to overwrite the existing file.
        Defaults to False.
    verbose : bool
        Set verbose output to true or false.

    """
    dig = raw.info['dig']
    coords = dict()
    fids = {d['ident']: d for d in dig if d['kind'] ==
            FIFF.FIFFV_POINT_CARDINAL}
    if fids:
        if FIFF.FIFFV_POINT_NASION in fids:
            coords['NAS'] = fids[FIFF.FIFFV_POINT_NASION]['r'].tolist()
        if FIFF.FIFFV_POINT_LPA in fids:
            coords['LPA'] = fids[FIFF.FIFFV_POINT_LPA]['r'].tolist()
        if FIFF.FIFFV_POINT_RPA in fids:
            coords['RPA'] = fids[FIFF.FIFFV_POINT_RPA]['r'].tolist()

    hpi = {d['ident']: d for d in dig if d['kind'] == FIFF.FIFFV_POINT_HPI}
    if hpi:
        for ident in hpi.keys():
            coords['coil%d' % ident] = hpi[ident]['r'].tolist()

    coord_frame = set([dig[ii]['coord_frame'] for ii in range(len(dig))])
    if len(coord_frame) > 1:
        err = 'All HPI and Fiducials must be in the same coordinate frame.'
        raise ValueError(err)

    fid_json = {'MEGCoordinateSystem': manufacturer,
                'MEGCoordinateUnits': unit,  # XXX validate this
                'HeadCoilCoordinates': coords,
                'HeadCoilCoordinateSystem': orient,
                'HeadCoilCoordinateUnits': unit  # XXX validate this
                }

    _write_json(fname, fid_json, overwrite, verbose)

    return fname


def _sidecar_json(raw, task, manufacturer, fname, kind, overwrite=False,
                  verbose=True):
    """Create a sidecar json file depending on the kind and save it.

    The sidecar json file provides meta data about the data of a certain kind.

    Parameters
    ----------
    raw : instance of Raw
        The data as MNE-Python Raw object.
    task : str
        Name of the task the data is based on.
    manufacturer : str
        Manufacturer of the acquisition system. For MEG also used to define the
        coordinate system for the MEG sensors.
    fname : str
        Filename to save the sidecar json to.
    kind : str
        Type of the data as in ALLOWED_KINDS.
    overwrite : bool
        Whether to overwrite the existing file.
        Defaults to False.
    verbose : bool
        Set verbose output to true or false. Defaults to true.

    """
    sfreq = raw.info['sfreq']
    powerlinefrequency = raw.info.get('line_freq', None)
    if powerlinefrequency is None:
        warn('No line frequency found, defaulting to 50 Hz')
        powerlinefrequency = 50

    if isinstance(raw, BaseRaw):
        rec_type = 'continuous'
    elif isinstance(raw, Epochs):
        rec_type = 'epoched'
    else:
        rec_type = 'n/a'

    # determine whether any channels have to be ignored:
    n_ignored = len([ch_name for ch_name in
                     IGNORED_CHANNELS.get(manufacturer, list()) if
                     ch_name in raw.ch_names])
    # all ignored channels are trigger channels at the moment...

    n_megchan = len([ch for ch in raw.info['chs']
                     if ch['kind'] == FIFF.FIFFV_MEG_CH])
    n_megrefchan = len([ch for ch in raw.info['chs']
                        if ch['kind'] == FIFF.FIFFV_REF_MEG_CH])
    n_eegchan = len([ch for ch in raw.info['chs']
                     if ch['kind'] == FIFF.FIFFV_EEG_CH])
    n_ecogchan = len([ch for ch in raw.info['chs']
                      if ch['kind'] == FIFF.FIFFV_ECOG_CH])
    n_seegchan = len([ch for ch in raw.info['chs']
                      if ch['kind'] == FIFF.FIFFV_SEEG_CH])
    n_eogchan = len([ch for ch in raw.info['chs']
                     if ch['kind'] == FIFF.FIFFV_EOG_CH])
    n_ecgchan = len([ch for ch in raw.info['chs']
                     if ch['kind'] == FIFF.FIFFV_ECG_CH])
    n_emgchan = len([ch for ch in raw.info['chs']
                     if ch['kind'] == FIFF.FIFFV_EMG_CH])
    n_miscchan = len([ch for ch in raw.info['chs']
                      if ch['kind'] == FIFF.FIFFV_MISC_CH])
    n_stimchan = len([ch for ch in raw.info['chs']
                      if ch['kind'] == FIFF.FIFFV_STIM_CH]) - n_ignored

    # Define modality-specific JSON dictionaries
    ch_info_json_common = [
        ('TaskName', task),
        ('Manufacturer', manufacturer),
        ('PowerLineFrequency', powerlinefrequency),
        ('SamplingFrequency', sfreq),
        ('SoftwareFilters', 'n/a'),
        ('RecordingDuration', raw.times[-1]),
        ('RecordingType', rec_type)]
    ch_info_json_meg = [
        ('DewarPosition', 'n/a'),
        ('DigitizedLandmarks', False),
        ('DigitizedHeadPoints', False),
        ('MEGChannelCount', n_megchan),
        ('MEGREFChannelCount', n_megrefchan)]
    ch_info_json_eeg = [
        ('EEGReference', 'n/a'),
        ('EEGGround', 'n/a'),
        ('EEGPlacementScheme', _infer_eeg_placement_scheme(raw)),
        ('Manufacturer', manufacturer)]
    ch_info_json_ieeg = [
        ('iEEGReference', 'n/a'),
        ('ECOGChannelCount', n_ecogchan),
        ('SEEGChannelCount', n_seegchan)]
    ch_info_ch_counts = [
        ('EEGChannelCount', n_eegchan),
        ('EOGChannelCount', n_eogchan),
        ('ECGChannelCount', n_ecgchan),
        ('EMGChannelCount', n_emgchan),
        ('MiscChannelCount', n_miscchan),
        ('TriggerChannelCount', n_stimchan)]

    # Stitch together the complete JSON dictionary
    ch_info_json = ch_info_json_common
    if kind == 'meg':
        append_kind_json = ch_info_json_meg
    elif kind == 'eeg':
        append_kind_json = ch_info_json_eeg
    elif kind == 'ieeg':
        append_kind_json = ch_info_json_ieeg
    else:
        raise ValueError('Unexpected "kind": {}'
                         ' Use one of: {}'.format(kind, ALLOWED_KINDS))

    ch_info_json += append_kind_json
    ch_info_json += ch_info_ch_counts
    ch_info_json = OrderedDict(ch_info_json)

    _write_json(fname, ch_info_json, overwrite, verbose)

    return fname


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
    if ext not in ['.fif', '.ds', '.vhdr', '.edf', '.bdf', '.set', '.con',
                   '.sqd']:
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
            prefix=data_path)
        sh.copyfile(hpi, marker_fname)

    return output_path
