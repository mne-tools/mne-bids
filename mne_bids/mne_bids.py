"""Make BIDS compatible directory structures and infer meta data from MNE."""
# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#          Chris Holdgraf <choldgraf@berkeley.edu>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)

import os
import shutil as sh
import pandas as pd
from collections import defaultdict, OrderedDict

import numpy as np
from mne.io.constants import FIFF
from mne.io.pick import channel_type
from mne.io import BaseRaw
from mne.channels.channels import _unit2human
from mne.externals.six import string_types

from datetime import datetime
from warnings import warn

from .utils import (make_bids_filename, make_bids_folders,
                    make_dataset_description, _write_json,
                    _read_events, _mkdir_p, age_on_date)
from .io import (_parse_ext, _read_raw, ALLOWED_EXTENSIONS)


ALLOWED_KINDS = ['meg']
ORIENTATION = {'.sqd': 'ALS', '.con': 'ALS', '.fif': 'RAS', '.pdf': 'ALS',
               '.ds': 'ALS'}

UNITS = {'.sqd': 'm', '.con': 'm', '.fif': 'm', '.pdf': 'm', '.ds': 'cm'}

MANUFACTURERS = {'.sqd': 'KIT/Yokogawa', '.con': 'KIT/Yokogawa',
                 '.fif': 'Elekta', '.pdf': '4D Magnes', '.ds': 'CTF',
                 '.meg4': 'CTF'}

# List of synthetic channels by manufacturer that are to be excluded from the
# channel list. Currently this is only for stimulus channels.
IGNORED_CHANNELS = {'KIT/Yokogawa': ['STI 014']}


def _channels_tsv(raw, fname, verbose):
    """Create a channels.tsv file and save it.

    Parameters
    ----------
    raw : instance of Raw
        The data as MNE-Python Raw object.
    fname : str
        Filename to save the channels.tsv to.
    verbose : bool
        Set verbose output to true or false.

    """
    map_chs = defaultdict(lambda: 'OTHER')
    map_chs.update(grad='MEGGRAD', mag='MEGMAG', stim='TRIG', eeg='EEG',
                   ecog='ECOG', seeg='SEEG', eog='EOG', ecg='ECG', misc='MISC',
                   resp='RESPONSE', ref_meg='REFMEG')
    map_desc = defaultdict(lambda: 'Other type of channel')
    map_desc.update(grad='Gradiometer', mag='Magnetometer',
                    stim='Trigger',
                    eeg='ElectroEncephaloGram',
                    ecog='Electrocorticography',
                    seeg='StereoEEG',
                    ecg='ElectroCardioGram',
                    eog='ElectrOculoGram', misc='Miscellaneous',
                    ref_meg='Reference channel')

    # get the manufacturer from the file in the Raw object
    manufacturer = None
    if hasattr(raw, 'filenames'):
        _, ext = _parse_ext(raw.filenames[0], verbose=verbose)
        manufacturer = MANUFACTURERS[ext]

    ignored_indexes = [raw.ch_names.index(ch_name) for ch_name in raw.ch_names
                       if ch_name in
                       IGNORED_CHANNELS.get(manufacturer, list())]

    status, ch_type, description = list(), list(), list()
    for idx, ch in enumerate(raw.info['ch_names']):
        status.append('bad' if ch in raw.info['bads'] else 'good')
        ch_type.append(map_chs[channel_type(raw.info, idx)])
        description.append(map_desc[channel_type(raw.info, idx)])
    low_cutoff, high_cutoff = (raw.info['highpass'], raw.info['lowpass'])
    units = [_unit2human.get(ch_i['unit'], 'n/a') for ch_i in raw.info['chs']]
    units = [u if u not in ['NA'] else 'n/a' for u in units]
    n_channels = raw.info['nchan']
    sfreq = raw.info['sfreq']

    df = pd.DataFrame(OrderedDict([
                      ('name', raw.info['ch_names']),
                      ('type', ch_type),
                      ('units', units),
                      ('description', description),
                      ('sampling_frequency', np.full((n_channels), sfreq)),
                      ('low_cutoff', np.full((n_channels), low_cutoff)),
                      ('high_cutoff', np.full((n_channels), high_cutoff)),
                      ('status', status)]))
    df.drop(ignored_indexes, inplace=True)
    df.to_csv(fname, sep='\t', index=False, na_rep='n/a')

    if verbose:
        print(os.linesep + "Writing '%s'..." % fname + os.linesep)
        print(df.head())

    return fname


def _events_tsv(events, raw, fname, trial_type, verbose):
    """Create an events.tsv file and save it.

    This function will write the mandatory 'onset', and 'duration' columns as
    well as the optional 'event_value' and 'event_sample'. The 'event_value'
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
    event_id : dict | None
        Dictionary mapping a brief description key to an event id (value). For
        example {'Go': 1, 'No Go': 2}.
    verbose : bool
        Set verbose output to true or false.

    Notes
    -----
    The function writes durations of zero for each event.

    """
    # Start by filling all data that we know into a df
    first_samp = raw.first_samp
    sfreq = raw.info['sfreq']
    events[:, 0] -= first_samp

    data = OrderedDict([('onset', events[:, 0]),
                        ('duration', np.zeros(events.shape[0])),
                        ('trial_type', events[:, 2]),
                        ('event_value', events[:, 2]),
                        ('event_sample', events[:, 0])])

    df = pd.DataFrame.from_dict(data)

    # Now check if trial_type is specified or should be removed
    if trial_type:
        trial_type_map = {v: k for k, v in trial_type.items()}
        df.trial_type = df.trial_type.map(trial_type_map)
    else:
        df.drop(labels=['trial_type'], axis=1, inplace=True)

    # Onset column needs to be specified in seconds
    df.onset /= sfreq

    # Save to file
    df.to_csv(fname, sep='\t', index=False, na_rep='n/a')
    if verbose:
        print(os.linesep + "Writing '%s'..." % fname + os.linesep)
        print(df.head())

    return fname


def _participants_tsv(raw, subject_id, group, fname, verbose):
    """Create a participants.tsv file and save it.

    This will append any new participant data to the current list if it
    exists. Otherwise a new file will be created with the provided information.

    Parameters
    ----------
    raw : instance of Raw
        The data as MNE-Python Raw object.
    subject_id : str
        The subject name in BIDS compatible format ('01', '02', etc.)
    group : str
        Name of group participant belongs to.
    fname : str
        Filename to save the participants.tsv to.
    verbose : bool
        Set verbose output to true or false.

    """
    subject_id = 'sub-' + subject_id
    data = {'participant_id': [subject_id]}

    subject_info = raw.info['subject_info']
    if subject_info is not None:
        genders = {0: 'U', 1: 'M', 2: 'F'}
        sex = genders[subject_info.get('sex', 0)]

        # determine the age of the participant
        age = subject_info.get('birthday', None)
        meas_date = raw.info.get('meas_date', None)
        if meas_date is not None and age is not None:
            bday = datetime(age[0], age[1], age[2])
            if isinstance(meas_date, (np.ndarray, list)):
                meas_date = meas_date[0]
            meas_date = datetime.fromtimestamp(meas_date)
            subject_age = age_on_date(bday, meas_date)
        else:
            subject_age = "n/a"

        data.update({'age': [subject_age], 'sex': [sex], 'group': [group]})

    # append the participant data to the existing file if it exists
    if os.path.exists(fname):
        df = pd.read_csv(fname, sep='\t')
        df = df.append(pd.DataFrame(data=data,
                                    columns=['participant_id', 'age',
                                             'sex', 'group']))
        df.drop_duplicates(subset='participant_id', keep='last',
                           inplace=True)
        df = df.sort_values(by='participant_id')
    else:
        df = pd.DataFrame(data=data,
                          columns=['participant_id', 'age', 'sex',
                                   'group'])

    df.to_csv(fname, sep='\t', index=False, na_rep='n/a')
    if verbose:
        print(os.linesep + "Writing '%s'..." % fname + os.linesep)
        print(df.head())

    return fname


def _scans_tsv(raw, raw_fname, fname, verbose):
    """Create a scans.tsv file and save it.

    Parameters
    ----------
    raw : instance of Raw
        The data as MNE-Python Raw object.
    raw_fname : str
        Relative path to the raw data file.
    fname : str
        Filename to save the scans.tsv to.
    verbose : bool
        Set verbose output to true or false.

    """
    # get MEASurement date from the data info
    meas_date = raw.info['meas_date']
    if isinstance(meas_date, (np.ndarray, list)):
        meas_date = meas_date[0]

    if meas_date is None:
        acq_time = 'n/a'
    else:
        acq_time = datetime.fromtimestamp(
            meas_date).strftime('%Y-%m-%dT%H:%M:%S')

    df = pd.DataFrame(data={'filename': ['%s' % raw_fname],
                            'acq_time': [acq_time]},
                      columns=['filename', 'acq_time'])

    df.to_csv(fname, sep='\t', index=False, na_rep='n/a')

    if verbose:
        print(os.linesep + "Writing '%s'..." % fname + os.linesep)
        print(df.head())

    return fname


def _coordsystem_json(raw, unit, orient, manufacturer, fname, verbose):
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
    _write_json(fid_json, fname)

    return fname


def _sidecar_json(raw, task, manufacturer, fname, kind,
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
    verbose : bool
        Set verbose output to true or false. Defaults to true.

    """
    sfreq = raw.info['sfreq']
    powerlinefrequency = raw.info.get('line_freq', None)
    if powerlinefrequency is None:
        warn('No line frequency found, defaulting to 50 Hz')
        powerlinefrequency = 50

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
        ('PowerLineFrequency', powerlinefrequency)]
    ch_info_json_meg = [
        ('SamplingFrequency', sfreq),
        ("DewarPosition", "XXX"),
        ("DigitizedLandmarks", False),
        ("DigitizedHeadPoints", False),
        ("SoftwareFilters", "n/a"),
        ('MEGChannelCount', n_megchan),
        ('MEGREFChannelCount', n_megrefchan)]
    ch_info_json_ieeg = [
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
    elif kind == 'ieeg':
        append_kind_json = ch_info_json_ieeg
    else:
        raise ValueError('Unexpected "kind": {}'
                         ' Use one of: {}'.format(kind, ALLOWED_KINDS))

    ch_info_json += append_kind_json
    ch_info_json += ch_info_ch_counts
    ch_info_json = OrderedDict(ch_info_json)

    _write_json(ch_info_json, fname, verbose=verbose)
    return fname


def raw_to_bids(subject_id, task, raw_file, kind, output_path,
                session_id=None, run=None, events_data=None, event_id=None,
                hpi=None, electrode=None, hsp=None, config=None,
                overwrite=True, verbose=True):
    """Walk over a folder of files and create BIDS compatible folder.

    Parameters
    ----------
    subject_id : str
        The subject name in BIDS compatible format ('01', '02', etc.)
    task : str
        Name of the task the data is based on.
    raw_file : str | instance of mne.Raw
        The raw data. If a string, it is assumed to be the path to the raw data
        file. Otherwise it must be an instance of mne.Raw
    kind : str, one of ('meg', 'ieeg')
        The kind of data being converted.
    output_path : str
        The path of the BIDS compatible folder
    session_id : str | None
        The session name in BIDS compatible format.
    run : int | None
        The run number for this dataset.
    events_data : str | array | None
        The events file. If a string, a path to the events file. If an array,
        the MNE events array (shape n_events, 3). If None, events will be
        inferred from the stim channel using `mne.find_events`.
    event_id : dict | None
        The event id dict used to create a 'trial_type' column in events.tsv
    hpi : None | str | list of str
        Marker points representing the location of the marker coils with
        respect to the MEG Sensors, or path to a marker file.
        If list, all of the markers will be averaged together.
    electrode : None | str
        Digitizer points representing the location of the fiducials and the
        marker coils with respect to the digitized head shape, or path to a
        file containing these points.
    hsp : None | str | array, shape = (n_points, 3)
        Digitizer head shape points, or path to head shape file. If more than
        10`000 points are in the head shape, they are automatically decimated.
    config : str | None
        A path to the configuration file to use if the data is from a BTi
        system.
    overwrite : bool
        If the file already exists, whether to overwrite it.
    verbose : bool
        If verbose is True, this will print a snippet of the sidecar files. If
        False, no content will be printed.

    Note
    ----
    For the participants.tsv file, the raw.info['subjects_info'] should be
    updated and raw.info['meas_date'] should not be None to compute the age
    of the participant correctly.
    """
    if isinstance(raw_file, string_types):
        # We must read in the raw data
        raw = _read_raw(raw_file, electrode=electrode, hsp=hsp, hpi=hpi,
                        config=config, verbose=verbose)
        _, ext = _parse_ext(raw_file, verbose=verbose)
        raw_fname = raw_file
    elif isinstance(raw_file, BaseRaw):
        # We got a raw mne object, get back the filename if possible
        # Assume that if no filename attr exists, it's a fif file.
        raw = raw_file.copy()
        if hasattr(raw, 'filenames'):
            _, ext = _parse_ext(raw.filenames[0], verbose=verbose)
            raw_fname = raw.filenames[0]
        else:
            # FIXME: How to get the filename if no filenames attribute?
            raw_fname = 'unknown_file_name'
            ext = '.fif'
    else:
        raise ValueError('raw_file must be an instance of str or BaseRaw, '
                         'got %s' % type(raw_file))

    data_path = make_bids_folders(subject=subject_id, session=session_id,
                                  kind=kind, root=output_path,
                                  overwrite=overwrite,
                                  verbose=verbose)
    if session_id is None:
        ses_path = data_path
    else:
        ses_path = make_bids_folders(subject=subject_id, session=session_id,
                                     root=output_path,
                                     overwrite=False,
                                     verbose=verbose)

    # create filenames
    scans_fname = make_bids_filename(
        subject=subject_id, session=session_id, suffix='scans.tsv',
        prefix=ses_path)
    participants_fname = make_bids_filename(prefix=output_path,
                                            suffix='participants.tsv')
    coordsystem_fname = make_bids_filename(
        subject=subject_id, session=session_id,
        suffix='coordsystem.json', prefix=data_path)
    data_meta_fname = make_bids_filename(
        subject=subject_id, session=session_id, task=task, run=run,
        suffix='%s.json' % kind, prefix=data_path)
    if ext in ['.fif', '.ds']:
        raw_file_bids = make_bids_filename(
            subject=subject_id, session=session_id, task=task, run=run,
            suffix='%s%s' % (kind, ext))
    else:
        raw_folder = make_bids_filename(
            subject=subject_id, session=session_id, task=task, run=run,
            suffix='%s' % kind)
        raw_file_bids = make_bids_filename(
            subject=subject_id, session=session_id, task=task, run=run,
            suffix='%s%s' % (kind, ext), prefix=raw_folder)
    events_tsv_fname = make_bids_filename(
        subject=subject_id, session=session_id, task=task,
        run=run, suffix='events.tsv', prefix=data_path)
    channels_fname = make_bids_filename(
        subject=subject_id, session=session_id, task=task, run=run,
        suffix='channels.tsv', prefix=data_path)

    # Read in Raw object and extract metadata from Raw object if needed
    if kind == 'meg':
        orient = ORIENTATION[ext]
        unit = UNITS[ext]
        manufacturer = MANUFACTURERS[ext]
    else:
        orient = 'n/a'
        unit = 'n/a'
        manufacturer = 'n/a'

    # save stuff
    if kind == 'meg':
        _scans_tsv(raw, os.path.join(kind, raw_file_bids), scans_fname,
                   verbose)
        _coordsystem_json(raw, unit, orient, manufacturer, coordsystem_fname,
                          verbose)

    make_dataset_description(output_path, name=" ",
                             verbose=verbose)
    _sidecar_json(raw, task, manufacturer, data_meta_fname, kind,
                  verbose)
    _participants_tsv(raw, subject_id, "n/a", participants_fname, verbose)
    _channels_tsv(raw, channels_fname, verbose)

    events = _read_events(events_data, raw)
    if len(events) > 0:
        _events_tsv(events, raw, events_tsv_fname, event_id, verbose)

    # set the raw file name to now be the absolute path to ensure the files
    # are placed in the right location
    raw_file_bids = os.path.join(data_path, raw_file_bids)
    if os.path.exists(raw_file_bids) and not overwrite:
        raise ValueError('"%s" already exists. Please set'
                         ' overwrite to True.' % raw_file_bids)
    _mkdir_p(os.path.dirname(raw_file_bids))

    if verbose:
        print('Writing data files to %s' % raw_file_bids)

    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError('ext must be in %s, got %s'
                         % (''.join(ALLOWED_EXTENSIONS), ext))

    # for FIF, we need to re-save the file to fix the file pointer
    # for files with multiple parts
    if ext in ['.fif']:
        raw.save(raw_file_bids, overwrite=overwrite)
    elif ext == '.ds':
        sh.copytree(raw_fname, raw_file_bids)
    else:
        sh.copyfile(raw_fname, raw_file_bids)

    return output_path
