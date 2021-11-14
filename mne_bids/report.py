"""Make BIDS report from dataset and sidecar files."""
# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD-3-Clause
import json
import os.path as op
import textwrap
from pathlib import Path

import numpy as np
from mne.externals.tempita import Template
from mne.utils import warn, logger, verbose

from mne_bids.config import DOI, ALLOWED_DATATYPES
from mne_bids.tsv_handler import _from_tsv
from mne_bids.path import (get_bids_path_from_fname, get_datatypes,
                           get_entity_vals, BIDSPath,
                           _parse_ext, _find_matching_sidecar)

# functions to be used inside Template strings
FUNCTION_TEMPLATE = """{{py:
def _pretty_str(listed):
    # make strings a sequence of ',' and 'and'
    if not isinstance(listed, list):
        listed = list(listed)

    if len(listed) <= 1:
        return ','.join(listed)
    return '{}, and {}'.format(', '.join(listed[:-1]), listed[-1])

def _range_str(minval, maxval, meanval, stdval, n_unknown, type):
    if minval == 'n/a':
        return 'ages all unknown'

    if n_unknown > 0:
        unknown_str = f'; {n_unknown} with unknown {type}'
    else:
        unknown_str = ''
    return f'ages ranged from {round(minval, 2)} to {round(maxval, 2)} '\
           f'(mean = {round(meanval, 2)}, std = {round(stdval, 2)}{unknown_str})'

def _summarize_participant_hand(hands):
    n_unknown = len([hand for hand in hands if hand == 'n/a'])

    if n_unknown == len(hands):
        return f'handedness were all unknown'
    n_rhand = len([hand for hand in hands if hand.upper() == 'R'])
    n_lhand = len([hand for hand in hands if hand.upper() == 'L'])
    n_ambidex = len([hand for hand in hands if hand.upper() == 'A'])

    return f'comprised of {n_rhand} right hand, {n_lhand} left hand ' \
           f'and {n_ambidex} ambidextrous'

def _summarize_participant_sex(sexs):
    n_unknown = len([sex for sex in sexs if sex == 'n/a'])

    if n_unknown == len(sexs):
        return f'sex were all unknown'
    n_males = len([sex for sex in sexs if sex.upper() == 'M'])
    n_females = len([sex for sex in sexs if sex.upper() == 'F'])

    return f'comprised of {n_males} male and {n_females} female participants'

def _length_recording_str(length_recordings):
    import numpy as np
    if length_recordings is None:
        return ''

    min_record_length = round(np.min(length_recordings), 2)
    max_record_length = round(np.max(length_recordings), 2)
    mean_record_length = round(np.mean(length_recordings), 2)
    std_record_length = round(np.std(length_recordings), 2)
    total_record_length = round(sum(length_recordings), 2)

    return f' Recording durations ranged from {min_record_length} to {max_record_length} seconds '\
           f'(mean = {mean_record_length}, std = {std_record_length}), '\
           f'for a total of {total_record_length} seconds of data recorded '\
           f'over all scans.'

def _summarize_software_filters(software_filters):
    if software_filters in [{}, 'n/a']:
        return ''

    msg = ''
    for key, value in software_filters.items():
        msg += f'{key}'

        if isinstance(value, dict) and value:
            parameters = []
            for param_name, param_value in value.items():
                if param_name and param_value:
                    parameters.append(f'{param_value} {param_name}')
            if parameters:
                msg += ' with parameters '
                msg += ', '.join(parameters)
    return msg

}}"""  # noqa

BIDS_DATASET_TEMPLATE = \
    """{{if name == 'n/a'}}This{{else}}The {{name}}{{endif}}
dataset was created by {{_pretty_str(authors)}} and conforms to BIDS version
{{bids_version}}. This report was generated with
MNE-BIDS ({{mne_bids_doi}}). """
BIDS_DATASET_TEMPLATE += \
    """The dataset consists of
{{n_subjects}} participants ({{PARTICIPANTS_TEMPLATE}}) {{if n_sessions}}and {{n_sessions}} recording sessions: {{(_pretty_str(sessions))}}.{{else}}.{{endif}} """  # noqa

PARTICIPANTS_TEMPLATE = \
    """{{_summarize_participant_sex(sexs)}};
{{_summarize_participant_hand(hands)}}; {{_range_str(min_age, max_age, mean_age, std_age, n_age_unknown, 'age')}}"""  # noqa

DATATYPE_AGNOSTIC_TEMPLATE = \
    """Data was recorded using a {{_pretty_str(system)}} system
{{if manufacturer}}({{_pretty_str(manufacturer)}} manufacturer){{endif}}
sampled at {{_pretty_str(sfreq)}} Hz
with line noise at {{_pretty_str(powerlinefreq)}} Hz.
{{if _summarize_software_filters(software_filters)}}
The following software filters were applied during recording: 
{{_summarize_software_filters(software_filters)}}.
{{endif}}
{{if n_scans > 1}}
There were {{n_scans}} scans in total.
{{else}}There was {{n_scans}} scan in total.
{{endif}}
{{_length_recording_str(length_recordings)}}
For each dataset, there were on average {{mean_chs}} (std = {{std_chs}}) recording channels per scan,
out of which {{mean_good_chs}} (std = {{std_good_chs}}) were used in analysis
({{mean_bad_chs}} +/- {{std_bad_chs}} were removed from analysis). """  # noqa


def _pretty_dict(template_dict):
    """Remove problematic blank spaces."""
    for key, val in template_dict.items():
        if val == ' ':
            template_dict[key] = 'n/a'


def _summarize_dataset(root):
    """Summarize the dataset_desecription.json file.

    Required dataset descriptors include:
        - Name
        - BIDSVersion

    Added descriptors include:
        - Authors
        - DOI

    Parameters
    ----------
    root : path-like
        The path of the root of the BIDS compatible folder.

    Returns
    -------
    template_dict : dict
        A dictionary of values for various template strings.
    """
    dataset_descrip_fpath = op.join(root,
                                    'dataset_description.json')
    if not op.exists(dataset_descrip_fpath):
        return dict()

    # read file and 'REQUIRED' components of it
    with open(dataset_descrip_fpath, 'r', encoding='utf-8-sig') as fin:
        dataset_description = json.load(fin)

    # create dictionary to pass into template string
    name = dataset_description['Name']
    bids_version = dataset_description['BIDSVersion']
    authors = dataset_description['Authors']
    template_dict = {
        'name': name,
        'bids_version': bids_version,
        'mne_bids_doi': DOI,
        'authors': authors,
    }
    _pretty_dict(template_dict)
    return template_dict


def _summarize_participants_tsv(root):
    """Summarize `participants.tsv` file in BIDS root directory.

    Parameters
    ----------
    root : path-like
        The path of the root of the BIDS compatible folder.

    Returns
    -------
    template_dict : dict
        A dictionary of values for various template strings.
    """
    participants_tsv_fpath = op.join(root, 'participants.tsv')
    if not op.exists(participants_tsv_fpath):
        return dict()

    participants_tsv = _from_tsv(str(participants_tsv_fpath))
    p_ids = participants_tsv['participant_id']
    logger.info(f'Summarizing participants.tsv {participants_tsv_fpath}...')

    # summarize sex count statistics
    keys = ['M', 'F', 'n/a']
    p_sex = participants_tsv.get('sex')
    # phrasing works for both sex and gender
    p_gender = participants_tsv.get('gender')
    sexs = ['n/a']
    if p_sex or p_gender:
        # only summarize sex if it conforms to `keys` referenced above
        p_sex = p_gender if p_sex is None else p_sex
        if all([sex.upper() in keys
                for sex in p_sex if sex != 'n/a']):
            sexs = p_sex

    # summarize hand count statistics
    keys = ['R', 'L', 'A', 'n/a']
    p_hands = participants_tsv.get('hand')
    hands = ['n/a']
    if p_hands:
        # only summarize handedness if it conforms to
        # mne-bids handedness
        if all([hand.upper() in keys
                for hand in p_hands if hand != 'n/a']):
            hands = p_hands

    # summarize age statistics: mean, std, min, max
    p_ages = participants_tsv.get('age')
    min_age, max_age = 'n/a', 'n/a'
    mean_age, std_age = 'n/a', 'n/a'
    n_age_unknown = len(p_ages)
    if p_ages:
        # only summarize age if they are numerics
        if all([age.isnumeric() for age in p_ages if age != 'n/a']):
            age_list = [float(age) for age in p_ages if age != 'n/a']
            n_age_unknown = len(p_ids) - len(age_list)
            if age_list:
                min_age, max_age = np.min(age_list), np.max(age_list)
                mean_age, std_age = np.mean(age_list), np.std(age_list)

    template_dict = {
        'sexs': sexs,
        'hands': hands,
        'n_age_unknown': n_age_unknown,
        'mean_age': mean_age,
        'std_age': std_age,
        'min_age': min_age,
        'max_age': max_age,
    }
    return template_dict


def _summarize_scans(root, session=None):
    """Summarize scans in BIDS root directory.

    Summarizes scans only if there is a *_scans.tsv file.

    Parameters
    ----------
    root : path-like
        The path of the root of the BIDS compatible folder.
    session : str, optional
        The session for a item. Corresponds to "ses".

    Returns
    -------
    template_dict : dict
        A dictionary of values for various template strings.

    """
    root = Path(root)
    if session is None:
        search_str = '*_scans.tsv'
    else:
        search_str = f'*ses-{session}' \
                     f'*_scans.tsv'
    scans_fpaths = list(root.rglob(search_str))
    if len(scans_fpaths) == 0:
        warn('No *scans.tsv files found. Currently, '
             'we do not generate a report without the scans.tsv files.')
        return dict()

    logger.info(f'Summarizing scans.tsv files {scans_fpaths}...')

    # summarize sidecar.json, channels.tsv template
    sidecar_dict = _summarize_sidecar_json(root, scans_fpaths)
    channels_dict = _summarize_channels_tsv(root, scans_fpaths)
    template_dict = dict()
    template_dict.update(**sidecar_dict)
    template_dict.update(**channels_dict)

    return template_dict


def _summarize_sidecar_json(root, scans_fpaths):
    """Summarize scans in BIDS root directory.

    Parameters
    ----------
    root : path-like
        The path of the root of the BIDS compatible folder.
    scans_fpaths : list
        A list of all *_scans.tsv files in ``root``. The summary
        will occur for all scans listed in the *_scans.tsv files.

    Returns
    -------
    template_dict : dict
        A dictionary of values for various template strings.

    """
    n_scans = 0
    powerlinefreqs, sfreqs = set(), set()
    manufacturers = set()
    length_recordings = []

    # loop through each scan
    for scan_fpath in scans_fpaths:
        # load in the scans.tsv file
        # and read metadata for each scan
        scans_tsv = _from_tsv(scan_fpath)
        scans = scans_tsv['filename']
        for scan in scans:
            # summarize metadata of recordings
            bids_path, ext = _parse_ext(scan)
            datatype = op.dirname(scan)
            if datatype not in ALLOWED_DATATYPES:
                continue

            n_scans += 1

            # convert to BIDS Path
            if not isinstance(bids_path, BIDSPath):
                bids_path = get_bids_path_from_fname(bids_path)
            bids_path.root = root

            # XXX: improve to allow emptyroom
            if bids_path.subject == 'emptyroom':
                continue

            sidecar_fname = _find_matching_sidecar(bids_path=bids_path,
                                                   suffix=datatype,
                                                   extension='.json')
            with open(sidecar_fname, 'r', encoding='utf-8-sig') as fin:
                sidecar_json = json.load(fin)

            # aggregate metadata from each scan
            # REQUIRED kwargs
            sfreq = sidecar_json['SamplingFrequency']
            powerlinefreq = str(sidecar_json['PowerLineFrequency'])
            software_filters = sidecar_json.get('SoftwareFilters')
            if not software_filters:
                software_filters = 'n/a'

            # RECOMMENDED kwargs
            manufacturer = sidecar_json.get('Manufacturer', 'n/a')
            record_duration = sidecar_json.get('RecordingDuration', 'n/a')

            sfreqs.add(str(np.round(sfreq, 2)))
            powerlinefreqs.add(str(powerlinefreq))
            if manufacturer != 'n/a':
                manufacturers.add(manufacturer)
            length_recordings.append(record_duration)

    # XXX: length summary is only allowed, if no 'n/a' was found
    if any([dur == 'n/a' for dur in length_recordings]):
        length_recordings = None

    template_dict = {
        'n_scans': n_scans,
        'manufacturer': list(manufacturers),
        'sfreq': sfreqs,
        'powerlinefreq': powerlinefreqs,
        'software_filters': software_filters,
        'length_recordings': length_recordings,
    }
    return template_dict


def _summarize_channels_tsv(root, scans_fpaths):
    """Summarize channels.tsv data in BIDS root directory.

    Currently, summarizes all REQUIRED components of channels
    data, and some RECOMMENDED and OPTIONAL components.

    Parameters
    ----------
    root : path-like
        The path of the root of the BIDS compatible folder.
    scans_fpaths : list
        A list of all *_scans.tsv files in ``root``. The summary
        will occur for all scans listed in the *_scans.tsv files.

    Returns
    -------
    template_dict : dict
        A dictionary of values for various template strings.
    """
    root = Path(root)

    # keep track of channel type, status
    ch_status_count = {'bad': [], 'good': []}
    ch_count = []

    # loop through each scan
    for scan_fpath in scans_fpaths:
        # load in the scans.tsv file
        # and read metadata for each scan
        scans_tsv = _from_tsv(scan_fpath)
        scans = scans_tsv['filename']
        for scan in scans:
            # summarize metadata of recordings
            bids_path, _ = _parse_ext(scan)
            datatype = op.dirname(scan)
            if datatype not in ['meg', 'eeg', 'ieeg']:
                continue

            # convert to BIDS Path
            if not isinstance(bids_path, BIDSPath):
                bids_path = get_bids_path_from_fname(bids_path)
            bids_path.root = root

            # XXX: improve to allow emptyroom
            if bids_path.subject == 'emptyroom':
                continue

            channels_fname = _find_matching_sidecar(bids_path=bids_path,
                                                    suffix='channels',
                                                    extension='.tsv')

            # summarize channels.tsv
            channels_tsv = _from_tsv(channels_fname)
            for status in ch_status_count.keys():
                ch_status = [ch for ch in channels_tsv['status']
                             if ch == status]
                ch_status_count[status].append(len(ch_status))
            ch_count.append(len(channels_tsv['name']))

    # create summary template strings for status
    template_dict = {
        'mean_chs': np.mean(ch_count),
        'std_chs': np.std(ch_count),
        'mean_good_chs': np.mean(ch_status_count['good']),
        'std_good_chs': np.std(ch_status_count['good']),
        'mean_bad_chs': np.mean(ch_status_count['bad']),
        'std_bad_chs': np.std(ch_status_count['bad']),
    }
    for key, val in template_dict.items():
        template_dict[key] = round(val, 2)
    return template_dict


@verbose
def make_report(root, session=None, verbose=None):
    """Create a methods paragraph string from BIDS dataset.

    Summarizes the REQUIRED components in the BIDS specification
    and also some RECOMMENDED components. Currently, the methods
    paragraph summarize the:

      - dataset_description.json file
      - (optional) participants.tsv file
      - (optional) datatype-agnostic files for (M/I)EEG data,
        which reads files from the ``*_scans.tsv`` file.

    Parameters
    ----------
    root : path-like
        The path of the root of the BIDS compatible folder.
    session : str | None
            The (optional) session for a item. Corresponds to "ses".
    %(verbose)s

    Returns
    -------
    paragraph : str
        The paragraph wrapped with 80 characters per line
        describing the summary of the subjects.
    """
    # high level summary
    subjects = get_entity_vals(root, entity_key='subject')
    sessions = get_entity_vals(root, entity_key='session')
    modalities = get_datatypes(root)

    # only summarize allowed modalities (MEG/EEG/iEEG) data
    # map them to a pretty looking string
    datatype_map = {
        'meg': 'MEG',
        'eeg': 'EEG',
        'ieeg': 'iEEG',
    }
    modalities = [datatype_map[datatype] for datatype in modalities
                  if datatype in datatype_map.keys()]

    # REQUIRED: dataset_description.json summary
    dataset_summary = _summarize_dataset(root)

    # RECOMMENDED: participants summary
    participant_summary = _summarize_participants_tsv(root)

    # RECOMMENDED: scans summary
    scans_summary = _summarize_scans(root, session=session)

    # turn off 'recommended' report summary
    # if files are not available to summarize
    if not participant_summary:
        participant_template = ''
    else:
        content = f'{FUNCTION_TEMPLATE}{PARTICIPANTS_TEMPLATE}'
        participant_template = Template(content=content)
        participant_template = participant_template.substitute(
            **participant_summary)
        logger.info(f'The participant template found: {participant_template}')

    dataset_summary['PARTICIPANTS_TEMPLATE'] = str(participant_template)

    if not scans_summary:
        datatype_agnostic_template = ''
    else:
        datatype_agnostic_template = DATATYPE_AGNOSTIC_TEMPLATE

    dataset_summary.update({
        'system': modalities,
        'n_subjects': len(subjects),
        'n_sessions': len(sessions),
        'sessions': sessions,
    })

    # XXX: add channel summary for modalities (ieeg, meg, eeg)
    # create the content and mne Template
    # lower-case templates are "Recommended",
    # while upper-case templates are "Required".
    content = f'{FUNCTION_TEMPLATE}{BIDS_DATASET_TEMPLATE}' \
              f'{datatype_agnostic_template}'

    paragraph = Template(content=content)
    paragraph = paragraph.substitute(**dataset_summary,
                                     **scans_summary)

    # Clean paragraph
    paragraph = paragraph.replace('\n', ' ')
    paragraph = paragraph.replace('  ', ' ')

    return '\n'.join(textwrap.wrap(paragraph, width=80))
