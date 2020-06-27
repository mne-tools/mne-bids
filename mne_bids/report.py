"""Make BIDS report from dataset and sidecar files."""
# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)
import json
import os.path as op
import textwrap
from pathlib import Path

import mne
import numpy as np
from mne.externals.tempita import Template, sub

from mne_bids.config import DOI, ALLOWED_KINDS
from mne_bids.tsv_handler import _from_tsv
from mne_bids.utils import (make_bids_basename, get_kinds,
                            get_entity_vals, _parse_ext,
                            _find_matching_sidecar, _parse_bids_filename,
                            BIDSPath)

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
        return 'all unknown'
        
    if n_unknown > 0:
        unknown_str = f'; {n_unknown} with unknown {type}'
    else:
        unknown_str = ''
    return f'{round(minval, 2)} to {round(maxval, 2)} '\
           f'({round(meanval, 2)} +/- {round(stdval, 2)}{unknown_str})'
           
def _summarize_participant_hand(hands):
    n_unknown = len([hand for hand in hands if hand == 'n/a'])

    if n_unknown == len(hands):
        return f'all unknown'
    n_rhand = len([hand for hand in hands if hand == 'R'])
    n_lhand = len([hand for hand in hands if hand == 'L'])
    n_ambidex = len([hand for hand in hands if hand == 'A'])

    return f'{n_rhand} right hand, {n_lhand} left hand ' \
           f'and {n_ambidex} ambidextrous subjects'

def _summarize_participant_sex(sexs):
    n_unknown = len([sex for sex in sexs if sex == 'n/a'])

    if n_unknown == len(sexs):
        return f'all unknown'
    n_males = len([sex for sex in sexs if sex == 'M'])
    n_females = len([sex for sex in sexs if sex == 'F'])

    return f'{n_males} males and {n_females} females'        

def _length_recording_str(length_recordings):
    import numpy as np
    if length_recordings is None:
        return ''
    
    min_record_length = round(np.min(length_recordings), 2)
    max_record_length = round(np.max(length_recordings), 2)
    mean_record_length = round(np.mean(length_recordings), 2)
    std_record_length = round(np.std(length_recordings), 2)
    total_record_length = round(sum(length_recordings), 2)
    
    return f'Each dataset is {min_record_length} to {max_record_length} seconds, '\
           f'for a total of {total_record_length} seconds of data recorded '\
           f'({mean_record_length} +/- {std_record_length}).' 
}}"""  # noqa

BIDS_DATASET_TEMPLATE = """{{if name == 'n/a'}}This{{else}}The {{name}}{{endif}}
dataset was created with BIDS version {{bids_version}}
by {{_pretty_str(authors)}}. This report was generated with
MNE-BIDS ({{mne_bids_doi}}). There are {{n_subjects}} subjects.
The dataset consists of {{n_sessions}} recording sessions
{{(_pretty_str(sessions))}}. """

PARTICIPANTS_TEMPLATE = """Sex of the subjects are {{_summarize_participant_sex(sexs)}}.
Handedness of the subjects are {{_summarize_participant_hand(hands)}}.
Ages of the subjects are
{{_range_str(min_age, max_age, mean_age, std_age, n_age_unknown, 'age')}}. """

MODALITY_AGNOSTIC_TEMPLATE = """
Data was acquired using a {{_pretty_str(system)}} system
({{_pretty_str(manufacturer)}} manufacturer) with line noise at
{{_pretty_str(powerlinefreq)}} Hz{{if software_filters != 'n/a'}} using filters
({{software_filters}}).{{else}}.{{endif}}
There are {{n_scans}} total scans, {{n_chs}} channels ({{n_good}} are used and
{{n_bad}} are removed from analysis).
{{_length_recording_str(length_recordings)}} """

IEEG_TEMPLATE = """There are {{n_ecog_chs}} ECoG and
{{n_seeg_chs}} SEEG channels. """


def _pretty_dict(template_dict):
    """Remove problematic blank spaces."""
    for key, val in template_dict.items():
        if val == ' ':
            template_dict[key] = 'n/a'


def _summarize_dataset(bids_root):
    """Summarize the dataset_desecription.json file.

    Required dataset descriptors include:
        - Name
        - BIDSVersion

    Added descriptors include:
        - Authors
        - DOI

    Parameters
    ----------
    bids_root : str | pathlib.Path
        The path of the root of the BIDS compatible folder.

    Returns
    -------
    template_dict : dict
        A dictionary of values for various template strings.
    """
    dataset_descrip_fpath = make_bids_basename(
        prefix=bids_root, suffix='dataset_description.json')
    if not op.exists(dataset_descrip_fpath):
        return dict()

    # read file and 'REQUIRED' components of it
    with open(dataset_descrip_fpath, 'r') as fin:
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


def _summarize_participants_tsv(bids_root, verbose=True):
    """Summarize `participants.tsv` file in BIDS root directory.

    Parameters
    ----------
    bids_root : str | pathlib.Path
        The path of the root of the BIDS compatible folder.
    verbose: bool
        Set verbose output to true or false.

    Returns
    -------
    template_dict : dict
        A dictionary of values for various template strings.
    """
    participants_tsv_fpath = make_bids_basename(prefix=bids_root,
                                                suffix='participants.tsv')
    if not op.exists(participants_tsv_fpath):
        return dict()

    participants_tsv = _from_tsv(str(participants_tsv_fpath))
    p_ids = participants_tsv['participant_id']
    if verbose:
        print(f'Trying to summarize participants.tsv...')

    # summarize sex count statistics
    keys = ['M', 'F', 'n/a']
    p_sex = participants_tsv.get('sex')
    sexs = ['n/a']
    if p_sex:
        # only summarize sex if it conforms to
        # mne-bids handedness
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


def _summarize_scans(bids_root, session=None, verbose=True):
    """Summarize scans in BIDS root directory.

    Parameters
    ----------
    bids_root : str | pathlib.Path
        The path of the root of the BIDS compatible folder.
    session : str, optional
        The session for a item. Corresponds to "ses".
    verbose : bool
        Set verbose output to true or false.

    Returns
    -------
    template_dict : dict
        A dictionary of values for various template strings.
    """
    bids_root = Path(bids_root)
    if session is None:
        search_str = '*_scans.tsv'
    else:
        search_str = f'*ses-{session}' \
                     f'*_scans.tsv'
    scans_fpaths = list(bids_root.rglob(search_str))
    if len(scans_fpaths) == 0:
        print('No *scans.tsv files found. Currently, '
              'we do not generate a report without the scans.tsv files.')
        return dict()

    # summarize sidecar.json, channels.tsv template
    sidecar_dict = _summarize_sidecar_json(bids_root, scans_fpaths,
                                           verbose=verbose)
    channels_dict = _summarize_channels_tsv(bids_root, scans_fpaths,
                                            verbose=verbose)
    template_dict = dict()
    template_dict.update(**sidecar_dict)
    template_dict.update(**channels_dict)

    return template_dict


def _summarize_sidecar_json(bids_root, scans_fpaths, verbose=True):
    """Summarize scans in BIDS root directory.

    Parameters
    ----------
    bids_root : str | pathlib.Path
        The path of the root of the BIDS compatible folder.
    scans_fpaths : list
        A list of all *_scans.tsv files in bids_root. The summary
        will occur for all scans listed in the *_scans.tsv files.
    verbose : bool
        Set verbose output to true or false.

    Returns
    -------
    template_dict : dict
        A dictionary of values for various template strings.
    """
    n_scans = 0
    n_ecog_chs, n_seeg_chs, n_eeg_chs = 0, 0, 0
    powerlinefreqs, sfreqs = set(), set()
    manufacturers = set('n/a')
    length_recordings = []

    # loop through each scan
    for scan_fpath in scans_fpaths:
        # load in the scans.tsv file
        # and read metadata for each scan
        scans_tsv = _from_tsv(scan_fpath)
        scans = scans_tsv['filename']
        for scan in scans:
            # summarize metadata of recordings
            bids_basename, ext = _parse_ext(scan)
            kind = op.dirname(scan)
            if kind not in ALLOWED_KINDS:
                continue

            n_scans += 1

            # convert to BIDS Path
            params = _parse_bids_filename(bids_basename, verbose)
            bids_basename = BIDSPath(subject=params.get('sub'),
                                     session=params.get('ses'),
                                     recording=params.get('rec'),
                                     acquisition=params.get('acq'),
                                     processing=params.get('proc'),
                                     space=params.get('space'),
                                     run=params.get('run'),
                                     task=params.get('task'),
                                     )

            # XXX: improve to allow emptyroom
            if bids_basename.subject == 'emptyroom':
                continue

            sidecar_fname = _find_matching_sidecar(bids_fname=bids_basename,
                                                   bids_root=bids_root,
                                                   suffix=f'{kind}.json')
            with open(sidecar_fname, 'r') as fin:
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
            n_eeg_chs += sidecar_json.get('EEGChannelCount', 0)
            n_ecog_chs += sidecar_json.get('ECOGChannelCount', 0)
            n_seeg_chs += sidecar_json.get('SEEGChannelCount', 0)

            sfreqs.add(str(sfreq))
            powerlinefreqs.add(str(powerlinefreq))
            manufacturers.add(manufacturer)
            length_recordings.append(record_duration)

    # XXX: length summary is only allowed, if no 'n/a' was found
    if any([dur == 'n/a' for dur in length_recordings]):
        length_recordings = None

    template_dict = {
        'n_scans': n_scans,
        'manufacturer': manufacturers,
        'sfreq': sfreqs,
        'powerlinefreq': powerlinefreqs,
        'software_filters': software_filters,
        'n_ecog_chs': n_ecog_chs,
        'n_seeg_chs': n_seeg_chs,
        'n_eeg_chs': n_eeg_chs,
        'length_recordings': length_recordings
    }
    return template_dict


def _summarize_channels_tsv(bids_root, scans_fpaths, verbose=True):
    """Summarize channels.tsv data in BIDS root directory.

    Currently, summarizes all REQUIRED components of channels
    data, and some RECOMMENDED and OPTIONAL components.

    Parameters
    ----------
    bids_root : str | pathlib.Path
        The path of the root of the BIDS compatible folder.
    scans_fpaths : list
        A list of all *_scans.tsv files in bids_root. The summary
        will occur for all scans listed in the *_scans.tsv files.
    verbose : bool

    Returns
    -------
    template_dict : dict
        A dictionary of values for various template strings.
    """
    bids_root = Path(bids_root)

    # keep track of channel type, status
    ch_status_count = {'bad': 0, 'good': 0}

    # loop through each scan
    for scan_fpath in scans_fpaths:
        # load in the scans.tsv file
        # and read metadata for each scan
        scans_tsv = _from_tsv(scan_fpath)
        scans = scans_tsv['filename']
        for scan in scans:
            # summarize metadata of recordings
            bids_basename, _ = _parse_ext(scan)
            kind = op.dirname(scan)
            if kind not in ['meg', 'eeg', 'ieeg']:
                continue

            # convert to BIDS Path
            params = _parse_bids_filename(bids_basename, verbose)
            bids_basename = BIDSPath(subject=params.get('sub'),
                                     session=params.get('ses'),
                                     recording=params.get('rec'),
                                     acquisition=params.get('acq'),
                                     processing=params.get('proc'),
                                     space=params.get('space'),
                                     run=params.get('run'),
                                     task=params.get('task'),
                                     prefix=bids_root)

            # XXX: improve to allow emptyroom
            if bids_basename.subject == 'emptyroom':
                continue

            channels_fname = _find_matching_sidecar(bids_fname=bids_basename,
                                                    bids_root=bids_root,
                                                    suffix='channels.tsv')

            # summarize channels.tsv
            channels_tsv = _from_tsv(channels_fname)
            for status in channels_tsv['status']:
                ch_status_count[status] += 1

    # create summary template strings for status
    n_good = ch_status_count['good']
    n_bad = ch_status_count['bad']
    template_dict = {
        'n_chs': n_good + n_bad,
        'n_good': n_good,
        'n_bad': n_bad
    }
    return template_dict


def create_methods_paragraph(bids_root, session=None, verbose=True):
    """Create a methods paragraph string from BIDS dataset.

    Parameters
    ----------
    bids_root : str | pathlib.Path
    session : str , optional
    summarize_participants : bool
    verbose : bool

    Returns
    -------
    paragraph : str
    """
    # high level summary
    subjects = get_entity_vals(bids_root, entity_key='sub')
    sessions = get_entity_vals(bids_root, entity_key='ses')
    kinds = get_kinds(bids_root)

    # only summarize allowed kinds (MEEG data)
    kinds = [kind.upper() for kind in kinds if kind in ALLOWED_KINDS]

    # dataset_description.json summary
    dataset_template = _summarize_dataset(bids_root)

    # participants summary
    participants_template = _summarize_participants_tsv(bids_root)

    # scans summary
    scans_template = _summarize_scans(bids_root, session=session,
                                      verbose=verbose)

    # turn off 'recommended' report summary
    # if files are not available to summarize
    if not participants_template:
        participant_template = ''
    else:
        participant_template = PARTICIPANTS_TEMPLATE

    if not scans_template:
        modality_agnostic_template = ''
    else:
        modality_agnostic_template = MODALITY_AGNOSTIC_TEMPLATE

    dataset_template.update({
        'system': kinds,
        'n_subjects': len(subjects),
        'n_sessions': len(sessions),
        'sessions': sessions,
    })

    # create the content and mne Template
    # lower-case templates are "Recommended",
    # while upper-case templates are "Required".
    content = f'{FUNCTION_TEMPLATE}{BIDS_DATASET_TEMPLATE}' \
              f'{participant_template}{modality_agnostic_template}'
    paragraph = Template(content=content)
    paragraph = paragraph.substitute(**dataset_template,
                                     **participants_template,
                                     **scans_template)

    # add channel summary for kinds
    if 'IEEG' in kinds:
        paragraph = paragraph + sub(IEEG_TEMPLATE, **scans_template)

    return '\n'.join(textwrap.wrap(paragraph, width=80))


if __name__ == '__main__':
    bids_root = mne.datasets.somato.data_path()
    bids_root = '/Users/adam2392/Dropbox/epilepsy_bids/'
    # bids_root = '/Users/adam2392/Downloads/ds001779-1.0.2'
    # bids_root = '/Users/adam2392/Downloads/ds002778-1.0.1'
    # bids_root = "/Users/adam2392/Downloads/ds002904-1.0.0"
    # bids_root = "/Users/adam2392/Downloads/ds000117-master"
    # bids_root = "/Users/adam2392/Downloads/ds000246-master"
    # bids_root = "/Users/adam2392/Downloads/ds000248-master"
    # bids_root = "/Users/adam2392/Downloads/ds001810-master"
    # bids_root = "/Users/adam2392/Downloads/ds001971-master"
    methods_paragraph = create_methods_paragraph(bids_root)
    print(methods_paragraph)
