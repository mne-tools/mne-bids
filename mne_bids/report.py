"""Make BIDS report from dataset and sidecar files."""
# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)
import json
import os.path as op
from pathlib import Path

import numpy as np

from mne_bids.config import DOI
from mne_bids.datasets import fetch_brainvision_testing_data
from mne_bids.tsv_handler import _from_tsv
from mne_bids.utils import (make_bids_basename, get_kinds,
                            get_entity_vals, _parse_ext,
                            _find_matching_sidecar, _parse_bids_filename,
                            BIDSPath)

BIDS_DATASET_TEMPLATE = 'Dataset {name} was created ' \
                        'with BIDS version {bids_version} using ' \
                        'MNE-BIDS ({doi}).'

PARTICIPANTS_TEMPLATE = \
    'We had {n_subjects} subjects amongst whom there were ' \
    '{n_males} males and {n_females} females ({n_sex_unknown} unknown). ' \
    'There were {n_rhand} right hand, {n_lhand} left hand ' \
    'and {n_ambidex} ambidextrous subjects. ' \
    'Their ages were {min_age}-{max_age} ({mean_age} +/- {std_age} ' \
    'with {n_age_unknown} unknown).'

MODALITY_AGNOSTIC_TEMPLATE = \
    'Data was acquired using a {system} system ({manufacturer} manufacturer ' \
    'with line noise at {powerlinefreq} Hz) using ' \
    'filters ({software_filters}).' \
    'Each dataset was {min_record_length} to {max_record_length} seconds, ' \
    'for a total of {total_record_length} seconds of data recorded ' \
    '({mean_record_length} +/- {std_record_length}). ' \
    'The dataset consisted of {n_sessions} recording sessions ({sessions}), ' \
    '{n_chs} channels ({n_good} were used and ' \
    '{n_bad} were removed from analysis). '

IEEG_TEMPLATE = 'There were {n_ecog_chs} ECoG and {n_seeg_chs} SEEG channels.'


def _summarize_dataset(bids_root):
    """Summarize the dataset_desecription.json file.

    Parameters
    ----------
    bids_root : str | pathlib.Path
        The path of the root of the BIDS compatible folder.

    Returns
    -------
    template_dict : dict
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
    template_dict = {
        'name': name,
        'bids_version': bids_version,
        'doi': DOI,
    }
    return template_dict


def _summarize_subs(bids_root):
    """Summarize subjects in BIDS root directory.

    Parameters
    ----------
    bids_root : str | pathlib.Path
        The path of the root of the BIDS compatible folder.

    Returns
    -------
    template_dict : dict
    """
    participants_tsv_fpath = make_bids_basename(prefix=bids_root,
                                                suffix='participants.tsv')
    if not op.exists(participants_tsv_fpath):
        return dict()

    participants_tsv = _from_tsv(str(participants_tsv_fpath))
    n_subjects = len(participants_tsv['participant_id'])

    # summarize sex count statistics
    n_males, n_females, n_sex_unknown = 0, 0, 0
    keys = {'M': n_males, 'F': n_females, 'n/a': n_sex_unknown}
    for key in keys:
        n_sex_ct = len([sex for sex in participants_tsv['sex']
                        if sex == key])
        if key == 'M':
            n_males = n_sex_ct
        elif key == 'F':
            n_females = n_sex_ct
        else:
            n_sex_unknown = n_sex_ct

    # summarize hand count statistics
    n_rhand, n_lhand, n_ambidex = 0, 0, 0
    n_hand_unknown = 0
    keys = {'R': n_rhand, 'L': n_lhand,
            'A': n_ambidex, 'n/a': n_hand_unknown}
    for key, str_val in keys.items():
        n_hand_ct = len([hand for hand in participants_tsv['hand']
                         if hand == key])
        if key == 'R':
            n_rhand = n_hand_ct
        elif key == 'L':
            n_lhand = n_hand_ct
        elif key == 'A':
            n_ambidex = n_hand_ct
        else:
            n_hand_unknown = n_hand_ct

    # summarize age statistics: mean, std, min, max
    age_list = [float(age) for age in participants_tsv['age'] if age != 'n/a']
    n_age_unknown = n_subjects - len(age_list)
    if age_list:
        min_age, max_age = np.min(age_list), np.max(age_list)
        mean_age, std_age = np.mean(age_list), np.std(age_list)
    else:
        min_age, max_age, mean_age, std_age = 'n/a', 'n/a', 'n/a', 'n/a'

    template_dict = {
        'n_subjects': n_subjects,
        'n_males': n_males,
        'n_females': n_females,
        'n_rhand': n_rhand,
        'n_lhand': n_lhand,
        'n_ambidex': n_ambidex,
        'n_hand_unknown': n_hand_unknown,
        'n_sex_unknown': n_sex_unknown,
        'n_age_unknown': n_age_unknown,
        'mean_age': mean_age,
        'std_age': std_age,
        'min_age': min_age,
        'max_age': max_age,
    }
    return template_dict


def _summarize_scans(bids_root, session=None, kind=None, verbose=True):
    """Summarize scans in BIDS root directory.

    Parameters
    ----------
    bids_root : str | pathlib.Path
        The path of the root of the BIDS compatible folder.
    session : str, optional
    kind : str, optional
    verbose : bool

    Returns
    -------
    template_dict : dict
    """
    bids_root = Path(bids_root)
    if session is None:
        search_str = '*_scans.tsv'
    else:
        search_str = f'*ses-{session}*_scans.tsv'
    scans_fpaths = list(bids_root.rglob(search_str))

    if kind is not None:
        scans_fpaths = [fpath for fpath in scans_fpaths
                        if not fpath.as_posix().endswith(f'*_{kind}.*')]
    if len(scans_fpaths) == 0:
        return dict()

    # keep track of number of files, length and sfreq
    n_scans = len(scans_fpaths)
    template_dict = {
        'n_scans': n_scans,
    }

    # summarize sidecar.json, channels.tsv template
    sidecar_dict = _summarize_sidecar_json(bids_root, scans_fpaths,
                                           verbose=verbose)
    channels_dict = _summarize_channels_tsv(bids_root, scans_fpaths,
                                            verbose=verbose)
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
    verbose : bool

    Returns
    -------
    template_dict : dict
    """
    n_ecog_chs, n_seeg_chs, n_eeg_chs = 0, 0, 0
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
            bids_basename, _ = _parse_ext(scan)
            kind = op.dirname(scan)
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
            sidecar_fname = _find_matching_sidecar(bids_basename,
                                                   bids_root,
                                                   suffix=f'{kind}.json')
            with open(sidecar_fname, 'r') as fin:
                sidecar_json = json.load(fin)

            # aggregate metadata from each scan
            sfreq = sidecar_json['SamplingFrequency']
            powerlinefreq = str(sidecar_json['PowerLineFrequency'])
            manufacturer = sidecar_json['Manufacturer']
            n_eeg_chs += sidecar_json.get('EEGChannelCount', 0)
            n_ecog_chs += sidecar_json.get('ECOGChannelCount', 0)
            n_seeg_chs += sidecar_json.get('SEEGChannelCount', 0)

            software_filters = sidecar_json.get('SoftwareFilters')
            sfreqs.add(sfreq)
            powerlinefreqs.add(powerlinefreq)
            manufacturers.add(manufacturer)
            length_recordings.append(sidecar_json['RecordingDuration'])

    # length summary
    min_record_length = min(length_recordings)
    max_record_length = max(length_recordings)
    mean_record_length = np.mean(length_recordings)
    std_record_length = np.std(length_recordings)

    template_dict = {
        'manufacturer': ','.join(manufacturers),
        'sfreq': ','.join(map(str, sfreqs)),
        'powerlinefreq': ','.join(map(str, powerlinefreqs)),
        'software_filters': software_filters,
        'n_ecog_chs': n_ecog_chs,
        'n_seeg_chs': n_seeg_chs,
        'n_eeg_chs': n_eeg_chs,
        'min_record_length': min_record_length,
        'max_record_length': max_record_length,
        'mean_record_length': mean_record_length,
        'std_record_length': std_record_length,
        'total_record_length': np.sum(length_recordings)
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
    verbose : bool

    Returns
    -------
    template_dict : dict
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
            if not scan.startswith('ieeg'):
                continue

            # summarize metadata of recordings
            bids_basename, _ = _parse_ext(scan)
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
    verbose : bool

    Returns
    -------
    paragraph : str
    """
    # high level summary
    sessions = get_entity_vals(bids_root, entity_key='ses')
    kinds = get_kinds(bids_root)

    # dataset_description.json summary
    dataset_template = _summarize_dataset(bids_root)

    # participants summary
    participants_template = _summarize_subs(bids_root)

    # scans summary
    scans_template = _summarize_scans(bids_root, session=session,
                                      verbose=verbose)
    scans_template.update({'system': ','.join(kinds),
                           'n_sessions': len(sessions),
                           'sessions': ','.join(sessions)})

    paragraph = \
        BIDS_DATASET_TEMPLATE.format(**dataset_template) + \
        PARTICIPANTS_TEMPLATE.format(**participants_template) + \
        MODALITY_AGNOSTIC_TEMPLATE.format(**scans_template)

    if 'ieeg' in kinds:
        paragraph = paragraph + IEEG_TEMPLATE.format(**scans_template)

    return paragraph


if __name__ == '__main__':
    import textwrap

    bids_root = fetch_brainvision_testing_data()
    bids_root = "/Users/adam2392/Downloads/tngpipeline"

    methods_paragraph = create_methods_paragraph(bids_root)
    print('\n'.join(textwrap.wrap(methods_paragraph, width=50)))
