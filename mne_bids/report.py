"""Make BIDS report from dataset and sidecar files."""
# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)
import collections
import json
import os
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


def _summarize_dataset(bids_root) -> str:
    """Summarize the dataset_desecription.json file.

    Parameters
    ----------
    bids_root : str | pathlib.Path

    Returns
    -------
    template : str
    """
    dataset_descrip_fpath = make_bids_basename(
        prefix=bids_root, suffix='dataset_description.json')
    if not op.exists(dataset_descrip_fpath):
        return ''

    # read file and 'REQUIRED' components of it
    with open(dataset_descrip_fpath, 'r') as fin:
        dataset_description = json.load(fin)

    name = dataset_description['Name']
    bids_version = dataset_description['BIDSVersion']

    template = f'Dataset {name} was created ' \
               f'with BIDS version {bids_version} using ' \
               f'MNE-BIDS ({DOI}).'
    return template


def _summarize_subs(bids_root) -> str:
    """Summarize subjects in BIDS root directory.

    Parameters
    ----------
    bids_root : str | pathlib.Path

    Returns
    -------
    template : str
    """
    participants_tsv_fpath = make_bids_basename(prefix=bids_root,
                                                suffix='participants.tsv')
    if not op.exists(participants_tsv_fpath):
        return ''

    participants_tsv = _from_tsv(participants_tsv_fpath)
    n_subjects = len(participants_tsv['participant_id'])

    # summarize sex
    sex_str = ""
    count = 0
    keys = {'M': 'male', 'F': 'female', 'n/a': 'unknown sex'}
    for key, str_val in keys.items():
        n_sex_ct = len([sex for sex in participants_tsv['sex'] if sex == key])
        if n_sex_ct > 0:
            sex_str = sex_str + f'{n_sex_ct} {str_val} '
            count += 1

    # summarize hand
    hand_str = ""
    count = 0
    keys = {'R': 'right', 'L': 'left',
            'A': 'ambidextrous', 'n/a': 'unknown hand'}
    for key, str_val in keys.items():
        n_hand_ct = len([hand for hand in participants_tsv['hand']
                         if hand == key])
        if n_hand_ct > 0:
            hand_str = hand_str + f'{n_hand_ct} {str_val} '
            count += 1

    # summarize age
    age_list = [float(age) for age in participants_tsv['age'] if age != 'n/a']
    n_unknown = n_subjects - len(age_list)
    age_str = 'ages '
    if age_list:
        avg_age, std_age = np.mean(age_list), np.std(age_list)
        age_str = f'{avg_age} +/- {std_age}'
    if n_unknown:
        age_str = age_str + f' with {n_unknown} unknown age'

    template = f'The dataset consists of {n_subjects} subjects ' \
               f'({age_str}; {hand_str}; {sex_str}).'
    return template


def _summarize_scans(bids_root, session=None, kind=None) -> str:
    """Summarize scans in BIDS root directory.

    Parameters
    ----------
    bids_root : str | pathlib.Path
    session : str, optional
    kind : str, optional

    Returns
    -------
    template : str
    """
    bids_root = Path(bids_root)
    if session is None:
        search_str = '*_scans.tsv'
    else:
        search_str = f'*ses-{session}*_scans.tsv'
    scans_fpaths = list(bids_root.rglob(search_str))

    if len(scans_fpaths) == 0:
        return ''

    # keep track of number of files, length and sfreq
    n_files = 0
    length_recordings = []
    sampling_frequency = collections.defaultdict(int)
    powerline_frequency = dict()

    # loop through each scan
    for scan_fpath in scans_fpaths:
        # get the current session directory
        ses_path = os.path.dirname(scan_fpath)

        # load in the scans.tsv file
        # and read metadata for each scan
        scans_tsv = _from_tsv(scan_fpath)
        scans = scans_tsv['filename']
        for scan in scans:
            if kind is not None:
                if not scan.startswith(kind):
                    continue

            # count each scan
            n_files += 1

            # summarize metadata of recordings
            bids_fname, _ = _parse_ext(scan)
            sidecar_fname = os.path.join(ses_path, bids_fname + '.json')
            with open(sidecar_fname, 'r') as fin:
                sidecar_json = json.load(fin)

            # aggregate metadata from each scan
            length_recordings.append(sidecar_json['RecordingDuration'])
            sfreq = sidecar_json['SamplingFrequency']
            powerlinefreq = str(sidecar_json['PowerLineFrequency'])
            powerline_frequency[powerlinefreq] = 1
            sampling_frequency[sfreq] += 1

    # length summary
    avg_length = np.mean(length_recordings)
    std_length = np.std(length_recordings)
    length_summary = f'{avg_length:.2f} +/- {std_length:.2f} seconds'

    # sampling frequency summary
    sfreq_summary = ''
    for idx, (sfreq, count) in enumerate(sampling_frequency.items()):
        if sfreq_summary == '':
            sfreq_summary = 'with sampling rates '
        if idx > 0:
            sfreq_summary = sfreq_summary + ', '
        sfreq_summary = sfreq_summary + f'{sfreq} (n={count})'

    # power line frequency summary
    powerlinefreq_summary = ','.join(powerline_frequency.keys())

    # report template
    template = f'There are {n_files} datasets ({length_summary}) ' \
               f'{sfreq_summary} with ' \
               f'power line frequency {powerlinefreq_summary}.'
    return template


def _summarize_ieeg(bids_root, session=None, verbose=True) -> str:
    """Summarize ieeg data in BIDS root directory.

    Currently, summarizes all REQUIRED components of iEEG
    data, and some RECOMMENDED and OPTIONAL components.

    Parameters
    ----------
    bids_root : str | pathlib.Path
    session : str, optional

    Returns
    -------
    template : str
    """
    bids_root = Path(bids_root)
    if session is None:
        search_str = '*_scans.tsv'
    else:
        search_str = f'*ses-{session}*_scans.tsv'
    scans_fpaths = list(bids_root.rglob(search_str))

    if len(scans_fpaths) == 0:
        return ''

    # keep track of channel type, status
    ch_type_count = collections.defaultdict(int)
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

            sidecar_fname = _find_matching_sidecar(bids_fname=bids_basename,
                                                   bids_root=bids_root,
                                                   suffix='ieeg.json')
            channels_fname = _find_matching_sidecar(bids_fname=bids_basename,
                                                    bids_root=bids_root,
                                                    suffix='channels.tsv')

            # summarize sidecar json
            with open(sidecar_fname, 'r') as fin:
                sidecar_json = json.load(fin)
            # aggregate metadata from each scan
            for key in ['SEEGChannelCount', 'ECOGChannelCount']:
                ch_type_count[key] += sidecar_json[key]

            # summarize channels.tsv
            channels_tsv = _from_tsv(channels_fname)
            for status in channels_tsv['status']:
                ch_status_count[status] += 1

    # create summary template strings for type
    ch_type_summary = f','.join([f'{key} (n={value})'
                                 for key, value in ch_type_count.items()])

    # create summary template strings for status
    n_good = ch_status_count['good']
    n_bad = ch_status_count['bad']
    ch_status_summary = f'used {n_good} channels and ' \
                        f'removed {n_bad} channels from analysis.'

    template = f'There were {ch_type_summary} ({ch_status_summary}).'
    return template


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
    subjects = get_entity_vals(bids_root, entity_key='sub')
    sessions = get_entity_vals(bids_root, entity_key='ses')
    kinds = get_kinds(bids_root)

    # summarizing statistics
    n_patients, n_sessions = len(subjects), len(sessions)
    n_kinds = len(kinds)

    # template string for high level summary
    session_summary = ', '.join(sessions)
    kind_summary = ', '.join(kinds)
    template = f"The dataset consists of {n_patients} patients " \
               f"with {n_sessions} sessions ({session_summary}) " \
               f"consisting of {n_kinds} kinds of data ({kind_summary})."

    # dataset_description.json summary
    dataset_template = _summarize_dataset(bids_root)

    # subject and scans summary
    sub_template = _summarize_subs(bids_root)
    scan_template = _summarize_scans(bids_root, session=session)

    paragraph = f'{dataset_template} {template} {sub_template} {scan_template}'

    if 'ieeg' in kinds:
        ieeg_summary = _summarize_ieeg(bids_root, session=session)
        paragraph = paragraph + f' {ieeg_summary}'

    return paragraph


if __name__ == '__main__':
    bids_root = fetch_brainvision_testing_data()
    bids_root = "/Users/adam2392/Downloads/tngpipeline"

    methods_paragraph = create_methods_paragraph(bids_root)
    print(methods_paragraph)
