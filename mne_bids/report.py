"""Make BIDS report from dataset and sidecar files."""
# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)
import collections
import json
from pathlib import Path
import os
from pprint import pprint

from mne_bids import make_bids_basename
from mne_bids.utils import get_kinds, get_entity_vals, _find_matching_sidecar
from mne_bids.tsv_handler import _from_tsv

PARTICIPANTS_TEMPLATE = "The dataset consists of {n_patients} patients " \
                        "({patient_count_summary})."
SCAN_TEMPLATE = "The session {session} consists of {n_scans} " \
                "with {scan_summary}."
DATASET_TEMPLATE = "The data was acquired using a {{system}} system ({{manufacturer}}) " \
                   "consisting of {{channel_summary}}."


def _create_subject_tree(bids_root, verbose=True):
    # create a tree for the patient summaries in the form of a nested dictionary
    patient_summaries = dict()

    # loop through each subject
    subjects = get_entity_vals(bids_root, entity_key='sub')

    for subject in subjects:
        ignore_sub = [sub for sub in subjects if sub != subject]
        sessions = get_entity_vals(bids_root, entity_key='ses', ignore_sub=ignore_sub)
        tasks = get_entity_vals(bids_root, entity_key='task', ignore_sub=ignore_sub)
        acquisitions = get_entity_vals(bids_root, entity_key='acq', ignore_sub=ignore_sub)

        # if subject not in patient_summaries:
        #     patient_summaries[subject] = dict()
        # patient_summaries[subject]['ses'] = sessions

        for session in sessions:
            # if session not in patient_summaries[subject]['ses']:
            scans_fpath = make_bids_basename(subject, session, suffix='scans.tsv')
            scans_tsv = _from_tsv(scans_fpath, verbose)
            scans = scans_tsv['filename']

            # for scan in scans:

            # patient_summaries[subject][session]['task'] = tasks
            # patient_summaries[subject][session]['acq'] = acquisitions
            # patient_summaries[subject][session]['scans'] = scans

def create_methods_paragraph(bids_root, verbose=True):
    # summarizing statistics
    n_kinds = 0
    kinds = get_kinds(bids_root)
    n_kinds += len(kinds)
    n_sessions = 0
    n_patients = 0



    # loop through each subject
    subjects = get_entity_vals(bids_root, entity_key='sub')
    sessions = get_entity_vals(bids_root, entity_key='ses')

    for subject in subjects:
        ignore_sub = [sub for sub in subjects if sub != subject]
        sessions = get_entity_vals(bids_root, entity_key='ses', ignore_sub=ignore_sub)
        tasks = get_entity_vals(bids_root, entity_key='task', ignore_sub=ignore_sub)
        acquisitions = get_entity_vals(bids_root, entity_key='acq', ignore_sub=ignore_sub)

        patient_summaries[subject]['ses'] = sessions

        for session in sessions:
            scans_fpath = make_bids_basename(subject, session, suffix='scans.tsv')
            scans_tsv = _from_tsv(scans_fpath, verbose)
            scans = scans_tsv['filename']

            patient_summaries[subject][session]['task'] = tasks
            patient_summaries[subject][session]['acq'] = acquisitions
            patient_summaries[subject][session]['scans'] = scans

    kind_summary = ', '.join(kinds)
    PARTICIPANTS_TEMPLATE = f"The dataset consists of {n_patients} patients " \
                            f"with {n_sessions} sessions consisting of {kind_summary} data."

    paragraph = PARTICIPANTS_TEMPLATE
    return paragraph

def _scan_summary(bids_basename, bids_root, kind):
    # XXX: to improve with separate summary dictionary for each kind
    scan_dict = dict()

    sidecar_fpath = _find_matching_sidecar(bids_basename, bids_root,
                                           suffix=f'{kind}.json')
    with open(sidecar_fpath, 'r') as fin:
        sidecarjson = json.load(fin)

    scan_dict['RecordingDuration'] = sidecarjson['RecordingDuration']
    scan_dict['PowerLineFrequeency'] = sidecarjson['PowerLineFrequency']
    scan_dict['SamplingFrequency'] = sidecarjson['SamplingFrequency']
    scan_dict['Manufacturer'] = sidecarjson['Manufacturer']

    return scan_dict