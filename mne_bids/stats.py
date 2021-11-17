"""Some functions to extract stats from a BIDS dataset."""

# Authors: Alex Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause


from mne_bids import BIDSPath, get_datatypes
from mne_bids.config import EPHY_ALLOWED_DATATYPES


def count_events(root_or_path, datatype='auto'):
    """Count events present in dataset.

    Parameters
    ----------
    root_or_path : path-like | mne_bids.BIDSPath
        If str or Path it is the root folder of the BIDS dataset.
        If a BIDSPath is passed it allows to limit the count
        to a subject, a session or a run by only considering
        the event files that match this BIDSPath.
    datatype : str
        Type of the data recording. Can be ``meg``, ``eeg``,
        ``ieeg`` or ``auto``. If ``auto`` and a :class:`mne_bids.BIDSPath`
        isinstance is passed as ``root_or_path`` which has a ``datatype``
        attribute set, then this data type will be used. Otherwise, only
        one data type should be present in the dataset to avoid any
        ambiguity.

    Returns
    -------
    counts : pandas.DataFrame
        The pandas dataframe containing all the counts of trial_type
        in all matching events.tsv files.
    """
    import pandas as pd

    if not isinstance(root_or_path, BIDSPath):
        bids_path = BIDSPath(root=root_or_path)
    else:
        bids_path = root_or_path.copy()

    bids_path.update(suffix='events', extension='tsv')

    datatypes = get_datatypes(bids_path.root)
    this_datatypes = list(set(datatypes).intersection(EPHY_ALLOWED_DATATYPES))

    if (datatype == 'auto') and (bids_path.datatype is not None):
        datatype = bids_path.datatype

    if datatype == 'auto':
        if len(this_datatypes) > 1:
            raise ValueError(f'Multiple datatypes present ({this_datatypes}).'
                             f' You need to specity datatype got: {datatype})')
        elif len(this_datatypes) == 0:
            raise ValueError('No valid datatype present.')

        datatype = this_datatypes[0]

    if datatype not in EPHY_ALLOWED_DATATYPES:
        raise ValueError(f'datatype ({datatype}) is not supported. '
                         f'It must be one of: {EPHY_ALLOWED_DATATYPES})')

    bids_path.update(datatype=datatype)

    tasks = sorted(set([bp.task for bp in bids_path.match()]))

    all_counts = []

    for task in tasks:
        bids_path.update(task=task)

        all_df = []
        for bp in bids_path.match():
            df = pd.read_csv(str(bp), delimiter='\t')
            df['subject'] = bp.subject
            if bp.session is not None:
                df['session'] = bp.session
            if bp.run is not None:
                df['run'] = bp.run
            all_df.append(df)

        if not all_df:
            continue

        df = pd.concat(all_df)
        groups = ['subject']
        if bp.session is not None:
            groups.append('session')
        if bp.run is not None:
            groups.append('run')

        if 'stim_type' in df.columns:
            # Deal with some old files that use stim_type rather than
            # trial_type
            df = df.rename(columns={"stim_type": "trial_type"})

        # There are datasets out there without a `trial_type` or `stim_type`
        # column.
        if 'trial_type' in df.columns:
            groups.append('trial_type')

        counts = df.groupby(groups).size()
        counts = counts.unstack()

        if 'BAD_ACQ_SKIP' in counts.columns:
            counts = counts.drop('BAD_ACQ_SKIP', axis=1)

        counts.columns = pd.MultiIndex.from_arrays(
            [[task] * counts.shape[1], counts.columns]
        )

        all_counts.append(counts)

    if not all_counts:
        raise ValueError('No events files found.')

    counts = pd.concat(all_counts, axis=1)

    return counts
