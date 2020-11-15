"""Some functions to extract stats from a BIDS dataset."""

# Authors: Alex Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)


from mne_bids import BIDSPath


def count_events(root, subject=None, session=None, task=None, run=None):
    """Count events

    Parameters
    ----------
    root : str | pathlib.Path | None
        The root folder of the BIDS dataset.
    subject : str | None
        The subject ID to consider. If None all subjects are included.
    session : str | None
        The session to consider. If None all sessions are included.
    task : str | None
        The task to consider. If None all tasks are included.
    run : int | None
        The run number to consider. If None all runs are included.

    Returns
    -------
    counts : pd.DataFrame
        The pandas dataframe containing all the counts of trial_type
        in all matching events.tsv files.
    """
    import pandas as pd
    bids_path = BIDSPath(
        root=root, suffix='events', extension='tsv', subject=subject,
        session=session, task=task, run=run
    )

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

        if len(all_df) == 0:
            continue

        df = pd.concat(all_df)
        groups = ['subject']
        if bp.session is not None:
            groups.append('session')
        if bp.run is not None:
            groups.append('run')

        groups.append('trial_type')
        counts = df.groupby(groups).size()
        counts = counts.unstack()

        if 'BAD_ACQ_SKIP' in counts.columns:
            counts = counts.drop('BAD_ACQ_SKIP', axis=1)

        counts.columns = pd.MultiIndex.from_arrays(
            [[task] * counts.shape[1], counts.columns]
        )

        all_counts.append(counts)

    if len(all_counts) == 0:
        raise ValueError('No events files found.')

    counts = pd.concat(all_counts, axis=1)

    return counts
