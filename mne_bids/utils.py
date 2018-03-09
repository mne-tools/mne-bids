import os
import errno
from collections import OrderedDict

def _mkdir_p(path):
    """Create a directory, making parent directories as needed.
    Copied from
    stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def filename_bids(subject=None, session=None, task=None,
                  acquisition=None, run=None, processing=None,
                  recording=None, suffix=None):
    """Create a BIDS filename from its component parts.

    BIDS filename prefixes have one or more pieces of metadata in them. They
    must follow a particular order, which is followed by this function. This
    will generate the *prefix* for a BIDS file name that can be used with many
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
    run : str | None
        The run with a task for this item. Corresponds to "run".
    processing : str | None
        The processing label for this item. Corresponds to "proc".
    recording : str | None
        The recording name for this item. Corresponds to "recording".
    suffix : str | None
        The suffix of a file that begins with this prefix. E.g., 'audio.wav'.

    Returns
    -------
    filename : str
        The BIDS filename you wish to create.
    """
    order = OrderedDict([('sub', subject),
                         ('ses', session),
                         ('task', task),
                         ('acq', acquisition),
                         ('run', run),
                         ('proc', processing),
                         ('recording', recording)])
    _check_types(order.values())

    if not any(isinstance(ii, str) for ii in order.keys()):
        raise ValueError("At least one parameter must be given.")
    filename = []
    for key, val in order.items():
        if val is not None:
            filename.append('%s-%s' % (key, val))
    if isinstance(suffix, str):
        filename.append(suffix)
    filename = '_'.join(filename)
    return filename


def create_folders(subject, session=None, kind=None, root=None, create=True):
    """Create a BIDS folder hierarchy.

    This creates a hierarchy of folders *within* a BIDS dataset. You should
    plan to create these folders *inside* the root folder of the dataset.

    Parameters
    ----------
    subject : str
        The subject ID. Corresponds to "sub".
    kind : str
        The kind of folder being created at the end of the hierarchy. E.g.,
        "anat", "func", etc.
    session : str | None
        The session for a item. Corresponds to "ses".
    root : str | None
        The root for the folders to be created. If None, folders will be
        created in the current working directory.
    create : bool
        Whether to create the folders specified.

    Returns
    -------
    path : str
        The (relative) path to the folder that was created.
    """
    _check_types((subject, kind, session, root))

    path = ['sub-%s' % subject]
    if isinstance(session, str):
        path.append('ses-%s' % session)
    if isinstance(kind, str):
        path.append(kind)
    path = os.path.join(*path)
    if isinstance(root, str):
        path = os.path.join(root, path)

    if create is True:
        _mkdir_p(path)
    return path


def _check_types(variables):
    """Make sure all variables are strings or None."""
    types = set(type(ii) for ii in variables)
    for itype in types:
        if not isinstance(itype, type(str)) and itype is not None:
            raise ValueError("All values must be either None or strings. "
                             "Found type %s." % itype)
