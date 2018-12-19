import numpy as np
from collections import OrderedDict


def _from_tsv(fname, dtypes=None):
    """Read a tsv file into an OrderedDict.

    Parameters
    ----------
    fname : str
        Path to the file being loaded.
    dtypes : list
        List of types to cast the values loaded as. This is specified column by
        column.
        Defaults to None. In this case all the data is loaded as strings.

    Returns
    -------
    data : collections.OrderedDict
        Keys are the column names, and values are the column data.
    """
    data = np.loadtxt(fname, dtype=str, delimiter='\t')
    column_names = data[0, :]
    info = data[1:, :]
    data_dict = OrderedDict()
    if dtypes is None:
        dtypes = [str] * info.shape[1]
    if not isinstance(dtypes, (list, tuple)):
        dtypes = [dtypes] * info.shape[1]
    if not len(dtypes) == info.shape[1]:
        raise ValueError('dtypes length mismatch. Provided: {0}, '
                         'Expected: {1}'.format(len(dtypes), info.shape[1]))
    for i, name in enumerate(column_names):
        data_dict[name] = info[:, i].astype(dtypes[i]).tolist()
    return data_dict


def _to_tsv(data, fname):
    """Write an OrderedDict into a tsv file.

    Parameters
    ----------
    data : collections.OrderedDict
        Ordered dictionary containing data to be written to a tsv file.
    fname : str
        Path to the file being written.
    """
    n_rows = len(data[list(data.keys())[0]])
    output = _tsv_to_str(data, n_rows)

    with open(fname, 'w') as f:
        f.write(output)


def _combine(data1, data2, drop_columns=None):
    """Add two OrderedDict's together and optionally drop repeated data.

    Parameters
    ----------
    data1 : collections.OrderedDict
        Original OrderedDict.
    data2 : collections.OrderedDict
        New OrderedDict to be added to the original.
    drop_columns : str | list of str's
        Name(s) of the column to check for duplicate values in.
        Any duplicates found will be dropped from the original data array (ie.
        most recent value are kept).
        If a list is provided, the rows will be dropped by checking equality
        between values in the column names provided in order.

    Returns
    -------
    data : collections.OrderedDict
        The new combined data.
    """
    data = data1.copy()
    for key, value in data2.items():
        data[key].extend(value)

    # check that all the columns have the same number of values, filling any
    # columns without the required amount with 'n/a'.
    max_rows = max([len(column) for column in data.values()])
    for key, value in data.items():
        if len(value) != max_rows:
            data[key].extend(['n/a'] * (max_rows - len(value)))

    if drop_columns is None:
        return data

    if isinstance(drop_columns, str):
        drop_columns = [drop_columns]
    idxs = []
    for drop_column in drop_columns:
        # for each column we wish to drop values from, find any repeated values
        # and remove all but the most recent version of
        n_rows = len(data[drop_column])
        _, _idxs = np.unique(data[drop_column][::-1], return_index=True)
        if idxs == []:
            idxs = _idxs.tolist()
        else:
            for idx in set(idxs) - set(_idxs):
                idxs.remove(idx)
    for key in data:
        data[key] = [data[key][n_rows - 1 - idx] for idx in idxs]

    return data


def _drop(data, values, column):
    """Remove rows from the OrderedDict.

    Parameters
    ----------
    data : collections.OrderedDict
        Data to drop values from.
    values : list
        List of values to drop. Any row containing this value in the specified
        column will be dropped.
    column : string
        Name of the column to check for the existence of `value` in.

    Returns
    -------
    data : collections.OrderedDict
        Copy of the original data with 0 or more rows dropped.
    """
    new_data = data.copy()
    mask = np.in1d(new_data[column], values, invert=True)
    for key in new_data.keys():
        new_data[key] = np.array(new_data[key])[mask].tolist()
    return new_data


def _contains_row(data, row_data):
    """Whether the specified row data exists in the OrderedDict.

    Parameters
    ----------
    data : collections.OrderedDict
        OrderedDict to check.
    row_data : list
        List of values to be searched for. This must match the contents of a
        row exactly for a positive match to be returned.
    """
    mask = None
    for idx, column_data in enumerate(data.values()):
        column_mask = np.in1d(np.array(column_data), row_data[idx])
        mask = column_mask if mask is None else (mask & column_mask)
    return np.any(mask)


def _tsv_to_str(data, rows=5):
    """Return a string representation of the OrderedDict.

    Parameters
    ----------
    data : collections.OrderedDict
        OrderedDict to return string representation of.
    rows : int
        Maximum number of rows of data to output.
    """
    col_names = list(data.keys())
    n_rows = len(data[col_names[0]])
    output = list()
    # write headings.
    output.append('\t'.join(col_names))

    # write column data.
    max_rows = min(n_rows, rows)
    for idx in range(max_rows):
        row_data = list(str(data[key][idx]) for key in data)
        output.append('\t'.join(row_data))

    return '\n'.join(output)
