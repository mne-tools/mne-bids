"""Private functions to handle tabular data."""
import numpy as np
from collections import OrderedDict
from copy import deepcopy


def _combine(data1, data2, drop_column=None):
    """Add two OrderedDict's together and optionally drop repeated data.

    Parameters
    ----------
    data1 : collections.OrderedDict
        Original OrderedDict.
    data2 : collections.OrderedDict
        New OrderedDict to be added to the original.
    drop_column : str, optional
        Name of the column to check for duplicate values in.
        Any duplicates found will be dropped from the original data array (ie.
        most recent value are kept).

    Returns
    -------
    data : collections.OrderedDict
        The new combined data.

    """
    data = deepcopy(data1)
    for key, value in data2.items():
        data[key].extend(value)
    # Make sure that if there are any columns in data1 that didn't get new
    # data they are populated with "n/a"'s.
    for key in set(data1.keys()) - set(data2.keys()):
        data[key].extend(["n/a"] * len(next(iter(data2.values()))))

    if drop_column is None:
        return data

    # Find any repeated values and remove all but the most recent value.
    n_rows = len(data[drop_column])
    _, idxs = np.unique(data[drop_column][::-1], return_index=True)
    for key in data:
        data[key] = [data[key][n_rows - 1 - idx] for idx in idxs]

    return data


def _contains_row(data, row_data):
    """Determine whether the specified row data exists in the OrderedDict.

    Parameters
    ----------
    data : collections.OrderedDict
        OrderedDict to check.
    row_data : dict
        Dictionary with column names as keys, and values being the column value
        to match within a row.

    Returns
    -------
    bool
        True if `row_data` exists in `data`.

    Note
    ----
    This function will return True if the supplied `row_data` contains less
    columns than the number of columns in the existing data but there is still
    a match for the partial row data.

    """
    mask = None
    for key, value in row_data.items():
        column_mask = np.in1d(np.array(data[key]), value)
        mask = column_mask if mask is None else (mask & column_mask)
    return np.any(mask)


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
    new_data : collections.OrderedDict
        Copy of the original data with 0 or more rows dropped.

    """
    new_data = deepcopy(data)
    mask = np.in1d(new_data[column], values, invert=True)
    for key in new_data.keys():
        new_data[key] = np.array(new_data[key])[mask].tolist()
    return new_data


def _from_tsv(fname, dtypes=None):
    """Read a tsv file into an OrderedDict.

    Parameters
    ----------
    fname : str
        Path to the file being loaded.
    dtypes : list, optional
        List of types to cast the values loaded as. This is specified column by
        column.
        Defaults to None. In this case all the data is loaded as strings.

    Returns
    -------
    data_dict : collections.OrderedDict
        Keys are the column names, and values are the column data.

    """
    data = np.loadtxt(fname, dtype=str, delimiter='\t', encoding='utf-8')
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

    with open(fname, 'wb') as f:
        f.write(output.encode('utf-8'))


def _tsv_to_str(data, rows=5):
    """Return a string representation of the OrderedDict.

    Parameters
    ----------
    data : collections.OrderedDict
        OrderedDict to return string representation of.
    rows : int, optional
        Maximum number of rows of data to output.

    Returns
    -------
    str
        String representation of the first `rows` lines of `data`.

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
