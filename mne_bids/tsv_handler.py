"""Private functions to handle tabular data."""
from collections import OrderedDict
from copy import deepcopy

from mne.utils import warn
import numpy as np


def _combine_rows(data1, data2, drop_column=None):
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
    # next extend the values in data1 with values in data2
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
    for key, row_value in row_data.items():
        # if any of the columns don't even exist in the keys
        # this data_value will return False
        data_value = np.array(data.get(key))

        # Cast row_value to the same dtype as data_value to avoid a NumPy
        # FutureWarning, see
        # https://github.com/mne-tools/mne-bids/pull/372
        row_value = np.array(row_value, dtype=data_value.dtype)

        column_mask = np.in1d(data_value, row_value)
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
    new_data_col = np.array(new_data[column])

    # Cast `values` to the same dtype as `new_data_col` to avoid a NumPy
    # FutureWarning, see
    # https://github.com/mne-tools/mne-bids/pull/372
    dtype = new_data_col.dtype
    if new_data_col.shape == (0,):
        dtype = np.array(values).dtype
    values = np.array(values, dtype=dtype)

    mask = np.in1d(new_data_col, values, invert=True)
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
    data = np.loadtxt(fname, dtype=str, delimiter='\t', ndmin=2,
                      comments=None, encoding='utf-8-sig')
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
    empty_cols = 0
    for i, name in enumerate(column_names):
        values = info[:, i].astype(dtypes[i]).tolist()
        data_dict[name] = values
        if len(values) == 0:
            empty_cols += 1

    if empty_cols == len(column_names):
        warn(f"TSV file is empty: '{fname}'")

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

    with open(fname, 'w', encoding='utf-8-sig') as f:
        f.write(output)
        f.write('\n')


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
