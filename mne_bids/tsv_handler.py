"""Private functions to handle tabular data."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

import gzip
import json
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
from mne.utils import _validate_type

from mne_bids._fileio import _open_lock


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
        if data_value.size > 0:
            row_value = np.array(row_value, dtype=data_value.dtype)
        column_mask = np.isin(data_value, row_value)
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

    mask = np.isin(new_data_col, values, invert=True)
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
        For gzipped files (``*.gz``), note there are no in-file header names;
        generated keys ``column_0``, ``column_1``, ... will be used.

    Returns
    -------
    data_dict : collections.OrderedDict
        Keys are the column names, and values are the column data.

    """
    from .utils import warn  # avoid circular import

    fname = Path(fname)
    compressed = fname.suffix == ".gz"

    data = np.loadtxt(
        fname, dtype=str, delimiter="\t", ndmin=2, comments=None, encoding="utf-8-sig"
    )
    # Handle empty files - data may be empty or only have a header
    if data.size == 0:
        warn(f"TSV file is empty: '{fname}'")
        return OrderedDict()

    # If data is 1-dimensional (only header), make it 2D
    data = np.atleast_2d(data)

    if compressed:
        # Compressed TSVs are headerless
        info = data
        column_names = [f"column_{i}" for i in range(info.shape[1])]
    else:
        # Cast to list to avoid `np.str_()` keys in dict
        column_names = data[0, :].tolist()
        info = data[1:, :]
    data_dict = OrderedDict()
    if dtypes is None:
        dtypes = [str] * info.shape[1]
    if not isinstance(dtypes, list | tuple):
        dtypes = [dtypes] * info.shape[1]
    if not len(dtypes) == info.shape[1]:
        raise ValueError(
            "dtypes length mismatch. "
            f"Provided: {len(dtypes)}, Expected: {info.shape[1]}"
        )
    empty_cols = 0
    for i, name in enumerate(column_names):
        values = info[:, i].astype(dtypes[i]).tolist()
        data_dict[name] = values
        if len(values) == 0:
            empty_cols += 1

    if empty_cols == len(column_names):
        warn(f"TSV file is empty: '{fname}'")

    return data_dict


def _from_compressed_tsv(fname, dtypes=None):
    """Wrap _from_tsv and then read column names from corresponding JSON."""
    fname = Path(fname)
    if fname.suffix != ".gz":
        raise ValueError(
            f"_from_compressed_tsv expects a .gz file, got '{fname.name}'."
        )

    data_dict = _from_tsv(fname=fname, dtypes=dtypes)

    sidecar_json = fname.with_suffix("").with_suffix(".json")
    if not sidecar_json.exists():
        raise ValueError(
            "To read a compressed tsv file, a corresponding sidecar JSON is needed. "
            f"searched for:\n {sidecar_json}"
        )

    sidecar = json.loads(sidecar_json.read_text(encoding="utf-8-sig"))
    columns = sidecar["Columns"]
    _validate_type(columns, list)

    if len(columns) != len(data_dict):
        raise ValueError(
            f"'{fname.name}' has {len(columns)} columns but '{sidecar_json.name}' only "
            f"provides names for {len(data_dict)} columns in its 'Columns' field."
        )

    renamed = dict()
    for idx, values in enumerate(data_dict.values()):
        renamed[columns[idx]] = values
    return renamed


def _to_tsv(data, fname, *, compress=False):
    """Write an OrderedDict into a tsv file.

    Parameters
    ----------
    data : collections.OrderedDict
        Ordered dictionary containing data to be written to a tsv file.
    fname : str
        Path to the file being written.
    """
    include_header = False if compress else True
    encoding = "utf-8-sig"

    n_rows = len(data[list(data.keys())[0]])
    output = _tsv_to_str(data, n_rows, include_header=include_header)
    output = f"{output}\n"

    if compress:
        # TODO: need to test that this works as expected during parallel write.
        with _open_lock(fname, "wb") as f:  # XXX: Would 'wt' mode work?
            f.write(gzip.compress(output.encode(encoding)))
    else:
        with _open_lock(fname, "w", encoding=encoding) as f:
            f.write(output)


def _tsv_to_str(data, rows=5, *, include_header=True):
    """Return a string representation of the OrderedDict.

    Parameters
    ----------
    data : collections.OrderedDict
        OrderedDict to return string representation of.
    rows : int, optional
        Maximum number of rows of data to output.
    include_header : bool
        Whether to include the column names in the TSV file. For writing gzipped text
        files, this should be False

    Returns
    -------
    str
        String representation of the first `rows` lines of `data`.

    """
    col_names = list(data.keys())
    n_rows = len(data[col_names[0]])
    output = list()
    # write headings.
    if include_header:
        output.append("\t".join(col_names))

    # write column data.
    max_rows = min(n_rows, rows)
    for idx in range(max_rows):
        row_data = list(str(data[key][idx]) for key in data)
        output.append("\t".join(row_data))

    return "\n".join(output)
