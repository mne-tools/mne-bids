import numpy as np
from collections import OrderedDict, Counter


def from_tsv(fname, dtypes=None):
    """Read a tsv file into an OrderedDict

    Parameters
    ----------
    fname : str
        Path to the file being loaded.
    dtypes : list
        List of types to case the values loaded as. This is specified column by
        column.
        Defaults to None. In this case all the data is loaded as strings."""
    data = np.loadtxt(fname, dtype=str, delimiter='\t')
    column_names = data[0, :]
    info = data[1:, :]
    data_dict = OrderedDict()
    if isinstance(dtypes, (list, tuple)):
        if len(dtypes) == info.shape[1]:
            for i, name in enumerate(column_names):
                data_dict[name] = info[:, i].astype(dtypes[i]).tolist()
    else:
        for i, name in enumerate(column_names):
            data_dict[name] = info[:, i].tolist()
    return data_dict


def to_tsv(data, fname):
    """ Write an OrderedDict into a tsv file """
    rows = len(data[list(data.keys())[0]])
    with open(fname, 'w') as f:
        f.write('\t'.join(list(data.keys())))
        f.write('\n')
        for i in range(rows):
            row_data = list(str(data[key][i]) for key in data)
            f.write('\t'.join(row_data))
            f.write('\n')


def combine(data1, data2, drop_column=None):
    """Add two OrderedDict's together with the option of dropping repeated data

    Parameters
    ----------
    data : numpy.ndarray
        Original array of data.
    other : numpy.ndarray
        New array of data.
    drop_column : str
        Name of the column to check for duplicate values in.
        Any duplicates found will be dropped from the original data array (ie.
        most recent value are kept).

    """
    for key, value in data2.items():
        data1[key].extend(value)
    if drop_column in data1:
        column_data = data1[drop_column]
        values_count = Counter(column_data)
        for key, value in values_count.items():
            for _ in range(value - 1):
                idx = data1[drop_column].index(key)
                for column in data1.values():
                    del column[idx]


def drop(data, values, column):
    """Remove specified values in a column. This occurs in-place.

    Parameters
    ----------
    data : collections.OrderedDict
        Data to drop values from.
    values : list
        List of values to drop. Any row containing this value in the specified
        column will be dropped.
    column : string
        Name of the column to check for the existence of `value` in.

    """
    for value in values:
        if value in data[column]:
            idx = data[column].index(value)
            for column_data in data.values():
                del column_data[idx]


def contains_row(data, row_data):
    """Whether the specified row data exists in the OrderedDict """
    first_column = list(data.keys())[0]
    potential_indexes = list()
    for i, value in enumerate(data[first_column]):
        if value == row_data[0]:
            potential_indexes.append(i)
    for i in potential_indexes:
        contained_row_data = list(data[key][i] for key in data)
        if row_data == contained_row_data:
            return True
    return False


def prettyprint(data, rows=5):
    """pretty print an ordered dictionary """
    data_rows = len(data[list(data.keys())[0]])
    out_str = ''
    out_str += '\t'.join(list(data.keys())) + '\n'
    for i in range(min(data_rows, rows)):
        row_data = list(str(data[key][i]) for key in data)
        out_str += '\t'.join(row_data) + '\n'
    return out_str
