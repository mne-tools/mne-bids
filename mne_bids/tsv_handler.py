import numpy as np
from collections import OrderedDict, Counter


def _from_tsv(fname, dtypes=None):
    """Read a tsv file into an OrderedDict

    Parameters
    ----------
    fname : str
        Path to the file being loaded.
    dtypes : list
        List of types to cast the values loaded as. This is specified column by
        column.
        Defaults to None. In this case all the data is loaded as strings.

    Returns a collections.OrderedDict where the keys are the column names, and
    values are the column data.

    """
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


def _to_tsv(data, fname):
    """ Write an OrderedDict into a tsv file """
    n_rows = len(data[list(data.keys())[0]])
    with open(fname, 'w') as f:
        f.write('\t'.join(list(data.keys())))
        f.write('\n')
        for idx in range(n_rows):
            row_data = list(str(data[key][idx]) for key in data)
            f.write('\t'.join(row_data))
            f.write('\n')


def _combine(data1, data2, drop_column=None):
    """Add two OrderedDict's together with the option of dropping repeated data

    Parameters
    ----------
    data1 : collections.OrderedDict
        Original OrderedDict.
    other : collections.OrderedDict
        New OrderedDict to be added to the original.
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
            # find the locations of the first n-1 values and remove the rows
            for _ in range(value - 1):
                idx = data1[drop_column].index(key)
                for column in data1.values():
                    del column[idx]


def _drop(data, values, column):
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
    mask = np.in1d(data[column], values, invert=True)
    for key in data.keys():
        data[key] = np.array(data[key])[mask].tolist()


def _contains_row(data, row_data):
    """Whether the specified row data exists in the OrderedDict """
    mask = None
    for idx, column_data in enumerate(data.values()):
        column_mask = np.in1d(np.array(column_data), row_data[idx])
        mask = column_mask if mask is None else (mask & column_mask)
    return np.any(mask)


def _prettyprint(data, rows=5):
    """pretty print an ordered dictionary """
    data_rows = len(data[list(data.keys())[0]])
    out_str = ''
    out_str += '\t'.join(list(data.keys())) + '\n'
    for i in range(min(data_rows, rows)):
        row_data = list(str(data[key][i]) for key in data)
        out_str += '\t'.join(row_data) + '\n'
    return out_str
