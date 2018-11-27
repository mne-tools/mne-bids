import numpy as np


def odict_to_ndarray(d):
    """ Cast an ordered dictionary to a numpy ndarray """
    titles = np.reshape(np.array(list(d.keys())), (1, len(d)))
    data = np.stack(d.values(), axis=-1).astype(str)
    return np.concatenate([titles, data])


def from_tsv(fname):
    """ Read a tsv file into an ndarray """
    return np.loadtxt(fname, dtype=str, delimiter='\t')


def to_tsv(data, fname):
    """ Write an ndarray to a tsv file """
    np.savetxt(fname, data, fmt='%s', delimiter='\t', comments='')


def is_contained(data, other):
    """ Checks whether `other` is contained within `data`

    Parameters
    ----------
    data : numpy.ndarray
    other : numpy.ndarray

    """
    for row in other[1:]:
        if not row.tolist() in data[1:].tolist():
            return False
    return True


def column_data(data, column):
    """ Returns the data in the specified column """
    col_num = data[0].tolist().index(column)
    return data[1:, col_num]


def combine_data(data, other, drop_column=None):
    """ Add two ndarrays of data with column names together

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
    data_head, data_data = np.split(data, [1])
    other_head, other_data = np.split(other, [1])
    if not np.array_equal(data_head, other_head):
        raise ValueError("data sets do not have the same column names")
    if drop_column is not None:
        col_num = data_head[0].tolist().index(drop_column)
        drop_indexes = np.where(np.isin(data_data[:, col_num],
                                        other_data[:, col_num]))[0]
        data_data = np.delete(data_data, drop_indexes, 0)

    return np.concatenate([data_head, data_data, other_data])


def drop(data, values, column):
    """ Remove specified values in a column

    Parameters
    ----------
    data : numpy.ndarray
        Data to drop values from.
    values : list
        List of values to drop. Any row containing this value in the specified
        column will be dropped.
    column : string
        Name of the column to check for the existence of `value` in.

    """
    if len(values) != 0:
        col_num = data[0].tolist().index(column)
        values = np.array(values).astype(str)
        loc_data = np.where(np.isin(data, values))
        if col_num not in loc_data[1]:
            raise ValueError('values not found in column "%s"' % column)
        row_indexes = loc_data[0][np.where(loc_data[1] == col_num)]
        return np.delete(data, row_indexes, 0)
    else:
        return data
