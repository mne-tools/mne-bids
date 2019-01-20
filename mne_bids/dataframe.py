import numpy as np
from collections import OrderedDict as odict


class DataFrame():
    def __init__(self, data):
        """ A stripped down clone of a Pandas DataFrame object

        Parameters
        -----------
        data : ordered dictionary
            An ordered dictionary containing the column names (keys) and
            associated column values (values).

        Note: All data is cast as a string to ensure uniformity and to avoid
        issues when comparing data. Because of this, operations on the data
        in this format should be avoided and any processing done before
        assignment.

        """
        # create a mapping between the column names and the column number
        self.columns = odict(zip(data.keys(), range(len(data))))
        self.arr = np.stack(data.values(), axis=-1).astype(str)

    def drop(self, values, column):
        """ Drop any rows with the specified values in the given column

        Parameters
        ----------
        values : list
            List of values to check for.
        column : str
            Name of the column to check values in.

        """
        # only need to drop values if there are some that need to be dropped
        if values != list():
            # if column not in self.columns we expect an error to be raised
            col_num = self.columns[column]
            # cast values to string type to be safe
            values = [str(i) for i in values]
            # create array with True values wherever the value is in values
            loc_data = np.where(np.isin(self.arr, values))
            if col_num not in loc_data[1]:
                raise ValueError('values not found in column "%s"' % column)
            row_indexes = loc_data[0][np.where(loc_data[1] == col_num)]
            self.arr = np.delete(self.arr, row_indexes, 0)

    def append(self, other, drop_column=None):
        """ Add one DataFrame to another

        Parameters
        ----------
        other : DataFrame
            The other DataFrame object to be appended to the end of the current
            one.
        drop_column : str
            The name of the column to check for multiple instances of the same
            value. If multiple values are found in the column, the rows
            containing all but the last value are removed.

        """
        self.arr = np.concatenate([self.arr, other.arr])
        if drop_column is not None:
            col_num = self.columns[drop_column]
            col_data = self.arr[:, col_num]
            drop_indexes = np.where(col_data == other.arr[0][col_num])[0]
            self.arr = np.delete(self.arr, drop_indexes[:-1], 0)

    def head(self, rows=5):
        """ Return a view of the first number of rows

        Parameters
        ----------
        rows : int
            Maximum number of rows to show

        """
        output = ''
        output += '\t'.join(list(self.columns.keys())) + '\n'
        count = min(rows, self.arr.shape[0])
        for i in range(count):
            output += '\t'.join(self.arr[i, :]) + '\n'
        return output

    def __contains__(self, item):
        """ Provide functionality for the `in` operator

        item may be either an numpy.ndarray or a DataFrame, however it may only
        be 1 dimensional.

        """
        if isinstance(item, np.ndarray):
            return item.tolist() in self.arr.tolist()
        elif isinstance(item, type(self)):
            return item.arr.flatten().tolist() in self.arr.tolist()

    def __getitem__(self, key):
        """ Return the data in the column specified """
        return self.arr[:, self.columns[key]]

    @classmethod
    def from_tsv(cls, fname):
        """ Generate a DataFrame object from a .tsv file

        Parameters
        ----------
        fname : str
            Path to the tsv to be loaded.

        """
        data = np.loadtxt(fname, dtype=str, delimiter='\t')
        # the first row will be the names
        column_names = data[0, :]
        info = data[1:, :]
        data_dict = odict()
        for i, name in enumerate(column_names):
            data_dict[name] = info[:, i]
        return cls(data_dict)

    def to_tsv(self, fname):
        """ Produce a tsv file

        Parameters
        ----------
        fname : str
            Path to the tsv to be generated.

        """
        header = '\t'.join(self.columns.keys())
        np.savetxt(fname, self.arr, fmt='%s', delimiter='\t', header=header,
                   comments='')
