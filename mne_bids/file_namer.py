# Authors: Matt Sanderson <msvalleyfields@hotmail.com>
#
# License: BSD (3-clause)

from collections import OrderedDict
from mne.externals.six import string_types
from os.path import join, splitext

from .utils import _check_types, _check_key_val

# for now this is only MEG compatible
class BIDSName():
    def __init__(self, subject=None, session=None, task=None,
                 acquisition=None, run=None, processing=None,
                 recording=None, space=None, kind='meg'):
        """
        Constructs an object that has all the information required to automatically construct
        the names for various files within the bids sub-directories

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
        run : int | None
            The run number for this item. Corresponds to "run".
        processing : str | None
            The processing label for this item. Corresponds to "proc".
        recording : str | None
            The recording name for this item. Corresponds to "recording".
        space : str | None
            The coordinate space for an anatomical file. Corresponds to "space".
        kind : str | "meg"
            The kind of data being converted. Defaults to "meg".
        """
        self.order = OrderedDict([('sub', subject),
                         ('ses', session),
                         ('task', task),
                         ('acq', acquisition),
                         ('run', run),
                         ('proc', processing),
                         ('space', space),
                         ('recording', recording)])

        self.kind = kind

        self._required_fields = ['sub']
        # A dictionary containing the different fields we may want in the name
        # these are fields on top of the required ones (which is only the 'sub' one)
        self._file_specific_requirements = {'coordsystem.json':['ses', 'acq'],
                                            'channels.tsv':['task', 'ses', 'acq', 'run', 'proc'],
                                            'meg.json':['task', 'ses', 'acq', 'run', 'proc'],
                                            'scans.tsv':['ses'],
                                            'events.tsv':['ses', 'task'],
                                            '_rawfiles':['task', 'ses', 'acq', 'run', 'proc']}

        if self.order['run'] is not None and not isinstance(self.order['run'], string_types):
            # Ensure that run is a string
            self.order['run'] = '{:02}'.format(self.order['run'])

        _check_types(self.order.values())

        if not any(isinstance(ii, string_types) for ii in self.order.keys()):
            raise ValueError("At least one parameter must be given.")

        # construct the base path:
        if session is not None:
            self._basepath = join("sub-{0}".format(subject), "ses-{0}".format(session))
        else:
            self._basepath = join("sub-{0}".format(subject))

    def get_filename(self, file=None, parent_directory=""):
        """
        This will return the filename associated with the file
        By default we will simply want the relative path, however, if a
        parent directory is specified it will be appended to the front of the path.

        Parameters
        ----------
        file : str | None
            The type of file to produce the name and path of.
            One of ('coordsystem.json', 'channels.tsv', 'meg.json', 'scans.tsv')
            for a specific file, or, to produce the names of raw files the original name,
            or even just file type may be entered (eg. 'data.con', or '.con')
        parent_directory : str | ""
            The parent directory to set all sub-folders relative to.
            Only needed when creating absolute paths.
        """
        if file in self._file_specific_requirements:
            required_fields = self._required_fields + self._file_specific_requirements[file]
            rawfile = False
        else:
            # assume that the default file name is required.
            # This is the going to be the same as the meg.json file so that this
            # name can be used to rename manufacturer specific files
            required_fields = self._required_fields + self._file_specific_requirements['_rawfiles']
            rawfile = True
        
        fields = []
        for key, val in self.order.items():
            if val is not None and key in required_fields:
                _check_key_val(key, val)
                fields.append("{0}-{1}".format(key, val))
        
        if rawfile:
            filename = "_".join(fields) + "_" + self.kind + self._get_ext(file)
        else:
            filename = "_".join(fields) + "_" + file

        if file != 'scans.tsv':
            path = join(parent_directory, self._basepath, self.kind, filename)
        else:
            path = join(parent_directory, self._basepath, filename)
        return path

    def _get_ext(self, fname):
        """ Get the extension for the file specified by fname """
        name, ext = splitext(fname)
        if ext == '':
            # in this case fname is simply the extension (possibly without the period, so fix this if needed)
            if name[0] == '.':
                return name
            else:
                return '.{0}'.format(name)
        else:
            return ext
