"""Configuration values for MNE-BIDS."""
from mne import io
from mne.io.constants import FIFF


BIDS_VERSION = "1.4.0"

DOI = """https://doi.org/10.21105/joss.01896"""

EPHY_ALLOWED_DATATYPES = ['meg', 'eeg', 'ieeg']

ALLOWED_DATATYPES = EPHY_ALLOWED_DATATYPES + ['anat', 'beh']

# Orientation of the coordinate system dependent on manufacturer
ORIENTATION = {
    '.con': 'KitYokogawa',
    '.ds': 'CTF',
    '.fif': 'ElektaNeuromag',
    '.pdf': '4DBti',
    '.sqd': 'KitYokogawa',
}

UNITS = {
    '.con': 'm',
    '.ds': 'cm',
    '.fif': 'm',
    '.pdf': 'm',
    '.sqd': 'm'
}

meg_manufacturers = {
    '.con': 'KIT/Yokogawa',
    '.ds': 'CTF',
    '.fif': 'Elekta',
    '.meg4': 'CTF',
    '.pdf': '4D Magnes',
    '.sqd': 'KIT/Yokogawa'
}

eeg_manufacturers = {'.vhdr': 'BrainProducts', '.eeg': 'BrainProducts',
                     '.edf': 'n/a', '.bdf': 'Biosemi', '.set': 'n/a',
                     '.fdt': 'n/a',
                     '.lay': 'Persyst', '.dat': 'Persyst',
                     '.EEG': 'Nihon Kohden'}

ieeg_manufacturers = {'.vhdr': 'BrainProducts', '.eeg': 'BrainProducts',
                      '.edf': 'n/a', '.set': 'n/a', '.fdt': 'n/a',
                      '.mef': 'n/a', '.nwb': 'n/a',
                      '.lay': 'Persyst', '.dat': 'Persyst',
                      '.EEG': 'Nihon Kohden'}

# file-extension map to mne-python readers
reader = {'.con': io.read_raw_kit, '.sqd': io.read_raw_kit,
          '.fif': io.read_raw_fif, '.pdf': io.read_raw_bti,
          '.ds': io.read_raw_ctf, '.vhdr': io.read_raw_brainvision,
          '.edf': io.read_raw_edf, '.bdf': io.read_raw_bdf,
          '.set': io.read_raw_eeglab, '.lay': io.read_raw_persyst,
          '.EEG': io.read_raw_nihon}


# Merge the manufacturer dictionaries in a python2 / python3 compatible way
# MANUFACTURERS dictionary only includes the extension of the input filename
# that mne-python accepts (e.g. BrainVision has three files, but the reader
# takes the filename for `.vhdr`)
MANUFACTURERS = dict()
MANUFACTURERS.update(meg_manufacturers)
MANUFACTURERS.update(eeg_manufacturers)
MANUFACTURERS.update(ieeg_manufacturers)

# List of synthetic channels by manufacturer that are to be excluded from the
# channel list. Currently this is only for stimulus channels.
IGNORED_CHANNELS = {'KIT/Yokogawa': ['STI 014'],
                    'BrainProducts': ['STI 014'],
                    'n/a': ['STI 014'],  # for unknown manufacturers, ignore it
                    'Biosemi': ['STI 014']}

allowed_extensions_meg = ['.con', '.sqd', '.fif', '.pdf', '.ds']
allowed_extensions_eeg = ['.vhdr',  # BrainVision, accompanied by .vmrk, .eeg
                          '.edf',  # European Data Format
                          '.bdf',  # Biosemi
                          '.set',  # EEGLAB, potentially accompanied by .fdt
                          ]

allowed_extensions_ieeg = ['.vhdr',  # BrainVision, accompanied by .vmrk, .eeg
                           '.edf',  # European Data Format
                           '.set',  # EEGLAB, potentially accompanied by .fdt
                           '.mef',  # MEF: Multiscale Electrophysiology File
                           '.nwb',  # Neurodata without borders
                           ]

# allowed extensions (data formats) in BIDS spec
ALLOWED_DATATYPE_EXTENSIONS = {'meg': allowed_extensions_meg,
                               'eeg': allowed_extensions_eeg,
                               'ieeg': allowed_extensions_ieeg}

# allow additional extensions that are not BIDS
# compliant, but we will convert to the
# recommended formats
ALLOWED_INPUT_EXTENSIONS = \
    allowed_extensions_meg + allowed_extensions_eeg + \
    allowed_extensions_ieeg + ['.lay', '.EEG']

# allowed suffixes (i.e. last "_" delimiter in the BIDS filenames before
# the extension)
ALLOWED_FILENAME_SUFFIX = [
    'meg', 'markers', 'eeg', 'ieeg', 'T1w', 'FLASH',  # datatype
    'participants', 'scans',
    'electrodes', 'channels', 'coordsystem', 'events',  # sidecars
    'headshape', 'digitizer',  # meg-specific sidecars
    'behav', 'phsyio', 'stim'  # behavioral
]

# converts suffix to known path modalities
SUFFIX_TO_DATATYPE = {
    'meg': 'meg', 'eeg': 'eeg', 'ieeg': 'ieeg', 'T1w': 'anat',
    'headshape': 'meg', 'digitizer': 'meg', 'markers': 'meg'
}

# allowed BIDS extensions (extension in the BIDS filename)
ALLOWED_FILENAME_EXTENSIONS = (
    ALLOWED_INPUT_EXTENSIONS +
    ['.json', '.tsv', '.tsv.gz', '.nii', '.nii.gz'] +
    ['.pos', '.eeg', '.vmrk'] +  # extra datatype-specific metadata files.
    ['.dat', '.EEG']  # extra eeg extensions
)

# allowed BIDS path entities
ALLOWED_PATH_ENTITIES = ('subject', 'session', 'task', 'run',
                         'processing', 'recording', 'space',
                         'acquisition', 'split',
                         'suffix', 'extension')
ALLOWED_PATH_ENTITIES_SHORT = {'sub': 'subject', 'ses': 'session',
                               'task': 'task', 'acq': 'acquisition',
                               'run': 'run', 'proc': 'processing',
                               'space': 'space', 'rec': 'recording',
                               'split': 'split', 'suffix': 'suffix'}

# See: https://bids-specification.readthedocs.io/en/latest/99-appendices/04-entity-table.html#encephalography-eeg-ieeg-and-meg  # noqa
ENTITY_VALUE_TYPE = {
    'subject': 'label',
    'session': 'label',
    'task': 'label',
    'run': 'index',
    'processing': 'label',
    'recording': 'label',
    'space': 'label',
    'acquisition': 'label',
    'split': 'index',
    'suffix': 'label',
    'extension': 'label'
}

# accepted BIDS formats, which may be subject to change
# depending on the specification
BIDS_IEEG_COORDINATE_FRAMES = ['acpc', 'pixels', 'other']
BIDS_MEG_COORDINATE_FRAMES = ['ctf', 'elektaneuromag',
                              '4dbti', 'kityokogawa',
                              'chietiitab', 'other']
BIDS_EEG_COORDINATE_FRAMES = ['captrak']

# accepted coordinate SI units
BIDS_COORDINATE_UNITS = ['m', 'cm', 'mm']

# mapping from BIDs coordinate frames -> MNE
BIDS_TO_MNE_FRAMES = {
    'ctf': 'ctf_head',
    '4dbti': 'ctf_head',
    'kityokogawa': 'ctf_head',
    'elektaneuromag': 'head',
    'chietiitab': 'head',
    'captrak': 'head',
    'acpc': 'ras',
    'mni': 'mni_tal',
    'fs': 'fs_tal',
    'ras': 'ras',
    'voxel': 'mri_voxels',
    'mri': 'mri',
    'unknown': 'unknown'
}
MNE_TO_BIDS_FRAMES = {val: key for key, val in BIDS_TO_MNE_FRAMES.items()}

# these coordinate frames in mne-python are related to scalp/meg
# 'meg', 'ctf_head', 'ctf_meg', 'head', 'unknown'
# copied from "mne.transforms.MNE_STR_TO_FRAME"
MNE_STR_TO_FRAME = dict(
    meg=FIFF.FIFFV_COORD_DEVICE,
    mri=FIFF.FIFFV_COORD_MRI,
    mri_voxel=FIFF.FIFFV_MNE_COORD_MRI_VOXEL,
    head=FIFF.FIFFV_COORD_HEAD,
    mni_tal=FIFF.FIFFV_MNE_COORD_MNI_TAL,
    ras=FIFF.FIFFV_MNE_COORD_RAS,
    fs_tal=FIFF.FIFFV_MNE_COORD_FS_TAL,
    ctf_head=FIFF.FIFFV_MNE_COORD_CTF_HEAD,
    ctf_meg=FIFF.FIFFV_MNE_COORD_CTF_DEVICE,
    unknown=FIFF.FIFFV_COORD_UNKNOWN
)
MNE_FRAME_TO_STR = {val: key for key, val in MNE_STR_TO_FRAME.items()}

# see BIDS specification for description we copied over from each
COORD_FRAME_DESCRIPTIONS = {
    'ctf': 'ALS orientation and the origin between the ears',
    'elektaneuromag': 'RAS orientation and the origin between the ears',
    '4dbti': 'ALS orientation and the origin between the ears',
    'kityokogawa': 'ALS orientation and the origin between the ears',
    'chietiitab': 'RAS orientation and the origin between the ears',
    'captrak': (
        'The X-axis goes from the left preauricular point (LPA) through '
        'the right preauricular point (RPA). '
        'The Y-axis goes orthogonally to the X-axis through the nasion (NAS). '
        'The Z-axis goes orthogonally to the XY-plane through the vertex of '
        'the head. '
        'This corresponds to a "RAS" orientation with the origin of the '
        'coordinate system approximately between the ears. '
        'See Appendix VIII in the BIDS specification.'),
    'mri': 'Defined by Freesurfer, the MRI (surface RAS) origin is at the '
           'center of a 256×256×256 1mm anisotropic volume '
           '(may not be in the center of the head).',
    'mri_voxel': 'Defined by Freesurfer, the MRI (surface RAS) origin '
                 'is at the center of a 256×256×256 voxel anisotropic '
                 'volume (may not be in the center of the head).',
    'mni_tal': 'MNI template in Talairach coordinates',
    'fs_tal': 'Freesurfer template in Talairach coordinates',
    'ras': 'RAS means that the first dimension (X) points towards '
           'the right hand side of the head, the second dimension (Y) '
           'points towards the Anterior aspect of the head, and the '
           'third dimension (Z) points towards the top of the head.',
}

REFERENCES = {'mne-bids':
              'Appelhoff, S., Sanderson, M., Brooks, T., Vliet, M., '
              'Quentin, R., Holdgraf, C., Chaumon, M., Mikulan, E., '
              'Tavabi, K., Höchenberger, R., Welke, D., Brunner, C., '
              'Rockhill, A., Larson, E., Gramfort, A. and Jas, M. (2019). '
              'MNE-BIDS: Organizing electrophysiological data into the '
              'BIDS format and facilitating their analysis. Journal of '
              'Open Source Software 4: (1896). '
              'https://doi.org/10.21105/joss.01896',
              'meg':
              'Niso, G., Gorgolewski, K. J., Bock, E., Brooks, T. L., '
              'Flandin, G., Gramfort, A., Henson, R. N., Jas, M., Litvak, '
              'V., Moreau, J., Oostenveld, R., Schoffelen, J., Tadel, F., '
              'Wexler, J., Baillet, S. (2018). MEG-BIDS, the brain '
              'imaging data structure extended to magnetoencephalography. '
              'Scientific Data, 5, 180110. '
              'https://doi.org/10.1038/sdata.2018.110',
              'eeg':
              'Pernet, C. R., Appelhoff, S., Gorgolewski, K. J., '
              'Flandin, G., Phillips, C., Delorme, A., Oostenveld, R. (2019). '
              'EEG-BIDS, an extension to the brain imaging data structure '
              'for electroencephalography. Scientific Data, 6, 103. '
              'https://doi.org/10.1038/s41597-019-0104-8',
              'ieeg':
              'Holdgraf, C., Appelhoff, S., Bickel, S., Bouchard, K., '
              'D\'Ambrosio, S., David, O., … Hermes, D. (2019). iEEG-BIDS, '
              'extending the Brain Imaging Data Structure specification '
              'to human intracranial electrophysiology. Scientific Data, '
              '6, 102. https://doi.org/10.1038/s41597-019-0105-7'}


# Mapping subject information between MNE-BIDS and MNE-Python.
HAND_BIDS_TO_MNE = {
    ('n/a',): 0,
    ('right', 'r', 'R', 'RIGHT', 'Right'): 1,
    ('left', 'l', 'L', 'LEFT', 'Left'): 2,
    ('ambidextrous', 'a', 'A', 'AMBIDEXTROUS', 'Ambidextrous'): 3
}

HAND_MNE_TO_BIDS = {0: 'n/a', 1: 'R', 2: 'L', 3: 'A'}

SEX_BIDS_TO_MNE = {
    ('n/a', 'other', 'o', 'O', 'OTHER', 'Other'): 0,
    ('male', 'm', 'M', 'MALE', 'Male'): 1,
    ('female', 'f', 'F', 'FEMALE', 'Female'): 2
}

SEX_MNE_TO_BIDS = {0: 'n/a', 1: 'M', 2: 'F'}


def _map_options(what, key, fro, to):
    if what == 'sex':
        mapping_bids_mne = SEX_BIDS_TO_MNE
        mapping_mne_bids = SEX_MNE_TO_BIDS
    elif what == 'hand':
        mapping_bids_mne = HAND_BIDS_TO_MNE
        mapping_mne_bids = HAND_MNE_TO_BIDS
    else:
        raise ValueError('Can only map `sex` and `hand`.')

    if fro == 'bids' and to == 'mne':
        # Many-to-one mapping
        mapped_option = None
        for bids_keys, mne_option in mapping_bids_mne.items():
            if key in bids_keys:
                mapped_option = mne_option
                break
    elif fro == 'mne' and to == 'bids':
        # One-to-one mapping
        mapped_option = mapping_mne_bids.get(key, None)
    else:
        raise RuntimeError("fro value {} and to value {} are not "
                           "accepted. Use 'mne', or 'bids'.".format(fro, to))

    return mapped_option
