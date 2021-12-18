"""Configuration values for MNE-BIDS."""
from mne import io
from mne.io.constants import FIFF


BIDS_VERSION = "1.6.0"

DOI = """https://doi.org/10.21105/joss.01896"""

EPHY_ALLOWED_DATATYPES = ['meg', 'eeg', 'ieeg']

ALLOWED_DATATYPES = EPHY_ALLOWED_DATATYPES + ['anat', 'beh']

MEG_CONVERT_FORMATS = ['FIF', 'auto']
EEG_CONVERT_FORMATS = ['BrainVision', 'auto']
IEEG_CONVERT_FORMATS = ['BrainVision', 'auto']
CONVERT_FORMATS = {
    'meg': MEG_CONVERT_FORMATS,
    'eeg': EEG_CONVERT_FORMATS,
    'ieeg': IEEG_CONVERT_FORMATS
}

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
                     '.edf': 'n/a', '.EDF': 'n/a', '.bdf': 'Biosemi',
                     '.BDF': 'Biosemi',
                     '.set': 'n/a', '.fdt': 'n/a',
                     '.lay': 'Persyst', '.dat': 'Persyst',
                     '.EEG': 'Nihon Kohden'}

ieeg_manufacturers = {'.vhdr': 'BrainProducts', '.eeg': 'BrainProducts',
                      '.edf': 'n/a', '.EDF': 'n/a', '.set': 'n/a',
                      '.fdt': 'n/a', '.mef': 'n/a', '.nwb': 'n/a',
                      '.lay': 'Persyst', '.dat': 'Persyst',
                      '.EEG': 'Nihon Kohden'}

# file-extension map to mne-python readers
reader = {'.con': io.read_raw_kit, '.sqd': io.read_raw_kit,
          '.fif': io.read_raw_fif, '.pdf': io.read_raw_bti,
          '.ds': io.read_raw_ctf, '.vhdr': io.read_raw_brainvision,
          '.edf': io.read_raw_edf, '.EDF': io.read_raw_edf,
          '.bdf': io.read_raw_bdf,
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
    'beh', 'physio', 'stim'  # behavioral
]

# converts suffix to known path modalities
SUFFIX_TO_DATATYPE = {
    'meg': 'meg', 'headshape': 'meg', 'digitizer': 'meg', 'markers': 'meg',
    'eeg': 'eeg', 'ieeg': 'ieeg',
    'T1w': 'anat', 'FLASH': 'anat'
}

# allowed BIDS extensions (extension in the BIDS filename)
ALLOWED_FILENAME_EXTENSIONS = (
    ALLOWED_INPUT_EXTENSIONS +
    ['.json', '.tsv', '.tsv.gz', '.nii', '.nii.gz'] +
    ['.pos', '.eeg', '.vmrk'] +  # extra datatype-specific metadata files.
    ['.dat', '.EEG'] +  # extra eeg extensions
    ['.mrk']  # KIT/Yokogawa/Ricoh marker coil
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
                               'split': 'split'}

# Annotations to never remove during reading or writing
ANNOTATIONS_TO_KEEP = ('BAD_ACQ_SKIP',)

coordsys_standard_template = [
    'ICBM452AirSpace',
    'ICBM452Warp5Space',
    'IXI549Space',
    'fsaverage',
    'fsaverageSym',
    'fsLR',
    'MNIColin27',
    'MNI152Lin',
    'MNI152NLin2009aSym',
    'MNI152NLin2009bSym',
    'MNI152NLin2009cSym',
    'MNI152NLin2009aAsym',
    'MNI152NLin2009bAsym',
    'MNI152NLin2009cAsym',
    'MNI152NLin6Sym',
    'MNI152NLin6ASym',
    'MNI305',
    'NIHPD',
    'OASIS30AntsOASISAnts',
    'OASIS30Atropos',
    'Talairach',
    'UNCInfant',
]

coordsys_standard_template_deprecated = [
    'fsaverage3',
    'fsaverage4',
    'fsaverage5',
    'fsaverage6',
    'fsaveragesym',
    'UNCInfant0V21',
    'UNCInfant1V21',
    'UNCInfant2V21',
    'UNCInfant0V22',
    'UNCInfant1V22',
    'UNCInfant2V22',
    'UNCInfant0V23',
    'UNCInfant1V23',
    'UNCInfant2V23',
]

coordsys_meg = ['CTF', 'ElektaNeuromag', '4DBti', 'KitYokogawa', 'ChietiItab']
coordsys_eeg = ['CapTrak']
coordsys_ieeg = ['Pixels', 'ACPC']
coordsys_wildcard = ['Other']
coordsys_shared = (coordsys_standard_template +
                   coordsys_standard_template_deprecated +
                   coordsys_wildcard)

ALLOWED_SPACES = dict()
ALLOWED_SPACES['meg'] = coordsys_shared + coordsys_meg + coordsys_eeg
ALLOWED_SPACES['eeg'] = coordsys_shared + coordsys_meg + coordsys_eeg
ALLOWED_SPACES['ieeg'] = coordsys_shared + coordsys_ieeg
ALLOWED_SPACES['anat'] = None
ALLOWED_SPACES['beh'] = None

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
BIDS_IEEG_COORDINATE_FRAMES = ['ACPC', 'Pixels', 'Other']
BIDS_MEG_COORDINATE_FRAMES = ['CTF', 'ElektaNeuromag',
                              '4DBti', 'KitYokogawa',
                              'ChietiItab', 'Other']
BIDS_EEG_COORDINATE_FRAMES = ['CapTrak']

# accepted coordinate SI units
BIDS_COORDINATE_UNITS = ['m', 'cm', 'mm']

# mapping from supported BIDs coordinate frames -> MNE
BIDS_TO_MNE_FRAMES = {
    'CTF': 'ctf_head',
    '4DBti': 'ctf_head',
    'KitYokogawa': 'ctf_head',
    'ElektaNeuromag': 'head',
    'ChietiItab': 'head',
    'CapTrak': 'head',
    'ACPC': 'mri',  # assumes T1 is ACPC-aligned, if not the coordinates are lost  # noqa
    'fsaverage': 'mni_tal',  # XXX: note fsaverage and MNI305 are the same  # noqa
    'MNI305': 'mni_tal',
    'Other': 'unknown'
}

# mapping from supported MNE coordinate frames -> BIDS
# XXX: note that there are a lot fewer MNE available coordinate
# systems so the range of BIDS supported coordinate systems we
# can write is limited.
MNE_TO_BIDS_FRAMES = {
    'ctf_head': 'CTF',
    'head': 'CapTrak',
    'mni_tal': 'fsaverage',
    # 'fs_tal': 'fsaverage',  # XXX: not used
    'unknown': 'Other',
    'ras': 'Other',
    'mri': 'Other'
}

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
BIDS_COORD_FRAME_DESCRIPTIONS = {
    'acpc': 'The origin of the coordinate system is at the Anterior '
            'Commissure and the negative y-axis is passing through the '
            'Posterior Commissure. The positive z-axis is passing through '
            'a mid-hemispheric point in the superior direction.',
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
    'fsaverage': 'Defined by FreeSurfer, the MRI (surface RAS) origin is '
                 'at the center of a 256×256×256 mm^3 anisotropic volume '
                 '(may not be in the center of the head).',
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


# Which JSON data can safely be transferred from a non-anonymized to an
# anonymized dataset without accidentally exposing personal identifiable
# information
ANONYMIZED_JSON_KEY_WHITELIST = [
    # Common
    'Manufacturer',
    'ManufacturersModelName',
    'InstitutionName',
    'InstitutionalDepartmentName',
    'InstitutionAddress',
    'DeviceSerialNumber',
    # MRI
    # Many of these are not standardized, but produced by dcm2niix.
    'Modality',
    'MagneticFieldStrength',
    'ImagingFrequency',
    'StationName',
    'SeriesInstanceUID',
    'StudyInstanceUID',
    'StudyID',
    'BodyPartExamined',
    'PatientPosition',
    'ProcedureStepDescription',
    'SoftwareVersions',
    'MRAcquisitionType',
    'SeriesDescription',
    'ProtocolName',
    'ScanningSequence',
    'SequenceVariant',
    'ScanOptions',
    'SequenceName',
    'ImageType',
    'SeriesNumber',
    'AcquisitionNumber',
    'SliceThickness',
    'SAR',
    'EchoTime',
    'RepetitionTime',
    'InversionTime',
    'FlipAngle',
    'PartialFourier',
    'BaseResolution',
    'ShimSetting',
    'TxRefAmp',
    'PhaseResolution',
    'ReceiveCoilName',
    'ReceiveCoilActiveElements',
    'PulseSequenceDetails',
    'ConsistencyInfo',
    'PercentPhaseFOV',
    'PercentSampling',
    'PhaseEncodingSteps',
    'AcquisitionMatrixPE',
    'PixelBandwidth',
    'DwellTime',
    'ImageOrientationPatientDICOM',
    'InPlanePhaseEncodingDirectionDICOM',
    'ConversionSoftware',
    'ConversionSoftwareVersion',
    # Electrophys common
    'TaskName',
    'TaskDescription',
    'Instructions',
    'PowerLineFrequency',
    'SamplingFrequency',
    'SoftwareFilters',
    'RecordingType',
    'EEGChannelCount',
    'EOGChannelCount',
    'ECGChannelCount',
    'EMGChannelCount',
    'MiscChannelCount',
    'TriggerChannelCount',
    'RecordingDuration',
    # EEG
    'EEGReference',
    'EEGPlacementScheme',
    # MEG
    'DewarPosition',
    'DigitizedLandmarks',
    'DigitizedHeadPoints',
    'MEGChannelCount',
    'MEGREFChannelCount',
    'ContinuousHeadLocalization',
    'HeadCoilFrequency'
]
