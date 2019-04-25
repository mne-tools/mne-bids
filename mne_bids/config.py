"""Configuration values for MNE-BIDS."""

BIDS_VERSION = "1.2.0"

ALLOWED_KINDS = ['meg', 'eeg', 'ieeg']

# Orientation of the coordinate system dependent on manufacturer
ORIENTATION = {'.sqd': 'ALS', '.con': 'ALS', '.fif': 'RAS', '.pdf': 'ALS',
               '.ds': 'ALS'}

UNITS = {'.sqd': 'm', '.con': 'm', '.fif': 'm', '.pdf': 'm', '.ds': 'cm'}

meg_manufacturers = {'.sqd': 'KIT/Yokogawa', '.con': 'KIT/Yokogawa',
                     '.fif': 'Elekta', '.pdf': '4D Magnes', '.ds': 'CTF',
                     '.meg4': 'CTF'}

eeg_manufacturers = {'.vhdr': 'BrainProducts', '.eeg': 'BrainProducts',
                     '.edf': 'n/a', '.bdf': 'Biosemi', '.set': 'n/a',
                     '.fdt': 'n/a'}

ieeg_manufacturers = {'.vhdr': 'BrainProducts', '.eeg': 'BrainProducts',
                      '.edf': 'n/a', '.set': 'n/a', '.fdt': 'n/a',
                      '.mef': 'n/a', '.nwb': 'n/a'}

# Merge the manufacturer dictionaries in a python2 / python3 compatible way
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

ALLOWED_EXTENSIONS = (allowed_extensions_meg +
                      allowed_extensions_eeg +
                      allowed_extensions_ieeg)
