Glossary
========

This glossary introduces terminology commonly used when working with
MNE-BIDS and :class:`mne_bids.BIDSPath`.

BIDS
    The Brain Imaging Data Structure: a standard for organizing and naming
    neuroimaging and electrophysiological datasets and their metadata.

BIDS path
    A path whose directory structure and filename encode BIDS metadata. A
    BIDS path can be represented and manipulated in MNE-BIDS with
    :class:`mne_bids.BIDSPath`.

    For example::

        /bids_dataset/sub-01/ses-01/eeg/
        sub-01_ses-01_task-rest_run-01_eeg.vhdr

BIDS entity
    A key-value component of a BIDS filename. For example, ``sub-01`` is the
    subject entity and ``task-rest`` is the task entity. Entities appear in a
    defined order in BIDS filenames.

subject (``sub``)
    The subject identifier. In :class:`mne_bids.BIDSPath`, this is passed as
    ``subject``.

session (``ses``)
    The acquisition session identifier.

task (``task``)
    The experimental task associated with a recording.

acquisition (``acq``)
    A label describing acquisition parameters that distinguish otherwise
    similar files.

run (``run``)
    The run number used to distinguish repeated recordings with otherwise
    matching entities.

processing (``proc``)
    A processing label used to distinguish files according to processing.

recording (``recording``)
    A recording label used to distinguish recordings when required by the
    BIDS naming rules.

space (``space``)
    The coordinate space for anatomical and sensor-location files. Valid
    values are constrained by the BIDS specification.

split (``split``)
    The split number of a continuous recording file, used by MNE-BIDS for
    split ``.fif`` data.

description (``desc``)
    Additional information used primarily for derivative data. For example,
    a preprocessed derivative may use a description such as ``cleaned``.

suffix
    The component after the final underscore and before the file extension.
    Examples include ``eeg``, ``meg``, ``events`` and ``channels``. The suffix
    describes the type or purpose of the file and is not a key-value entity.

extension
    The filename extension, for example ``.json``, ``.tsv`` or a supported
    recording-data extension.

datatype
    The BIDS data-type directory, for example ``eeg``, ``meg``, ``ieeg``,
    ``anat`` or ``func``. It is part of the directory structure rather than a
    filename entity.

root
    The root directory of the BIDS dataset. In
    :class:`mne_bids.BIDSPath`, setting ``root`` allows generation of a full
    path instead of only a relative BIDS path.
