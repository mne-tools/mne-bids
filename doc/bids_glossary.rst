.. _bids-glossary:

Glossary of BIDS terms in MNE-BIDS
==================================

MNE-BIDS is a Python library built on the Brain Imaging Data Structure (BIDS)
standard. Some BIDS ideas have different names in MNE-BIDS, and MNE-BIDS has a
few terms of its own. This page explains these terms and lists where MNE-BIDS
differs from the BIDS specification.

Anatomy of a BIDS path
----------------------

Take this file. Its parts are called:

.. code-block:: text

    /bids_dataset/sub-01/ses-02/eeg/sub-01_ses-02_task-rest_run-01_eeg.vhdr

    root       /bids_dataset
    directory  /bids_dataset/sub-01/ses-02/eeg
    basename   sub-01_ses-02_task-rest_run-01_eeg.vhdr
    entities   sub-01  ses-02  task-rest  run-01
    suffix     eeg
    extension  .vhdr
    fpath      the whole path above (the directory plus the basename)

The same file described with :class:`mne_bids.BIDSPath`:

.. code-block:: python

    from mne_bids import BIDSPath

    bids_path = BIDSPath(
        subject="01", session="02", task="rest", run="01",
        suffix="eeg", extension=".vhdr", datatype="eeg",
        root="/bids_dataset",
    )
    bids_path.basename   # 'sub-01_ses-02_task-rest_run-01_eeg.vhdr'
    bids_path.directory  # Path('/bids_dataset/sub-01/ses-02/eeg')
    bids_path.fpath

Glossary
--------

.. glossary::
   :sorted:

   BIDSPath
      The MNE-BIDS class that stores the pieces of a BIDS path (the
      :term:`entities <entity>`, :term:`suffix`, :term:`extension`,
      :term:`datatype`, and :term:`root`) and builds names and paths from
      them. It is a standalone class, not a subclass of
      :class:`pathlib.Path`, but its ``root``, ``directory`` and ``fpath``
      properties return :class:`pathlib.Path` objects. Its entities can be
      changed in place with ``update()``.

   basename
      The file name without any parent folders, for example
      ``sub-01_ses-02_task-rest_run-01_eeg.vhdr``. This is the
      ``BIDSPath.basename`` property. BIDS simply calls this the file name.
      The name matches Python's :func:`os.path.basename`, but ``BIDSPath``
      builds it from its entities, not from an existing path.

   check
      A ``BIDSPath`` option. If ``True`` (the default), MNE-BIDS enforces
      BIDS conformity, for example only allowing known suffixes and
      extensions. Setting ``check=False`` allows paths that MNE-BIDS does not
      natively support, such as ``dataset_description.json``.

   datatype
      The kind of data a file holds, which is also the name of the folder it
      sits in: for example ``anat``, ``eeg``, ``ieeg``, ``meg``. BIDS writes
      it as "data type". It is related to, but not the same as, the
      :term:`modality`. Since version 0.11, ``BIDSPath`` no longer infers the
      datatype from the :term:`suffix`, so you need to give it explicitly
      wherever it is needed.

   directory
      The folder that contains a file: the :term:`root`, followed by
      ``sub-<label>``, ``ses-<label>``, and the :term:`datatype` folder when
      they are set. This is the ``BIDSPath.directory`` property and returns a
      :class:`pathlib.Path`. Like ``basename``, it only builds text from the
      entities and never looks at the disk.

   entity
      A key-value piece of a BIDS file name, such as ``sub-01`` or
      ``task-rest``. Entities are separated by ``_``, and the key and value
      within an entity are separated by ``-``. They must appear in a fixed
      order, and no entity may appear more than once in a file name.
      ``BIDSPath.entities`` returns them as a dictionary.

   extension
      The file type at the end of the file name, including the leading
      period, for example ``.vhdr`` or ``.tsv``.

   fpath
      The full path to the file, as a :class:`pathlib.Path`. This is the
      ``BIDSPath.fpath`` property. The path is relative if :term:`root` is not
      set or is itself relative, and absolute only if ``root`` is absolute.

      Unlike ``basename`` and ``directory``, ``fpath`` can search the
      dataset. If ``root`` is set and the ``suffix`` or ``extension`` is
      missing, it looks for the matching file in the dataset and returns that
      file's path. It also looks for recordings that are split into several
      files, and returns the first part (``split-01``) if it finds one. If
      more than one file matches, it raises a ``RuntimeError``. If no file is
      found, it returns the path built from the entities.

   modality
      The category of brain data that a file holds, for example MEG, EEG, or
      MRI. BIDS says that a modality may overlap with the :term:`datatype`,
      but should not be confused with it. ``BIDSPath`` has no ``modality``
      argument. Comments and docstrings in older MNE-BIDS versions sometimes
      used the word loosely for what is now called the ``datatype``.

   root
      The top-level folder of a BIDS dataset. This is the ``BIDSPath.root``
      property. BIDS calls it the "root directory" or "dataset root".

   suffix
      The last part of the file name before the extension, after the final
      ``_``, for example ``eeg``, ``channels``, ``events``, or ``T1w``. It
      tells you what kind of file this is. Not to be confused with the
      :term:`datatype`: ``eeg`` can be both a suffix and a datatype, while
      ``channels`` is only a suffix.

Entity reference
----------------

MNE-BIDS uses full words for its Python arguments, while BIDS file names use
short keys. The rows follow the order in which the BIDS specification requires
entities to appear in a file name.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - ``BIDSPath`` argument
     - Key in file name
     - Notes
   * - ``subject``
     - ``sub``
     -
   * - ``session``
     - ``ses``
     -
   * - ``task``
     - ``task``
     -
   * - ``tracking_system``
     - ``tracksys``
     - Used for Motion-BIDS data.
   * - ``acquisition``
     - ``acq``
     -
   * - ``run``
     - ``run``
     - A non-negative integer in the spec. Since version 0.15, ``run=1`` is
       not converted to ``run-01``. Pass ``'01'`` if you want two digits.
   * - ``processing``
     - ``proc``
     - BIDS calls it "Processed (on device)". In the spec's file tables,
       allowed for MEG files, but not for EEG, iEEG, or Motion files.
   * - ``space``
     - ``space``
     - For electrode, coordinate system, and marker files. With
       ``check=True``, ``datatype`` must be set. For ``meg``, ``eeg`` and
       ``ieeg``, ``space`` must be a coordinate system name that BIDS allows
       for that datatype. For ``emg``, any non-empty text is accepted. For
       ``anat``, ``beh``, ``motion`` and ``nirs``, ``check=True`` rejects
       ``space``. With ``check=False`` the value is not checked.
   * - ``split``
     - ``split``
     - For recordings that are split into several ``.fif`` files. Follows the
       same rule as ``run``.
   * - ``recording``
     - ``recording``
     - For continuous recordings such as ``physio`` and ``stim`` files.
   * - ``description``
     - ``desc``
     - Derivative data only, according to the spec.

``BIDSPath`` has arguments for the entities above only. The BIDS specification
defines many more entities, mostly for other data types (for example ``echo``,
``flip``, ``dir``, ``ce``, ``sample``, and ``part``).

Where MNE-BIDS differs from the BIDS specification
--------------------------------------------------

- ``basename``, ``directory``, ``fpath`` and ``check`` describe how MNE-BIDS
  works with paths in Python. They are not terms from the BIDS specification,
  which talks about file names and the root directory of a dataset instead.
- A ``BIDSPath`` does not have to point to a single existing file. It can hold
  only some entities and act as a search pattern, which ``BIDSPath.match()``
  uses to find real files in the dataset.
- The BIDS specification gives entity values one of two forms. A *label* is
  made of letters and digits, such as ``sub-patient`` or ``task-rest``. An
  *index* is a non-negative integer, such as ``run-01``. ``BIDSPath`` follows
  this for ``run`` and ``split``: with ``check=True`` they must be indices, so
  ``run='5H'`` raises an error, while with ``check=False`` other values are
  accepted. The specification recommends zero padding for indices, but
  ``BIDSPath`` writes the value as given, so ``run=1`` gives ``run-1``. For
  labels, ``BIDSPath`` only rejects the characters ``-``, ``_`` and ``/``, so
  it also accepts some values that the specification does not allow, such as
  ``a.b``.
- With ``check=True``, ``BIDSPath`` accepts the data types ``eeg``, ``emg``,
  ``ieeg``, ``meg``, ``nirs``, ``motion``, ``anat`` and ``beh``, and a fixed
  list of suffixes. BIDS defines more. For example, ``func`` and ``dwi`` are
  data types that ``BIDSPath`` does not accept by default, and ``photo`` and
  ``physioevents`` are suffixes that it rejects. Use ``check=False`` to build
  such paths anyway.

Old names
---------

Version 0.5 introduced ``BIDSPath`` and renamed several things. If you see
these names in old code or tutorials, this is what they became:

- ``make_bids_basename`` became ``BIDSPath``.
- ``prefix`` became ``root``.
- ``kind`` became ``datatype``.
- ``get_kinds`` became ``get_datatypes``.
- The ``bids_basename`` and ``bids_root`` arguments of the read and write
  functions became a single ``bids_path`` argument.

More changes are listed in :doc:`whats_new_previous_releases`.
