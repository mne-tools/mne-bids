.. _bids-glossary:

Glossary of BIDS terms in MNE-BIDS
==================================

MNE-BIDS uses Python to implement the Brain Imaging Data Structure (BIDS)
specification. Working with BIDS in Python results in a few special terms that
MNE-BIDS uses but the BIDS specification does not. MNE-BIDS does not rename
BIDS concepts. This glossary defines the key terms of MNE-BIDS and explains how
they relate to BIDS terms.

Anatomy of a BIDS path
----------------------

:class:`~mne_bids.BIDSPath` objects are the primary way that MNE-BIDS stores
and organizes information about subject data. To see what a ``BIDSPath`` holds,
take this file. Its parts are listed below, grouped by whether the term comes
from the BIDS specification or from MNE-BIDS:

.. code-block:: text

    /bids_dataset/sub-01/ses-02/eeg/sub-01_ses-02_task-rest_run-01_eeg.vhdr

    BIDS terms
      root       /bids_dataset
      datatype   eeg
      entities   sub-01  ses-02  task-rest  run-01
      suffix     eeg
      extension  .vhdr

    MNE-BIDS terms
      directory  /bids_dataset/sub-01/ses-02/eeg
      basename   sub-01_ses-02_task-rest_run-01_eeg.vhdr
      fpath      the whole path above (the directory plus the basename)

The BIDS specification talks about a file name and the location of a file in
the dataset. MNE-BIDS adds the ``basename``, ``directory`` and ``fpath``
properties to describe the path in Python. ``root`` is also a BIDS term (the
specification says "root directory").

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
      changed in place with ``update()``. It does not have to point to a single
      existing file: it can hold only some entities and act as a search
      pattern, which ``BIDSPath.match()`` uses to find matching files in the
      dataset.

   basename
      The file name without any parent folders, for example
      ``sub-01_ses-02_task-rest_run-01_eeg.vhdr``. This is the
      ``BIDSPath.basename`` property. BIDS simply calls this the file name.
      The name matches Python's :func:`os.path.basename`, but ``BIDSPath``
      builds it from its entities, not from an existing path.

   check
      A ``BIDSPath`` option. If ``True`` (the default), MNE-BIDS enforces
      BIDS conformity, for example only allowing known data types, suffixes
      and extensions. Setting ``check=False`` allows paths that MNE-BIDS does
      not natively support, such as ``dataset_description.json``.

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
      ``BIDSPath.entities`` returns them as a dictionary. A value is either a
      *label* (letters and digits, such as the ``patient`` in ``sub-patient``)
      or an *index* (a non-negative integer, such as the ``01`` in ``run-01``).

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

Legacy MNE-BIDS terms
---------------------

Version 0.5 introduced ``BIDSPath`` and renamed several terms. The old names
below are no longer used in MNE-BIDS, but you may still see them in release
notes and in code written for older versions:

- ``make_bids_basename`` became ``BIDSPath``.
- ``prefix`` became ``root``.
- ``kind``, the name of the data type, became ``datatype``.
- ``get_kinds`` became ``get_datatypes``.
- The ``bids_basename`` and ``bids_root`` arguments of functions such as
  ``write_raw_bids`` and ``read_raw_bids`` became a single ``bids_path``
  argument.

More changes are listed in :doc:`whats_new_previous_releases`.