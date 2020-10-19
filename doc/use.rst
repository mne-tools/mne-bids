:orphan:

Using MNE-BIDS
==============

Contents
--------

- `Quickstart`_
- `Examples`_
- :doc:`Python API reference <api>`
- :doc:`Command-line interface (CLI) reference<generated/cli>`


Quickstart
----------
MNE-BIDS fully supports writing of BIDS datasets for MEG and EEG. Support for
iEEG is experimental at the moment.

Python
~~~~~~

.. code:: python

    >>> import mne
    >>> from mne_bids import write_raw_bids
    >>> raw = mne.io.read_raw_fif('my_old_file.fif')
    >>> write_raw_bids(raw, 'sub-01_ses-01_run-05', bids_root='./bids_dataset')

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

In addition to ``import mne_bids``, you can use the command line interface.
Simply type ``mne_bids`` in your command line and press enter, to see the
accepted commands. Then type ``mne_bids <command> --help``, where ``<command>``
is one of the accepted commands, to get more information about that
``<command>``.

Example:

.. code-block:: bash

  $ mne_bids raw_to_bids --subject_id sub01 --task rest --raw data.edf --bids_root new_path


.. include:: auto_examples/index.rst
   :start-after: :orphan: 
