[![Join the chat at https://gitter.im/mne-tools/mne-bids](https://badges.gitter.im/mne-tools/mne-bids.svg)](https://gitter.im/mne-tools/mne-bids?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

MNE-BIDS
========

This is a repository for experimental code for creating BIDS-compatible folders with MNE. 
All of this is considered as work-in-progress.

Installation
------------

We recommend the [Anaconda Python distribution](https://www.continuum.io/downloads).
To install ``mne_bids``, you first need to install its dependencies:

	$ pip install pandas mne

Then install mne_bids::

	$ pip install git+https://github.com/mne-tools/mne-bids.git#egg=mne-bids

If you do not have admin privileges on the computer, use the ``--user`` flag
with `pip`. To upgrade, use the ``--upgrade`` flag provided by `pip`.

To check if everything worked fine, you can do:

	$ python -c 'import mne_bids'

and it should not give any error messages.

Examples
--------
https://mne-tools.github.io/mne-bids/auto_examples/index.html

Contributions
-------------
Contributions are welcome in the form of pull requests.
Once the implementation of a piece of functionality is considered to be bug
free and properly documented (both API docs and an example script),
it can be incorporated into the master branch.
