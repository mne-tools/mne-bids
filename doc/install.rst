:orphan:

Installation
============

MNE-BIDS is included in the `official MNE-Python installers <https://mne.tools/stable/install/installers.html>`_.

If you want to install MNE-BIDS manually instead, please continue reading.

Dependencies
------------

Required:

* ``mne`` (>=1.8)
* ``numpy`` (>=1.23)
* ``scipy`` (>=1.9)

Optional:

* ``nibabel`` (>=3.2.1, for processing MRI data)
* ``pybv`` (>=0.7.5, for writing BrainVision data)
* ``eeglabio`` (>=0.0.2, for writing EEGLAB data)
* ``pymatreader`` (for other operations with EEGLAB data)
* ``matplotlib`` (>=3.6, for using the interactive data inspector)
* ``pandas`` (>=1.3.2, for generating event statistics)
* ``edfio`` (>=0.2.1, for writing EDF data)
* ``defusedxml`` (for writing reading EGI MFF data and BrainVision montages)

We recommend installing ``mne-bids`` into an isolated Python environment,
for example created via ``conda``
(may be obtained through `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_).
We require that you **use Python 3.10 or higher**.
You may choose to install ``mne-bids`` into your isolated Python environment
`via pip <#installation-via-pip>`_ or
`via conda <#installation-via-conda>`_.

Installation via pip
--------------------

To install MNE-BIDS including all dependencies required to use all features,
simply run:

.. code-block:: bash

   pip install --upgrade mne-bids[full]

This ``pip`` command will also work if you want to upgrade if a newer version
of MNE-BIDS is available.

If you don't require advanced features like interactive visual data inspection,
you may also install a basic version of MNE-BIDS via

.. code-block:: bash

   pip install --upgrade mne-bids

If you want to install a snapshot of the current development version, run:

.. code-block:: bash

   pip install --upgrade https://github.com/mne-tools/mne-bids/archive/refs/heads/main.zip

To check if everything worked fine, the following command should
print a version number and not give any error messages:

.. code-block:: bash

   python -c 'import mne_bids; print(mne_bids.__version__)'

MNE-BIDS works best with the latest stable release of MNE-Python (the ``mne`` package).
To ensure MNE-Python is up-to-date, follow the
`MNE-Python installation instructions <https://mne.tools/stable/install/#>`_.

Installation via conda
----------------------

If you have followed the
`MNE-Python installation instructions <https://mne.tools/stable/install/#>`_,
all that's left to do is to install MNE-BIDS without its dependencies,
as they've already been installed during the MNE-Python installation process.

Activate the correct ``conda`` environment and install ``mne-bids``:

.. code-block:: bash

   conda activate mne
   conda install --channel conda-forge --no-deps mne-bids

This approach ensures that the installation of MNE-BIDS doesn't alter any
other packages in your existing ``conda`` environment.

Alternatively, you may wish to take advantage of the fact that the
``mne-bids`` package on ``conda-forge`` in fact depends on ``mne``,
meaning that a "full" installation of ``mne-bids`` (i.e., including its
dependencies) will provide you with a working copy of of both ``mne`` and
``mne-bids`` at once:

.. code-block:: bash

   conda create --name mne --channel conda-forge mne-bids

After activating the environment, you should be ready to use MNE-BIDS:

.. code-block:: bash

   conda activate mne
   python -c 'import mne_bids; print(mne_bids.__version__)'
