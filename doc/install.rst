:orphan:

Installation
============

MNE-BIDS is included in the `official MNE-Python installers <https://mne.tools/stable/install/installers.html>`_.

If you want to install MNE-BIDS manually instead, please continue reading.

Dependencies
------------

Required:

* ``mne`` (>=1.2)
* ``numpy`` (>=1.18.1)
* ``scipy`` (>=1.4.1, or >=1.5.0 for certain operations with EEGLAB data)
* ``setuptools`` (>=46.4.0)

Optional:

* ``nibabel`` (>=2.5, for processing MRI data)
* ``pybv`` (>=0.7.5, for writing BrainVision data)
* ``pymatreader`` (>=0.0.30 , for operations with EEGLAB data)
* ``matplotlib`` (>=3.1.0, for using the interactive data inspector)
* ``pandas`` (>=1.0.0, for generating event statistics)
* ``EDFlib-Python`` (>=1.0.6, for writing EDF data)

We recommend the `Anaconda <https://www.anaconda.com/download/>`_ Python distribution.
We require that you **use Python 3.8 or higher**.
You may choose to install ``mne-bids``
`via pip <#installation-via-pip>`_ or
`via conda <#installation-via-conda>`_.

Installation via pip
--------------------

To install MNE-BIDS including all dependencies required to use all features,
simply run:

.. code-block:: bash

   pip install --upgrade mne-bids[full]

This ``pip`` command will also work if you want to upgrade if a newer version
of ``mne-bids`` is available.

If you don't require advanced features like interactive visual data inspection,
you may also install a basic version of ``mne-bids`` via

.. code-block:: bash

   pip install --upgrade mne-bids

If you want to install a snapshot of the current development version, run:

.. code-block:: bash

   pip install --upgrade https://api.github.com/repos/mne-tools/mne-bids/zipball/main

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
all that's left to do is to install ``mne-bids`` without its dependencies, as
they've already been installed during the MNE-Python installation process.

Activate the correct ``conda`` environment and install ``mne-bids``:

.. code-block:: bash

   conda activate mne
   conda install --channel conda-forge --no-deps mne-bids

This approach ensures that the installation of ``mne-bids`` doesn't alter any
other packages in your existing ``conda`` environment.

Alternatively, you may wish to take advantage of the fact that the
``mne-bids`` package on ``conda-forge`` in fact depends on ``mne``,
meaning that a "full" installation of ``mne-bids`` (i.e., including its
dependencies) will provide you with a working copy of of both ``mne`` and
``mne-bids`` at once:

.. code-block:: bash

   conda create --name mne --channel conda-forge mne-bids

After activating the environment, you should be ready to use ``mne-bids``:

.. code-block:: bash

   conda activate mne
   python -c 'import mne_bids; print(mne_bids.__version__)'
