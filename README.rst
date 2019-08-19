

.. image:: https://badges.gitter.im/mne-tools/mne-bids.svg
   :target: https://gitter.im/mne-tools/mne-bids?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
   :alt: Gitter

.. image:: https://travis-ci.org/mne-tools/mne-bids.svg?branch=master
   :target: https://travis-ci.org/mne-tools/mne-bids
   :alt: Travis

.. image:: https://ci.appveyor.com/api/projects/status/d4u70pht341cwqxb/branch/master?svg=true
   :target: https://ci.appveyor.com/project/mne-tools/mne-bids/branch/master
   :alt: Appveyor


.. image:: https://codecov.io/gh/mne-tools/mne-bids/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/mne-tools/mne-bids
   :alt: codecov


.. image:: https://circleci.com/gh/mne-tools/mne-bids.svg?style=svg
   :target: https://circleci.com/gh/mne-tools/mne-bids
   :alt: CircleCi


.. image:: https://pepy.tech/badge/mne-bids
   :target: https://pepy.tech/project/mne-bids
   :alt: Downloads


MNE-BIDS
========

This is a repository for creating
`BIDS <http://bids.neuroimaging.io/>`_\ -compatible datasets with
`MNE <https://mne.tools/stable/index.html>`_.

The documentation can be found under the following links:

- for the `stable release <https://mne.tools/mne-bids/>`_
- for the `latest (development) version <https://circleci.com/api/v1.1/project/github/mne-tools/mne-bids/latest/artifacts/0/html/index.html?branch=master>`_

Installation
------------

We recommend the `Anaconda <https://www.anaconda.com/download/>`_ Python
distribution. We require that you use Python 3.
Besides ``numpy`` and ``scipy`` (which are included in the standard Anaconda
installation), you will need to install the most recent version of ``MNE``
using the ``pip`` tool:

.. code-block:: bash

   $ pip install -U mne


Then install ``mne-bids``\ :

.. code-block:: bash

   $ pip install -U mne-bids


These ``pip`` commands also work if you want to upgrade if a newer version of
``mne-bids`` is available. If you do not have administrator privileges on the
computer, use the ``--user`` flag with ``pip``.

To check if everything worked fine, the following command should not give any
error messages:

.. code-block:: bash

   $ python -c 'import mne_bids'

For full functionality of ``mne-bids``, you will also need to ``pip install``
the following packages:

- ``nibabel``, for interacting with MRI data

If you want to use the latest development version of ``mne-bids``, use the
following command:

.. code-block:: bash

   $ pip install https://api.github.com/repos/mne-tools/mne-bids/zipball/master

Quickstart
----------

Currently, we support writing of BIDS datasets for MEG and EEG. Support for
iEEG is experimental at the moment.

.. code:: python

    >>> from mne import io
    >>> from mne_bids import write_raw_bids
    >>> raw = io.read_raw_fif('my_old_file.fif')
    >>> write_raw_bids(raw, 'sub-01_ses-01_run-05', output_path='./bids_dataset')

Command Line Interface
----------------------

In addition to ``import mne_bids``, you can use the command line interface.
Simply type ``mne_bids`` in your command line and press enter, to see the
accepted commands. Then type ``mne_bids <command> --help``, where ``<command>``
is one of the accepted commands, to get more information about that
``<command>``.

Example:

.. code-block:: bash

  $ mne_bids raw_to_bids --subject_id sub01 --task rest --raw data.edf --output_path new_path

Bug reports
-----------

Use the `github issue tracker <https://github.com/mne-tools/mne-bids/issues>`_
to report bugs.

Cite
----

If you use ``mne-bids`` in your work, please cite one of the following papers,
depending on which modality you used:

`MEG <http://doi.org/10.1038/sdata.2018.110>`_
##############################################

.. code-block:: Text

   Niso, G., Gorgolewski, K. J., Bock, E., Brooks, T. L., Flandin, G., Gramfort, A.,
   Henson, R. N., Jas, M., Litvak, V., Moreau, J., Oostenveld, R., Schoffelen, J.,
   Tadel, F., Wexler, J., Baillet, S. (2018). MEG-BIDS, the brain imaging data
   structure extended to magnetoencephalography. Scientific Data, 5, 180110.
   http://doi.org/10.1038/sdata.2018.110


`EEG <https://doi.org/10.1038/s41597-019-0104-8>`_
##################################################

.. code-block:: Text

   Pernet, C. R., Appelhoff, S., Gorgolewski, K. J., Flandin, G.,
   Phillips, C., Delorme, A., Oostenveld, R. (2019). EEG-BIDS, an extension
   to the brain imaging data structure for electroencephalography. Scientific
   Data, 6, 103. https://doi.org/10.1038/s41597-019-0104-8


`iEEG <https://doi.org/10.1038/s41597-019-0105-7>`_
###################################################

.. code-block:: Text

   Holdgraf, C., Appelhoff, S., Bickel, S., Bouchard, K., D'Ambrosio, S.,
   David, O., … Hermes, D. (2019). iEEG-BIDS, extending the Brain Imaging Data
   Structure specification to human intracranial electrophysiology. Scientific
   Data, 6, 102. https://doi.org/10.1038/s41597-019-0105-7
