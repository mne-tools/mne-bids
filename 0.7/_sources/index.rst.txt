What is MNE-BIDS?
=================

MNE-BIDS is a Python package that allows you to read and write
`BIDS <https://bids.neuroimaging.io/>`_\ -compatible datasets with the help of
`MNE-Python <https://mne.tools/stable/index.html>`_.

.. image:: http://mne.tools/mne-bids/assets/MNE-BIDS.png
   :alt: Schematic: From raw data to BIDS using MNE-BIDS

Why?
----
MNE-BIDS links BIDS and MNE-Python with the goal to make your analyses faster
to code, more robust, and facilitate data and code sharing with co-workers
and collaborators.


What is this BIDS thing, anyway?
--------------------------------
BIDS, the Brain Imaging Data Structure, is a standard that describes how to
organize neuroimaging and electrophysiological data. In particular, it defines:

- which file formats to use
- how to name your files
- where to place your files within a directory structure
- what additional metadata to store

The complete set of rules is written down in the
`BIDS specification <https://bids-specification.readthedocs.io/>`_.


Supported file formats
----------------------

Currently, we support all file formats that are in the BIDS specification for MEG, EEG, and iEEG data.
We also support a range of additional manufacturer formats to facilitate converting them to their BIDS-recommended
formats. For example, if you have a Nihon Kohden file, you can read that file using MNE-Python, and
then use MNE-BIDS convert it to the BrainVision format and store it according to BIDS.

Citing MNE-BIDS
---------------

If you use MNE-BIDS in your work, please cite our
`publication in JOSS <https://doi.org/10.21105/joss.01896>`_.

Appelhoff, S., Sanderson, M., Brooks, T., Vliet, M., Quentin, R., Holdgraf, C.,
Chaumon, M., Mikulan, E., Tavabi, K., Höchenberger, R., Welke, D., Brunner, C.,
Rockhill, A., Larson, E., Gramfort, A., & Jas, M. (2019). **MNE-BIDS:
Organizing electrophysiological data into the BIDS format and facilitating
their analysis.**
*Journal of Open Source Software,* 4:1896.
DOI: `10.21105/joss.01896 <https://doi.org/10.21105/joss.01896>`_

Please also cite one of the following papers to credit BIDS, depending on which
data type you used:

- `BIDS-MEG <http://doi.org/10.1038/sdata.2018.110>`_
- `BIDS-EEG <https://doi.org/10.1038/s41597-019-0104-8>`_
- `BIDS-iEEG <https://doi.org/10.1038/s41597-019-0105-7>`_


.. contents:: :local:
    :depth: 3