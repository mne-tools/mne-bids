---
title: "MNE-BIDS: Organizing electrophysiological data into the BIDS format and facilitating their analysis"
tags:
  - Python
  - BIDS
  - MNE
  - neuroimaging
  - MEG
  - EEG
  - iEEG
authors:
  - affiliation: 1
    name: Stefan Appelhoff
    orcid: 0000-0001-8002-0877
  - affiliation: 2
    name: Matthew Sanderson
    orcid: 0000-0002-4645-8979
  - affiliation: 3
    name: Teon L Brooks
    orcid: 0000-0001-7344-3230
  - affiliation: 4
    name: Marijn van Vliet
    orcid: 0000-0002-6537-6899
  - affiliation: 5
    name: Romain Quentin
    orcid: 0000-0001-7659-3605
  - affiliation: 6
    name: Chris Holdgraf
    orcid: 0000-0002-2391-0678
  - affiliation: 7
    name: Maximilien Chaumon
    orcid: 0000-0001-9664-8861
  - affiliation: 8
    name: Ezequiel Mikulan
    orcid: 0000-0001-7259-6120
  - affiliation: 9
    name: Kambiz Tavabi
    orcid: 0000-0003-1881-892X
  - affiliation: 10
    name: Richard Höchenberger
    orcid: 0000-0002-0380-4798
  - affiliation: 11
    name: Dominik Welke
    orcid: 0000-0002-5529-1998
  - affiliation: 12
    name: Clemens Brunner
    orcid: 0000-0002-6030-2233
  - affiliation: 13
    name: Alexander P Rockhill
    orcid: 0000-0003-3868-7453
  - affiliation: 9
    name: Eric Larson
    orcid: 0000-0003-4782-5360
  - affiliation: 14
    name: Alexandre Gramfort
    orcid: 0000-0001-9791-4404
  - affiliation: 15
    name: Mainak Jas
    orcid: 0000-0002-3199-9027
affiliations:
- index: 1
  name: Center for Adaptive Rationality, Max Planck Institute for Human Development, Berlin, Germany
- index: 2
  name: Department of Cognitive Sciences, Macquarie University, Sydney, Australia
- index: 3
  name: Mozilla
- index: 4
  name: Department of Neuroscience and Biomedical Engineering, Aalto University, Espoo, Finland
- index: 5
  name: Human Cortical Physiology and Neurorehabilitation Section, NINDS, NIH, Bethesda, Maryland 20892
- index: 6
  name: UC Berkeley, Project Jupyter
- index: 7
  name: Institut du cerveau et de la moelle épinière (ICM), Paris, France
- index: 8
  name: Department of Biomedical and Clinical Sciences 'L. Sacco', University of Milan, Milan, Italy
- index: 9
  name: Institute for Learning and Brain Sciences, University of Washington, Seattle, WA, USA
- index: 10
  name: Institute of Neuroscience and Medicine (INM-3), Research Center Jülich, Germany
- index: 11
  name: Max-Planck-Institute for Empirical Aesthetics, Frankfurt a.M., Germany
- index: 12
  name: Institute of Psychology, University of Graz, Austria
- index: 13
  name: University of Oregon, Eugene OR, USA
- index: 14
  name: Université Paris-Saclay, Inria, CEA, Palaiseau, France
- index: 15
  name: Athinoula A. Martinos Center for Biomedical Imaging, Massachusetts General Hospital, Charlestown, MA, USA
date: 16 December 2019
bibliography: paper.bib
---

# Summary

The development of the Brain Imaging Data Structure
[BIDS; @Gorgolewski2016] gave the neuroscientific community
a standard to organize and share data.
BIDS prescribes file naming conventions and a folder structure to
store data in a set of already existing file formats.
Next to rules about organization of the data itself, BIDS provides standardized
templates to store associated metadata in the form of Javascript Object
Notation (JSON) and tab separated value (TSV) files.
It thus facilitates data sharing, eases metadata querying, and enables
automatic data analysis pipelines.
BIDS is a rich system to curate, aggregate, and annotate neuroimaging
databases.

While BIDS was originally intended for magnetic resonance imaging (MRI)
data, it has extensions for other data modalities including: magnetoencephalography
[MEG; @Niso2018], electroencephalography [EEG; @Pernet2019], and
intracranial encephalography [iEEG; @Holdgraf2019].
Software packages analyzing MEG, EEG, and iEEG are
now starting to support data organized using the BIDS standard, thereby
becoming "BIDS compatible".
Within the Python ecosystem, ``MNE-Python`` [@Agramfort2013] is a major
software package for electrophysiology data analysis, and extending its
functionality to support BIDS would be a great benefit for its growing user
base.
For this reason, we developed a dedicated Python software package *``MNE-BIDS`` with
the goal of providing a programmable interface for BIDS datasets in
electrophysiology with ``MNE-Python``*.
``MNE-BIDS`` allows users to re-organize data into BIDS formats, store
associated metadata after anonymization, extract information necessary for
preprocessing, and read the data into ``MNE-Python`` objects,
ready for source localization.

Starting with a single directory full of data files with arbitrary
names, ``MNE-BIDS`` can be used to extract existing metadata, reorganize the
files into the BIDS format, and write additional metadata.
All the conversion routines are thoroughly tested by running the output through
the [BIDS validator](https://github.com/bids-standard/bids-validator).
Moreover, ``MNE-BIDS`` supports converting data formats that are not BIDS
compatible into permissible formats.
These utilities allow users to easily convert their datasets to BIDS in a
matter of minutes, rather than hours of manual labour.

In addition to this core functionality, ``MNE-BIDS`` is continuously being
extended to facilitate the analysis of BIDS formatted data.
Some features include: reading a BIDS dataset as a set of Python objects for
analysis with ``MNE-Python``,
defacing T1-weighted anatomical MRI images to anonymize data and facilitate sharing,
and saving anatomical landmark coordinates to enable
coregistration between the MEG/EEG and MRI data, which is necessary for
computation of forward and inverse solutions.

Users can easily install ``MNE-BIDS`` on all major platforms via `pip` and
`conda`, and its functionality is continuously tested on Windows, macOS, and
Linux.
Other than the core dependencies for scientific computing (`numpy`, `scipy`)
and handling of MEG/EEG data (`mne`), ``MNE-BIDS`` has minimal dependencies,
all of which are optional.
The Application Programming Interface (API) of the package is stable and
extensively documented and explained in examples
([https://mne.tools/mne-bids/](https://mne.tools/mne-bids/)).
In addition, a command-line interface is provided that allows non-Python
users to benefit from the core functionality.

As of writing, ``MNE-BIDS`` has received code contributions from 15
contributors and its user base is steadily growing.
Code development is
[active](https://github.com/mne-tools/mne-bids/graphs/commit-activity) and the
developer team is committed to provide timely support for issues opened on the
GitHub issue tracker.

``MNE-BIDS`` is used as a dependency in several other software packages such as
the [MNE-study-template](https://github.com/mne-tools/mne-study-template), an
automated pipeline for group analysis with MNE [@Mainak2018], and
[Biscuit](https://github.com/Macquarie-MEG-Research/Biscuit), a graphical
user interface to format BIDS data.
Lastly, several large institutions have adopted ``MNE-BIDS`` for their
workflows such as the Martinos Center for Biomedical Imaging.

The developer team is excited to improve the state of the art in data handling
and looking forward to welcoming new contributors and users.

# Acknowledgements

``MNE-BIDS`` development is partly supported by
the Academy of Finland (grant 310988),
NIH NINDS R01-NS104585,
ERC Starting Grant SLAB ERC-YStG-676943,
ANR meegBIDS.fr,
BRAIN Initiative and National Institute of Mental Health (1R24MH114705),
the Bezos Family Foundation,
the Simms Mann Foundation,
and
the Google Summer of Code 2019.

# References
