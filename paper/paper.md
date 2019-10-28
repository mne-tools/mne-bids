---
title: 'MNE-BIDS'
tags:
  - Python
  - BIDS
  - MNE
  - neuroimaging
  - MEG
  - EEG
authors:
  - affiliation: 1
    name: Stefan Appelhoff
    orcid: 0000-0001-8002-0877
  - affiliation: 2
    name: Mainak Jas
    orcid: 0000-0002-3199-9027    
  - affiliation: 3
    name: monkeyman192
    orcid: xxxx-xxxx-xxxx-xxxx
  - affiliation: 4
    name: Teon Brooks
    orcid: xxxx-xxxx-xxxx-xxxx
  - affiliation: 5
    name: wmvanvliet
    orcid:
  - affiliation: 6
    name: romquentin
    orcid: xxxx-xxxx-xxxx-xxxx
  - affiliation: 7
    name: Chris Holdgraf
    orcid: 0000-0002-2391-0678
  - affiliation: 8
    name: dnacombo
    orcid: xxxx-xxxx-xxxx-xxxx
  - affiliation: 9
    name: ezemikulan
    orcid: xxxx-xxxx-xxxx-xxxx
  - affiliation: 10
    name: ktavabi
    orcid: xxxx-xxxx-xxxx-xxxx
  - affiliation: 11
    name: hoechenberger
    orcid: xxxx-xxxx-xxxx-xxxx    
  - affiliation: 12
    name: dominikwelke
    orcid: xxxx-xxxx-xxxx-xxxx          
  - affiliation: 13
    name: cbrnr
    orcid: xxxx-xxxx-xxxx-xxxx     
  - affiliation: 14
    name: alexrockhill
    orcid: xxxx-xxxx-xxxx-xxxx
  - affiliation: 15
    name: Alexandre Gramfort
    orcid: xxxx-xxxx-xxxx-xxxx                     
affiliations:
- index: 1
  name: Center for Adaptive Rationality, Max Planck Institute for Human Development, Berlin, Germany
- index: 2
  name: Télécom ParisTech, France  
- index: 3
  name:
- index: 4
  name:
- index: 5
  name:
- index: 6
  name:
- index: 7
  name: University of California at Berkeley
- index: 8
  name:
- index: 9
  name:
- index: 10
  name:
- index: 11
  name:
- index: 12
  name:
- index: 13
  name:
- index: 14
  name:
- index: 15
  name:
date: 27 October 2019
bibliography: paper.bib
---

# Summary

With the emergence of the Brain Imaging Data Structure
(``BIDS``; [@Gorgolewski2016]) in 2016, the scientific community received a
standard to organize and share data in the broad domain of neuroscience.

Originally limited to MRI data types, BIDS is continuously being extended and
by now also supports other neuroimaging modalities such as MEG [@Niso2018],
EEG [@Pernet2019], and iEEG [@Holdgraf2019].

BIDS prescribes how complex experiment data should be structured and which
metadata should be encoded next to the raw data. This set of guidelines
allows users to reap several benefits, such as:

1. **sharing data within a lab and between labs:** Through reading the BIDS
   specification, users are empowered to understand any BIDS dataset
   without requiring specialist knowledge.
1. **data validation:** Dedicated software can check whether the rules of
   BIDS are being followed in a dataset, and alert users to missing or broken
   data (see
   [bids-validator](https://github.com/bids-standard/bids-validator)).
1. **automatic data handling and analysis pipelines:** With a perfect
   knowledge about which data to expect and where to search for it, many
   common workflows in data handling and analysis can be simplified or
   automated through specialized software.

To make use of these benefits however, datasets first must be converted to
BIDS format. Furthermore, already existing data analysis software must be
extended to be aware of a BIDS format and how to properly use it.

This is where ``MNE-BIDS`` steps in: Harnessing BIDS and the data analysis
software MNE-Python [@Agramfort2013], it is the goal of ``MNE-BIDS`` to
automatically organize raw MEG and EEG data into BIDS format and to facilitate
their analysis.

Starting with, for example, a single directory full of datafiles with arbitrary
naming, ``MNE-BIDS`` can be used to extract present meta data, reorganize the
files into BIDS format, and write additional meta data. Moreover,
``MNE-BIDS`` supports conversion from raw data formats that are not BIDS
compatible into permissible formats. As a result, users can easily convert
their datasets to BIDS in a matter of minutes, rather than after hours of
manual labour. All conversions performed by MNE-BIDS are validated with the
bids-validator for a maximum certainty that the results hold up to the BIDS
format.

Next to this core functionality, ``MNE-BIDS`` is continuously being extended
to facilitate the analysis of BIDS formatted data. To name some features, it is
possible to read a raw BIDS dataset and obtain a Python object, ready for
analyis with MNE-Python, or to save a T1 weighted anatomical MRI image
alongside the MEG or EEG data and apply an automatic defacing algorithm for
anonymization purposes

Users can easily install ``MNE-BIDS`` on all platforms via `pip` and `conda`
and its functionality is continuously tested on Windows and Linux.
Next to three core dependencies for scientific computation (`numpy`, `scipy`),
and handling of MEG and EEG data (`mne`), ``MNE-BIDS`` has only minimal
dependencies, all of which are optional. The API of the package is stable and
extensively documented and explained in examples (https://mne.tools/mne-bids/).

As of writing, ``MNE-BIDS`` has received code contributions from 15
contributors and its' user base is steadily growing. Code development is
active and users are typically receiving support within a few hours of opening
an issue on our dedicated issue tracker.

MNE-BIDS is used in automated analysis pipelines such as the
MNE-study-template (https://github.com/mne-tools/mne-study-template) and
several large institutions such as Martinos and "insert Matt's institution" have started to use MNE-BIDS.

The developer team is excited to improve the state of the art in data handling
and looking forward to welcoming new contributors and users.

# Acknowledgements

MNE-BIDS development is partly supported by XYZ.

# References
