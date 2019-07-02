"""
=======================================================================
Save a T1 weighted MRI scan to BIDS, and encode a transformation matrix
=======================================================================

When working with MEEG data in the domain of source localization, we usually
have to deal with aligning several coordinate systems, such as the coordinate
systems of ...

- the head of a study participant
- the recording device (in the case of MEG)
- the anatomical MRI scan of a study participant

The process of aligning these frames is also called coregistration, and is
performed with the help of transformation matrices, called ``trans`` in MNE.

In this tutorial, we show how ``MNE-BIDS`` can be used to save a T1 weighted
MRI scan in BIDS format, and to encode all information of the ``trans`` object
in a BIDS compatible way.

Finally, we will automatically reproduce our ``trans`` object from a BIDS
directory.

See the documentation pages in the MNE docs for more information on
`source alignment and coordinate frames <mne_source_coords_>`_



"""
# Authors: Stefan Appelhoff <stefan.appelhoff@mailbox.org>
# License: BSD (3-clause)

###############################################################################
# We are importing everything we need for this example:
#
#
#
#
# .. LINKS
#
# .. _mne_source_coords:
#    https://www.martinos.org/mne/stable/auto_tutorials/source-modeling/plot_source_alignment.html  # noqa: E501
