:orphan:

.. _whats_new:

.. currentmodule:: mne_bids

What's new?
===========

.. _changes_0_16:

Version 0.16 (unreleased)
-------------------------

👩🏽‍💻 Authors
~~~~~~~~~~~~~~~

The following authors contributed for the first time. Thank you so much! 🤩

* `Aaron Earle-Richardson`_
* `Amaia Benitez`_
* `Kaare Mikkelsen`_
* `Simon Kern`_
* `Thomas Hartmann`_
* `William Turner`_

The following authors had contributed before. Thank you for sticking around! 🤘

* `Daniel McCloy`_
* `Eric Larson`_
* `Stefan Appelhoff`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

🚀 Enhancements
^^^^^^^^^^^^^^^

- :meth:`mne_bids.BIDSPath.match()` and :func:`mne_bids.find_matching_paths` now have additional parameters ``ignore_json`` and ``ignore_nosub``, to give users more control over which type of files are matched, by `Kaare Mikkelsen`_ (:gh:`1281`)
- :func:`mne_bids.write_raw_bids()` can now handle event metadata as a pandas DataFrame, by `Thomas Hartmann`_ (:gh:`1285`)

🧐 API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :func:`mne_bids.read_raw_bids` no longer warns about unit changes in channels upon reading, as that information is taken from ``channels.tsv`` and judged authorative, by `Stefan Appelhoff`_ (:gh:`1282`)
- MEG OPM channels are now experimentally included, by `Amaia Benitez`_ (:gh:`1222`)
- :func:`mne_bids.mark_channels` will no longer create a ``status_description`` column filled with ``n/a`` in the ``channels.tsv`` file, by `Stefan Appelhoff`_ (:gh:`1293`)
- :func:`mark_channels(..., ch_names=[]) <mne_bids.mark_channels>` now raises a deprecation warning, and in future its behavior will change from marking *all* channels to marking *no* channels; to avoid the warning use ``mark_channels(..., ch_names="all")``, by `Daniel McCloy`_ (:gh:`1307`)


🛠 Requirements
^^^^^^^^^^^^^^^

- MNE-BIDS now requires Python 3.10 or higher.

🪲 Bug fixes
^^^^^^^^^^^^

- Writing MEGIN data with MNE channel types `chpi` will now map to BIDS type HLU by `Simon Kern`_ (:gh:`1325`)
- When anonymizing the date of a recording, MNE-BIDS will no longer error during `~mne_bids.write_raw_bids` if passing a `~mne.io.Raw` instance to ``empty_room``, by `Daniel McCloy`_ (:gh:`1270`)
- Dealing with alphanumeric ``sub`` entity labels is now fixed for :func:`~mne_bids.write_raw_bids`, by `Aaron Earle-Richardson`_ (:gh:`1291`)
- When processing subject_info data that MNE Python imports as numpy arrays with only one item, MNE-BIDS now unpacks these, resulting in a correct participants.tsv, by `Thomas Hartmann`_ (:gh:`1310`)
- Fixed broken links in examples 7 and 8, by `William Turner`_ (:gh:`1316`)
- All valid extensions for ``README`` files are now accepted. This prevents an extra ``README`` file being created, when one with a ``.txt``, ``.md``, or ``.rst`` extension is already present. By `Thomas Hartmann`_ (:gh:`1318`)
- A warning was given if no events were provided but the task was starting with 'rest' as recommended by `Simon Kern`_ (:gh:`1327`)

⚕️ Code health
^^^^^^^^^^^^^^

- Keep MNE-BIDS up to date with recent changes on participant birthday date handling in MNE-Python, by `Eric Larson`_ (:gh:`1278`)
- Make rules for linting more strict, make quality assessment exceptions less permissive, by `Stefan Appelhoff`_ (:gh:`1283`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.rst
