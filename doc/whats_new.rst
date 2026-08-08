:orphan:

.. _whats_new:

.. currentmodule:: mne_bids

.. include:: authors.rst

What's new?
===========

.. _changes_0_20:

Version 0.20 (unreleased)
-------------------------

👩🏽‍💻 Authors
~~~~~~~~~~~~~~~

The following authors contributed for the first time. Thank you so much! 🤩

* `Daria Agafonova`_
* `Vincent Gao`_

The following authors had contributed before. Thank you for sticking around! 🤘

* `Bruno Aristimunha`_

Detailed list of changes
~~~~~~~~~~~~~~~~~~~~~~~~

🚀 Enhancements
^^^^^^^^^^^^^^^

- Clarify that the ``tracking_system`` parameter of :class:`mne_bids.BIDSPath` and the ``tracking_systems`` parameter of :func:`mne_bids.find_matching_paths` correspond to the Motion-BIDS ``tracksys`` entity, by `Daria Agafonova`_ (:gh:`1562`)
- Add :func:`mne_bids.read_epochs_bids` to read epoched BIDS recordings (``"RecordingType": "epoched"``) as :class:`mne.Epochs`; :func:`mne_bids.read_raw_bids` now raises a helpful error for such recordings, by `Bruno Aristimunha`_ (:gh:`1605`)

🧐 API and behavior changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- None yet

🛠 Requirements
^^^^^^^^^^^^^^^

- None yet

🪲 Bug fixes
^^^^^^^^^^^^

- Preserve the actual root of files returned by :meth:`mne_bids.BIDSPath.match`, avoiding duplicate paths and paths that do not exist when matching inside nested BIDS roots, by `Daria Agafonova`_ (:gh:`1637`)
- :func:`mne_bids.write_raw_bids` no longer raises a ``TypeError`` when writing ANT Neuro eego recordings (``.cnt``), by `Vincent Gao`_ (:gh:`1617`)

⚕️ Code health
^^^^^^^^^^^^^^

- Sped up writing of recordings with many channels by avoiding redundant per-channel work in :func:`mne_bids.write_raw_bids` (single-pass channel-type counting, cached coil-type lookup, and a fixed quadratic loop when writing BrainVision units), by `Stefan Appelhoff`_ (:gh:`1620`)

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`
