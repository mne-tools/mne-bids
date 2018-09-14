:orphan:

.. _whats_new:

What's new?
===========

.. currentmodule:: mne_bids

.. _current:

Current
-------

Changelog
~~~~~~~~~

- Update :func:`mne_bids.raw_to_bids` to work for KIT and BTi systems by `Teon Brooks`_ (`#16 <https://github.com/mne-tools/mne-bids/pull/16/files>`_)
- Add support for iEEG and add :func:`mne_bids.make_bids_folders` and :func:`mne_bids.make_bids_folders` by `Chris Holdgraf`_ (`#28 <https://github.com/mne-tools/mne-bids/pull/28>`_ and `#37 <https://github.com/mne-tools/mne-bids/pull/37>`_)
- Add command line interface by `Teon Brooks`_ (`#31 <https://github.com/mne-tools/mne-bids/pull/31>`_)
- Add :func:`mne_bids.utils.print_dir_tree` for visualizing directory structures by `Stefan Appelhoff`_ (`#55 <https://github.com/mne-tools/mne-bids/pull/55>`_)
- Automatically generate participants.tsv by `Matt Sanderson`_ (`#70 <https://github.com/mne-tools/mne-bids/pull/70>`_)

Bug
~~~

- Correctly handle the case when measurement date is not available by `Mainak Jas`_ (`#23 <https://github.com/mne-tools/mne-bids/pull/23>`_)
- Fix counting of miscellaneous channels by `Matt Sanderson`_ (`#49 <https://github.com/mne-tools/mne-bids/pull/49>`_)
- Fix ordering of columns in scans.tsv by `Matt Sanderson`_ (`#68 <https://github.com/mne-tools/mne-bids/pull/68>`_)
- Fix bug in how artificial trigger channel STI014 is handled in channels.tsv for KIT systems by `Matt Sanderson`_ (`#72 <https://github.com/mne-tools/mne-bids/pull/72>`_)
- Fix channel types for KIT system in channels.tsv by `Matt Sanderson`_ (`#76 <https://github.com/mne-tools/mne-bids/pull/76>`_)

.. _Mainak Jas: https://perso.telecom-paristech.fr/mjas/
.. _Teon Brooks: http://teonbrooks.github.io/
.. _Chris Holdgraf: https://bids.berkeley.edu/people/chris-holdgraf
.. _Matt Sanderson: http://github.com/monkeyman192
.. _Stefan Appelhoff: http://stefanappelhoff.com/