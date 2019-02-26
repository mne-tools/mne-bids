"""Temporary fixes/workarounds for bugs in upstream code."""

# Authors: Mainak Jas <mainakjas@gmail.com>

import shutil
import os.path as op


def _copytree(src, dst, **kwargs):
    """See: https://github.com/jupyterlab/jupyterlab/pull/5150."""
    try:
        shutil.copytree(src, dst, **kwargs)
    except shutil.Error as error:
        # `copytree` throws an error if copying to + from NFS even though
        # the copy is successful (see https://bugs.python.org/issue24564)
        if '[Errno 22]' not in str(error) or not op.exists(dst):
            raise
