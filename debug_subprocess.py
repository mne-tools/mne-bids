import sys
from queue import Queue, Empty
from contextlib import contextmanager
import subprocess
import inspect
from threading import Thread
import platform
# import os


# helpers to get function arguments
def _get_args(function, varargs=False):
    params = inspect.signature(function).parameters
    args = [key for key, param in params.items()
            if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)]
    if varargs:
        varargs = [param.name for param in params.values()
                   if param.kind == param.VAR_POSITIONAL]
        if len(varargs) == 0:
            varargs = None
        return args, varargs
    else:
        return args


def _enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)


@contextmanager
def running_subprocess(command, after="wait", verbose=None, *args, **kwargs):
    """Context manager to do something with a command running via Popen.

    Parameters
    ----------
    command : list of str | str
        Command to run as subprocess (see :class:`python:subprocess.Popen`).
    after : str
        Can be:

        - "wait" to use :meth:`~python:subprocess.Popen.wait`
        - "communicate" to use :meth:`~python.subprocess.Popen.communicate`
        - "terminate" to use :meth:`~python:subprocess.Popen.terminate`
        - "kill" to use :meth:`~python:subprocess.Popen.kill`

    %(verbose)s
    *args, **kwargs : arguments
        Additional arguments to pass to subprocess.Popen.

    Returns
    -------
    p : instance of Popen
        The process.
    """
    # _validate_type(after, str, 'after')
    # _check_option('after', after, ['wait', 'terminate', 'kill', 'communicate'])
    for stdxxx, sys_stdxxx in (['stderr', sys.stderr], ['stdout', sys.stdout]):
        if stdxxx not in kwargs:
            kwargs[stdxxx] = subprocess.PIPE

    # Check the PATH environment variable. If run_subprocess() is to be called
    # frequently this should be refactored so as to only check the path once.
    # env = kwargs.get('env', os.environ)
    # if any(p.startswith('~') for p in env['PATH'].split(os.pathsep)):
    #     warn('Your PATH environment variable contains at least one path '
    #          'starting with a tilde ("~") character. Such paths are not '
    #          'interpreted correctly from within Python. It is recommended '
    #          'that you use "$HOME" instead of "~".')
    if isinstance(command, str):
        command_str = command
    else:
        command = [str(s) for s in command]
        command_str = ' '.join(s for s in command)
    # logger.info("Running subprocess: %s" % command_str)
    print("Running subprocess: %s" % command_str)
    try:
        p = subprocess.Popen(command, *args, **kwargs)
    except Exception:
        if isinstance(command, str):
            command_name = command.split()[0]
        else:
            command_name = command[0]
        # logger.error('Command not found: %s' % command_name)
        print('Command not found: %s' % command_name)
        raise
    try:
        yield p
    finally:
        getattr(p, after)()
        p.wait()


def run_subprocess(command, return_code=False, verbose=None, *args, **kwargs):
    """Run command using subprocess.Popen.

    Run command and wait for command to complete. If the return code was zero
    then return, otherwise raise CalledProcessError.
    By default, this will also add stdout= and stderr=subproces.PIPE
    to the call to Popen to suppress printing to the terminal.

    Parameters
    ----------
    command : list of str | str
        Command to run as subprocess (see subprocess.Popen documentation).
    return_code : bool
        If True, return the return code instead of raising an error if it's
        non-zero.

        .. versionadded:: 0.20
    %(verbose)s
    *args, **kwargs : arguments
        Additional arguments to pass to subprocess.Popen.

    Returns
    -------
    stdout : str
        Stdout returned by the process.
    stderr : str
        Stderr returned by the process.
    code : int
        The return code, only returned if ``return_code == True``.
    """
    all_out = ''
    all_err = ''
    # non-blocking adapted from https://stackoverflow.com/questions/375427/non-blocking-read-on-a-subprocess-pipe-in-python#4896288  # noqa: E501
    out_q = Queue()
    err_q = Queue()
    with running_subprocess(command, *args, **kwargs) as p:
        out_t = Thread(target=_enqueue_output, args=(p.stdout, out_q))
        err_t = Thread(target=_enqueue_output, args=(p.stderr, err_q))
        out_t.daemon = True
        err_t.daemon = True
        out_t.start()
        err_t.start()
        while True:
            do_break = p.poll() is not None
            # read all current lines without blocking
            while True:
                try:
                    out = out_q.get(timeout=0.01)
                except Empty:
                    break
                else:
                    out = out.decode('utf-8')
                    # logger.info(out)
                    print(out)
                    all_out += out
            while True:
                try:
                    err = err_q.get(timeout=0.01)
                except Empty:
                    break
                else:
                    err = err.decode('utf-8')
                    # logger.warning(err)
                    print(err)
                    all_err += err
            if do_break:
                break
    output = (all_out, all_err)

    if return_code:
        output = output + (p.returncode,)
    elif p.returncode:
        print(output)
        err_fun = subprocess.CalledProcessError.__init__
        if 'output' in _get_args(err_fun):
            raise subprocess.CalledProcessError(p.returncode, command, output)
        else:
            raise subprocess.CalledProcessError(p.returncode, command)

    return output


if __name__ == '__main__':
    if platform.system() == 'Windows':
        run_subprocess(['node', 'bids-validator', '--version'], shell=True)
    else:
        run_subprocess(['bids-validator', '--version'], shell=False)
