# This file is part of utils.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# Use of this source code is governed by a 3-clause BSD-style
# license that can be found in the LICENSE file.
#

"""Utilities for measuring execution time."""

from __future__ import annotations

__all__ = ["duration_from_timeMethod", "logInfo", "profile", "timeMethod", "time_this"]

import datetime
import functools
import logging
import socket
import sys
import time
import traceback
from collections.abc import Callable, Collection, Iterable, Iterator, MutableMapping
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any

from astropy import units as u
import psutil

from .introspection import find_outside_stacklevel
from .logging import LsstLoggers
from .usage import _get_current_rusage, get_current_mem_usage, get_peak_mem_usage

if TYPE_CHECKING:
    import cProfile


_LOG = logging.getLogger(__name__)


def _add_to_metadata(metadata: MutableMapping, name: str, value: Any) -> None:
    """Add a value to dict-like object, creating list as needed.

    The list grows as more values are added for that key.

    Parameters
    ----------
    metadata : `dict`-like, optional
        `dict`-like object that can store keys. Uses `add()` method if
        one is available, else creates list and appends value if needed.
    name : `str`
        The key to use in the metadata dictionary.
    value : Any
        Value to store in the list.
    """
    try:
        try:
            # PropertySet should always prefer LongLong for integers
            metadata.addLongLong(name, value)  # type: ignore
        except (TypeError, AttributeError):
            metadata.add(name, value)  # type: ignore
    except AttributeError:
        pass
    else:
        return

    # Fallback code where `add` is not implemented.
    if name not in metadata:
        metadata[name] = []
    metadata[name].append(value)


def logPairs(
    obj: Any,
    pairs: Collection[tuple[str, Any]],
    logLevel: int = logging.DEBUG,
    metadata: MutableMapping | None = None,
    logger: LsstLoggers | None = None,
    stacklevel: int | None = None,
) -> None:
    """Log ``(name, value)`` pairs to ``obj.metadata`` and ``obj.log``.

    Parameters
    ----------
    obj : `object`
        An object with one or both of these two attributes:

        - ``metadata`` a `dict`-like container for storing metadata.
          Can use the ``add(name, value)`` method if found, else will append
          entries to a list.
        - ``log`` an instance of `logging.Logger` or subclass.

        If `None`, at least one of ``metadata`` or ``logger`` should be passed
        or this function will do nothing.
    pairs : sequence
        A sequence of ``(name, value)`` pairs, with value typically numeric.
    logLevel : `int, optional
        Log level (an `logging` level constant, such as `logging.DEBUG`).
    metadata : `collections.abc.MutableMapping`, optional
        Metadata object to write entries to.  Ignored if `None`.
    logger : `logging.Logger` or `lsst.utils.logging.LsstLogAdapter`
        Log object to write entries to.  Ignored if `None`.
    stacklevel : `int`, optional
        The stack level to pass to the logger to adjust which stack frame
        is used to report the file information. If `None` the stack level
        is computed such that it is reported as the first package outside
        of the utils package. If a value is given here it is adjusted by
        1 to account for this caller.
    """
    if obj is not None:
        if metadata is None:
            with suppress(AttributeError):
                metadata = obj.metadata

        if logger is None:
            with suppress(AttributeError):
                logger = obj.log

    strList = []
    for name, value in pairs:
        if metadata is not None:
            _add_to_metadata(metadata, name, value)
        strList.append(f"{name}={value}")
    if logger is not None:
        # Want the file associated with this log message to be that
        # of the caller. This is expensive so only do it if we know the
        # message will be issued.
        timer_logger = logging.getLogger("timer." + logger.name)
        if timer_logger.isEnabledFor(logLevel):
            if stacklevel is None:
                stacklevel = find_outside_stacklevel("lsst.utils")
            else:
                # Account for the caller stack.
                stacklevel += 1
            timer_logger.log(logLevel, "; ".join(strList), stacklevel=stacklevel)


def logInfo(
    obj: Any,
    prefix: str,
    logLevel: int = logging.DEBUG,
    metadata: MutableMapping | None = None,
    logger: LsstLoggers | None = None,
    stacklevel: int | None = None,
) -> None:
    """Log timer information to ``obj.metadata`` and ``obj.log``.

    Parameters
    ----------
    obj : `object`
        An object with both or one these two attributes:

        - ``metadata`` a `dict`-like container for storing metadata.
          Can use the ``add(name, value)`` method if found, else will append
          entries to a list.
        - ``log`` an instance of `logging.Logger` or subclass.

        If `None`, at least one of ``metadata`` or ``logger`` should be passed
        or this function will do nothing.
    prefix : `str`
        Name prefix, the resulting entries are ``CpuTime``, etc.. For example
        `timeMethod` uses ``prefix = Start`` when the method begins and
        ``prefix = End`` when the method ends.
    logLevel : optional
        Log level (a `logging` level constant, such as `logging.DEBUG`).
    metadata : `collections.abc.MutableMapping`, optional
        Metadata object to write entries to, overriding ``obj.metadata``.
    logger : `logging.Logger` or `lsst.utils.logging.LsstLogAdapter`
        Log object to write entries to, overriding ``obj.log``.
    stacklevel : `int`, optional
        The stack level to pass to the logger to adjust which stack frame
        is used to report the file information. If `None` the stack level
        is computed such that it is reported as the first package outside
        of the utils package. If a value is given here it is adjusted by
        1 to account for this caller.

    Notes
    -----
    Logged items include:

    - ``Utc``: UTC date in ISO format (only in metadata since log entries have
      timestamps).
    - ``CpuTime``: System + User CPU time (seconds). This should only be used
        in differential measurements; the time reference point is undefined.
    - ``MaxRss``: maximum resident set size. Always in bytes.

    All logged resource information is only for the current process; child
    processes are excluded.

    The metadata will be updated with a ``__version__`` field to indicate the
    version of the items stored. If there is no version number it is assumed
    to be version 0.

    * Version 0: ``MaxResidentSetSize`` units are platform-dependent.
    * Version 1: ``MaxResidentSetSize`` will be stored in bytes.
    """
    if metadata is None and obj is not None:
        with suppress(AttributeError):
            metadata = obj.metadata

    if metadata is not None:
        # Log messages already have timestamps.
        if sys.version_info < (3, 11, 0):
            now = datetime.datetime.utcnow()
        else:
            now = datetime.datetime.now(datetime.UTC)
        utcStr = now.isoformat()
        _add_to_metadata(metadata, name=prefix + "Utc", value=utcStr)

        # Force a version number into the metadata.
        # v1: Ensure that max_rss field is always bytes.
        metadata["__version__"] = 1

        # Add node information.
        metadata["nodeName"] = socket.gethostname()
        metadata["nodeCpuCount"] = psutil.cpu_count()
    if stacklevel is not None:
        # Account for the caller of this routine not knowing that we
        # are going one down in the stack.
        stacklevel += 1

    usage = _get_current_rusage()
    logPairs(
        obj=obj,
        pairs=[(prefix + k[0].upper() + k[1:], v) for k, v in usage.dict().items()],
        logLevel=logLevel,
        metadata=metadata,
        logger=logger,
        stacklevel=stacklevel,
    )


def timeMethod(
    _func: Any | None = None,
    *,
    metadata: MutableMapping | None = None,
    logger: LsstLoggers | None = None,
    logLevel: int = logging.DEBUG,
) -> Callable:
    """Measure duration of a method.

    Parameters
    ----------
    _func : `~collections.abc.Callable` or `None`
        The method to wrap.
    metadata : `collections.abc.MutableMapping`, optional
        Metadata to use as override if the instance object attached
        to this timer does not support a ``metadata`` property.
    logger : `logging.Logger` or `lsst.utils.logging.LsstLogAdapter`, optional
        Logger to use when the class containing the decorated method does not
        have a ``log`` property.
    logLevel : `int`, optional
        Log level to use when logging messages. Default is `~logging.DEBUG`.

    Notes
    -----
    Writes various measures of time and possibly memory usage to the
    metadata; all items are prefixed with the function name.

    .. warning::

       This decorator only works with instance methods of any class
       with these attributes:

       - ``metadata``: an instance of `collections.abc.Mapping`. The ``add``
         method will be used if available, else entries will be added to a
         list.
       - ``log``: an instance of `logging.Logger` or subclass.

    Examples
    --------
    To use:

    .. code-block:: python

        import lsst.utils as utils
        import lsst.pipe.base as pipeBase
        class FooTask(pipeBase.Task):
            pass

            @utils.timeMethod
            def run(self, ...): # or any other instance method you want to time
                pass
    """

    def decorator_timer(func: Callable) -> Callable:
        @functools.wraps(func)
        def timeMethod_wrapper(self: Any, *args: Any, **keyArgs: Any) -> Any:
            # Adjust the stacklevel to account for the wrappers.
            # stacklevel 1 would make the log message come from this function
            # but we want it to come from the file that defined the method
            # so need to increment by 1 to get to the caller.
            stacklevel = 2
            logInfo(
                obj=self,
                prefix=func.__name__ + "Start",
                metadata=metadata,
                logger=logger,
                logLevel=logLevel,
                stacklevel=stacklevel,
            )
            try:
                res = func(self, *args, **keyArgs)
            finally:
                logInfo(
                    obj=self,
                    prefix=func.__name__ + "End",
                    metadata=metadata,
                    logger=logger,
                    logLevel=logLevel,
                    stacklevel=stacklevel,
                )
            return res

        return timeMethod_wrapper

    if _func is None:
        return decorator_timer
    else:
        return decorator_timer(_func)


@contextmanager
def time_this(
    log: LsstLoggers | None = None,
    msg: str | None = None,
    level: int = logging.DEBUG,
    prefix: str | None = "timer",
    args: Iterable[Any] = (),
    mem_usage: bool = False,
    mem_child: bool = False,
    mem_unit: u.Quantity = u.byte,
    mem_fmt: str = ".0f",
) -> Iterator[None]:
    """Time the enclosed block and issue a log message.

    Parameters
    ----------
    log : `logging.Logger`, optional
        Logger to use to report the timer message. The root logger will
        be used if none is given.
    msg : `str`, optional
        Context to include in log message.
    level : `int`, optional
        Python logging level to use to issue the log message. If the
        code block raises an exception the log message will include some
        information about the exception that occurred.
    prefix : `str`, optional
        Prefix to use to prepend to the supplied logger to
        create a new logger to use instead. No prefix is used if the value
        is set to `None`. Defaults to "timer".
    args : iterable of any
        Additional parameters passed to the log command that should be
        written to ``msg``.
    mem_usage : `bool`, optional
        Flag indicating whether to include the memory usage in the report.
        Defaults, to False.
    mem_child : `bool`, optional
        Flag indication whether to include memory usage of the child processes.
    mem_unit : `astropy.units.Unit`, optional
        Unit to use when reporting the memory usage. Defaults to bytes.
    mem_fmt : `str`, optional
        Format specifier to use when displaying values related to memory usage.
        Defaults to '.0f'.
    """
    if log is None:
        log = logging.getLogger()
    if prefix:
        log_name = f"{prefix}.{log.name}" if not isinstance(log, logging.RootLogger) else prefix
        log = logging.getLogger(log_name)

    start = time.time()

    if mem_usage and not log.isEnabledFor(level):
        mem_usage = False

    if mem_usage:
        current_usages_start = get_current_mem_usage()
        peak_usages_start = get_peak_mem_usage()

    errmsg = ""
    try:
        yield
    except BaseException as e:
        frame, lineno = list(traceback.walk_tb(e.__traceback__))[-1]
        errmsg = f"{e!r} @ {frame.f_code.co_filename}:{lineno}"
        raise
    finally:
        end = time.time()

        # The message is pre-inserted to allow the logger to expand
        # the additional args provided. Make that easier by converting
        # the None message to empty string.
        if msg is None:
            msg = ""

        # Convert user provided parameters (if any) to mutable sequence to make
        # mypy stop complaining when additional parameters will be added below.
        params = list(args) if args else []

        # Specify stacklevel to ensure the message is reported from the
        # caller (1 is this file, 2 is contextlib, 3 is user)
        params += (": " if msg else "", end - start)
        msg += "%sTook %.4f seconds"
        if errmsg:
            params += (f" (timed code triggered exception of {errmsg!r})",)
            msg += "%s"
        if mem_usage:
            current_usages_end = get_current_mem_usage()
            peak_usages_end = get_peak_mem_usage()

            current_deltas = [end - start for end, start in zip(current_usages_end, current_usages_start)]
            peak_deltas = [end - start for end, start in zip(peak_usages_end, peak_usages_start)]

            current_usage = current_usages_end[0]
            current_delta = current_deltas[0]
            peak_delta = peak_deltas[0]
            if mem_child:
                current_usage += current_usages_end[1]
                current_delta += current_deltas[1]
                peak_delta += peak_deltas[1]

            if not mem_unit.is_equivalent(u.byte):
                _LOG.warning("Invalid memory unit '%s', using '%s' instead", mem_unit, u.byte)
                mem_unit = u.byte

            msg += (
                f"; current memory usage: {current_usage.to(mem_unit):{mem_fmt}}"
                f", delta: {current_delta.to(mem_unit):{mem_fmt}}"
                f", peak delta: {peak_delta.to(mem_unit):{mem_fmt}}"
            )
        log.log(level, msg, *params, stacklevel=3)


@contextmanager
def profile(filename: str | None, log: LsstLoggers | None = None) -> Iterator[cProfile.Profile | None]:
    """Profile the enclosed code block and save the result to a file.

    Parameters
    ----------
    filename : `str` or `None`
        Filename to which to write profile (profiling disabled if `None` or
        empty string).
    log : `logging.Logger` or `lsst.utils.logging.LsstLogAdapter`, optional
        Log object for logging the profile operations.

    Yields
    ------
    prof : `cProfile.Profile` or `None`
        If profiling is enabled, the context manager returns the
        `cProfile.Profile` object (otherwise it returns `None`),
        which allows additional control over profiling.

    Examples
    --------
    You can obtain the `cProfile.Profile` object using the "as" clause, e.g.:

    .. code-block:: python

        with profile(filename) as prof:
            runYourCodeHere()

    The output cumulative profile can be printed with a command-line like:

    .. code-block:: bash

        python -c 'import pstats; \
            pstats.Stats("<filename>").sort_stats("cumtime").print_stats(30)'
    """
    if not filename:
        # Nothing to do
        yield None
        return
    from cProfile import Profile

    profile = Profile()
    if log is not None:
        log.info("Enabling cProfile profiling")
    profile.enable()
    yield profile
    profile.disable()
    profile.dump_stats(filename)
    if log is not None:
        log.info("cProfile stats written to %s", filename)


def duration_from_timeMethod(
    metadata: MutableMapping | None, method_name: str, clock: str = "Cpu"
) -> float | None:
    """Parse the metadata entries from ``timeMethod`` and return a duration.

    Parameters
    ----------
    metadata : `collections.abc.MutableMapping`
        The Task metadata that timing metrics were added to.
    method_name : `str`
        Name of the timed method to extract a duration for.
    clock : `str`, optional
        Options are "Cpu", "User", or "System".

    Returns
    -------
    duration : `float`
        The time elapsed between the start and end of the timed method.
    """
    if metadata is None:
        return None
    start = metadata[method_name + "Start" + clock + "Time"]
    if isinstance(start, list):
        start = start[-1]
    end = metadata[method_name + "End" + clock + "Time"]
    if isinstance(end, list):
        end = end[-1]
    return end - start
