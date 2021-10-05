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

from __future__ import annotations

__all__ = ("TRACE", "VERBOSE", "getLogger", "LsstLogAdapter")

import logging
from logging import LoggerAdapter
from deprecated.sphinx import deprecated
from contextlib import contextmanager

from typing import (
    Any,
    Generator,
    List,
    Optional,
    Union,
)

# log level for trace (verbose debug).
TRACE = 5
logging.addLevelName(TRACE, "TRACE")

# Verbose logging is midway between INFO and DEBUG.
VERBOSE = (logging.INFO + logging.DEBUG) // 2
logging.addLevelName(VERBOSE, "VERBOSE")


class _F:
    """
    Format, supporting str.format() syntax
    see using-custom-message-objects in logging-cookbook.html
    """
    def __init__(self, fmt: str, /, *args: Any, **kwargs: Any):
        self.fmt = fmt
        self.args = args
        self.kwargs = kwargs

    def __str__(self) -> str:
        return self.fmt.format(*self.args, **self.kwargs)


class LsstLogAdapter(LoggerAdapter):
    """A special logging adapter to provide log features for LSST code.

    Expected to be instantiated initially by a call to `getLogger()`.

    This class provides enhancements over `logging.Logger` that include:

    * Methods for issuing trace and verbose level log messages.
    * Provision of a context manager to temporarily change the log level.
    * Attachment of logging level constants to the class to make it easier
      for a Task writer to access a specific log level without having to
      know the underlying logger class.
    """

    # Store logging constants in the class for convenience. This is not
    # something supported by Python loggers but can simplify some
    # logic if the logger is available.
    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING

    # Python supports these but prefers they are not used.
    FATAL = logging.FATAL
    WARN = logging.WARN

    # These are specific to Tasks
    TRACE = TRACE
    VERBOSE = VERBOSE

    @contextmanager
    def temporary_log_level(self, level: Union[int, str]) -> Generator:
        """A context manager that temporarily sets the level of this logger.

        Parameters
        ----------
        level : `int`
            The new temporary log level.
        """
        old = self.level
        self.setLevel(level)
        try:
            yield
        finally:
            self.setLevel(old)

    @property
    def level(self) -> int:
        """Current level of this logger (``int``)."""
        return self.logger.level

    def getChild(self, name: str) -> LsstLogAdapter:
        """Get the named child logger.

        Parameters
        ----------
        name : `str`
            Name of the child relative to this logger.

        Returns
        -------
        child : `LsstLogAdapter`
            The child logger.
        """
        return getLogger(name=name, logger=self.logger)

    @deprecated(reason="Use Python Logger compatible isEnabledFor Will be removed after v23.",
                version="v23", category=FutureWarning)
    def isDebugEnabled(self) -> bool:
        return self.isEnabledFor(self.DEBUG)

    @deprecated(reason="Use Python Logger compatible 'name' attribute. Will be removed after v23.",
                version="v23", category=FutureWarning)
    def getName(self) -> str:
        return self.name

    @deprecated(reason="Use Python Logger compatible .level property. Will be removed after v23.",
                version="v23", category=FutureWarning)
    def getLevel(self) -> int:
        return self.logger.level

    def fatal(self, msg: str, *args: Any, **kwargs: Any) -> None:
        # Python does not provide this method in LoggerAdapter but does
        # not formally deprecated it in favor of critical() either.
        # Provide it without deprecation message for consistency with Python.
        # stacklevel=5 accounts for the forwarding of LoggerAdapter.
        self.critical(msg, *args, **kwargs, stacklevel=4)

    def verbose(self, fmt: str, *args: Any, **kwargs: Any) -> None:
        """Issue a VERBOSE level log message.

        Arguments are as for `logging.info`.
        ``VERBOSE`` is between ``DEBUG`` and ``INFO``.
        """
        # There is no other way to achieve this other than a special logger
        # method.
        # Stacklevel is passed in so that the correct line is reported
        # in the log record and not this line. 3 is this method,
        # 2 is the level from `self.log` and 1 is the log infrastructure
        # itself.
        self.log(VERBOSE, fmt, *args, stacklevel=3, **kwargs)

    def trace(self, fmt: str, *args: Any) -> None:
        """Issue a TRACE level log message.

        Arguments are as for `logging.info`.
        ``TRACE`` is lower than ``DEBUG``.
        """
        # There is no other way to achieve this other than a special logger
        # method. For stacklevel discussion see `verbose()`.
        self.log(TRACE, fmt, *args, stacklevel=3)

    @deprecated(reason="Use Python Logger compatible method. Will be removed after v23.",
                version="v23", category=FutureWarning)
    def tracef(self, fmt: str, *args: Any, **kwargs: Any) -> None:
        # Stacklevel is 4 to account for the deprecation wrapper
        self.log(TRACE, _F(fmt, *args, **kwargs), stacklevel=4)

    @deprecated(reason="Use Python Logger compatible method. Will be removed after v23.",
                version="v23", category=FutureWarning)
    def debugf(self, fmt: str, *args: Any, **kwargs: Any) -> None:
        self.log(logging.DEBUG, _F(fmt, *args, **kwargs), stacklevel=4)

    @deprecated(reason="Use Python Logger compatible method. Will be removed after v23.",
                version="v23", category=FutureWarning)
    def infof(self, fmt: str, *args: Any, **kwargs: Any) -> None:
        self.log(logging.INFO, _F(fmt, *args, **kwargs), stacklevel=4)

    @deprecated(reason="Use Python Logger compatible method. Will be removed after v23.",
                version="v23", category=FutureWarning)
    def warnf(self, fmt: str, *args: Any, **kwargs: Any) -> None:
        self.log(logging.WARNING, _F(fmt, *args, **kwargs), stacklevel=4)

    @deprecated(reason="Use Python Logger compatible method. Will be removed after v23.",
                version="v23", category=FutureWarning)
    def errorf(self, fmt: str, *args: Any, **kwargs: Any) -> None:
        self.log(logging.ERROR, _F(fmt, *args, **kwargs), stacklevel=4)

    @deprecated(reason="Use Python Logger compatible method. Will be removed after v23.",
                version="v23", category=FutureWarning)
    def fatalf(self, fmt: str, *args: Any, **kwargs: Any) -> None:
        self.log(logging.CRITICAL, _F(fmt, *args, **kwargs), stacklevel=4)

    def setLevel(self, level: Union[int, str]) -> None:
        """Set the level for the logger, trapping lsst.log values.

        Parameters
        ----------
        level : `int`
            The level to use. If the level looks too big to be a Python
            logging level it is assumed to be a lsst.log level.
        """
        if isinstance(level, int) and level > logging.CRITICAL:
            self.logger.warning("Attempting to set level to %d -- looks like an lsst.log level so scaling it"
                                " accordingly.", level)
            level //= 1000

        self.logger.setLevel(level)

    @property
    def handlers(self) -> List[logging.Handler]:
        """Log handlers associated with this logger."""
        return self.logger.handlers

    def addHandler(self, handler: logging.Handler) -> None:
        """Add a handler to this logger.

        The handler is forwarded to the underlying logger.
        """
        self.logger.addHandler(handler)

    def removeHandler(self, handler: logging.Handler) -> None:
        """Remove the given handler from the underlying logger."""
        self.logger.removeHandler(handler)


def getLogger(name: Optional[str] = None, logger: logging.Logger = None) -> LsstLogAdapter:
    """Get a logger compatible with LSST usage.

    Parameters
    ----------
    name : `str`, optional
        Name of the logger. Root logger if `None`.
    logger : `logging.Logger` or `LsstLogAdapter`
        If given the logger is converted to the relevant logger class.
        If ``name`` is given the logger is assumed to be a child of the
        supplied logger.

    Returns
    -------
    logger : `LsstLogAdapter`
        The relevant logger.

    Notes
    -----
    A `logging.LoggerAdapter` is used since it is easier to provide a more
    uniform interface than when using `logging.setLoggerClass`. An adapter
    can be wrapped around the root logger and the `~logging.setLoggerClass`
    will return the logger first given that name even if the name was
    used before the `Task` was created.
    """
    if not logger:
        logger = logging.getLogger(name)
    elif name:
        logger = logger.getChild(name)
    return LsstLogAdapter(logger, {})


LsstLoggers = Union[logging.Logger, LsstLogAdapter]
