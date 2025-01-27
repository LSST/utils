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

"""Utilities for measuring resource consumption."""

from __future__ import annotations

__all__ = ["get_current_mem_usage", "get_peak_mem_usage"]

import dataclasses
import platform
import resource
import time

import astropy.units as u
import psutil


def _get_rusage_multiplier() -> int:
    """Return the multiplier to use for memory usage returned by getrusage.

    Returns
    -------
    unit : `int`
        The multiplier that should be applied to the memory usage numbers
        returned by `resource.getrusage` to convert them to bytes.
    """
    system = platform.system().lower()
    if system == "darwin":
        # MacOS uses bytes
        return 1
    elif "solaris" in system or "sunos" in system:
        # Solaris and SunOS use pages
        return resource.getpagesize()
    else:
        # Assume Linux/FreeBSD etc, which use kibibytes
        return 1024


_RUSAGE_MEMORY_MULTIPLIER = _get_rusage_multiplier()


def get_current_mem_usage() -> tuple[u.Quantity, u.Quantity]:
    """Report current memory usage.

    Returns
    -------
    usage_main : `astropy.units.Quantity`
        Current memory usage of the calling process expressed in bytes.
    usage_child : `astropy.units.Quantity`
        Current memory usage of the child processes (zero if there are none)
        expressed in bytes.

    Notes
    -----
    Function reports current memory usage using resident set size as a proxy.
    As such the values it reports are capped at available physical RAM and may
    not reflect the actual memory allocated to the process and its children.
    """
    proc = psutil.Process()
    with proc.oneshot():
        usage_main = proc.memory_info().rss * u.byte
        usage_child = sum([child.memory_info().rss for child in proc.children()]) * u.byte
    return usage_main, usage_child


def get_peak_mem_usage() -> tuple[u.Quantity, u.Quantity]:
    """Report peak memory usage.

    Returns
    -------
    peak_main: `astropy.units.Quantity`
        Peak memory usage (maximum resident set size) of the calling process.
    peak_child: `astropy.units.Quantity`
        Peak memory usage (resident set size) of the largest child process.

    Notes
    -----
    Function reports peak memory usage using the maximum resident set size as
    a proxy. As such the value it reports is capped at available physical RAM
    and may not reflect the actual maximal value.
    """
    peak_main = _get_current_rusage().maxResidentSetSize * u.byte
    peak_child = _get_current_rusage(for_children=True).maxResidentSetSize * u.byte
    return peak_main, peak_child


@dataclasses.dataclass(frozen=True)
class _UsageInfo:
    """Summary of process usage."""

    cpuTime: float
    """CPU time in seconds."""
    userTime: float
    """User time in seconds."""
    systemTime: float
    """System time in seconds."""
    maxResidentSetSize: int
    """Maximum resident set size in bytes."""
    minorPageFaults: int
    majorPageFaults: int
    blockInputs: int
    blockOutputs: int
    voluntaryContextSwitches: int
    involuntaryContextSwitches: int
    nodeCpuPercent: float
    """Node-wide CPU utilization expressed as a percentage."""
    nodeLoadAverage: float
    """Number of node-wide processes in a runnable state over the last minute."""

    def dict(self) -> dict[str, float | int]:
        return dataclasses.asdict(self)


def _get_current_rusage(for_children: bool = False) -> _UsageInfo:
    """Get information about this (or the child) process.

    Parameters
    ----------
    for_children : `bool`, optional
        Whether the information should be requested for child processes.
        Default is for the current process.

    Returns
    -------
    info : `_UsageInfo`
        The information obtained from the process.
    """
    who = resource.RUSAGE_CHILDREN if for_children else resource.RUSAGE_SELF
    res = resource.getrusage(who)

    # Convert the memory usage to bytes.
    max_rss = res.ru_maxrss * _RUSAGE_MEMORY_MULTIPLIER

    return _UsageInfo(
        cpuTime=time.process_time(),
        userTime=res.ru_utime,
        systemTime=res.ru_stime,
        maxResidentSetSize=max_rss,
        minorPageFaults=res.ru_minflt,
        majorPageFaults=res.ru_majflt,
        blockInputs=res.ru_inblock,
        blockOutputs=res.ru_oublock,
        voluntaryContextSwitches=res.ru_nvcsw,
        involuntaryContextSwitches=res.ru_nivcsw,
        nodeCpuPercent=psutil.cpu_percent(),
        nodeLoadAverage=psutil.getloadavg()[0],
    )
