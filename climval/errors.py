"""Exception types raised by climval.

Every error climval raises on purpose derives from :class:`ClimvalError`, so a
pipeline can catch them in one place.
"""

from __future__ import annotations


class ClimvalError(Exception):
    """Base class for all climval errors."""


class InputError(ClimvalError, ValueError):
    """An input could not be resolved to a variable (missing data, unknown name)."""


class AlignmentError(ClimvalError, ValueError):
    """The two inputs cannot be put on a common grid, period or set of dimensions."""


class CurvilinearGridError(AlignmentError):
    """A grid has two-dimensional latitude and longitude and cannot be interpolated."""


class NoOverlapError(AlignmentError):
    """The inputs share no points in space or no periods in time."""


class UnitsError(ClimvalError, ValueError):
    """Units are missing, unknown or not convertible."""


class MissingDependencyError(ClimvalError, ImportError):
    """An optional dependency needed for the requested option is not installed."""
