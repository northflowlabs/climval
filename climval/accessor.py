"""xarray accessors: ``candidate.climval.compare(reference, ...)``.

Registered when climval is imported. The object the accessor is called on is
the candidate; every keyword of :func:`climval.compare` is accepted.
"""

from __future__ import annotations

from typing import Any

import xarray as xr

from climval.result import Comparison


@xr.register_dataarray_accessor("climval")  # type: ignore[no-untyped-call]
class DataArrayAccessor:
    def __init__(self, obj: xr.DataArray) -> None:
        self._obj = obj

    def compare(self, reference: Any, **kwargs: Any) -> Comparison:
        """Compare this DataArray (the candidate) with ``reference``."""
        from climval.api import compare

        return compare(reference, self._obj, **kwargs)


@xr.register_dataset_accessor("climval")  # type: ignore[no-untyped-call]
class DatasetAccessor:
    def __init__(self, obj: xr.Dataset) -> None:
        self._obj = obj

    def compare(self, reference: Any, variable: Any = None, **kwargs: Any) -> Comparison:
        """Compare a variable of this Dataset (the candidate) with ``reference``."""
        from climval.api import compare

        return compare(reference, self._obj, variable, **kwargs)
