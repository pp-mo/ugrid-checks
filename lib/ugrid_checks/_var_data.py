"""
Routines for fetching variable data, and performing key checks on it.

Rather than introducing Dask as a dependency, we will assume that the array
calculations we need can all be done in memory, but we must still implement
a 'lazy' array fetch.

In future, we may use Dask for this + allow more sophisticated out-of-memory
calculations on large data.  But that also requires a chunking policy (since
the default is poorly suited), so let's not do that for now.

For now instead, we fetch the *whole* variable data on demand.

"""
from typing import Callable, Union

import netCDF4 as nc
import numpy as np

from .nc_dataset_scan import NcVariableSummary


class VariableDataProxy:
    """
    Record details of a file variable, and provide deferred fetch of its data.

    These are what is stored in a NcVariableSummary.data by file-scanning.

    """

    def __init__(self, datapath, varname):
        self.datapath = datapath
        self.varname = varname

    def fetch_array(self):
        try:
            ds = nc.Dataset(self.datapath)
            result = ds.variables[self.varname][:]  # Fetch whole array
        finally:
            ds.close()
        return result


class VariableDataStats:
    def __init__(
        self,
        var: NcVariableSummary,
        max_datasize_mb: float = -1.0,
        data_skipped_callback: Union[Callable, None] = None,
    ):
        """
        An object to efficiently calculate key statistics of a variable's data.

        Parameters
        ----------
        var : NcVariableSummary
            the dataset variable we are referencing, whose ".data" is the
            relevant VariableDataProxy, if there is one.

        max_datasize_mb : float, default -1.0
            A size threshold, in Mb.  If 'var' is bigger than this, then its
            data will not be fetched, and all properties return 'safe' values.
            A value == 0 means never fetch, and < 0 means always fetch.

        data_skipped_callback: callable or None, default None
            An argless function, to be called if the variable data was not
            fetched on request, because it failed the size check.

        When var data is fetched, it is cached in this object.

        The 'key statistics' of the data are calculated independently on demand,
        and the results cached.
        The statistic calculations are individually deferred, and the results
        cached.  If the variable exceeded the size threshold, and was not
        fetched, then all statistics return "safe" values instead.

        """
        if max_datasize_mb < 0:
            max_datasize_mb = 1.0e30  # stupidly large
        # Public state
        self.var = var
        self.max_datasize = max_datasize_mb
        self.has_data = False  # public version of "self._data is not None"
        # Private state
        self._data = None
        self._decide_data_fetch = True  # We only do it once.
        self._data_skipped_event = data_skipped_callback

        # Create an empty cache = values for each key statistic.
        self._cached_values = {
            "has_missing_values": None,
            "has_duplicate_values": None,
            "min_value": None,
            "max_value": None,
        }

    def get_data(self) -> np.ndarray:
        """
        Get the var data, if the var is not too large.

        Returns the array, or None.
        When loaded, the data is cached ==> "self.has_data == True".

        """
        if self._decide_data_fetch:
            # only decide this *once*.
            self._decide_data_fetch = False
            if self.var:
                # Decide *whether* to fetch data, and record it in self._data.
                # If not, self._data == None, and will stay that way.
                var_size_mb = (
                    np.prod(self.var.shape) * self.var.dtype.itemsize * 1.0e-6
                )
                if var_size_mb < self.max_datasize:
                    self._data = self.var.data.fetch_array()
                    self.has_data = True
                else:
                    if self._data_skipped_event is not None:
                        self._data_skipped_event()
        return self._data

    @property
    def has_missing_values(self) -> bool:
        """Whether the var has any missing data values (or default=False)."""
        if self._cached_values["has_missing_values"] is None:
            data = self.get_data()
            if data is None:
                has_missing = False  # "Safe" answer
            else:
                has_missing = np.ma.is_masked(data)
            self._cached_values["has_missing_values"] = has_missing
        return self._cached_values["has_missing_values"]

    @property
    def has_duplicate_values(self) -> bool:
        """Whether the var has any duplicate data values (or default=False)."""
        if self._cached_values["has_duplicate_values"] is None:
            data = self.get_data()
            if data is None:
                has_duplicates = False  # "Safe" answer
            else:
                data = data.flatten()
                # for masked data, consider only non-masked points
                if np.ma.is_masked(data):
                    data = data.data[~data.mask]
                # check for number of distinct values
                (n_data_values,) = data.shape
                n_distinct_values = len(set(data))
                has_duplicates = n_distinct_values < n_data_values
            self._cached_values["has_duplicate_values"] = has_duplicates
        return self._cached_values["has_duplicate_values"]

    @property
    def min_value(self):
        """The minimum (un-masked) value in the var (or default=1)."""
        if self._cached_values["min_value"] is None:
            data = self.get_data()
            if data is None:
                min_index = 1  # "Safe" answer
            else:
                min_index = data.min()  # also works if masked
            self._cached_values["min_value"] = min_index
        return self._cached_values["min_value"]

    @property
    def max_value(self):
        """The maximum (un-masked) value in the var (or default=1)."""
        if self._cached_values["max_value"] is None:
            data = self.get_data()
            if data is None:
                max_index = 1  # "Safe" answer
            else:
                max_index = data.max()  # also works if masked
            self._cached_values["max_value"] = max_index
        return self._cached_values["max_value"]
