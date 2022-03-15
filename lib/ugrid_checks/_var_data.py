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
import netCDF4 as nc
import numpy as np

from .nc_dataset_scan import NcVariableSummary


class VariableDataProxy:
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
    def __init__(self, var: NcVariableSummary, max_datasize_mb: float = -1.0):
        """
        An object to efficiently calculate key statistics of a variables' data.

        Parameters
        ----------
        var : NcVariableSummary
            the dataset variable we are referencing, whose ".data" is the
            relevant VariableDataProxy, if there is one.

        max_datasize_mb : float, default -1.0
            A size threshold, in Mb. If 'var' is bigger than this, then its
            data will not be fetched, and all properties return 'safe' values.
            A value == 0 means never fetch, and < 0 means always fetch.

        Actual calculations are deferred, in addition to returning the "safe"
        values without calculation if the variable exceeds the size threshold.

        The 'key statistics' of the data are calculated independently on demand,
        and the results cached.

        When var data is fetched, it is cached in this object.
        This is discarded on request, or when all the key statistics have been
        calculated.

        """
        if max_datasize_mb < 0:
            max_datasize_mb = 1.0e30  # stupidly large
        # Public state
        self.max_datasize = max_datasize_mb
        self.has_data = False  # public version of "self._data is not None"
        self.var = var
        # Private state
        self._data = None
        self._decide_data_fetch = True  # We only do it once.

        # Create an empty cache = value defaults for each key statistic.
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
            self._decide_data_fetch = False
            if self.var:
                # Decide *whether* to fetch data, and record it in self._data.
                # If not, self._data == None, and will stay that way.
                var_size_mb = (
                    np.prod(self.var.shape) * self.var.dtype.itemsize * 1.0e-6
                )
                if var_size_mb < self.max_datasize:
                    self._data = self.var.fetch_array()
                    self.has_data = True
        return self._data

    def discard_data(self):
        """
        Remove the cached data values array.

        This is done automatically when all the required properties have been
        calculated.

        """
        self._data = None
        self.has_data = False

    def get_all_stats(self):
        """
        Calculate all the statistics : this also discards the cached data.

        """
        for name in self._cached_values.keys():
            getattr(self, name)()  # Call each access routine.

    def _discard_data_if_alldone(self):
        if all(value is not None for value in self._cached_values.values()):
            self.discard_data()

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
            self._discard_data_if_alldone()
        return self._cached_values["has_missing_values"]

    @property
    def has_duplicate_values(self) -> bool:
        """Whether the var has any duplicate data values (or default=False)."""
        if self._cached_values["has_duplicate_values"] is None:
            data = self.get_data()
            if data is None:
                has_duplicates = False  # "Safe" answer
            else:
                has_duplicates = len(set(data.flatten())) > 1
            self._cached_values["has_duplicate_values"] = has_duplicates
            self._discard_data_if_alldone()
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
            self._discard_data_if_alldone()
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
            self._discard_data_if_alldone()
        return self._cached_values["max_value"]
