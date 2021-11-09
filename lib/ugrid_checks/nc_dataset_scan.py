from dataclasses import dataclass
from typing import Mapping, Text, Tuple, Union

import dask.array as da
import numpy as np


@dataclass
class NcVariableSummary:
    """
    An object containing details of a Netcdf file variable.
    The actual file data may also be fetched from 'data', if it is available.

    """

    #: Variable name in the file, consistent with containing file.variables
    name: Text

    #: Dimension names, consistent with containing file
    dimensions: Tuple[Text, ...]

    #: Variable array shape, consistent with 'dims', containing file
    #: and 'data', if any
    shape: Tuple[int, ...]  #

    #: Numpy dtype, consistent with 'data', if any
    dtype: np.dtype

    #: Attributes : values are numpy scalars or 0-1 dimensional arrays
    attributes: Mapping[Text, np.ndarray]

    #: Data : a Dask lazy array, or None if actual file not available.
    #: If data is not None, it is still possible for access to fail if the
    #: original file has since been modified or removed.
    data: Union[da.Array, None]


@dataclass
class NcDimSummary:
    """An object containing details of a Netcdf file dimension."""

    length: int
    is_unlimited: bool


@dataclass
class NcFileSummary:
    """
    An object containing details of the contents of a Netcdf file.

    Actual file data may be available from the variables, but also may not be.

    """

    #: file dimensions: the values represent dimension lengths
    dimensions: Mapping[Text, NcDimSummary]
    variables: Mapping[Text, NcVariableSummary]
    attributes: Mapping[Text, np.ndarray]


def scan_dataset(filepath):
    """
    Snapshot a netcdf dataset (the key metadata).

    Returns:
        dimsdict, varsdict
        * dimsdict (dict):
            A map of dimension-name: length.
        * varsdict (dict):
            A map of each variable's properties, {var_name: propsdict}
            Each propsdict is {attribute-name: value} over the var's ncattrs().
            Each propsdict ALSO contains a [_VAR_DIMS] entry listing the
            variable's dims.

    """
    import netCDF4 as nc

    ds = nc.Dataset(filepath)
    # dims dict is {name: len}
    dims_summary = {
        name: NcDimSummary(length=dim.size, is_unlimited=dim.isunlimited())
        for name, dim in ds.dimensions.items()
    }

    def allattrs(item):
        # get all attributes as a map, whose values are all numpy objects
        # NB 'item' can be a variable, or the whole dataset
        # (potentially a Group, but we are not doing that at present)
        return {
            attr: np.asanyarray(item.getncattr(attr))
            for attr in item.ncattrs()
        }

    vars_summary = {
        name: NcVariableSummary(
            name=name,
            dimensions=var.dimensions,
            shape=var.shape,
            dtype=var.dtype,
            attributes=allattrs(var),
            data=None,  # *could* be a DataProxy, but let's not go there yet
        )
        for name, var in ds.variables.items()
    }

    attrs_summary = allattrs(ds)
    file_summary = NcFileSummary(
        dimensions=dims_summary,
        variables=vars_summary,
        attributes=attrs_summary,
    )

    ds.close()
    return file_summary
