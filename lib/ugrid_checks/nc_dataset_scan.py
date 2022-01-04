from dataclasses import dataclass
from typing import Mapping, Text, Tuple

import numpy as np


class Dict_1dArray(dict):
    """
    A specialised dictionary type to contain netcdf attributes.

    All values are cast as Numpy arrays of <= 1 dimension.
    Thus, they will all have dtype, ndim, etc.
    This then matches what would be read from a file with netCDF4-python.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            # Rewrite all content to ensure conversion.
            self[key] = value

    def __setitem__(self, key, value):
        value = np.asarray(value)
        if value.ndim > 1:
            msg = (
                f"{self.__class__} value [{key!r}] = {value!r} is an error: "
                "values may not have more than one dimension."
            )
            raise ValueError(msg)
        super().__setitem__(key, value)


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
    dtype: object

    #: Attributes : values are numpy scalars or 0-1 dimensional arrays
    attributes: Mapping[Text, np.ndarray]

    #: Data : a Dask lazy array, or None if actual file not available.
    #: If data is not None, it is still possible for access to fail if the
    #: original file has since been modified or removed.
    data = None

    def __init__(
        self, name, dimensions, shape, dtype, attributes=None, data=None
    ):
        """Enhanced init to support optional 'attributes' + 'data'."""
        self.name = name
        self.dimensions = dimensions
        self.shape = shape
        self.dtype = dtype
        # Attributes defaults to an empty dict.
        if attributes is None:
            attributes = {}
        self.attributes = attributes
        # Data may be also None (defaults to None).
        self.data = data

    def __setattr__(self, key, value):
        # Ensure correct types for 'attributes' and 'data' properties.
        if key == "attributes":
            # 'attributes' is always a dictionary with 0d or 1d array values.
            value = Dict_1dArray(value)
        elif key == "data":
            # 'data' may be None, or an array matching self.shape.
            if value is not None:
                value = np.asarray(value)
                if value.shape != self.shape:
                    msg = (
                        f"Can't assign '.data' with shape {value.shape} "
                        f"to {self.__class__} with shape of {self.shape}."
                    )
                    raise ValueError(msg)
        super().__setattr__(key, value)


@dataclass
class NcDimSummary:
    """An object containing details of a Netcdf file dimension."""

    length: int
    is_unlimited: bool

    def __init__(self, length, is_unlimited=False):
        """Enhanced init to support optional is_unlimited arg."""
        self.length = length
        self.is_unlimited = is_unlimited


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
