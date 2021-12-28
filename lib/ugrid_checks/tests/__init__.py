"""
Unittests for the ugrid_checks package.

TODO: definitely a good idea to convert all this to PyTest.

"""
import copy
from pathlib import Path
from subprocess import check_call
from typing import Text, Union

from pytest import fixture
from ugrid_checks.check import (
    _VALID_CONNECTIVITY_ROLES,
    _VALID_MESHCOORD_ATTRS,
)
from ugrid_checks.nc_dataset_scan import (
    NcFileSummary,
    NcVariableSummary,
    scan_dataset,
)
from ugrid_checks.scan_utils import property_namelist


def cdl_scan(
    cdl: Text, tempdir_path: Path, tempfile_name: Union[Text, None] = None
):
    """
    Create a dataset "scan" :class:`~ugrid_checks.nc_data_scan.NcFileSummary`
    from a CDL string.

    Requires a temporary directory to create temporary intermediate files in.
    Calls 'ncgen' to create a netcdf file from the CDL.

    """

    tempfile_name = tempfile_name or "tmp.cdl"
    temp_nc_file_name = tempfile_name.replace(".cdl", ".nc")
    temp_cdl_path = Path(tempdir_path).resolve() / tempfile_name
    temp_nc_path = Path(tempdir_path).resolve() / temp_nc_file_name
    with open(temp_cdl_path, "w") as tempfile:
        tempfile.write(cdl)
    cmd = f"ncgen -k4  -o {temp_nc_path} {temp_cdl_path}"
    check_call(cmd, shell=True)
    scan = scan_dataset(temp_nc_path)
    return scan


@fixture()
def cdl_scanner(tmp_path):
    """
    A pytest fixture returning an object which can convert a CDL string to
    a dataset "scan" :class:`~ugrid_checks.nc_data_scan.NcFileSummary`.

    Since the operation uses temporary files, the 'cdl_scanner' embeds a
    'tmp_path' fixture, which determines where they are created.

    """

    class CdlScanner:
        def __init__(self, tmp_path):
            self.tmp_path = tmp_path

        def scan(self, cdl_string):
            return cdl_scan(cdl=cdl_string, tempdir_path=self.tmp_path)

    return CdlScanner(tmp_path)


#
# Utilities to make duplicated structures within file-scans.
# N.B. these could perhaps be written for more generalised use, but at present
# their only expected usage is in tests.
#
def next_name(name: str) -> str:
    # Convert 'xxx' to 'xxx_2' and 'xxx_<N>' to 'xxx_<N+1>'
    if name[-2] == "_" and name[-1].isdigit():
        digit = int(name[-1:])
        name = name[:-1] + str(digit + 1)
    else:
        name += "_2"  # NB don't use '_1'

    return name


def next_dim(file_scan: NcFileSummary, dim_name: str) -> str:
    # Get the 'next-named' dimension from an existing one,
    # creating + installing it if required.
    # N.B. we return only the *name*, suitable for setting variable dims.
    new_name = next_name(dim_name)
    new_dim = file_scan.dimensions.get(new_name)
    if new_dim is None:
        old_dim = file_scan.dimensions[dim_name]
        new_dim = copy.deepcopy(old_dim)
        file_scan.dimensions[new_name] = new_dim
    return new_name


def next_var(
    file_scan: NcFileSummary, var_name: str, ref_attrs=None
) -> NcVariableSummary:
    # Return the 'next-named' var, creating it if necessary.
    # This also 'bumps' its dimensions, and any referenced variables.
    # Referenced variables are those named in attribute 'ref_attrs', plus
    # 'mesh', 'coordinates' and 'bounds'.
    new_name = next_name(var_name)
    new_var = file_scan.variables.get(new_name)
    if new_var is None:
        old_var = file_scan.variables[var_name]  # This *should* exist !
        new_var = copy.deepcopy(old_var)
        new_var.name = new_name
        new_var.dimensions = [
            next_dim(file_scan, dimname) for dimname in old_var.dimensions
        ]
        # Shift any simple var refs (for datavar usage)
        if ref_attrs is None:
            ref_attrs = []
        # always do these ones
        ref_attrs += ["mesh", "bounds", "coordinates"]
        for attrname in ref_attrs:
            inner_varsattr = old_var.attributes.get(attrname)
            inner_varnames = property_namelist(inner_varsattr)
            if inner_varnames:
                new_names = [
                    next_var(file_scan, inner_name).name
                    for inner_name in inner_varnames
                ]
                new_var.attributes[attrname] = " ".join(new_names)
        file_scan.variables[new_var.name] = new_var
    return new_var


def next_mesh(file_scan: NcFileSummary, mesh_name: str) -> NcVariableSummary:
    # Return the 'next-named' mesh, creating it if needed.
    # This means duplicating its dimensions, coords and connectivities.
    # N.B. unlike the checker code itself, we here assume that the original
    # mesh is complete + consistent.
    new_name = next_name(mesh_name)
    new_mesh = file_scan.variables.get(new_name)
    if not new_mesh:
        # Copy the variable, also duplicating any coord+connectivity variables.
        extra_ref_attrs = _VALID_MESHCOORD_ATTRS + _VALID_CONNECTIVITY_ROLES
        new_mesh = next_var(file_scan, mesh_name, ref_attrs=extra_ref_attrs)
        # Similarly 'bump' any mesh-dimension attributes.
        for location in ("face", "edge"):
            coords_attr = f"{location}_dimension"
            dimname = new_mesh.attributes.get(coords_attr)
            if dimname:
                new_mesh.attributes[coords_attr] = next_name(dimname)
    return new_mesh
