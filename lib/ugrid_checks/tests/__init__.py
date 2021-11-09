"""
Unittests for the ugrid_checks package.

TODO: definitely a good idea to convert all this to PyTest.

"""
from pathlib import Path
from subprocess import check_call
from typing import Text, Union

from ugrid_checks.nc_dataset_scan import scan_dataset


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
    cmd = f"ncgen -k4 {temp_cdl_path} -o {temp_nc_path}"
    check_call(cmd, shell=True)
    scan = scan_dataset(temp_nc_path)
    return scan
