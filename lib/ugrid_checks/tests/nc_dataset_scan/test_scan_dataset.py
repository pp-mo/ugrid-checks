"""
Tests for ugrid_checks.nc_scan_dataset.scan_dataset

"""
import unittest as tests

from ugrid_checks.nc_dataset_scan import scan_dataset


class TestScan(tests.TestCase):
    def test(self):
        # Crude exercise test of code
        # TODO: remove dependence on Iris testing
        # TODO: replace with proper testing
        import iris.tests as itsts

        filepath = itsts.get_data_path(
            ["NetCDF", "global", "xyt", "SMALL_hires_wind_u_for_ipcc4.nc"]
        )
        data = scan_dataset(filepath)
        print(data)


if __name__ == "__main__":
    tests.main()
