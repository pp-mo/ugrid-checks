"""
Tests for ugrid_checks.nc_scan_dataset.scan_dataset

"""
from shutil import rmtree
from tempfile import mkdtemp
import unittest as tests

import numpy as np
from numpy import asanyarray as nparray
from ugrid_checks.nc_dataset_scan import NcDimSummary
from ugrid_checks.tests import cdl_scan


class TestScan(tests.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir_path = mkdtemp()

    @classmethod
    def tearDownClass(cls):
        rmtree(cls.temp_dir_path)

    def setUp(self):
        # Crude exercise test of code
        self.test_cdl = """
netcdf small_test {
dimensions:
    x = 3 ;
    y = 4 ;
    t = UNLIMITED ;

variables:
    double x(x) ;
        x:name = "x-coord" ;
    int y(y) ;
        y:name = "y-coord" ;
    float t(t) ;
        t:name = "time" ;
        t:units = "s" ;
    float aux_x(x) ;
        aux_x:units = "1" ;
    float data_var(t, y, x) ;
        data_var:name = "multidim_data" ;
        data_var:units = "1" ;
        data_var:_FillValue = 9.999e25f ;
        data_var:coordinates = "aux_x" ;

// global attributes:
    :Conventions = "CF-1.0" ;
    :comment = "global comment attribute" ;

data:
    t = 1.0, 2.0 ;
}
"""
        self.scan = cdl_scan(self.test_cdl, self.temp_dir_path)
        print(self.scan)

    def test_dimensions(self):
        dims = self.scan.dimensions
        self.assertEqual(3, len(dims))
        self.assertEqual(["x", "y", "t"], list(dims.keys()))
        self.assertEqual(dims["x"], NcDimSummary(length=3, is_unlimited=False))
        self.assertEqual(dims["y"], NcDimSummary(length=4, is_unlimited=False))
        self.assertEqual(dims["t"], NcDimSummary(length=2, is_unlimited=True))

    def test_variables(self):
        vars = self.scan.variables
        self.assertEqual(5, len(vars))
        self.assertEqual(
            set(["x", "y", "t", "aux_x", "data_var"]), set(vars.keys())
        )

        # Note: testing with individual asserts ... it is _possible_ to check
        # for equality with an expected, matching NcVariableSummary, but this
        # makes any failure messages very hard to interpret.
        var_x = vars["x"]
        self.assertEqual("x", var_x.name)
        self.assertEqual(("x",), var_x.dimensions)
        self.assertEqual((3,), var_x.shape)
        self.assertEqual(np.float64, var_x.dtype)
        self.assertEqual({"name": "x-coord"}, var_x.attributes)

        var_y = vars["y"]
        self.assertEqual("y", var_y.name)
        self.assertEqual(("y",), var_y.dimensions)
        self.assertEqual((4,), var_y.shape)
        self.assertEqual(np.int32, var_y.dtype)
        self.assertEqual({"name": "y-coord"}, var_y.attributes)

        var_t = vars["t"]
        self.assertEqual("t", var_t.name)
        self.assertEqual(("t",), var_t.dimensions)
        self.assertEqual((2,), var_t.shape)
        self.assertEqual(np.float32, var_t.dtype)
        self.assertEqual({"name": "time", "units": "s"}, var_t.attributes)

        var_auxx = vars["aux_x"]
        self.assertEqual("aux_x", var_auxx.name)
        self.assertEqual(("x",), var_auxx.dimensions)
        self.assertEqual((3,), var_auxx.shape)
        self.assertEqual(np.float32, var_auxx.dtype)
        self.assertEqual({"units": "1"}, var_auxx.attributes)

        var_data = vars["data_var"]
        self.assertEqual("data_var", var_data.name)
        self.assertEqual(("t", "y", "x"), var_data.dimensions)
        self.assertEqual((2, 4, 3), var_data.shape)
        self.assertEqual(np.float32, var_data.dtype)
        self.assertEqual(
            {
                "units": nparray("1"),
                "name": nparray("multidim_data"),
                "_FillValue": nparray(9.999e25, dtype=np.float32),
                "coordinates": nparray("aux_x"),
            },
            var_data.attributes,
        )

    def test_global_attributes(self):
        attrs = self.scan.attributes
        self.assertEqual(2, len(attrs))
        self.assertEqual(["Conventions", "comment"], list(attrs.keys()))
        self.assertEqual(nparray("CF-1.0"), attrs["Conventions"])
        self.assertEqual(nparray("global comment attribute"), attrs["comment"])


if __name__ == "__main__":
    tests.main()
