"""
Tests for ugrid_checks.check.check_data

"""
from pathlib import Path
import unittest as tests

import ugrid_checks
from ugrid_checks.nc_dataset_scan import scan_dataset
from ugrid_checks.check import check_data


class TestCheck(tests.TestCase):
    def setUp(self):
        # Crude exercise test of code
        # TODO: replace with abstract generated testdata.
        filepath = Path(ugrid_checks.__file__).parent  # package dir
        filepath = filepath.parent.parent  # repo base dir
        filepath = filepath / "test_data" / "data_C4.nc"
        self.scan = scan_dataset(filepath)
        # print(data)

    def test_basic(self):
        check_data(self.scan)
        t_dbg = 0

if __name__ == '__main__':
    tests.main()
