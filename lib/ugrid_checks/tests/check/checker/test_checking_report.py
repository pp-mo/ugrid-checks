"""
Tests for :meth:`ugrid_checks.check.Checker.checking_report()`.

"""

import re

from pytest import fixture

from .. import simple_incorrect_scan_and_codes, simple_scan
from ... import cdl_scanner
from ..test_check_dataset__checks import DatasetChecker

# Yes, we do need these imports.
cdl_scanner
simple_scan
simple_incorrect_scan_and_codes


from ugrid_checks.check import Checker


@fixture
def test_checker_incorrect(simple_incorrect_scan_and_codes):
    scan, _ = simple_incorrect_scan_and_codes
    return Checker(scan, max_mb_checks=0)


@fixture
def test_checker_noerror(simple_scan):
    return Checker(simple_scan, max_mb_checks=0)


class Test_CheckReport(DatasetChecker):
    # Just one basic function test, for now.
    def test_basic(self, test_checker_incorrect):
        text = test_checker_incorrect.checking_report()
        # For now, this is identical to that printed by
        # check_dataset(print_output=True).
        # See: tests.check.test_check_dataset__interface...
        # ... Test_CheckerControls.test_printout_on
        check_re = (
            "conformance checks complete.*"
            "List of checker messages.*"
            "Total of 4 problems.*"
            "2 Rxxx requirement failures.*"
            "2 Axxx advisory recommendation warnings.*"
            "Done."
        )
        assert re.search(check_re, text, re.DOTALL)

    def test_noerror(self, test_checker_noerror):
        text = test_checker_noerror.checking_report()
        # For now, this is identical to that printed by
        # check_dataset(print_output=True).
        # See: tests.check.test_check_dataset__interface...
        # ... Test_CheckerControls.test_printout_on
        print(text)
        check_re = (
            "conformance checks complete.*" "No problems found.*" "Done."
        )
        assert re.search(check_re, text, re.DOTALL)
