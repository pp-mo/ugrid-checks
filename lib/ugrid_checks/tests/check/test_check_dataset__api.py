"""
Tests for ugrid_checks.check.check_dataset API

N.B. the actual checker function is tested in
:mod:tests.check.test_check_dataset__checks

"""

import re

from . import simple_incorrect_scan_and_codes, simple_scan
from .. import cdl_scanner
from .test_check_dataset__checks import DatasetChecker, scan_mini_w_data

# Yes, we do need these imports.
cdl_scanner
simple_scan
simple_incorrect_scan_and_codes
scan_mini_w_data


from ugrid_checks.check import check_dataset


class Test_CheckerControls(DatasetChecker):
    def check(
        self,
        # These args are pass-through to the check_dataset call
        arg,
        print_summary=False,
        omit_advisories=False,
        ignore_codes=None,
        max_data_mb=200.0,
        # This arg defines the expected results
        expected_codes=None,
    ):
        # Call the checker with the test object.
        # N.B. if setting 'print_summary', best to hide the print output
        # with the PyTest standard 'capsys' fixture.
        checker = check_dataset(
            arg,
            print_summary=print_summary,
            omit_advisories=omit_advisories,
            ignore_codes=ignore_codes,
            max_data_mb=max_data_mb,
        )
        logs = checker.logger.report_statement_logrecords()

        # Check that the list of statements is as expected
        expected_statements = [(code, "") for code in expected_codes]
        self._expect_notes(logs, expected_statements)

        # Also check that output counts the expected number of errors+warnings
        expect_n_err = sum(
            1 if record.levelname == "ERROR" else 0 for record in logs
        )
        expect_n_warn = sum(
            1 if record.levelname == "WARNING" else 0 for record in logs
        )
        assert checker.logger.N_FAILURES == expect_n_err
        assert checker.logger.N_WARNINGS == expect_n_warn
        return checker

    def test_noerrors(self, simple_scan):
        self.check(simple_scan, expected_codes=[])

    def test_basic(self, simple_incorrect_scan_and_codes):
        scan, codes = simple_incorrect_scan_and_codes
        self.check(scan, expected_codes=codes)

    def test_printout_on(self, simple_incorrect_scan_and_codes, capsys):
        scan, codes = simple_incorrect_scan_and_codes
        self.check(scan, print_summary=True, expected_codes=codes)
        text = capsys.readouterr().out
        check_re = (
            "conformance checks complete.*"
            "List of checker messages.*"
            "Total of 4 problems.*"
            "2 Rxxx requirement failures.*"
            "2 Axxx advisory recommendation warnings.*"
            "Done."
        )
        assert re.search(check_re, text, re.DOTALL)

    def test_printout_off(self, simple_incorrect_scan_and_codes, capsys):
        scan, codes = simple_incorrect_scan_and_codes
        self.check(scan, print_summary=False, expected_codes=codes)
        assert capsys.readouterr().out == ""

    def test_no_warnings(self, simple_incorrect_scan_and_codes):
        scan, codes = simple_incorrect_scan_and_codes
        errors = [code for code in codes if code.startswith("R")]
        self.check(scan, omit_advisories=True, expected_codes=errors)

    def test_ignore_codes(self, simple_incorrect_scan_and_codes):
        scan, codes = simple_incorrect_scan_and_codes
        ignores = ["R101", "A903"]
        codes = [code for code in codes if code not in ignores]
        self.check(scan, ignore_codes=ignores, expected_codes=codes)

    def test_size_threshold(self, scan_mini_w_data):
        scan = scan_mini_w_data
        checker = self.check(scan, expected_codes=[])
        assert not checker.data_skipped
        checker = self.check(scan, max_data_mb=0.0, expected_codes=[])
        assert checker.data_skipped
