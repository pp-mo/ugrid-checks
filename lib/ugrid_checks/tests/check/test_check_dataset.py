"""
Tests for ugrid_checks.check.check_dataset

"""
import logging
from pathlib import Path
import unittest as tests

import numpy as np
import ugrid_checks
from ugrid_checks.check import check_dataset
from ugrid_checks.nc_dataset_scan import scan_dataset


class TestCheck(tests.TestCase):
    def setUp(self):
        # Crude exercise test of code
        # TODO: replace with abstract generated testdata.
        filepath = Path(ugrid_checks.__file__).parent  # package dir
        filepath = filepath.parent.parent  # repo base dir
        filepath = filepath / "test_data" / "data_C4.nc"
        self.scan = scan_dataset(filepath)

    def _check_dataset(self, scan):
        return check_dataset(scan, print_summary=False, print_results=False)

    def _expect_notes(self, statements, expected_notes):
        def note_from_logrecord(record):
            statement_code = None
            if record.levelname == "INFO":
                statement_code = ""
            elif record.levelname == "WARNING":
                statement_code = "A" + str(record.args[0])
            elif record.levelname == "ERROR":
                statement_code = "R" + str(record.args[0])
            assert statement_code is not None
            return (statement_code, record.msg)

        actual_notes = [
            note_from_logrecord(record)
            for record in statements
            if record.levelno >= logging.INFO
        ]
        # Replace each expected item with a matching actual one,
        # if it can be found.
        for i_expected, expected_note in enumerate(expected_notes.copy()):
            expected_code, expected_msg = expected_note
            for actual_note in actual_notes:
                actual_code, actual_msg = actual_note
                if (
                    expected_code is None or expected_code == actual_code
                ) and expected_msg in actual_msg:
                    expected_notes[i_expected] = actual_note
                    break  # Only ever take first match
        # This should result in a matching list.
        self.assertEqual(set(expected_notes), set(actual_notes))

    def test_basic(self):
        logs = self._check_dataset(self.scan)
        self._expect_notes(
            logs, []
        )  # *NO* recorded statements (nothing wrong !).

    def test_mesh_missing_cf_role(self):
        del self.scan.variables["topology"].attributes["cf_role"]
        logs = self._check_dataset(self.scan)
        self._expect_notes(
            logs,
            [
                (
                    "R102",
                    "no \"cf_role\" property, which should be 'mesh_topology'",
                )
            ],
        )

    def test_mesh_bad_cf_role(self):
        self.scan.variables["topology"].attributes["cf_role"] = "something odd"
        logs = self._check_dataset(self.scan)
        self._expect_notes(
            logs,
            [
                ("R102", "should be 'mesh_topology'"),
                (
                    "",
                    "not a valid UGRID cf_role",
                ),  # N.B. this one doesn't have a code yet
            ],
        )

    def test_mesh_no_topology_dimension(self):
        del self.scan.variables["topology"].attributes["topology_dimension"]
        logs = self._check_dataset(self.scan)
        self._expect_notes(logs, [("R103", 'no "topology_dimension"')])

    def test_mesh_unknown_topology_dimension(self):
        self.scan.variables["topology"].attributes["topology_dimension"] = 4
        logs = self._check_dataset(self.scan)
        self._expect_notes(logs, [("R104", "not 0, 1 or 2")])

    def test_nonexistent_mesh(self):
        self.scan.variables["sample_data"].attributes["mesh"] = np.array(
            "other_mesh"
        )
        logs = self._check_dataset(self.scan)
        self._expect_notes(
            logs, [("R502", 'no "other_mesh" variable in the dataset.')]
        )

    def test_bad_mesh_name(self):
        self.scan.variables["sample_data"].attributes["mesh"] = np.array(
            "this that other"
        )
        logs = self._check_dataset(self.scan)
        self._expect_notes(
            logs,
            [
                (
                    "R502",
                    "\"mesh='this that other'\", "
                    "which is not a valid variable name.",
                )
            ],
        )

    def test_empty_mesh_name(self):
        self.scan.variables["sample_data"].attributes["mesh"] = np.array("")
        logs = self._check_dataset(self.scan)
        self._expect_notes(
            logs,
            [("R502", "\"mesh=''\", which is not a valid variable name.")],
        )


if __name__ == "__main__":
    tests.main()
