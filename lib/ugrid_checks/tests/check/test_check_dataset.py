"""
Tests for ugrid_checks.check.check_dataset

"""
import logging

import numpy as np
from pytest import fixture
from ugrid_checks.check import check_dataset
from ugrid_checks.tests import cdl_scanner

# Prevent error from 'black' about unused import.
# NOTE : the import is *not* in fact redundant, since pytest requires it.
cdl_scanner


@fixture
def scan_2d_mesh(cdl_scanner):
    """
    Return a scan representing a 'typical' testfile with a mesh.

    Being re-created for every test, this can be post-modified to exercise the
    individual conformance tests.
    (which is easier than making modified actual files or CDL strings)

    """
    test_cdl = """
    netcdf data_C4 {
    dimensions:
        dim0 = 6 ;
        num_node = 8 ;
        num_vertices = 4 ;

    variables:
        double sample_data(dim0) ;
            sample_data:long_name = "sample_data" ;
            sample_data:coordinates = "latitude longitude" ;
            sample_data:location = "face" ;
            sample_data:mesh = "topology" ;
        double latitude(dim0) ;
            latitude:units = "degrees_north" ;
            latitude:standard_name = "latitude" ;
            latitude:long_name = "latitude of 2D face centres" ;
            latitude:bounds_long_name = "latitude of 2D mesh nodes." ;
        double longitude(dim0) ;
            longitude:units = "degrees_east" ;
            longitude:standard_name = "longitude" ;
            longitude:long_name = "longitude of 2D face centres" ;
            longitude:bounds_long_name = "longitude of 2D mesh nodes." ;
        double node_lat(num_node) ;
            node_lat:standard_name = "latitude" ;
            node_lat:long_name = "latitude of 2D mesh nodes." ;
            node_lat:units = "degrees_north" ;
        double node_lon(num_node) ;
            node_lon:standard_name = "longitude" ;
            node_lon:long_name = "longitude of 2D mesh nodes." ;
            node_lon:units = "degrees_east" ;
        int face_nodes(dim0, num_vertices) ;
            face_nodes:long_name = "Map every face to its corner nodes." ;
            face_nodes:cf_role = "face_node_connectivity" ;
            face_nodes:start_index = 1 ;
        int topology ;
            topology:cf_role = "mesh_topology" ;
            topology:topology_dimension = 2L ;
            topology:node_coordinates = "node_lat node_lon" ;
            topology:face_coordinates = "latitude longitude" ;
            topology:face_node_connectivity = "face_nodes" ;
            topology:face_dimension = "dim0" ;
            topology:long_name = "Topology data of 2D unstructured mesh" ;

    // global attributes:
            :Conventions = "UGRID-1.0" ;
    }
    """
    return cdl_scanner.scan(test_cdl)


class TestCheckDataset:
    def _check_dataset(self, scan):
        # Conformance-check the given scan.
        return check_dataset(scan, print_summary=False, print_results=False)

    @staticmethod
    def _expect_notes(statements, expected_notes):
        # Test that each entry in 'expected' matches one of 'statements',
        # and that all 'statements' were matched.
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
        # if one can be found.
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
        assert set(expected_notes) == set(actual_notes)

    def test_basic(self, scan_2d_mesh):
        logs = self._check_dataset(scan_2d_mesh)
        self._expect_notes(
            logs, []
        )  # *NO* recorded statements, as there is nothing wrong !

    def test_mesh_missing_cf_role(self, scan_2d_mesh):
        del scan_2d_mesh.variables["topology"].attributes["cf_role"]
        logs = self._check_dataset(scan_2d_mesh)
        self._expect_notes(
            logs,
            [
                (
                    "R101",
                    "no \"cf_role\" property, which should be 'mesh_topology'",
                )
            ],
        )

    def test_mesh_bad_cf_role(self, scan_2d_mesh):
        scan_2d_mesh.variables["topology"].attributes[
            "cf_role"
        ] = "something odd"
        logs = self._check_dataset(scan_2d_mesh)
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

    def test_mesh_no_topology_dimension(self, scan_2d_mesh):
        del scan_2d_mesh.variables["topology"].attributes["topology_dimension"]
        logs = self._check_dataset(scan_2d_mesh)
        self._expect_notes(logs, [("R103", 'no "topology_dimension"')])

    def test_mesh_unknown_topology_dimension(self, scan_2d_mesh):
        scan_2d_mesh.variables["topology"].attributes["topology_dimension"] = 4
        logs = self._check_dataset(scan_2d_mesh)
        self._expect_notes(logs, [("R104", "not 0, 1 or 2")])

    def test_nonexistent_mesh(self, scan_2d_mesh):
        scan_2d_mesh.variables["sample_data"].attributes["mesh"] = np.array(
            "other_mesh"
        )
        logs = self._check_dataset(scan_2d_mesh)
        self._expect_notes(
            logs, [("R502", 'no "other_mesh" variable in the dataset.')]
        )

    def test_bad_mesh_name(self, scan_2d_mesh):
        scan_2d_mesh.variables["sample_data"].attributes["mesh"] = np.array(
            "this that other"
        )
        logs = self._check_dataset(scan_2d_mesh)
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

    def test_empty_mesh_name(self, scan_2d_mesh):
        scan_2d_mesh.variables["sample_data"].attributes["mesh"] = np.array("")
        logs = self._check_dataset(scan_2d_mesh)
        self._expect_notes(
            logs,
            [("R502", "\"mesh=''\", which is not a valid variable name.")],
        )
