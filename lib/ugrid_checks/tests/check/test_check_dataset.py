"""
Tests for ugrid_checks.check.check_dataset

"""
import logging
import re

import numpy as np
from numpy import array
from pytest import fixture
from ugrid_checks.check import check_dataset
from ugrid_checks.nc_dataset_scan import NcVariableSummary
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
        face_dim = 6 ;
        num_node = 8 ;
        num_vertices = 4 ;

    variables:
        double sample_data(face_dim) ;
            sample_data:long_name = "sample_data" ;
            sample_data:coordinates = "latitude longitude" ;
            sample_data:location = "face" ;
            sample_data:mesh = "topology" ;
        double latitude(face_dim) ;
            latitude:units = "degrees_north" ;
            latitude:standard_name = "latitude" ;
            latitude:long_name = "latitude of 2D face centres" ;
            latitude:bounds_long_name = "latitude of 2D mesh faces." ;
        double longitude(face_dim) ;
            longitude:units = "degrees_east" ;
            longitude:standard_name = "longitude" ;
            longitude:long_name = "longitude of 2D face centres" ;
            longitude:bounds_long_name = "longitude of 2D mesh faces." ;
        double node_lat(num_node) ;
            node_lat:standard_name = "latitude" ;
            node_lat:long_name = "latitude of 2D mesh nodes." ;
            node_lat:units = "degrees_north" ;
        double node_lon(num_node) ;
            node_lon:standard_name = "longitude" ;
            node_lon:long_name = "longitude of 2D mesh nodes." ;
            node_lon:units = "degrees_east" ;
        int face_nodes(face_dim, num_vertices) ;
            face_nodes:long_name = "Map every face to its corner nodes." ;
            face_nodes:cf_role = "face_node_connectivity" ;
            face_nodes:start_index = 1 ;
        int topology ;
            topology:cf_role = "mesh_topology" ;
            topology:topology_dimension = 2L ;
            topology:node_coordinates = "node_lat node_lon" ;
            topology:face_coordinates = "latitude longitude" ;
            topology:face_node_connectivity = "face_nodes" ;
            topology:face_dimension = "face_dim" ;
            topology:long_name = "Topology data of 2D unstructured mesh" ;

    // global attributes:
            :Conventions = "UGRID-1.0" ;
    }
    """
    return cdl_scanner.scan(test_cdl)


@fixture
def scan_1d_mesh(cdl_scanner):
    """Return a scan representing a 1d (edges only) mesh."""
    test_cdl = """
    netcdf data_C4 {
    dimensions:
        edge_dim = 6 ;
        num_node = 8 ;
        num_ends = 2 ;

    variables:
        double sample_data(edge_dim) ;
            sample_data:long_name = "sample_data" ;
            sample_data:coordinates = "latitude longitude" ;
            sample_data:location = "edge" ;
            sample_data:mesh = "topology" ;
        double latitude(edge_dim) ;
            latitude:units = "degrees_north" ;
            latitude:standard_name = "latitude" ;
            latitude:long_name = "latitude of 2D edge centres" ;
            latitude:bounds_long_name = "latitude of 2D mesh edges." ;
        double longitude(edge_dim) ;
            longitude:units = "degrees_east" ;
            longitude:standard_name = "longitude" ;
            longitude:long_name = "longitude of 2D edge centres" ;
            longitude:bounds_long_name = "longitude of 2D mesh edges." ;
        double node_lat(num_node) ;
            node_lat:standard_name = "latitude" ;
            node_lat:long_name = "latitude of 2D mesh nodes." ;
            node_lat:units = "degrees_north" ;
        double node_lon(num_node) ;
            node_lon:standard_name = "longitude" ;
            node_lon:long_name = "longitude of 2D mesh nodes." ;
            node_lon:units = "degrees_east" ;
        int edge_nodes(edge_dim, num_ends) ;
            edge_nodes:long_name = "Map every edge to its endpoints." ;
            edge_nodes:cf_role = "edge_node_connectivity" ;
            edge_nodes:start_index = 1 ;
        int topology ;
            topology:cf_role = "mesh_topology" ;
            topology:topology_dimension = 1L ;
            topology:node_coordinates = "node_lat node_lon" ;
            topology:edge_coordinates = "latitude longitude" ;
            topology:edge_node_connectivity = "edge_nodes" ;
            topology:edge_dimension = "edge_dim" ;
            topology:long_name = "Topology data of 1D unstructured mesh" ;

    // global attributes:
            :Conventions = "UGRID-1.0" ;
    }
    """
    return cdl_scanner.scan(test_cdl)


@fixture
def scan_0d_mesh(cdl_scanner):
    """Return a scan representing a 0d (nodes only) mesh."""
    test_cdl = """
    netcdf data_C4 {
    dimensions:
        num_node = 8 ;
        num_ends = 2 ;

    variables:
        double sample_data(num_node) ;
            sample_data:long_name = "sample_data" ;
            sample_data:location = "node" ;
            sample_data:mesh = "topology" ;
        double node_lat(num_node) ;
            node_lat:standard_name = "latitude" ;
            node_lat:long_name = "latitude of 2D mesh nodes." ;
            node_lat:units = "degrees_north" ;
        double node_lon(num_node) ;
            node_lon:standard_name = "longitude" ;
            node_lon:long_name = "longitude of 2D mesh nodes." ;
            node_lon:units = "degrees_east" ;
        int topology ;
            topology:cf_role = "mesh_topology" ;
            topology:topology_dimension = 0L ;
            topology:node_coordinates = "node_lat node_lon" ;
            topology:long_name = "Topology data of 0D unstructured mesh" ;

    // global attributes:
            :Conventions = "UGRID-1.0" ;
    }
    """
    return cdl_scanner.scan(test_cdl)


class TestCheckDataset:
    #
    # Tests for the main "ugrid_checks.check_dataset" call.
    #
    # TODO: at present these tests mostly target "check_dataset_inner", while
    #   *not* actually testing the wrapper functions of "check_dataset".
    #   Actually, most really target "check_meshvar", so we should separate
    #   that all out, too.
    #
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
                ) and re.search(expected_msg, actual_msg):
                    expected_notes[i_expected] = actual_note
                    break  # Only ever take first match
        # This should result in a matching list.
        assert set(actual_notes) == set(expected_notes)

    def _expect_1(self, statements, code, message):
        self._expect_notes(statements, [(code, message)])

    def test_basic_2d_noerror(self, scan_2d_mesh):
        logs = self._check_dataset(scan_2d_mesh)
        self._expect_notes(
            logs, []
        )  # *NO* recorded statements, as there is nothing wrong !

    def test_basic_1d_noerror(self, scan_1d_mesh):
        logs = self._check_dataset(scan_1d_mesh)
        self._expect_notes(
            logs, []
        )  # *NO* recorded statements, as there is nothing wrong !

    def test_basic_0d_noerror(self, scan_0d_mesh):
        logs = self._check_dataset(scan_0d_mesh)
        self._expect_notes(
            logs, []
        )  # *NO* recorded statements, as there is nothing wrong !

    def test_r101_mesh_missing_cf_role(self, scan_2d_mesh):
        del scan_2d_mesh.variables["topology"].attributes["cf_role"]
        logs = self._check_dataset(scan_2d_mesh)
        msg = "no \"cf_role\" property, which should be 'mesh_topology'"
        self._expect_1(logs, "R101", msg)

    def test_r102_mesh_bad_cf_role(self, scan_2d_mesh):
        scan_2d_mesh.variables["topology"].attributes[
            "cf_role"
        ] = "something odd"
        logs = self._check_dataset(scan_2d_mesh)
        self._expect_notes(
            logs,
            [
                ("R102", "should be 'mesh_topology'"),
                (
                    "",  # N.B. this one doesn't have a code yet
                    "not a valid UGRID cf_role",
                ),
            ],
        )

    def test_r103_mesh_no_topology_dimension(self, scan_2d_mesh):
        del scan_2d_mesh.variables["topology"].attributes["topology_dimension"]
        logs = self._check_dataset(scan_2d_mesh)
        self._expect_1(logs, "R103", 'no "topology_dimension"')

    def test_r104_mesh_unknown_topology_dimension(self, scan_2d_mesh):
        scan_2d_mesh.variables["topology"].attributes["topology_dimension"] = 4
        logs = self._check_dataset(scan_2d_mesh)
        self._expect_1(logs, "R104", "not 0, 1 or 2")

    def test_r502_nonexistent_mesh(self, scan_2d_mesh):
        scan_2d_mesh.variables["sample_data"].attributes["mesh"] = array(
            "other_mesh"
        )
        logs = self._check_dataset(scan_2d_mesh)
        self._expect_1(logs, "R502", 'no "other_mesh" variable')

    def test_r502_datavar_bad_mesh_name(self, scan_2d_mesh):
        scan_2d_mesh.variables["sample_data"].attributes["mesh"] = array(
            "this that other"
        )
        logs = self._check_dataset(scan_2d_mesh)
        msg = (
            "\"mesh='this that other'\", "
            "which is not a valid variable name."
        )
        self._expect_1(logs, "R502", msg)

    def test_r502_datavar_empty_mesh_name(self, scan_2d_mesh):
        scan_2d_mesh.variables["sample_data"].attributes["mesh"] = array("")
        logs = self._check_dataset(scan_2d_mesh)
        msg = "\"mesh=''\", which is not a valid variable name."
        self._expect_1(logs, "R502", msg)

    def test_r502_datavar_missing_meshvar(self, scan_2d_mesh):
        scan_2d_mesh.variables["sample_data"].attributes["mesh"] = array(
            "absent"
        )
        logs = self._check_dataset(scan_2d_mesh)
        msg = "mesh='absent'.*but there is no \"absent\" variable"
        self._expect_1(logs, "R502", msg)

    def test_r118_mesh_missing_face_dimension(self, scan_2d_mesh):
        # Swap the dim order of the face_nodes_connectivity
        conn_var = scan_2d_mesh.variables["face_nodes"]
        conn_var.dimensions = conn_var.dimensions[::-1]

        # This in itself should be valid + generate no logs.
        logs = self._check_dataset(scan_2d_mesh)
        self._expect_notes(logs, [])

        # Now remove the 'node_dimension' attribute
        del scan_2d_mesh.variables["topology"].attributes["face_dimension"]

        # Unfortunately that doesn't work on it's own, as "the" face dimension
        # is now *determined* by the dimensions of the face-nodes conn.
        # So, add an extra (optional) connectivity, with "standard" dim order,
        # to conflict with the "face_nodes" conn.
        role = "face_face_connectivity"
        varname = "face_face"
        dims = ["face_dim", "num_vertices"]
        shape = [scan_2d_mesh.dimensions[name].length for name in dims]
        conn_var = NcVariableSummary(
            name=varname,
            dimensions=["face_dim", "num_vertices"],
            attributes={"cf_role": np.asanyarray(role)},
            shape=shape,
            dtype=np.dtype(np.int64),
            data=None,
        )
        scan_2d_mesh.variables[varname] = conn_var
        scan_2d_mesh.variables["topology"].attributes[role] = np.asanyarray(
            varname
        )

        logs = self._check_dataset(scan_2d_mesh)
        msg = r'no "face_dimension".*with non-standard dim.*: "face_face"\.'
        self._expect_1(logs, "R118", msg)

    def test_r105_r107_mesh_badcoordattr_nodecoords_nonstring(
        self, scan_2d_mesh
    ):
        # An invalid mesh-coord attribute.
        # This is always caused by a subsidiary specific error, so for testing
        # it doesn't much matter which we choose.
        scan_2d_mesh.variables["topology"].attributes[
            "node_coordinates"
        ] = array(3)
        logs = self._check_dataset(scan_2d_mesh)
        self._expect_notes(
            logs,
            [
                (
                    "R105",
                    (
                        "\"topology\" attribute 'node_coordinates'.*"
                        "does not have string type"
                    ),
                ),
                ("R107", '"topology".*not a list of variables in the dataset'),
            ],
        )

    def test_r105_r107_mesh_badcoordattr_nodecoords_empty(self, scan_2d_mesh):
        # An invalid mesh-coord attribute.
        # This is always caused by a subsidiary specific error, so for testing
        # it doesn't much matter which we choose.
        scan_2d_mesh.variables["topology"].attributes[
            "node_coordinates"
        ] = array("")
        logs = self._check_dataset(scan_2d_mesh)
        self._expect_notes(
            logs,
            [
                (
                    "R105",
                    (
                        '"topology" attribute \'node_coordinates\' = "".*'
                        "is not a valid list of netcdf variable"
                    ),
                ),
                ("R107", '"topology".*not a list of variables'),
            ],
        )

    def test_r105_r107_mesh_badcoordattr_nodecoords_invalidname(
        self, scan_2d_mesh
    ):
        # An invalid mesh-coord attribute.
        # This is always caused by a subsidiary specific error, so for testing
        # it doesn't much matter which we choose.
        scan_2d_mesh.variables["topology"].attributes[
            "node_coordinates"
        ] = array("$123")
        logs = self._check_dataset(scan_2d_mesh)
        self._expect_notes(
            logs,
            [
                (
                    "R105",
                    (
                        r'"topology" attribute \'node_coordinates\'.*"\$123"'
                        ".*not a valid netcdf variable name"
                    ),
                ),
                ("R107", '"topology".*not a list of variables'),
            ],
        )

    def test_r105_r107_mesh_badcoordattr_nodecoords_missingvar(
        self, scan_2d_mesh
    ):
        # An invalid mesh-coord attribute.
        # This is always caused by a subsidiary specific error, so for testing
        # it doesn't much matter which we choose.
        scan_2d_mesh.variables["topology"].attributes[
            "node_coordinates"
        ] = array("$123")
        logs = self._check_dataset(scan_2d_mesh)
        self._expect_notes(
            logs,
            [
                (
                    "R105",
                    (
                        r'"topology" attribute \'node_coordinates\'.*"\$123"'
                        ".*not a valid netcdf variable name"
                    ),
                ),
                ("R107", '"topology".*not a list of variables in the dataset'),
            ],
        )

    def test_r108_mesh_badconn(self, scan_2d_mesh):
        # Checking the catchall error for an invalid mesh-coord attribute.
        # This is always caused by a subsidiary specific error, so for testing
        # it doesn't much matter which we choose.
        meshvar = scan_2d_mesh.variables["topology"]
        meshvar.attributes["face_node_connectivity"] = array("")
        logs = self._check_dataset(scan_2d_mesh)
        self._expect_notes(
            logs,
            [
                (
                    "R105",
                    (
                        "\"topology\" attribute 'face_node_connectivity' "
                        '= "".*is not a valid list of netcdf variable'
                    ),
                ),
                (
                    "R108",
                    (
                        '"topology" attribute "face_node_connectivity" = ""'
                        ".*not a list of variables in the dataset"
                    ),
                ),
            ],
        )

    def test_r109_missing_node_coords(self, scan_2d_mesh):
        del scan_2d_mesh.variables["topology"].attributes["node_coordinates"]
        logs = self._check_dataset(scan_2d_mesh)
        msg = "does not have a 'node_coordinates' attribute"
        self._expect_1(logs, "R109", msg)

    def test_r110_mesh_topologydim0_extra_edgeconn(self, scan_0d_mesh):
        meshvar = scan_0d_mesh.variables["topology"]
        # An odd var to choose, but unchecked so avoids additional errors
        meshvar.attributes["edge_node_connectivity"] = array("node_lat")
        logs = self._check_dataset(scan_0d_mesh)
        msg = (
            'has "topology_dimension=0", but the presence of.*'
            "'edge_node_connectivity'.*implies it should be 1."
        )
        self._expect_1(logs, "R110", msg)

    def test_r111_mesh_topologydim1_missing_edgeconn(self, scan_1d_mesh):
        meshvar = scan_1d_mesh.variables["topology"]
        del meshvar.attributes["edge_node_connectivity"]
        # Also remove this, just to avoid an additional error
        del meshvar.attributes["edge_dimension"]
        logs = self._check_dataset(scan_1d_mesh)
        msg = 'has "topology_dimension=1", but.*' "no 'edge_node_connectivity'"
        self._expect_1(logs, "R111", msg)

    def test_r113_mesh_topologydim2_missing_faceconn(self, scan_1d_mesh):
        meshvar = scan_1d_mesh.variables["topology"]
        meshvar.attributes["topology_dimension"] = array(2)
        logs = self._check_dataset(scan_1d_mesh)
        msg = 'has "topology_dimension=2", but.*' "no 'face_node_connectivity'"
        self._expect_1(logs, "R113", msg)

    def test_r113_mesh_topologydim0_extra_faceconn(self, scan_0d_mesh):
        meshvar = scan_0d_mesh.variables["topology"]
        meshvar.attributes["face_node_connectivity"] = array("node_lat")
        logs = self._check_dataset(scan_0d_mesh)
        msg = (
            'has "topology_dimension=0", but the presence of.*'
            "'face_node_connectivity'.*implies it should be 2."
        )
        self._expect_1(logs, "R113", msg)

    def test_r114_mesh_topologydim1_extra_boundsconn(self, scan_1d_mesh):
        meshvar = scan_1d_mesh.variables["topology"]
        meshvar.attributes["boundary_node_connectivity"] = array("node_lat")
        logs = self._check_dataset(scan_1d_mesh)
        msg = (
            'has a "boundary_node_connectivity".*'
            'there is no "face_node_connectivity"'
        )
        self._expect_1(logs, "R114", msg)

    def test_r115_mesh_edgedim_unknown(self, scan_1d_mesh):
        meshvar = scan_1d_mesh.variables["topology"]
        meshvar.attributes["edge_dimension"] = array("unknown_dim")
        logs = self._check_dataset(scan_1d_mesh)
        msg = 'edge_dimension="unknown_dim".* not a dimension'
        self._expect_1(logs, "R115", msg)
