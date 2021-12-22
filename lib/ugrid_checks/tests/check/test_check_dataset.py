"""
Tests for ugrid_checks.check.check_dataset

"""
from copy import deepcopy
import logging
import re

import numpy as np
from pytest import fixture
from ugrid_checks.check import check_dataset
from ugrid_checks.nc_dataset_scan import NcDimSummary, NcVariableSummary
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


class DatasetChecker:
    # Generic helper functions for dataset-scan testing.
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

    def check(self, scan, code=None, message=None, statements=None):
        # Check statements generated by checking a dataset scan.
        # Usage forms :
        #   check(scan)  : expect no statements
        #   check(scan, code, message)  : expect exactly 1 statement
        #   check(scan, statements=[(code1, message1), ...])  : multiples
        logs = self._check_dataset(scan)
        if statements is None:
            if code is None and message is None:
                # Expect *no* statements.
                statements = []
            else:
                # Expect code+message to specify a single problem.
                statements = [(code, message)]
        self._expect_notes(logs, statements)


class TestChecker_Dataset(DatasetChecker):
    # Test dataset-level checking.
    def test_basic_2d_noerror(self, scan_2d_mesh):
        # Check that "2d" example, unmodified is all OK
        self.check(scan_2d_mesh)

    def test_basic_1d_noerror(self, scan_1d_mesh):
        # Check that "1d" example, unmodified is all OK
        self.check(scan_1d_mesh)

    def test_basic_0d_noerror(self, scan_0d_mesh):
        # Check that "0d" example, unmodified, is all OK
        self.check(scan_0d_mesh)

    def test_a104_dataset_shared_meshdims_2(self, scan_0d_mesh):
        # Create a minimal additional mesh, just a copy of the given one
        meshname2 = "mesh2"
        meshvar2 = deepcopy(scan_0d_mesh.variables["topology"])
        meshvar2.name = meshname2
        scan_0d_mesh.variables[meshname2] = meshvar2
        msg = (
            'Dimension "num_node" is mapped by both mesh "mesh2" and '
            'mesh "topology"'
        )
        self.check(scan_0d_mesh, "A104", msg)

    def test_a104_dataset_shared_meshdims_3(self, scan_0d_mesh):
        # Create 2 additional meshes
        for i_mesh in (2, 3):
            meshname = f"mesh_{i_mesh}"
            meshvar = deepcopy(scan_0d_mesh.variables["topology"])
            meshvar.name = meshname
            scan_0d_mesh.variables[meshname] = meshvar
        msg = (
            'Dimension "num_node" is mapped by multiple meshes : '
            '"mesh_2", "mesh_3" and "topology".'
        )
        self.check(scan_0d_mesh, "A104", msg)


class TestChecker_MeshVariables(DatasetChecker):
    # Simplified fixtures, when the meshvar is what is wanted.
    @fixture
    def meshvar_scan_2d(self, scan_2d_mesh):
        return scan_2d_mesh.variables["topology"], scan_2d_mesh

    @fixture
    def meshvar_scan_1d(self, scan_1d_mesh):
        return scan_1d_mesh.variables["topology"], scan_1d_mesh

    @fixture
    def meshvar_scan_0d(self, scan_0d_mesh):
        return scan_0d_mesh.variables["topology"], scan_0d_mesh

    # Test mesh-variable checking.
    def test_r101_mesh_missing_cf_role(self, meshvar_scan_2d):
        meshvar, scan = meshvar_scan_2d
        del meshvar.attributes["cf_role"]
        msg = "no \"cf_role\" property, which should be 'mesh_topology'"
        self.check(scan, "R101", msg)

    def test_r102_mesh_bad_cf_role(self, meshvar_scan_2d):
        meshvar, scan = meshvar_scan_2d
        meshvar.attributes["cf_role"] = "something odd"
        self.check(
            scan,
            statements=[
                ("R102", "should be 'mesh_topology'"),
                (
                    "",  # N.B. this one doesn't have a code yet
                    "not a valid UGRID cf_role",
                ),
            ],
        )

    def test_r103_mesh_no_topology_dimension(self, meshvar_scan_2d):
        meshvar, scan = meshvar_scan_2d
        del meshvar.attributes["topology_dimension"]
        self.check(scan, "R103", 'no "topology_dimension"')

    def test_r104_mesh_unknown_topology_dimension(self, meshvar_scan_2d):
        meshvar, scan = meshvar_scan_2d
        meshvar.attributes["topology_dimension"] = 4
        self.check(scan, "R104", "not 0, 1 or 2")

    def test_r105_r107_mesh_badcoordattr_nonstring(self, meshvar_scan_2d):
        meshvar, scan = meshvar_scan_2d
        # An invalid mesh-coord attribute.
        # This is always caused by a subsidiary specific error, so for testing
        # it doesn't much matter which we choose.
        meshvar.attributes["node_coordinates"] = 3
        self.check(
            scan,
            statements=[
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

    def test_r105_r107_mesh_badcoordattr_empty(self, meshvar_scan_2d):
        meshvar, scan = meshvar_scan_2d
        meshvar.attributes["node_coordinates"] = ""
        self.check(
            scan,
            statements=[
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

    def test_r105_r107_mesh_badcoordattr_invalidname(self, meshvar_scan_2d):
        meshvar, scan = meshvar_scan_2d
        meshvar.attributes["node_coordinates"] = "$123"
        self.check(
            scan,
            statements=[
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

    def test_r105_r107_mesh_badcoordattr_missingvar(self, meshvar_scan_2d):
        meshvar, scan = meshvar_scan_2d
        meshvar.attributes["node_coordinates"] = "$123"
        self.check(
            scan,
            statements=[
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

    def test_r108_mesh_badconn(self, meshvar_scan_2d):
        meshvar, scan = meshvar_scan_2d
        # Checking the catchall error for an invalid mesh-coord attribute.
        # This is always caused by a subsidiary specific error, so for testing
        # it doesn't much matter which we choose.
        meshvar.attributes["face_node_connectivity"] = ""
        self.check(
            scan,
            statements=[
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

    def test_r109_mesh_missing_node_coords(self, meshvar_scan_2d):
        meshvar, scan = meshvar_scan_2d
        del meshvar.attributes["node_coordinates"]
        msg = "does not have a 'node_coordinates' attribute"
        self.check(scan, "R109", msg)

    def test_r110_mesh_topologydim0_extra_edgeconn(self, meshvar_scan_0d):
        meshvar, scan = meshvar_scan_0d
        # This isn't a suitable target, but avoids a 'no such variable' error.
        meshvar.attributes["edge_node_connectivity"] = "node_lat"
        msg = (
            'has "topology_dimension=0", but the presence of.*'
            "'edge_node_connectivity'.*implies it should be 1."
        )
        self.check(scan, "R110", msg)

    def test_r111_mesh_topologydim1_missing_edgeconn(self, meshvar_scan_1d):
        meshvar, scan = meshvar_scan_1d
        del meshvar.attributes["edge_node_connectivity"]
        # Also remove this, just to avoid an additional error
        del meshvar.attributes["edge_dimension"]
        msg = 'has "topology_dimension=1", but.*' "no 'edge_node_connectivity'"
        self.check(scan, "R111", msg)

    # NOTE: R112 is to be removed

    def test_r113_mesh_topologydim2_missing_faceconn(self, meshvar_scan_1d):
        meshvar, scan = meshvar_scan_1d
        meshvar.attributes["topology_dimension"] = 2
        msg = 'has "topology_dimension=2", but.*' "no 'face_node_connectivity'"
        self.check(scan, "R113", msg)

    def test_r113_mesh_topologydim0_extra_faceconn(self, meshvar_scan_0d):
        meshvar, scan = meshvar_scan_0d
        meshvar.attributes["face_node_connectivity"] = "node_lat"
        msg = (
            'has "topology_dimension=0", but the presence of.*'
            "'face_node_connectivity'.*implies it should be 2."
        )
        self.check(scan, "R113", msg)

    def test_r114_mesh_topologydim1_extra_boundsconn(self, meshvar_scan_1d):
        meshvar, scan = meshvar_scan_1d
        meshvar.attributes["boundary_node_connectivity"] = "node_lat"
        msg = (
            'has a "boundary_node_connectivity".*'
            'there is no "face_node_connectivity"'
        )
        self.check(scan, "R114", msg)

    def test_r115_mesh_edgedim_unknown(self, meshvar_scan_1d):
        meshvar, scan = meshvar_scan_1d
        meshvar.attributes["edge_dimension"] = "unknown_dim"
        msg = 'edge_dimension="unknown_dim".* not a dimension'
        self.check(scan, "R115", msg)

    def test_r117_mesh_facedim_unknown(self, meshvar_scan_2d):
        meshvar, scan = meshvar_scan_2d
        meshvar.attributes["face_dimension"] = "unknown_dim"
        msg = 'face_dimension="unknown_dim".* not a dimension'
        self.check(scan, "R117", msg)

    def test_r116_mesh_missing_needed_edgedim(self, meshvar_scan_2d):
        # Check that, if some edge connectivities have a non-standard dim
        # order, we then require an edge-dim attribute.
        meshvar, scan = meshvar_scan_2d

        # First add edges and a face-edges map to the 2d mesh
        # (because there are no optional edge connectivities in a 1d mesh)
        scan.dimensions["edge_dim"] = NcDimSummary(3, False)
        scan.dimensions["n_edge_ends"] = NcDimSummary(2, False)
        edgenodes_name = "edge_nodes_var"
        edgenodes_conn = NcVariableSummary(
            name=edgenodes_name,
            dimensions=["edge_dim", "n_edge_ends"],
            shape=(3, 2),
            dtype=np.int64,
        )
        # Now add the (optional) edge-face connectivity.
        edgeface_name = "edge_faces_var"
        edgeface_conn = NcVariableSummary(
            name=edgeface_name,
            dimensions=["edge_dim", "num_vertices"],
            shape=(6, 4),
            dtype=np.int64,
        )
        scan.variables[edgenodes_name] = edgenodes_conn
        scan.variables[edgeface_name] = edgeface_conn
        meshvar.attributes["edge_node_connectivity"] = edgenodes_name
        meshvar.attributes["edge_face_connectivity"] = edgeface_name

        # Check this is still ok, with no edge-dim attribute.
        assert "edge_dimension" not in meshvar.attributes
        self.check(scan)

        # Now swap the dim-order of the 'edge_faces' variable.
        edgeface_conn.dimensions = edgeface_conn.dimensions[::-1]
        # That should trigger the error
        msg = (
            r'has no "edge_dimension" attribute.*'
            "edge connectivities with non-standard dimension order : "
            f'"{edgeface_name}"'
        )
        self.check(scan, "R116", msg)

        # Reinstating the 'edge_dimension' attribute should make it OK again.
        meshvar.attributes["edge_dimension"] = "edge_dim"
        self.check(scan)

    def test_r118_mesh_missing_needed_facedim_2(self, meshvar_scan_2d):
        # Check that, if some edge connectivities have a non-standard dim
        # order, we then require an edge-dim attribute.
        meshvar, scan = meshvar_scan_2d

        # Add a face-face map to the 2d mesh
        faceface_name = "face_face_var"
        faceface_conn = NcVariableSummary(
            name=faceface_name,
            dimensions=["face_dim", "n_vertices"],
            shape=(6, 4),
            dtype=np.int64,
        )
        scan.variables[faceface_name] = faceface_conn
        meshvar.attributes["face_face_connectivity"] = faceface_name

        # Remove the face-dim attribute
        del meshvar.attributes["face_dimension"]
        # Check that this it is still OK, with no checking errors
        self.check(scan)

        # Now swap the dim-order of the 'face_edges' variable.
        faceface_conn.dimensions = faceface_conn.dimensions[::-1]
        # That should trigger the error
        msg = (
            r'has no "face_dimension" attribute.*'
            "face connectivities with non-standard dimension order : "
            f'"{faceface_name}"'
        )
        self.check(scan, "R118", msg)

        # Reinstating the 'face_dimension' attribute should make it OK again.
        meshvar.attributes["face_dimension"] = "face_dim"
        self.check(scan)

    def test_r119_mesh_faceface_no_faces(self, meshvar_scan_1d):
        meshvar, scan = meshvar_scan_1d
        # This isn't a suitable target, but avoids a 'no such variable' error.
        meshvar.attributes["face_face_connectivity"] = "node_lat"
        msg = (
            'has a "face_face_connectivity".*'
            'there is no "face_node_connectivity"'
        )
        self.check(scan, "R119", msg)

    def test_r120_mesh_faceedge_no_faces(self, meshvar_scan_1d):
        meshvar, scan = meshvar_scan_1d
        # This isn't a suitable target, but avoids a 'no such variable' error.
        meshvar.attributes["face_edge_connectivity"] = "node_lat"
        msg = (
            'has a "face_edge_connectivity".*'
            'there is no "face_node_connectivity"'
        )
        self.check(scan, "R120", msg)

    def test_r120_mesh_faceedge_no_edges(self, meshvar_scan_2d):
        meshvar, scan = meshvar_scan_2d
        meshvar = scan.variables["topology"]
        # This isn't a suitable target, but avoids a 'no such variable' error.
        meshvar.attributes["face_edge_connectivity"] = "node_lat"
        msg = (
            'has a "face_edge_connectivity".*'
            'there is no "edge_node_connectivity"'
        )
        self.check(scan, "R120", msg)

    def test_r121_mesh_edgeface_no_faces(self, meshvar_scan_1d):
        meshvar, scan = meshvar_scan_1d
        # This isn't a suitable target, but avoids a 'no such variable' error.
        meshvar.attributes["edge_face_connectivity"] = "node_lat"
        msg = (
            'has a "edge_face_connectivity".*'
            'there is no "face_node_connectivity"'
        )
        self.check(scan, "R121", msg)

    def test_r121_mesh_edgeface_no_edges(self, meshvar_scan_2d):
        meshvar, scan = meshvar_scan_2d
        # This isn't a suitable target, but avoids a 'no such variable' error.
        meshvar.attributes["edge_face_connectivity"] = "node_lat"
        msg = (
            'has a "edge_face_connectivity".*'
            'there is no "edge_node_connectivity"'
        )
        self.check(scan, "R121", msg)

    def test_a101_mesh_variable_dimensions(self, meshvar_scan_0d):
        meshvar, scan = meshvar_scan_0d
        meshvar.dimensions = ["num_node"]
        meshvar.shape = (8,)  # to make consistent, but probably unnecessary?
        self.check(scan, "A101", 'Mesh variable "topology" has dimensions')

    def test_a102_mesh_variable_stdname(self, meshvar_scan_0d):
        meshvar, scan = meshvar_scan_0d
        meshvar.attributes["standard_name"] = "air_temperature"
        self.check(
            scan,
            "A102",
            'Mesh variable "topology" has a "standard_name" attribute',
        )

    def test_a103_mesh_variable_units(self, meshvar_scan_0d):
        meshvar, scan = meshvar_scan_0d
        meshvar.attributes["units"] = "K"
        self.check(
            scan,
            "A103",
            'Mesh variable "topology" has a "units" attribute',
        )

    # Note: A104 is tested in TestChecker_Dataset

    def test_a105_mesh_invalid_boundarydim(self, meshvar_scan_2d):
        meshvar, scan = meshvar_scan_2d
        meshvar.attributes["boundary_dimension"] = "odd_name"
        msg = (
            'has an attribute "boundary_dimension", which is '
            "not a valid UGRID term"
        )
        self.check(scan, "", msg)  # TODO: will be "A105"

    def test_a105_mesh_invalid_nodedim(self, meshvar_scan_2d):
        meshvar, scan = meshvar_scan_2d
        meshvar.attributes["node_dimension"] = "odd_name"
        msg = (
            'has an attribute "node_dimension", which is '
            "not a valid UGRID term"
        )
        self.check(scan, "", msg)  # TODO: will be "A105"

    def test_a106_mesh_unwanted_edgedim(self, meshvar_scan_0d):
        meshvar, scan = meshvar_scan_0d
        meshvar.attributes["edge_dimension"] = "odd_name"
        msg = (
            'has an attribute "edge_dimension", which is not valid.*'
            'no "edge_node_connectivity"'
        )
        #  TODO: will be "A106" -- or possibly "Rxxx" ?
        self.check(scan, "", msg)

    def test_a106_mesh_unwanted_facedim(self, meshvar_scan_0d):
        meshvar, scan = meshvar_scan_0d
        meshvar.attributes["face_dimension"] = "odd_name"
        msg = (
            'has an attribute "face_dimension", which is not valid.*'
            'no "face_node_connectivity"'
        )
        #  TODO: will be "A106" -- or possibly "Rxxx" ?
        self.check(scan, "", msg)


class TestChecker_DataVariables(DatasetChecker):
    @fixture
    def datavar_scan_2d(self, scan_2d_mesh):
        return scan_2d_mesh, scan_2d_mesh.variables["sample_data"]

    # Test data-variable checking.
    def test_r502_datavar_nonexistent_mesh(self, datavar_scan_2d):
        scan, datavar = datavar_scan_2d
        datavar.attributes["mesh"] = "other_mesh"
        self.check(scan, "R502", 'no "other_mesh" variable')

    def test_r502_datavar_bad_mesh_name(self, datavar_scan_2d):
        scan, datavar = datavar_scan_2d
        datavar.attributes["mesh"] = "this that other"
        msg = (
            "\"mesh='this that other'\", "
            "which is not a valid variable name."
        )
        self.check(scan, "R502", msg)

    def test_r502_datavar_empty_mesh_name(self, datavar_scan_2d):
        scan, datavar = datavar_scan_2d
        datavar.attributes["mesh"] = ""
        msg = "\"mesh=''\", which is not a valid variable name."
        self.check(scan, "R502", msg)

    def test_r502_datavar_missing_meshvar(self, datavar_scan_2d):
        scan, datavar = datavar_scan_2d
        datavar.attributes["mesh"] = "absent"
        msg = "mesh='absent'.*but there is no \"absent\" variable"
        self.check(scan, "R502", msg)
