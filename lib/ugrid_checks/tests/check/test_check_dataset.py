"""
Tests for ugrid_checks.check.check_dataset

"""
import logging
import re

import numpy as np
from pytest import fixture
from ugrid_checks.check import check_dataset
from ugrid_checks.nc_dataset_scan import NcDimSummary, NcVariableSummary
from ugrid_checks.tests import cdl_scanner, next_mesh, next_var

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
        # Copy 'topology' mesh and all its components to 'topology_2' etc
        next_mesh(scan_0d_mesh, "topology")
        # Fix the new node-coord variables to use the OLD nodes dim
        for xy in ("lon", "lat"):
            new_node_coord = scan_0d_mesh.variables[f"node_{xy}_2"]
            assert new_node_coord.dimensions == ["num_node_2"]
            new_node_coord.dimensions = ["num_node"]
        msg = (
            'Dimension "num_node" is mapped by both mesh "topology" and '
            'mesh "topology_2"'
        )
        self.check(scan_0d_mesh, "A104", msg)

    def test_a104_dataset_shared_meshdims_3(self, scan_0d_mesh):
        # Create 2 additional meshes
        next_mesh(scan_0d_mesh, "topology")  # makes 'topology_2'
        next_mesh(scan_0d_mesh, "topology_2")  # makes 'topology_3'
        # Fix the new node-coord variables to use the OLD nodes dim
        for imesh in (2, 3):
            for xy in ("lon", "lat"):
                new_node_coord = scan_0d_mesh.variables[f"node_{xy}_{imesh}"]
                assert new_node_coord.dimensions == [f"num_node_{imesh}"]
                new_node_coord.dimensions = ["num_node"]
        msg = (
            'Dimension "num_node" is mapped by multiple meshes : '
            '"topology", "topology_2" and "topology_3".'
        )
        self.check(scan_0d_mesh, "A104", msg)


class TestChecker_MeshVariables(DatasetChecker):
    # Simplified fixtures, when the meshvar is what is wanted.
    @fixture
    def scan_2d_and_meshvar(self, scan_2d_mesh):
        return scan_2d_mesh, scan_2d_mesh.variables["topology"]

    @fixture
    def scan_1d_and_meshvar(self, scan_1d_mesh):
        return scan_1d_mesh, scan_1d_mesh.variables["topology"]

    @fixture
    def scan_0d_and_meshvar(self, scan_0d_mesh):
        return scan_0d_mesh, scan_0d_mesh.variables["topology"]

    # Test mesh-variable checking.
    def test_r101_mesh_missing_cf_role(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
        del meshvar.attributes["cf_role"]
        msg = "no \"cf_role\" property, which should be 'mesh_topology'"
        self.check(scan, "R101", msg)

    def test_r102_mesh_bad_cf_role(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
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

    def test_r103_mesh_no_topology_dimension(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
        del meshvar.attributes["topology_dimension"]
        self.check(scan, "R103", 'no "topology_dimension"')

    def test_r104_mesh_unknown_topology_dimension(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
        meshvar.attributes["topology_dimension"] = 4
        self.check(scan, "R104", "not 0, 1 or 2")

    def test_r105_r107_mesh_badcoordattr_nonstring(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
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

    def test_r105_r107_mesh_badcoordattr_empty(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
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

    def test_r105_r107_mesh_badcoordattr_invalidname(
        self, scan_2d_and_meshvar
    ):
        scan, meshvar = scan_2d_and_meshvar
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

    def test_r105_r107_mesh_badcoordattr_missingvar(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
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

    def test_r108_mesh_badconn_empty(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
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

    def test_r108a_mesh_badconn_multiplevars(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
        # Checking for multiple entries in a connectivity attribute.
        # Create extra custom variables, copying 'node_lat'
        next_var(scan, "node_lat")
        next_var(scan, "node_lat_2")
        # point a connectivity at those
        meshvar.attributes["face_node_connectivity"] = "node_lat_2 node_lat_3"

        msg = (
            "face_node_connectivity=\"\\['node_lat_2', 'node_lat_3'\\]\""
            ", which contains 2 names, instead of 1."
        )
        self.check(scan, "", msg)

    def test_r109_mesh_missing_node_coords(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
        del meshvar.attributes["node_coordinates"]
        msg = "does not have a 'node_coordinates' attribute"
        self.check(scan, "R109", msg)

    def test_r110_mesh_topologydim0_extra_edgeconn(self, scan_0d_and_meshvar):
        scan, meshvar = scan_0d_and_meshvar
        # Add extra dims and a variable, to mimic edge-node connectivity.
        # (which is not valid in a 0-d mesh)
        scan.dimensions["edges"] = NcDimSummary(3)
        scan.dimensions["edge_ends"] = NcDimSummary(2)
        edgenodes_name = "fake_edge_nodes"
        scan.variables[edgenodes_name] = NcVariableSummary(
            name=edgenodes_name,
            dimensions=["edges", "edge_ends"],
            shape=(3, 2),
            dtype=np.dtype(np.int32),
            attributes={"cf_role": "edge_node_connectivity"},
        )
        meshvar.attributes["edge_node_connectivity"] = edgenodes_name
        msg = (
            'has "topology_dimension=0", but the presence of.*'
            "'edge_node_connectivity'.*implies it should be 1."
        )
        self.check(scan, "R110", msg)

    def test_r111_mesh_topologydim1_missing_edgeconn(
        self, scan_1d_and_meshvar
    ):
        scan, meshvar = scan_1d_and_meshvar
        del meshvar.attributes["edge_node_connectivity"]
        # Remove this, just to avoid an additional error
        del meshvar.attributes["edge_dimension"]
        # Avoid checking the data-variable, which now has a dim problem.
        del scan.variables["sample_data"].attributes["mesh"]
        msg = 'has "topology_dimension=1", but.*' "no 'edge_node_connectivity'"
        self.check(scan, "R111", msg)

    # NOTE: R112 is to be removed

    def test_r113_mesh_topologydim2_missing_faceconn(
        self, scan_1d_and_meshvar
    ):
        scan, meshvar = scan_1d_and_meshvar
        meshvar.attributes["topology_dimension"] = 2
        msg = 'has "topology_dimension=2", but.*' "no 'face_node_connectivity'"
        self.check(scan, "R113", msg)

    def test_r113_mesh_topologydim0_extra_faceconn(self, scan_0d_and_meshvar):
        scan, meshvar = scan_0d_and_meshvar
        # Add extra dims + a variable, to mimic face-node connectivity.
        # (which is not valid in a 0-d mesh)
        scan.dimensions["faces"] = NcDimSummary(3)
        scan.dimensions["face_vertices"] = NcDimSummary(4)
        facenodes_name = "fake_face_nodes"
        scan.variables[facenodes_name] = NcVariableSummary(
            name=facenodes_name,
            dimensions=["faces", "face_vertices"],
            shape=(3, 4),
            dtype=np.dtype(np.int32),
            attributes={"cf_role": "face_node_connectivity"},
        )
        meshvar.attributes["face_node_connectivity"] = facenodes_name
        msg = (
            'has "topology_dimension=0", but the presence of.*'
            "'face_node_connectivity'.*implies it should be 2."
        )
        self.check(scan, "R113", msg)

    def test_r114_mesh_topologydim1_extra_boundsconn(
        self, scan_1d_and_meshvar
    ):
        scan, meshvar = scan_1d_and_meshvar
        # Add extra dims and a variable, to mimic a bounds connectivity.
        # (which is not valid in a 1-d mesh)
        scan.dimensions["bounds"] = NcDimSummary(3)
        scan.dimensions["bounds_ends"] = NcDimSummary(2)
        boundnodes_name = "fake_bounds"
        scan.variables[boundnodes_name] = NcVariableSummary(
            name=boundnodes_name,
            dimensions=["bounds", "bounds_ends"],
            shape=(3, 2),
            dtype=np.dtype(np.int32),
            attributes={"cf_role": "boundary_node_connectivity"},
        )
        meshvar.attributes["boundary_node_connectivity"] = boundnodes_name
        msg = (
            'has a "boundary_node_connectivity".*'
            'there is no "face_node_connectivity"'
        )
        self.check(scan, "R114", msg)

    def _check_w_r305(self, scan, errcode, msg):
        # For multiple cases which generate an annoying extra R305
        statements = [
            (errcode, msg),
            (
                "R305",
                "does not contain any element dimension of the parent mesh",
            ),
        ]
        self.check(scan, statements=statements)

    def test_r115_mesh_edgedim_unknown(self, scan_1d_and_meshvar):
        scan, meshvar = scan_1d_and_meshvar
        meshvar.attributes["edge_dimension"] = "unknown_dim"
        # Avoid checking the data-variable, which now has a dim problem.
        del scan.variables["sample_data"].attributes["mesh"]
        msg = 'edge_dimension="unknown_dim".* not a dimension'
        self._check_w_r305(scan, "R115", msg)

    def test_r117_mesh_facedim_unknown(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
        meshvar.attributes["face_dimension"] = "unknown_dim"
        # Avoid checking the data-variable, which now has a dim problem.
        del scan.variables["sample_data"].attributes["mesh"]
        msg = 'face_dimension="unknown_dim".* not a dimension'
        self._check_w_r305(scan, "R117", msg)

    def test_r116_mesh_missing_needed_edgedim(self, scan_2d_and_meshvar):
        # Check that, if some edge connectivities have a non-standard dim
        # order, we then require an edge-dim attribute.
        scan, meshvar = scan_2d_and_meshvar

        # First add edges and a face-edges map to the 2d mesh
        # (because there are no optional edge connectivities in a 1d mesh)
        scan.dimensions["edge_dim"] = NcDimSummary(3, False)
        scan.dimensions["n_edge_ends"] = NcDimSummary(2, False)
        edgenodes_name = "edge_nodes_var"
        edgenodes_conn = NcVariableSummary(
            name=edgenodes_name,
            dimensions=["edge_dim", "n_edge_ends"],
            shape=(3, 2),
            dtype=np.dtype(np.int64),
            attributes={"cf_role": "edge_node_connectivity"},
        )
        # Now add the (optional) edge-face connectivity.
        edgeface_name = "edge_faces_var"
        edgeface_conn = NcVariableSummary(
            name=edgeface_name,
            dimensions=["edge_dim", "num_vertices"],
            shape=(6, 4),
            dtype=np.dtype(np.int64),
            attributes={"cf_role": "edge_face_connectivity"},
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

    def test_r118_mesh_missing_needed_facedim_2(self, scan_2d_and_meshvar):
        # Check that, if some edge connectivities have a non-standard dim
        # order, we then require an edge-dim attribute.
        scan, meshvar = scan_2d_and_meshvar

        # Add a face-face map to the 2d mesh
        faceface_name = "face_face_var"
        faceface_conn = NcVariableSummary(
            name=faceface_name,
            dimensions=["face_dim", "n_vertices"],
            shape=(6, 4),
            dtype=np.dtype(np.int64),
            attributes={"cf_role": "face_face_connectivity"},
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

    def test_r119_mesh_faceface_no_faces(self, scan_1d_and_meshvar):
        scan, meshvar = scan_1d_and_meshvar
        # Add extra dims + a variable to mimic a 'face-face' connectivity.
        # (which is not valid in a 1-d mesh)
        scan.dimensions["faces"] = NcDimSummary(3)
        scan.dimensions["face_N_faces"] = NcDimSummary(4)
        faceface_name = "fake_face_faces"
        scan.variables[faceface_name] = NcVariableSummary(
            name=faceface_name,
            dimensions=["faces", "face_N_faces"],
            shape=(3, 4),
            dtype=np.dtype(np.int32),
            attributes={"cf_role": "face_face_connectivity"},
        )
        meshvar.attributes["face_face_connectivity"] = faceface_name
        msg = (
            'has a "face_face_connectivity".*'
            'there is no "face_node_connectivity"'
        )
        self._check_w_r305(scan, "R119", msg)

    def test_r120_mesh_faceedge_no_faces(self, scan_1d_and_meshvar):
        scan, meshvar = scan_1d_and_meshvar
        # Add extra dims + a variable, to mimic face-edge connectivity.
        # (which is not valid for a 1-d mesh)
        scan.dimensions["faces"] = NcDimSummary(5)
        scan.dimensions["face_N_edges"] = NcDimSummary(3)
        faceedges_name = "fake_face_edges"
        scan.variables[faceedges_name] = NcVariableSummary(
            name=faceedges_name,
            dimensions=["faces", "face_N_edges"],
            shape=(5, 3),
            dtype=np.dtype(np.int32),
            attributes={"cf_role": "face_edge_connectivity"},
        )
        meshvar.attributes["face_edge_connectivity"] = faceedges_name
        msg = (
            'has a "face_edge_connectivity".*'
            'there is no "face_node_connectivity"'
        )
        self._check_w_r305(scan, "R120", msg)

    def test_r120_mesh_faceedge_no_edges(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
        meshvar = scan.variables["topology"]
        # Add extra dims + a variable, to mimic face-edge connectivity.
        # (which is not valid, as the mesh has no edges)
        scan.dimensions["face_N_edges"] = NcDimSummary(3)
        faceedges_name = "fake_face_edges"
        scan.variables[faceedges_name] = NcVariableSummary(
            name=faceedges_name,
            dimensions=["faces", "face_N_edges"],
            shape=(6, 3),
            dtype=np.dtype(np.int32),
            attributes={"cf_role": "face_edge_connectivity"},
        )
        meshvar.attributes["face_edge_connectivity"] = faceedges_name
        msg = (
            'has a "face_edge_connectivity".*'
            'there is no "edge_node_connectivity"'
        )
        self._check_w_r305(scan, "R120", msg)

    def test_r121_mesh_edgeface_no_faces(self, scan_1d_and_meshvar):
        scan, meshvar = scan_1d_and_meshvar
        # Add extra dims + a variable, to mimic edge-face connectivity.
        # (which is not valid for a 1-d mesh)
        scan.dimensions["edges"] = NcDimSummary(5)
        scan.dimensions["edge_N_faces"] = NcDimSummary(3)
        edgefaces_name = "fake_edge_faces"
        scan.variables[edgefaces_name] = NcVariableSummary(
            name=edgefaces_name,
            dimensions=["edges", "edge_N_faces"],
            shape=(5, 3),
            dtype=np.dtype(np.int32),
            attributes={"cf_role": "edge_face_connectivity"},
        )
        meshvar.attributes["edge_face_connectivity"] = edgefaces_name
        msg = (
            'has a "edge_face_connectivity".*'
            'there is no "face_node_connectivity"'
        )
        self._check_w_r305(scan, "R121", msg)

    def test_r121_mesh_edgeface_no_edges(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
        # Add extra dims + a variable, to mimic edge-face connectivity.
        # (which is not valid as the mesh has no edges)
        scan.dimensions["edges"] = NcDimSummary(5)
        scan.dimensions["edge_N_faces"] = NcDimSummary(3)
        edgefaces_name = "fake_edge_faces"
        scan.variables[edgefaces_name] = NcVariableSummary(
            name=edgefaces_name,
            dimensions=["edges", "edge_N_faces"],
            shape=(5, 3),
            dtype=np.dtype(np.int32),
            attributes={"cf_role": "edge_face_connectivity"},
        )
        meshvar.attributes["edge_face_connectivity"] = edgefaces_name
        msg = (
            'has a "edge_face_connectivity".*'
            'there is no "edge_node_connectivity"'
        )
        self._check_w_r305(scan, "R121", msg)

    def test_a101_mesh_variable_dimensions(self, scan_0d_and_meshvar):
        scan, meshvar = scan_0d_and_meshvar
        meshvar.dimensions = ["num_node"]
        meshvar.shape = (8,)  # to make consistent, but probably unnecessary?
        self.check(scan, "A101", 'Mesh variable "topology" has dimensions')

    def test_a102_mesh_variable_stdname(self, scan_0d_and_meshvar):
        scan, meshvar = scan_0d_and_meshvar
        meshvar.attributes["standard_name"] = "air_temperature"
        self.check(
            scan,
            "A102",
            'Mesh variable "topology" has a "standard_name" attribute',
        )

    def test_a103_mesh_variable_units(self, scan_0d_and_meshvar):
        scan, meshvar = scan_0d_and_meshvar
        meshvar.attributes["units"] = "K"
        self.check(
            scan,
            "A103",
            'Mesh variable "topology" has a "units" attribute',
        )

    # Note: A104 is tested in TestChecker_Dataset

    def test_a105_mesh_invalid_boundarydim(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
        meshvar.attributes["boundary_dimension"] = "odd_name"
        msg = (
            'has an attribute "boundary_dimension", which is '
            "not a valid UGRID term"
        )
        self.check(scan, "", msg)  # TODO: will be "A105"

    def test_a105_mesh_invalid_nodedim(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
        meshvar.attributes["node_dimension"] = "odd_name"
        msg = (
            'has an attribute "node_dimension", which is '
            "not a valid UGRID term"
        )
        self.check(scan, "", msg)  # TODO: will be "A105"

    def test_a106_mesh_unwanted_edgedim(self, scan_0d_and_meshvar):
        scan, meshvar = scan_0d_and_meshvar
        meshvar.attributes["edge_dimension"] = "odd_name"
        msg = (
            'has an attribute "edge_dimension", which is not valid.*'
            'no "edge_node_connectivity"'
        )
        #  TODO: will be "A106" -- or possibly "Rxxx" ?
        self.check(scan, "", msg)

    def test_a106_mesh_unwanted_facedim(self, scan_0d_and_meshvar):
        scan, meshvar = scan_0d_and_meshvar
        meshvar.attributes["face_dimension"] = "odd_name"
        msg = (
            'has an attribute "face_dimension", which is not valid.*'
            'no "face_node_connectivity"'
        )
        #  TODO: will be "A106" -- or possibly "Rxxx" ?
        self.check(scan, "", msg)


class TestChecker_DataVariables(DatasetChecker):
    @fixture
    def scan_2d_and_datavar(self, scan_2d_mesh):
        return scan_2d_mesh, scan_2d_mesh.variables["sample_data"]

    def test_r502_datavar_empty_mesh(self, scan_2d_and_datavar):
        scan, datavar = scan_2d_and_datavar
        datavar.attributes["mesh"] = ""
        msg = r'mesh="", which is not a single variable name\.'
        self.check(scan, "R502", msg)

    def test_r502_datavar_invalid_mesh_name(self, scan_2d_and_datavar):
        scan, datavar = scan_2d_and_datavar
        datavar.attributes["mesh"] = "$123"
        msg = r'mesh="\$123", which is not a valid netcdf variable name\.'
        self.check(scan, "R502", msg)

    def test_r502_datavar_bad_mesh_dtype(self, scan_2d_and_datavar):
        scan, datavar = scan_2d_and_datavar
        datavar.attributes["mesh"] = 2.0
        msg = r'mesh="2.0", which is not a string value\.'
        self.check(scan, "R502", msg)

    def test_r502_datavar_multi_mesh_name(self, scan_2d_and_datavar):
        scan, datavar = scan_2d_and_datavar
        datavar.attributes["mesh"] = "this that other"
        msg = r'mesh="this that other", which is not a single variable name\.'
        self.check(scan, "R502", msg)

    def test_r502_datavar_missing_meshvar(self, scan_2d_and_datavar):
        scan, datavar = scan_2d_and_datavar
        datavar.attributes["mesh"] = "absent"
        msg = r'mesh="absent", which is not a variable in the dataset\.'
        self.check(scan, "R502", msg)

    # Test data-variable checking.
    def test_r502_datavar_nonexistent_mesh(self, scan_2d_and_datavar):
        scan, datavar = scan_2d_and_datavar
        datavar.attributes["mesh"] = "other_mesh"
        msg = r'mesh="other_mesh", which is not a variable in the dataset\.'
        self.check(scan, "R502", msg)


class TestChecker_Coords(DatasetChecker):
    @fixture
    def scan_2d_and_coordvar(self, scan_2d_mesh):
        return scan_2d_mesh, scan_2d_mesh.variables["node_lon"]

    # Test mesh-coordinate checking.
    def test_r201_coord_multidim(self, scan_2d_and_coordvar):
        scan, coord = scan_2d_and_coordvar
        coord.dimensions = coord.dimensions + ("num_vertices",)
        msg = '"node_lon".*one dimension.*has 2 dimensions'
        self.check(scan, "R201", msg)

    def test_r202_coord_wrongdim(self, scan_2d_and_coordvar):
        scan, coord = scan_2d_and_coordvar
        coord.dimensions = ("face_dim",)
        msg = (
            'dimension "face_dim", but the parent mesh '
            'node dimension is "num_node".'
        )
        self.check(scan, "R202", msg)

    def test_r203_coord_bounds_multiple_names(self, scan_2d_and_coordvar):
        scan, coord = scan_2d_and_coordvar
        # Add an invalid bounds attribute to the node_lon coord.
        coord.attributes["bounds"] = "a b"
        msg = (
            '"node_lon" within topology:node_coordinates has bounds="a b", '
            "which is not a single variable"
        )
        self.check(scan, "R203", msg)

    def test_r203_coord_bounds_bad_attrdtype(self, scan_2d_and_coordvar):
        scan, coord = scan_2d_and_coordvar
        # Add an invalid bounds attribute to the node_lon coord.
        coord.attributes["bounds"] = 2.0
        msg = r'has bounds="2.0", which is not a string value\.'
        self.check(scan, "R203", msg)

    def test_r203_coord_bounds_bad_name(self, scan_2d_and_coordvar):
        scan, coord = scan_2d_and_coordvar
        # Add an invalid bounds attribute to the node_lon coord.
        coord.attributes["bounds"] = "$123"
        msg = (
            r'"node_lon" within topology:node_coordinates has bounds="\$123", '
            "which is not a valid netcdf variable name"
        )
        self.check(scan, "R203", msg)

    def test_r203_coord_bounds_missing_var(self, scan_2d_and_coordvar):
        scan, coord = scan_2d_and_coordvar
        # Add an invalid bounds attribute to the node_lon coord.
        coord.attributes["bounds"] = "other_var"
        msg = (
            '"node_lon" within topology:node_coordinates has '
            'bounds="other_var", which is not a variable in the dataset'
        )
        self.check(scan, "R203", msg)

    def test_r203_coord_bounds_missing_element_dim(self, scan_2d_and_coordvar):
        scan, coord = scan_2d_and_coordvar
        # Add node-lon bounds, with inappropriate dims.
        scan.dimensions["extra1"] = NcDimSummary(3)
        scan.dimensions["extra2"] = NcDimSummary(2)
        nodelon_bds_name = "node_lons_bounds"
        scan.variables[nodelon_bds_name] = NcVariableSummary(
            name=nodelon_bds_name,
            dimensions=("extra1", "extra2"),
            shape=(3, 2),
            dtype=coord.dtype,
        )
        coord.attributes["bounds"] = nodelon_bds_name
        msg = (
            r"dimensions \('extra1', 'extra2'\).*"
            'does not include the parent variable dimension, "num_node".'
        )
        self.check(scan, "R203", msg)

    def test_r203_coord_bounds_bad_ndims(self, scan_2d_mesh):
        coord = scan_2d_mesh.variables["longitude"]
        # Add node-lon bounds, with wrong n-dims (but does include the parent).
        facelons_bds_name = "face_lons_bounds"
        scan_2d_mesh.variables[facelons_bds_name] = NcVariableSummary(
            name=facelons_bds_name,
            dimensions=("face_dim",),
            shape=(6,),
            dtype=coord.dtype,
        )
        coord.attributes["bounds"] = facelons_bds_name
        msg = r"dimensions \('face_dim',\).* should be 2, instead of 1\."
        self.check(scan_2d_mesh, "R203", msg)

    def test_r203_coord_bounds_stdname_clash(self, scan_2d_mesh):
        coord = scan_2d_mesh.variables["longitude"]
        # Add node-lon bounds, with wrong n-dims (but does include the parent).
        facelons_bds_name = "face_lons_bounds"
        scan_2d_mesh.variables[facelons_bds_name] = NcVariableSummary(
            name=facelons_bds_name,
            dimensions=("face_dim", "num_vertices"),
            shape=(6,),
            dtype=coord.dtype,
            attributes={"standard_name": "junk"},
        )
        coord.attributes["bounds"] = facelons_bds_name
        msg = (
            'standard_name="junk", which does not match the parent '
            r'standard_name of "longitude"\.'
        )
        self.check(scan_2d_mesh, "R203", msg)

    def test_r203_coord_bounds_stdname_noparentstdname(self, scan_2d_mesh):
        coord = scan_2d_mesh.variables["longitude"]
        # Add node-lon bounds, with wrong n-dims (but does include the parent).
        facelons_bds_name = "face_lons_bounds"
        scan_2d_mesh.variables[facelons_bds_name] = NcVariableSummary(
            name=facelons_bds_name,
            dimensions=("face_dim", "num_vertices"),
            shape=(6,),
            dtype=coord.dtype,
            attributes={"standard_name": coord.attributes["standard_name"]},
        )
        coord.attributes["bounds"] = facelons_bds_name
        # Fix so the bounds has a standard-name, but the parent does not
        del coord.attributes["standard_name"]
        msg = (
            'standard_name="longitude", which does not match the parent '
            r'standard_name of "<none>"\.'
        )
        # N.B. this causes an additional A203 statement, which can't be avoided
        msg2 = 'coordinate.* "longitude".* has no "standard_name"'
        self.check(scan_2d_mesh, statements=[("R203", msg), ("A203", msg2)])

    def test_r203_coord_bounds_units_clash(self, scan_2d_mesh):
        coord = scan_2d_mesh.variables["longitude"]
        # Add node-lon bounds, with wrong n-dims (but does include the parent).
        facelons_bds_name = "face_lons_bounds"
        scan_2d_mesh.variables[facelons_bds_name] = NcVariableSummary(
            name=facelons_bds_name,
            dimensions=("face_dim", "num_vertices"),
            shape=(6,),
            dtype=coord.dtype,
            attributes={"units": "junk"},
        )
        coord.attributes["bounds"] = facelons_bds_name
        msg = (
            'units="junk", which does not match the parent '
            r'units of "degrees_east"\.'
        )
        self.check(scan_2d_mesh, "R203", msg)

    def test_r203_coord_bounds_units_noparentunits(self, scan_2d_mesh):
        coord = scan_2d_mesh.variables["longitude"]
        # Add node-lon bounds, with wrong n-dims (but does include the parent).
        facelons_bds_name = "face_lons_bounds"
        scan_2d_mesh.variables[facelons_bds_name] = NcVariableSummary(
            name=facelons_bds_name,
            dimensions=("face_dim", "num_vertices"),
            shape=(6,),
            dtype=coord.dtype,
            attributes={"units": "degrees_east"},
        )
        coord.attributes["bounds"] = facelons_bds_name
        # Fix so the bounds has a standard-name, but the parent does not
        del coord.attributes["units"]
        msg = (
            'units="degrees_east", which does not match the parent '
            r'units of "<none>"\.'
        )
        # N.B. this causes an additional A204 statement, which can't be avoided
        msg2 = 'coordinate.* "longitude".* has no "units" attribute'
        self.check(scan_2d_mesh, statements=[("R203", msg), ("A204", msg2)])

    #
    # Advisory warnings
    #

    def test_a201_coord_multiple_meshes(self, scan_2d_and_coordvar):
        scan, coord = scan_2d_and_coordvar
        # Copy 'topology' mesh and all its components to make 'topology_2'
        mesh2 = next_mesh(scan, "topology")
        # Reference the old 'node_lon' variable in the new 'topology_2' mesh
        assert mesh2.attributes["node_coordinates"] == "node_lat_2 node_lon_2"
        mesh2.attributes["node_coordinates"] = "node_lat_2 node_lon"
        msg = (
            '"node_lon" is referenced by multiple mesh variables : '
            "topology:node_coordinates, topology_2:node_coordinates."
        )
        # Note: we also get a '2 meshes using 1 dim' error
        # - this can't be avoided.
        msg2 = (
            'has dimension "num_node", but the parent mesh node dimension '
            'is "num_node_2"'
        )
        self.check(scan, statements=[("A201", msg), ("R202", msg2)])

    def test_a202_coord_not_floating_point(self, scan_2d_and_coordvar):
        scan, coord = scan_2d_and_coordvar
        coord.dtype = np.dtype(np.int32)
        msg = (
            'variable "node_lon" within topology:node_coordinates has '
            r'dtype "int32", which is not floating-point\.'
        )
        self.check(scan, "A202", msg)

    def test_a203_coord_no_stdname(self, scan_2d_and_coordvar):
        scan, coord = scan_2d_and_coordvar
        del coord.attributes["standard_name"]
        msg = 'has no "standard_name" attribute'
        self.check(scan, "A203", msg)

    def test_a204_coord_no_units(self, scan_2d_and_coordvar):
        scan, coord = scan_2d_and_coordvar
        del coord.attributes["units"]
        msg = 'has no "units" attribute'
        self.check(scan, "A204", msg)

    # A205 : bounds data matching expected -- not yet implemented


class TestChecker_Connectivities(DatasetChecker):
    @fixture
    def scan_2d_and_connvar(self, scan_2d_mesh):
        return scan_2d_mesh, scan_2d_mesh.variables["face_nodes"]

    def test_r301_conn_missing_cfrole(self, scan_2d_and_connvar):
        scan, conn = scan_2d_and_connvar
        del conn.attributes["cf_role"]
        msg = "\"face_nodes\".* has no 'cf_role' attribute."
        self.check(scan, "R301", msg)

    def test_r302_conn_bad_cfrole(self, scan_2d_and_connvar):
        scan, conn = scan_2d_and_connvar
        # Set to a valid CF discrete-sampling term, to avoid a warning about
        # valid CF values (a possible future check)
        conn.attributes["cf_role"] = "profile_id"
        msg = (
            '"face_nodes".* has cf_role="profile_id".*'
            "not a valid UGRID connectivity"
        )
        self.check(scan, "R302", msg)

    def test_r303_conn_cfrole_differentfromparent(self, scan_2d_and_connvar):
        scan, conn = scan_2d_and_connvar
        conn.attributes["cf_role"] = "face_face_connectivity"
        msg = (
            'cf_role="face_face_connectivity".*'
            "different from its role in the parent.*"
            '"face_node_connectivity"'
        )
        self.check(scan, "R303", msg)

    def test_r304_conn_bad_ndims(self, scan_2d_and_connvar):
        scan, conn = scan_2d_and_connvar
        conn.dimensions = conn.dimensions + ("num_vertices",)
        msg = (
            r"dimensions \('face_dim', 'num_vertices', 'num_vertices'\), "
            "of which there are 3, instead of 2"
        )
        self.check(scan, "R304", msg)

    def test_r305_conn_no_parent_dim(self, scan_2d_and_connvar):
        scan, conn = scan_2d_and_connvar
        conn.dimensions = ("num_vertices", "num_vertices")
        msg = (
            r"dimensions \('num_vertices', 'num_vertices'\).*"
            "does not contain any element dimension of the parent mesh"
        )
        self.check(scan, "R305", msg)

    def test_r306_conn_no_nonparent_dim(self, scan_2d_and_connvar):
        scan, conn = scan_2d_and_connvar
        conn.dimensions = ("face_dim", "face_dim")
        msg = (
            r"dimensions \('face_dim', 'face_dim'\).*"
            "does not contain any dimension which is "
            "not an element dimension of the parent mesh"
        )
        self.check(scan, "R306", msg)

    def test_r307_conn_wrong_element_dim(self, scan_2d_and_connvar):
        scan, conn = scan_2d_and_connvar
        conn.dimensions = ("num_node", "num_vertices")
        msg = (
            "does not include the expected face dimension "
            r'of the parent mesh, "face_dim"\.'
        )
        self.check(scan, "R307", msg)

    @staticmethod
    def _add_edges(scan):
        meshvar = scan.variables["topology"]
        # Add edges, and an edge-node connectivity ..
        scan.dimensions["edges"] = NcDimSummary(5)
        scan.dimensions["n_edge_ends"] = NcDimSummary(2)
        edgenodes_name = "edge_nodes"
        scan.variables[edgenodes_name] = NcVariableSummary(
            name=edgenodes_name,
            dimensions=["edges", "n_edge_ends"],
            shape=(5, 2),
            dtype=np.dtype(np.int32),
            attributes={"cf_role": "edge_node_connectivity"},
        )
        meshvar.attributes["edge_node_connectivity"] = edgenodes_name

    def test_r308_conn_bad_num_edge_ends(self, scan_2d_and_connvar):
        scan, conn = scan_2d_and_connvar
        self._add_edges(scan)
        # Adjust the length of the 'n_edge_ends' dim : this is the error.
        scan.dimensions["n_edge_ends"] = NcDimSummary(7)
        msg = (
            '"edge_nodes".* contains the non-mesh dimension "n_edge_ends", '
            r"but this has length 7 instead of 2\."
        )
        self.check(scan, "R308", msg)

    def test_r309_conn_bad_startindex_value(self, scan_2d_and_connvar):
        scan, conn = scan_2d_and_connvar
        conn.attributes["start_index"] = np.array(3, dtype=conn.dtype)
        msg = (
            '"face_nodes".* has a start_index of "3", '
            r"which is not either 0 or 1\."
        )
        self.check(scan, "R309", msg)

    # R310 -- not implementing data checks yet

    #
    # Advisory checks
    #

    def test_a301_conn_multiple_meshes(self, scan_2d_and_connvar):
        scan, conn = scan_2d_and_connvar

        # Copy 'topology' mesh and all its components to make 'topology_2'
        mesh2 = next_mesh(scan, "topology")
        # Reference the old 'face_nodes' variable in the new 'topology_2' mesh
        assert mesh2.attributes["face_node_connectivity"] == "face_nodes_2"
        mesh2.attributes["face_node_connectivity"] = "face_nodes"
        msg = (
            '"face_nodes" is referenced by multiple mesh variables : '
            "topology:face_node_connectivity, "
            r"topology_2:face_node_connectivity\."
        )
        # Note: we also get a 'doesn't match parent dim' error
        # - this can't easily be avoided.
        msg2 = "does not contain any element dimension of the parent mesh"
        self.check(scan, statements=[("A301", msg), ("R305", msg2)])

    def test_a302_conn_nonintegraltype(self, scan_2d_and_connvar):
        scan, conn = scan_2d_and_connvar
        conn.dtype = np.dtype(np.float32)
        # N.B. also adjust dtype of start_index, to avoid an A303
        conn.attributes["start_index"] = np.array(0.0, dtype=conn.dtype)
        msg = (
            '"face_nodes".* has dtype "float32", '
            r"which is not an integer type\."
        )
        self.check(scan, "A302", msg)

    def test_a303_conn_bad_startindex_type(self, scan_2d_and_connvar):
        scan, conn = scan_2d_and_connvar
        conn.attributes["start_index"] = np.array(1.0)
        msg = (
            'start_index of dtype "float64".* different from the '
            'variable dtype, "int32"'
        )
        self.check(scan, "A303", msg)

    def test_a304_conn_unexpected_fillvalue(self, scan_2d_and_connvar):
        # A "xxx_node_connectivity" should not have a fill-value
        scan, conn = scan_2d_and_connvar
        conn.attributes["_FillValue"] = np.array(-1, dtype=conn.dtype)
        msg = (
            '"face_nodes".* has a _FillValue attribute, which '
            "should not be present on a "
            r'"face_node_connectivity" connectivity\.'
        )
        self.check(scan, "A304", msg)

    # A305 checks for missing data -- not yet implemented

    def _add_edges_and_faceedges(self, scan):
        self._add_edges(scan)
        faceedge_varname = "face_edges"
        scan.variables[faceedge_varname] = NcVariableSummary(
            name=faceedge_varname,
            dimensions=["face_dim", "n_vertices"],
            shape=(6, 4),
            dtype=np.dtype(np.int32),
            attributes={"cf_role": "face_edge_connectivity"},
        )
        meshvar = scan.variables["topology"]
        meshvar.attributes["face_edge_connectivity"] = faceedge_varname

    def test_a306_conn_fillvalue_baddtype(self, scan_2d_mesh):
        # Add edges and a face-edge connectivity, because face_nodes can't have
        # missing indices.
        self._add_edges_and_faceedges(scan_2d_mesh)
        conn = scan_2d_mesh.variables["face_edges"]
        conn.attributes["_FillValue"] = np.array(-1, dtype=np.int16)
        msg = (
            '"face_edges".* has a _FillValue of dtype "int16", '
            r'which is different from the variable dtype, "int32"\.'
        )
        self.check(scan_2d_mesh, "A306", msg)

    def test_a307_conn_fillvalue_badvalue(self, scan_2d_mesh):
        # Add edges and a face-edge connectivity, because face_nodes can't have
        # missing indices.
        self._add_edges_and_faceedges(scan_2d_mesh)
        conn = scan_2d_mesh.variables["face_edges"]
        conn.attributes["_FillValue"] = np.array(999, dtype=conn.dtype)
        msg = r'"face_edges".* has _FillValue="999", which is not negative\.'
        self.check(scan_2d_mesh, "A307", msg)

    # A308 checks values within the dim range -- not yet implemented
