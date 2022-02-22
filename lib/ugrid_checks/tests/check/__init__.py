"""
Some common test fixtures for checker testing.

"""
from pytest import fixture

from .. import cdl_scanner

cdl_scanner  # Yes, we do need this import.


@fixture
def simple_scan(cdl_scanner):
    """
    Return a scan representing a small mesh.
    As follows:
        mesh missing cf_role (but identified from datavar) --> R101
        node_lon_bounds is missing --> R203
        node_lat missing units --> A204
        invalid UGRID 'Conventions' attribute --> A903

    """
    test_cdl = """
    netcdf data_C4 {
    dimensions:
        num_node = 8 ;

    variables:
        int topology ;
            topology:cf_role = "mesh_topology" ;
            topology:topology_dimension = 0L ;
            topology:node_coordinates = "node_lat node_lon" ;
        double node_lat(num_node) ;
            node_lat:standard_name = "latitude" ;
            node_lat:units = "degrees_north" ;
        double node_lon(num_node) ;
            node_lon:standard_name = "longitude" ;
            node_lon:units = "degrees_east" ;
        double sample_data(num_node) ;
            sample_data:location = "node" ;
            sample_data:mesh = "topology" ;

    // global attributes:
            :Conventions = "UGRID-1.0" ;
    }
    """
    return cdl_scanner.scan(test_cdl)


@fixture
def simple_incorrect_scan_and_codes(simple_scan):
    """
    Return a scan and error codes, representing a small mesh which produces a
    couple of known requirement and advisory statements, and the relevant
    statement codes.

    Errors as follows:
        mesh missing cf_role (but identified from datavar) --> R101
        node_lon specifies a 'bounds', which is missing --> R203
        node_lat missing units --> A204
        invalid UGRID 'Conventions' attribute --> A903

    """
    scan = simple_scan
    # Remove topology:cf_role --> R101
    del scan.variables["topology"].attributes["cf_role"]
    # Add node_lon:bounds --> R203
    scan.variables["node_lon"].attributes["bounds"] = "missing_lon_bounds"
    # Remove node_lat:units --> A204
    del scan.variables["node_lat"].attributes["units"]
    # Invalidate global 'Conventions' UGRID part --> A903
    assert scan.attributes["Conventions"] == "UGRID-1.0"
    scan.attributes["Conventions"] = "UG-1.0"
    # Also return a list of the expected statement codes.
    expected_codes = ["R101", "R203", "A204", "A903"]
    return scan, expected_codes
