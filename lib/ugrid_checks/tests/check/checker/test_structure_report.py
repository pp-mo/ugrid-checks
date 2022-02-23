"""
Tests for :meth:`ugrid_checks.check.Checker.structure_report()`.

"""
import re

import numpy as np
from pytest import fixture
from ugrid_checks.nc_dataset_scan import NcDimSummary, NcVariableSummary

from .. import simple_scan
from ... import cdl_scanner
from ..test_check_dataset__checks import DatasetChecker, scan_0d_mesh

# Yes, we do need these imports.
cdl_scanner
simple_scan
scan_0d_mesh


from ugrid_checks.check import Checker


@fixture
def nomesh_scan(cdl_scanner):
    test_cdl = """
    netcdf data_C4 {
    dimensions:
        x = 8 ;

    variables:
        double var_1(x) ;
    }
    """
    scan = cdl_scanner.scan(test_cdl)
    return scan


@fixture
def nomesh_checker(nomesh_scan):
    return Checker(nomesh_scan)


@fixture
def simple_mesh_checker(simple_scan):
    # Add an extra non-mesh dim and variable, for nonmesh output testing.
    simple_scan.dimensions["x_dim"] = NcDimSummary(3)
    simple_scan.variables["x_data"] = NcVariableSummary(
        name="x_data", dimensions=("x_dim",), shape=(3,), dtype=np.int32
    )
    return Checker(simple_scan)


class Test_CheckReport(DatasetChecker):
    def expect_in_text(self, text, expects):
        """
        Check that a (list of) expected regular-expressions occur in a text,
        in the order given.

        If they aren't all present, produce an informative message showing
        which are missing.
        N.B. uses "assert list == []", for intelligible error messages.
        E.G.
            self.expect_in_text(['XXXX', '.*'], "this"") == []
            -->
                ['XXXX'] != []

                Expected :[]
                Actual   :['XXXX']

        """
        if isinstance(expects, str):
            expects = [expects]
        unmatched_expects = []
        for expect in expects:
            # Match each 'expect' entry to the result text
            match = re.search(expect, text, re.DOTALL)
            if match:
                # Item found.  Continue from after match
                text = text[match.end() :]
            else:
                # This not found.  Continue from same place
                unmatched_expects.append(expect)
        assert unmatched_expects == []

    def test_empty__nomesh__meshonly(self, nomesh_checker):
        result = nomesh_checker.structure_report()
        expect = "Meshes : <none>"
        self.expect_in_text(result, expect)

    def test_empty__nomesh__w_nonmesh(self, nomesh_checker):
        # Do "include_nonmesh", on a no-mesh dataset.
        result = nomesh_checker.structure_report(include_nonmesh=True)
        expects = [
            "Meshes : <none>",
            r'Non-mesh[^\n]*\n *dimensions:\n *"x"',
            r'variables:\n *"var_1"',
        ]
        self.expect_in_text(result, expects)

    def test_basic__meshonly(self, simple_mesh_checker):
        # Simple case : just for this, do a full text comparison.
        text = simple_mesh_checker.structure_report()
        text_lines = text.split("\n")
        expected_lines = [
            "Meshes",
            '    "topology"',
            '        node("num_node")',
            '            coordinates : "node_lat", "node_lon"',
            "",
            "Mesh Data Variables",
            '    "sample_data"',
            '        mesh : "topology"',
            '        location : "node"',
        ]
        assert text_lines == expected_lines

    def test_basic__w_nonmesh(self, simple_mesh_checker):
        # Check the "nonmesh" output : another full-text check.
        text = simple_mesh_checker.structure_report(include_nonmesh=True)
        text_lines = text.split("\n")
        expected_lines = [
            "Meshes",
            '    "topology"',
            '        node("num_node")',
            '            coordinates : "node_lat", "node_lon"',
            "",
            "Mesh Data Variables",
            '    "sample_data"',
            '        mesh : "topology"',
            '        location : "node"',
            "",
            "Non-mesh variables and/or dimensions",
            "    dimensions:",
            '        "x_dim"',
            "    variables:",
            '        "x_data"',
        ]
        assert text_lines == expected_lines

    def test_lis(self, scan_0d_mesh):
        # Check reporting of a location-index set
        scan = scan_0d_mesh
        self._convert_to_lis(scan)
        text = Checker(scan).structure_report()
        print(text)
        expects = [
            "Location Index Sets",
            r'"lis" \(lis_nodes\)',
            'mesh : "topology"',
            "location : node",
        ]
        self.expect_in_text(text, expects)
