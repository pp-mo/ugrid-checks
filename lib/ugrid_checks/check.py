from pathlib import Path
from typing import AnyStr, Union

from .logging import LOG
from .nc_dataset_scan import NcFileSummary, scan_dataset
from .scan_utils import property_as_single_name, vars_w_props

__all__ = ["check_dataset"]


_VALID_CONNECTIVITY_ROLES = [
    "edge_node_connectivity",
    "face_node_connectivity" "face_edge_connectivity",
    "edge_face_connectivity",
    "face_face_connectivity",
    "boundary_node_connectivity",
]

_VALID_CF_ROLES = [
    "mesh_topology" "location_index_set",
] + _VALID_CONNECTIVITY_ROLES


def check_meshvar(meshname, meshvar, meshvars_by_cf_role, meshvar_referrers):
    # First check for bad 'cf_role' :
    # if wrong, meshvar can only have been identified by reference.
    if meshname in meshvars_by_cf_role:
        assert (
            meshvar.attributes["cf_role"] == "mesh_topology"
        )  # This is already guaranteed
    else:
        # This variable is referred as 'mesh' by some variable,
        # but does not have the expected 'cf_role'.
        # Either there is no 'cf_role', or it is "wrong".
        referring_var = meshvar_referrers[meshname]
        cfrole_prop = meshvar.attributes.get("cf_role", None)
        assert cfrole_prop != "mesh_topology"
        msg = (
            f"appears to be a mesh, "
            f'since it is the value of "{referring_var}::mesh". '
            "But it has "
        )
        if cfrole_prop is None:
            msg += 'no "cf_role" property,'
            errcode = "R101"
        else:
            msg += f'a "cf_role" of "{cfrole_prop}",'
            errcode = "R102"
        msg += " which should be 'mesh_topology'."
        LOG.state(errcode, "", meshvar, msg)
        # Also, if the 'cf_role' was something else,
        # check it is a valid option + emit an additional message if needed.
        if cfrole_prop is not None and cfrole_prop not in _VALID_CF_ROLES:
            msg = (
                f'has a "cf_role" of "{cfrole_prop}", '
                "which is not a valid UGRID cf_role."
            )
            LOG.state("?", "Mesh", meshvar, msg)

    # Check all other attributes of mesh vars.
    topology_dimension = meshvar.attributes.get("topology_dimension")
    if topology_dimension is None:
        LOG.state("R103", "Mesh", meshvar, 'has no "topology_dimension".')
    else:
        # Check the topology dimension.
        # In principle, this controls which other connectivity properties may
        # appear : In practice, it is better to parse those independently,
        # and then cross-check.
        # TODO
        if topology_dimension not in (0, 1, 2):
            LOG.state(
                "R104",
                "Mesh",
                meshvar,
                (
                    f'has "topology_dimension={topology_dimension}", '
                    "which is not 0, 1 or 2."
                ),
            )


def check_dataset_inner(scan):
    #
    # Phase#1 : identify mesh data variables
    #
    vars = scan.variables
    meshdata_vars = vars_w_props(vars, mesh="*")

    # Check that all vars with a 'mesh' properties name valid variables.
    all_meshvars = {}
    meshvar_referrers = {}
    for mrv_name, mrv_var in meshdata_vars.items():
        meshprop = mrv_var.attributes["mesh"]
        meshvar_name = property_as_single_name(meshprop)
        if not meshvar_name:
            LOG.state(
                "R502",
                "Mesh data",
                mrv_var,
                (
                    f"variable {mrv_name} has attribute "
                    f"\"mesh='{meshprop}'\", which is not a "
                    "valid variable name."
                ),
            )
        elif meshvar_name not in all_meshvars:
            if meshvar_name not in vars:
                LOG.state(
                    "R502",
                    "Mesh data",
                    mrv_var,
                    (
                        f"variable {mrv_name} has attribute "
                        "\"mesh='{meshvar_name}'\", but there is no "
                        f'"{meshvar_name}" variable in the dataset.'
                    ),
                )
            else:
                # Include this one in those we check as mesh-vars.
                all_meshvars[meshvar_name] = vars[meshvar_name]
                meshvar_referrers[meshvar_name] = mrv_name

    #
    # Phase#2 : check all mesh variables
    #
    vars = scan.variables
    # Add all vars with 'cf_role="mesh_topology"' to the all-meshvars dict
    meshvars_by_cf_role = vars_w_props(vars, cf_role="mesh_topology")
    all_meshvars.update(meshvars_by_cf_role)

    # Check all putative mesh vars.
    for meshname, meshvar in all_meshvars.items():
        check_meshvar(
            meshname, meshvar, meshvars_by_cf_role, meshvar_referrers
        )


def printout_reports():
    LOG.printonly("")
    LOG.printonly("UGRID conformance checks complete.")
    if LOG.N_FAILURES + LOG.N_WARNINGS == 0:
        LOG.printonly("No problems found.")
    else:
        LOG.printonly(
            f"Total of {LOG.N_WARNINGS + LOG.N_FAILURES} problems found: \n"
            f"  {LOG.N_FAILURES} Rxxx requirement failures\n"
            f"  {LOG.N_WARNINGS} Axxx advisory recommendation warnings"
        )
    LOG.printonly("Done.")


def check_dataset(
    scan: Union[NcFileSummary, AnyStr, Path],
    print_summary=True,
    print_results=True,
):
    """
    Run UGRID conformance checks on file content.

    Log statements regarding problems found to the :data:`_LOG` logger.
    The built-in logger prints the statements, and records them

    Args:
    * scan:
        Item to scan : an :class:`~ugrid_checks.nc_dataset_scan.NcFileSummary`,
        or a string or filepath object.

    Return:
    * logs
        A list of LogRecords representing checker statements.

    """
    LOG.reset()
    # print_results, print_summary = True, True
    LOG.enable_reports_printout(print_results)

    if isinstance(scan, str):
        scan = Path(scan)
    if isinstance(scan, Path):
        scan = scan_dataset(scan)

    check_dataset_inner(scan)

    # from .nc_dataset_scan import NcVariableSummary
    # dummyvar = NcVariableSummary('dummyvar', (), (), None, {}, None)
    # LOG.state('?', 'Dummy', dummyvar, 'test warning message')
    # LOG.state('R123', 'Dummy', dummyvar, 'failure')
    # LOG.state('A123', 'Dummy', dummyvar, 'recommendation')
    # report('test non-warning message')

    if print_summary:
        printout_reports()
        print("")
        print("Logged reports:")
        for log_report in LOG.report_statement_logrecords():
            print(f"  {log_report}")

    return LOG.report_statement_logrecords()
