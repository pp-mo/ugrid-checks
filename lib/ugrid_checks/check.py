from pathlib import Path
from typing import Union, AnyStr
import warnings


from .nc_dataset_scan import NcFileSummary, scan_dataset
from .scan_utils import vars_w_props, property_as_single_name


# class UgridConformanceFailWarning(warnings.WarningMessage):
#     """
#     A warning whose message details how a dataset fails to meet a UGRID conformance requirement (R-code).
#
#     Extra properties encode certain standard details:
#     * format : string
#         the format string for the warning
#     * code : int
#         the 'R' code number matching the warning.
#
#     """
#     pass
#
#
# class UgridConformanceRecommendationWarning(UgridConformanceFailWarning):
#     """A warning whose message details how a dataset doesn't meet a UGRID conformance advisory recommendation (A-code)."""
#     pass

_N_WARNINGS = 0
_N_FAILURES = 0

def reset_reports():
    global _N_WARNINGS, _N_FAILURES
    _N_WARNINGS = 0
    _N_FAILURES = 0

def report(msg):
    print(msg)

def warn(msg, a_num=0):
    global _N_WARNINGS
    _N_WARNINGS += 1
    report(f'... WARN A{a_num:03d} : {msg}')

def fail(msg, r_num=0):
    global _N_FAILURES
    _N_FAILURES += 1
    report(f'*** FAIL R{r_num:03d} : {msg}')


def _check_meshvar(meshname, meshvar, meshvars_by_cf_role, referred_meshvars_by_referrer):
    # First check for bad 'cf_role' : if wrong, meshvar can only have been identified by reference.
    if meshname not in meshvars_by_cf_role:
        referring_var = referred_meshvars_by_referrer[meshname]
        cfrole_prop = meshvar.attributes.get('cf_role', None)
        assert cfrole_prop != 'mesh_topology'
        msg = (f'variable "{meshname}" appears to be a mesh, '
               f'since it is the value of "{referring_var}::mesh". '
               'But it has ')
        if cfrole_prop is None:
            msg += 'no "cf_role" property,'
        else:
            msg += f'a "cf_role" property of "{cfrole_prop}",'
        msg += " which should be 'mesh_topology'."
        fail(msg)

    # Check all other attributes of mesh vars.
    pass


def check_data(scan: Union[NcFileSummary, AnyStr, Path]):
    """
    Run UGRID conformance checks on file content.

    Raises a series of

    Args:
    * scan:
        Item to scan : an :class:`~ugrid_checks.nc_dataset_scan.NcFileSummary` or a string or filepath object.

    """
    if isinstance(scan, str):
        scan = Path(scan)
    if isinstance(scan, Path):
        scan = scan_dataset(scan)

    #
    # Phase#1 : identify mesh variables
    #
    vars = scan.variables
    meshvars_by_cf_role = vars_w_props(vars, cf_role='mesh_topology')
    all_meshvars = meshvars_by_cf_role.copy()

    # Check that all vars with a 'mesh' properties name valid variables.
    mesh_referring_vars = vars_w_props(vars, mesh='*')
    referred_meshvars_by_referrer = {}
    for mrv_name, mrv_var in mesh_referring_vars.items():
        meshprop = mrv_var.attributes['mesh']
        meshvar_name = property_as_single_name(meshprop)
        if not meshvar_name:
            warn(f'variable {mrv_name} has attribute "mesh=\'{meshprop}\'", which is not a valid variable name.')
        elif meshvar_name not in all_meshvars:
            if meshvar_name not in vars:
                warn(f'variable {mrv_name} has attribute "mesh=\'{meshvar_name}\'", '
                     f'but there is no "{meshvar_name}" variable in the dataset.')
            else:
                # Include this one in those we check as mesh-vars.
                all_meshvars[meshvar_name] = vars[meshvar_name]
                referred_meshvars_by_referrer[meshvar_name] = mrv_name  # A 'backwards' lookup.

    # Check all putative mesh vars.
    for meshname, meshvar in all_meshvars.items():
        _check_meshvar(meshname, meshvar, meshvars_by_cf_role, referred_meshvars_by_referrer)

    # All ok.

    # warn('odd')
    # fail('oops')
    report(f"UGRID conformance checks complete.")
    if _N_FAILURES + _N_WARNINGS == 0:
        report('No problems found.')
    else:
        report(f"Total of {_N_WARNINGS + _N_FAILURES} problems found: \n"
           f"  {_N_FAILURES} Rxxx requirement failures\n"
           f"  {_N_WARNINGS} Axxx advisory recommendation warnings")
    report('Done.')
