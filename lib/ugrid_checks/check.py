from pathlib import Path
from typing import Union, AnyStr

from .nc_dataset_scan import NcFileSummary, scan_dataset
from .scan_utils import vars_w_props, property_as_single_name

#
# Crude message logging, for now
#
_N_WARNINGS = 0
_N_FAILURES = 0

import logging
class UgridLogHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def emit(self, record):
        self.logs.append(record)

    def reset(self):
        self.logs = []


_LOG = logging.Logger('ugrid_conformance')
_HANDLER = UgridLogHandler(level=logging.INFO)
_LOG.addHandler(_HANDLER)
_ENABLE_PRINT = True

def reset_reports():
    """Reset the report log."""
    global _N_WARNINGS, _N_FAILURES
    _N_WARNINGS = 0
    _N_FAILURES = 0
    _HANDLER.reset()


def enable_printout(print_statements=True):
    global _ENABLE_PRINT
    _ENABLE_PRINT = print_statements


def report_statement_logrecords():
    """Return the report log as a list of :class:`~logging.LogRecords`."""
    return _HANDLER.logs


def report(msg, level=logging.INFO, *args):
    if _ENABLE_PRINT:
        print(msg)
    _LOG.log(level, msg, *args)


def printonly(msg, *args):
    report(msg, logging.DEBUG, *args)


def _statement(vartype, var, msg):
    if vartype:
        result = vartype + f' variable {var.name}'
    else:
        result = f'Variable {var.name}'
    result += f' {msg}'
    return result


def warn(vartype, var, msg, a_num=0):
    global _N_WARNINGS
    _N_WARNINGS += 1
    msg = f'... WARN A{a_num:03d} : ' + _statement(vartype, var, msg)
    report(msg, logging.WARN, a_num)


def fail(vartype, var, msg, r_num=0):
    global _N_FAILURES
    _N_FAILURES += 1
    msg = f'*** FAIL R{r_num:03d} : ' + _statement(vartype, var, msg)
    report(msg, logging.ERROR, r_num)


def state(statement: str, vartype, var, msg):
    statement_type = statement[0]
    try:
        statement_num = int(statement[1:])
    except ValueError:
        statement_num = 0
    if statement_type == 'R':
        fail(vartype, var, msg, statement_num)
    elif statement_type == 'A':
        warn(vartype, var, msg, statement_num)
    else:
        report(_statement(vartype, var, msg))


_VALID_CONNECTIVITY_ROLES = [
    'edge_node_connectivity',
    'face_node_connectivity'
    'face_edge_connectivity',
    'edge_face_connectivity',
    'face_face_connectivity',
    'boundary_node_connectivity',
]

_VALID_CF_ROLES = [
    'mesh_topology'
    'location_index_set',
] + _VALID_CONNECTIVITY_ROLES


def check_meshvar(meshname, meshvar, meshvars_by_cf_role, meshvar_referrers):
    # First check for bad 'cf_role' : if wrong, meshvar can only have been identified by reference.
    if meshname in meshvars_by_cf_role:
        assert meshvar.attributes['cf_role'] == 'mesh_topology'  # This is already guaranteed
    else:
        # This variable is referred as 'mesh' by some variable, but does not have the expected 'cf_role'.
        # Either there is no 'cf_role', or it is "wrong".
        referring_var = meshvar_referrers[meshname]
        cfrole_prop = meshvar.attributes.get('cf_role', None)
        assert cfrole_prop != 'mesh_topology'
        msg = (f'appears to be a mesh, '
               f'since it is the value of "{referring_var}::mesh". '
               'But it has ')
        if cfrole_prop is None:
            msg += 'no "cf_role" property,'
        else:
            msg += f'a "cf_role" of "{cfrole_prop}",'
        msg += " which should be 'mesh_topology'."
        state('R102', '', meshvar, msg)
        # Also, if the 'cf_role' was something else, check it is a valid option + emit an additional message if needed.
        if cfrole_prop is not None and cfrole_prop not in _VALID_CF_ROLES:
            msg = f'has a "cf_role" of "{cfrole_prop}", which is not a valid UGRID cf_role.'
            state('?', 'Mesh', meshvar, msg)

    # Check all other attributes of mesh vars.
    topology_dimension = meshvar.attributes.get('topology_dimension')
    if topology_dimension is None:
        state('R103', 'Mesh', meshvar, 'has no "topology_dimension".')
    else:
        # Check the topology dimension.
        # In principle, this controls which other connectivity properties may appear.
        # In practice, it is better to parse those independently, and then cross-check.
        # TODO
        if topology_dimension not in (0, 1, 2):
            state('R104', 'Mesh', meshvar, f'has "topology_dimension={topology_dimension}", which is not 0, 1 or 2.')


def check_dataset_inner(scan):
    #
    # Phase#1 : identify mesh data variables
    #
    vars = scan.variables
    meshdata_vars = vars_w_props(vars, mesh='*')

    # Check that all vars with a 'mesh' properties name valid variables.
    all_meshvars = {}
    meshvar_referrers = {}
    for mrv_name, mrv_var in meshdata_vars.items():
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
                meshvar_referrers[meshvar_name] = mrv_name  # A 'backwards' lookup.

    #
    # Phase#2 : check all mesh variables
    #
    vars = scan.variables
    # Add all vars with 'cf_role="mesh_topology"' to the all-meshvars dict
    meshvars_by_cf_role = vars_w_props(vars, cf_role='mesh_topology')
    all_meshvars.update(meshvars_by_cf_role)  # Add these into the "all vars to check"

    # Check all putative mesh vars.
    for meshname, meshvar in all_meshvars.items():
        check_meshvar(meshname, meshvar, meshvars_by_cf_role, meshvar_referrers)


def printout_reports():
    printonly('')
    printonly(f"UGRID conformance checks complete.")
    if _N_FAILURES + _N_WARNINGS == 0:
        printonly('No problems found.')
    else:
        printonly(f"Total of {_N_WARNINGS + _N_FAILURES} problems found: \n"
           f"  {_N_FAILURES} Rxxx requirement failures\n"
           f"  {_N_WARNINGS} Axxx advisory recommendation warnings")
    printonly('Done.')


def check_dataset(scan: Union[NcFileSummary, AnyStr, Path], print_summary=True, print_results=True):
    """
    Run UGRID conformance checks on file content.

    Logs a series of statements regarding problems found to the :data:`_LOG` logger.
    The built-in logger prints the statements, and records them

    Args:
    * scan:
        Item to scan : an :class:`~ugrid_checks.nc_dataset_scan.NcFileSummary` or a string or filepath object.

    """
    reset_reports()
    enable_printout(print_results)

    if isinstance(scan, str):
        scan = Path(scan)
    if isinstance(scan, Path):
        scan = scan_dataset(scan)

    check_dataset_inner(scan)

    # fail('oops')
    # from .nc_dataset_scan import NcVariableSummary
    # dummyvar = NcVariableSummary('dummyvar', (), (), None, {}, None)
    # warn('Dummy', dummyvar, 'test warning message')
    # report('test non-warning message')
    if print_summary:
        printout_reports()
        print('')
        print('Logged reports:')
        for log_report in _HANDLER.logs:
            print(f'  {log_report}')
