from logging import LogRecord
from pathlib import Path
import re
from typing import AnyStr, Dict, List, Tuple, Union

from .logging import LOG
from .nc_dataset_scan import NcFileSummary, NcVariableSummary, scan_dataset
from .scan_utils import (
    property_as_single_name,
    property_namelist,
    vars_w_props,
)

__all__ = ["check_dataset"]

_VALID_CONNECTIVITY_ROLES = [
    "edge_node_connectivity",
    "face_node_connectivity",
    "face_edge_connectivity",
    "edge_face_connectivity",
    "face_face_connectivity",
    "boundary_node_connectivity",
]

_VALID_CF_ROLES = [
    "mesh_topology",
    "location_index_set",
] + _VALID_CONNECTIVITY_ROLES

_VALID_MESHCOORD_ATTRS = [
    f"{location}_coordinates" for location in ("face", "edge", "node")
]

# Valid cf varname regex : copied from iris.common.metadata code.
_VALID_NAME_REGEX = re.compile(r"""^[a-zA-Z][a-zA-Z0-9]*[\w.+\-@]*$""")


def check_mesh_attr_is_varlist(
    file_scan: NcFileSummary, meshvar: NcVariableSummary, attrname: str
):
    """
    Check that a mesh-var attribute, if it exists, is a valid varlist.

    Parameters
    ----------
    file_scan : :class:`NcFileSummary`
        file containing the mesh-var
    meshvar : class:`NcVariableSummary`
        mesh variable
    attrname : str
        name of the attribute of 'meshvar' to check

    """
    value = meshvar.attributes.get(attrname)
    if value is None:
        # Missing is ok.  But NB *not* an empty string (see below).
        success = True
    else:
        success = value.dtype.kind == "U"
        if not success:
            LOG.state(
                "R105",
                "Mesh",
                meshvar,
                f"attribute '{attrname}' does not have string type.",
            )
        if success:
            varnames = property_namelist(value)
            if not varnames:
                # Empty is *not* a valid content.
                # N.B. this includes non-string contents.
                LOG.state(
                    "R105",
                    "Mesh",
                    meshvar,
                    f"attribute '{attrname}' = \"{value}\", "
                    "which is not a valid list of netcdf variable names.",
                )
                success = False
        if success:
            for varname in varnames:
                if not varname:  # skip any extra blanks
                    continue
                if not _VALID_NAME_REGEX.match(varname):
                    LOG.state(
                        "R105",
                        "Mesh",
                        meshvar,
                        f"attribute '{attrname}' = \"{varname}\", "
                        "which is not a valid netcdf variable name.",
                    )
                    success = False
                elif varname not in file_scan.variables:
                    LOG.state(
                        "R106",
                        "Mesh",
                        meshvar,
                        f"attribute '{attrname}' refers to a variable "
                        f'"{varname}", but there is no such variable '
                        "in the dataset.",
                    )
                    success = False
    return success


def var_ref_problem(file_scan: NcFileSummary, attr_value: str):
    """
    Make a text description of any problems with a single-variable reference.

    Check that the input contains a single, valid name, referring to an
    existing variable.
    If no problem, returns an empty string.

    """
    succeed = True
    names = property_namelist(attr_value)
    if len(names) != 1:
        result = "is not a single variable name."
        succeed = False
    if succeed:
        boundsvar_name = property_as_single_name(attr_value)
        if not _VALID_NAME_REGEX.match(boundsvar_name):
            result = "is not a valid netcdf variable name."
            succeed = False
    if succeed:
        bounds_var = file_scan.variables.get(boundsvar_name)
        if bounds_var is None:
            result = "is not a variable in the dataset."
            succeed = False
    if succeed:
        result = ""
    return result


def check_coord_bounds(
    file_scan: NcFileSummary, coord: NcVariableSummary
) -> List[Tuple[str]]:
    """
    Validity-check the bounds of a coordinate (if any).

    Ok if bounds-attribute does not exist, but not if it is an empty string.
    Check: existence, n-dims, parent dimension, standard-name and units.

    Returns
    codes_and_messages : List[tuple(str, str)]
        a list of codes and messages, to be logged in the context of the parent
        coordinate variable.

    """
    result_codes_and_messages = []

    def log_bounds_statement(code, msg):
        msg = f'has bounds "{bounds_name}", which ' + msg
        result_codes_and_messages.append((code, msg))

    bounds_name = coord.attributes.get("bounds")
    has_bounds = bounds_name is not None
    if has_bounds:
        msg = var_ref_problem(file_scan, bounds_name)
        if msg != "":
            log_bounds_statement("R203", msg)
            has_bounds = False

    if has_bounds:
        # NB from the above check, we do have a bounds variable.
        bounds_var = file_scan.variables[bounds_name]
        bounds_dims = bounds_var.dimensions
        (coord_dim,) = coord.dimensions  # NB always has exactly 1
        if coord_dim not in bounds_dims:
            msg = (
                f"has dimensions {bounds_dims!r}, which does not include "
                f"the parent variable dimension, {coord_dim}."
            )
            log_bounds_statement("R203", msg)

        n_bounds_dims = bounds_dims
        if n_bounds_dims != 2:
            msg = (
                f"has dimensions {bounds_dims!r}, of which there should be "
                f"2, instead of {n_bounds_dims}."
            )
            log_bounds_statement("R203", msg)

        # Do the data-values check.  This is potentially costly.
        if coord.data is not None:
            # TODO: enable data-value checks by attaching lazy data arrays
            # to scan variables.
            assert bounds_var.data is not None
            raise ValueError("Not ready for data-value checks.")
            log_bounds_statement("A205", "???")

        def check_attr_mismatch(attr_name):
            coord_attr, bounds_attr = [
                var.attributes.get(attr_name) for var in (coord, bounds_var)
            ]
            if bounds_attr is not None and bounds_attr != coord_attr:
                msg = (
                    f'has {attr_name}="{bounds_attr}", which does not match '
                    f'the parent {attr_name} of "{coord_attr}".'
                )
                log_bounds_statement("R203", msg)

        check_attr_mismatch("standard_name")
        check_attr_mismatch("units")

    return result_codes_and_messages


def check_mesh_coordinates(
    file_scan: NcFileSummary,
    meshvar: NcVariableSummary,
    attr_name: str,
    mesh_dims: Dict[str, str],
    all_meshvars: List[NcVariableSummary],
):
    """Validity-check a coordinate attribute of a mesh-variable."""
    # Note: the content of the coords attribute was already checked

    # Elements which change as we scan the various coords.
    coord = None  # Changes as we scan them.
    common_message_prefix = ""
    coord_context_str = ""

    # Function to emit a statement message, adding context as to the specific
    # coord variable.
    def log_coord(code, msg):
        LOG.state(code, "Mesh coordinate", coord, common_message_prefix + msg)

    coord_names = property_namelist(meshvar.attributes.get(attr_name))
    for coord_name in coord_names:
        if coord_name not in file_scan.variables:
            # This problem will already have been detected + logged.
            continue
        coord = file_scan.variables[coord_name]
        common_message_prefix = f"within {meshvar.name}:{attr_name} "
        coord_ndims = len(coord.dimensions)
        if coord_ndims != 1:
            msg = (
                f"should have exactly one dimension, but has "
                f"{coord_ndims} dimensions : {coord.dimensions!r}."
            )
            log_coord("R201", msg)
        else:
            # Check the dimension is the correct one according to location.
            (coord_dim,) = coord.dimensions
            location = attr_name.split("_")[0]
            mesh_dim = mesh_dims[location]
            if coord_dim != mesh_dim:
                msg = coord_context_str + (
                    f'has dimension "{coord_dim}", but the parent mesh '
                    f'{location} dimension is "{mesh_dim}".'
                )
                log_coord("R202", msg)
            # Check coord bounds (if any)
            # N.B. this *also* assumes a single dim for the primary var
            codes_and_messages = check_coord_bounds(file_scan, coord)
            for code, msg in codes_and_messages:
                log_coord(code, msg)

        # Advisory notes..
        # A201 should have 1-and-only-1 parent mesh
        parent_meshes = []
        for meshname in sorted(all_meshvars):
            meshvar = all_meshvars[meshname]
            for attr in _VALID_MESHCOORD_ATTRS:
                attrval = meshvar.attributes.get(attr, "")
                names = str(attrval).split(" ")
                if coord_name in names:
                    parent_meshes.append(meshname)
        if len(parent_meshes) != 1:
            msg = (
                "has multiple mesh variables referring to it : "
                f"{sorted(parent_meshes)!r}."
            )
            log_coord("A201", msg)

        # A202 floating-point type
        dtype = coord.dtype
        if dtype.kind != "f":
            log_coord(
                "A202", f"has a dtype which is not floating-point : {dtype}."
            )

        # A203 standard-name : has+valid (can't handle fully ??)
        stdname = coord.attributes.get("standard_name")
        if not stdname:
            log_coord("A203", 'has no "standard_name" attribute.')

        # A204 units : has+valid (can't handle fully ??)
        stdname = coord.attributes.get("units")
        if not stdname:
            log_coord("A204", 'has no "units" attribute.')

        # A205 bounds data values match derived ones
        # - did this already above, within "check_coord_bounds"


def check_mesh_connectivity(file_scan, meshvar, attr_name, mesh_dims):
    """Validity-check a connectivity attribute of a mesh-variable."""
    value = meshvar.attributes.get(attr_name)
    more_todo = value is not None
    if more_todo:
        pass


def check_meshvar(
    meshvar: NcVariableSummary,
    file_scan: NcFileSummary,
    all_meshvars: List[NcVariableSummary],
    name_of_a_referring_var: Union[str, None] = None,
) -> Dict[str, str]:
    """
    Run checks on a mesh variable.

    Parameters
    ----------
    meshvar : :class:`NcVariableSummary`
        meshvar to check
    file_scan : :class:`NcFileSummary`
        a scan of the entire file that this meshvar exists in
    name_of_a_referring_var : str or None, default = None
        a variable which names *this* var in its 'mesh' property,
        if there are any.

    Returns
    -------
    mesh_dims : Dict[str, str]
        A dictionary mapping each mesh location to the name of the relevant
        location dimension in this mesh, if any.
        Unused locations are present with value None.

    """
    # First check for bad 'cf_role' :
    # if wrong, meshvar can only have been identified by reference.
    cfrole_prop = meshvar.attributes.get("cf_role", None)
    if cfrole_prop != "mesh_topology":
        assert name_of_a_referring_var is not None
        # This variable does not have the expected 'cf_role', so if we are
        # checking it, it must be referred to as 'mesh' by some variable.
        # Either there is no 'cf_role', or it is "wrong".
        msg = (
            f"appears to be a mesh, "
            f'since it is the value of "{name_of_a_referring_var}::mesh". '
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
            # "R102.a"
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
            # Handle this subsequently as if it was missing
            topology_dimension = None

    # Work out what topology-dimension is implied by the available mesh
    # properties, which we will use *instead* of the declared one in
    # subsequent tests (and check the declared one against it).
    highest_connectivity = None
    appropriate_dim = 0
    if "face_node_connectivity" in meshvar.attributes:
        appropriate_dim = 2
        highest_connectivity = "face_node_connectivity"
    elif "edge_node_connectivity" in meshvar.attributes:
        appropriate_dim = 1
        highest_connectivity = "edge_node_connectivity"

    if topology_dimension is not None:
        # Emit an error if the attributes present don't match the stated
        # topology-dimension.  If *no* topology-dimension, skip this : we
        # already flagged that it was missing, above.
        if topology_dimension != appropriate_dim:
            if topology_dimension == 0:
                if appropriate_dim == 1:
                    errcode = "R110"  # unexpected edge-node
                else:
                    assert appropriate_dim == 2
                    errcode = "R113"  # unexpected face-node
            elif topology_dimension == 1:
                if appropriate_dim == 0:
                    errcode = "R111"  # missing edge-node
                else:
                    assert appropriate_dim == 2
                    errcode = "R113"  # unexpected face-node
            else:
                assert topology_dimension == 2
                errcode = "R113"  # missing face-node
            #
            # TODO: remove R112 -- "may" is not testable !!
            #

            if topology_dimension < appropriate_dim:
                # something is extra
                msg = (
                    f'has "topology_dimension={topology_dimension}", '
                    f"but the presence of a '{highest_connectivity}' "
                    f"attribute implies it should be {appropriate_dim}."
                )
            else:
                # something is missing
                toplogy_required_attribute = {
                    0: "face_node",
                    1: "edge_node_connectivity",
                    2: "face_node_connectivity",
                }[int(topology_dimension)]
                msg = (
                    f'has "topology_dimension={topology_dimension}", '
                    f"but it has no '{toplogy_required_attribute}' "
                    f"attribute."
                )
            LOG.state(errcode, "Mesh", meshvar, msg)

    # We will use the 'calculated' one to scope any remaining checks.
    # TODO: remove this, if it continues to be unused.
    # topology_dimension = appropriate_dim

    # Check that coordinate and connectivity attributes are valid "varlists"
    varlist_names = _VALID_MESHCOORD_ATTRS + _VALID_CONNECTIVITY_ROLES
    for attr in varlist_names:
        ok = check_mesh_attr_is_varlist(file_scan, meshvar, attr)
        if not ok:
            value = meshvar.attributes.get(attr, "")
            is_conn = attr in _VALID_CONNECTIVITY_ROLES
            LOG.state(
                "R108" if is_conn else "R107",
                "Mesh",
                meshvar,
                (
                    f'attribute "{attr}" = "{value}", which is not a list of '
                    "variables in the dataset."
                ),
            )

    # Work out the actual mesh dimensions.
    mesh_dims = {name: None for name in ("face", "edge", "node", "boundary")}

    if "node_coordinates" not in meshvar.attributes:
        LOG.state(
            "R109",
            "Mesh",
            meshvar,
            "does not have a 'node_coordinates' attribute.",
        )
    else:
        # Note: if a 'node_coordinates' attribute exists, then we already
        # checked that it is a valid varlist.
        # So don't re-raise any problems here, just press on.
        coord_names = property_namelist(meshvar.attributes["node_coordinates"])
        if coord_names:
            coord_var = file_scan.variables.get(coord_names[0])
            if coord_var:
                # Answer is the first dimension, if any.
                if len(coord_var.dimensions) > 0:
                    mesh_dims["node"] = coord_var.dimensions[0]

    def deduce_face_or_edge_dim(location):
        # Identify the dim, and check the consistency of relevant attributes.
        # If found, set it in 'mesh_dims'
        dimattr_name = f"{location}_dimension"
        connattr_name = f"{location}_node_connectivity"
        dimension_name = property_as_single_name(
            meshvar.attributes.get(dimattr_name)
        )
        if location in ("boundary", "node"):
            # No 'boundary_dimension' attribute is supported.
            if dimension_name:
                dimension_name = None
                # "A105 ?"
                LOG.state(
                    "?",
                    "Mesh",
                    meshvar,
                    (
                        f'has an attribute "{dimattr_name}", which is not '
                        'a valid UGRID term, and may be a mistake (ADVISORY)."'
                    ),
                )
                # TODO: add ADVISE code for this ?

        if dimension_name:
            # There is an explicit 'xxx_dimension' property.
            if connattr_name not in meshvar.attributes:
                # "A106 ?"
                LOG.state(
                    "?",
                    "Mesh",
                    meshvar,
                    f'has an attribute "{dimattr_name}", which is not valid '
                    f'since there is no "{connattr_name}".',
                )
            elif dimension_name in file_scan.dimensions:
                mesh_dims[location] = dimension_name
            else:
                errcode = {"edge": "R115", "face": "R117"}[location]
                LOG.state(
                    errcode,
                    "Mesh",
                    meshvar,
                    f'has {dimattr_name}="{dimension_name}", which is not '
                    "a dimension in the dataset.",
                )
        elif connattr_name in meshvar.attributes:
            # No "xxx_dimension" attribute, but we *do* have
            # "xxx_node_connectivity", so the mesh does _have_ this location.
            connvar_name = property_as_single_name(
                meshvar.attributes[connattr_name]
            )
            conn_var = file_scan.variables.get(connvar_name)
            if conn_var:
                # Answer is the first dimension, if any.
                if len(conn_var.dimensions) > 0:
                    mesh_dims[location] = conn_var.dimensions[0]

    deduce_face_or_edge_dim("node")
    deduce_face_or_edge_dim("boundary")
    deduce_face_or_edge_dim("edge")
    deduce_face_or_edge_dim("face")

    # Check that, if any connectivities have non-standard dim order, then a
    # dimension attribute exists.
    def var_has_nonfirst_dim(varname, dimname):
        conn_var = file_scan.variables.get(varname)
        result = conn_var is not None
        if result:
            result = dimname in conn_var.dimensions
        if result:
            result = conn_var.dimensions[0] != dimname
        return result

    location_altordered_conns = {}
    for attr in _VALID_CONNECTIVITY_ROLES:
        maindim_location = attr.split("_")[0]
        assert maindim_location != "node"  # no such connectivities
        maindim_name = mesh_dims[maindim_location]
        for conn_name in property_namelist(meshvar.attributes.get(attr)):
            if var_has_nonfirst_dim(conn_name, maindim_name):
                # We found a connectivity with a nonstandard dim order
                dim_attr = f"{maindim_location}_dimension"
                if dim_attr not in meshvar.attributes:
                    # There is no corresponding 'xxx_dimension', so warn.
                    conns = location_altordered_conns.get(
                        maindim_location, set()
                    )
                    conns.add(conn_name)
                    location_altordered_conns[maindim_location] = conns

    for location, conns in location_altordered_conns.items():
        # We found connectivities with a nonstandard dim order for this dim.
        assert location in ("face", "edge")
        errcode = {"edge": "R116", "face": "R118"}[location]
        conn_names = [f'"{name}"' for name in conns]
        conn_names_str = ", ".join(conn_names)
        msg = (
            f'has no "{dim_attr}" attribute, but there are '
            f"{location} connectivities "
            f"with non-standard dimension order : {conn_names_str}."
        )
        LOG.state(errcode, "Mesh", meshvar, msg)

    # Check that all existing coordinates are valid.
    for coords_name in _VALID_MESHCOORD_ATTRS:
        location = coords_name.split("_")[0]
        # Only check coords of locations present in the mesh.
        # This avoids complaints about coords dis-connected by problems with
        # the topology identification.
        if mesh_dims[location]:
            check_mesh_coordinates(
                file_scan, meshvar, coords_name, mesh_dims, all_meshvars
            )

    # Check that all existing connectivities are valid.
    for attr in _VALID_CONNECTIVITY_ROLES:
        check_mesh_connectivity(file_scan, meshvar, attr, mesh_dims)

    # deal with the optional elements (connectivities)
    def check_requires(errcode, attrname, location_1, location_2=None):
        exist = attrname in meshvar.attributes
        if exist:
            elems = [location_1]
            if location_2:
                elems.append(location_2)
            required_elements = [f"{name}_node_connectivity" for name in elems]
            missing_elements = [
                f'"{name}"'
                for name in required_elements
                if name not in meshvar.attributes
            ]
            if missing_elements:
                err_msg = (
                    f'has a "{attrname}" attribute, which is not valid '
                    f"since there is no "
                )
                err_msg += "or ".join(missing_elements)
                err_msg += " attribute present."
                LOG.state(errcode, "Mesh", meshvar, err_msg)

    check_requires("R114", "boundary_node_connectivity", "face")
    check_requires("R119", "face_face_connectivity", "face")
    check_requires("R120", "face_edge_connectivity", "face", "edge")
    check_requires("R121", "edge_face_connectivity", "face", "edge")

    # Advisory checks.
    if meshvar.dimensions:
        LOG.state("A101", "Mesh", meshvar, "has dimensions.")
    if "standard_name" in meshvar.attributes:
        LOG.state("A102", "Mesh", meshvar, 'has a "standard_name" attribute.')
    if "units" in meshvar.attributes:
        LOG.state("A103", "Mesh", meshvar, 'has a "units" attribute.')
    # NOTE: "A104" relates to multiple meshvars, so is handled in caller.

    return mesh_dims


def check_dataset_inner(file_scan: NcFileSummary):
    """
    Run UGRID conformance checks on a representation of a file.

    This low-level routine operates on an abstract "file-scan" representation
    rather than a real file.
    All checking messages are recorded with :meth:`LOG.state`.

    Parameters
    ----------
    file_scan : :class:`NcFileSummary`
        representation of a netcdf file to check

    """
    #
    # Phase#1 : identify mesh data variables
    #
    all_vars = file_scan.variables
    meshdata_vars = vars_w_props(all_vars, mesh="*")

    # Check that any var 'mesh' property names a valid mesh variable.
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
            if meshvar_name not in all_vars:
                LOG.state(
                    "R502",
                    "Mesh data",
                    mrv_var,
                    (
                        f"has attribute \"mesh='{meshvar_name}'\", but there "
                        f'is no "{meshvar_name}" variable in the dataset.'
                    ),
                )
            else:
                # Include this one in those we check as mesh-vars.
                all_meshvars[meshvar_name] = all_vars[meshvar_name]
                # Record name of referring var.
                # N.B. potentially this can overwrite a previous referrer,
                # but "any one of several" will be OK for our purpose.
                meshvar_referrers[meshvar_name] = mrv_name

    #
    # Phase#2 : check all mesh variables
    #
    # Add all vars with 'cf_role="mesh_topology"' to the all-meshvars dict
    meshvars_by_cf_role = vars_w_props(all_vars, cf_role="mesh_topology")
    all_meshvars.update(meshvars_by_cf_role)

    # Check all putative mesh vars.

    # Build a map of mesh dimensions,
    # all_meshes_dims: {meshname: {location: dimname}}
    all_meshes_dims = {}

    # Also keep an account of coordinates + connectivities which are
    # cross-referred between different meshes.
    # Just so that we can avoid warning about these more than once.
    # cross_mapped_coords

    for meshname, meshvar in all_meshvars.items():
        dims_dict = check_meshvar(
            meshvar,
            file_scan=file_scan,
            all_meshvars=all_meshvars,
            name_of_a_referring_var=meshvar_referrers.get(meshname, None),
        )
        # (NB names should be var_names of the vars, so all different)
        assert meshname not in all_meshes_dims
        all_meshes_dims[meshname] = dims_dict

    # Check for dimensions shared between meshes, which is an advisory warning.

    # Convert all_meshes_dims: {meshname: {location: dimname}}
    # .. to dim_meshes: {dimname: [meshnames]}
    dim_meshes = {}
    for mesh, location_dims in all_meshes_dims.items():
        for location, dim in location_dims.items():
            # Fetch list
            meshnames = dim_meshes.get(dim, set())
            if dim:
                # TODO: what if a dim is used by 2 different locations of
                #  of the same mesh ?
                meshnames.add(mesh)
            # Write list back
            dim_meshes[dim] = meshnames

    # Check for any dims which are used more than once.
    for dim, meshnames in dim_meshes.items():
        if len(meshnames) > 1:
            # TODO: what if a dim is used by 2 different locations of
            #  of the same mesh ?
            #  We would get a repeated meshname here...
            meshnames = sorted(meshnames)
            other_meshes, last_mesh = meshnames[:-1], meshnames[-1]
            if len(other_meshes) == 1:
                other_mesh = other_meshes[0]
                msg = (
                    f'Dimension "{dim}" is mapped by both '
                    f'mesh "{other_mesh}" and mesh "{last_mesh}".'
                )
            else:
                msg = f'Dimension "{dim}" is mapped by multiple meshes : '
                msg += ", ".join(f'"{mesh}"' for mesh in other_meshes)
                msg += f' and "{last_mesh}".'
            LOG.state("A104", None, None, msg)


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
    file: Union[NcFileSummary, AnyStr, Path],
    print_results: bool = True,
    print_summary: bool = True,
) -> List[LogRecord]:
    """
    Run UGRID conformance checks on a file.

    Log statements regarding any problems found to the :data:`_LOG` logger.
    Return the accumulated log records of problems found.
    Optionally print logger messages and/or a summary of results.

    Parameters
    ----------
    file : filepath string, Path or :class:`NcFileSummary`
        path to, or representation of a netcdf input file
    print_results : bool, default=True
        whether to print warnings as they are logged
    print_summary : bool, default=True
        whether to print a results summary at the end

    Returns
    -------
    checker_warnings : list of :class:LogRecord
            A list of logged message records, one for each problem identified.
    """
    LOG.reset()
    # print_results, print_summary = True, True
    LOG.enable_reports_printout(print_results)

    if isinstance(file, str):
        file_path = Path(file)
    elif isinstance(file, Path):
        file_path = file
    if isinstance(file, NcFileSummary):
        file_scan = file
    else:
        file_scan = scan_dataset(file_path)

    check_dataset_inner(file_scan)

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
