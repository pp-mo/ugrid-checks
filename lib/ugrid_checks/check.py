import logging
from logging import LogRecord
from pathlib import Path
import re
from typing import AnyStr, Dict, List, Tuple, Union

import numpy as np

from .nc_dataset_scan import NcFileSummary, NcVariableSummary, scan_dataset
from .scan_utils import (
    property_as_single_name,
    property_namelist,
    vars_w_props,
)
from .ugrid_logger import LOG

__all__ = ["check_dataset"]

_VALID_CONNECTIVITY_ROLES = [
    "edge_node_connectivity",
    "face_node_connectivity",
    "face_edge_connectivity",
    "edge_face_connectivity",
    "face_face_connectivity",
    "boundary_node_connectivity",
]

_VALID_UGRID_CF_ROLES = [
    "mesh_topology",
    "location_index_set",
] + _VALID_CONNECTIVITY_ROLES

_VALID_MESHCOORD_ATTRS = [
    f"{location}_coordinates" for location in ("face", "edge", "node")
]

_VALID_CF_CF_ROLES = [
    "timeseries_id",
    "profile_id",
    "trajectory_id",
]

# Valid cf varname regex : copied from iris.common.metadata code.
_VALID_NAME_REGEX = re.compile(r"""^[a-zA-Z][a-zA-Z0-9]*[\w.+\-@]*$""")


class Checker:
    def __init__(self, file_scan: NcFileSummary, do_data_checks: bool = False):
        self.file_scan = file_scan
        self.do_data_checks = do_data_checks
        self._all_vars = file_scan.variables
        # Note: the following are filled in by 'dataset_identify_containers'
        self._meshdata_vars: Dict[str, NcVariableSummary] = {}
        self._mesh_vars: Dict[str, NcVariableSummary] = {}
        self._lis_vars: Dict[str, NcVariableSummary] = {}
        self._mesh_referrers: Dict[str, str] = {}
        self._lis_referrers: Dict[str, str] = {}
        # Note: these are filled by 'dataset_check_containers_and_map_dims'
        self._all_mesh_dims: Dict[str, Dict[str, Union[None, str]]] = {}
        self._allowed_cfrole_varnames: List[str]

    def check_mesh_attr_is_varlist(
        self, meshvar: NcVariableSummary, attrname: str
    ):
        """
        Check that a mesh-var attribute, if it exists, is a valid varlist.

        Parameters
        ----------
        meshvar : class:`NcVariableSummary`
            mesh variable
        attrname : str
            name of the attribute of 'meshvar' to check

        Returns
        -------
        ok : bool
            True iff no problems were found

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
                    meshvar.name,
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
                        meshvar.name,
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
                            meshvar.name,
                            f"attribute '{attrname}' = \"{varname}\", "
                            "which is not a valid netcdf variable name.",
                        )
                        success = False
                    elif varname not in self._all_vars:
                        LOG.state(
                            "R106",
                            "Mesh",
                            meshvar.name,
                            f"attribute '{attrname}' refers to a variable "
                            f'"{varname}", but there is no such variable '
                            "in the dataset.",
                        )
                        success = False
        return success

    def var_ref_problem(self, attr_value: np.ndarray):
        """
        Make a text description of any problems of a single-variable reference.

        Check that the input contains a single, valid name, referring to an
        existing variable.
        If no problem, returns an empty string.

        """
        succeed = True
        if attr_value.dtype.kind != "U":
            result = "is not a string value"
            succeed = False
        if succeed:
            names = property_namelist(attr_value)
            if len(names) != 1:
                result = "is not a single variable name"
                succeed = False
        if succeed:
            boundsvar_name = property_as_single_name(attr_value)
            if not _VALID_NAME_REGEX.match(boundsvar_name):
                result = "is not a valid netcdf variable name"
                succeed = False
        if succeed:
            bounds_var = self._all_vars.get(boundsvar_name)
            if bounds_var is None:
                result = "is not a variable in the dataset"
                succeed = False
        if succeed:
            result = ""
        return result

    def check_coord_bounds(self, coord: NcVariableSummary) -> List[Tuple[str]]:
        """
        Validity-check the bounds of a coordinate (if any).

        Ok for _no_ bounds-attribute, but not if it is an empty string.
        Check: existence, n-dims, parent dimension, standard-name and units.

        Returns
        codes_and_messages : List[tuple(str, str)]
            a list of codes and messages, to be logged in the context of the
            parent coordinate variable.

        """
        bounds_name = coord.attributes.get("bounds")
        result_codes_and_messages = []

        def log_bounds_statement(code, msg):
            msg = f'has bounds="{bounds_name}", which {msg}.'
            result_codes_and_messages.append((code, msg))

        has_bounds = bounds_name is not None
        if has_bounds:
            msg = self.var_ref_problem(bounds_name)
            if msg != "":
                log_bounds_statement("R203", msg)
                has_bounds = False

        if has_bounds:
            # NB from the above check, we do have a bounds variable.
            bounds_var = self._all_vars[str(bounds_name)]
            bounds_dims = bounds_var.dimensions
            (coord_dim,) = coord.dimensions  # NB always has exactly 1
            if coord_dim not in bounds_dims:
                msg = (
                    f"has dimensions {bounds_dims!r}, which does not include "
                    f'the parent variable dimension, "{coord_dim}".'
                )
                log_bounds_statement("R203", msg)

            n_bounds_dims = len(bounds_dims)
            if n_bounds_dims != 2:
                msg = (
                    f"has dimensions {bounds_dims!r}, of which there should "
                    f"be 2, instead of {n_bounds_dims}."
                )
                log_bounds_statement("R203", msg)

            #
            # Advisory checks
            #

            def check_attr_mismatch(attr_name):
                coord_attr, bounds_attr = [
                    var.attributes.get(attr_name)
                    for var in (coord, bounds_var)
                ]
                if bounds_attr is not None and bounds_attr != coord_attr:
                    if coord_attr is None:
                        coord_attr = "<none>"
                    msg = (
                        f'has {attr_name}="{bounds_attr}", which does not '
                        f'match the parent {attr_name} of "{coord_attr}".'
                    )
                    log_bounds_statement("R203", msg)

            check_attr_mismatch("standard_name")
            check_attr_mismatch("units")

            # Do the data-values check.  This is potentially costly.
            if self.do_data_checks:
                # TODO: enable data-value checks by attaching lazy data arrays
                # to scan variables.
                assert bounds_var.data is not None
                raise ValueError("Not ready for data-value checks.")
                log_bounds_statement("A205", "???")

        return result_codes_and_messages

    def check_mesh_coordinates(
        self,
        meshvar: NcVariableSummary,
        attr_name: str,
    ):
        """Validity-check a coordinate attribute of a mesh-variable."""
        # Note: the content of the coords attribute was already checked

        # Elements which change as we scan the various coords.
        coord = None  # Changes as we scan them.
        common_msg_prefix = ""
        coord_context_str = ""

        # Function to emit a statement message, adding context as to the
        # specific coord variable.
        def log_coord(code, msg):
            LOG.state(
                code, "Mesh coordinate", coord.name, common_msg_prefix + msg
            )

        coord_names = property_namelist(meshvar.attributes.get(attr_name))
        for coord_name in coord_names:
            if coord_name not in self._all_vars:
                # This problem will already have been detected + logged.
                continue
            coord = self._all_vars[coord_name]
            common_msg_prefix = f"within {meshvar.name}:{attr_name} "
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
                mesh_dim = self._all_mesh_dims[meshvar.name][location]
                if coord_dim != mesh_dim:
                    msg = coord_context_str + (
                        f'has dimension "{coord_dim}", but the parent mesh '
                        f'{location} dimension is "{mesh_dim}".'
                    )
                    log_coord("R202", msg)
                # Check coord bounds (if any)
                # N.B. this *also* assumes a single dim for the primary var
                codes_and_messages = self.check_coord_bounds(coord)
                for code, msg in codes_and_messages:
                    log_coord(code, msg)

            #
            # Advisory notes..
            #

            # A201 should have 1-and-only-1 parent mesh : this is handled by
            # 'check_dataset', as it involves multiple meshes.

            # A202 floating-point type
            dtype = coord.dtype
            if dtype.kind != "f":
                log_coord(
                    "A202",
                    f'has type "{dtype}", which is not floating-point.',
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

    def check_mesh_connectivity(
        self,
        meshvar: NcVariableSummary,
        attr_name: str,
    ):
        """Validity-check a connectivity attribute of a mesh-variable."""
        attr_value = meshvar.attributes.get(attr_name)
        ok = attr_value is not None
        if ok:
            conn_name = property_as_single_name(attr_value)
            ok = conn_name is not None
        if ok:
            conn_var = self._all_vars.get(conn_name)
            ok = conn_var is not None
        if ok:
            # Add to our list variables 'allowed' to have a UGRID cf-role.
            self._allowed_cfrole_varnames.append(conn_name)

            msg_prefix = f'of mesh "{meshvar.name}" '

            def log_conn(errcode, msg):
                LOG.state(
                    errcode, "Mesh connectivity", conn_name, msg_prefix + msg
                )

            cf_role = conn_var.attributes.get("cf_role")
            if cf_role is None:
                log_conn("R301", "has no 'cf_role' attribute.")
            elif cf_role not in _VALID_CONNECTIVITY_ROLES:
                msg = (
                    f'has cf_role="{cf_role}", '
                    "which is not a valid UGRID connectivity attribute name."
                )
                log_conn("R302", msg)
            elif cf_role != attr_name:
                msg = (
                    f'has cf_role="{cf_role}", which is different from its '
                    f'role in the parent mesh, which is "{attr_name}".'
                )
                log_conn("R303", msg)

            conn_dims = conn_var.dimensions
            dims_msg = f"has dimensions {conn_dims!r}"
            if len(conn_dims) != 2:
                msg = (
                    f"{dims_msg}, of which there are "
                    f"{len(conn_dims)}, instead of 2."
                )
                log_conn("R304", msg)

            mesh_dims = self._all_mesh_dims[meshvar.name]
            is_parent_dim = [dim in mesh_dims.values() for dim in conn_dims]
            n_parent_dims = sum(is_parent_dim)
            if n_parent_dims == 0:
                msg = (
                    f"{dims_msg}, which does not contain any element "
                    f"dimension of the parent mesh."
                )
                log_conn("R305", msg)
            elif n_parent_dims == len(conn_dims):
                msg = (
                    f"{dims_msg}, which does not contain any dimension "
                    f"which is not an element dimension of the parent mesh."
                )
                log_conn("R306", msg)
            else:
                # Some are parent mesh-dims, and some not.
                # Just check that the *expected* mesh-dim is there.
                location = attr_name.split("_")[0]
                parent_dim = mesh_dims[location]
                if parent_dim not in conn_dims:
                    msg = (
                        f"{dims_msg}, which does not include the expected "
                        f"{location} dimension of the parent mesh, "
                        f'"{parent_dim}".'
                    )
                    log_conn("R307", msg)

            edgelike_conns = (
                "edge_node_connectivity",
                "boundary_node_connectivity",
            )
            if attr_name in edgelike_conns and n_parent_dims == 1:
                (conn_nonmesh_dim,) = (
                    dim
                    for dim, in_parent in zip(conn_dims, is_parent_dim)
                    if not in_parent
                )
                nonmesh_dim = self.file_scan.dimensions[conn_nonmesh_dim]
                nonmesh_length = nonmesh_dim.length
                if nonmesh_length != 2:
                    msg = (
                        f"{dims_msg}, which contains the non-mesh "
                        f'dimension "{conn_nonmesh_dim}", but this has '
                        f"length {nonmesh_length} instead of 2."
                    )
                    log_conn("R308", msg)

            index_value = conn_var.attributes.get("start_index")
            if index_value is not None:
                # Note: check value, converted to int.
                # This avoids an extra warning for strings like "0", "1",
                # since a non-integral type triggers an A302 warning anyway.
                if int(index_value) not in (0, 1):
                    msg = (
                        f'has start_index="{index_value}", which is not '
                        "either 0 or 1."
                    )
                    log_conn("R309", msg)

            if self.do_data_checks:
                if attr_name.endswith("_node_connectivity"):
                    # Check for missing values
                    msg = "may have missing indices (NOT YET CHECKED)."
                    log_conn("R310", msg)

            #
            # Advisory checks
            #

            # A301 1-and-only-1 parent mesh
            # done in 'check_dataset', as it involves multiple meshes

            if conn_var.dtype.kind != "i":
                msg = (
                    f'has type "{conn_var.dtype}", '
                    "which is not an integer type."
                )
                log_conn("A302", msg)

            if index_value is not None and index_value.dtype != conn_var.dtype:
                msg = (
                    f'has a start_index of type "{index_value.dtype}", '
                    "which is different from the variable type, "
                    f'"{conn_var.dtype}".'
                )
                log_conn("A303", msg)

            fill_value = conn_var.attributes.get("_FillValue")
            if (
                attr_name.endswith("_node_connectivity")
                and fill_value is not None
            ):
                msg = (
                    f"has a _FillValue attribute, which should not be present "
                    f'on a "{attr_name}" connectivity.'
                )
                log_conn("A304", msg)

            if self.do_data_checks:
                # check for missing indices
                msg = "may have missing indices (NOT YET CHECKED)."
                log_conn("A305", msg)

            if fill_value is not None and fill_value.dtype != conn_var.dtype:
                msg = (
                    f'has a _FillValue of type "{fill_value.dtype}", '
                    "which is different from the variable type, "
                    f'"{conn_var.dtype}".'
                )
                log_conn("A306", msg)

            if fill_value is not None and fill_value >= 0:
                msg = f'has _FillValue="{fill_value}", which is not negative.'
                log_conn("A307", msg)

            if self.do_data_checks:
                # check for missing indices
                msg = (
                    "may have indices which exceed the length of the element "
                    "dimension (NOT YET CHECKED)."
                )
                log_conn("A308", msg)

    def check_mesh_var(self, meshvar: NcVariableSummary) -> Dict[str, str]:
        """
        Run checks on a mesh variable.

        Parameters
        ----------
        meshvar : :class:`NcVariableSummary`
            meshvar to check

        """

        def log_meshvar(code, msg):
            LOG.state(code, "Mesh", meshvar.name, msg)

        # First check for bad 'cf_role' :
        # if wrong, meshvar can only have been identified by reference.
        cfrole_prop = meshvar.attributes.get("cf_role", None)
        if cfrole_prop != "mesh_topology":
            # This variable does not have the expected 'cf_role', so if we are
            # checking it, it must be referred to as 'mesh' by some variable.
            referring_var_name = self._mesh_referrers[meshvar.name]
            # Either there is no 'cf_role', or it is "wrong".
            msg = (
                f"appears to be a mesh, "
                f'since it is the value of "{referring_var_name}:mesh". '
                "But it has "
            )
            if cfrole_prop is None:
                msg += 'no "cf_role" property,'
                errcode = "R101"
            else:
                msg += f'a "cf_role" of "{cfrole_prop}",'
                errcode = "R102"
            msg += " which should be 'mesh_topology'."
            # N.B. do not identify as a Mesh, statement just says "variable"
            LOG.state(errcode, "", meshvar.name, msg)
            # Also, if the 'cf_role' was something else, then check it is a
            # valid option + emit an additional message if needed.
            if (
                cfrole_prop is not None
                and cfrole_prop not in _VALID_UGRID_CF_ROLES
            ):
                msg = (
                    f'has a "cf_role" of "{cfrole_prop}", '
                    "which is not a valid UGRID cf_role."
                )
                # "R102.a"
                log_meshvar("?", msg)

        # Check all other attributes of mesh vars.
        topology_dimension = meshvar.attributes.get("topology_dimension")
        if topology_dimension is None:
            log_meshvar("R103", 'has no "topology_dimension".')
        else:
            # Check the topology dimension.
            # In principle, this controls which other connectivity properties
            # may appear : In practice, it is better to parse those
            # independently, and then cross-check.
            if topology_dimension not in (0, 1, 2):
                msg = (
                    f'has topology_dimension="{topology_dimension}", '
                    "which is not 0, 1 or 2."
                )
                log_meshvar("R104", msg)
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
                log_meshvar(errcode, msg)

        # We will use the 'calculated' one to scope any remaining checks.
        # TODO: remove this, if it continues to be unused.
        # topology_dimension = appropriate_dim

        # Check all coordinate and connectivity attributes are valid "varlists"
        varlist_names = _VALID_MESHCOORD_ATTRS + _VALID_CONNECTIVITY_ROLES
        for attr in varlist_names:
            is_conn = attr in _VALID_CONNECTIVITY_ROLES
            attr_value = meshvar.attributes.get(attr)
            if attr_value is not None:
                ok = self.check_mesh_attr_is_varlist(meshvar, attr)
                var_names = property_namelist(attr_value)
                if not ok:
                    errcode = "R108" if is_conn else "R107"
                    msg = (
                        f'attribute "{attr}" = "{attr_value}", which is not '
                        "a list of variables in the dataset."
                    )
                    log_meshvar(errcode, msg)
                elif is_conn and len(var_names) != 1:
                    # "R106.a"
                    msg = (
                        f'has {attr}="{var_names!r}", which contains '
                        f"{len(var_names)} names, instead of 1."
                    )
                    log_meshvar("?", msg)

        # Work out the actual mesh dimensions.
        mesh_dims = {
            name: None for name in ("face", "edge", "node", "boundary")
        }
        self._all_mesh_dims[meshvar.name] = mesh_dims

        if "node_coordinates" not in meshvar.attributes:
            log_meshvar(
                "R109", "does not have a 'node_coordinates' attribute."
            )
        else:
            # Note: if a 'node_coordinates' attribute exists, then we already
            # checked that it is a valid varlist.
            # So don't re-raise any problems here, just press on.
            coord_names = property_namelist(
                meshvar.attributes["node_coordinates"]
            )
            if coord_names:
                coord_var = self._all_vars.get(coord_names[0])
                if coord_var:
                    # Answer is the first dimension, if any.
                    if len(coord_var.dimensions) > 0:
                        mesh_dims["node"] = coord_var.dimensions[0]

        def deduce_element_dim(location):
            # Identify the dim, and check consistency of relevant attributes.
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
                    msg = (
                        f'has an attribute "{dimattr_name}", which is not '
                        "a valid UGRID term, and may be "
                        'a mistake (ADVISORY)."'
                    )
                    log_meshvar("?", msg)
                    # TODO: add ADVISE code for this ?

            if dimension_name:
                # There is an explicit 'xxx_dimension' property.
                if connattr_name not in meshvar.attributes:
                    # "A106 ?"
                    msg = (
                        f'has an attribute "{dimattr_name}", '
                        "which is not valid "
                        f'since there is no "{connattr_name}".',
                    )
                    log_meshvar("?", msg)
                elif dimension_name in self.file_scan.dimensions:
                    mesh_dims[location] = dimension_name
                else:
                    errcode = {"edge": "R115", "face": "R117"}[location]
                    msg = (
                        f'has {dimattr_name}="{dimension_name}", which is not '
                        "a dimension in the dataset.",
                    )
                    log_meshvar(errcode, msg)
            elif connattr_name in meshvar.attributes:
                # No "xxx_dimension" attribute, but we *do* have
                # "xxx_node_connectivity", so mesh does _have_ this location.
                connvar_name = property_as_single_name(
                    meshvar.attributes[connattr_name]
                )
                conn_var = self._all_vars.get(connvar_name)
                if conn_var:
                    # Answer is the first dimension, if any.
                    if len(conn_var.dimensions) > 0:
                        mesh_dims[location] = conn_var.dimensions[0]

        deduce_element_dim("node")
        deduce_element_dim("boundary")
        deduce_element_dim("edge")
        deduce_element_dim("face")

        # Check that, if any connectivities have non-standard dim order, then a
        # dimension attribute exists.
        def var_has_nonfirst_dim(varname, dimname):
            conn_var = self._all_vars.get(varname)
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
            # Found connectivities with a nonstandard dim order for this dim.
            assert location in ("face", "edge")
            errcode = {"edge": "R116", "face": "R118"}[location]
            conn_names = [f'"{name}"' for name in conns]
            conn_names_str = ", ".join(conn_names)
            msg = (
                f'has no "{dim_attr}" attribute, but there are '
                f"{location} connectivities "
                f"with non-standard dimension order : {conn_names_str}."
            )
            log_meshvar(errcode, msg)

        # Check that all existing coordinates are valid.
        for coords_name in _VALID_MESHCOORD_ATTRS:
            location = coords_name.split("_")[0]
            # Only check coords of locations present in the mesh.
            # This avoids complaints about coords dis-connected by problems
            # with the topology identification.
            if mesh_dims[location]:
                self.check_mesh_coordinates(meshvar, coords_name)

        # Check that all existing connectivities are valid.
        for attr in _VALID_CONNECTIVITY_ROLES:
            self.check_mesh_connectivity(meshvar, attr)

        # deal with the optional elements (connectivities)
        def check_requires(errcode, attrname, location_1, location_2=None):
            exist = attrname in meshvar.attributes
            if exist:
                elems = [location_1]
                if location_2:
                    elems.append(location_2)
                required_elements = [
                    f"{name}_node_connectivity" for name in elems
                ]
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
                    log_meshvar(errcode, err_msg)

        check_requires("R114", "boundary_node_connectivity", "face")
        check_requires("R119", "face_face_connectivity", "face")
        check_requires("R120", "face_edge_connectivity", "face", "edge")
        check_requires("R121", "edge_face_connectivity", "face", "edge")

        # Advisory checks.
        if meshvar.dimensions:
            log_meshvar("A101", "has dimensions.")
        if "standard_name" in meshvar.attributes:
            log_meshvar("A102", 'has a "standard_name" attribute.')
        if "units" in meshvar.attributes:
            log_meshvar("A103", 'has a "units" attribute.')
        # NOTE: "A104" relates to multiple meshvars, so is handled in caller.

        return mesh_dims

    def check_meshdata_var(self, datavar: NcVariableSummary):
        def log_meshdata(errcode, msg):
            LOG.state(errcode, "Mesh data", datavar.name, msg)

        lis_name = datavar.attributes.get("location_index_set")
        mesh_name = datavar.attributes.get("mesh")
        location = datavar.attributes.get("location")
        # At least one of these is true, or we would not have identified this
        # as a mesh-data var.
        assert mesh_name is not None or lis_name is not None

        # Decide whether to check this as a lis-datavar or a mesh-datavar
        # This is designed to produce 3 possible "clash" errors:
        #   lis & mesh & ~location --> R506
        #   lis & location & ~mesh --> R507
        #   mesh & lis --> R501
        treat_as_lis = lis_name is not None and (
            mesh_name is None or location is None
        )

        # Initialise reference used for the generic parent dimension check
        parent_varname = None  # Can be either a meshvar or a lis
        parent_location = None
        if treat_as_lis:
            # Treat the datavar as a 'lis-datavar'
            #  --> has "location_index_set", but no "mesh" or "location"
            ref_msg = self.var_ref_problem(lis_name)
            if ref_msg:
                # Invalid 'location_index_set' reference
                msg = f'has location_index_set="{lis_name}", which {ref_msg}.'
                log_meshdata("R508", msg)
            else:
                # We have a valid lis var.  Take this as the 'parent' for
                # the generic dimension test R510
                parent_varname = str(lis_name)
                lis_var = self._lis_vars[parent_varname]
                # Also set the parent-location.
                # NOTE: we are not checking the lis-var here, only the datavar,
                # so just get a value that works if the lis is valid.
                parent_location = str(lis_var.attributes.get("location", ""))

            if mesh_name is not None:
                msg = (
                    "has a 'mesh' attribute, which is invalid since it is "
                    "based on a 'location_index_set' attribute."
                )
                log_meshdata("R506", msg)

            if location is not None:
                msg = (
                    "has a 'location' attribute, which is invalid since it is "
                    "based on a 'location_index_set' attribute."
                )
                log_meshdata("R507", msg)

        else:
            # Treat the datavar as a 'mesh-datavar'
            #  --> has "mesh" and "location", but no "location_index_set"
            ref_msg = self.var_ref_problem(mesh_name)
            if ref_msg:
                # Invalid 'mesh' reference
                msg = f'has mesh="{mesh_name}", which {ref_msg}.'
                log_meshdata("R502", msg)

            if lis_name is not None:
                msg = (
                    "has a 'location_index_set' attribute, which is invalid "
                    "since it is based on a 'mesh' attribute."
                )
                log_meshdata("R501", msg)

            if location is None:
                log_meshdata("R503", "has no 'location' attribute.")
            elif str(location) not in ("face", "edge", "node"):
                msg = (
                    f'has location="{location}", which is not one of '
                    '"face", "edge" or "node".'
                )
                log_meshdata("R504", msg)
            else:
                # Given a valid location, check that it exists in the parent
                if not ref_msg:
                    parent_varname = str(mesh_name)
                    parent_location = str(location)
                    assert parent_varname in self._all_mesh_dims
                    mesh_dims = self._all_mesh_dims[parent_varname]
                    parent_dim = mesh_dims[parent_location]
                    if parent_dim is None:
                        msg = (
                            f'has location="{location}", which is a location '
                            "that does not exist in the parent mesh, "
                            f'"{parent_varname}".'
                        )
                        log_meshdata("R505", msg)

        # Generic dimension testing, for either lis- or mesh-type datavars
        # First check there is only 1 mesh-dim
        data_dims = datavar.dimensions
        data_mesh_dims = [
            dim
            for dim in data_dims
            if any(
                dim in self._all_mesh_dims[some_mesh_name].values()
                for some_mesh_name in self._all_mesh_dims
            )
        ]
        n_data_mesh_dims = len(data_mesh_dims)
        if n_data_mesh_dims != 1:
            msg = (
                f"has dimensions {data_dims}, of which {n_data_mesh_dims} "
                "are mesh dimensions, instead of 1."
            )
            log_meshdata("R509", msg)
            data_meshdim = None  # cannot check against parent
        else:
            # We have a single element-dim : check against a parent mesh or lis
            (data_meshdim,) = data_mesh_dims

        if parent_varname and parent_location and data_meshdim:
            # If we have a valid parent ref, and single mesh dimension of the
            # datavar, check that they match
            mesh_dims = self._all_mesh_dims[parent_varname]
            parent_dim = mesh_dims[parent_location]
            if parent_dim is not None and data_meshdim != parent_dim:
                # Warn only if the parent_dim *exists*, but does not match
                # N.B. missing parent dim is checked elsewhere : R505 or R404
                if parent_varname in self._lis_vars:
                    typename = "location_index_set"
                else:
                    typename = "mesh"
                msg = (
                    f'has the element dimension "{data_meshdim}", which does '
                    f"not match the {parent_location} dimension of the "
                    f'"{parent_varname}" {typename}, which is "{parent_dim}".'
                )
                log_meshdata("R510", msg)

    def check_lis_var(self, lis_var: NcVariableSummary):
        """Validity-check a location-index-set variable."""

        # Add the lis element dimension into self._all_mesh_dims
        location = str(lis_var.attributes.get("location", ""))
        if location in ("node", "edge", "face"):
            dims = lis_var.dimensions
            if len(dims) == 1:
                # Lis has a valid location and single dim
                # So we can record 'our' dim as an element-dim
                (lis_dim,) = dims
                dims = self._all_mesh_dims.get(lis_var.name)
                if dims is None:
                    dims = {name: None for name in ("node", "edge", "face")}
                dims[location] = lis_dim
                self._all_mesh_dims[lis_var.name] = dims

        def log_lis(errcode, msg):
            LOG.state(errcode, "location-index-set", lis_var.name, msg)

        cf_role = lis_var.attributes.get("cf_role")
        if cf_role is None:
            log_lis("R401", "has no 'cf_role' attribute.")
        elif cf_role != "location_index_set":
            msg = f'has cf_role="{cf_role}", instead of "location_index_set".'
            log_lis("R401", msg)

        mesh_var = None  # Used to skip additional checks when mesh is bad
        mesh_name = lis_var.attributes.get("mesh")
        if mesh_name is not None:
            msg_ref = self.var_ref_problem(mesh_name)
            if msg_ref:
                msg = f'has mesh="{mesh_name}", which {msg_ref}.'
                log_lis("R402", msg)
            else:
                mesh_name = str(mesh_name)
                mesh_var = self._mesh_vars.get(mesh_name)
                if mesh_var is None:
                    msg = (
                        f'has mesh="{mesh_name}", '
                        "which is not a valid mesh variable."
                    )
                    log_lis("R402", msg)

        location = lis_var.attributes.get("location")
        parent_dim = None
        if location is None:
            log_lis("R403", "has no 'location' attribute.")
        elif str(location) not in ("face", "edge", "node"):
            msg = (
                f'has location="{location}", which is not one of '
                '"face", "edge" or "node".'
            )
            log_lis("R403", msg)
        elif mesh_var:
            # check the location exists in the parent mesh
            location = str(location)
            mesh_dims = self._all_mesh_dims[mesh_name]
            parent_dim = mesh_dims[location]
            if parent_dim is None:
                msg = (
                    f'has location="{location}", which is a location '
                    "that does not exist in the indicated mesh, "
                    f'"{mesh_name}".'
                )
                log_lis("R404", msg)
                # Don't attempt any further checks against the mesh
                mesh_var = None

        lis_dims = lis_var.dimensions
        n_lis_dims = len(lis_dims)
        if n_lis_dims != 1:
            msg = (
                f"has dimensions {lis_dims!r}, of which there are "
                f"{n_lis_dims} instead of 1."
            )
            log_lis("R405", msg)
        else:
            (lis_dim,) = lis_dims

        index_value = lis_var.attributes.get("start_index")
        if index_value is not None:
            # Note: check value, converted to int.
            # This avoids an extra warning for strings like "0", "1",
            # since a non-integral type triggers an A407 warning anyway.
            if int(index_value) not in (0, 1):
                msg = (
                    f'has start_index="{index_value}", which is not '
                    "either 0 or 1."
                )
                log_lis("R406", msg)

        #
        # Advisory checks
        #
        if lis_var.dtype.kind != "i":
            msg = f'has type "{lis_var.dtype}", which is not an integer type.'
            log_lis("A401", msg)

        if self.do_data_checks:
            # TODO: data checks
            log_lis("A402", "contains missing indices.")

        if "_FillValue" in lis_var.attributes:
            msg = (
                "has a _FillValue attribute, which should not be present "
                "on a location-index-set."
            )
            log_lis("A403", msg)

        if mesh_var:
            len_lis = self.file_scan.dimensions[lis_dim].length
            len_parent = self.file_scan.dimensions[parent_dim].length
            if len_lis >= len_parent:
                msg = (
                    f'has dimension "{lis_dim}", length {len_lis}, which is '
                    f"longer than the {location} dimension of the indicated "
                    f'mesh "{mesh_name}" : '
                    f'"{parent_dim}", length {len_parent}.'
                )
                log_lis("A404", msg)

        if self.do_data_checks:
            # TODO: data checks
            msg = "contains repeated index values."
            log_lis(
                "A405",
            )
            if mesh_var:
                msg = (
                    "contains index values which are outside the range of the "
                    f'indicated mesh "{mesh_name}" {location} dimension, '
                    f' : "{parent_dim}", range 1..{len_parent}.'
                )
                log_lis(
                    "A406",
                )

        if index_value is not None and index_value.dtype != lis_var.dtype:
            msg = (
                f'has a start_index of type "{index_value.dtype}", '
                "which is different from the variable type, "
                f'"{lis_var.dtype}".'
            )
            log_lis("A407", msg)

    def dataset_identify_containers(self):
        """
        Find "mesh" , "mesh data", and "location index set" variables,

        Also include possibles due to mesh/lis references from data variables.

        Results set as self properties :
            self._meshdata_vars
            self._mesh_vars
            self._lis_vars
            self._mesh_referrers
            self._lis_referrers

        """
        # Location index sets are those with a cf_role of 'location_index_set'
        self._lis_vars = vars_w_props(
            self._all_vars, cf_role="location_index_set"
        )

        # Mesh data variables are those with either a 'mesh' or
        # 'location_index_set' attribute, but excluding the lis-vars.
        self._meshdata_vars = {
            varname: var
            for varname, var in self._all_vars.items()
            if (
                varname not in self._lis_vars
                and (
                    "mesh" in var.attributes
                    or "location_index_set" in var.attributes
                )
            )
        }
        # Mesh vars are those with cf_role="mesh_topology".
        self._mesh_vars = vars_w_props(self._all_vars, cf_role="mesh_topology")

        # Scan for any meshvars referred to by 'mesh' or 'location_index_set'
        # properties in mesh-data vars.
        # These are included among potential meshdata- and lis- variables
        # (so they are detected + checked even without the correct cf_role)
        self._mesh_referrers = {}
        self._lis_referrers = {}
        for referrer_name, referrer_var in self._meshdata_vars.items():
            meshprop = referrer_var.attributes.get("mesh")
            meshvar_name = property_as_single_name(meshprop)
            if (
                meshvar_name is not None
                and meshvar_name in self._all_vars
                and meshvar_name not in self._mesh_vars
            ):
                # Add this reference to out list of all meshvars
                self._mesh_vars[meshvar_name] = self._all_vars[meshvar_name]
                # Record name of referring var.
                # N.B. potentially this can overwrite a previous referrer,
                # but "any one of several" will be OK for our purpose.
                self._mesh_referrers[meshvar_name] = referrer_name

            # Do something similar with lis references.
            meshprop = referrer_var.attributes.get("location_index_set")
            lisvar_name = property_as_single_name(meshprop)
            if (
                lisvar_name is not None
                and lisvar_name in self._all_vars
                and lisvar_name not in self._lis_vars
            ):
                # Add this reference to out list of all meshvars
                self._lis_vars[lisvar_name] = self._all_vars[lisvar_name]
                # Record name of referring var.
                self._lis_referrers[lisvar_name] = referrer_name

    def dataset_check_containers_and_map_dims(self):
        """
        Check all putative mesh + lis variables and collect dimension maps.
        Writes self._all_mesh_dims: {<mesh or lis name>: {location: dim-name}}

        Note: in checking the individual mesh variables, we also check all
        the coordinates and connectivities.

        This routine also sets self._allowed_cfrole_varnames

        """

        # Build a map of the dimensions of all the meshes,
        # all_meshes_dims: {meshname: {location: dimname}}
        self._all_mesh_dims = {}

        # This list of "UGRID variables" is used by 'dataset_global_checks' to
        # find any vars with a UGRID-style 'cf_role' that should not have one.
        # We don't include meshdata, or coordinate variables, which should
        # *not* have a cf_role anyway.
        # The connectivities are added by 'check_mesh_connectivity'.
        self._allowed_cfrole_varnames = list(self._mesh_vars.keys()) + list(
            self._lis_vars.keys()
        )

        # Check all mesh vars
        # Note: this call also fills in 'self._all_mesh_dims', and checks all
        # the attached coordinates and connectivites for each mesh.
        for meshvar in self._mesh_vars.values():
            self.check_mesh_var(meshvar)

        # Check all lis-vars
        # Note: this call also fills in 'self._all_mesh_dims'.
        for lis_var in self._lis_vars.values():
            self.check_lis_var(lis_var)

    def dataset_detect_shared_dims(self):
        """
        Check for any dimensions shared between meshes - an advisory warning.

        """
        # Convert all_meshes_dims: {meshname: {location: dimname}}
        # .. to dim_meshes: {dimname: [meshnames]}
        dim_meshes = {}
        for mesh, location_dims in self._all_mesh_dims.items():
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

    def dataset_detect_multiple_refs(self):
        """
        Check for any coords and conns referenced by multiple meshes.

        N.B. relevant errors are :
            * A201 coord should have 1-and-only-1 parent mesh
            * A301 connectivity should have 1-and-only-1 parent mesh

        """
        var_refs_meshes_attrs = {}
        all_ref_attrs = _VALID_MESHCOORD_ATTRS + _VALID_CONNECTIVITY_ROLES
        for some_meshname in sorted(self._mesh_vars):
            some_meshvar = self._mesh_vars[some_meshname]
            for some_refattr in all_ref_attrs:
                is_coord = some_refattr in _VALID_MESHCOORD_ATTRS
                attrval = some_meshvar.attributes.get(some_refattr, None)
                somevar_names = property_namelist(attrval)
                for somevar_name in somevar_names:
                    # NB only collect valid refs (to real variables)
                    if somevar_name in self._all_vars:
                        meshes = var_refs_meshes_attrs.get(somevar_name, set())
                        meshes.add((some_meshname, some_refattr))
                        var_refs_meshes_attrs[somevar_name] = meshes

        for some_varname, meshes_and_attrs in var_refs_meshes_attrs.items():
            some_var = self._all_vars[some_varname]  # NB have only 'real' refs
            if len(meshes_and_attrs) > 1:
                meshes_and_attrs = sorted(
                    meshes_and_attrs, key=lambda pair: pair[0]
                )
                refs_msg = ", ".join(
                    [
                        f"{some_mesh}:{attr_name}"
                        for some_mesh, attr_name in meshes_and_attrs
                    ]
                )
                msg = f"is referenced by multiple mesh variables : {refs_msg}."

                # Structurally, a var *could* be referenced as both a coord
                # *and* a connectivity.  But they have different required
                # numbers of dims, so we use that to decide what to call it.
                is_coord = len(some_var.dimensions) == 1
                if is_coord:
                    vartype = "Mesh coordinate"
                    code = "A201"
                else:
                    vartype = "Mesh connectivity"
                    code = "A301"
                LOG.state(code, vartype, some_varname, msg)

    def dataset_global_checks(self):
        """Do file-level checks not based on any particular variable type."""

        def log_dataset(errcode, msg):
            LOG.state(errcode, "", "", msg)

        # A901 "dataset contents should also be CF compliant" -- not checkable,
        # unless we integrate this code with cf-checker.

        # Check the global Conventions attribute for a UGRID version.
        conventions = self.file_scan.attributes.get("Conventions")
        if conventions is None:
            msg = ""
            log_dataset("A902", "dataset has no 'Conventions' attribute.")
        else:
            conventions = str(conventions)
            re_conventions = re.compile(r"UGRID-[0-9]+\.[0-9]+")
            if not re_conventions.search(conventions):
                # NOTE: just search.  Don't attempt to split, as usage of
                # comma/space/semicolon might be inconsistent, and we don't
                # need to care about that here.
                msg = (
                    f'dataset has Conventions="{conventions}", which does not '
                    "contain a UGRID convention statement of the form "
                    '"UGRID-<major>.<minor>".'
                )
                log_dataset("A903", msg)

        # Check for any unexpected 'cf_role' usages.
        # N.B. the logic here is that
        #   1) if it has a UGRID-type cf-role, then *either* it was already
        #       identified (and checked), *or* it generates a A904 warning
        #   2) if it has a CF cf-role, we don't comment
        #   3) if it has some other cf-role, this is unrecognised -> A905
        for var_name, var in self._all_vars.items():
            if (
                "cf_role" in var.attributes
                and var_name not in self._allowed_cfrole_varnames
            ):
                cf_role = str(var.attributes["cf_role"])
                if cf_role in _VALID_UGRID_CF_ROLES:
                    msg = (
                        f'has cf_role="{cf_role}", which is a UGRID defined '
                        "cf_role term, but the variable is not recognised as "
                        "a UGRID mesh, location-index-set or connectivity "
                        "variable."
                    )
                    LOG.state("A904", "netcdf", var_name, msg)
                elif cf_role not in _VALID_CF_CF_ROLES:
                    msg = (
                        f'has cf_role="{cf_role}", which is not a recognised '
                        "value defined in either CF or UGRID."
                    )
                    LOG.state("A905", "netcdf", var_name, msg)

    def check_dataset(self):
        """
        Run all conformance checks on the contained file scan.

        All results logged via :data:`LOG.state`.

        """

        self.dataset_identify_containers()
        self.dataset_check_containers_and_map_dims()

        # Check all the mesh-data vars
        for meshdata_var in self._meshdata_vars.values():
            self.check_meshdata_var(meshdata_var)

        # Do the checks which cut across different meshes
        self.dataset_detect_shared_dims()
        self.dataset_detect_multiple_refs()

        # Do the miscellaneous dataset-level checks
        # TODO: enable, when we fix the 'orphan connectivity' problem.
        # self.dataset_global_checks()


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
    checker = Checker(file_scan)
    checker.check_dataset()


def output_report(print_func=print):
    # Emit the output results.
    logs = LOG.report_statement_logrecords()
    print_func("")
    print_func("UGRID conformance checks complete.")
    print_func("")
    if LOG.N_FAILURES + LOG.N_WARNINGS == 0:
        print_func("No problems found.")
    else:
        if logs:
            print_func("List of checker messages :")
            for log in logs:
                print_func("  " + log.msg)
            print_func("")
        print_func(
            f"Total of {LOG.N_WARNINGS + LOG.N_FAILURES} problems logged : \n"
            f"  {LOG.N_FAILURES} Rxxx requirement failures\n"
            f"  {LOG.N_WARNINGS} Axxx advisory recommendation warnings"
        )
    print_func("")
    print_func("Done.")


def check_dataset(
    file: Union[NcFileSummary, AnyStr, Path],
    print_results: bool = False,
    log_summary: bool = False,
    filter_level: int = logging.INFO,
) -> List[LogRecord]:
    """
    Run UGRID conformance checks on a file.

    Log statements regarding any problems found to the :data:`_LOG` logger.
    Return the accumulated log records of problems found.
    Optionally print logger messages and/or log a summary of results.

    Parameters
    ----------
    file : filepath string, Path or :class:`NcFileSummary`
        path to, or representation of a netcdf input file
    print_results : bool, default=True
        whether to print warnings as they are logged
    log_summary : bool, default=False
        whether to add a results summary to the log record

    Returns
    -------
    checker_warnings : list of :class:LogRecord
            A list of log records, one for each problem identified.
    """
    LOG.reset()
    LOG.enable_reports_printout(print_results)
    LOG.set_filter_level(filter_level)

    if isinstance(file, str):
        file_path = Path(file)
    elif isinstance(file, Path):
        file_path = file
    if isinstance(file, NcFileSummary):
        file_scan = file
    else:
        file_scan = scan_dataset(file_path)

    check_dataset_inner(file_scan)

    if log_summary:
        # Add a summary into the log record.
        LOG.set_filter_level(logging.DEBUG)  # log the 'printonly' calls
        output_report(print_func=LOG.printonly)

    return LOG.report_statement_logrecords()
