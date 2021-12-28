"""
Utility routines for handling netcdf file information stored in a
:class:`ugrid_checks.nc_dataset_scan.NcFileSummary`.

"""


def property_namelist(np_property_value):
    # Return a list of names from a numpy string property value.
    # Treat a None value as an empty string.
    result = []
    if np_property_value is not None:
        if np_property_value.dtype.kind == "U":
            # Don't handle non-string values.  Simply return empty.
            result = str(np_property_value).split()
            result = [val for val in result if val]
    return result


def property_as_single_name(np_property_value):
    # Return the single name in a property value.
    # If there is not exactly one name, return None.
    result = None
    if np_property_value is not None:
        nameslist = property_namelist(np_property_value)
        if len(nameslist) == 1:
            result = nameslist[0]
    return result


def vars_w_props(varsdict, **kwargs):
    """
    Extract vars from a dataset scan  dict, {name:props}, returning only those
    where each <attribute>=<value>, defined by the given keywords.
    Except that '<key>="*"' means that '<key>' merely _exists_, with any value.

    Kwargs:
    * varsdict (map name: NcVariableSummary):
        dataset vars, as in  a :class:`NcFileSummary`.variables.

    Returns:
    * extracted_vars (map name: NcVariableSummary):
        a map containing the dataset variables with the required properties.

    """

    def check_attrs_match(var):
        result = True
        for key, val in kwargs.items():
            # if key == '_DIMS':
            #     result = var.dimensions == val
            # else:
            attrs = var.attributes
            result = key in attrs
            if result:
                # val='*'' for a simple existence check
                result = (val == "*") or attrs[key] == val
            if not result:
                break
        return result

    result = {
        name: var for name, var in varsdict.items() if check_attrs_match(var)
    }

    return result


# def vars_meshnames(varsdict):
#     """Return the names of all the mesh variables (found by cf_role)."""
#     return list(vars_w_props(varsdict, cf_role="mesh_topology").keys())
#
#
# def vars_w_dims(varsdict, dim_names):
#     """Subset a vars dict, returning those which map all the
#     specified dims."""
#     result = {
#         name: var
#         for name, var in varsdict.items()
#         if all(dim in var.dimensions for dim in dim_names)
#     }
#     return result
#
#
# def vars_meshdim(varsdict, location, mesh_name=None):
#     """
#     Extract a dim-name for a given element location.
#
#     Args:
#         * vars (varsdict):
#             file varsdict, as returned from 'snapshot_dataset'.
#         * location (string):
#             a mesh location : 'node' / 'edge' / 'face'
#         * mesh_name (string or None):
#             If given, identifies the mesh var.
#             Otherwise, find a unique mesh var (i.e. there must be exactly 1).
#
#     Returns:
#         dim_name (string)
#             The dim-name of the mesh dim for the given location.
#
#     TODO: relies on the element having coordinates, which in future will not
#         always be the case. This can be fixed
#
#     """
#     if mesh_name is None:
#         # Find "the" meshvar -- assuming there is just one.
#         (mesh_name,) = vars_meshnames(varsdict)
#     mesh_props = varsdict[mesh_name].attributes
#     loc_coords = property_namelist(mesh_props[f"{location}_coordinates"])
#     (single_location_dim,) = varsdict[loc_coords[0]].dimensions
#     return single_location_dim
