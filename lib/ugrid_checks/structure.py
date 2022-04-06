import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Union

from ugrid_checks.nc_dataset_scan import NcDimSummary, NcVariableSummary

# Basic classes with convenience names
Varname = str
Location = str  # From _VALID_UGRID_LOCATIONS only
Role = str  # From _VALID_UGRID_CF_ROLES only
Var = NcVariableSummary
Dim = NcDimSummary
VarsMap = Dict[Varname, Var]
DimsMap = Dict[Varname, Dim]


# Initially pre-set these types, to avoid circular dependencies
UgridMesh = Any
UgridLis = Any


# Assign this to make a dataclass field default to an empty dict.
# Required for dataclass fields, as mutable init values are not permitted.
def emptydict():
    return dataclasses.field(default_factory=dict)


@dataclass()
class UgridDatavar:
    """
    Represents a mesh-data-variable.
    """

    name: str
    var: Var  # original low-level data
    # N.B. UgridMesh/UgridLis are *not* the actual final types at this point,
    # to avoid circular definitions
    lis: Union[UgridLis, None] = None
    mesh: Union[UgridMesh, None] = None
    location: Union[Location, None] = None


@dataclass
class _CoordsLocationMap:
    node: VarsMap = emptydict()
    edge: VarsMap = emptydict()
    face: VarsMap = emptydict()


@dataclass()
class UgridMesh:
    """
    Represents a mesh.
    """

    name: str
    var: Var  # original low-level data
    # N.B. 'UgridLis' is *not* the final type, to avoid circular definitions
    location_coords: Dict[Location, VarsMap] = _CoordsLocationMap()
    role_conns: Dict[Role, Var] = emptydict()
    # Additional summary info
    location_dims: DimsMap = emptydict()
    all_dims: DimsMap = emptydict()
    lisets: Dict[Varname, UgridLis] = emptydict()
    datavars: Dict[Varname, UgridDatavar] = emptydict()


@dataclass()
class UgridLis:
    """
    Represents a location-index-set.
    """

    name: str
    var: Var
    mesh: UgridMesh
    location: Location
    # Additional summary info
    datavars: Dict[Varname, UgridDatavar] = emptydict()


@dataclass
class _Dims_MeshNonmeshAll:
    all: DimsMap = emptydict()
    mesh: DimsMap = emptydict()
    nonmesh: DimsMap = emptydict()


@dataclass
class _Vars_MeshNonmeshAll:
    all: VarsMap = emptydict()
    mesh: VarsMap = emptydict()
    nonmesh: VarsMap = emptydict()


@dataclass()
class UgridDataset:
    """
    Represents the mesh structure of an entire dataset.
    """

    dims: _Dims_MeshNonmeshAll = _Dims_MeshNonmeshAll()
    vars: _Vars_MeshNonmeshAll = _Vars_MeshNonmeshAll()
    meshes: Dict[Varname, UgridMesh] = emptydict()
    lises: Dict[Varname, UgridLis] = emptydict()
    datavars: Dict[Varname, UgridDatavar] = emptydict()
