# ugrid-checks
**Version : 0.1 (beta)**

A utility to check netcdf files to the [UGRID specification](http://ugrid-conventions.github.io/ugrid-conventions/).

It tests files against the UGRID [conformance rules](https://ugrid-conventions.readthedocs.io/en/conformance/conformance/),
and can also produce a summary of the mesh content of a file.

  * [Installation](#installation)
  * [Command Line : checking](#command-line--checking)
    * [Basic usage](#basic-usage)
    * [Controlling checks](#controlling-checks)
  * [Command Line : structure analysis](#command-line--structure-analysis)
  * [List of Conformance Rules and codes](#list-of-conformance-rules-and-codes)
  * [Limitations](#limitations)
  * [Python API](#python-api)
  * [Runtime Requirements](#runtime-requirements)


## Installation
Available on PyPI.

To install:
```commandline
> pip install ugrid-checker
```

## Command Line : checking
```commandline
$ ugrid-checker -h
usage: ugrid-checker [-h] [-q] [-e] [-s] [--nonmesh] [-i IGNORE] [-v] file

Check a netcdf-CF file for conformance to the UGRID conventions.

positional arguments:
  file                  File to check.

optional arguments:
  -h, --help            show this help message and exit
  -q, --quiet           don't print a checking report if there were no
                        problems
  -e, --errorsonly      ignore all warnings ("Axxx"= advise codes), and only
                        report errors ("Rxxx"= require codes).
  -s, --summary         print a summary of UGRID mesh information found in the
                        file
  --nonmesh             include a list of non-mesh variables in the summary
  -i IGNORE, --ignore IGNORE
                        a list of errorcodes to ignore.
  -v, --version         print version information
$
```

### Basic usage
```commandline
$ ugrid-checker data_C4.nc

UGRID conformance checks complete.

No problems found.

Done.
$
```
```commandline
$ ugrid-checker data_C4_warn_error.nc 

UGRID conformance checks complete.

List of checker messages :
  *** FAIL R106 : Mesh variable "topology" attribute 'face_coordinates' refers to a variable "longitude", but there is no such variable in the dataset.
  *** FAIL R108 : Mesh variable "topology" has face_coordinates="latitude longitude", which is not a list of variables in the dataset.
  ... WARN A304 : Mesh connectivity variable "face_nodes" of mesh "topology" has a '_FillValue' attribute, which should not be present on a "face_node_connectivity" connectivity.

Total of 3 problems logged :
  2 Rxxx requirement failures
  1 Axxx advisory recommendation warnings

Done.
$
```

### Controlling checks
The ``-e`` / ``--errorsonly`` option checks only against the requirements (aka "errors"),
and skips the recommendations (aka "warnings").

The ``-i`` / ``--ignore`` option skips particular checks according to their Axxx/Rxxx codes.  
See [List of codes](#list-of-conformance-rules-and-codes).

Example:
```commandline
$ ugrid-checker data_C4_warn_error.nc --errorsonly --ignore r106,r108

Ignoring codes:
  R106, R108

UGRID conformance checks complete.

No problems found.

Done.
$
```

## Command line : structure analysis
The ``-s`` / ``--summary`` option prints a summary of a file's mesh content.
```commandline
$ ugrid-checker data_C4.nc --summary --quiet

File mesh structure
-------------------
Meshes
    "topology"
        node("num_node")
            coordinates : "node_lat", "node_lon"
        face("dim0")
            face_node_connectivity : "face_nodes"
            coordinates : "latitude", "longitude"

Mesh Data Variables
    "sample_data"
        mesh : "topology"
        location : "face"

$
```

## List of Conformance Rules and codes
All the error/warning codes used are defined in the UGRID conformance rules. 
Each has an identifying code : "Rxxx" for requirements or "Axxx" for advisory rules.

See the list here : [UGRID Draft Conformance Rules](https://ugrid-conventions.readthedocs.io/en/conformance/conformance/) 

Note : these are currently *only* available in this preliminary draft version,
not yet accepted into the UGRID spec.

## Limitations
#### No data value checks
At present, none of the rules which test *actual data values in variables* are implemented.  
   - For example : [A305](https://ugrid-conventions.readthedocs.io/en/conformance/conformance/#A305) --
    "a connectivity that contains missing indices should have a `_Fillvalue` property"

It is intended that these checks will be added later, enabled by a new flag such as ```--datachecks```


## Python API
The checker is provided as an importable module "ugrid_checks".

Running the module directly calls the command-line interface,  
e.g. `$ python -m ugrid_checks file.nc` is the same as `$ ugrid-checker file.nc`. 

Within Python, the module can be used like this : 
```ignorelang
>>> from ugrid_checks.check import check_dataset
>>> print(check_dataset.__doc__)

    Run UGRID conformance checks on a file.

    Optionally print a result summary.
    Optionally ignore errors below a logging level.
    Returns a checker object with a file analysis and checking log records.

    Parameters
    ----------
    file : string, Path or :class:`NcFileSummary`
        path to, or representation of a netcdf input file
    print_summary : bool, default=True
        print a results summary at the end
    omit_advisories : bool, default False
        If set, log only 'requirements' Rxxx statements, and ignore the
        advisory 'Axxx' ones.
    ignore_codes : list(str) or None, default None
        A list of error codes to ignore.

    Returns
    -------
    checker : Checker
        A checker for the file.

    
>>>
>>> checker = check_dataset('data_C4_warn.nc', print_summary=False)
>>> 
>>> type(checker)
<class 'ugrid_checks.check.Checker'>
>>>
>>> print(checker.checking_report())
UGRID conformance checks complete.

List of checker messages :
  ... WARN A304 : Mesh connectivity variable "face_nodes" of mesh "topology" has a '_FillValue' attribute, which should not be present on a "face_node_connectivity" connectivity.

Total of 1 problems logged :
  0 Rxxx requirement failures
  1 Axxx advisory recommendation warnings

Done.
>>> 
>>> print(checker.structure_report())
Meshes
    "topology"
        node("num_node")
            coordinates : "node_lat", "node_lon"
        face("dim0")
            face_node_connectivity : "face_nodes"
            coordinates : "latitude", "longitude"

Mesh Data Variables
    "sample_data"
        mesh : "topology"
        location : "face"
>>>
```

For the time being, there is no built API documentation :
Please refer to code docstrings for more detail. 


## Runtime Requirements
  * Python >= 3.7
  * [netCDF4](https://github.com/Unidata/netcdf4-python)

