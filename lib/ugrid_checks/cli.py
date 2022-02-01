from argparse import ArgumentParser

import ugrid_checks
from ugrid_checks.check import check_dataset


def make_parser():
    # Process cli
    parser = ArgumentParser(
        description=(
            "Check a netcdf-CF file for conformance to the UGRID conventions."
        )
    )
    parser.add_argument("file", type=str, help="File to check.")
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="don't print a checking report if there were no problems",
    )
    parser.add_argument(
        "-e",
        "--errorsonly",
        action="store_true",
        help=(
            'only report errors ("Rxxx"= require codes), '
            'i.e. suppress warnings ("Axxx"= advise codes)'
        ),
    )
    parser.add_argument(
        "-s",
        "--summary",
        action="store_true",
        help="print a summary of UGRID mesh information found in the file",
    )
    parser.add_argument(
        "--nonmesh",
        action="store_true",
        help="include a list of non-mesh variables in the summary",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        help="print version information",
    )
    return parser


def call_cli(args=None):
    parser = make_parser()
    args = parser.parse_args(args)

    if args.version:
        checker_vsn = ugrid_checks.__version__
        cf_vsn = ugrid_checks._cf_version
        print("")
        print(f"UGRID file checker, version {checker_vsn}")
        print(f"    valid to CF version : {cf_vsn}")

    checker = check_dataset(
        file=args.file,
        print_summary=False,  # print summary separately, if needed
        omit_advisories=args.errorsonly,
    )

    if args.summary:
        print("")
        print("File mesh structure")
        print("-------------------")
        print(checker.structure_report(include_nonmesh=args.nonmesh))
        print("")

    rc = 0
    log = checker.logger
    if log.N_FAILURES > 0:
        rc = 1
    if not args.errorsonly:
        if log.N_WARNINGS > 0:
            rc = 1

    if rc != 0 or not args.quiet:
        print(checker.checking_report())

    return rc
