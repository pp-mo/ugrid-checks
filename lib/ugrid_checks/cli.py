from argparse import ArgumentParser
import re

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
            'ignore all warnings ("Axxx"= advise codes), '
            'and only report errors ("Rxxx"= require codes).'
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
        "-i",
        "--ignore",
        type=str,
        action="append",  # multiple occurs can add together
        help="a list of errorcodes to ignore.",
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

    if not args.ignore:
        ignore_codes = None
    else:
        # Find matches for the
        ignore_codes_iter = re.finditer(
            "(A|R)[0-9]{3}", " ".join(args.ignore).upper()
        )
        # NOTE: this list comprehension is currently triggering a war between
        # black v22.1 and flake8 v4.0
        # ignore_codes = [
        #     match.string[match.start() : match.end()]
        #     for match in ignore_codes
        # ]
        # So, here is an ugly alternative..
        ignore_codes = []
        for match in ignore_codes_iter:
            i_start, i_end = match.start(), match.end()
            match_text = match.string[i_start:i_end]
            ignore_codes.append(match_text)
        print(f"\nIgnoring codes:\n  {', '.join(ignore_codes)}")

    checker = check_dataset(
        file=args.file,
        print_summary=False,  # print summary separately, if needed
        omit_advisories=args.errorsonly,
        ignore_codes=ignore_codes,
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
