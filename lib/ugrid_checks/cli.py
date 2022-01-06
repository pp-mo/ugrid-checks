from argparse import ArgumentParser

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
        help="print nothing unless there are problems.",
    )
    parser.add_argument(
        "-e",
        "--errorsonly",
        action="store_true",
        help=(
            'Only report errors ("Rxxx"= require codes), '
            'i.e. suppress warnings ("Axxx"= advise codes).'
        ),
    )
    return parser


def call_cli(args=None):
    parser = make_parser()
    args = parser.parse_args(args)

    checker = check_dataset(
        file=args.file,
        print_summary=False,  # print summary separately, if needed
        omit_advisories=args.errorsonly,
    )
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
