"""Slightly crude, but ugrid-check-specific logging."""
import logging


class UgridLogHandler(logging.Handler):
    """A logging handler which simply accumulates logged messages in a list."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def emit(self, record):
        self.logs.append(record)

    def reset(self):
        self.logs = []


class CheckLoggingInterface:
    """
    Purpose-specific interface for logging ugrid conformance check messages.

    Contains a regular :class:`~logging.Handler` and a
    :class:`UgridLogHandler` which is connectedd to it.

    """

    def __init__(self, logger=None, handler=None):
        if logger is None:
            logger = logging.Logger("ugrid_conformance")
        self._logger = logger
        if handler is None:
            handler = UgridLogHandler(level=logging.INFO)
        self._handler = handler
        #: Total number of warning =ADVISE statements logged (since reset).
        self.N_WARNINGS = 0
        #: Total number of error =REQUIRE statements logged (since reset).
        self.N_FAILURES = 0
        self._logger.addHandler(self._handler)
        self.reset()

    def reset(self):
        self._handler.reset()
        self.N_WARNINGS = 0
        self.N_FAILURES = 0
        self._ENABLE_PRINT = True

    def enable_reports_printout(self, print_statements=True):
        """Enable/disable printing of logged messages."""
        self._ENABLE_PRINT = print_statements

    def report_statement_logrecords(self):
        """Return the report log as a list of :class:`~logging.LogRecords`."""
        return self._handler.logs

    def report(self, msg, level=logging.INFO, *args):
        """
        Output a free-form message with a given level.

        A low-level routine.  Use state/printonly in preference.

        """
        if self._ENABLE_PRINT:
            print(msg)
        self._logger.log(level, msg, *args)

    @staticmethod
    def _statement(vartype, var, msg):
        if vartype:
            result = vartype + f' variable "{var.name}"'
        else:
            result = f'Variable "{var.name}"'
        result += f" {msg}"
        return result

    def state(self, statement: str, vartype, var, msg):
        """
        Log a message relating to a given statement code.

        This constructs a semi-automated message, referencing the statement
        code and a primary dataset variable 'var', of type 'vartype'
        (i.e. its role in the dataset, e.g 'mesh' or 'connectivity').

        Associates a statment code with the logged message.
        The statement code also determines the message level :
        * 'R' ("require") codes log with logging.ERROR.
        * 'A' ("advise") codes log with logging.WARN.
        * '?' (informational) codes log with logging.INFO.

        """
        assert len(statement) >= 1
        statement_type = statement[0]
        assert statement_type in ("R", "A", "?")
        try:
            statement_num = int(statement[1:])
        except ValueError:
            statement_num = 0
        if statement_type == "R":
            # For messages with a 'REQUIRE' type code.
            self.N_FAILURES += 1
            msg = f"*** FAIL R{statement_num:03d} : " + self._statement(
                vartype, var, msg
            )
            self.report(msg, logging.ERROR, statement_num)
        elif statement_type == "A":
            # For messages with an 'ADVISE' type code.
            self.N_WARNINGS += 1
            msg = f"... WARN A{statement_num:03d} : " + self._statement(
                vartype, var, msg
            )
            self.report(msg, logging.WARN, statement_num)
        elif statement_type == "?":
            # For messages which don't have a code assigned.
            self.report(self._statement(vartype, var, msg))

    def printonly(self, msg, *args):
        """
        Log a debug-level message.

        A low-level alternative to 'state'.
        By default, these messages will be printed but *not* logged.
        ( to log them, change the filter level of the logger).

        """
        self.report(msg, logging.DEBUG, *args)


LOG = CheckLoggingInterface()
