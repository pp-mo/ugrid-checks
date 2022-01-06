"""Slightly crude, but ugrid-check-specific logging."""
import logging
from typing import List, Union


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
    :class:`UgridLogHandler` which is connected to it.

    """

    def __init__(self, logger=None, handler=None):
        if logger is None:
            logger = logging.Logger("ugrid_conformance")
        self._logger = logger
        if handler is None:
            handler = UgridLogHandler(level=logging.INFO)
        self._handler = handler
        #: Control printing of messages as they are logged (debug).
        self._ENABLE_PRINT = False
        self._print_func = print
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
        self._ENABLE_PRINT = False

    def enable_reports_printout(self, print_statements=False, print_func=None):
        """Enable/disable printing of logged messages."""
        if print_func:
            self._print_func = print_func
        self._ENABLE_PRINT = print_statements

    def set_filter_level(self, level: int):
        self._handler.setLevel(level)

    def report_statement_logrecords(self) -> List[logging.LogRecord]:
        """Return the report log as a list of :class:`~logging.LogRecords`."""
        # Note: return a list copy, as a snapshot
        return self._handler.logs[:]

    def report(self, msg, level=logging.INFO, *args):
        """
        Output a free-form message with a given level.

        A low-level routine.  Use state/printonly in preference.

        """
        if self._ENABLE_PRINT:
            self._print_func(msg)
        self._logger.log(level, msg, *args)

    @staticmethod
    def _statement(vartype: str, varname: str, msg: str):
        if vartype:
            result = vartype + f' variable "{varname}" '
        elif varname:
            result = f'Variable "{varname}" '
        else:
            result = ""
        result += f"{msg}"
        return result

    def state(
        self,
        statement: str,
        vartype: Union[str, None],
        varname: Union[str, None],
        msg: str,
    ):
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

        Parameters
        ----------
        statement : str
            the code for the specific problem (a short string like "A103")
        vartype : str or None
            the 'role' of a variable which is the focus of the problem (if any)
        varname : str or None
            the name of a variable which is the focus of the problem (if any)
        msg : str
            a description of the problem

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
                vartype, varname, msg
            )
            self.report(msg, logging.ERROR, statement_num)
        elif statement_type == "A":
            # For messages with an 'ADVISE' type code.
            self.N_WARNINGS += 1
            msg = f"... WARN A{statement_num:03d} : " + self._statement(
                vartype, varname, msg
            )
            self.report(msg, logging.WARN, statement_num)
        elif statement_type == "?":
            # For messages which don't have a code assigned.
            self.report(self._statement(vartype, varname, msg))

    def printonly(self, msg, *args):
        """
        Log a debug-level message.

        A low-level alternative to 'state'.
        By default, these messages will be printed but *not* logged.
        ( to log them, change the filter level of the logger).

        """
        self.report(msg, logging.DEBUG, *args)
