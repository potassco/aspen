"""
Setup project wide loggers.

This is a thin wrapper around Python's logging module. It supports colored
logging.
"""

import logging
import sys
from functools import partial
from logging import StreamHandler
from typing import Callable, Optional, TextIO
from unittest import TestCase

import tree_sitter as ts

# pylint: disable=import-error,no-name-in-module
from clingo.core import MessageCode

COLORS = {
    "GREY": "\033[90m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "RED": "\033[91m",
    "NORMAL": "\033[0m",
}


class SingleLevelFilter(logging.Filter):
    """
    Filter levels.
    """

    passlevel: int
    reject: bool

    def __init__(self, passlevel: int, reject: bool):
        # pylint: disable=super-init-not-called
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record: logging.LogRecord) -> bool:
        if self.reject:
            return record.levelno != self.passlevel  # nocoverage

        return record.levelno == self.passlevel


def configure_logging(stream: TextIO, level: int, use_color: bool) -> None:
    """
    Configure application logging.
    """

    def format_str(color: str) -> str:
        if use_color:
            return (
                f"{COLORS[color]}%(levelname)s:{COLORS['GREY']}"
                f"  - %(message)s{COLORS['NORMAL']}"
            )
        return "%(levelname)s:  - %(message)s"  # nocoverage

    def make_handler(level: int, color: str) -> "logging.StreamHandler[TextIO]":
        handler = logging.StreamHandler(stream)
        handler.addFilter(SingleLevelFilter(level, False))
        handler.setLevel(level)
        formatter = logging.Formatter(format_str(color))
        handler.setFormatter(formatter)
        return handler

    handlers = [
        make_handler(logging.INFO, "GREEN"),
        make_handler(logging.WARNING, "YELLOW"),
        make_handler(logging.DEBUG, "BLUE"),
        make_handler(logging.ERROR, "RED"),
    ]
    logging.basicConfig(handlers=handlers, level=level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.
    """
    return logging.getLogger(name)


CLINGO_FSTRING = "clingo: %s"


def log_clingo_message(
    message_code: MessageCode, message: str, logger: logging.Logger
) -> None:  # nocoverage
    "Log clingo message at the appropriate level"
    if message_code is MessageCode.AtomUndefined:
        logger.info(CLINGO_FSTRING, message)
    elif message_code is MessageCode.FileIncluded:
        logger.warn(CLINGO_FSTRING, message)
    elif message_code is MessageCode.GlobalVariable:
        logger.info(CLINGO_FSTRING, message)
    elif message_code is MessageCode.OperationUndefined:
        logger.info(CLINGO_FSTRING, message)
    # not sure what the appropriate log level for "Other" is... just do info for now
    elif message_code is MessageCode.Other:
        logger.info(CLINGO_FSTRING, message)
    elif message_code is MessageCode.RuntimeError:
        logger.error(CLINGO_FSTRING, message)
    elif message_code is MessageCode.VariableUnbounded:
        logger.info(CLINGO_FSTRING, message)


def get_clingo_logger(
    logger: logging.Logger,
) -> Callable[[MessageCode, str], None]:
    """Return a callback function to be used by a clingo.Control
    object to log to input logger."""
    return partial(log_clingo_message, logger=logger)


def log_ts_message(logtype: ts.LogType, message: str, logger: logging.Logger) -> None:
    """Log tree-sitter message."""
    if logtype is ts.LogType.PARSE:
        logger.debug("tree-sitter parser: %s", message)
    elif logtype is ts.LogType.LEX:
        logger.debug("tree-sitter lexer: %s", message)


def get_ts_logger(
    logger: logging.Logger,
) -> Callable[[ts.LogType, str], None]:
    """Return a callback function to be used when initializing a
    clingo.core.Library object to log to input logger.

    """
    return partial(log_ts_message, logger=logger)


# class to redirect logs
# https://stackoverflow.com/questions/69200881/how-to-get-python-unittest
# -to-show-log-messages-only-on-failed-tests
class LoggerRedirector:  # nocoverage
    """Class with methods to update the stream on the log handler to
    point to the buffer unittest has set up for capturing the test
    output, as well as to reset them after test has run.

    """

    # Keep a reference to the real streams so we can revert
    _real_stdout = sys.stdout
    _real_stderr = sys.stderr

    @staticmethod
    def all_loggers() -> list[logging.Logger]:
        "Store all loggers."
        loggers = [logging.getLogger()]
        loggers += [
            logging.getLogger(name)
            for name in logging.root.manager.loggerDict  # pylint: disable=no-member
        ]
        return loggers

    @classmethod
    def redirect_loggers(
        cls, fake_stdout: Optional[TextIO] = None, fake_stderr: Optional[TextIO] = None
    ) -> None:
        "Redirect loggers before test."
        if (not fake_stdout or fake_stdout is cls._real_stdout) and (
            not fake_stderr or fake_stderr is cls._real_stderr
        ):
            return
        for logger in cls.all_loggers():
            for handler in logger.handlers:
                if isinstance(handler, StreamHandler):
                    if handler.stream is cls._real_stdout:
                        handler.setStream(fake_stdout)
                    if handler.stream is cls._real_stderr:
                        handler.setStream(fake_stderr)

    @classmethod
    def reset_loggers(
        cls, fake_stdout: Optional[TextIO] = None, fake_stderr: Optional[TextIO] = None
    ) -> None:
        "Reset loggers after test"
        if (not fake_stdout or fake_stdout is cls._real_stdout) and (
            not fake_stderr or fake_stderr is cls._real_stderr
        ):
            return
        for logger in cls.all_loggers():
            for handler in logger.handlers:
                if isinstance(handler, StreamHandler):
                    if handler.stream is fake_stdout:
                        handler.setStream(cls._real_stdout)
                    if handler.stream is fake_stderr:
                        handler.setStream(cls._real_stderr)


class TestCaseWithRedirectedLogs(TestCase):
    """A TestCase subclass that redirects stdin and stderr in such a
    fashion as to redirect log handlers to the buffer unittest
    uses. This allows us to display logs when a test fails."""

    def setUp(self) -> None:
        # unittest has reassigned sys.stdout and sys.stderr by this point
        LoggerRedirector.redirect_loggers(fake_stdout=sys.stdout, fake_stderr=sys.stderr)

    def tearDown(self) -> None:
        LoggerRedirector.reset_loggers(fake_stdout=sys.stdout, fake_stderr=sys.stderr)
        # unittest will revert sys.stdout and sys.stderr after this
