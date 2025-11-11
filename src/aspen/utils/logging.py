"""
Setup project wide loggers.

This is a thin wrapper around Python's logging module. It supports colored
logging.
"""

import logging
from functools import partial
from typing import Callable, TextIO

import tree_sitter as ts

# pylint: disable=import-error,no-name-in-module
from clingo.core import MessageCode

NOTSET = logging.NOTSET
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

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
