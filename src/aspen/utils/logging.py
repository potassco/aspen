"""
Setup project wide loggers.

This is a thin wrapper around Python's logging module. It supports colored
logging.
"""

import logging
from typing import TextIO, Callable
from functools import partial

from clingo.core import MessageType
import tree_sitter as ts

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
            return f"{COLORS[color]}%(levelname)s:{COLORS['GREY']}  - %(message)s{COLORS['NORMAL']}"
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


def log_clingo_message(message_code: MessageType, message: str, logger: logging.Logger) -> None:  # nocoverage
    """Log clingo message at the appropriate level"""
    clingo_fstring = "clingo: %s"
    if message_code is MessageType.Trace:
        logger.debug(clingo_fstring, message)
    elif message_code is MessageType.Debug:
        logger.debug(clingo_fstring, message)
    elif message_code is MessageType.Info:
        logger.info(clingo_fstring, message)
    elif message_code is MessageType.Warn:
        logger.warn(clingo_fstring, message)
    elif message_code is MessageType.Error:
        logger.error(clingo_fstring, message)
    elif message_code is MessageType.AtomUndefined:
        logger.info(clingo_fstring, message)
    elif message_code is MessageType.FileIncluded:
        logger.warn(clingo_fstring, message)
    elif message_code is MessageType.GlobalVariable:
        logger.info(clingo_fstring, message)
    elif message_code is MessageType.OperationUndefined:
        logger.info(clingo_fstring, message)


def get_clingo_logger(
    logger: logging.Logger,
) -> Callable[[MessageType, str], None]:
    """Return a callback function to be used when initializing a
    clingo.core.Library object to log to input logger.

    """
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
