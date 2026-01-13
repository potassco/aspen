"""
Test cases for main application functionality.
"""

import logging
from io import StringIO
from unittest import TestCase

from aspen.utils import logging
from aspen.utils.logging import configure_logging, get_logger
from aspen.utils.parser import get_parser


class TestMain(TestCase):
    """
    Test cases for main application functionality.
    """

    def test_logger(self) -> None:
        """
        Test the logger.
        """
        sio = StringIO()
        configure_logging(sio, logging.INFO, True)
        log = get_logger("main")
        with self.assertLogs("main", level=logging.INFO):
            log.info("test123")

    def test_parser(self) -> None:
        """
        Test the parser.
        """
        parser = get_parser()
        ret = parser.parse_args(["--log", "info"])
        self.assertEqual(ret.log, logging.INFO)
