"""Common definitions for tests."""

import logging
import sys
from pathlib import Path

import tree_sitter_clingo as ts_clingo
from tree_sitter import Language

from aspen.utils.log import configure_logging

asp_dir = (Path(__file__) / ".." / "asp").resolve()
encoding_dir = asp_dir / "encodings"
input_dir = asp_dir / "inputs"
output_dir = asp_dir / "outputs"

clingo_lang = Language(ts_clingo.language())

configure_logging(sys.stderr, logging.DEBUG, sys.stderr.isatty())
