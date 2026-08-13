"""Common definitions for tests."""

import logging
import sys
from pathlib import Path

import tree_sitter_aspcore2 as ts_aspcore2
import tree_sitter_clingo as ts_clingo
import tree_sitter_metasp as ts_metasp
from tree_sitter import Language

from aspen.utils.log import configure_logging

asp_dir = (Path(__file__) / ".." / "asp").resolve()
encoding_dir = asp_dir / "encodings"
input_dir = asp_dir / "inputs"
output_dir = asp_dir / "outputs"

clingo_lang = Language(ts_clingo.language())
metasp_lang = Language(ts_metasp.language())
aspcore2_lang = Language(ts_aspcore2.language())

configure_logging(sys.stderr, logging.INFO, sys.stderr.isatty())
