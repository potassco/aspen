"""Utilities for testing."""

import logging
import re
from pathlib import Path
from typing import Literal, Optional, Sequence

from clingo.control import Control
from clingo.solving import Model
from clingo.symbol import Function, Number, Symbol, parse_term
from tree_sitter import Language

from aspen.tree import (
    AspenTree,
    SourceInput,
    TransformError,
    generic_util_path,
    id_counter,
)
from aspen.utils.logging import TestCaseWithRedirectedLogs

aspen_tree_logger = logging.getLogger("aspen.tree")


class AspenTestCase(TestCaseWithRedirectedLogs):
    """Base class for building test cases related to AspenTree class."""

    def assert_parse_equals_file(
        self,
        language: Language,
        source: SourceInput,
        path: Path,
        additional_expected_facts: Optional[list[Symbol]] = None,
    ) -> None:
        """Assert that parsing string of the given language results in
        symbols contained in the given file."""
        tree = AspenTree(default_language=language)
        tree.parse(source)
        # we have to parse and then turn back into string due to
        # clingo6 bug: https://github.com/potassco/clingo/issues/579
        with path.open() as f:
            expected_symbols = [str(parse_term(s)) for s in f.readlines()]
        if additional_expected_facts is not None:
            expected_symbols.extend([str(f) for f in additional_expected_facts])
        expected_symbols.sort()
        symbols = [str(parse_term(str(s))) for s in tree.facts]
        symbols.sort()
        self.assertListEqual(symbols, expected_symbols)

    def assert_transform_isomorphic(
        self,
        *,
        language: Language,
        source: SourceInput,
        expected: str | Path,
        meta_files: Optional[Sequence[Path]] = None,
        meta_string: Optional[str] = None,
        initial_program: tuple[str, Sequence[Symbol]] = ("base", ()),
        control_options: Optional[Sequence[str]] = None,
    ) -> None:
        """Assert that transformation results in expected string, and
        check that reified representation is isomorphic."""
        tree = AspenTree(default_language=language)
        s = tree.parse(source)
        parsed_source = tree.sources[s]
        tree.transform(
            meta_files=meta_files,
            meta_string=meta_string,
            initial_program=initial_program,
            control_options=control_options,
        )
        transformed_source_str = str(
            tree.sources[s].source_bytes, encoding=parsed_source.encoding
        )
        if isinstance(expected, Path):
            expected = expected.read_text()
        self.assertEqual(transformed_source_str, expected)
        # don't clutter logs generated during testing
        lvl = aspen_tree_logger.level
        aspen_tree_logger.setLevel(logging.ERROR)
        tree2 = AspenTree(
            default_language=language, id_generator=id_counter(start=-1, step=-1)
        )
        tree2.parse(expected)
        control = Control()
        iso_query_path = generic_util_path / "queries" / "isomorphic.lp"
        control.load(str(iso_query_path))
        query_symb = Function("isomorphic", [Number(0), Number(-1)])
        with control.backend() as backend:
            facts = tree.facts + tree2.facts
            for f in facts:
                f_atom = backend.add_atom(f)
                backend.add_rule([f_atom])
            query_atom = backend.add_atom(
                Function("aspen", [Function("query", [query_symb])])
            )
            backend.add_rule([query_atom])
        control.ground()
        query_return_facts: set[Symbol] = set()

        def on_iso_model(model: Model) -> Literal[False]:
            for symb in model.symbols(shown=True):
                if (
                    symb.match("aspen", 1)
                    and symb.arguments[0].match("return", 2)
                    and symb.arguments[0].arguments[0] == query_symb
                ):
                    query_return_facts.add(symb.arguments[0].arguments[1])
            return False

        control.solve(on_model=on_iso_model)
        aspen_tree_logger.setLevel(lvl)
        query_return_facts_str = {str(s) for s in query_return_facts}
        expected_return_facts_str: set[str] = set()
        expected_return_facts_str.add(str(Function("isomorphic", [])))
        self.assertSetEqual(query_return_facts_str, expected_return_facts_str)

    def assert_transform_logs(
        self,
        *,
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"],
        message2num_matches: dict[str, int],
        language: Language,
        source: SourceInput,
        meta_files: Optional[Sequence[Path]] = None,
        meta_string: Optional[str] = None,
        initial_program: tuple[str, Sequence[Symbol]] = ("base", ()),
        control_options: Optional[Sequence[str]] = None,
    ) -> None:
        """Assert that transformation logs messages, or raises error."""
        tree = AspenTree(default_language=language)
        tree.parse(source)
        with self.assertLogs("aspen.tree", level=log_level) as cm:
            tree.transform(
                meta_files=meta_files,
                meta_string=meta_string,
                initial_program=initial_program,
                control_options=control_options,
            )
            logs = "\n".join(cm.output)
            for message, expected_num in message2num_matches.items():
                assert_msg = (
                    f"Expected {expected_num} "
                    "matches for log message pattern "
                    f"'{message}' in {logs}, found "
                )
                reo = re.compile(message)
                num_log_matches = len(reo.findall(logs))
                self.assertEqual(
                    num_log_matches, expected_num, msg=assert_msg + str(num_log_matches)
                )

    def assert_transform_raises(
        self,
        *,
        message_regex: str,
        language: Language,
        source: SourceInput,
        meta_files: Optional[Sequence[Path]] = None,
        meta_string: Optional[str] = None,
        initial_program: tuple[str, Sequence[Symbol]] = ("base", ()),
        control_options: Optional[Sequence[str]] = None,
    ) -> None:
        """Assert that transformation raises error."""
        tree = AspenTree(default_language=language)
        tree.parse(source)
        with self.assertRaisesRegex(TransformError, message_regex):
            tree.transform(
                meta_files=meta_files,
                meta_string=meta_string,
                initial_program=initial_program,
                control_options=control_options,
            )
