"""Unit tests for module aspen.tree"""

import logging
import re
import sys
from pathlib import Path
from typing import Literal, Optional, Sequence

import tree_sitter_clingo as ts_clingo
from clingo.control import Control
from clingo.solving import Model

# pylint: disable=import-error,no-name-in-module
from clingo.symbol import Function, Number, String, Symbol, parse_term
from tree_sitter import Language

from aspen.tree import (
    AspenTree,
    SourceInput,
    TransformError,
    generic_util_path,
    id_counter,
)
from aspen.utils.logging import TestCaseWithRedirectedLogs, configure_logging

asp_dir = (Path(__file__) / ".." / "asp").resolve()
encoding_dir = asp_dir / "encodings"
input_dir = asp_dir / "inputs"
output_dir = asp_dir / "outputs"

clingo_lang = Language(ts_clingo.language())

configure_logging(sys.stderr, logging.DEBUG, sys.stderr.isatty())

aspen_tree_logger = logging.getLogger("aspen.tree")


class TestAspenTree(
    TestCaseWithRedirectedLogs
):  # pylint: disable=too-many-public-methods
    """Test AspenTree class."""

    maxDiff = None

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
        expected_return: set[Symbol] = set()
        expected_return.add(Function("isomorphic", []))
        self.assertSetEqual(query_return_facts, expected_return)

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
    ):
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

    def test_path2py(self):
        """Test conversion of symbolic path to python list"""
        tree = AspenTree(default_language=clingo_lang)
        good_path = parse_term("(1, (2, ()))")
        self.assertListEqual(
            tree._path2py(good_path), [1, 2]  # pylint: disable=protected-access
        )
        inverted_path = parse_term("(((), 2), 1)")
        re_str = r"Malformed path symbol"
        with self.assertRaisesRegex(ValueError, re_str):
            tree._path2py(inverted_path)  # pylint: disable=protected-access
        bad_element_path = parse_term("(a, (b, ()))")
        with self.assertRaisesRegex(ValueError, re_str):
            tree._path2py(bad_element_path)  # pylint: disable=protected-access

    def test_conslist2pylist(self):
        """Test conversion of symbolic cons list to python list"""
        tree = AspenTree(default_language=clingo_lang)
        good_path = parse_term("(1, (2, ()))")
        self.assertListEqual(
            tree._cons_list2py(good_path),  # pylint: disable=protected-access
            [Number(1), Number(2)],
        )
        inverted_path = parse_term("(((), 2), 1)")
        re_str = r"Expected tuple of arity 2"
        with self.assertRaisesRegex(ValueError, re_str):
            tree._cons_list2py(inverted_path)  # pylint: disable=protected-access
        const_cons_list = parse_term("(a, (b, ()))")
        const_py_list = [Function("a", []), Function("b", [])]
        self.assertListEqual(
            tree._cons_list2py(const_cons_list),  # pylint: disable=protected-access
            const_py_list,
        )

    def test_node2path_symb(self):
        """Test calculation of path symbol of a tree sitter node."""
        tree = AspenTree(default_language=clingo_lang)
        source = tree.parse("a(1).")
        node = tree.sources[source].tree.root_node.child(0).child(0).child(0).child(2)
        expected_path_symb = parse_term("(0, (0, (0, (2, ()))))")
        path_symb = tree._py_node2path_symb(node)  # pylint: disable=protected-access
        self.assertEqual(path_symb, expected_path_symb)

    def test_source_path_symb2node(self):
        """Test conversion of node id to tree sitter tree node."""
        tree = AspenTree(default_language=clingo_lang)
        source_id = parse_term("test(42)")
        tree.parse("a.", identifier=source_id)
        source_path_syb = parse_term("(test(42), (0, ( 0, ())))")
        source, node = (
            tree._source_path2py_source_node(  # pylint: disable=protected-access
                source_path_syb
            )
        )
        expected_source = tree.sources[source_id]
        self.assertEqual(source, expected_source)
        expected_node = tree.sources[source_id].tree.root_node.child(0).child(0)
        self.assertEqual(node, expected_node)
        unknown_source_node_id = parse_term("(foo(41),())")
        with self.assertRaisesRegex(ValueError, r"Unknown source symbol."):
            tree._source_path2py_source_node(  # pylint: disable=protected-access
                unknown_source_node_id
            )
        non_existent_path_node_id = parse_term("(test(42),(2, (0, ())))")
        regex_str = r"No node found in tree at path"
        with self.assertRaisesRegex(ValueError, regex_str):
            tree._source_path2py_source_node(  # pylint: disable=protected-access
                non_existent_path_node_id
            )

    def test_parse_strings(self):
        """Test parsing of input strings."""
        self.assert_parse_equals_file(
            clingo_lang, "a :- b.", output_dir / "ab_reified.txt"
        )

    def test_parse_files(self):
        """Test parsing of input files."""
        s0 = Function("s", [Number(0)])
        path_fact = Function(
            "source_path",
            [s0, String(str(input_dir / "ab.lp"))],
        )
        self.assert_parse_equals_file(
            clingo_lang,
            input_dir / "ab.lp",
            output_dir / "ab_reified.txt",
            additional_expected_facts=[path_fact],
        )

    def test_reify_error_missing_node(self):
        """Test reification of error and missing node."""
        self.assert_parse_equals_file(
            clingo_lang, "+a.", output_dir / "error_missing_reified.txt"
        )

    def test_parse_no_lang(self):
        """Test that parse method raises ValueError when no language
        is given and no default language is given."""
        tree = AspenTree()
        regex = r"No language specified, and no default language is set."
        with self.assertRaisesRegex(ValueError, regex):
            tree.parse("a.")

    def test_transform_add_vars(self):
        """Test transformation, adding variables to atoms."""
        self.assert_transform_isomorphic(
            language=clingo_lang,
            source="a :- b.",
            meta_files=[encoding_dir / "add_var.lp"],
            initial_program=("add_var_to_atoms", [String("X")]),
            expected="a(X) :- b(X).",
        )

    def test_transform_join(self):
        """Test transformation that uses a string join."""
        self.assert_transform_isomorphic(
            language=clingo_lang,
            source="a.",
            meta_files=[encoding_dir / "add_body_to_facts.lp"],
            expected="a :- b; c.",
        )

    def test_transform_join_dependency(self):
        """Test transformation that uses a string join and dependencies."""
        self.assert_transform_isomorphic(
            language=clingo_lang,
            source="a. :- b.",
            meta_files=[encoding_dir / "add_body_to_facts_depend.lp"],
            expected="a :- d; c. :- d.",
        )

    def test_transform_multiple_edits_same_node(self):
        """Test that transformation raises error when defining
        multiple edits for the same node."""
        tree = AspenTree(default_language=clingo_lang)
        tree.parse("p(1).")
        meta_str = 'aspen(edit(node(N),format(S,()))) :- source_root(_,N); S=("a.";"b.").'
        regex_str = (
            r"Multiple edits defined for following nodes;"
            r" expected one each: node\(0\)."
        )
        with self.assertRaisesRegex(ValueError, regex_str):
            tree.transform(meta_string=meta_str)

    def test_transform_spanning_ancestor(self):
        """Test that transformation succeeds when editing a node which
        has ancestors that span the same byte range."""
        self.assert_transform_isomorphic(
            language=clingo_lang,
            source=b"a.",
            meta_string='aspen(edit(node(N),"b")) :- leaf_text(N, "a").',
            expected="b.",
        )

    def test_transform_bad_edit(self):
        """Test that tranformation raises error when invalid
        replacement is used in aspen(edit(S,R))."""
        tree = AspenTree(default_language=clingo_lang)
        tree.parse("p(1).")
        meta_str = "aspen(edit(node(N),foo(1))) :- source_root(_,N)."
        regex_str = r"Symbol foo\(1\) could not be converted to string\."
        with self.assertRaisesRegex(ValueError, regex_str):
            tree.transform(meta_string=meta_str)

    def test_transform_dependencies(self):
        """Test that edits in transformation are applied in the
        correct order to satisfy implicit dependencies between edits"""
        tree = AspenTree(default_language=clingo_lang)
        tree.parse("a :- b.")
        meta_str = (
            'aspen(edit(node(N), format("{0}", (node(M), ()))))'
            ' :- leaf_text(N, "a"), leaf_text(M, "b").'
            'aspen(edit(node(M), format("{0}", (node(N), ()))))'
            ' :- leaf_text(N, "a"), leaf_text(M, "b").'
        )
        error_regex = (
            r"Transformation edits define cyclic dependencies via format strings\. "
        )
        with self.assertRaisesRegex(ValueError, error_regex):
            tree.transform(meta_string=meta_str)
        self.assert_transform_isomorphic(
            language=clingo_lang,
            source="a(b).",
            meta_files=[encoding_dir / "add_var.lp"],
            meta_string=(
                '#program add_var_to_atoms(var). aspen(edit(node(N),"c")) '
                ':- leaf_text(N,"b").'
            ),
            initial_program=("add_var_to_atoms", [String("X")]),
            expected="a(c,X).",
        )

    def test_transform_multiple_steps(self):
        """Test that transformation works as expected when multiple
        steps are defined."""
        self.assert_transform_isomorphic(
            language=clingo_lang,
            source="a :- b.",
            meta_files=[
                encoding_dir / "rename_a_to_b.lp",
                encoding_dir / "add_var.lp",
            ],
            initial_program=("rename_x_to_y", [String("a"), String("b")]),
            expected="b(X) :- b(X).",
        )

    def test_transform_bad_next_step(self):
        """Test that transformation raises error when multiple next
        programs are derived."""
        tree = AspenTree(default_language=clingo_lang)
        tree.parse("")
        meta_str = 'aspen(next_program("foo", (,))). aspen(next_program("bar", (,))).'
        error_regex = r"Multiple next_program-s defined, expected one"
        with self.assertRaisesRegex(ValueError, error_regex):
            tree.transform(meta_string=meta_str)
        meta_str = "aspen(next_program(a,b))."
        error_regex = (
            r"First argument of next_program must be a string, "
            r"second must be a tuple, found"
        )
        with self.assertRaisesRegex(ValueError, error_regex):
            tree.transform(meta_string=meta_str)

    def test_transform_multiline(self):
        """Test transform where multiline replacement occurs."""
        meta_str = """aspen(edit(node(N),format("{0}", (node(M),())))) :-
 leaf_text(N,"a"), type(M,"symbolic_atom"), child(M,L), type(L,"terms")."""
        self.assert_transform_isomorphic(
            language=clingo_lang,
            source="""
a.
p(1
,
2).""",
            meta_string=meta_str,
            expected="""
p(1
,
2).
p(1
,
2).""",
        )

    def test_transform_logs_info(self):
        """Test that tranformation logs messages as expected."""
        self.assert_transform_logs(
            log_level="INFO",
            message2num_matches={
                r"/var/home/amicsi/ghq/github.com/krr-up/aspen/tests/asp/"
                r"encodings/multiline.lp:0:0-2:0: This is a log for a node": 1,
                r" This is a log without location.": 1,
            },
            language=clingo_lang,
            source=encoding_dir / "multiline.lp",
            meta_files=[encoding_dir / "log_info.lp"],
        )

    def test_transform_logs_warn(self):
        """Test that tranformation logs messages as expected."""
        self.assert_transform_logs(
            log_level="WARNING",
            message2num_matches={
                r"/var/home/amicsi/ghq/github.com/krr-up/aspen/tests/asp/"
                r"encodings/a.lp:0:0-2: This is a log for node 'a.'.": 1,
                r" This is a log without location.": 1,
            },
            language=clingo_lang,
            source=encoding_dir / "a.lp",
            meta_files=[encoding_dir / "log_warning.lp"],
        )

    def test_transform_raises(self):
        """Test that transformation raises error as expected."""
        self.assert_transform_raises(
            message_regex=r"s\(0\):0:0-1: This is an error for node 'a'.",
            language=clingo_lang,
            source="a.",
            meta_files=[encoding_dir / "raise_error.lp"],
        )

    def test_transform_raises_no_loc(self):
        """Test that transformation raises error as expected."""
        self.assert_transform_raises(
            message_regex=r"This is an error with no location.",
            language=clingo_lang,
            source="a.",
            meta_files=[encoding_dir / "raise_error_no_loc.lp"],
        )

    # def test_transform_metasp_telingo_sugar(self):
    #     """Integration test for transformation - transform input,
    #     replacing syntactic sugar."""
    #     self.assert_transform_isomorphic(
    #         language=clingo_lang,
    #         source=(input_dir / "telingo_sugar_input.lp"),
    #         meta_files=[
    #             encoding_dir / "replace_sugar.lp",
    #             input_dir / "telingo_sugar.lp",
    #         ],
    #         expected=output_dir / "telingo_sugar_output_intermediate.lp",
    #     )


#             language=clingo_lang,
#             source=(input_dir / "telingo_sugar_input.lp"),
#             meta_files=[
#                 encoding_dir / "replace_sugar.lp",
#                 input_dir / "telingo_sugar.lp",
#             ],
#             util_encodings={
#                 "generic": ("all.lp", "show_all.lp"),
#                 "clingo": ("symbol_signature.lp", "show_all.lp"),
#             },
#             expected=output_dir / "telingo_sugar_output_intermediate.lp",
#         )
#             language=clingo_lang,
#             source=(input_dir / "telingo_sugar_input.lp"),
#             meta_files=[
#                 encoding_dir / "replace_sugar.lp",
#                 input_dir / "telingo_sugar.lp",
#             ],
#             util_encodings={
#                 "generic": ("all.lp", "show_all.lp"),
#                 "clingo": ("symbol_signature.lp", "show_all.lp"),
#             },
#             expected=output_dir / "telingo_sugar_output_intermediate.lp",
#         )
