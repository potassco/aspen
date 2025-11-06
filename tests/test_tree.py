"""Unit tests for module aspen.tree"""

from pathlib import Path
from typing import Optional, Sequence
from unittest import TestCase

import tree_sitter_clingo as ts_clingo

# pylint: disable=import-error,no-name-in-module
from clingo.core import Library
from clingo.symbol import String, Symbol, parse_term
from tree_sitter import Language

import aspen
from aspen.tree import AspenTree, SourceInput

aspen_path = Path(aspen.__file__)
asp_dir = aspen_path.parent.parent.parent / "tests" / "files"
lib = Library()
clingo_lang = Language(ts_clingo.language())


class TestAspenTree(TestCase):
    """Test AspenTree class."""

    maxDiff = None

    def assert_parse_equals_file(
        self, language: Language, source: SourceInput, path: Path
    ):
        """Assert that parsing string of the given language results in
        symbols contained in the given file."""
        tree = AspenTree(default_language=language)
        tree.parse(source)
        # we have to parse and then turn back into string due to
        # clingo6 bug: https://github.com/potassco/clingo/issues/579
        with path.open() as f:
            expected_symbols = [str(parse_term(lib, s)) for s in f.readlines()]
        expected_symbols.sort()
        symbols = [str(parse_term(lib, str(s))) for s in tree.facts]
        symbols.sort()
        self.assertListEqual(symbols, expected_symbols)

    def assert_transform_equals(
        self,
        *,
        language: Language,
        source: SourceInput,
        expected_str: str,
        meta_files: Optional[Sequence[Path]] = None,
        meta_string: Optional[str] = None,
        initial_program: tuple[str, Sequence[Symbol]] = ("base", ()),
        util_encodings: Sequence[str] = ("all.lp",),
        control_options: Optional[Sequence[str]] = None,
    ):
        """Assert that transformation results in expected string, and
        check that reified representation matches."""
        tree = AspenTree(default_language=language)
        s = tree.parse(source)
        parsed_source = tree.sources[s]
        tree.transform(
            meta_files=meta_files,
            meta_string=meta_string,
            initial_program=initial_program,
            util_encodings=util_encodings,
            control_options=control_options,
        )
        transformed_source_str = str(
            tree.sources[s].source_bytes, encoding=parsed_source.encoding
        )
        self.assertEqual(transformed_source_str, expected_str)
        tree2 = AspenTree(default_language=language)
        tree2.parse(expected_str)
        expected_symbols = [str(f) for f in tree2.facts]
        symbols = [str(f) for f in tree.facts]
        expected_symbols.sort()
        symbols.sort()
        self.assertListEqual(symbols, expected_symbols)

    def test_parse_strings(self):
        """Test parsing of input strings."""
        self.assert_parse_equals_file(clingo_lang, "a :- b.", asp_dir / "ab_reified.txt")

    def test_parse_files(self):
        """Test parsing of input files."""
        self.assert_parse_equals_file(
            clingo_lang, asp_dir / "ab.lp", asp_dir / "ab_reified.txt"
        )

    def test_reify_missing_node(self):
        """Test reification of missing node."""
        self.assert_parse_equals_file(clingo_lang, "=2.", asp_dir / "missing_reified.txt")

    def test_reify_error_node(self):
        """Test reification of error node."""
        self.assert_parse_equals_file(clingo_lang, "+a.", asp_dir / "error_reified.txt")

    def test_path2py(self):
        """Test conversion of symbolic path to python list"""
        tree = AspenTree(default_language=clingo_lang)
        good_path = parse_term(tree.lib, "(((), 2), 1)")
        self.assertListEqual(tree.path2py(good_path), [1, 2])
        inverted_path = parse_term(tree.lib, "(1, (2, ()))")
        re_str = r"Malformed path symbol"
        with self.assertRaisesRegex(ValueError, re_str):
            tree.path2py(inverted_path)
        bad_element_path = parse_term(tree.lib, "(((), b), a)")
        with self.assertRaisesRegex(ValueError, re_str):
            tree.path2py(bad_element_path)

    def test_node_id2ts(self):
        """Test conversion of node id to tree sitter tree node."""
        tree = AspenTree(default_language=clingo_lang)
        source_id = parse_term(tree.lib, "test(42)")
        tree.parse("a.", identifier=source_id)
        node_id = parse_term(tree.lib, "(test(42),(((),0),0))")
        source, node = tree.node_id2ts(node_id)
        expected_source = tree.sources[source_id]
        self.assertEqual(source, expected_source)
        expected_node = tree.sources[source_id].tree.root_node.child(0).child(0)
        self.assertEqual(node, expected_node)
        unknown_source_node_id = parse_term(tree.lib, "(foo(41),())")
        with self.assertRaisesRegex(ValueError, r"Unknown source symbol."):
            tree.node_id2ts(unknown_source_node_id)
        non_existent_path_node_id = parse_term(tree.lib, "(test(42),(((),0),2))")
        regex_str = r"No node found in tree at path"
        with self.assertRaisesRegex(ValueError, regex_str):
            tree.node_id2ts(non_existent_path_node_id)

    def test_parse_no_lang(self):
        """Test that parse method raises ValueError when no language
        is given and no default language is given."""
        tree = AspenTree()
        regex = r"No language specified, and no default language is set."
        with self.assertRaisesRegex(ValueError, regex):
            tree.parse("a.")

    def test_transform_add_vars_(self):
        """Test transformation, adding variables to atoms."""
        self.assert_transform_equals(
            language=clingo_lang,
            source="a :- b.",
            meta_files=[asp_dir / "add_var.lp"],
            initial_program=("add_var_to_atoms", [String(lib, "X")]),
            expected_str="a(X) :- b(X).",
        )

    def test_transform_multiple_edits_same_node(self):
        """Test that transformation raises error when defining
        multiple edits for the same node."""
        tree = AspenTree(default_language=clingo_lang)
        tree.parse("p(1).")
        meta_str = (
            'aspen(edit(N,format(S,()))) :- N=(s(0),((),0)); node(N); S=("a.";"b.").'
        )
        regex_str = (
            r"Multiple edits defined for following nodes;"
            r" expected one each: \(s\(0\),\(\(\),0\)\)."
        )
        with self.assertRaisesRegex(ValueError, regex_str):
            tree.transform(meta_string=meta_str)

    def test_transform_spanning_ancestor(self):
        """Test that transformation succeeds when editing a node which
        has ancestors that span the same byte range."""
        self.assert_transform_equals(
            language=clingo_lang,
            source=b"a.",
            meta_string='aspen(edit((s(0),(((((),0),0),0),0)),"b")).',
            expected_str="b.",
        )

    def test_transform_bad_edit(self):
        """Test that tranformation raises error when invalid
        replacement is used in aspen(edit(S,R))."""
        tree = AspenTree(default_language=clingo_lang)
        tree.parse("p(1).")
        meta_str = "aspen(edit(N,foo(1))) :- N=(s(0),((),0)); node(N)."
        regex_str = r"Symbol foo\(1\) does not match any allowed replacement symbols\."
        with self.assertRaisesRegex(ValueError, regex_str):
            tree.transform(meta_string=meta_str)

    def test_transform_dependencies(self):
        """Test that edits in transformation are applied in the
        correct order to satisfy implicit dependencies between edits"""
        tree = AspenTree(default_language=clingo_lang)
        tree.parse("a :- b.")
        meta_str = (
            'aspen(edit((s(0),(((),0),0)), format("{0}", ((s(0),(((),0),2)), ())))).'
            'aspen(edit((s(0),(((),0),2)), format("{0}", ((s(0),(((),0),0)), ())))).'
        )
        error_regex = (
            r"Transformation edits define cyclic dependencies via format strings\. "
        )
        with self.assertRaisesRegex(ValueError, error_regex):
            tree.transform(meta_string=meta_str)
        self.assert_transform_equals(
            language=clingo_lang,
            source="a(b).",
            meta_files=[asp_dir / "add_var.lp"],
            meta_string=(
                '#program add_var_to_atoms(var). aspen(edit(N,"c")) :- leaf_text(N,"b").'
            ),
            initial_program=("add_var_to_atoms", [String(lib, "X")]),
            expected_str="a(c,X).",
        )

    def test_transform_multiple_steps(self):
        """Test that transformation works as expected when multiple
        steps are defined."""
        self.assert_transform_equals(
            language=clingo_lang,
            source="a :- b.",
            meta_files=[asp_dir / "rename_a_to_b.lp", asp_dir / "add_var.lp"],
            initial_program=("rename_x_to_y", [String(lib, "a"), String(lib, "b")]),
            expected_str="b(X) :- b(X).",
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
            r"First argument of next_transform_program must be a string, "
            r"second must be a tuple, found"
        )
        with self.assertRaisesRegex(ValueError, error_regex):
            tree.transform(meta_string=meta_str)

    def test_transform_multiline(self):
        """Test transform where multiline replacement occurs."""
        meta_str = """aspen(edit(N,format("{0}", (M,())))) :-
 leaf_text(N,"a"), type(M,"symbolic_atom"), child(M,L), type(L,"terms")."""
        self.assert_transform_equals(
            language=clingo_lang,
            source="""
a.
p(1
,
2).""",
            meta_string=meta_str,
            expected_str="""
p(1
,
2).
p(1
,
2).""",
        )
