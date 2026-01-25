"""Unit tests for module aspen.tree"""

from contextlib import redirect_stdout
from io import StringIO

# pylint: disable=import-error,no-name-in-module
from clingo.symbol import Function, Number, String, parse_term

from aspen.tree import AspenTree
from aspen.utils.testing import AspenTestCase
from aspen.utils.tree_sitter_utils import get_node_at_path

from .common import clingo_lang, encoding_dir, input_dir, output_dir


class TestAspenTree(AspenTestCase):  # pylint: disable=too-many-public-methods
    """Test suite for AspenTree class."""

    def test_path2py(self) -> None:
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

    def test_conslist2pylist(self) -> None:
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

    def test_node2path_symb(self) -> None:
        """Test calculation of path symbol of a tree sitter node."""
        tree = AspenTree(default_language=clingo_lang)
        source = tree.parse("a(1).")
        node = get_node_at_path(tree.sources[source].tree, [0, 0, 0, 2], reverse=True)
        expected_path_symb = parse_term("(0, (0, (0, (2, ()))))")
        path_symb = tree._py_node2path_symb(node)  # pylint: disable=protected-access
        self.assertEqual(path_symb, expected_path_symb)

    def test_source_path_symb2node(self) -> None:
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
        expected_node = get_node_at_path(
            tree.sources[source_id].tree, [0, 0], reverse=True
        )
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

    def test_parse_strings(self) -> None:
        """Test parsing of input strings."""
        self.assert_parse_equals_file(
            clingo_lang, "a :- b.", output_dir / "ab_reified.txt"
        )

    def test_parse_files(self) -> None:
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

    def test_reify_error_missing_node(self) -> None:
        """Test reification of error and missing node."""
        self.assert_parse_equals_file(
            clingo_lang, "+a.", output_dir / "error_missing_reified.txt"
        )

    def test_parse_no_lang(self) -> None:
        """Test that parse method raises ValueError when no language
        is given and no default language is given."""
        tree = AspenTree()
        regex = r"No language specified, and no default language is set."
        with self.assertRaisesRegex(ValueError, regex):
            tree.parse("a.")

    def test_transform_add_vars(self) -> None:
        """Test transformation, adding variables to atoms."""
        self.assert_transform_isomorphic(
            language=clingo_lang,
            sources=["a :- b."],
            meta_files=[encoding_dir / "add_var.lp"],
            initial_program=("add_var_to_atoms", [String("X")]),
            expected_sources=["a(X) :- b(X)."],
        )

    def test_transform_join(self) -> None:
        """Test transformation that uses a string join."""
        self.assert_transform_isomorphic(
            language=clingo_lang,
            sources=["a."],
            meta_files=[encoding_dir / "add_body_to_facts.lp"],
            expected_sources=["a :- b; c."],
        )

    def test_transform_join_dependency(self) -> None:
        """Test transformation that uses a string join and dependencies."""
        self.assert_transform_isomorphic(
            language=clingo_lang,
            sources=["a. :- b."],
            meta_files=[encoding_dir / "add_body_to_facts_depend.lp"],
            expected_sources=["a :- d; c. :- d."],
        )

    def test_transform_multiple_edits_same_node(self) -> None:
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

    def test_transform_spanning_ancestor(self) -> None:
        """Test that transformation succeeds when editing a node which
        has ancestors that span the same byte range."""
        self.assert_transform_isomorphic(
            language=clingo_lang,
            sources=[b"a."],
            meta_string='aspen(edit(node(N),"b")) :- leaf_text(N, "a").',
            expected_sources=["b."],
        )

    def test_transform_bad_edit(self) -> None:
        """Test that tranformation raises error when invalid
        replacement is used in aspen(edit(S,R))."""
        tree = AspenTree(default_language=clingo_lang)
        tree.parse("p(1).")
        meta_str = "aspen(edit(node(N),foo(1))) :- source_root(_,N)."
        regex_str = r"Symbol foo\(1\) could not be converted to string\."
        with self.assertRaisesRegex(ValueError, regex_str):
            tree.transform(meta_string=meta_str)

    def test_transform_dependencies(self) -> None:
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
            sources=["a(b)."],
            meta_files=[encoding_dir / "add_var.lp"],
            meta_string=(
                '#program add_var_to_atoms(var). aspen(edit(node(N),"c")) '
                ':- leaf_text(N,"b").'
            ),
            initial_program=("add_var_to_atoms", [String("X")]),
            expected_sources=["a(c,X)."],
        )

    def test_transform_multiple_steps(self) -> None:
        """Test that transformation works as expected when multiple
        steps are defined."""
        self.assert_transform_isomorphic(
            language=clingo_lang,
            sources=["a :- b."],
            meta_files=[
                encoding_dir / "rename_a_to_b.lp",
                encoding_dir / "add_var.lp",
            ],
            initial_program=("rename_x_to_y", [String("a"), String("b")]),
            expected_sources=["b(X) :- b(X)."],
        )

    def test_transform_bad_next_step(self) -> None:
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

    def test_transform_multiline(self) -> None:
        """Test transform where multiline replacement occurs."""
        meta_str = """aspen(edit(node(N),format("{0}", (node(M),())))) :-
 leaf_text(N,"a"), type(M,"symbolic_atom"), child(M,L), type(L,"terms")."""
        self.assert_transform_isomorphic(
            language=clingo_lang,
            sources=[
                """
a.
p(1
,
2)."""
            ],
            meta_string=meta_str,
            expected_sources=[
                """
p(1
,
2).
p(1
,
2)."""
            ],
        )

    def test_transform_logs_info(self) -> None:
        """Test that tranformation logs messages as expected."""
        source_path = encoding_dir / "multiline.lp"
        loc_log_str = str(source_path).replace("\\", "\\\\")
        loc_log_str += r":1:0-2:0: This is a log for a node"
        self.assert_transform_logs(
            log_level="INFO",
            message2num_matches={
                loc_log_str: 1,
                r" This is a log without location.": 1,
            },
            language=clingo_lang,
            source=source_path,
            meta_files=[encoding_dir / "log_info.lp"],
        )

    def test_transform_logs_warn(self) -> None:
        """Test that tranformation logs messages as expected."""
        source_path = encoding_dir / "a.lp"
        loc_log_str = str(source_path).replace("\\", "\\\\")
        loc_log_str += r":1:0-2: This is a log for node 'a.'."
        self.assert_transform_logs(
            log_level="WARNING",
            message2num_matches={
                loc_log_str: 1,
                r" This is a log without location.": 1,
            },
            language=clingo_lang,
            source=source_path,
            meta_files=[encoding_dir / "log_warning.lp"],
        )

    def test_transform_raises(self) -> None:
        """Test that transformation raises error as expected."""
        self.assert_transform_raises(
            message_regex=r"s\(0\):1:0-1: This is an error for node 'a'.",
            language=clingo_lang,
            sources=["a."],
            meta_files=[encoding_dir / "raise_error.lp"],
        )

    def test_transform_raises_no_loc(self) -> None:
        """Test that transformation raises error as expected."""
        self.assert_transform_raises(
            message_regex=r"This is an error with no location.",
            language=clingo_lang,
            sources=["a."],
            meta_files=[encoding_dir / "raise_error_no_loc.lp"],
        )

    def test_transform_print(self) -> None:
        """Test that aspen(print(String)) symbols derived during
        transformation prints String to stdout."""
        tree = AspenTree(default_language=clingo_lang)
        with redirect_stdout(StringIO()) as buf:
            tree.transform(meta_string='aspen(print(format("Hello {}", ("World", ())))).')
            printed_text = buf.getvalue()
            self.assertEqual(printed_text, "Hello World\n")

    def test_transform_print_to_io(self) -> None:
        """Test that aspen(print(String)) symbols derived during
        transformation prints String to stdout."""
        tree = AspenTree(default_language=clingo_lang)
        with StringIO() as buf:
            tree.textio_symbols[Function("io", [])] = buf
            tree.transform(
                meta_string='aspen(print(format("Hello {}", ("World", ())), io)).'
            )
            printed_text = buf.getvalue()
            self.assertEqual(printed_text, "Hello World\n")

    def test_transform_print_to_source(self) -> None:
        """Test that if Source is a valid source symbol, then
        aspen(print(String, Source)) symbols derived during
        transformation append String to the given Source."""
        self.assert_transform_isomorphic(
            language=clingo_lang,
            sources=[""],
            meta_string='aspen(print("a.", s(0))).',
            expected_sources=["a.\n"],
        )
