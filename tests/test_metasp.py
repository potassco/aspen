"""Test for metasp-related applications of aspen."""

from contextlib import redirect_stdout
from io import StringIO

from clingo import Function

from aspen.tree import AspenTree
from aspen.utils.testing import AspenTestCase

from .common import clingo_lang, encoding_dir, input_dir, output_dir

remove_amp_program = ("metasp_remove_ampersand", ())
preprocess_program = ("metasp_preprocess", ())


class TestMetaAsp(AspenTestCase):
    """Test suite for metasp-related applications of aspen."""

    maxDiff = None

    def test_metasp_remove_ampersand(self) -> None:
        """Test  removal of & from metasp"""
        self.assert_transform_isomorphic(
            language=clingo_lang,
            sources=[input_dir / "metasp_ids.lp"],
            meta_files=[encoding_dir / "metasp_remove_ampersand.lp"],
            expected_sources=[output_dir / "metasp_ids.lp"],
            initial_program=remove_amp_program,
        )

    def test_metasp_generate_type_facts(self) -> None:
        """Test generation of type facts for metasp"""
        with StringIO() as buf:
            tree = AspenTree(
                default_language=clingo_lang,
            )
            tree.parse(input_dir / "telingo_type.lp")
            tree.textio_symbols[Function("fact_file", [])] = buf
            tree.transform(
                meta_files=[encoding_dir / "metasp_generate_type_facts.lp"],
                initial_program=preprocess_program,
            )
            facts_str = buf.getvalue().strip().replace("&", "__")
        expected_facts = sorted(
            (output_dir / "generated_telingo_type_facts.lp").read_text().splitlines()
        )
        facts = sorted(facts_str.splitlines())
        self.assertEqual(facts, expected_facts)

    def test_metasp_generate_externals(self) -> None:
        """Test generation of external statements for metasp."""
        tree = AspenTree(default_language=clingo_lang)
        tree.parse(input_dir / "telingo_type.lp")
        s = tree.parse(input_dir / "metasp_telingo_gen_externals.lp")
        source = tree.sources[s]
        tree.transform(
            meta_files=[
                encoding_dir / "metasp_main.lp",
                encoding_dir / "metasp_exceptions.lp",
            ],
            initial_program=preprocess_program,
        )
        source_text_str = str(source.source_bytes, encoding=source.encoding)
        source_text = source_text_str.strip().splitlines()
        source_text = [l for l in source_text if l != "" and not l.startswith("%")]
        expected_externals = sorted(
            (output_dir / "metasp_telingo_gen_externals.lp").read_text().splitlines()
        )
        expected_externals = [
            l for l in expected_externals if l != "" and not l.startswith("%")
        ]
        source_text.sort()
        expected_externals.sort()
        self.assertListEqual(source_text, expected_externals)

    def test_metasp_generate_externals_with_condition(self) -> None:
        """Test generation of external statements for metasp with more
        advanced language constructs such as aggregates and
        conditional literals."""
        tree = AspenTree(default_language=clingo_lang)
        tree.parse(input_dir / "telingo_type.lp")
        s = tree.parse(input_dir / "metasp_telingo_with_conditions_gen_externals.lp")
        source = tree.sources[s]
        tree.transform(
            meta_files=[
                encoding_dir / "metasp_main.lp",
                encoding_dir / "metasp_exceptions.lp",
            ],
            initial_program=preprocess_program,
        )
        source_text_str = str(source.source_bytes, encoding=source.encoding).strip()
        code = [
            l for l in source_text_str.splitlines() if l != "" and not l.startswith("%")
        ]
        expected_code = sorted(
            (output_dir / "metasp_telingo_with_conditions_gen_externals.lp")
            .read_text()
            .splitlines()
        )
        expected_code = [l for l in expected_code if l != "" and not l.startswith("%")]
        code.sort()
        expected_code.sort()
        self.assertListEqual(code, expected_code)

    def test_metasp_bad_syntax(self) -> None:
        """Test that error is raised when a metasp function is not reachable."""
        self.assert_transform_raises(
            message_regex=(
                r"2:8-16: Metasp expression &next\(a\) is not reachable from"
                r" metasp symbolic atom through other metasp expressions."
            ),
            language=clingo_lang,
            sources=[input_dir / "metasp_bad_syntax.lp", input_dir / "telingo_type.lp"],
            meta_files=[
                encoding_dir / "metasp_main.lp",
                encoding_dir / "metasp_exceptions.lp",
            ],
            initial_program=preprocess_program,
        )
        self.assert_transform_raises(
            message_regex=(
                r"4:6-7: The operand of metasp expression must be a metasp "
                r"expression or symbolic atoms, found: 1."
            ),
            language=clingo_lang,
            sources=[input_dir / "metasp_bad_syntax.lp", input_dir / "telingo_type.lp"],
            meta_files=[
                encoding_dir / "metasp_main.lp",
                encoding_dir / "metasp_exceptions.lp",
            ],
            initial_program=preprocess_program,
        )

    def test_metasp_occurrence(self) -> None:
        """Test that occurrences are detected correctly."""
        with redirect_stdout(StringIO()) as buf:
            tree = AspenTree(
                default_language=clingo_lang,
            )
            print_str = (
                'aspen(print(format("head({}).", (node(S), ())))) '
                ':- extended_atom_occurrence(S,"head").'
                'aspen(print(format("directive({}).", (node(S), ())))) '
                ':- extended_atom_occurrence(S,"directive").'
            )
            tree.parse(input_dir / "metasp_occurrence_head.lp")
            tree.transform(
                meta_files=[
                    encoding_dir / "metasp_main.lp",
                ],
                meta_string=print_str,
                initial_program=preprocess_program,
            )
            print_output = buf.getvalue()
        print_facts = sorted(print_output.splitlines())
        expected_facts = sorted(
            (output_dir / "metasp_head_occurrence.lp").read_text().splitlines()
        )
        self.assertListEqual(print_facts, expected_facts)

    def test_metasp_unknown_symbol(self) -> None:
        """ "Test that exception is raised when an unknown metasp
        expression is encountered."""
        self.assert_transform_raises(
            message_regex=(r"1:0-2: Undefined metasp expression &a/0."),
            language=clingo_lang,
            sources=["&a."],
            meta_files=[
                encoding_dir / "metasp_main.lp",
                encoding_dir / "metasp_exceptions.lp",
            ],
            initial_program=preprocess_program,
        )

    def test_metasp_bad_occurrence(self) -> None:
        """ "Test that exception is raised when"""
        self.assert_transform_raises(
            message_regex=(r"8:0-7: Occurence of &bar/1 in forbidden position: head."),
            language=clingo_lang,
            sources=[input_dir / "metasp_bad_occurrence.lp"],
            meta_files=[
                encoding_dir / "metasp_main.lp",
                encoding_dir / "metasp_exceptions.lp",
            ],
            initial_program=preprocess_program,
        )

    def test_metasp_no_safety(self) -> None:
        """Test that exception is raised when no safety is given for
        an argument of a metasp expression."""
        self.assert_transform_raises(
            message_regex=(
                r"3:4-10: One or more arguments of expression definition '&bar/1' "
                r"does not have provided safety information."
            ),
            language=clingo_lang,
            sources=[input_dir / "metasp_bad_occurrence.lp"],
            meta_files=[
                encoding_dir / "metasp_main.lp",
                encoding_dir / "metasp_exceptions.lp",
            ],
            initial_program=preprocess_program,
        )

    def test_metasp_rewrite_show(self) -> None:
        """Test that rewriting of show statements works as expected."""
        self.assert_transform_isomorphic(
            language=clingo_lang,
            sources=[input_dir / "metasp_rewrite_show.lp"],
            meta_files=[encoding_dir / "metasp_rewrite_show.lp"],
            expected_sources=[output_dir / "metasp_rewrite_show.lp"],
            initial_program=preprocess_program,
        )

    def test_metasp_integration(self) -> None:
        """Integration test for metasp preprocessing."""
        self.assert_transform_isomorphic(
            language=clingo_lang,
            sources=[input_dir / "metasp_integration.lp", input_dir / "telingo_type.lp"],
            meta_files=[
                encoding_dir / "metasp_all.lp",
                encoding_dir / "metasp_remove_ampersand.lp",
            ],
            meta_string=(
                "#program metasp_preprocess. "
                'aspen(next_program("metasp_remove_ampersand", ())).'
            ),
            expected_sources=[
                output_dir / "metasp_integration.lp",
                input_dir / "telingo_type.lp",
            ],
            initial_program=preprocess_program,
        )
