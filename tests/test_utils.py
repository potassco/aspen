"""Unit tests for utilities."""

from typing import Optional

from tree_sitter import Language, Node, Parser, Point, Tree

from aspen.utils.log import TestCaseWithRedirectedLogs
from aspen.utils.tree_sitter_utils import (
    Change,
    EditRange,
    calc_node_append_range,
    calc_node_edit_range,
    edit_tree,
    get_node_at_path,
    get_tree_changes,
)

from .common import clingo_lang

Path = list[int]
Length = int
# a seqence of sibling nodes can be described via the path (encoded as
# a list of intiger indices) at which the first sibling can be found,
# and the length of the sequence of siblings
SiblingsDescriptor = tuple[Path, Length]
ExpectedChangeDescriptor = tuple[
    Optional[SiblingsDescriptor], Optional[SiblingsDescriptor]
]


class TestTreeSitterUtils(TestCaseWithRedirectedLogs):
    """Test tree-sitter related utilities"""

    maxDiff = None

    def get_siblings_from_descriptor(
        self,
        descriptor: Optional[SiblingsDescriptor],
        tree: Tree,
    ) -> list[Node]:
        """Given a tree and an input descriptor, describing a list of
        sibling nodes via a path to the first sibling and a length,
        return the described siblings from the tree.

        """
        if descriptor is None:
            return []
        path, length = descriptor
        sibling = get_node_at_path(tree, path, reverse=True)
        parent = sibling.parent
        assert parent is not None
        idx = parent.children.index(sibling)
        return parent.children[idx : idx + length]

    def get_changes_from_descriptors(
        self,
        old_tree: Tree,
        new_tree: Tree,
        expected_change_descriptors: list[ExpectedChangeDescriptor],
    ) -> list[Change]:
        """Given an old tree, a new tree (generated via re-parsing),
        retrieve the list of changes desribed by the input list of
        change descriptors."""
        expected_changes = []
        for exp_change_desc in expected_change_descriptors:
            old_siblings = self.get_siblings_from_descriptor(exp_change_desc[0], old_tree)
            new_siblings = self.get_siblings_from_descriptor(exp_change_desc[1], new_tree)
            expected_changes.append((old_siblings, new_siblings))
        return expected_changes

    def assert_edits_changes_equal(
        self,
        *,
        language: Language = clingo_lang,
        input_bytes: bytes,
        edits: list[tuple[EditRange, bytes] | tuple[Path, str, bytes]],
        expected_final_bytes: bytes,
        expected_change_descriptors: list[ExpectedChangeDescriptor],
    ) -> None:
        """Assert that tree edit results in expected bytes, and expected changes."""
        parser = Parser(language)
        old_tree = parser.parse(input_bytes)
        current_bytes = input_bytes
        for edit in edits:
            # case: node edit/append
            if isinstance(edit, tuple) and len(edit) == 3:
                path, mode, replace_bytes = edit
                target_node = get_node_at_path(old_tree, path, reverse=True)
                if mode == "edit":
                    replace_range = calc_node_edit_range(target_node, replace_bytes)
                elif mode == "append":
                    replace_range = calc_node_append_range(target_node, replace_bytes)
                else:
                    raise ValueError
                current_bytes = edit_tree(
                    old_tree, replace_range, replace_bytes, old_source=current_bytes
                )
            # case: range edit
            elif isinstance(edit, tuple) and len(edit) == 2:
                edit_range, replace_bytes = edit
                current_bytes = edit_tree(
                    old_tree, edit_range, replace_bytes, old_source=current_bytes
                )
        self.assertEqual(current_bytes, expected_final_bytes)
        new_tree = parser.parse(current_bytes, old_tree)
        changes = get_tree_changes(old_tree, new_tree)
        expected_changes = self.get_changes_from_descriptors(
            old_tree, new_tree, expected_change_descriptors
        )
        self.assertListEqual(changes, expected_changes)

    def test_changes_no_edit(self) -> None:
        """Test that there are no reported changes when no edit is performed."""
        self.assert_edits_changes_equal(
            input_bytes=b"a.",
            edits=[],
            expected_final_bytes=b"a.",
            expected_change_descriptors=[],
        )

    def test_changes_range_edit1(self) -> None:
        """Test that editing based on range results in expected changes."""
        self.assert_edits_changes_equal(
            input_bytes=b"a(1). z :- c, x.",
            edits=[
                (
                    (
                        11,
                        14,
                        11,
                        Point(row=0, column=11),
                        Point(row=0, column=14),
                        Point(row=0, column=11),
                    ),
                    b"",
                ),
            ],
            expected_final_bytes=b"a(1). z :- x.",
            expected_change_descriptors=[(([1, 1], 3), ([1, 1], 3))],
        )

    def test_changes_range_edit2(self) -> None:
        """Test that editing based on range results in expected changes."""
        self.assert_edits_changes_equal(
            input_bytes=b"a(1;4). z :- c, x.",
            edits=[
                (
                    (
                        2,
                        9,
                        16,
                        Point(row=0, column=2),
                        Point(row=0, column=9),
                        Point(row=0, column=16),
                    ),
                    b"3;1,2;4). z; g",
                )
            ],
            expected_final_bytes=b"a(3;1,2;4). z; g :- c, x.",
            expected_change_descriptors=[(([0], 2), ([0], 2))],
        )

    def test_changes_range_edit3(self) -> None:
        """Test that editing based on range results in expected changes."""
        self.assert_edits_changes_equal(
            input_bytes=b"a(1). z :- c, x.",
            edits=[
                (
                    (
                        8,
                        15,
                        8,
                        Point(row=0, column=8),
                        Point(row=0, column=15),
                        Point(row=0, column=8),
                    ),
                    b"",
                )
            ],
            expected_final_bytes=b"a(1). z .",
            expected_change_descriptors=[(([1, 0], 4), ([1, 0], 2))],
        )

    def test_changes_multiple_range_edit_overlap(self) -> None:
        """Test that editing based on range results in expected changes."""
        self.assert_edits_changes_equal(
            input_bytes=b"a :- b. z :- y, x. w :- s. unchanged.",
            edits=[
                (
                    (
                        5,
                        9,
                        9,
                        Point(row=0, column=5),
                        Point(row=0, column=9),
                        Point(row=0, column=9),
                    ),
                    b"n. n",
                ),
                (
                    (
                        16,
                        20,
                        20,
                        Point(row=0, column=16),
                        Point(row=0, column=20),
                        Point(row=0, column=20),
                    ),
                    b"m. m",
                ),
            ],
            expected_final_bytes=b"a :- n. n :- y, m. m :- s. unchanged.",
            expected_change_descriptors=[(([0], 3), ([0], 3))],
        )

    def test_changes_node_append1(self) -> None:
        """Test that appending after node results in expected changes."""
        self.assert_edits_changes_equal(
            input_bytes=b"a :- b. z :- x.",
            edits=[([0], "append", b" asd :- wasd. dsa :- asd.")],
            expected_final_bytes=b"a :- b. asd :- wasd. dsa :- asd. z :- x.",
            expected_change_descriptors=[(([0], 1), ([0], 3))],
        )

    def test_changes_node_append2(self) -> None:
        """Test that appending after node results in expected changes."""
        self.assert_edits_changes_equal(
            input_bytes=b"a(1). z :- x",
            edits=[([0, 0, 0, 2], "append", b"1, 2; 3, 4")],
            expected_final_bytes=b"a(11, 2; 3, 4). z :- x",
            expected_change_descriptors=[(([0], 2), ([0], 2))],
        )

    def test_changes_node_append3(self) -> None:
        """Test that appending after node results in expected changes."""
        self.assert_edits_changes_equal(
            input_bytes=b"a(1). z :- x",
            edits=[([0, 0, 0], "append", b"; b. p")],
            expected_final_bytes=b"a(1); b. p. z :- x",
            expected_change_descriptors=[(([0], 2), ([0], 3))],
        )

    def test_changes_node_append4(self) -> None:
        """Test that appending after node results in expected changes."""
        self.assert_edits_changes_equal(
            input_bytes=b"a(1). z :- x",
            edits=[([0, 0, 0, 2], "append", b",2")],
            expected_final_bytes=b"a(1,2). z :- x",
            expected_change_descriptors=[(([0], 2), ([0], 2))],
        )

    def test_changes_node_edit1(self) -> None:
        """Test that editing node results in expected changes."""
        self.assert_edits_changes_equal(
            input_bytes=b"a :- b. z :- x. a.",
            edits=[([1], "edit", b"z :- x. as :- sd.")],
            expected_final_bytes=b"a :- b. z :- x. as :- sd. a.",
            expected_change_descriptors=[(([0], 2), ([0], 3))],
        )

    def test_changes_node_edit2(self) -> None:
        """Test that editing node results in expected changes."""
        self.assert_edits_changes_equal(
            input_bytes=b"aast :- bdsrr. z :- x. a. c.",
            edits=[([1], "edit", b"")],
            expected_final_bytes=b"aast :- bdsrr.  a. c.",
            expected_change_descriptors=[(([0], 3), ([0], 2))],
        )

    def test_changes_node_edit3(self) -> None:
        """Test that editing node results in expected changes."""
        self.assert_edits_changes_equal(
            input_bytes=b"aast :- bdsrr. z :- x. a. b.",
            edits=[([1], "edit", b""), ([3], "edit", b"")],
            expected_final_bytes=b"aast :- bdsrr.  a. ",
            expected_change_descriptors=[(([0], 4), ([0], 2))],
        )

    def test_changes_node_edit4(self) -> None:
        """Test that editing node results in expected changes."""
        self.assert_edits_changes_equal(
            input_bytes=b"a.",
            edits=[([0], "edit", b"")],
            expected_final_bytes=b"",
            expected_change_descriptors=[(([0], 1), None)],
        )

    def test_changes_total_parse_failure_raises(self) -> None:
        """Test that an edit which leaves the source unparsable (so that
        the new tree's root itself is an ERROR node, not a source_file)
        raises, since such a change cannot be expressed as a change to a
        list of siblings anywhere in the old tree."""
        parser = Parser(clingo_lang)
        input_bytes = b"a :- b."
        old_tree = parser.parse(input_bytes)
        edit_range = (
            3,
            7,
            3,
            Point(row=0, column=3),
            Point(row=0, column=7),
            Point(row=0, column=3),
        )
        current_bytes = edit_tree(old_tree, edit_range, b"", old_source=input_bytes)
        self.assertEqual(current_bytes, b"a :")
        new_tree = parser.parse(current_bytes, old_tree)
        with self.assertRaises(ValueError):
            get_tree_changes(old_tree, new_tree)

    def test_changes_node_edit_multiple_overlap(self) -> None:
        """Test that the detection of changes with multiple edits
        correctly merges overlapping changes into a single change."""
        self.assert_edits_changes_equal(
            input_bytes=b"a :- b. z :- x",
            edits=[
                ([0], "append", b" d."),
                ([1], "edit", b"very_long_name :- longer_name."),
            ],
            expected_final_bytes=b"a :- b. d. very_long_name :- longer_name.",
            expected_change_descriptors=[(([0], 2), ([0], 3))],
        )
