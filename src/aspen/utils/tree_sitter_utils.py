"""
Utilities for printing.
"""

import os
import re
from typing import Optional

import tree_sitter as ts

from aspen.utils.logging import get_logger

logger = get_logger(__name__)

EditRange = tuple[int, int, int, ts.Point, ts.Point, ts.Point]

NL_PATTERN = re.compile(b"\n")

ByteRange = tuple[int, int]


def pprint_node(ts_node: ts.Node) -> None:  # nocoverage
    """Pretty-prints the string representation of a tree-sitter Tree."""

    formatted_str = ""
    sep = " " * 2
    tab_count = 0
    for token in str(ts_node).split():
        if token.endswith(":"):
            formatted_str += " " + token + " "
            continue
        l = len(token) - len(token.lstrip("("))
        r = len(token) - len(token.rstrip(")"))
        tab_count = tab_count + l - r
        formatted_str += token + os.linesep + sep * tab_count
    print(formatted_str)


def print_supertypes(lang: ts.Language) -> None:  # nocoverage
    """Prints all supertype and their respective subtypes defined in a grammar."""
    for supertype in lang.supertypes:
        supertype_name = lang.node_kind_for_id(supertype)
        if supertype_name is not None:
            print(supertype_name + ":")
        subtype_str = ""
        for subtype_id in lang.subtypes(supertype):
            subtype = lang.node_kind_for_id(subtype_id)
            if subtype is None:
                continue
            subtype_str += subtype + ", "
        print(subtype_str)


def print_changed_ranges(old_tree: ts.Tree, new_tree: ts.Tree) -> None:  # nocoverage
    """Print changed ranges after editing tree."""
    for changed_range in old_tree.changed_ranges(new_tree):
        print("Changed range:")
        print(f"  Start point {changed_range.start_point}")
        print(f"  Start byte {changed_range.start_byte}")
        print(f"  End point {changed_range.end_point}")
        print(f"  End byte {changed_range.end_byte}")


def get_path_of_node(node: ts.Node) -> list[int]:
    """Given an input node, calculate it's path in the tree,
    represented as a list of indices to traverse to reach the node
    from the root, in reverse order.

    """
    path: list[int] = []
    parent = node.parent
    while parent is not None:
        path.append(parent.children.index(node))
        node = parent
        parent = node.parent
    return path


def get_node_at_path(tree: ts.Tree, path: list[int], reverse: bool = False) -> ts.Node:
    """Given an input tree and a path, represented as a sequence of
    integer indices (in reverse order by default), retrieve the node found at the
    path in the tree.

    """
    if reverse:
        path.reverse()
    current_node = tree.root_node
    while path:
        idx = path.pop()
        child_node = current_node.child(idx)
        if child_node is None:  # nocoverage
            raise ValueError("No node found at path.")
        current_node = child_node
    return current_node


def calc_edit_range(
    start_byte: int,
    old_end_byte: int,
    start_point: ts.Point,
    old_end_point: ts.Point,
    replacement: bytes,
) -> EditRange:
    """Calculate start/end bytes and points for an arbitrary replacement"""
    new_end_byte = start_byte + len(replacement)
    num_newline = len(NL_PATTERN.findall(replacement))
    if num_newline == 0:
        new_end_col = start_point.column + len(replacement)
    else:
        new_end_col = len(replacement.split(b"\n")[-1])
    new_end_point = ts.Point(row=start_point.row + num_newline, column=new_end_col)
    edit_range = (
        start_byte,
        old_end_byte,
        new_end_byte,
        start_point,
        old_end_point,
        new_end_point,
    )
    logger.debug(
        (
            "Calculated range for edit: start byte: %s, old end byte: %s, "
            "new end byte: %s, start point: %s, old end point: %s, new end point:%s."
        ),
        *edit_range,
    )
    return edit_range


def calc_node_edit_range(edit_node: ts.Node, replacement: bytes) -> EditRange:
    """Calculate start/end bytes and points for a node edit."""
    start_byte = edit_node.start_byte
    old_end_byte = edit_node.end_byte
    start_point = edit_node.start_point
    old_end_point = edit_node.end_point
    return calc_edit_range(
        start_byte, old_end_byte, start_point, old_end_point, replacement
    )


def calc_append_range(append_node: ts.Node, to_append: bytes) -> EditRange:
    """Calculate start/end bytes and points for appending text after node."""
    start_byte = append_node.end_byte
    old_end_byte = append_node.end_byte
    start_point = append_node.end_point
    old_end_point = append_node.end_point
    return calc_edit_range(
        start_byte, old_end_byte, start_point, old_end_point, to_append
    )


def edit_tree(
    tree: ts.Tree,
    edit_range: EditRange,
    replacement: bytes,
    old_source: Optional[bytes] = None,
) -> bytes:
    """Edit a node in the tree by replacing with the given text.

    Note that this function does not re-parse the tree."""

    start_byte, old_end_byte, new_end_byte, start_point, old_end_point, new_end_point = (
        edit_range
    )
    if old_source is None:  # nocoverage
        if tree.root_node.text is None:
            raise ValueError
        old_source = tree.root_node.text
    new_source = old_source[0:start_byte] + replacement + old_source[old_end_byte:]
    tree.edit(
        start_byte, old_end_byte, new_end_byte, start_point, old_end_point, new_end_point
    )
    return new_source


def _accumulate_changed_nodes(cursor: ts.TreeCursor, acc: list[list[ts.Node]]) -> bool:
    """Accumulate nodes for which all descendants have changes."""
    node = cursor.node
    if not node.has_changes:  # type: ignore
        return False
    if cursor.goto_first_child():
        all_changed_children: list[list[ts.Node]] = []
        changed_siblings: list[ts.Node] = []
        num_changed = 0
        if _accumulate_changed_nodes(cursor, acc):
            num_changed += 1
            changed_siblings.append(cursor.node)  # type: ignore
        while cursor.goto_next_sibling():
            if _accumulate_changed_nodes(cursor, acc):
                num_changed += 1
                changed_siblings.append(cursor.node)  # type: ignore
            else:
                if len(changed_siblings) > 0:
                    all_changed_children.append(changed_siblings)
                changed_siblings = []
        if len(changed_siblings) > 0:
            all_changed_children.append(changed_siblings)
        cursor.goto_parent()
        if num_changed != cursor.node.child_count:  # type: ignore
            if len(all_changed_children) != 0:
                acc.extend(all_changed_children)
            return False
    return True


def find_changed_nodes(tree: ts.Tree) -> list[list[ts.Node]]:
    """Walk edited tree to find nodes that have been changed via edit.

    A node is considered changed if all of it's descendants have changes."""
    largest_changed_nodes: list[list[ts.Node]] = []
    cursor = tree.walk()
    source_changed = _accumulate_changed_nodes(cursor, largest_changed_nodes)
    if source_changed:
        return [[tree.root_node]]
    if len(largest_changed_nodes) == 0:
        return largest_changed_nodes
    largest_changed_nodes.sort(key=lambda l: l[0].start_byte)
    final_changed_nodes = [largest_changed_nodes[0]]
    for node_list, next_node_list in zip(
        largest_changed_nodes, largest_changed_nodes[1:]
    ):
        if node_list[-1].end_byte == next_node_list[0].start_byte:
            final_changed_nodes[-1].extend(next_node_list)
        else:
            final_changed_nodes.append(next_node_list)
    return final_changed_nodes


def get_largest_ancestor_with_same_range(node: ts.Node) -> ts.Node:
    """Get largest ancestor of input node that has the same range."""
    start, end = node.start_byte, node.end_byte
    while (
        node.parent is not None
        and node.parent.start_byte == start
        and node.parent.end_byte == end
    ):
        node = node.parent
    return node


def get_cover(ancestor_node: ts.Node, start: int, end: int) -> list[ts.Node]:
    smallest_span_node = ancestor_node.descendant_for_byte_range(start, end)
    if smallest_span_node is None:  # nocoverage
        raise ValueError(
            f"Expected to find node at byte range {start}, {end}" " ; found none."
        )
    # walk up tree, as in case there are adjacent zero length nodes,
    # we also want to catch these
    if (
        smallest_span_node.start_byte == start and smallest_span_node.end_byte == end
    ) or smallest_span_node.child_count == 0:
        smallest_span_node = get_largest_ancestor_with_same_range(smallest_span_node)
        smallest_span_node = (
            smallest_span_node
            if smallest_span_node.parent is None
            else smallest_span_node.parent
        )
    # collect children that intersect the byte range
    cover = [
        child
        for child in smallest_span_node.children
        if start < child.end_byte and child.start_byte < end
    ]
    return cover


Change = tuple[list[ts.Node], list[ts.Node]]


def get_tree_changes(old_tree: ts.Tree, new_tree: ts.Tree) -> list[Change]:
    """Given an old tree, and a new tree that has just been re-parsed
    from the old one, calculate the changes that need to be made to
    old_tree (and data structures derived from old_tree) to get a tree
    isomorphic to new_tree.

    The necessary changes are returned a list of pairs. Each pair in
    turn contains two lists of sibling nodes, the first one from the
    old tree and the second one from the new tree. By replacing the
    old siblings with the new siblings in the old tree for each such
    pair in the returned list, the old tree becomes isomorphic to the
    new tree.

    """
    covers: list[tuple[list[ts.Node], list[ts.Node]]] = []
    changed_nodes = find_changed_nodes(old_tree)
    changed_nodes.sort(key=lambda n: n[-1].end_byte - n[0].start_byte, reverse=True)
    for siblings in changed_nodes:
        start, end = siblings[0].start_byte, siblings[-1].end_byte
        new_cover = get_cover(new_tree.root_node, start, end)
        if len(new_cover) > 0:
            new_start, new_end = new_cover[0].start_byte, new_cover[-1].end_byte
            old_cover = get_cover(old_tree.root_node, new_start, new_end)
        # for deletions and some weird cases when the ranges shrink,
        # we just keep the list of sibling changed nodes as well
        if (
            len(new_cover) == 0
            or start < new_cover[0].start_byte
            or new_cover[-1].end_byte < end
        ):
            old_cover = siblings
        covers.append((old_cover, new_cover))
    if len(covers) == 0:
        return covers
    covers.sort(
        key=lambda t: t[0][0].start_byte if len(t[1]) == 0 else t[1][0].start_byte
    )
    final: list[tuple[list[ts.Node], list[ts.Node]]] = [covers[0]]
    for (old_cover, new_cover), (next_old_cover, next_new_cover) in zip(
        covers, covers[1:]
    ):
        # case: covers are adjacent
        if new_cover[-1].next_sibling == next_new_cover[0]:
            assert old_cover[-1].next_sibling == next_old_cover[0]
            final[-1][0].extend(next_old_cover)
            final[-1][1].extend(next_new_cover)
        # case: covers overlap
        elif new_cover[-1].end_byte > next_new_cover[0].start_byte:
            assert old_cover[-1].end_byte > next_old_cover[0].start_byte
            new_idx = new_cover.index(next_new_cover[0])
            old_idx = old_cover.index(next_old_cover[0])
            final_old = final[-1][0][:old_idx] + next_old_cover
            final_new = final[-1][1][:new_idx] + next_new_cover
            final[-1] = (final_old, final_new)
        else:
            final.append((next_old_cover, next_new_cover))
    return final
