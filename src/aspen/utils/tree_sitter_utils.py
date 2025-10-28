"""
Utilities for printing.
"""

import os
import re
from typing import Optional

import tree_sitter as ts

EditRange = tuple[int, int, int, ts.Point, ts.Point, ts.Point]


def ts_pprint_node(ts_node: ts.Node) -> None:
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


def ts_print_supertypes(lang: ts.Language) -> None:
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


def ts_print_changed_ranges(old_tree: ts.Tree, new_tree: ts.Tree) -> None:
    """Print changed ranges after editing tree."""
    for changed_range in old_tree.changed_ranges(new_tree):
        print("Changed range:")
        print(f"  Start point {changed_range.start_point}")
        print(f"  Start byte {changed_range.start_byte}")
        print(f"  End point {changed_range.end_point}")
        print(f"  End byte {changed_range.end_byte}")


def ts_calc_edit_range(edit_node: ts.Node, replacement: bytes) -> EditRange:
    """Calculate start/end bytes and points for an edit."""
    start_byte = edit_node.start_byte
    old_end_byte = edit_node.end_byte
    new_end_byte = start_byte + len(replacement)
    start_point = edit_node.start_point
    old_end_point = edit_node.end_point
    pattern = re.compile(b"\n")
    num_newline = len(pattern.findall(replacement))
    if num_newline == 0:
        new_end_col = start_point.column + len(replacement)
    else:
        new_end_col = len(replacement.split(b"\n")[-1])
    new_end_point = ts.Point(row=start_point.row + num_newline, column=new_end_col)
    return start_byte, old_end_byte, new_end_byte, start_point, old_end_point, new_end_point


def ts_edit_tree(tree: ts.Tree, edit_node: ts.Node, replacement: bytes, old_source: Optional[bytes] = None) -> bytes:
    """Edit a node in the tree by replacing with the given text.

    Node that this function does not re-parse the tree."""

    start_byte, old_end_byte, new_end_byte, start_point, old_end_point, new_end_point = ts_calc_edit_range(
        edit_node, replacement
    )
    if old_source is None:
        if tree.root_node.text is None:
            raise ValueError
        old_source = tree.root_node.text
    new_source = old_source[0:start_byte] + replacement + old_source[old_end_byte:]
    tree.edit(start_byte, old_end_byte, new_end_byte, start_point, old_end_point, new_end_point)
    return new_source


def ts_sync_node_after_tree_edit(sync_node: ts.Node, tree: ts.Tree, edit_node: ts.Node, replacement: bytes) -> None:
    """Call edit on a node after a tree edit."""
    sync_node.edit(*ts_calc_edit_range(tree, edit_node, replacement))
