"""
Utilities related to tree-sitter.
"""

import os
import re
from typing import Optional

import tree_sitter as ts

from aspen.utils.log import get_logger

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


def calc_node_append_range(append_node: ts.Node, to_append: bytes) -> EditRange:
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


Change = tuple[list[ts.Node], list[ts.Node]]


def _find_reused_children(
    old_children: list[ts.Node], new_children: list[ts.Node]
) -> list[tuple[int, int]]:
    """Find pairs of indices (old_idx, new_idx) of children that are
    literally the same node, reused verbatim across the edit.

    Tree-sitter reuses a node's identity (its ``id``) across an
    incremental edit and re-parse whenever that node's subtree is
    unaffected by the edit: "if a new tree is created based on an older
    tree, and a node from the old tree is reused in the process, then
    that node will have the same id in both trees" (tree-sitter docs).
    Two children sharing an id are therefore guaranteed structurally
    identical, and since edits never reorder siblings, matches occur in
    the same relative order in both sequences.

    """
    new_index_by_id = {child.id: idx for idx, child in enumerate(new_children)}
    matches: list[tuple[int, int]] = []
    last_new_idx = -1
    for old_idx, child in enumerate(old_children):
        new_idx = new_index_by_id.get(child.id)
        if new_idx is not None and new_idx > last_new_idx:
            matches.append((old_idx, new_idx))
            last_new_idx = new_idx
    return matches


def _narrow_change(
    old_siblings: list[ts.Node], new_siblings: list[ts.Node]
) -> list[Change]:
    """A change consisting of a single old and a single new node of the
    same grammar type can be described equally well, and more precisely,
    as a change to that node's children. Descend in that case; otherwise
    the change cannot be narrowed further."""
    if (
        len(old_siblings) == 1
        and len(new_siblings) == 1
        and old_siblings[0].child_count > 0
        and new_siblings[0].child_count > 0
        and old_siblings[0].type == new_siblings[0].type
    ):
        return _diff_children(old_siblings[0], new_siblings[0])
    return [(old_siblings, new_siblings)]


def _diff_children(
    old_node: ts.Node, new_node: ts.Node, top_level: bool = False
) -> list[Change]:
    """Diff the children of two corresponding nodes (nodes occupying the
    same structural position in the old and new tree, e.g. both roots),
    returning the changes needed to turn old_node's children into
    new_node's children.

    A change with an empty list of old siblings describes a pure
    insertion with no old node to anchor it to; callers of
    get_tree_changes can only make sense of such a change if it is a
    plain append at the very end of the whole source (``top_level``),
    since that is the only position that can be identified without an
    anchor. Everywhere else, a pure insertion is folded together with
    one adjacent reused node (which is then reported as changed too,
    even though only its neighbourhood changed) so that every change has
    a real node to anchor it to.

    """
    old_children = old_node.children
    new_children = new_node.children
    reused = _find_reused_children(old_children, new_children)
    changes: list[Change] = []
    old_pos = new_pos = 0
    last_fold_was_used = False
    for old_idx, new_idx in reused:
        old_gap = old_children[old_pos:old_idx]
        new_gap = new_children[new_pos:new_idx]
        last_fold_was_used = False
        if not old_gap and new_gap:
            # pure insertion right before this reused node: fold the
            # node in as an anchor, in place of excluding it as unchanged
            # note: in practice, I could not produce an instance where this
            # happens; it seems that a pure insertion always causes the
            # previous node before to get a new id, eveng though it is unchanged.
            # We might remove this code in the future, but leave it here for
            # now to be defensive
            old_gap = [old_children[old_idx]]
            new_gap = new_gap + [new_children[new_idx]]
            last_fold_was_used = True
        if old_gap or new_gap:
            changes.extend(_narrow_change(old_gap, new_gap))
        old_pos, new_pos = old_idx + 1, new_idx + 1
    # gap after the last reused node
    old_gap = old_children[old_pos:]
    new_gap = new_children[new_pos:]
    if not old_gap and new_gap and not top_level:
        # trailing pure insertion with no root to fall back on: anchor
        # it to the last reused node, which is the only one left that
        # could possibly serve, extending the change it was already
        # folded into, or folding it in now if it wasn't needed there.
        if last_fold_was_used:
            last_old, last_new = changes[-1]
            changes[-1] = (last_old, last_new + new_gap)
            old_gap, new_gap = [], []
        elif reused:
            last_reused_old, last_reused_new = reused[-1]
            old_gap = [old_children[last_reused_old]]
            new_gap = [new_children[last_reused_new]] + new_gap
        # else: old_node had no children of its own to anchor to; only
        # reachable when old_node is the (possibly empty) tree root
    if old_gap or new_gap:
        changes.extend(_narrow_change(old_gap, new_gap))
    return changes


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
    # note: when re-parsing, the root node id is always new, so I think the branch
    # below is not needed.
    # if old_tree.root_node.id == new_tree.root_node.id:
    #     return []
    if old_tree.root_node.type != new_tree.root_node.type:
        # The edit left text that tree-sitter cannot parse into the
        # expected top-level grammar rule at all (e.g. an unterminated
        # rule at the very end of the source), so the *root* node itself
        # would need replacing - not just some of its children. A
        # replacement of the root can't be expressed as a change to a
        # list of siblings (the root has no parent to splice it into),
        # so this can't be diffed incrementally; the caller needs to
        # re-reify the whole source instead.
        raise ValueError(
            "Old and new tree root nodes have different types "
            f"({old_tree.root_node.type!r} vs {new_tree.root_node.type!r}); "
            "the new tree cannot be expressed as sibling-level changes to "
            "the old tree and must be reified from scratch."
        )
    return _diff_children(old_tree.root_node, new_tree.root_node, top_level=True)
