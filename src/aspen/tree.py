"""Module defining the AspenTree class."""

from collections import defaultdict
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import List, Literal, Optional, Sequence

import tree_sitter as ts

# pylint: disable=import-error,no-name-in-module
from clingo.control import Control
from clingo.core import Library
from clingo.solve import Model
from clingo.symbol import Function, Number, String, Symbol, SymbolType, Tuple_

import aspen
from aspen.utils.logging import get_clingo_logger, get_logger, get_ts_logger
from aspen.utils.tree_sitter_utils import ts_edit_tree

logger = get_logger(__name__)
clingo_logger = get_clingo_logger(logger)
ts_logger = get_ts_logger(logger)

Id = int
Bytes = tuple[int, int]
StringEncoding = Literal["utf8", "utf16"]
StringName = str

SourceInput = bytes | str | Path


@dataclass
class Source:
    """Source processed by an AspenTree."""

    id: Symbol
    source_bytes: bytes
    path: Optional[Path]
    encoding: StringEncoding
    parser: ts.Parser
    tree: ts.Tree


SourceNode = tuple[Source, ts.Node]
FormatEdit = tuple[SourceNode, tuple[str, list[SourceNode]]]


class AspenTree:
    """A tree that wraps a tree-sitter tree and it's representation as
    a set ASP facts.

    The class further defined various transformation operations on
    both representations, and syncs the two in case of
    changes. Transformations to the tree-sitter tree can be made via
    the wrapped tree's API, while trasformations on the ASP fact
    reperesentation can be defined via a logic program, which deriving
    special transformation predicates in it's answer sets with the
    fact representation of the tree as input.

    """

    def __init__(
        self,
        default_language: Optional[ts.Language] = None,
        default_encoding: StringEncoding = "utf8",
        clingo_lib: Optional[Library] = None,
    ):
        self.sources: dict[Symbol, Source] = {}
        self.default_language = default_language
        self.default_encoding = default_encoding
        self.lib = clingo_lib if clingo_lib is not None else Library(logger=clingo_logger)
        self.facts: List[Symbol] = []
        self.next_transform_program: Optional[tuple[str, Sequence[Symbol]]] = None
        self.id_counter = 0

    def parse(
        self,
        /,
        source: SourceInput,
        *,
        language: Optional[ts.Language] = None,
        encoding: Optional[StringEncoding] = None,
        included_ranges: Optional[Sequence[ts.Range]] = None,
        identifier: Optional[Symbol] = None,
    ) -> Symbol:
        """Parse input strings, and generate fact representation."""
        language = language if language is not None else self.default_language
        if language is None:
            raise ValueError("No language specified, and no default language is set.")
        encoding = encoding if encoding is not None else self.default_encoding
        parser = ts.Parser(language, included_ranges=included_ranges, logger=ts_logger)
        path: Optional[Path] = None
        if isinstance(source, bytes):
            source_bytes = source
        elif isinstance(source, str):
            source_bytes = bytes(source, encoding)
        elif isinstance(source, Path):
            path = source.resolve().resolve()
            if not path.is_file():  # nocoverage
                raise IOError(f"File {path} not found.")
            source_bytes = path.read_bytes()
        else:  # nocoverage
            raise TypeError(
                f"Argument 'source' must be of type {SourceInput}, got: {type(source)}."
            )
        tree = parser.parse(source_bytes, encoding=encoding)
        if identifier is None:
            identifier = Function(self.lib, "s", [Number(self.lib, self.id_counter)])
            self.id_counter += 1
        processed_source = Source(identifier, source_bytes, path, encoding, parser, tree)
        self.sources[identifier] = processed_source
        self._reify_ts_tree(tree, processed_source)
        return identifier

    def _reify_node_attrs(
        self, node: ts.Node, node_id: Symbol, encoding: StringEncoding
    ) -> list[Symbol]:
        """Reify a tree-sitter node and it's attributes into a (set of) fact(s)."""
        facts: list[Symbol] = []
        facts.append(Function(self.lib, "node", [node_id]))
        if node.is_named:
            facts.append(
                Function(self.lib, "type", [node_id, String(self.lib, node.type)])
            )
        if node.is_error:
            facts.append(Function(self.lib, "error", [node_id]))
        if node.is_missing:
            facts.append(Function(self.lib, "missing", [node_id]))
        if node.is_extra:  # nocoverage
            facts.append(Function(self.lib, "extra", [node_id]))
        if node.child_count == 0 and node.text is not None:
            facts.append(
                Function(
                    self.lib,
                    "leaf_text",
                    [node_id, String(self.lib, node.text.decode(encoding))],
                )
            )
        return facts

    def _reify_ts_subtree(
        self,
        node: ts.Node,
        subtree_path: Symbol,
        source: Source,
    ) -> list[Symbol]:
        """Reify tree-sitter subtree with input root node into a set of facts."""
        stack: list[tuple[ts.Node, Symbol]] = [(node, subtree_path)]
        facts: list[Symbol] = []
        while len(stack) > 0:
            parent, parent_path = stack.pop()
            parent_id = Tuple_(self.lib, [source.id, parent_path])
            facts.extend(self._reify_node_attrs(parent, parent_id, source.encoding))
            for idx, child in enumerate(parent.children):
                child_path = Tuple_(self.lib, [parent_path, Number(self.lib, idx)])
                child_id = Tuple_(self.lib, [source.id, child_path])
                field_name = parent.field_name_for_child(idx)
                if field_name is not None:
                    facts.append(
                        Function(
                            self.lib, "field", [child_id, String(self.lib, field_name)]
                        )
                    )
                stack.append((child, child_path))
        return facts

    def _reify_ts_tree(self, tree: ts.Tree, source: Source) -> None:
        """Reify tree-sitter tree into a set of facts."""
        root_node = tree.root_node
        root_path = Tuple_(self.lib, [])
        if source.parser.language is None:  # nocoverage
            raise ValueError(f"Parser of source should not be None: {source}.")
        lang_name = source.parser.language.name
        if lang_name is None:  # nocoverage
            raise ValueError(f"Language of parser of source cannot be None: {source}.")
        lang_fact = Function(
            self.lib,
            "language",
            [
                source.id,
                String(self.lib, source.parser.language.name),  # type: ignore
            ],
        )
        self.facts.append(lang_fact)
        self.facts.extend(self._reify_ts_subtree(root_node, root_path, source))

    def path2py(self, path_symb: Symbol) -> list[int]:
        """Convert path expression from symbolic to list form."""
        l: list[int] = []
        nil = Tuple_(self.lib, [])
        while path_symb != nil:
            if not path_symb.match(2) or path_symb.arguments[1].type != SymbolType.Number:
                raise ValueError(f"Malformed path symbol: {path_symb}.")
            l.append(path_symb.arguments[1].number)
            path_symb = path_symb.arguments[0]
        return l

    def conslist2list(self, l: Symbol) -> list[Symbol]:
        """Convert symbolic cons list consisting of nested tuples into
        a python list of symbols."""
        l_py: list[Symbol] = []
        nil = Tuple_(self.lib, [])
        while l != nil:
            l_py.append(l.arguments[0])
            l = l.arguments[1]
        return l_py

    def node_id2ts(self, node_id: Symbol) -> SourceNode:
        """Retrieve tree-sitter node from node identifier symbol."""
        source_symb, path_symb = node_id.arguments
        path_list = self.path2py(path_symb)
        try:
            source = self.sources[source_symb]
        except KeyError as exc:
            raise ValueError(f"Unknown source symbol: {source_symb}.") from exc
        tree = source.tree
        node = tree.root_node
        while True:
            try:
                idx = path_list.pop()
            except IndexError:
                break
            try:
                tmp_node = node.child(idx)
                if tmp_node is None:  # nocoverage
                    raise ValueError(f"No node found at path: {path_symb}.")
            except IndexError as exc:
                raise ValueError(f"No node found in tree at path: {path_symb}.") from exc
            node = tmp_node
        return source, node

    def _get_descendant_ids(
        self, node: ts.Node, node_path_symb: Symbol, source: Source
    ) -> list[Symbol]:
        """Return list of identifiers for all descendents of node."""
        source_symb = source.id
        desc_ids: list[Symbol] = []
        stack: list[tuple[ts.Node, Symbol]] = [(node, node_path_symb)]
        while len(stack) > 0:
            parent, parent_path = stack.pop()
            parent_id = Tuple_(self.lib, [source_symb, parent_path])
            desc_ids.append(parent_id)
            for idx, child in enumerate(parent.children):
                child_path = Tuple_(self.lib, [parent_path, Number(self.lib, idx)])
                child_id = Tuple_(self.lib, [source_symb, child_path])
                desc_ids.append(child_id)
                stack.append((child, child_path))
        return desc_ids

    def _node2path_symb(self, node: ts.Node) -> Symbol:
        """Given a tree-sitter node, calculate the corresponding path symbol."""
        path: list[int] = []
        parent = node.parent
        while parent is not None:
            path.append(parent.children.index(node))
            node = parent
            parent = node.parent
        path_symb = Tuple_(self.lib, [])
        while True:
            try:
                idx = path.pop()
            except IndexError:
                break
            path_symb = Tuple_(self.lib, [path_symb, Number(self.lib, idx)])
        return path_symb

    def _re_reify_changed_subtrees(
        self, subtrees: defaultdict[Symbol, list[ts.Node]]
    ) -> None:
        """Re-reify subtrees who's syntactic structure changed due to
        edit, and delete outdated facts from before edit."""
        # drop nodes to be re-reified that are descendants of other nodes to be re-reified
        delete_ids: set[Symbol] = set()
        new_facts: list[Symbol] = []
        contained_nodes: set[ts.Node] = set()
        for source_symb, nodes in subtrees.items():
            node_ranges = [(n, n.start_byte, n.end_byte) for n in nodes]
            for idx, (n1, s1, e1) in enumerate(node_ranges):
                if n1 in contained_nodes:
                    continue
                for n2, s2, e2 in node_ranges[idx + 1 :]:
                    if s1 <= s2 and e2 <= e1:
                        contained_nodes.add(n2)
                    elif s2 <= s1 and e1 <= e2:  # nocoverage
                        break
                else:
                    path = self._node2path_symb(n1)
                    source = self.sources[source_symb]
                    new_facts.extend(self._reify_ts_subtree(n1, path, source))
                    delete_ids.update(self._get_descendant_ids(n1, path, source))
                    if n1.parent is not None:
                        parent = n1.parent
                        node_idx = parent.children.index(n1)
                        field_name = parent.field_name_for_child(node_idx)
                        if field_name is not None:
                            node_id = Tuple_(self.lib, [source_symb, path])
                            field_fact = Function(
                                self.lib, "field", [node_id, String(self.lib, field_name)]
                            )
                            new_facts.append(field_fact)

        # print("Ids to be deleted:")
        # print([str(s) for s in delete_ids])
        self.facts = [f for f in self.facts if f.arguments[0] not in delete_ids]
        # print("Reified facts to be added by transform:")
        # print([str(s) for s in new_facts])
        self.facts.extend(new_facts)

    def _reparse_sources(self, edited_sources: dict[Symbol, set[ts.Range]]) -> None:
        """Re-parse sources that have been edited, and update fact
        representation based on the changed ranges of sources.

        A set of additional ranges who's fact representation should be
        updated can be provided as the value for the given source in
        the edited_sources dict argument.
        """

        re_reify_subtree_roots: defaultdict[Symbol, list[ts.Node]] = defaultdict(list)
        for source_symb, changed_ranges in edited_sources.items():
            try:
                source = self.sources[source_symb]
            except KeyError as exc:  # nocoverage
                raise ValueError(f"Unknown source symbol: {source_symb}.") from exc
            old_tree = source.tree
            # print(source.source_bytes)
            new_tree = source.parser.parse(
                source.source_bytes, old_tree, encoding=source.encoding
            )
            changed_ranges.update(old_tree.changed_ranges(new_tree))
            # print(changed_ranges)
            source.tree = new_tree
            # print(f"Changed ranges in source {source}:.")
            # print(changed_ranges)
            for changed_range in changed_ranges:
                start, end = changed_range.start_byte, changed_range.end_byte
                node = new_tree.root_node.descendant_for_byte_range(start, end)
                if node is None:  # nocoverage
                    raise RuntimeError("Code should be unreachable.")
                # Not sure if this is necessary, but we walk up to the
                # greatest node that spans the changed range.
                parent = node.parent
                while (
                    parent is not None
                    and parent.start_byte == start
                    and parent.end_byte == end
                ):
                    node = parent
                    parent = node.parent

                re_reify_subtree_roots[source.id].append(node)
        self._re_reify_changed_subtrees(re_reify_subtree_roots)

    def _topological_sort_edits(
        self, edit_symbols: Sequence[Symbol], deps: dict[Symbol, List[Symbol]]
    ) -> Sequence[Symbol]:
        """Toplogically sort edits, so they are processed in correct
        order. We edit (children of) replacement nodes before the
        target node of any given derived edit fact.

        """
        for s in edit_symbols:
            if s not in deps.keys():
                deps[s] = []
        tsorter = TopologicalSorter(deps)
        try:
            edit_symbols = list(tsorter.static_order())
        except CycleError as exc:
            cycle = [str(symb) for symb in exc.args[1]]
            msg = (
                "Transformation edits define cyclic dependencies via format strings. "
                f"One such cycle: {cycle}."
            )
            raise ValueError(msg) from exc
        return edit_symbols

    def _edit_sources_from_symbs(
        self, edit_symbols: Sequence[Symbol]
    ) -> dict[Symbol, set[ts.Range]]:
        """Apply edits to tree according to operations given in
        edit_symbols, and return dictionary who's keys are source
        symbols that have been edited, and the values are a set of
        additional changed ranges that will not (necessarily) be
        detected by tree-sitter when re-parsing after the edit.

        The edited symbols must first be sorted via
        _topological_sort_edits.

        """
        seen = set()
        dupes = {
            f.arguments[0]
            for f in edit_symbols
            if f.arguments[0] in seen or seen.add(f.arguments[0])  # type: ignore
        }
        if len(dupes) > 0:
            dupes_str = ",".join(str(d) for d in dupes)
            raise ValueError(
                "Multiple edits defined for following nodes; "
                f"expected one each: {dupes_str}."
            )
        # print(edit_symbols)
        edited_sources: dict[Symbol, set[ts.Range]] = {}
        for symb in edit_symbols:
            replacement = symb.arguments[1]
            target_source, target_node = self.node_id2ts(symb.arguments[0])
            if replacement.match("format", 2):
                format_string = replacement.arguments[0].string
                replacement_tup = replacement.arguments[1]
                insert_texts: list[str] = []
                inserts = self.conslist2list(replacement_tup)
                for insert in inserts:
                    # insert is a node
                    if insert.match(2):
                        insert_source, insert_node = self.node_id2ts(insert)
                        start, end = insert_node.start_byte, insert_node.end_byte
                        insert_text = insert_source.source_bytes[start:end].decode(
                            insert_source.encoding
                        )
                    # insert is not a node
                    else:
                        insert_text = str(insert)
                        if insert_text.startswith('"') and insert_text.endswith('"'):
                            insert_text = insert_text[1:-1]
                    insert_texts.append(insert_text)
                replacement_text = format_string.format(*insert_texts)

            elif replacement.type == SymbolType.String:
                replacement_text = replacement.string
            else:
                raise ValueError(
                    f"Symbol {replacement} does not match any "
                    "allowed replacement symbols."
                )
            replacement_bytes = bytes(replacement_text, target_source.encoding)
            target_source.source_bytes = ts_edit_tree(
                target_source.tree,
                target_node,
                replacement_bytes,
                target_source.source_bytes,
            )
            if target_source.id not in edited_sources:
                edited_sources[target_source.id] = set()
            # changes to leaf nodes do not show up in tree changed_ranges method,
            # so we manually mark these nodes as changed instead
            if target_node.child_count == 0:
                edited_sources[target_source.id].add(target_node.range)
        # print(edited_sources)
        return edited_sources

    def _on_transform_model(self, model: Model) -> Literal[False]:
        """Model callback for transformation. Returns False as we only
        expect one model."""
        if logger.level == 10:  # nocoverage
            logger.debug(
                "Stable model obtained by applying transformation meta-encoding:"
            )
            for s in model.symbols(shown=True):
                logger.debug(str(s))

        edit_symbols: list[Symbol] = []
        next_transform_symbols: list[Symbol] = []
        deps: defaultdict[Symbol, list[Symbol]] = defaultdict(list)
        for symb in model.symbols(shown=True):
            if symb.match("aspen", 1):
                arg = symb.arguments[0]
                if arg.match("edit", 2):
                    logger.info(
                        "Edit derived by transformation meta-encoding: '%s'",
                        str(symb),
                    )
                    edit_symbols.append(arg)
                elif arg.match("comes_before", 2):
                    deps[arg.arguments[1]].append(arg.arguments[0])
                elif arg.match("next_program", 2):
                    next_transform_symbols.append(arg)
        if len(next_transform_symbols) > 1:
            raise ValueError(
                (
                    "Multiple next_program-s defined, expected one: "
                    f"{next_transform_symbols}."
                )
            )
        if len(next_transform_symbols) > 0:
            next_symb = next_transform_symbols[0]
            if (
                next_symb.arguments[0].type != SymbolType.String
                or next_symb.arguments[1].type != SymbolType.Tuple
            ):
                raise ValueError(
                    "First argument of next_transform_program must be a string, "
                    f"second must be a tuple, found: {next_symb}."
                )
            self.next_transform_program = (
                next_symb.arguments[0].string,
                next_symb.arguments[1].arguments,
            )
        sorted_edit_symbols = self._topological_sort_edits(edit_symbols, deps)
        edited_sources = self._edit_sources_from_symbs(sorted_edit_symbols)
        self._reparse_sources(edited_sources)
        return False

    def transform(
        self,
        *,
        meta_files: Optional[Sequence[Path]] = None,
        meta_string: Optional[str] = None,
        initial_program: tuple[str, Sequence[Symbol]] = ("base", ()),
        util_encodings: Sequence[str] = ("all.lp",),
        control_options: Optional[Sequence[str]] = None,
    ) -> None:
        """Transform fact base via a meta-encoding."""
        options = control_options if control_options is not None else []
        if meta_files is not None:  # nocoverage
            for f in meta_files:
                if not f.is_file():
                    raise IOError(f"File {f} not found.")
        encoding_files = [str(f) for f in meta_files] if meta_files is not None else []
        aspen_init_path = Path(aspen.__file__)
        encoding_path = (aspen_init_path / ".." / "asp").resolve()
        base_encodings = [encoding_path / "defined.lp", encoding_path / "edit.lp"]
        encoding_files.extend([str(p) for p in base_encodings])
        encoding_files.extend(
            [str(encoding_path / "utils" / name) for name in util_encodings]
        )
        base_program = ("base", ())
        self.next_transform_program = initial_program
        while self.next_transform_program is not None:
            # print(self.next_transform_program)
            control = Control(self.lib, options=options)
            control.parse_files(encoding_files)
            if meta_string is not None:
                control.parse_string(meta_string)
            with control.backend as backend:
                for fact in self.facts:
                    atom = backend.atom(fact)
                    backend.rule([atom])
            parts = (
                None
                if self.next_transform_program == base_program
                else [base_program, self.next_transform_program]
            )
            control.ground(parts=parts)
            self.next_transform_program = None
            with control.solve(on_model=self._on_transform_model) as handle:
                handle.get()
