"""Module defining the AspenTree class."""

import logging
from collections import defaultdict
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Generator, List, Literal, Optional, Sequence

import tree_sitter as ts

# pylint: disable=import-error,no-name-in-module
from clingo.control import Control
from clingo.solving import Model
from clingo.symbol import Function, Number, String, Symbol, SymbolType, Tuple_

import aspen
from aspen.utils.logging import get_clingo_logger, get_logger, get_ts_logger
from aspen.utils.tree_sitter_utils import (
    Change,
    calc_node_edit_range,
    edit_tree,
    get_path_of_node,
    get_tree_changes,
)

logger = get_logger(__name__)
clingo_logger = get_clingo_logger(logger)
ts_logger = get_ts_logger(logger)

Id = int
Bytes = tuple[int, int]
StringEncoding = Literal["utf8", "utf16"]
StringName = str
RelatedNodes = tuple[
    Optional[Symbol],
    Optional[Symbol],
    Optional[Symbol],
    Optional[Symbol],
    Optional[Symbol],
]

SourceInput = bytes | str | Path

log_lvl_strs = {"debug", "info", "warning"}
log_lvl_str2int = {"debug": 10, "info": 20, "warning": 30}

base_program = ("base", ())

aspen_init_path = Path(aspen.__file__)
encoding_path = (aspen_init_path / ".." / "asp").resolve()
generic_util_path = encoding_path / "utils" / "generic"


class TransformError(Exception):
    """Exception raised when a transformation meta-encoding derives an
    error."""


def id_counter(start: int = 0, step: int = 1) -> Generator[Symbol]:
    """Simple infinite generator that yields clingo Numbers, starting
    from start, with increment given by step."""
    n = start
    while True:
        yield Number(n)
        n += step


@dataclass
class Source:
    """Source processed by an AspenTree."""

    id: Symbol
    source_bytes: bytes
    path: Optional[Path]
    encoding: StringEncoding
    parser: ts.Parser
    tree: ts.Tree


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
        id_generator: Optional[Generator[Symbol]] = None,
    ):
        self.sources: dict[Symbol, Source] = {}
        self.default_language = default_language
        self.default_encoding = default_encoding
        self.facts: List[Symbol] = []
        self.next_transform_program: Optional[tuple[str, Sequence[Symbol]]] = None
        self._id_generator = id_counter() if id_generator is None else id_generator
        self._node_id2source_node: dict[Symbol, tuple[Source, ts.Node]] = {}

    def _path2py(self, path_symb: Symbol) -> list[int]:
        """Convert path expression from symbolic to list form."""
        l: list[int] = []
        nil = Tuple_([])
        while path_symb != nil:
            if (
                not path_symb.match("", 2)
                or path_symb.arguments[0].type != SymbolType.Number
            ):
                raise ValueError(f"Malformed path symbol: {path_symb}.")
            l.append(path_symb.arguments[0].number)
            path_symb = path_symb.arguments[1]
        return l

    def _cons_list2py(self, l: Symbol) -> list[Symbol]:
        """Convert symbolic cons list consisting of nested tuples into
        a python list of symbols."""
        l_py: list[Symbol] = []
        nil = Tuple_([])
        while l != nil:
            if not l.match("", 2):
                raise ValueError(f"Expected tuple of arity 2, found: {l}.")  # nocoverage
            l_py.append(l.arguments[0])
            l = l.arguments[1]
        return l_py

    def _py_node2path_symb(self, node: ts.Node) -> Symbol:
        """Given a tree-sitter node, calculate the corresponding path symbol."""
        path = get_path_of_node(node)
        path_symb = Tuple_([])
        for idx in path:
            path_symb = Tuple_([Number(idx), path_symb])
        return path_symb

    def _source_path2py_source_node(
        self, source_path_symb: Symbol
    ) -> tuple[Source, ts.Node]:
        """Retrieve tree-sitter node from node identifier symbol."""
        source_symb, path_symb = (
            source_path_symb.arguments[0],
            source_path_symb.arguments[1],
        )
        logger.debug(
            "Retrieving node from tree of source %s at path %s.", source_symb, path_symb
        )
        path_list = self._path2py(path_symb)
        try:
            source = self.sources[source_symb]
        except KeyError as exc:
            raise ValueError(f"Unknown source symbol: {source_symb}.") from exc
        tree = source.tree
        node = tree.root_node
        for idx in path_list:
            try:
                tmp_node = node.child(idx)
                if tmp_node is None:  # nocoverage
                    raise ValueError(f"No node found in tree at path: {path_symb}.")
            except IndexError as exc:
                raise ValueError(f"No node found in tree at path: {path_symb}.") from exc
            node = tmp_node
        return source, node

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
            path = source.resolve()
            if not path.is_file():  # nocoverage
                raise IOError(f"File {path} not found.")
            source_bytes = path.read_bytes()
        else:  # nocoverage
            raise TypeError(
                f"Argument 'source' must be of type {SourceInput}, got: {type(source)}."
            )
        if identifier is None:
            identifier = Function("s", [Number(len(self.sources))])
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "Assigning identifier %s to new source with contents: '%s'",
                identifier,
                source_bytes.decode(encoding).replace("\n", "\\n"),
            )
            logger.info(
                "Parsing source %s with language set to %s and encoding set to %s.",
                identifier,
                language.name,
                encoding,
            )
        tree = parser.parse(source_bytes, encoding=encoding)
        processed_source = Source(identifier, source_bytes, path, encoding, parser, tree)
        self.sources[identifier] = processed_source
        self.facts.extend(self._reify_ts_tree(tree, processed_source))
        if path is not None:
            self.facts.append(Function("source_path", [identifier, String(str(path))]))
        return identifier

    def _reify_node_attrs(
        self, node: ts.Node, node_id: Symbol, encoding: StringEncoding
    ) -> list[Symbol]:
        """Reify a tree-sitter node and it's attributes into a (set of) fact(s)."""
        facts: list[Symbol] = []
        facts.append(Function("node", [node_id]))
        # we don't reify the type of error nodes as we mark them via
        # error(Node), making the fact type(Node, "ERROR") redundant.
        if node.is_named and not node.is_error:
            facts.append(Function("type", [node_id, String(node.type)]))
        if node.is_error:
            facts.append(Function("error", [node_id]))
        if node.is_missing:
            facts.append(Function("missing", [node_id]))
        if node.is_extra:  # nocoverage
            facts.append(Function("extra", [node_id]))
        if node.child_count == 0 and node.text is not None:
            facts.append(
                Function(
                    "leaf_text",
                    [node_id, String(node.text.decode(encoding))],
                )
            )
        return facts

    def _reify_ts_subtree(
        self, subtree_root_node: ts.Node, encoding: StringEncoding
    ) -> list[Symbol]:
        """Reify tree-sitter subtree with input root node into a list of facts.

        The first element of this list is guarenteed to be the node/1
        fact corresponding to the root of the subtree.
        """
        subtree_root_id = next(self._id_generator)
        stack: list[tuple[Symbol, ts.Node]] = [(subtree_root_id, subtree_root_node)]
        facts: list[Symbol] = []
        while len(stack) > 0:
            parent_id, parent = stack.pop()
            facts.extend(self._reify_node_attrs(parent, parent_id, encoding))
            prev_child_id: Optional[Symbol] = None
            for idx, child in enumerate(parent.children):
                child_id = next(self._id_generator)
                if idx != 0:
                    facts.append(
                        Function("next_sibling", [prev_child_id, child_id])  # type:ignore
                    )
                facts.append(Function("child", [parent_id, child_id]))
                field_name = parent.field_name_for_child(idx)
                if field_name is not None:
                    facts.append(Function("field", [child_id, String(field_name)]))
                prev_child_id = child_id
                stack.append((child_id, child))
        return facts

    def _reify_ts_tree(self, tree: ts.Tree, source: Source) -> list[Symbol]:
        """Reify tree-sitter tree into a set of facts."""
        logger.info("Reifying parse tree of source %s.", source.id)
        facts: list[Symbol] = []
        root_node = tree.root_node
        if source.parser.language is None:  # nocoverage
            raise ValueError(f"Parser of source should not be None: {source}.")
        lang_name = source.parser.language.name
        if lang_name is None:  # nocoverage
            raise ValueError(f"Language of parser of source cannot be None: {source}.")
        lang_fact = Function(
            "language",
            [
                source.id,
                String(source.parser.language.name),  # type: ignore
            ],
        )
        facts.append(lang_fact)
        tree_facts = self._reify_ts_subtree(root_node, source.encoding)
        root_id = tree_facts[0].arguments[0]
        facts.append(Function("source_root", [source.id, root_id]))
        facts.extend(tree_facts)
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "Resulting facts from reifying source %s: %s",
                source.id,
                " ".join([str(s) + "." for s in facts]),
            )
        return facts

    def transform(
        self,
        *,
        meta_files: Optional[Sequence[Path]] = None,
        meta_string: Optional[str] = None,
        initial_program: tuple[str, Sequence[Symbol]] = ("base", ()),
        control_options: Optional[Sequence[str]] = None,
    ) -> None:
        """Transform fact base via a meta-encoding."""
        options = control_options if control_options is not None else []
        if meta_files is not None:  # nocoverage
            for f in meta_files:
                if not f.is_file():
                    raise IOError(f"File {f} not found.")
        encoding_files = [str(f) for f in meta_files] if meta_files is not None else []

        base_encodings = [encoding_path / "transform" / "all.lp"]
        encoding_files.extend([str(p) for p in base_encodings])
        self.next_transform_program = initial_program
        parts: Sequence[tuple[str, Sequence[Symbol]]]
        while self.next_transform_program is not None:
            logger.debug("Initializing clingo.Control with options %s.", options)
            control = Control(arguments=options, logger=clingo_logger)
            parts = (
                [base_program]
                if self.next_transform_program == base_program
                else [base_program, self.next_transform_program]
            )
            logger.info(
                (
                    "Grounding and solving program parts "
                    "%s of transformation meta-encoding files %s and string '%s'."
                ),
                parts,
                encoding_files,
                meta_string,
            )
            for fi in encoding_files:
                control.load(fi)
            if meta_string is not None:
                control.add(meta_string)
            with control.backend() as backend:
                for fact in self.facts:
                    atom = backend.add_atom(fact)
                    backend.add_rule([atom])
            control.ground(parts=parts)
            self.next_transform_program = None
            control.solve(on_model=self._on_transform_model)

    def _on_transform_model(self, model: Model) -> Literal[False]:
        """Model callback for transformation. Returns False as we only
        expect one model."""
        if logger.isEnabledFor(logging.INFO):  # nocoverage
            logger.info(
                ("Found stable model with shown atoms: %s"),
                " ".join([str(s) + "." for s in model.symbols(shown=True)]),
            )
        edit_symbols: list[Symbol] = []
        next_transform_symbols: list[Symbol] = []
        deps: defaultdict[Symbol, list[Symbol]] = defaultdict(list)
        log_symbols: list[Symbol] = []
        exception_symbols: list[Symbol] = []
        for symb in model.symbols(shown=True):
            if symb.match("aspen", 1):
                arg = symb.arguments[0]
                if arg.match("edit", 2):
                    edit_symbols.append(arg)
                elif arg.match("comes_before", 2):
                    deps[arg.arguments[1]].append(arg.arguments[0])
                elif arg.match("next_program", 2):
                    next_transform_symbols.append(arg)
                elif arg.match("log", 3) or arg.match("log", 2):
                    log_symbols.append(arg)
                elif arg.match("exception", 2) or arg.match("exception", 1):
                    exception_symbols.append(arg)
                elif arg.match("return", 2) and arg.arguments[0].match("path_of_node", 1):
                    node_id = arg.arguments[0].arguments[0]
                    self._node_id2source_node[Function("node", [node_id])] = (
                        self._source_path2py_source_node(arg.arguments[1])
                    )
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
                or next_symb.arguments[1].type != SymbolType.Function
                or next_symb.arguments[1].name != ""
            ):
                raise ValueError(
                    "First argument of next_program must be a string, "
                    f"second must be a tuple, found: {next_symb}."
                )
            prog_name = next_symb.arguments[0].string
            prog_params = next_symb.arguments[1].arguments
            logger.info(
                "Setting next transformation program to (%s, %s)",
                prog_name,
                [str(s) for s in prog_params],
            )
            self.next_transform_program = (prog_name, prog_params)
        self._process_log_symbs(log_symbols)
        self._process_exception_symbs(exception_symbols)
        sorted_edit_symbols = self._topological_sort_edits(edit_symbols, deps)
        edited_sources = self._edit_sources_from_symbs(sorted_edit_symbols)
        self._reparse_sources(edited_sources)
        return False

    def _get_loc_prefix_from_source_node(self, source: Source, node: ts.Node) -> str:
        """Given a symbolic tuple of a source and path, calculate the
        node's location prefix string for error and log messages.

        """
        if source.path is not None:
            source_str = str(source.path)
        else:
            source_str = str(source.id)
        start_p, end_p = node.start_point, node.end_point
        if start_p.row == end_p.row:
            span_str = f"{start_p.row}:{start_p.column}-{end_p.column}"
        else:
            span_str = f"{start_p.row}:{start_p.column}-{end_p.row}:{end_p.column}"
        loc_prefix = f"{source_str}:{span_str}: "
        return loc_prefix

    def _process_log_symbs(self, log_symbols: list[Symbol]) -> None:
        """Emit logs based on log symbols."""
        for symb in log_symbols:
            logger.debug("Processing log symbol %s.", symb)
            if len(symb.arguments) == 2:
                log_level_symb = symb.arguments[0]
                loc_prefix = " "
                text = self._format_str_symb2str(symb.arguments[1])
            # case when len(symb.arguments) == 3
            else:
                log_level_symb = symb.arguments[1]
                source, node = self._node_id2source_node[symb.arguments[0]]
                loc_prefix = self._get_loc_prefix_from_source_node(source, node)
                text = self._format_str_symb2str(symb.arguments[2])
            if (
                log_level_symb.type == SymbolType.String
                and log_level_symb.string in log_lvl_strs
            ):
                log_lvl_str = log_level_symb.string
                log_lvl = log_lvl_str2int[log_lvl_str]
            else:  # nocoverage
                raise ValueError(
                    f"Level of log symbol {symb} must be" f" one of {log_lvl_strs}."
                )

            log_msg = f"{loc_prefix}{text}"
            logger.debug(
                "Log level and text of symbol after processing: %s, %s", log_lvl, log_msg
            )
            logger.log(log_lvl, log_msg)

    def _process_exception_symbs(self, exception_symbols: list[Symbol]) -> None:
        """Raise errors based on exception symbols."""
        error_msgs: list[str] = []
        for symb in exception_symbols:
            logger.debug("Processing exception symbol %s.", symb)
            if len(symb.arguments) == 1:
                loc_prefix = ""
                text = self._format_str_symb2str(symb.arguments[0])
            # case when len(symb.arguments) == 2
            else:
                source, node = self._node_id2source_node[symb.arguments[0]]
                loc_prefix = self._get_loc_prefix_from_source_node(source, node)
                text = self._format_str_symb2str(symb.arguments[1])
            error_msgs.append(f"{loc_prefix}{text}")
        if len(error_msgs) > 0:
            raise TransformError("\n".join(error_msgs))

    def _format_str_symb2str(self, symb: Symbol) -> str:
        """Convert format string symbol to python str."""
        py_str: str
        if symb.type == SymbolType.String:
            py_str = symb.string
        elif symb.match("node", 1):
            try:
                source, node = self._node_id2source_node[symb]
                start, end = node.start_byte, node.end_byte
                py_str = source.source_bytes[start:end].decode(source.encoding)
            # if the tuple is not a node id
            except ValueError as exc:  # nocoverage
                raise ValueError(
                    f"Symbol {symb} could not be converted to string."
                ) from exc
        elif (
            symb.match("format", 2)
            and symb.arguments[0].type == SymbolType.String
            and symb.arguments[1].match("", 2)
        ):
            format_string = symb.arguments[0].string
            inserts = self._cons_list2py(symb.arguments[1])
            insert_strs: list[str] = [self._format_str_symb2str(s) for s in inserts]
            py_str = format_string.format(*insert_strs)
        elif (
            symb.match("join", 2)
            and symb.arguments[0].type == SymbolType.String
            and symb.arguments[1].match("", 2)
        ):
            join_str = symb.arguments[0].string
            inserts = self._cons_list2py(symb.arguments[1])
            insert_strs = [self._format_str_symb2str(s) for s in inserts]
            py_str = join_str.join(insert_strs)
        else:
            raise ValueError(f"Symbol {symb} could not be converted to string.")
        return py_str

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
            cycle_str = " <-\n".join([str(symb) for symb in exc.args[1]])
            msg = (
                "Transformation edits define cyclic dependencies via format strings. "
                f"One such cycle:\n{cycle_str}"
            )
            raise ValueError(msg) from exc
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Topological ordering found for edit symbols: %s",
                " -> ".join([str(s) for s in edit_symbols]),
            )
        return edit_symbols

    def _edit_sources_from_symbs(self, edit_symbols: Sequence[Symbol]) -> set[Symbol]:
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
        edited_sources: set[Symbol] = set()
        for symb in edit_symbols:
            logger.info("Processing edit symbol: %s.", symb)
            replacement = symb.arguments[1]
            target_source, target_node = self._node_id2source_node[symb.arguments[0]]
            replacement_text = self._format_str_symb2str(replacement)
            logger.debug(
                "Formatted replacement text of of edit symbol: '%s'", replacement_text
            )
            replacement_bytes = bytes(replacement_text, target_source.encoding)
            if logger.isEnabledFor(logging.INFO):
                logger.info(
                    "Text of source %s before applying edit: '%s'",
                    target_source.id,
                    str(
                        target_source.source_bytes, encoding=target_source.encoding
                    ).replace("\n", "\\n"),
                )
            edit_range = calc_node_edit_range(target_node, replacement_bytes)
            target_source.source_bytes = edit_tree(
                target_source.tree,
                edit_range,
                replacement_bytes,
                target_source.source_bytes,
            )
            if logger.isEnabledFor(logging.INFO):
                logger.info(
                    "Text of source %s after applying edit: '%s'",
                    target_source.id,
                    str(
                        target_source.source_bytes, encoding=target_source.encoding
                    ).replace("\n", "\\n"),
                )
            edited_sources.add(target_source.id)
        return edited_sources

    def _reparse_sources(self, edited_sources: set[Symbol]) -> None:
        """Re-parse sources that have been edited, and update fact
        representation based on the changed ranges of sources.
        """
        source_changes: dict[Symbol, list[Change]] = {}
        for source_symb in edited_sources:
            try:
                source = self.sources[source_symb]
            except KeyError as exc:  # nocoverage
                raise ValueError(f"Unknown source symbol: {source_symb}.") from exc
            logger.info("Reparsing source %s after edit.", source_symb)
            old_tree = source.tree
            new_tree = source.parser.parse(
                source.source_bytes, old_tree, encoding=source.encoding
            )
            changes = get_tree_changes(old_tree, new_tree)
            source.tree = new_tree
            source_changes[source_symb] = changes
        self._re_reify_changed_subtrees(source_changes)

    def _re_reify_changed_subtrees(
        self,
        source_changes: dict[Symbol, list[Change]],
    ) -> None:
        """Re-reify subtrees who's syntactic structure changed due to
        edit, and delete outdated facts from before edit."""

        query2siblings: dict[Symbol, list[ts.Node]] = {}
        new_facts: list[Symbol] = []
        for source_symb, changes in source_changes.items():
            for change in changes:
                old_siblings, new_siblings = change
                logger.debug(
                    "Processing change in old tree at range: %s, %s",
                    old_siblings[0].start_point,
                    old_siblings[-1].end_point,
                )
                first_old_sib_path = self._py_node2path_symb(old_siblings[0])
                query = Function(
                    "re_reify_siblings",
                    [source_symb, first_old_sib_path, Number(len(old_siblings))],
                )
                query2siblings[query] = new_siblings
        control = Control(logger=clingo_logger)
        parts = [base_program]
        encodings = [
            generic_util_path / "queries" / "re_reify_siblings.lp",
            encoding_path / "transform" / "defined.lp",
        ]
        for encoding in encodings:
            control.load(str(encoding))
        with control.backend() as backend:
            for query in query2siblings:
                query = Function("aspen", [Function("query", [query])])
                atom = backend.add_atom(query)
                backend.add_rule([atom])
            for f in self.facts:
                atom = backend.add_atom(f)
                backend.add_rule([atom])
        control.ground(parts=parts)
        delete_facts: set[Symbol] = set()
        query2_related_dict: defaultdict[Symbol, dict[str, Symbol]] = defaultdict(dict)

        def _on_re_reify_model(model: Model) -> Literal[False]:
            if logger.isEnabledFor(logging.INFO):  # nocoverage
                logger.info(
                    ("Found stable model with shown atoms: %s"),
                    " ".join([str(s) + "." for s in model.symbols(shown=True)]),
                )
            related_sib_names = {
                "parent",
                "prev_sibling",
                "next_sibling",
            }
            for symb in model.symbols(shown=True):
                if (
                    symb.match("aspen", 1)
                    and symb.arguments[0].match("return", 2)
                    and symb.arguments[0].arguments[0].match("re_reify_siblings", 3)
                ):
                    query = symb.arguments[0].arguments[0]
                    ret_value = symb.arguments[0].arguments[1]
                    if ret_value.match("delete", 1):
                        delete_facts.add(ret_value.arguments[0])
                    elif (
                        ret_value.type == SymbolType.Function
                        and ret_value.positive
                        and len(ret_value.arguments) == 1
                        and ret_value.name in related_sib_names
                    ):
                        query2_related_dict[query][ret_value.name] = ret_value.arguments[
                            0
                        ]
            return False

        control.solve(on_model=_on_re_reify_model)
        for query, siblings in query2siblings.items():
            kwargs = query2_related_dict[query]
            logger.debug(
                "Processing query %s with siblings %s and args %s",
                query,
                siblings,
                kwargs,
            )
            source_symb = query.arguments[0]
            source = self.sources[source_symb]
            facts = self._reify_changed_siblings(source, siblings, **kwargs)
            new_facts.extend(facts)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Deleting following obsolete facts before adding new facts: %s",
                " ".join([str(s) for s in delete_facts]),
            )
            logger.debug(
                "Adding following new facts from re-reification(s): %s",
                " ".join([str(s) for s in new_facts]),
            )
        self.facts = [f for f in self.facts if f not in delete_facts]
        self.facts.extend(new_facts)

    def _reify_changed_siblings(
        self,
        source: Source,
        siblings: list[ts.Node],
        *,
        parent: Optional[Symbol] = None,
        prev_sibling: Optional[Symbol] = None,
        next_sibling: Optional[Symbol] = None,
    ) -> list[Symbol]:
        """Given a new list of sibling nodes that have changed after
        re-parsing, generate the list of facts that need to be added
        to the factbase to reflect the new sibling nodes.

        """
        facts: list[Symbol] = []
        encoding = source.encoding
        for idx, node in enumerate(siblings):
            subtree_facts = self._reify_ts_subtree(node, encoding)
            node_id = subtree_facts[0].arguments[0]
            facts.extend(subtree_facts)
            if prev_sibling is not None:
                facts.append(Function("next_sibling", [prev_sibling, node_id]))
            if parent is not None and node.parent is not None:
                facts.append(Function("child", [parent, node_id]))
                child_index = node.parent.children.index(node)
                field_name = node.parent.field_name_for_child(child_index)
                if field_name is not None:
                    facts.append(Function("field", [node_id, String(field_name)]))
            if idx == len(siblings) - 1:
                if next_sibling is not None:
                    facts.append(Function("next_sibling", [node_id, next_sibling]))
            prev_sibling = node_id
        return facts
