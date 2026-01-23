% generate externals for metasp
% currently, we just handle normal rules, no disjunction, conditionals or aggregates.
#include "./metasp_extract_type_info.lp".
#include "./metasp_print_externals.lp".
#include "./metasp_rewrite_show.lp".
#include "../../../src/aspen/asp/utils/clingo/symbol_signature.lp".
#include "../../../src/aspen/asp/utils/generic/derived_predicates/named_child_index.lp".

#program metasp_preprocess.

% some definitions.


%% extended_expression_sig(TypeName, ("true", 0), ("true", 0)) :- type_name(_, TypeName).
%% extended_expression_sig(TypeName, ("false", 0), ("false", 0)) :- type_name(_, TypeName).
%% extended_expression_sig(TypeName, ("not", 1), ("not", 1)) :- type_name(_, TypeName).

%% extended_arg_kwarg(("not", 1), 0, "safety", "unsafe").

negation(N) :- type(N, ("default_negation"; "double_default_negation")).
positive(N)
  :- type(N, ("literal"; "body_literal")),
		 #false: child(N,S), negation(S).

extended_expression_sig(TypeName, ExpDef, Sig) :- expression_sig(TypeName, ExpDef, Sig).
extended_arg_kwarg(Sig, Idx, Name, Val) :- arg_kwarg(Sig, Idx, Name, Val).

metasp_symbol(N) :- type(N,("metasp_symbolic_atom";"metasp_function")).
macro_root_symbol((Head;Body)) :- valid_macro(_,Head,Body).
extended_root_symbol(N) :- type(N, ("metasp_symbolic_atom";"symbolic_atom")).
extended_root_symbol(N) :- macro_root_symbol(N).

symbol_signature(N,(Name,Arity))
  :- metasp_symbol(N), child(N,M), type(M,"terms"), field(M,"arguments"),
		 MaxIdx = #max{ Idx : child(M,L), named_child_index(L,Idx) }, Arity=MaxIdx+1,
		 child(N,J), type(J,"identifier"), field(J,"name"), leaf_text(J,Name).

symbol_signature(N,(Name,0))
  :- metasp_symbol(N), #false: child(N,M), type(M,"terms");
		 child(N,J), type(J,"identifier"), field(J,"name"), leaf_text(J,Name).

metasp_atom_type(N, TypeName)
  :- type(N, "metasp_symbolic_atom"),
		 symbol_signature(N, Sig),
		 extended_expression_sig(TypeName, _, Sig).

metasp_atom_allowed_occurrence(N, Occ)
  :- type(N, "metasp_symbolic_atom"),
		 symbol_signature(N, Sig),
		 metasp_atom_type(N, TypeName),
		 occurrence(TypeName, Occ).

metasp_atom_allowed_occurrence(N,("head";"body")) :- metasp_atom_allowed_occurrence(N, "any").
metasp_atom_allowed_occurrence(N,"directive") :- metasp_atom_allowed_occurrence(N, "head").

metasp_operand(N,Idx,O)
  :- metasp_symbol(N), child(N,M), type(M,"terms"),
		 child(M,O), named_child_index(O,Idx).

% deterine occurrences of symbols in program.

% possible occurrences

possible_head_occurrence(N)
  :- type(M,"literal"),
		 field(M,"head"),
		 child(M,N),
		 extended_root_symbol(N).

possible_head_occurrence(O)
  :- type(M,"disjunction"),
		 child(M,N),
		 type(N,"literal"),
		 child(N,O),
		 extended_root_symbol(O).

possible_head_occurrence(P)
  :- type(M, "disjunction"),
		 child(M,N),
		 type(N,"conditional_literal"),
		 child(N,O),
		 type(O,"literal"),
		 child(O,P),
	   extended_root_symbol(P).

possible_head_occurrence(Q)
  :- type(M, "set_aggregate"),
		 field(M, "head"),
		 child(M, N),
		 type(N, "set_aggregate_elements"),
		 child(N,O),
		 type(O, "set_aggregate_element"),
		 child(O,P),
		 type(P,"literal"),
		 child(P,Q),
		 extended_root_symbol(Q).

possible_head_occurrence(Q)
  :- type(M, "head_aggregate"),
		 child(M, N),
		 type(N, "head_aggregate_elements"),
		 child(N,O),
		 type(O, "head_aggregate_element"),
		 child(O,P),
		 type(P,"literal"),
		 child(P,Q),
		 extended_root_symbol(Q).

extended_atom_occurrence(N, "directive")
  :- type(M,"literal"),
		 positive(M),
		 field(M, "head"),
		 child(M,N),
		 extended_root_symbol(N),
		 child(L, M),
		 type(L, "rule"),
		 #false: child(L, B), type(B, "body").

% we do not count negated literals as head occurences
extended_atom_occurrence(N, "head")
  :- possible_head_occurrence(N),
		 child(Lit, N),
		 positive(Lit).

extended_atom_occurrence(M, "macrodef") :- macro_root_symbol(M).

extended_atom_occurrence(M, "body")
  :- extended_root_symbol(M),
		 not extended_atom_occurrence(M, "head"),
		 not extended_atom_occurrence(M, "macrodef").

%% print_symb(extended_atom_occurrence(M, Sig, Occ)) :- extended_atom_occurrence(M, Occ), symbol_signature(M, Sig).


% determine safety of atom operands.
metasp_operand_safety(N,O,Safety)
  :- symbol_signature(N,(Name, Arity)),
		 metasp_operand(N,Idx,O),
		 arg_safety((Name, Arity), Idx, Safety).

% base cases
% everything starts out as safe
root_symbol2operand(Root, Root, "safe")
  :- extended_root_symbol(Root).

% recursion
root_symbol2operand(Root, Operand, Safety)
  :- root_symbol2operand(Root, Symb, Safety),
		 metasp_operand_safety(Symb, Operand, "safe").

root_symbol2operand(Root, Operand, "unsafe")
  :- root_symbol2operand(Root, Symb, _),
		 metasp_operand_safety(Symb, Operand, "unsafe").

root_symbol2leaf_operand(Root, Symb, Safety)
  :- root_symbol2operand(Root, Symb, Safety),
		 not metasp_operand(Symb,_,_).


% deal with macros


macro_head_var(TypeName, Head, Body, Idx, Name)
  :- macro(TypeName, Head, Body),
		 metasp_operand(Head, Idx, Func),
		 type(Func, "function"),
		 symbol_signature(Func, (Name, 0)).

valid_macro(TypeName, Head, Body)
  :- macro(TypeName, Head, Body),
		 macro_head_var(Type, Head, Body, Idx, _): metasp_operand(Head, Idx, Operand).

extended_expression_sig(TypeName, Sig, Sig)
  :- valid_macro(TypeName, Head, Body),
		 symbol_signature(Head, Sig).

aspen(query(descendant(Body)))
  :- valid_macro(TypeName, Head, Body).

in_macro_body(Body, Node)
  :- valid_macro(TypeName, Head, Body),
		 aspen(return(query(descendant(Body)), Node)).

extended_arg_kwarg(Sig, Idx, "safety", "safe")
  :- valid_macro(TypeName, Head, Body),
		 in_macro_body(Body, Var),
		 root_symbol2leaf_operand(Body, Var, "safe"),
		 symbol_signature(Var, (Name, 0)),
		 macro_head_var(TypeName, Head, Body, Idx, Name),
		 symbol_signature(Head, Sig).

extended_arg_kwarg(Sig, Idx, "safety", "unsafe")
  :- valid_macro(TypeName, Head, Body),
		 symbol_signature(Head, Sig),
		 macro_head_var(TypeName, Head, Body, Idx, Name),
		 not extended_arg_kwarg(Sig, Idx, "safety", "safe").


% generate externals

safe_occurrence(N)
  :- extended_atom_occurrence(N, "body"),
		 child(O, N),
		 type(O, "body_literal"),
		 positive(O).

statement(SourceSymb, Statement)
  :- type(Source, "source_file"),
     child(Source, Statement),
		 source_root(SourceSymb, Source),
     not type(Statement, "metasp_type").

aspen(query(descendant(Statement))) :- statement(_, Statement).

statement_desc(Statement, Node)
  :- aspen(return(query(descendant(Statement)), Node)),
		 statement(_, Statement).

metasp_statement(Statement)
  :- statement_desc(Statement, Node),
		 metasp_symbol(Node).

metasp_statement_desc(Statement, Node)
  :- metasp_statement(Statement),
		 statement_desc(Statement, Node).

ext_head(HeadLeaf)
  :- metasp_statement_desc(Statement, HeadRoot),
		 extended_atom_occurrence(HeadRoot, "head"),
		 root_symbol2leaf_operand(HeadRoot, HeadLeaf, _).

ext_head(BodyRoot)
  :- metasp_statement_desc(Statement, BodyRoot),
		 type(BodyRoot, "metasp_symbolic_atom"),
		 extended_atom_occurrence(BodyRoot, "body").

ext_head_source(Head, Source)
  :- ext_head(Head),
		 metasp_statement_desc(Statement, Head),
		 statement(Source, Statement).

ext_body(HeadLeaf, BodyLeaf)
  :- metasp_statement_desc(Statement, HeadLeaf),
		 ext_head(HeadLeaf),
		 metasp_statement_desc(Statement, BodyRoot),
		 safe_occurrence(BodyRoot),
		 root_symbol2leaf_operand(BodyRoot, BodyLeaf, "safe"),
		 symbol(BodyLeaf).

% If the operator is part of a condition (as head or body), we need to
% include the condition as well in the external statement protecting
% the operator.

% 1. case: member is head of condition.
cond_member2cond(E,C)
  :- type(N, "literal"),
		 child(N,E),
		 extended_root_symbol(E),
		 child(P,N),
		 child(P,C),
		 type(C, "condition").

% 2. case: member is body of condition.
cond_member2cond(E,C)
  :- type(N, "literal"),
		 child(N,E),
		 extended_root_symbol(E),
		 child(C,N),
		 type(C, "condition").

% when ext_head is a head atom of the original statement
ext_body(Leaf, CondBodyLeaf)
  :- metasp_statement_desc(Statement, Root),
		 root_symbol2leaf_operand(Root,Leaf,_),
		 ext_head(Leaf),
		 cond_member2cond(Root, Cond),
		 child(Cond, Lit),
		 type(Lit, "literal"),
		 positive(Lit),
		 child(Lit, CondBodyRoot),
		 root_symbol2leaf_operand(CondBodyRoot, CondBodyLeaf, "safe"),
		 symbol(CondBodyLeaf).

% when ext_head is a body expression of the original statement
ext_body(Root, CondBodyLeaf)
  :- metasp_statement_desc(Statement, Root),
		 ext_head(Root),
		 cond_member2cond(Root, Cond),
		 child(Cond, Lit),
		 type(Lit, "literal"),
		 positive(Lit),
		 child(Lit, CondBodyRoot),
		 root_symbol2leaf_operand(CondBodyRoot, CondBodyLeaf, "safe"),
		 symbol(CondBodyLeaf).

%% aspen(print(format("cond member 2 cond: {}: {}", (node(E), (node(C), ()))))) :- cond_member2cond(E,C).
%% aspen(print(format("Root symbol 2 root operand: {}: {}, {}",
%% 									 (node(R), (node(L), (S, ()))))))
%%   :- root_symbol2leaf_operand(R,L,S).

%% print_symb(Sig) :- ext_head(H), symbol_signature(H, Sig).



% auxiliary predicates for generation of externals

ext_body_idx(Head, Body, Idx)
  :- ext_body(Head, Body),
		 Idx = #count{ B: ext_body(Head, B), B < Body }.

_ext_head_cons(node(Head), (), -1) :- ext_head(Head).

_ext_head_cons(node(Head), (node(Body), Cons), Idx+1)
  :- _ext_head_cons(node(Head), Cons, Idx),
		 ext_body_idx(Head, Body, Idx+1).

ext_head_cons(node(Head), Cons)
  :- _ext_head_cons(node(Head), Cons, Idx),
		 not ext_body_idx(Head, _, Idx+1).

aspen(print(@symbol_to_string(Symb))) :- print_symb(Symb).
