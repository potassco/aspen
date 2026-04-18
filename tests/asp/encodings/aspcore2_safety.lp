% helper predicates
stm(S) :- type(S, ("rule";"weak_constraint")).

var(V) :- type(V, "variable").

eq(A) :- type(A, "builtin_atom"), child(A, R), leaf_text(R, "=").

eq_expr(A, E) :- eq(A), child(A, E), not leaf_text(E, "=").

agg_eq(A) :-
  type(A, "aggregate"),
  child(A, U), type(U, "upper"),
  child(U, R), leaf_text(R, "=").

agg_element(A, E) :-
  type(A, "aggregate"), child(A, ES),
  type(ES, "aggregate_elements"), child(ES, E),
  type(E, "aggregate_element").

agg_rhs(A, V) :-
  type(A, "aggregate"), child(A, U),
  type(U, "upper"), child(U, V),
  var(V).

% construct sets of atoms that might bind variables.
atom_set(S, A) :-
  child(S, B),
  type(B, "body"),
  child(B, NL), type(NL, "naf_literal"),
  child(NL, A), not type(A, "naf").

atom_set(S, A) :- child(S, B), type(B, "body"), child(B, A), type(A, "aggregate").

atom_set(E, A) :-
  type(E, "aggregate_element"),
  child(E, NLS), type(NLS, "naf_literals"),
  child(NLS, NL), type(NL, "naf_literal"),
  child(NL, A), not type(A, "naf").

negated(A) :- atom_set(B, A), next_sibling(N, A), type(N, "naf").

% find variables and deterimine if they are in scope of relevant language constructs

in_scope(S, stm(S)) :- stm(S).
in_scope(E, eq_expr(E)) :- eq_expr(B, E).
in_scope(B, binop) :- type(B, "binary_operation").
in_scope(B, agg_element) :- type(B, "aggregate_element").
in_scope(A, pos_atom(A)) :- type(A, "classical_atom"), not negated(A).
in_scope(S, atom_set(S)) :- atom_set(S, _).

in_scope(C, A) :- in_scope(P, A), child(P, C).

% global vs local variables.

context(S, N, "global") :-
  var(V), leaf_text(V, N),
  stm(S), in_scope(V, atom_set(S)), not in_scope(V, agg_element).

context(E, N, "global") :-
  var(V), leaf_text(V, N),
  stm(S), context(S, N, "global"),
  atom_set(S, A), agg_element(A, E),
  in_scope(V, atom_set(E)).

context(S, N, "local") :-
  var(V), leaf_text(V, N), in_scope(V, atom_set(S)), not context(S, N, "global").

% define when a set of atoms bind a variable

% If a variable is bound by atom A in atom set S,
% all variables of the same name are also bound by it.
binds_occur(S, A, V') :-
  binds_occur(S, A, V), leaf_text(V, N),
  var(V'), leaf_text(V', N), in_scope(V', atom_set(S)).

% a non-negated classical atom binds variables
% that are not in the scope of an arithmetic term
binds_occur(S, A, V) :-
  var(V), not in_scope(V, binop),
  atom_set(S, A), in_scope(V, pos_atom(A)).

% a built-in equality atom in an atom set binds a variable that is
% the right- or left-hand side of the equation if all variables in the
% other side of the equation are bound by some other atom in the set.

% Todo: if the equality atom is in an aggregate, we should only need to check variables that are local

bound_by_other(S, A, V) :-
  atom_set(S, A),
  eq_expr(A, E),
  in_scope(V, eq_expr(E)),
  binds_occur(S, A', V), A' != A.

binds_occur(S, A, V) :-
  atom_set(S, A),
  eq_expr(A, V), var(V), leaf_text(V, N), context(S, N, C),
  eq_expr(A, E), E!=V,
  bound_by_other(S, A, V'): var(V'), leaf_text(V',N'), context(S, N', C),
  		       	    in_scope(V', eq_expr(E)).

% An equality aggregate atom member of an atom set binds a variable on it's right-hand side
% if all global variables occurring in the aggregate's elements are bound by some other atom in the atom set
bound_by_other(S, A, V) :-
  atom_set(S, A), agg_eq(A), not negated(A),
  agg_element(A, E),
  in_scope(V, atom_set(E)), leaf_text(V, N), context(S, N, "global"),
  binds_occur(S, A', V), A' != A.

binds_occur(S, A, V) :-
  atom_set(S, A), agg_eq(A), not negated(A), agg_rhs(A, V),
  bound_by_other(S, A, V'): agg_element(A, E), var(V'),in_scope(V', atom_set(E)),
  		       	    leaf_text(V', N), context(S, N, "global").

% todo: nice error message highlighting where the variable is unsafe

% unsafe_set() :- in_scope(S, ).
% aspen(exception(node(V), format("Variable '{}' is unsafe.", (node(V), ())))) :-
%   var(V), not binds_occur(_, _, V).

% log error if variable is unsafe
aspen(exception(node(V), format("Variable '{}' is unsafe.", (node(V), ())))) :-
  var(V), not binds_occur(_, _, V).

% raise syntax errors
syntax_error(N) :- error(N).
syntax_error(N) :- missing(N).
aspen(exception(node(N), format("Syntax error at node '{}'.", (node(N), ())))) :- syntax_error(N).

% #show negated/1.
% #show in_scope/2.
% #show atom_set/2.
% #show binds_occur/3.
% #show bound_by_other/3.
% #show agg_eq/1.
% #show agg_element/2.
% #show agg_rhs/2.
% #show context/3.
% #show eq_expr/2.
