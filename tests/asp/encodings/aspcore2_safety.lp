% helper predicates
stm(S) :- type(S, ("rule";"weak_constraint")).
var(V, N) :- type(V, "variable"), leaf_text(V, N).
eq(A) :- type(A, "builtin_atom"), child(A, R), leaf_text(R, "=").
eq_expr(A, E) :- eq(A), child(A, E), not leaf_text(E, "=").

agg_eq(A, V) :-
  type(A, "aggregate"), child(A, U),
  type(U, "upper"), child(U, R), child(U, V),
  leaf_text(R, "="), var(V,_).

agg_element(A, E) :-
  type(A, "aggregate"), child(A, ES),
  type(ES, "aggregate_elements"), child(ES, E),
  type(E, "aggregate_element").

% construct sets of atoms that might bind variables.
atom_set(S, A) :-
  child(S, B),
  type(B, "body"), child(B, NL),
  type(NL, "naf_literal"), child(NL, A),
  not type(A, "naf").

atom_set(S, A) :- child(S, B), type(B, "body"), child(B, A), type(A, "aggregate").

atom_set(E, A) :-
  type(E, "aggregate_element"), child(E, NLS),
  type(NLS, "naf_literals"), child(NLS, NL),
  type(NL, "naf_literal"), child(NL, A),
  not type(A, "naf").

negated(A) :- atom_set(B, A), next_sibling(N, A), type(N, "naf").

% find variables and deterimine if they are descendants of relevant language constructs
desc(E, eq_expr(E)) :- eq_expr(B, E).
desc(B, binop) :- type(B, "binary_operation").
desc(A, pos_atom(A)) :- type(A, "classical_atom"), not negated(A).
desc(S, atom_set(S)) :- atom_set(S, _).
desc(C, A) :- desc(P, A), child(P, C).

% global vs local variables.
context(S, N, "global") :-
  var(V, N),
  stm(S), desc(V, atom_set(S)),
  not desc(V, atom_set(E)): atom_set(S, A), agg_element(A, E).

context(E, N, "global") :-
  var(V, N),
  stm(S), context(S, N, "global"),
  atom_set(S, A), agg_element(A, E),
  desc(V, atom_set(E)).

context(S, N, "local") :-
  var(V, N), desc(V, atom_set(S)), not context(S, N, "global").

% define when a set of atoms bind a variable

% If a variable is bound by atom A in atom set S,
% all variables of the same name are also bound by it.
binds_occur(S, A, V') :-
  binds_occur(S, A, V), var(V, N),
  var(V', N), desc(V', atom_set(S)).

% a non-negated classical atom binds variables
% that are not in the scope of an arithmetic term
binds_occur(S, A, V) :-
  var(V,_), not desc(V, binop),
  atom_set(S, A), desc(V, pos_atom(A)).

% a built-in equality atom in an atom set binds a variable that is
% the right- or left-hand side of the equation if all variables in the
% other side of the equation are bound by some other atom in the set.
bound_by_other(S, A, V) :-
  atom_set(S, A),
  eq_expr(A, E),
  desc(V, eq_expr(E)),
  binds_occur(S, A', V), A' != A.

binds_occur(S, A, V) :-
  atom_set(S, A),
  eq_expr(A, V), var(V,N), context(S, N, C),
  eq_expr(A, E), E!=V,
  bound_by_other(S, A, V'): var(V',N'), context(S, N', C),
  		       	    desc(V', eq_expr(E)).

% An equality aggregate atom member of an atom set
% binds a variable on it's right-hand side
% if all global variables occurring in the aggregate's elements
% are bound by some other atom in the atom set
bound_by_other(S, A, V) :-
  atom_set(S, A), agg_eq(A,_), not negated(A),
  agg_element(A, E),
  desc(V, atom_set(E)), leaf_text(V, N), context(S, N, "global"),
  binds_occur(S, A', V), A' != A.

binds_occur(S, A, V) :-
  atom_set(S, A), agg_eq(A, V), not negated(A),
  bound_by_other(S, A, V'): agg_element(A, E), var(V', N), desc(V', atom_set(E)),
  		       	    context(S, N, "global").

% log error if variable is unsafe
unsafe_occ(V) :-  var(V,_), not binds_occur(_, _, V).

aspen(exception(node(S), format("Unsafe variable(s) in:\n  {}",
			 	(node(S), ())))) :-
  unsafe_occ(V), desc(V, atom_set(S)).


aspen(exception(node(V), format("Variable '{}' is unsafe.", (node(V), ())))) :-
  unsafe_occ(V).

% raise syntax errors
syntax_error(N) :- error(N).
syntax_error(N) :- missing(N).
aspen(exception(node(N), format("Syntax error at node '{}'.", (node(N), ())))) :- syntax_error(N).

% #show negated/1.
% #show desc/2.
% #show atom_set/2.
% #show binds_occur/3.
% #show bound_by_other/3.
% #show agg_eq/2.
% #show agg_element/2.
% #show context/3.
% #show eq_expr/2.
