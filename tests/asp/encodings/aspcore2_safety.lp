stm(S) :- type(S, ("rule";"weak_constraint")).                               %%#(\label{lst:safe:util-start}#)
var(V,N) :- type(V,"variable"), leaf_text(V,N).
eq(A,E) :- type(A,"builtin_atom"), child(A,R), leaf_text(R,"="),
  child(A,E), E != R.

agg_eq(A,V) :- type(A,"aggregate"), child(A,U), type(U,"upper"),
  child(U,R), child(U,V), leaf_text(R,"="), var(V,_).

pos_atom(A) :- type(A,"classical_atom"), not negated(A).

agg_element(A,E) :- type(A,"aggregate"), child(A,ES),
  type(ES,"aggregate_elements"), child(ES,E), type(E,"aggregate_element").   %%#(\label{lst:safe:util-end}#)

set(S,A) :- child(S,B), type(B,"body"), child(B,NL),                         %%#(\label{lst:safe:set-rule-naf-start}#)
  type(NL,"naf_literal"), child(NL,A), not type(A,"naf").                    %%#(\label{lst:safe:set-rule-naf-end}#)

set(S,A) :- child(S,B), type(B,"body"), child(B,A), type(A,"aggregate").     %%#(\label{lst:safe:set-rule-agg}#)

set(E,A) :- type(E,"aggregate_element"), child(E,NLS),                       %%#(\label{lst:safe:set-agg-element-start}#)
  type(NLS,"naf_literals"), child(NLS,NL),
  type(NL,"naf_literal"), child(NL,A), not type(A,"naf").                    %%#(\label{lst:safe:set-agg-element-end}#)

negated(A) :- set(B,A), next_sibling(N,A), type(N,"naf").                    %%#(\label{lst:safe:neg}#)
desc(E,E) :- eq(B,E).                                                   %%#(\label{lst:safe:desc-start}#)
desc(B,binop) :- type(B,"binary_operation").
desc(A,A) :- pos_atom(A).
desc(S,S;A,A) :- set(S,A).
desc(C,A) :- desc(P,A), child(P,C).                                          %%#(\label{lst:safe:desc-end}#)

context(S,N,"global") :- var(V,N), stm(S), desc(V,S),                        %%#(\label{lst:safe:global-start}#)
  not desc(V,E): set(S,A), agg_element(A,E).

context(E,N,"global") :- var(V,N), stm(S), context(S,N,"global"),
  set(S,A), agg_element(A,E), desc(V, E).                                    %%#(\label{lst:safe:global-end}#)

context(S,N,"local") :- var(V,N), set(S,_), desc(V,S), not context(S,N,"global"). %%#(\label{lst:safe:local}#)
binds(S,A,V') :- binds(S,A,V), var(V,N), var(V',N), desc(V',S).               %%#(\label{lst:safe:binds-shared-name}#)
binds(S,A,V) :- var(V,_), not desc(V,binop), set(S,A), desc(V,A), pos_atom(A).%%#(\label{lst:safe:binds-classic}#)

other_binds(S,A,V) :- set(S,A), eq(A,E), desc(V,E),                     %%#(\label{lst:safe:binds-eq-other-start}#)
  binds(S,A',V), A' != A.                                                    %%#(\label{lst:safe:binds-eq-other-end}#)

binds(S,A,V) :- set(S,A), eq(A,V), var(V,N), context(S,N,C),            %%#(\label{lst:safe:binds-eq-start}#)
  eq(A,E), E!=V, other_binds(S,A,V'): var(V',N'), context(S,N',C),
  		      			   desc(V',E).                       %%#(\label{lst:safe:binds-eq-end}#)

other_binds(S,A,V) :- set(S,A), agg_eq(A,_), not negated(A),                 %%#(\label{lst:safe:binds-agg-other-start}#)
  agg_element(A,E), desc(V,E), var(V,N),
  context(S,N,"global"), binds(S,A',V), A' != A.                             %%#(\label{lst:safe:binds-agg-other-end}#)

binds(S,A,V) :- set(S,A), agg_eq(A,V), not negated(A),                       %%#(\label{lst:safe:binds-agg-start}#)
  other_binds(S,A,V'): agg_element(A,E), var(V',N),
		       desc(V',E), context(S,N,"global").                    %%#(\label{lst:safe:binds-agg-end}#)

aspen(exception(node(V),format("Variable '{}' is unsafe.",(node(V),())))) :- %%#(\label{lst:safe:aspen-exc-start}#)
  var(V,_), not binds(_,_,V).                                                %%#(\label{lst:safe:aspen-exc-end}#)
