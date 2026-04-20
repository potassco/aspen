stm(S) :- type(S, ("rule";"weak_constraint")).                               %%#(\label{lst:safe:util-start}#)
var(V,N) :- type(V,"variable"), leaf_text(V,N).

agg_element(A,E) :- type(A,"aggregate"), child(A,ES), child(ES,E),
  type(ES,"aggregate_elements"), type(E,"aggregate_element").                %%#(\label{lst:safe:util-end}#)

eq(A,E) :- type(A,"builtin_atom"), child(A,R), leaf_text(R,"="),
  child(A,E), E != R.

eq(A,V) :- type(A,"aggregate"), child(A,U), type(U,"upper"),
  child(U,R), child(U,V), leaf_text(R,"="), var(V,_).

atoms(S,A) :- child(S,B), type(B,"body"), child(B,NL),                         %%#(\label{lst:safe:atoms-rule-naf-start}#)
  type(NL,"naf_literal"), child(NL,A), not type(A,"naf").                    %%#(\label{lst:safe:atoms-rule-naf-end}#)

atoms(S,A) :- child(S,B), type(B,"body"), child(B,A), type(A,"aggregate").     %%#(\label{lst:safe:atoms-rule-agg}#)

atoms(E,A) :- type(E,"aggregate_element"), child(E,NLS),                       %%#(\label{lst:safe:atoms-agg-element-start}#)
  type(NLS,"naf_literals"), child(NLS,NL),
  type(NL,"naf_literal"), child(NL,A), not type(A,"naf").                    %%#(\label{lst:safe:atoms-agg-element-end}#)

negated(A) :- atoms(_,A), next_sibling(N,A), type(N,"naf").                    %%#(\label{lst:safe:neg}#)

desc(B,binop) :- type(B,"binary_operation").
desc(S,S;A,A) :- atoms(S,A).
desc(C,A) :- desc(P,A), child(P,C).                                          %%#(\label{lst:safe:desc-end}#)

scope(S,N,"global") :- var(V,N), stm(S), desc(V,S),                          %%#(\label{lst:safe:global-start}#)
  not desc(V,E): atoms(S,A), agg_element(A,E).

scope(E,N,"global") :- var(V,N), stm(S), scope(S,N,"global"),
  atoms(S,A), agg_element(A,E), desc(V, E).                                    %%#(\label{lst:safe:global-end}#)

scope(S,N,"local") :- var(V,N), atoms(S,_), desc(V,S),                       %%#(\label{lst:safe:local}#)
  not scope(S,N,"global").

binds(S,A,V') :- binds(S,A,V), var(V,N), var(V',N), desc(V',S).              %%#(\label{lst:safe:binds-shared-name}#)

binds(S,A,V) :- atoms(S,A), type(A,"classical_atom"), not negated(A),        %%#(\label{lst:safe:binds-classic}#)
  var(V,_), not desc(V,binop), desc(V,A).

other_binds(S,A,V') :- atoms(S,A), not negated(A), eq(A,V), var(V,N),          %%#(\label{lst:safe:binds-eq-other-start}#)
  scope(S,N,C), desc(V',A), var(V',N'), scope(S,N',C), binds(S,A',V'), A'!=A.

binds(S,A,V) :- atoms(S,A), not negated(A), eq(A,V), var(V,N), scope(S,N,C),   %%#(\label{lst:safe:binds-eq-start}#)
  other_binds(S,A,V'): var(V',N'), scope(S,N',C), V'!=V, desc(V',A).         %%#(\label{lst:safe:binds-eq-end}#)

aspen(exception(node(V),format("Variable '{}' is unsafe.",(node(V),())))) :- %%#(\label{lst:safe:aspen-exc-start}#)
  var(V,_), not binds(_,_,V).                                                %%#(\label{lst:safe:aspen-exc-end}#)
