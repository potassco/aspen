stm(S) :- type(S,("rule";"weak_constraint")).                               %%#(\label{lst:safe:util-start}#)

var(V;V,N) :- type(V,"variable"), leaf_text(V,N).                            %%#(\label{lst:safe:util-end}#)

eq(A,T,U;A,U,T) :- type(A,"builtin_atom"), child(A,U),                      %%#(\label{lst:safe:eq-start}#)
   next_sibling(U,R), leaf_text(R,"="), next_sibling(R,T).                  %%#(\label{lst:safe:eq-end}#)

eq(A,E,U) :- type(A,"aggregate"), child(A,B), type(B,"upper"),              %%#(\label{lst:safe:agg-eq-start}#)
  child(B,R), leaf_text(R,"="), child(B,U), next_sibling(R,U),
  child(A,E), type(E,"aggregate_elements").                                %%#(\label{lst:safe:agg-eq-end}#)

atoms(S,A) :- child(S,B), type(B,"body"), child(B,NL),                       %%#(\label{lst:safe:atoms-rule-naf-start}#)
  type(NL,"naf_literal"), child(NL,A), not type(A,"naf").                    %%#(\label{lst:safe:atoms-rule-naf-end}#)

atoms(S,A) :- child(S,B), type(B,"body"), child(B,A), type(A,"aggregate").   %%#(\label{lst:safe:atoms-rule-agg}#)

atoms(S,A) :- type(S,"aggregate_element"), child(S,NLS),                     %%#(\label{lst:safe:atoms-agg-element-start}#)
  type(NLS,"naf_literals"), child(NLS,NL),
  type(NL,"naf_literal"), child(NL,A), not type(A,"naf").                    %%#(\label{lst:safe:atoms-agg-element-end}#)

negated(A) :- atoms(_,A), next_sibling(N,A), type(N,"naf").                  %%#(\label{lst:safe:neg}#)

desc(S,S) :- atoms(S,_).                                                     %%#(\label{lst:safe:desc-start}#)
desc(arith,B) :- type(B,("binary_operation";"unary_operation")).
desc(T,T;U,U) :- eq(A,T,U).
desc(A,A) :- type(A,"classical_atom").
desc(A,C) :- desc(A,P), child(P,C).                                          %%#(\label{lst:safe:desc-end}#)

scope(V,"global") :- var(V), stm(S), desc(S,V),                              %%#(\label{lst:safe:global-start}#)
  not desc(S',V): atoms(S',_), desc(S,S'), S'!=S.

scope(V',"global") :- var(V,N), scope(V,"global"),                           %%#(\label{lst:safe:global-end}#)
  stm(S), desc(S,V), desc(S,V'), var(V',N).

scope(V,"local") :- var(V), not scope(V,"global").                           %%#(\label{lst:safe:local}#)

binds(S,A,V') :- binds(S,A,V), var(V,N), var(V',N), desc(S,V').              %%#(\label{lst:safe:binds-shared-name}#)

binds(S,A,V) :- atoms(S,A), type(A,"classical_atom"), not negated(A),        %%#(\label{lst:safe:binds-classic}#)
  var(V), not desc(arith,V), desc(A,V).

other_binds(S,A,V) :- atoms(S,A), not negated(A), eq(A,T,_),                  %%#(\label{lst:safe:binds-eq-other-start}#)
  desc(T,V), var(V), binds(S,A',V), A'!=A.                                  %%#(\label{lst:safe:binds-eq-other-end}#)

binds(S,A,V) :- atoms(S,A), not negated(A), eq(A,T,U), desc(U,V), var(V),  %%#(\label{lst:safe:binds-eq-start}#)
  scope(V,C), not desc(arith,V),
  other_binds(S,A,V'): var(V'), scope(V',C), V'!=V, desc(T,V').            %%#(\label{lst:safe:binds-eq-end}#)

aspen(exception(node(S),format("Unsafe variables in:\n{}",(node(S),())));   %%#(\label{lst:safe:aspen-exc-start}#)
      exception(node(V),format("Variable '{}' is unsafe.",(node(V),())))) :-
  stm(S), desc(S,V), var(V), not binds(_,_,V).                               %%#(\label{lst:safe:aspen-exc-end}#)
