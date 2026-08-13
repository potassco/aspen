#include "../../../../src/aspen/asp/utils/aspcore2/derived_predicates/named_child_index.lp".

#show named_child_index/2.

list(N, I, P) :-
  type(N,"function"),
  child(N, M),
  type(M,"identifier"),
  leaf_text(M,"list"),
  child(N, O),
  type(O,"terms"),
  child(O, P),
  named_child_index(P, I).

cons_list(L, I+1, "nil") :- list(L, I, E), not list(L, I+1, _).

cons_list(L, I, format("cons({}, {})", (node(Car), (Cdr, ())))) :-
  list(L, I, Car),
  cons_list(L, I+1, Cdr).

aspen(replace(node(L), F)) :- cons_list(L, 0, F).
