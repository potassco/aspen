#include "../../generic/derived_predicates/named_child_index.lp".

symbol(N) :- type(N,("classical_atom";"function")).

symbol_signature(N,(Name,Arity))
  :- symbol(N), child(N,M), type(M,"terms"),
		 MaxIdx = #max{ Idx : child(M,L), named_child_index(L,Idx) }, Arity=MaxIdx+1,
		 child(N,J), type(J,"identifier"), leaf_text(J,Name).

symbol_signature(N,(Name,0))
  :- symbol(N), #false: child(N,M), type(M,"terms");
		child(N,J), type(J,"identifier"), leaf_text(J,Name).

#defined symbol_signature/3.
