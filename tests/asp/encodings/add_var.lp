#include "../../../src/aspen/asp/utils/generic/derived_predicates/named_child_index.lp".

#program add_var_to_atoms(var).

aspen(edit(node(N), format("{0}({1})", (node(N),(var, ())))))
  :- node(N); type(N,"symbolic_atom"); #false : child(N,M), type(M,"terms").

aspen(edit(node(P), format("{0},{1}", (node(P), (var, ())))))
  :- node(N); type(N,"symbolic_atom"); child(N,M); type(M,"terms");
		 Max = #max{ Idx: child(M,O), named_child_index(O,Idx) }; child(M,P);
     named_child_index(P,Max).
