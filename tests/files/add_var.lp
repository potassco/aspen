#program add_var_to_atoms(var).

aspen(edit(N, format("{0}({1})", (N,(var, ())))))
  :- node(N); type(N,"symbolic_atom"); #false : child(N,M), type(M,"terms").

aspen(edit(L, format("{0},{1}", (L, (var, ())))))
  :- node(N); type(N,"symbolic_atom"); child(N,M); M=(S,P); type(M,"terms");
		 Max = #max{ I: named_child_idx(M,I) }, node((S,(P,I))), L = (S,(P,I)).
