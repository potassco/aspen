#program add_var_to_atoms(var).

aspen(append(node(N), format("({0})", (var, ()))))
  :- node(N); type(N,"symbolic_atom"); #false : child(N,M), type(M,"terms").

aspen(append(node(P), format(",{0}", (var, ()))))
  :- node(N); type(N,"symbolic_atom"); child(N,M); type(M,"terms");
     Max = #max{ Idx: child(M,O), named_child_index(O,Idx) }; child(M,P);
     named_child_index(P,Max).
