% we assume that there are no pools in the input
% TODO: write a version that works with pools, or a transform encoding
% that unpools a program.
symbol(N) :- type(N,("symbolic_atom";"function")).

symbol_signature(N,Name,Arity)
  :- symbol(N), child(N,M), type(M,"terms"), field(M,"arguments"),
		 MaxIdx = #max{ Idx : named_child_index(M,L,Idx) }, Arity=MaxIdx+1,
		 child(N,J), type(J,"identifier"), field(J,"name"), leaf_text(J,Name).

symbol_signature(N,Name,0)
  :- symbol(N), #false: child(N,M), type(M,"terms");
		 child(N,J), type(J,"identifier"), field(J,"name"), leaf_text(J,Name).

#defined symbol_signature/3.
