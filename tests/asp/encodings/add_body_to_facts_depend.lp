fact(L) :-
  type(N,"rule"); child(N,L); field(L,"head"); #false: child(N,M), type(M,"body").

b(N) :- type(N,"identifier"), leaf_text(N,"b").

aspen(replace(node(N),"d")) :- b(N).

aspen(replace(node(L),
	      format("{0} :- {1}",
	             (node(L),(join("; ",(node(N),("c", ()))),()))))) :-
  fact(L), b(N).
