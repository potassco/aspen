fact(L)
  :- type(N,"rule"); child(N,L); field(L,"head");
		 #false: child(N,M), type(M,"body").

aspen(edit(node(L),
					 format("{0} :- {1}", (node(L),
																(join("; ", ("b", ("c", ()))),
																())))))
  :- fact(L).
