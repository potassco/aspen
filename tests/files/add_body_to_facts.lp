fact(L) :- type(N,"rule"); child(N,L); field(L,"head"); #false: child(N,M), type(M,"body").
aspen(edit(L,format("{0} :- {1}", (L, (join("; ", ("b", ("c", ()))), ()))))) :- fact(L).
