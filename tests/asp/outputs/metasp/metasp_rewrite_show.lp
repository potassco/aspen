_show.
_show_term(&next(good(X))) :- &next(good(X)), dog(X).
_show_atom(dog(A)) :- dog(A).
_show_atom(&until(B, A)) :- &until(B, A).
