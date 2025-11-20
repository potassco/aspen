eventually(always(neg(alive(M)))) :- alive(M), initial.

wnext(release(eventually(p(X)), wnext(q(X)))) :- p(X).
