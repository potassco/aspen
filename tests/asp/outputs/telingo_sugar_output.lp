until(true, release(false, neg(alive(M)))) :- alive(M), initial.

or(next(until(or(next(q(X)), final), and(or(next(q(X)), final), or(until(true, p(X)), final)))), final) :- p(X).
