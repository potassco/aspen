{ &next(a(X)): &next(&until(b(X), c(Y)))} :- {&next(d(X)) : &release(e(X), &next(f(X)))}, g(X).

#max{ 1:&next(h(X)): &next(&until(i(X), j(Y)))} = 1 :- 1 < #min{k(X) : &release(l(X), &next(m(X)))}, n(X), not &next(o(X)).

&next(p(X)): &wnext(q(X)), &until(r(X), s(X)) :- t(X) : &next(u(X)), v(X); w(X).

% externals for set aggregate rule
#external a(X): g(X), c(Y).
#external &next(&until(b(X), c(Y))): g(X), c(Y).
#external &next(d(X)): g(X), f(X).
#external &release(e(X), &next(f(X))): g(X), f(X).

% externals for head/body aggregate rule
#external h(X): n(X), j(Y).
#external &next(&until(i(X), j(Y))): n(X), j(Y).
#external &release(l(X), &next(m(X))): n(X), m(X).
#external &next(o(X)): n(X).

% externals for conditional literal rule
#external p(X): w(X), s(X).
#external &wnext(q(X)): w(X), s(X).
#external &until(r(X), s(X)): w(X), s(X).
#external &next(u(X)): w(X), v(X), u(X).
