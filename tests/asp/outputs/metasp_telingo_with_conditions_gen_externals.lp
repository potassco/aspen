{ &next(a(X)): &next(&until(b(X), c(Y)))}
  :- {&next(d(X)) : &release(e(X), &next(f(X)))}, g(X).

#max{ 1:&next(h(X)): &next(&until(i(X), j(Y)))} = 1
  :- 1 < #min{k(X) : &release(l(X), &next(m(X)))}, n(X), not &next(o(X)).

&next(p(X)): &wnext(q(X)), &until(r(X), s(X))
  :- t(X) : &next(u(X)), v(X); w(X).

% externals for set aggregate rule
#external a(X): c(Y), g(X).
#external &next(&until(b(X), c(Y))): c(Y), g(X).
#external &next(d(X)): f(X), g(X).
#external &release(e(X), &next(f(X))): f(X), g(X).

% externals for head/body aggregate rule
#external h(X): j(Y), n(X).
#external &next(&until(i(X), j(Y))): j(Y), n(X).
#external &release(l(X), &next(m(X))): m(X), n(X).
#external &next(o(X)): n(X).

% externals for conditional literal rule
#external p(X): s(X), w(X).
#external &wnext(q(X)): s(X), w(X).
#external &until(r(X), s(X)): s(X), w(X).
#external &next(u(X)): u(X), v(X), w(X).
