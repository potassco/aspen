{ &next(a(X)): &next(&until(b(X), c(Y)))}
  :- {&next(d(X)) : &release(e(X), &next(f(X)))}, g(X).

#max{ 1:&next(h(X)): &next(&until(i(X), j(Y)))} = 1
  :- 1 < #min{k(X) : &release(l(X), &next(m(X)))}, n(X), not &next(o(X)).

&next(p(X)): &wnext(q(X)), &until(r(X), s(X))
  :- t(X) : &next(u(X)), v(X); w(X).
