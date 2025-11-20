until(a(X),b(Y)); not next(c((Y+X))) :- prev(d(X)); eventually_after(e(Y)).
f(X) :- weak_prev(g(X)).
h(X); i(Y) :- since(j(X),k(Y)).
h(X); i(Y) :- trigger(j(X),k(Y)).
prev(l(X)) :- m(X); not n(X); not weak_prev(o(X)).

#external prev(d(X)) : d(X); e(Y). [false]
#external eventually_after(e(Y)) : d(X); e(Y). [false]
#external weak_prev(g(X)). [false]
#external since(j(X),k(Y)) : k(Y). [false]
#external trigger(j(X),k(Y)) : k(Y). [false]
#external weak_prev(o(X)) : m(X). [false]
#external a(X) : d(X); e(Y). [false]
#external b(Y) : d(X); e(Y). [false]
#external c((Y+X)) : d(X); e(Y). [false]
#external l(X) : m(X). [false]
