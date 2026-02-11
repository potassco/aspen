&next(p(X)):- &and(&wnext(q(Y)), &next(r(X))), d(X).

s(X) :- &true, &not(&false), &or(t(X), u(X)), &until(v(X), w(X)).

x(X,Y) :- &eventually(y(X)), &release(z(X), a(Y)), not &next(b(X)), &next(c(X)).

% externals for first rule
#external p(X): r(X), d(X).
#external &and(&wnext(q(Y)), &next(r(X))): r(X), d(X).

% externals for second rule
#external &true: w(X).
#external &not(&false): w(X).
#external &or(t(X), u(X)): w(X).
#external &until(v(X), w(X)): w(X).

% externals for third rule
#external &eventually(y(X)): y(X), a(Y), c(X).
#external &release(z(X), a(Y)): y(X), a(Y), c(X).
#external &next(b(X)): y(X), a(Y), c(X).
#external &next(c(X)): y(X), a(Y), c(X).
