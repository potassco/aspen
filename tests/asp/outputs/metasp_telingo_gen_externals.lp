&next(p(X)):- &and(&wnext(q(Y)), &next(r(X))), d(X).

s(X) :- &true, &not(&false), &or(t(X), u(X)), &until(v(X), w(X)).

x(X,Y) :- &eventually(y(X)), &release(z(X), a(Y)), not &next(b(X)), &next(c(X)).

% externals for first rule
#external p(X): d(X), r(X).
#external &and(&wnext(q(Y)), &next(r(X))): d(X), r(X).

% externals for second rule
#external &true: w(X).
#external &not(&false): w(X).
#external &or(t(X), u(X)): w(X).
#external &until(v(X), w(X)): w(X).

% externals for third rule
#external &eventually(y(X)): c(X), a(Y), y(X).
#external &release(z(X), a(Y)): c(X), a(Y), y(X).
#external &next(b(X)): c(X), a(Y), y(X).
#external &next(c(X)): c(X), a(Y), y(X).
