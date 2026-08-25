#program color(k).

{ map(X,C) } :- vtx(X), col(C), new(X,k;C,k).

:- map(X,C), map(Y,C), arc(X,Y), new(X,k;Y,k;C,k).

mapped(X,k) :- map(X,C), new(X,k;C,k).
:- vtx(X), not mapped(X), new(X,k).
#external mapped(X,k): vtx(X), new(X,k).

:- vtx(X), map(X,C), map(X,C'), C != C', new(X,k;C,k;C',k).

mapped(X) :- mapped(X,k), new(X,k).

mapped(X,k) :- mapped(X,k+1).
#external mapped(X,k+1): mapped(X,k).

#show vtx/1.
#show col/1.
#show arc/2.
#show map/2.
