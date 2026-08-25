{ map(X,C) } :- vtx(X), col(C).

mapped(X) :- map(X,C).
:- vtx(X), not mapped(X).

has_in_edge(X) :- arc(Y,X).
source(X) :- vtx(X), no has_in_edge(X).

index(X,0) :- source(X).
index(Y,I+1) :- index(X,I), arc(X,Y).

:- map(X,C), map(X,C'), C != C'.

:- map(X,C), map(Y,C), arc(X,Y).


#show vtx/1.
#show col/1.
#show arc/2.
#show map/2.
