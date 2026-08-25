#program color_initial(k).

{ map(X,C) } :- vtx(X), col(C).

mapped(X,k) :- map(X,C).
:- vtx(X), not mapped(X).
#external mapped(X): vtx(X).

has_in_edge(X,k) :- arc(Y,X).
source(X) :- vtx(X), not has_in_edge(X).
#external has_in_edge(X): vtx(X).

not_source(X) :- has_in_edge(X,k).

index(X,0,k) :- source(X,k).
index(Y,I+1,k) :- index(X,I,k), arc(X,Y).

:- vtx(X), map(X,C), map(X,C'), C < C'.

:- map(X,C), map(Y,C), arc(X,Y).

mapped(X) :- mapped(X,k).
has_in_edge(X) :- has_in_edge(X,k).
index(X,I) :- index(X,I,k).

mapped(X,k) :- mapped(X,k+1).
has_in_edge(X,k) :- has_in_edge(X,k+1).
index(X,I,k) :- index(X,I,k+1).

#external mapped(X,k+1): mapped(X,k).
#external has_in_edge(X,k+1): has_in_edge(X,k).
#external index(X,I,k+1): index(X,I,k).

#show col/1.
#show vtx/1.
#show arc/2.
#show map/2.
#show mapped/1.
#show done/1.
#show has_in_edge/1.
#show index/2.
#show source/1.
#show not_source/1.
