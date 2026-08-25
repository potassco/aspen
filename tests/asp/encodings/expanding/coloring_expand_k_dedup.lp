#program color(k).

{ map(X,C) } :- vtx(X), col(C), #false: defined(vtx(X)), defined(col(C)).

mapped(X,k) :- map(X,C), not defined(map(X,C)).
:- vtx(X), not mapped(X), not defined(vtx(X)).
#external mapped(X): vtx(X), not defined(vtx(X)).

has_in_edge(X,k) :- arc(Y,X), not defined(arc(Y,X)).
source(X) :- vtx(X), not has_in_edge(X), not defined(vtx(X)).
#external has_in_edge(X): vtx(X), not defined(vtx(X)).

index(X,0,k) :- source(X), not defined(source(X)).
index(Y,I+1,k) :- index(X,I), arc(X,Y), #false: defined(index(X,I)), defined(arc(X,Y)).

:- map(X,C), map(X,C'), C != C', #false: defined(map(X,C)), defined(map(X,C')).

:- map(X,C), map(Y,C), arc(X,Y), #false: defined(map(X,C)), defined(map(Y,C)), defined(arc(X,Y)).

mapped(X) :- mapped(X,k), not defined(mapped(X)).
has_in_edge(X) :- has_in_edge(X,k), not defined(has_in_edge(X)).
index(X,I) :- index(X,I,k), not defined(index(X,I)).

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
