#program color(k).

{ map(X,C,k) } :- vtx(X,k), col(C).
{ map(X,C,k) } :- vtx(X), col(C,k).

mapped(X,k) :- map(X,C,k).
:- vtx(X,k), not mapped(X).
#external mapped(X): vtx(X,k).

has_in_edge(X,k) :- arc(Y,X,k).
source(X,k) :- vtx(X,k), not has_in_edge(X).
#external has_in_edge(X): vtx(X,k).

% since we are trying to spare ourselves from generating expansions for not_source(X),
% as it has only one defining rule, where all the vars in the head occur in the body,
% we need in turn to check that only fresh derivations of has_in_edge are used.

% we might consider defining an auxiliary predicate
% fresh_has_in_edge(X,k) :- has_in_edge(X,k), not defined(has_in_edge(X)).
not_source(X,k) :- has_in_edge(X,k), not defined(has_in_edge(X)).

% here
index(X,0,k) :- source(X,k).
index(Y,I+1,k) :- index(X,I,k), arc(X,Y).
index(Y,I+1,k) :- index(X,I), arc(X,Y,k).

:- vtx(X,k), map(X,C), map(X,C'), C < C'.
:- vtx(X), map(X,C,k), map(X,C'), C < C'.
:- vtx(X), map(X,C), map(X,C',k), C < C'.

:- map(X,C,k), map(Y,C), arc(X,Y).
:- map(X,C), map(Y,C,k), arc(X,Y).
:- map(X,C), map(Y,C), arc(X,Y,k).

col(C) :- col(C,k).
vtx(X) :- vtx(X,k).
arc(X,Y) :- arc(X,Y,k).
map(X,C) :- map(X,C,k).
mapped(X) :- mapped(X,k), not defined(mapped(X)).
source(X) :- source(X,k).
has_in_edge(X) :- has_in_edge(X,k), not defined(has_in_edge(X)).
index(X,I) :- index(X,I,k), not defined(index(X,I)).
not_source(X) :- not_source(X,k).

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
