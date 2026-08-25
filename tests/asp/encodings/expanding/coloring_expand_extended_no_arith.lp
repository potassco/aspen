#program color(k).

{ map(X,C) } :- vtx(X), col(C), new(X,k;C,k).

:- map(X,C), map(Y,C), arc(X,Y), new(X,k;Y,k;C,k).

mapped(X,k) :- map(X,C), new(X,k;C,k).
:- vtx(X), not mapped(X), new(X,k).
#external mapped(X,k): vtx(X), new(X,k).

has_in_edge(X,k) :- arc(Y,X), new(X,k;Y,k).
source(X) :- vtx(X), not has_in_edge(X), new(X,k).
#external has_in_edge(X,k): vtx(X), new(X,k).

% We need to do some extra work, since arithmetic can create new terms during grounding.
% I think there is a similar issue for function symbols, those also create
% new terms. Also, we need something similar if a rule just introduces a new constant term

index(X,0,k) :- source(X), new(X,k).
index(Y,I+1,k) :- index(X,I), arc(X,Y), new(X,k;Y,k;I,k).

new(0,k) :- source(X), new(X,k).
new(I+1,k) :- index(X,I), arc(X,Y), new(X,k;Y,k;I,k).

:- vtx(X), map(X,C), map(X,C'), C != C', new(X,k;C,k;C',k).

mapped(X) :- mapped(X,k), new(X,k).
has_in_edge(X) :- has_in_edge(X,k), new(X,k).
index(X,I) :- index(X,I,k), new(X,k;I,k).

mapped(X,k) :- mapped(X,k+1).
#external mapped(X,k+1): mapped(X,k).
has_in_edge(X,k) :- has_in_edge(X,k+1).
#external has_in_edge(X,k+1): has_in_edge(X,k).
index(X,I,k) :- index(X,I,k+1).
#external index(X,I,k+1): index(X,I,k).


#show vtx/1.
#show col/1.
#show arc/2.
#show map/2.

#show index(X,I): index(X,arith(X,I)).
#show source/1.
