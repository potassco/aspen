#program color(k).

#external __new(col(C)): col(C), not __defined(col(C)). [true]
#external __new(vtx(X)): vtx(X), not __defined(vtx(X)). [true]
#external __new(arc(X,Y)): arc(X,Y), not __defined(arc(X,Y)). [true]

#external __new(map(X,C)): map(X,C), not __defined(map(X,C)). [true]
#external __new(mapped(X)): mapped(X), not __defined(mapped(X)). [true]
#external __new(has_in_edge(X)): has_in_edge(X), not __defined(has_in_edge(X)). [true]
#external __new(source(X)): source(X), not __defined(source(X)). [true]
#external __new(index(X,I)): index(X,I), not __defined(index(X,I)). [true]
#external __new(mapped(X)): mapped(X), not __defined(mapped(X)). [true]

% We would like to simply assert __defined(X) as a fact in the next solving step
% for each __new(X) we derived in the current one. We need it to be a fact,
% so it can be used for grounding simplifications below, and as far as i can tell,
% there is no good way of doing this without relying on externals scripting -
% so that's how we do it

{ map(X,C) } :- vtx(X), col(C), #false: __defined(vtx(X)), __defined(col(C)).

mapped(X,k) :- map(X,C), not __defined(map(X,C)).
:- vtx(X), not mapped(X), not __defined(vtx(X)).
#external mapped(X): vtx(X), not __defined(vtx(X)).

has_in_edge(X,k) :- arc(Y,X), not __defined(arc(Y,X)).
source(X) :- vtx(X), not has_in_edge(X), not __defined(vtx(X)).
#external has_in_edge(X): vtx(X), not __defined(vtx(X)).

index(X,0,k) :- source(X), not __defined(source(X)).
index(Y,I+1,k) :- index(X,I), arc(X,Y), #false: __defined(index(X,I)), __defined(arc(X,Y)).

:- map(X,C), map(X,C'), C != C', #false: __defined(map(X,C)), __defined(map(X,C')).

:- map(X,C), map(Y,C), arc(X,Y), #false: __defined(map(X,C)), __defined(map(Y,C)), __defined(arc(X,Y)).

mapped(X) :- mapped(X,k), not __defined(mapped(X)).
has_in_edge(X) :- has_in_edge(X,k), not __defined(has_in_edge(X)).
index(X,I) :- index(X,I,k), not __defined(index(X,I)).

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
