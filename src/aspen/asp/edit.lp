%% edit((s(0),(((((),0),0),0),0)), format("{0}", ((s(0),(((((),0),2),0),0)), ()))).
%% edit((s(0),((((((((),0),2),0),0),2),0),0)), format("d", ())).

_replacement(edit(N,format(S,R)),R) :- edit(N,format(S,R)).
_replacement(G,R) :- _replacement(G,(F,R)).
replacement(G,F) :- _replacement(G,(F,R)).

target(edit(N,format(S,R)),N) :- edit(N,format(S,R)).

desc((S,P), (S,P)) :- query_desc((S,P)).
desc((S,P), (S,(P',I))) :- desc((S,P), (S,P')), node((S,(P',I))).

query_desc(T):- replacement(G,T).
depends(G,G') :- replacement(G,R), target(G',T), desc(R,T).

%% #show replacement/2.
%% #show target/2.
%% #show desc/2.
%% #show depends/2.

