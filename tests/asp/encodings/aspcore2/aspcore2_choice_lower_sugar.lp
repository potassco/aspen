#program aspcore_choice_double_bound.

choice(R,ES,L,U) :- type(R,"rule"), child(R,C), type(C,"choice"),
  child(C,ES), type(ES,"choice_elements"), child(C,L), type(L,"lower"),
  child(C,U), type(U,"upper").

templ(body(R),format(" :- {}", (node(B),()))) :- choice(R,_,_,_),
  child(R,B), type(B,"body").

templ(body(R),".") :- choice(R,_,_,_), #false: child(R,B), type(B,"body").

templ(lower(R),format("{} {{ {} }}{}.", (node(L),(node(ES),(T,()))))) :-
  choice(R,ES,L,_), templ(body(R),T).

templ(upper(R),format("{{ {} }} {}{}.", (node(ES),(node(U),(T,()))))) :-
  choice(R,ES,_,U), templ(body(R),T).

aspen(replace(node(R), join("\n", (LT,(UT,()))))) :-
  choice(R,_,_,_), templ(lower(R),LT), templ(upper(R),UT).

#program aspcore_choice_rewrite_lower.
inverse("<",">";"<=",">=";"=","=";"!=","!=").
inverse(B,A) :- inverse(A,B).

aspen(delete(node(L));
      append(node(C),format(" {0} {1}",(S',(node(T),()))))) :-
  type(C,"choice"), child(C,L), type(L,"lower"),
  child(L,T), child(L,O), next_sibling(T,O), leaf_text(O,S), inverse(S,S').
