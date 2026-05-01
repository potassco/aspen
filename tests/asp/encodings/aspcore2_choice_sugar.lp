#include "../../../src/aspen/asp/utils/generic/derived_predicates/named_child_index.lp".
#include "../../../src/aspen/asp/utils/generic/queries/make_list.lp".

choice(R,E) :-
  type(R,"rule"),
  child(R,C), type(C,"choice"),
  child(C,ES), type(ES,"choice_elements"),
  child(ES,E), type(E,"choice_element").

e_head(E,CA) :- choice(R,E), child(E,CA), type(CA,"classical_atom").

default(body(R);upper(R);sign(E);args(E);cond(E)) :- choice(R,E).
templ(I,T) :- _templ(I,T).
templ(I,"") :- default(I), not _templ(I,_).

_templ(body(R),node(B)) :- choice(R,E), child(R,B), type(B,"body").
_templ(sign(E),"neg_") :- e_head(E,CA), child(CA,S), leaf_text(S,"-").
_templ(args(E),node(T)) :- e_head(E,CA), child(CA,T), type(T, "terms").
_templ(upper(R),node(U)) :-
  choice(R,E), child(R,C), child(C,U), type(U,"upper").
_templ(cond(E),format(", {}",(node(LS),())))
  :- choice(R,E), child(E,LS), type(LS,"naf_literals").

templ(hat(E),format("__choice_{}{}({})",(S,(node(I),(A,()))))) :-
  e_head(E,CA), child(CA,I), type(I,"identifier"),
  templ(sign(E),S), templ(args(E),A).

templ(rule(E),format("{} | {} :- {}{}.",(node(CA),(H,(B,(LS,())))))) :-
  choice(R,E), e_head(E,CA), templ(hat(E),H),
  templ(body(R),B), templ(cond(E),LS).

templ(agg_element(E),format("{} : {}{}",(H,(node(CA),(LS,()))))) :-
  choice(R,E), e_head(E,CA), templ(hat(E),H), templ(cond(E),LS).

e_idx(R,E,I) :- choice(R,E), named_child_index(E,I).

aspen(query(make_list(agg_elements(R), T, I))) :-
  choice(R,E), named_child_index(E,I), templ(agg_element(E),T).

templ(constraint(R),
      format(":- {}, not #count{{ {} }} {}.",(B,(J,(U,()))))) :-
  choice(R,E), templ(body(R),B), templ(upper(R),U),
  aspen(return(make_list(agg_elements(R)), C)), J=join("; ", C).

aspen(query(make_list(rules(R), T, I))) :-
  choice(R,E), named_child_index(E,I), templ(rule(E),T).

aspen(replace(node(R),join("\n",(T, C)))) :-
     choice(R,_), aspen(return(make_list(rules(R)), C)),
     templ(constraint(R),T).

% #show choice/2.
% #show templ/2.
% #show agg_cons/2.
% #show rule_cons/2.
% #show e_idx/3.
% #show aspen/1.
