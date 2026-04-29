#include "../../../src/aspen/asp/utils/generic/derived_predicates/named_child_index.lp".
#include "../../../src/aspen/asp/utils/generic/queries/make_list.lp".

choice(R,E) :-
  type(R,"rule"),
  child(R,C), type(C,"choice"),
  child(C,ES), type(ES,"choice_elements"),
  child(ES,E), type(E,"choice_element").

element_head(E,CA) :- choice(R,E), child(E,CA), type(CA,"classical_atom").

_default_template(body(R);upper(R);sign(E);args(E);cond(E)) :- choice(R,E).
template(I,T) :- _template(I,T).
template(I,"") :- _default_template(I), not _template(I,_).

_template(body(R),node(B)) :- choice(R,E), child(R,B), type(B,"body").
_template(upper(R),node(U)) :- choice(R,E), child(R,C), child(C,U), type(U,"upper").
_template(sign(E),"neg_") :- element_head(E,CA), child(CA,S), leaf_text(S,"-").
_template(args(E),node(T)) :- element_head(E,CA), child(CA,T), type(T, "terms").
_template(cond(E),format(", {}",(node(LS),()))) :- choice(R,E), child(E,LS), type(LS,"naf_literals").

template(hat(E),format("__choice_{}{}({})",(S,(node(I),(A,()))))) :-
  element_head(E,CA), child(CA,I), type(I,"identifier"),
  template(sign(E),S), template(args(E),A).

template(rule(E),format("{} | {} :- {}{}.",(node(CA),(H,(B,(LS,())))))) :-
  choice(R,E), element_head(E,CA), template(hat(E),H),
  template(body(R),B), template(cond(E),LS).

template(agg_element(E),format("{} : {}{}",(H,(node(CA),(LS,()))))) :-
  choice(R,E), element_head(E,CA), template(hat(E),H), template(cond(E),LS).

e_idx(R,E,I) :- choice(R,E), named_child_index(E,I).

aspen(query(make_list(agg_elements(R), T, I))) :-
  choice(R,E), named_child_index(E,I), template(agg_element(E),T).

template(constraint(R),format(":- {}, not #count{{ {} }} {}.",(B,(join("; ", CONS),(U,()))))) :-
  choice(R,E), template(body(R),B),
  aspen(return(make_list(agg_elements(R)), CONS)), template(upper(R),U).

aspen(query(make_list(rules(R), T, I))) :-
  choice(R,E), named_child_index(E,I), template(rule(E),T).

aspen(edit(node(R),join("\n",(T, CONS))))
  :- choice(R,_), aspen(return(make_list(rules(R)), CONS)), template(constraint(R),T).

% #show choice/2.
% #show template/2.
% #show agg_cons/2.
% #show rule_cons/2.
% #show e_idx/3.
#show aspen/1.
