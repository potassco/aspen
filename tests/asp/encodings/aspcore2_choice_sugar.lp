#include "../../../src/aspen/asp/utils/generic/raise_syntax_error.lp".
#include "../../../src/aspen/asp/utils/generic/derived_predicates/named_child_index.lp".
#include "../../../src/aspen/asp/utils/generic/queries/make_list.lp".

#include "../../../src/aspen/asp/utils/aspcore2/derived_predicates/symbol_signature.lp".

choice(R,E) :-
  type(R,"rule"),
  child(R,C), type(C,"choice"),
  child(C,ES), type(ES,"choice_elements"),
  child(ES,E), type(E,"choice_element").

e_head(E,CA) :- choice(R,E), child(E,CA), type(CA,"classical_atom").

default(body(R);upper(R);cond(E)) :- choice(R,E).
templ(I,T) :- _templ(I,T).
templ(I,"") :- default(I), not _templ(I,_).
_templ(body(R),node(B)) :- choice(R,E), child(R,B), type(B,"body").
_templ(upper(R),node(U)) :-
  choice(R,E), child(R,C), child(C,U), type(U,"upper").
_templ(cond(E),format(", {}",(node(LS),())))
  :- choice(R,E), child(E,LS), type(LS,"naf_literals").

templ(hat(E),format("not_choose__({0})",(node(CA),()))) :- e_head(E,CA).

templ(rule(E),format("{0} | {1} :- {2}{3}.",
	             (node(CA),(H,(B,(LS,())))))) :-
  choice(R,E), e_head(E,CA), templ(hat(E),H),
  templ(body(R),B), templ(cond(E),LS).

templ(agg_element(E),format("{0} : {1}{2}",(H,(node(CA),(LS,()))))) :-
  choice(R,E), e_head(E,CA), templ(hat(E),H), templ(cond(E),LS).

aspen(query(make_list(agg_elements(R), T, I))) :-
  choice(R,E), named_child_index(E,I), templ(agg_element(E),T).

templ(constraint(R),
      format(":- {0}, not #count{{ {1} }} {2}.",(B,(J,(U,()))))) :-
  choice(R,E), templ(body(R),B), templ(upper(R),U),
  aspen(return(make_list(agg_elements(R)), C)), J=join("; ", C).

aspen(query(make_list(rules(R), T, I))) :-
  choice(R,E), named_child_index(E,I), templ(rule(E),T).

aspen(replace(node(R),join("\n",(T, C)))) :-
  choice(R,_), aspen(return(make_list(rules(R)), C)),
  templ(constraint(R),T).

aspen(exception(node(A),"Predicate not_choose__/1 is system reserved.")) :-
  type(A,"classical_atom"), symbol_signature(A,("not_choose__",1)).

% #show choice/2.
% #show templ/2.
% #show agg_cons/2.
% #show rule_cons/2.
% #show e_idx/3.
% #show symbol_signature/3.
% #show aspen/1.
% #show aspen/1.
