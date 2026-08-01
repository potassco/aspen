#include "../../../src/aspen/asp/utils/generic/raise_syntax_error.lp".    %%#(\label{lst:choice:include-start}#)
#include "../../../src/aspen/asp/utils/clingo/derived_predicates/named_child_index.lp".
#include "../../../src/aspen/asp/utils/generic/queries/make_list.lp".

#include "../../../src/aspen/asp/utils/aspcore2/derived_predicates/symbol_signature.lp". %%#(\label{lst:choice:include-end}#)

choice(R,E) :-                                                             %%%%#(\label{lst:choice:choice-start}#)
  type(R,"rule"),
  child(R,C), type(C,"choice"),
  child(C,ES), type(ES,"choice_elements"),
  child(ES,E), type(E,"choice_element").                                   %%%%#(\label{lst:choice:choice-end}#)

e_head(E,A) :- choice(R,E), child(E,A), type(A,"classical_atom").          %%%%#(\label{lst:choice:ehead}#)

_templ(body(R),node(B)) :- choice(R,E), child(R,B), type(B,"body").        %%%%#(\label{lst:choice:templ-start}#)
_templ(upper(R),node(U)) :-
  choice(R,E), child(R,C), child(C,U), type(U,"upper").
_templ(cond(E),format(", {}",(node(LS),()))) :-
  choice(R,E), child(E,LS), type(LS,"naf_literals").                       %%%%#(\label{lst:choice:templ-end}#)

templ(I,T) :- _templ(I,T).                                                 %%%%#(\label{lst:choice:default-start}#)
templ(I,"") :- default(I), not _templ(I,_).
default(body(R);upper(R);cond(E)) :- choice(R,E).                          %%%%#(\label{lst:choice:default-end}#)

templ(hat(E),format("not_choose({0})",(node(A),()))) :- e_head(E,A).       %%%%#(\label{lst:choice:hat}#)

aspen(exception(node(A),"Predicate not_choose/1 is system reserved.")) :-  %%%%#(\label{lst:choice:exception-start}#)
  type(A,"classical_atom"), symbol_signature(A,("not_choose",1)).          %%%%#(\label{lst:choice:exception-end}#)

templ(rule(E),format("{0} | {1} :- {2}{3}.", (node(A),(H,(B,(LS,())))))) :- %%%%#(\label{lst:choice:rule-start}#)
  choice(R,E), e_head(E,A), templ(hat(E),H),
  templ(body(R),B), templ(cond(E),LS).                                     %%%%#(\label{lst:choice:rule-end}#)

templ(agg_element(E),format("{0} : {1}{2}",(H,(node(A),(LS,()))))) :-      %%%%#(\label{lst:choice:agg-ele-start}#)
  choice(R,E), e_head(E,A), templ(hat(E),H), templ(cond(E),LS).            %%%%#(\label{lst:choice:agg-ele-end}#)

aspen(query(make_list(agg_elements(R),T,I))) :-                            %%%%#(\label{lst:choice:agg-list-start}#)
  choice(R,E), named_child_index(E,I), templ(agg_element(E),T).            %%%%#(\label{lst:choice:agg-list-end}#)

templ(constraint(R), format(":- {0}, not #count{{ {1} }} {2}.",(B,(J,(U,()))))) :- %%%%#(\label{lst:choice:constraint-start}#)
  choice(R,E), templ(body(R),B), templ(upper(R),U),
  aspen(return(make_list(agg_elements(R)),C)), J=join("; ",C).             %%%%#(\label{lst:choice:constraint-end}#)

aspen(query(make_list(rules(R), T, I))) :-                                 %%%%#(\label{lst:choice:rules-list-start}#)
  choice(R,E), named_child_index(E,I), templ(rule(E),T).                   %%%%#(\label{lst:choice:rules-list-end}#)

aspen(replace(node(R),join("\n",(T, C)))) :-                               %%%%#(\label{lst:choice:replace-start}#)
  choice(R,_), aspen(return(make_list(rules(R)), C)),
  templ(constraint(R),T).                                                  %%%%#(\label{lst:choice:replace-end}#)
