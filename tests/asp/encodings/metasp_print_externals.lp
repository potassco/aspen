#include "../../../src/aspen/asp/utils/generic/queries/make_list.lp".

#program metasp_preprocess.

aspen(query(make_list(Head, node(Body), Body))) :- ext_body(Head, Body).


ext_head_source(Head, Source)
  :- ext_head(_,Head),
     metasp_statement_desc(Statement, Head),
     statement(Source, Statement).

aspen(print(format("#external {}: {}.", (node(Head), (join(", ", Cons), ()))), Source))
  :- aspen(return(make_list(Head), Cons)),
     ext_head_source(Head, Source).

aspen(print(format("#external {}.", (node(Head), ())), Source))
  :- ext_head_source(Head, Source),
     not ext_body(Head, _).
