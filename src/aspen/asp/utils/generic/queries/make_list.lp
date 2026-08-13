aspen(query(make_list(Id))) :- aspen(query(make_list(Id, Element, Rank))).

aspen(_list_idx(Id, Element, Index)) :-
  aspen(query(make_list(Id, Element, Rank))),
  Index = #count{ Rank': aspen(query(make_list(Id, _, Rank'))), Rank < Rank' }.

aspen(_list(Id, (), -1)) :- aspen(query(make_list(Id))).

aspen(_list(Id, (Element, Cons), Index+1)) :-
  aspen(_list(Id, Cons, Index)), aspen(_list_idx(Id, Element, Index+1)).

aspen(return(make_list(Id), Cons)) :-
  aspen(_list(Id, Cons, Index)), not aspen(_list_idx(Id, _, Index+1)).
