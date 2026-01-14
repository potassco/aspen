aspen(exception(
					format("Expected input N to query delete_subtree(N) to be node id, found: {0}",
								 (N, ()))))
  :- aspen(query(delete_subtree(N))), not node(N).

aspen(delete_in_subtree(delete_subtree(N), node(N))) :- aspen(query(delete_subtree(N))), node(N).
aspen(delete_in_subtree(delete_subtree(N), child(P, N))) :- aspen(query(delete_subtree(N))), node(N), child(P, N).
% effects of deleting a node

aspen(delete_in_subtree(Query, (node(Id');child(Id,Id'))))
  :- aspen(delete_in_subtree(Query, node(Id))),
		 child(Id,Id').

aspen(delete_in_subtree(Query, next_sibling(Id,Id')))
  :- aspen(delete_in_subtree(Query, node(Id))),
		 next_sibling(Id,Id').

aspen(delete_in_subtree(Query, next_sibling(Id',Id)))
  :- aspen(delete_in_subtree(Query, node(Id))),
		 next_sibling(Id',Id).

aspen(delete_in_subtree(Query, type(Id,T)))
  :- aspen(delete_in_subtree(Query, node(Id))),
		 type(Id,T).

aspen(delete_in_subtree(Query, error(Id)))
  :- aspen(delete_in_subtree(Query, node(Id))),
		 error(Id).

aspen(delete_in_subtree(Query, missing(Id)))
  :- aspen(delete_in_subtree(Query, node(Id))),
		 missing(Id).

aspen(delete_in_subtree(Query, extra(Id)))
  :- aspen(delete_in_subtree(Query, node(Id))),
		 extra(Id).

aspen(delete_in_subtree(Query, leaf_text(Id,Text)))
  :- aspen(delete_in_subtree(Query, node(Id))),
		 leaf_text(Id,Text).

aspen(delete_in_subtree(Query, field(Id,Name)))
  :- aspen(delete_in_subtree(Query, node(Id))),
		 field(Id,Name).

% return from query

aspen(return(Query, Fact)) :- aspen(delete_in_subtree(Query, Fact)).
