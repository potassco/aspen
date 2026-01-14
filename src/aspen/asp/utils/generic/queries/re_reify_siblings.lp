#include "./node_at_path.lp".
#include "./delete_subtree.lp".
#include "../derived_predicates/named.lp".

% find siblings of the given length starting at path
%%
aspen(exception(
					format("Expected positive integer for length of delete_siblings, got: {}",
								 (Length, ()))))
  :- aspen(query(re_reify_siblings(Source, Path, Length))),
		 Length < 1.

aspen(query(node_at_path(Source, Path)))
  :- aspen(query(re_reify_siblings(Source, Path, Length))).

aspen(re_sibling(re_reify_siblings(Source, Path, Length), NodeId, 0, Length - 1))
  :- aspen(query(re_reify_siblings(Source, Path, Length))),
		 aspen(return(node_at_path(Source, Path), NodeId)).

aspen(re_sibling(Query, NextNodeId, Idx + 1, Remainder - 1))
  :- aspen(re_sibling(Query, NodeId, Idx, Remainder)),
		 next_sibling(NodeId, NextNodeId),
		 Remainder > 0.

aspen(exception(format("Error in query {}: node '{}' with id {} has no next siblings, should have {} more.",
											 (Query, (node(NodeId), (NodeId, (Remainder, ())))))))
  :- aspen(re_sibling(Query, NodeId, Idx, Remainder)),
		 not next_sibling(NodeId, _),
		 Remainder > 0.

% return nodes to delete

aspen(query(delete_subtree(NodeId)))
  :- aspen(re_sibling(Query, NodeId, _, _)).

aspen(return(Query, delete(Fact)))
  :- aspen(re_sibling(Query, NodeId, _, _)),
		 aspen(return(delete_subtree(NodeId), Fact)).

% return related nodes of interest

% parent of siblings
aspen(return(Query, parent(ParentId)))
  :- aspen(re_sibling(Query, NodeId, 0, _)),
	   child(ParentId, NodeId).

% previous (named) sibling of first sibling
aspen(return(Query, prev_sibling(PrevNodeId)))
  :- aspen(re_sibling(Query, NodeId, 0, _)),
		 next_sibling(PrevNodeId, NodeId).

% next (named) sibling of last sibling
aspen(return(Query, next_sibling(NextNodeId)))
  :- aspen(re_sibling(Query, NodeId, _, 0)),
		 next_sibling(NodeId, NextNodeId).
