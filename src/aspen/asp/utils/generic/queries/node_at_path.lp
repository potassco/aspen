#include "../derived_predicates/child_index.lp".

% Initialize the query.
aspen(node_on_path((Source, Path), RootId, Path))
  :- aspen(query(node_at_path(Source, Path))), source_root(Source, RootId).

% Raise error if there is no such source
aspen(exception(format("Source {} not found.", (Source, ()))))
  :- aspen(query(node_at_path(Source, Path))), not source_root(Source, _).

% Recursive step, following the path down the tree.
aspen(node_on_path(Query, ChildId, Path))
  :- aspen(node_on_path(Query, NodeId, (Idx, Path))),
		 child(NodeId, ChildId), child_index(ChildId, Idx).

% Raise error if we cannot follow path all the way to a node
aspen(log(NodeId,
					"error",
					format("No node found at path: {}; path broken at node in range with remainder: {}.",
								 (Query, ((Idx, Path), ())))))
  :- aspen(node_on_path(Query, NodeId, (Idx, Path))),
		 #false : child(NodeId, ChildId), child_index(ChildId, Idx).

% If we have exhausted the path, we return the current node as answer to the query.
aspen(return(node_at_path(Source, QueryPath), NodeId))
  :- aspen(node_on_path((Source, QueryPath), NodeId, ())).
