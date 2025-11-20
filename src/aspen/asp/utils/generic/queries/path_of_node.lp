#include "../derived_predicates/child_index.lp".

% raise error if we are not given a node identifier

aspen(exception(format("Expected a node identifier as input for path_of_node query, found {}.",
									 (Node, ()))))
  :- aspen(query(path_of_node(Node))), not node(Node).

% base
aspen(acc_path(Node, Node, ()))
  :- aspen(query(path_of_node(Node))), child_index(Node,Idx).

% recur
aspen(acc_path(Query, Parent, (Idx, Path)))
  :- aspen(acc_path(Query, Node, Path)), child(Parent, Node), child_index(Node, Idx).

aspen(return(path_of_node(Query), (Source, Path)))
  :- aspen(acc_path(Query, Node, Path)), source_root(Source, Node).
