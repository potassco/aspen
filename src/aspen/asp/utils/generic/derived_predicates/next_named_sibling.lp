#include "./named_child_index.lp".

next_named_sibling(Node, Node')
  :- child(Parent, Node),
		 child(Parent, Node'),
		 named_child_index(Node, Idx),
		 named_child_index(Node', Idx + 1).
