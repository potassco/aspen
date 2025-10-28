% Calculate absolute byte offsets ranges.
call(node_range(ParentNode))
  :- call(node_range(Node)); child(ParentNode,_,Node).

return(node_range(Node), (PStart + Start, PStart + End))
  :- call(node_range(Node)); offset_byte_range(Node, Start, End);
		 child(ParentNode, _, Node); return(node_range(ParentNode), (PStart, PEnd)).

return(node_range(Node), (Start, End))
  :- call(node_range(Node)); root(Node); offset_byte_range(Node, Start, End).

% Find the smallest node within node that spans the given byte range.
% Replicates Node.descendant_for_byte_range(start_byte, end_byte, /) from tree-sitter.
_descendent_for_byte_range(Node, "mark", Node, Start, End)
  :- call(descendent_for_byte_range(Node, Start, End)).

_descendent_for_byte_range(N, Node, ChildNode, Start - ChildStart, End - ChildStart)
  :- _descendent_for_byte_range(N, _, Node, Start, End); child(Node, _, ChildNode);
		 offset_byte_range(ChildNode, ChildStart, ChildEnd);
     not Start < ChildStart; not ChildEnd < End.

return(descendent_for_byte_range(N, Start, End), Node)
	:- call(descendent_for_byte_range(N, Start, End));
		 _descendent_for_byte_range(N,_,Node,_,_);
		 not _descendent_for_byte_range(N,Node,_,_,_).

%% #show call/1.
%% #show return/2.

