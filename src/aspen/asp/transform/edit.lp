#include "../utils/generic/queries/descendant.lp".
#include "../utils/generic/queries/path_of_node.lp".


aspen(node_in_fstring(Fstring, Item))
  :- aspen(item_in_fstring(Fstring, Item)), Item=node(N).

aspen(list_in_fstring(Fstring, List))
  :- aspen(item_in_fstring(Fstring, Item)), Item=join(Sep, List).

aspen(list_in_fstring(Fstring, List))
  :- aspen(item_in_fstring(Fstring, Item)), Item=format(FormatStr, List).

aspen(list_in_fstring(Fstring, Rest)) :- aspen(list_in_fstring(Fstring, (First, Rest))).

aspen(item_in_fstring(Fstring, First)) :- aspen(list_in_fstring(Fstring, (First, Rest))).



aspen(item_in_fstring(Fstring, Fstring)) :- aspen(edit(Node, Fstring)).

aspen(query(descendant(N))) :- aspen(edit(_,S)), aspen(node_in_fstring(S,node(N))).

aspen(target(edit(N,S),N)) :- aspen(edit(N,S)).

% aspen(comes_before(Edit',Edit)) means Edit' must occur before Edit
% for the text replacement to be correct.
aspen(comes_before(edit(node(TargetNode), S'), edit(N, S)))
  :- aspen(edit(N, S)),
		 aspen(node_in_fstring(S, node(InsertNode))),
		 aspen(edit(node(TargetNode), S')),
		 edit(N, S) != edit(node(TargetNode), S'),
		 aspen(return(query(descendant(InsertNode)), TargetNode)).

% query the paths of node ids that occur as targets or as inserts
% in fstrings, so that they can be used by aspen to find the
% corresponding tree-sitter nodes for processing.

% targets
aspen(query(path_of_node(N))) :- aspen(log(node(N), _, _)).
aspen(query(path_of_node(N))) :- aspen(exception(node(N), _)).
aspen(query(path_of_node(N))) :- aspen(edit(node(N), _)).

% fstring inserts
aspen(item_in_fstring(Fstring, Fstring)) :- aspen(log(_, _, Fstring)).
aspen(item_in_fstring(Fstring, Fstring)) :- aspen(log(_, Fstring)).

aspen(item_in_fstring(Fstring, Fstring)) :- aspen(exception(_, Fstring)).
aspen(item_in_fstring(Fstring, Fstring)) :- aspen(exception(Fstring)).

aspen(query(path_of_node(Node))) :- aspen(node_in_fstring(_, node(Node))).
