#include "../utils/generic/queries/descendant.lp".
#include "../utils/generic/queries/path_of_node.lp".

#script (python)

from clingo.symbol import Symbol, String

def symbol_to_string(symbol: Symbol) -> String:
    return String(str(symbol))

#end.

aspen(replace(N,"")) :- aspen(delete(N)).

aspen(node_in_template(Template, Item)) :-
  aspen(item_in_template(Template, Item)), Item=node(N).

aspen(list_in_template(Template, List)) :-
  aspen(item_in_template(Template, Item)), Item=join(Sep, List).

aspen(list_in_template(Template, List)) :-
  aspen(item_in_template(Template, Item)), Item=format(FormatStr, List).

aspen(list_in_template(Template, Rest)) :-
  aspen(list_in_template(Template, (First, Rest))).

aspen(item_in_template(Template, First)) :-
  aspen(list_in_template(Template, (First, Rest))).

aspen(item_in_template(Template, Template)) :-
  aspen(replace(Node, Template);append(Node, Template)).

aspen(query(descendant(N))) :-
  aspen(replace(_,S);append(_,S)), aspen(node_in_template(S,node(N))).

aspen(target(replace(N,S),N)) :- aspen(replace(N,S)).

% aspen(comes_before(Edit',Edit)) means Edit' must occur before Edit
% for the text replacement to be correct.
aspen(comes_before(replace(node(TargetNode), S'), Edit)) :-
  aspen(Edit),
  Edit = (replace(N, S);append(N, S)),
  aspen(node_in_template(S, node(InsertNode))),
  aspen(replace(node(TargetNode), S')),
  Edit != replace(node(TargetNode), S'),
  aspen(return(query(descendant(InsertNode)), TargetNode)).

% query the paths of node ids that occur as targets or as inserts
% in templates, so that they can be used by aspen to find the
% corresponding tree-sitter nodes for processing.

% targets
aspen(query(path_of_node(N))) :-
  aspen(log(node(N), _, _);exception(node(N), _);
        replace(node(N), _);append(node(N), _)).

% template inserts
aspen(item_in_template(Template, Template)) :-
  aspen(log(_, _, Template);log(_, Template);print(Template);print(Template, _);
     	exception(_, Template);exception(Template)).

aspen(query(path_of_node(Node))) :- aspen(node_in_template(_, node(Node))).
